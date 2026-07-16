"""
Orchestrator Module
===================
The master controller for the entire Agentic AI system.
Implements a Map-Reduce concurrency pattern, capable of breaking down
a complex user query into sub-tasks (Map), spinning up independent asynchronous
LangGraph workers to research each sub-task in parallel, and aggregating the results
into a cohesive final document (Reduce).

Conversational upgrade:
- Per-session sliding-window chat memory (SessionStore) keyed by frontend session_id.
- Worker graph compiled with a LangGraph MemorySaver checkpointer; every run gets
  a unique thread_id derived from the session.
- Structured streaming: `assurance` (instant intent acknowledgment tokens),
  `tool_telemetry` (exact concurrent tool queries), `progress`, `telemetry`,
  and `complete` events.
"""

import asyncio
import time
import json
import uuid

from langgraph.checkpoint.memory import MemorySaver

from core_engine.nodes.strategy_planner import strategy_planner_node
from core_engine.nodes.intent_classifier import (
    classify_intent,
    resolve_workflow_route,
    CONVERSATIONAL,
    DEEP_RESEARCH,
    RESEARCH_AGENT,
)
from core_engine.loop_worker import build_loop_worker, ResearchState
from core_engine.llm_router import LLMRouter
from core_engine.session_memory import SessionStore, format_messages_for_prompt
from core_engine.utilities.progress import (
    sanitize_status,
    emit_progress,
    register_progress_stream,
    release_progress_stream,
)

ASSURANCE_PROMPT = (
    "You are A.R.I.A., an autonomous research assistant. "
    "In one or two short sentences, tell the user exactly what you are about to research "
    "and which kinds of sources you will consult (web results, academic papers on arXiv, "
    "or their uploaded documents). Start with a phrase like \"I'll\". "
    "Return plain text only: no markdown, no lists, no emojis."
)


def _merge_matrices(matrices: list) -> list:
    """Merge per-section extraction matrices, deduplicating by (title, url_or_source)."""
    merged, seen = [], set()
    for matrix in matrices:
        for record in matrix or []:
            key = (
                str(record.get("title", "")).strip().lower(),
                str(record.get("url_or_source", "")).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)
    return merged


def _chunk_text(chunk) -> str:
    """Extract plain text from a streamed LLM chunk (Gemini may return part lists)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
        )
    return ""


class ResearchOrchestrator:
    """
    Manages the lifecycle and execution of LangGraph research loops.
    """

    def __init__(self):
        # Checkpointer enables thread-scoped LangGraph state persistence.
        self.checkpointer = MemorySaver()
        self.worker_graph = build_loop_worker(checkpointer=self.checkpointer)
        # Conversational memory: session_id -> sliding window of chat messages.
        self.sessions = SessionStore()

    def _thread_config(self, session_id: str | None) -> dict:
        """Build a unique LangGraph thread config for a single worker run."""
        thread_id = f"{session_id or 'anonymous'}::{uuid.uuid4().hex}"
        return {"configurable": {"thread_id": thread_id}}

    async def _run_single_section(
        self,
        overall_query: str,
        section_title: str,
        section_question: str,
        semaphore: asyncio.Semaphore,
        stream_id: str | None = None,
        chat_history: list | None = None,
        session_id: str | None = None,
    ) -> tuple[str, list]:
        async with semaphore:
            initial_state: ResearchState = {
                "query": overall_query,
                "current_section_title": section_title,
                "current_section": section_question,
                "research_history": "",
                "research_complete": False,
                "current_gaps": [section_question],
                "pending_tool_tasks": [],
                "completed_sections": [],
                "loop_count": 0,
                "chat_history": chat_history or [],
                # Deep-mode section workers are always research runs: preset the
                # route so the Intent Classifier skips its LLM call.
                "intent_route": DEEP_RESEARCH,
                "extraction_matrix": [],
            }

            print(f"\U0001f9f5 [Worker Spawned] Starting research loop for: '{section_title}'")

            if stream_id is not None:
                initial_state["progress_stream_id"] = stream_id
                emit_progress(initial_state, f"[Worker] Starting research for {section_title}.")

            final_state = await self.worker_graph.ainvoke(
                initial_state, config=self._thread_config(session_id)
            )
            return (
                final_state["completed_sections"][0],
                final_state.get("extraction_matrix") or [],
            )

    async def run_deep_research(
        self,
        query: str,
        progress_queue: asyncio.Queue | None = None,
        session_id: str | None = None,
    ) -> dict:
        print(f"\n\U0001f680 [Orchestrator] Starting Deep Research for: '{query}'\n")

        stream_id = None
        if progress_queue is not None:
            stream_id = register_progress_stream(progress_queue)
            progress_queue.put_nowait(sanitize_status("[Orchestrator] Starting deep parallel research."))

        chat_history = self.sessions.get_messages(session_id)

        try:
            planner_state = {"query": query}
            if progress_queue is not None:
                # The planner runs outside the checkpointed graph, so the live
                # queue handle is safe to pass directly here.
                planner_state["progress_queue"] = progress_queue
            plan_state = strategy_planner_node(planner_state)
            report_plan = plan_state["report_plan"]

            print(f"\n\U0001f4cb [Orchestrator] Blueprint generated! {len(report_plan.report_outline)} sections required.")

            semaphore = asyncio.Semaphore(2)

            tasks = []
            for section in report_plan.report_outline:
                task_coroutine = self._run_single_section(
                    query,
                    section.title,
                    section.key_question,
                    semaphore,
                    stream_id,
                    chat_history,
                    session_id,
                )
                tasks.append(asyncio.create_task(task_coroutine))

            print("\n\u26a1 [Orchestrator] Firing parallel research loops...\n")

            try:
                section_results = await asyncio.gather(*tasks)
            except Exception as e:
                print(f"\n\U0001f6a8 [Orchestrator] CRITICAL WORKER FAILURE: {str(e)}")
                for t in tasks:
                    if not t.done():
                        t.cancel()
                raise e

            print("\n\U0001f4da [Orchestrator] All workers finished. Compiling final report...\n")

            completed_sections = [result[0] for result in section_results]
            # Unify the per-section extraction matrices into one deduplicated grid.
            extraction_matrix = _merge_matrices([result[1] for result in section_results])

            final_report = f"# {report_plan.report_title}\n\n"
            final_report += f"**Background Context:**\n{report_plan.background_context}\n\n"
            final_report += "---\n\n"
            final_report += "\n\n".join(completed_sections)

            # Record this turn so follow-up questions can reference the report.
            self.sessions.append_turn(
                session_id, query, final_report,
                extraction_matrix=extraction_matrix, kind="research",
            )

            return {"report": final_report, "extraction_matrix": extraction_matrix, "kind": "research"}
        finally:
            if stream_id:
                release_progress_stream(stream_id)

    async def run_single_research(
        self,
        query: str,
        progress_queue: asyncio.Queue | None = None,
        session_id: str | None = None,
        target_doc_id: str | None = None,
        intent_route: str | None = None,
        workflow_mode: str | None = None,
    ) -> dict:
        print(f"\n\U0001f680 [Orchestrator] Starting Single Iterative Research for: '{query}'\n")

        initial_state: ResearchState = {
            "query": query,
            "current_section_title": "Research Summary",
            "current_section": query,
            "research_history": "",
            "research_complete": False,
            "current_gaps": [query],
            "pending_tool_tasks": [],
            "completed_sections": [],
            "loop_count": 0,
            "chat_history": self.sessions.get_messages(session_id),
            "extraction_matrix": [],
        }

        if workflow_mode:
            # Explicit Elicit command-bar workflow: the graph's Intent Classifier
            # maps it to the right route (and flags such as rag_only) itself.
            initial_state["workflow_mode"] = workflow_mode

        if intent_route in (CONVERSATIONAL, DEEP_RESEARCH):
            # Preset by the streaming intent gate: the graph's Intent Classifier
            # will respect this and skip its own LLM call.
            initial_state["intent_route"] = intent_route

        if target_doc_id:
            # "Chat with Paper": scope this entire run to one ingested document.
            # The Tool Router bypasses external search tools accordingly.
            initial_state["target_doc_id"] = target_doc_id
            initial_state["intent_route"] = DEEP_RESEARCH

        stream_id = None
        if progress_queue is not None:
            stream_id = register_progress_stream(progress_queue)
            initial_state["progress_stream_id"] = stream_id
            if intent_route == CONVERSATIONAL:
                progress_queue.put_nowait(sanitize_status("[Orchestrator] Composing a conversational reply."))
            else:
                progress_queue.put_nowait(sanitize_status("[Orchestrator] Starting single-agent iterative research."))

        try:
            final_state = await self.worker_graph.ainvoke(
                initial_state, config=self._thread_config(session_id)
            )
        finally:
            if stream_id:
                release_progress_stream(stream_id)

        report = final_state["completed_sections"][0]
        extraction_matrix = final_state.get("extraction_matrix") or []
        # Conversational replies and Research Agent clarification questions are
        # rendered as chat bubbles, not report cards.
        is_chat_turn = (
            final_state.get("intent_route") == CONVERSATIONAL
            or bool(final_state.get("needs_clarification"))
        )
        kind = "chat" if is_chat_turn else "research"

        # Record this turn (bubble + matrix) so follow-ups and the session
        # timeline can reference it.
        self.sessions.append_turn(
            session_id, query, report,
            extraction_matrix=extraction_matrix, kind=kind,
        )

        return {"report": report, "extraction_matrix": extraction_matrix, "kind": kind}

    async def _assurance_stream(self, query: str, session_id: str | None = None):
        """
        Stream an instant, Elicit-style intent acknowledgment from the fast model.
        Falls back to a deterministic sentence if the LLM call fails, so the UI
        always receives an assurance message first.
        """
        fallback = (
            f"I'll research \"{query}\" across the web and recent academic literature, "
            "then synthesize a cited report for you."
        )
        try:
            router = LLMRouter()
            llm = router.fast_model
            history = format_messages_for_prompt(
                self.sessions.get_messages(session_id), max_chars=2000
            )
            messages = [
                ("system", ASSURANCE_PROMPT),
                ("human", f"Prior conversation:\n{history}\n\nNew research request: {query}"),
            ]
            streamed_any = False
            async for chunk in llm.astream(messages):
                text = _chunk_text(chunk)
                if text:
                    streamed_any = True
                    yield text
            if not streamed_any:
                yield fallback
        except Exception:
            yield fallback

    async def stream_research(
        self,
        query: str,
        mode: str,
        session_id: str | None = None,
        target_doc_id: str | None = None,
        workflow_mode: str | None = None,
    ):
        """
        Streams structured events: assurance tokens first, then progress and
        tool_telemetry packets while workers run, then telemetry stats and the
        final report.
        """
        progress_queue: asyncio.Queue = asyncio.Queue()

        start_time = time.time()
        cache_hits = 0
        api_calls = 0

        if mode not in ("single", "deep"):
            yield {"event": "error", "data": "Invalid mode. Use 'single' or 'deep'."}
            return

        # 0. INTENT GATE: explicit command-bar workflows resolve deterministically;
        # otherwise classify so greetings never trigger assurance or tool loops.
        normalized_workflow = str(workflow_mode or "").strip().lower() or None
        workflow_route = resolve_workflow_route(normalized_workflow)

        if workflow_route:
            route = workflow_route
        elif target_doc_id:
            route = DEEP_RESEARCH
        else:
            route = await classify_intent(query, self.sessions.get_messages(session_id))

        if route == CONVERSATIONAL:
            research_task = asyncio.create_task(
                self.run_single_research(
                    query, progress_queue, session_id, None, intent_route=CONVERSATIONAL
                )
            )
        elif workflow_route:
            research_task = asyncio.create_task(
                self.run_single_research(
                    query, progress_queue, session_id, target_doc_id,
                    workflow_mode=normalized_workflow,
                )
            )
        elif mode == "single":
            research_task = asyncio.create_task(
                self.run_single_research(
                    query, progress_queue, session_id, target_doc_id, intent_route=DEEP_RESEARCH
                )
            )
        else:
            research_task = asyncio.create_task(
                self.run_deep_research(query, progress_queue, session_id)
            )

        # 1. ASSURANCE FIRST (research routes only): the workers are already
        # spinning up in the background while we stream the acknowledgment.
        # The Research Agent route may stream a clarification question instead,
        # so it also skips the assurance text.
        if route not in (CONVERSATIONAL, RESEARCH_AGENT):
            if target_doc_id:
                # Document focus mode: deterministic assurance (no LLM call) so the
                # message accurately reflects the isolated single-paper scope.
                focused_assurance = (
                    "I'll answer this using only your pinned document, retrieving "
                    "and citing the most relevant passages from that paper."
                )
                for word in focused_assurance.split(" "):
                    yield {"event": "assurance", "data": word + " "}
            else:
                async for token in self._assurance_stream(query, session_id):
                    yield {"event": "assurance", "data": token}

        # 2. LIVE PROGRESS + STRUCTURED TELEMETRY LOOP
        while not research_task.done():
            try:
                message = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            # Structured packets (e.g. tool_telemetry) pass through verbatim.
            if isinstance(message, dict) and "event" in message:
                yield message
                continue

            clean_msg = sanitize_status(message)

            # Dynamically Track Network vs Cache
            if "Cache hit" in clean_msg:
                cache_hits += 1
            elif "Searching" in clean_msg or "Evaluating" in clean_msg or "Querying" in clean_msg:
                api_calls += 1

            yield {"event": "progress", "data": clean_msg}

        # Drain the remaining queue
        while not progress_queue.empty():
            message = progress_queue.get_nowait()
            if isinstance(message, dict) and "event" in message:
                yield message
                continue
            clean_msg = sanitize_status(message)
            if "Cache hit" in clean_msg:
                cache_hits += 1
            yield {"event": "progress", "data": clean_msg}

        try:
            result = await research_task

            if result["kind"] == "chat":
                # Conversational turn: tokens already streamed as `message` events.
                # Send the full reply so the frontend can finalize the bubble.
                yield {"event": "complete", "data": result["report"]}
                return

            # CALCULATE FINAL TELEMETRY
            execution_time = round(time.time() - start_time, 2)
            tokens_saved = cache_hits * 4500  # Estimate of tokens saved by bypassing the scrape

            telemetry_payload = json.dumps({
                "execution_time": execution_time,
                "cache_hits": cache_hits,
                "api_calls": api_calls,
                "tokens_saved": tokens_saved
            })

            yield {"event": "telemetry", "data": telemetry_payload}
            # Unified structured payload for the Elicit grid, delivered once at
            # the end of the execution, right before the final report.
            yield {"event": "matrix", "data": result["extraction_matrix"]}
            yield {"event": "complete", "data": result["report"]}

        except Exception as e:
            yield {"event": "error", "data": sanitize_status(str(e))}
