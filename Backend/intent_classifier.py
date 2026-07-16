"""
Intent Classifier Node
======================
The entry-point node of the LangGraph worker. Classifies the user's message into:

- CONVERSATIONAL: greetings, small talk, system commands, or conversational
  follow-ups. Routed to `conversational_node`, which streams a standard chat
  reply and updates `chat_history`.
- DEEP_RESEARCH: core research queries. Routed into the existing
  Gap Analyzer multi-agent research loop.

Orchestrators may preset `intent_route` in the initial state (e.g. deep-mode
section workers are always DEEP_RESEARCH); in that case the LLM call is skipped.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from core_engine.llm_router import LLMRouter
from core_engine.session_memory import format_messages_for_prompt
from core_engine.utilities.progress import emit_progress

# Route constants shared across the engine.
CONVERSATIONAL = "CONVERSATIONAL"
DEEP_RESEARCH = "DEEP_RESEARCH"
RESEARCH_AGENT = "RESEARCH_AGENT"
SYSTEMATIC_REVIEW = "SYSTEMATIC_REVIEW"

# Elicit command-bar workflow -> graph route mapping.
# 'find_papers' and 'extract_data' reuse the parallel search + extraction
# matrix loop; 'chat_with_papers' also sets rag_only so the Tool Router
# bypasses all external search tools; 'report' runs the loop but exits via
# heavy synthesis (long-form literature review, no matrix).
WORKFLOW_ROUTE_MAP = {
    "find_papers": DEEP_RESEARCH,
    "extract_data": DEEP_RESEARCH,
    "chat_with_papers": DEEP_RESEARCH,
    "report": DEEP_RESEARCH,
    "research_agent": RESEARCH_AGENT,
    "systematic_review": SYSTEMATIC_REVIEW,
}


def resolve_workflow_route(workflow_mode) -> str | None:
    """Map an explicit command-bar workflow to a graph route (None if unknown/absent)."""
    if not workflow_mode:
        return None
    return WORKFLOW_ROUTE_MAP.get(str(workflow_mode).strip().lower())


# --- 1. PYDANTIC SCHEMA ---
class IntentClassification(BaseModel):
    """Structured output schema for the intent gate."""
    route: Literal["CONVERSATIONAL", "DEEP_RESEARCH"] = Field(
        description=(
            "CONVERSATIONAL for greetings, small talk, thanks, meta questions, or system "
            "commands; DEEP_RESEARCH for anything requiring new research or a report"
        )
    )


# --- 2. THE SYSTEM PROMPT ---
# NOTE: Keep this prompt free of literal curly braces; they break LangChain templates.
INTENT_PROMPT = """
You are the intent gate for an autonomous research assistant.
Classify the user's latest message into exactly one route.

Routes:
- CONVERSATIONAL: greetings, small talk, thanks, meta questions about the assistant, system commands, or short conversational follow-ups that do not require gathering new evidence.
- DEEP_RESEARCH: any request that requires researching a topic, finding sources or papers, comparing evidence, or producing a report.

When in doubt, choose DEEP_RESEARCH.
"""

# Deterministic fallback markers used when the LLM call fails.
_GREETING_MARKERS = (
    "hi", "hello", "hey", "yo", "thanks", "thank you", "good morning",
    "good afternoon", "good evening", "who are you", "what can you do",
    "help", "ok", "okay", "cool", "nice", "great", "bye", "goodbye",
)


def _heuristic_route(query: str) -> str:
    """Deterministic fallback when the LLM classification fails. Biased to DEEP_RESEARCH."""
    text = (query or "").strip().lower().rstrip("!.?")
    if not text:
        return CONVERSATIONAL
    if text in _GREETING_MARKERS:
        return CONVERSATIONAL
    if len(text.split()) <= 3 and any(text.startswith(marker) for marker in _GREETING_MARKERS):
        return CONVERSATIONAL
    return DEEP_RESEARCH


async def classify_intent(query: str, chat_history: Optional[List] = None) -> str:
    """
    Classify a message with the fast model; fall back to the deterministic
    heuristic on any failure so the pipeline never stalls on the intent gate.
    """
    try:
        router = LLMRouter()
        structured_llm = router.fast_model.with_structured_output(IntentClassification)
        prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_PROMPT),
            ("human", "Prior conversation:\n{conversation}\n\nLatest user message: {query}\n\nClassify the route."),
        ])
        chain = prompt | structured_llm
        result: IntentClassification = await chain.ainvoke({
            "conversation": format_messages_for_prompt(chat_history or [], max_chars=2000),
            "query": query,
        })
        if result and result.route in (CONVERSATIONAL, DEEP_RESEARCH):
            return result.route
    except Exception as e:
        print(f"[Intent Classifier] LLM classification failed ({e}). Using heuristic fallback.")
    return _heuristic_route(query)


# --- 3. THE LANGGRAPH NODE ---
async def intent_classifier_node(state: dict):
    """LangGraph entry node: resolve workflow, preset, or classified intent route."""
    # 1. Explicit workflow selection from the Elicit command bar wins outright.
    workflow_mode = str(state.get("workflow_mode") or "").strip().lower()
    workflow_route = resolve_workflow_route(workflow_mode)
    if workflow_route:
        updates: dict = {"intent_route": workflow_route}
        if workflow_mode == "chat_with_papers":
            # Force local-RAG-only tool routing (no external web/arXiv search).
            updates["rag_only"] = True
        print(f"[Intent Classifier] Workflow '{workflow_mode}' routed to: {workflow_route}")
        return updates

    # 2. Preset route (deep-mode sections, paper focus, SSE gate).
    preset = state.get("intent_route")
    if preset in (CONVERSATIONAL, DEEP_RESEARCH, RESEARCH_AGENT, SYSTEMATIC_REVIEW):
        # Orchestrator already decided (deep-mode sections, paper focus, SSE gate).
        return {"intent_route": preset}

    emit_progress(state, "[Intent Classifier] Determining whether this is a chat message or a research request.")
    route = await classify_intent(state.get("query", ""), state.get("chat_history"))
    print(f"[Intent Classifier] Routed message to: {route}")
    return {"intent_route": route}
