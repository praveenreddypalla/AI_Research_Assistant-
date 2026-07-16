"""
Global State Management Module
==============================
Defines the `ResearchState` schema utilized by the LangGraph state machine.
By centralizing the state definition here, we adhere to the Dependency Inversion Principle,
allowing all independent cognitive nodes to import the state schema without triggering
circular dependency errors in Python.
"""

import operator
from typing import TypedDict, Annotated, List, Any, Dict, Optional
from typing_extensions import NotRequired

from langgraph.graph.message import add_messages

# --- GLOBAL LANGGRAPH STATE ---

class ResearchState(TypedDict):
    """
    The shared memory dictionary passed between all LangGraph nodes during a single iteration.
    Acts as the ephemeral context window for the autonomous agent.
    """
    
    # 1. Core Identifiers: Defines the current scope of the worker
    query: str
    current_section_title: str
    current_section: str
    
    # 2. Accumulators: 
    # The `Annotated[..., operator.add]` syntax is crucial for LangGraph. 
    # It instructs the state machine to append new text to `research_history` 
    # rather than overwriting it during subsequent loop iterations.
    research_history: Annotated[str, operator.add] 
    
    # 3. Decision Flags: Controls the cyclical execution flow
    research_complete: bool
    current_gaps: List[str]
    
    # 4. Action Queues: Passes instructions from the Router to the Executor
    pending_tool_tasks: List[Any]
    completed_sections: List[str]
    
    # 5. Safety Limits: Prevents infinite API loops
    loop_count: int

    # 6. Optional Streaming (legacy path): Carries progress events to the SSE endpoint.
    # NOTE: Do not set this when the graph is compiled with a checkpointer; an
    # asyncio.Queue is not serializable. Prefer `progress_stream_id` below.
    progress_queue: NotRequired[Any]

    # 6b. Checkpoint-safe streaming handle: a string key resolved to the live
    # asyncio.Queue through the registry in `core_engine.utilities.progress`.
    progress_stream_id: NotRequired[str]

    # 7. Conversational Memory: Sliding window of prior chat turns for this session.
    # Tracked with LangGraph's `add_messages` reducer so nodes can reference
    # previously generated reports when handling follow-up questions.
    chat_history: NotRequired[Annotated[List[Any], add_messages]]

    # 8. Document Focus ("Chat with Paper"): When set, the Tool Router bypasses
    # external search tools (web_searcher, web_crawler, arxiv_researcher) and
    # routes exclusively to the local RAG retriever, with retrieval filtered to
    # this document ID's chunks. TypedDicts cannot carry runtime defaults, so
    # NotRequired[Optional[str]] is the schema equivalent of `= None`.
    target_doc_id: NotRequired[Optional[str]]

    # 9. Intent Routing: Set by the Intent Classifier entry node. One of
    # "CONVERSATIONAL" (chat reply path) or "DEEP_RESEARCH" (multi-agent loop).
    # Orchestrators may preset this to skip re-classification (e.g. deep-mode
    # section workers are always DEEP_RESEARCH).
    intent_route: NotRequired[str]

    # 10. Extraction Matrix: Structured per-document records produced by the
    # Synthesizer (Elicit-style grid). Each record carries: title,
    # url_or_source, abstract_summary, methodology, sample_size, key_findings.
    extraction_matrix: NotRequired[List[Dict[str, Any]]]

    # 11. Workflow Selection (Elicit command bar): one of 'find_papers',
    # 'chat_with_papers', 'extract_data', 'research_agent', 'report', or
    # 'systematic_review'. Mapped to a graph route by the Intent Classifier.
    workflow_mode: NotRequired[str]

    # 12. RAG-only flag ('Chat with papers' workflow): forces the Tool Router
    # to bypass all external search tools even without a pinned target_doc_id.
    rag_only: NotRequired[bool]

    # 13. Clarification flag (Research Agent workflow): True when the agent
    # asked the user a clarifying question instead of running the research loop.
    needs_clarification: NotRequired[bool]
