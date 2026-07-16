"""
Research Agent Node
===================
Handles the 'Research agent' workflow selected from the Elicit command bar.

Before firing the deep research loop, it performs a clarification check:
- If the query is too broad or ambiguous, it streams a conversational
  clarifying question back to the user (as standard `message` chat tokens,
  e.g. "Are you focusing on a specific demographic or time period?") and ends
  the turn without running any research tools.
- If the query is specific enough, or the user has just answered a prior
  clarifying question, it refines the research question and routes into the
  deep search tools (Gap Analyzer loop).
"""

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from core_engine.llm_router import LLMRouter
from core_engine.session_memory import format_messages_for_prompt
from core_engine.utilities.progress import emit_event, emit_progress


# --- 1. PYDANTIC SCHEMA ---
class ClarificationCheck(BaseModel):
    """Structured output for the intake clarification gate."""
    needs_clarification: bool = Field(
        description="True only if the request is too broad or ambiguous to research well"
    )
    clarifying_question: str = Field(
        default="",
        description="One short, friendly question asking for the missing parameters (only when needs_clarification is true)",
    )
    refined_query: str = Field(
        default="",
        description="A single precise research question combining the request with any clarification answers from the conversation (only when needs_clarification is false)",
    )


# --- 2. THE SYSTEM PROMPT ---
# NOTE: Keep this prompt free of literal curly braces; they break LangChain templates.
CLARIFICATION_PROMPT = """
You are the intake step of an autonomous research agent.
Decide whether the user's research request is specific enough to start deep research.

Rules:
1. If the request is too broad or ambiguous (no clear scope, population, demographic, time period, domain, or outcome), set needs_clarification to true and write ONE short, friendly clarifying question asking for the missing parameters. Example: are you focusing on a specific demographic, time period, or subfield?
2. If the prior conversation shows the assistant already asked a clarifying question and the user has now answered it, set needs_clarification to false. Never ask twice.
3. If the request is already specific, set needs_clarification to false.
4. When needs_clarification is false, produce refined_query: a single, precise research question that combines the request with any clarification answers from the conversation.
"""


# --- 3. THE LANGGRAPH NODE ---
async def research_agent_node(state: dict):
    """Clarification gate: ask for parameters when the query is too broad, else refine and research."""
    query = state.get("query", "")
    conversation = format_messages_for_prompt(state.get("chat_history", []) or [])
    emit_progress(state, "[Research Agent] Checking whether the request needs clarification.")
    print(f"[Research Agent] Clarification check for: '{query}'")

    try:
        router = LLMRouter()
        structured_llm = router.fast_model.with_structured_output(ClarificationCheck)
        prompt = ChatPromptTemplate.from_messages([
            ("system", CLARIFICATION_PROMPT),
            ("human", "Prior conversation:\n{conversation}\n\nResearch request: {query}"),
        ])
        check: ClarificationCheck = await (prompt | structured_llm).ainvoke({
            "conversation": conversation,
            "query": query,
        })
    except Exception as e:
        # Fail open: never block research on a broken clarification check.
        print(f"[Research Agent] Clarification check failed ({e}). Proceeding to research.")
        check = ClarificationCheck(needs_clarification=False)

    if check.needs_clarification and check.clarifying_question.strip():
        question = check.clarifying_question.strip()
        print(f"[Research Agent] Asking for clarification: {question}")
        emit_progress(state, "[Research Agent] Asking a clarifying question before researching.")
        # Stream the question as standard chat text tokens (event: message).
        for word in question.split(" "):
            emit_event(state, "message", {"token": word + " "})
        return {
            "needs_clarification": True,
            "research_complete": True,
            "completed_sections": [question],
            "extraction_matrix": [],
            # add_messages reducer records the clarification turn so the next
            # user reply carries the full context back into this node.
            "chat_history": [HumanMessage(content=query), AIMessage(content=question)],
        }

    refined = check.refined_query.strip() or query
    print(f"[Research Agent] Query is specific enough. Researching: '{refined}'")
    emit_progress(state, "[Research Agent] Request is specific enough. Starting deep research.")
    return {
        "needs_clarification": False,
        "current_section": refined,
        "current_gaps": [refined],
    }
