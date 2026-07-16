"""
Conversational Node
===================
Handles the CONVERSATIONAL route selected by the Intent Classifier.
Streams a standard chat reply token-by-token through the checkpoint-safe
progress stream registry (forwarded by the API as `event: message`) and
records the turn in `chat_history` via the add_messages reducer.
"""

from langchain_core.messages import AIMessage, HumanMessage

from core_engine.llm_router import LLMRouter
from core_engine.session_memory import format_messages_for_prompt
from core_engine.utilities.progress import emit_event, emit_progress

CONVERSATIONAL_PROMPT = (
    "You are A.R.I.A., a professional autonomous research assistant. "
    "The user's latest message is conversational rather than a research request. "
    "Reply naturally, briefly, and helpfully in plain Markdown. "
    "You may reference the prior conversation, including previously generated reports. "
    "If the user seems to want new research, invite them to phrase a research question. "
    "Never fabricate research findings."
)


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


async def conversational_node(state: dict):
    """Stream a chat reply and finish the graph without touching research tools."""
    query = state.get("query", "")
    conversation = format_messages_for_prompt(state.get("chat_history", []) or [])
    emit_progress(state, "[Conversation] Composing a reply.")
    print(f"[Conversation] Handling conversational turn: '{query}'")

    fallback = (
        "Hello! I'm A.R.I.A., your research assistant. "
        "Ask me to research any topic and I'll synthesize a cited report for you."
    )

    response_parts: list = []
    try:
        router = LLMRouter()
        llm = router.fast_model
        messages = [
            ("system", CONVERSATIONAL_PROMPT),
            ("human", f"Prior conversation:\n{conversation}\n\nUser message: {query}"),
        ]
        async for chunk in llm.astream(messages):
            text = _chunk_text(chunk)
            if text:
                response_parts.append(text)
                # Standard chat tokens: the API forwards these as `event: message`.
                emit_event(state, "message", {"token": text})
    except Exception as e:
        print(f"[Conversation] Streaming failed ({e}). Using fallback reply.")

    reply = "".join(response_parts).strip() or fallback
    if not response_parts:
        # Nothing was streamed; push the fallback so the UI still receives text.
        emit_event(state, "message", {"token": reply})

    return {
        "completed_sections": [reply],
        "extraction_matrix": [],
        "research_complete": True,
        # add_messages reducer appends this turn to the in-graph chat history.
        "chat_history": [HumanMessage(content=query), AIMessage(content=reply)],
    }
