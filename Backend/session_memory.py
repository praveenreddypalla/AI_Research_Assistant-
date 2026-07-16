"""
Session Memory Module
=====================
Lightweight conversational memory for the research engine.

Each frontend chat session (identified by a `session_id` generated in the browser)
maps to a sliding window of LangChain messages (HumanMessage / AIMessage pairs).
The orchestrator seeds every LangGraph worker with this history via the
`chat_history` state field (tracked with LangGraph's `add_messages` reducer),
so the Gap Analyzer and Synthesizer can reference previously generated reports
when the user asks a follow-up question.

Elicit upgrade: alongside the sliding LLM window, the store now persists the
full session timeline (every chat bubble and its associated extraction matrix)
grouped under a persistent `session_id`. This powers the "Recent Chat Sessions"
sidebar and full-session restore in the frontend.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Sliding window: keep the last N user/assistant turn pairs per session.
MAX_TURNS = 4
# Cap each stored message so old reports cannot blow up the LLM context window.
MAX_MESSAGE_CHARS = 6000
# Cap the persisted timeline so a very long-lived session cannot grow unbounded.
MAX_TIMELINE_MESSAGES = 200


def _utc_now() -> str:
    """ISO-8601 UTC timestamp for timeline bookkeeping."""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """In-process store mapping session_id -> LLM chat window + full timeline."""

    def __init__(self, max_turns: int = MAX_TURNS):
        self._sessions: Dict[str, List[BaseMessage]] = {}
        # Full persistent timeline per session (chat bubbles + extraction matrices).
        self._timelines: Dict[str, Dict[str, Any]] = {}
        self.max_turns = max_turns

    def get_messages(self, session_id: Optional[str]) -> List[BaseMessage]:
        """Return a copy of the stored chat history for a session (empty if unknown)."""
        if not session_id:
            return []
        return list(self._sessions.get(session_id, []))

    def append_turn(
        self,
        session_id: Optional[str],
        query: str,
        report: str,
        extraction_matrix: Optional[List[Dict[str, Any]]] = None,
        kind: str = "research",
    ) -> None:
        """
        Record one completed turn (user message + assistant reply) in both the
        sliding LLM window and the persistent session timeline.

        Args:
            session_id: Stable browser-generated session identifier.
            query: The user's message for this turn.
            report: The assistant output (Markdown report or chat reply).
            extraction_matrix: Structured per-document records for this turn.
            kind: 'research' (report + matrix) or 'chat' (conversational reply).
        """
        if not session_id:
            return

        # 1. Sliding LLM window (unchanged behaviour).
        history = self._sessions.setdefault(session_id, [])
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=(report or "")[:MAX_MESSAGE_CHARS]))
        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            del history[: len(history) - max_messages]

        # 2. Full persistent timeline (chat bubbles + extraction matrices).
        now = _utc_now()
        timeline = self._timelines.setdefault(session_id, {
            "session_id": session_id,
            "title": (query or "Untitled session").strip()[:80] or "Untitled session",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        })
        timeline["messages"].append({
            "role": "user",
            "content": query,
            "kind": kind,
            "extraction_matrix": [],
            "timestamp": now,
        })
        timeline["messages"].append({
            "role": "assistant",
            "content": report or "",
            "kind": kind,
            "extraction_matrix": extraction_matrix or [],
            "timestamp": now,
        })
        if len(timeline["messages"]) > MAX_TIMELINE_MESSAGES:
            del timeline["messages"][: len(timeline["messages"]) - MAX_TIMELINE_MESSAGES]
        timeline["updated_at"] = now

    def get_timeline(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the full timeline for a session, or None if unknown."""
        if not session_id:
            return None
        return self._timelines.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return session summaries sorted by most recently updated first."""
        summaries = [
            {
                "session_id": timeline["session_id"],
                "title": timeline["title"],
                "created_at": timeline["created_at"],
                "updated_at": timeline["updated_at"],
                "turns": len(timeline["messages"]) // 2,
            }
            for timeline in self._timelines.values()
        ]
        return sorted(summaries, key=lambda s: s["updated_at"], reverse=True)

    def clear(self, session_id: Optional[str]) -> None:
        """Drop all memory (window + timeline) for a session."""
        if session_id:
            self._sessions.pop(session_id, None)
            self._timelines.pop(session_id, None)


def format_messages_for_prompt(messages: List[BaseMessage], max_chars: int = 8000) -> str:
    """Render chat messages as a plain-text transcript for prompt injection."""
    if not messages:
        return "No prior conversation."

    lines = []
    for message in messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        content = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{role}: {content}")

    text = "\n\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text
