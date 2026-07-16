"""
Progress Streaming Utilities
============================
Shared helpers for emitting clean, UI-safe progress messages from LangGraph nodes.

Checkpoint-safe design: live asyncio.Queue objects are held in a module-level
registry keyed by an opaque string (`progress_stream_id`). Graph state only
carries the string, which keeps the state fully serializable for LangGraph
checkpointers while nodes can still stream events in real time.
"""

import re
import uuid
from typing import Any

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
EMOJI_SYMBOL_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u27BF"
    "\u200D"
    "\uFE0F"
    "]+",
    flags=re.UNICODE,
)
NON_ASCII_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
WHITESPACE_RE = re.compile(r"[ \t]+")

# --- LIVE STREAM REGISTRY (checkpoint-safe queue lookup) ---
_PROGRESS_STREAMS: dict[str, Any] = {}


def register_progress_stream(queue: Any) -> str:
    """Register a live queue and return the serializable id to place in graph state."""
    stream_id = uuid.uuid4().hex
    _PROGRESS_STREAMS[stream_id] = queue
    return stream_id


def release_progress_stream(stream_id: str) -> None:
    """Remove a queue from the registry once its research run has finished."""
    _PROGRESS_STREAMS.pop(stream_id, None)


def _resolve_queue(state: dict[str, Any]) -> Any:
    """Find the active stream queue via the legacy direct handle or the registry."""
    queue = state.get("progress_queue")
    if queue is not None:
        return queue
    stream_id = state.get("progress_stream_id")
    if stream_id:
        return _PROGRESS_STREAMS.get(stream_id)
    return None


def sanitize_status(message: str) -> str:
    """
    Strip terminal color codes, emojis, Unicode symbols, and control characters.
    Returns clean ASCII text suitable for enterprise UI progress messages.
    """
    cleaned = ANSI_ESCAPE_RE.sub("", str(message))
    cleaned = EMOJI_SYMBOL_RE.sub("", cleaned)
    cleaned = CONTROL_RE.sub("", cleaned)
    cleaned = NON_ASCII_RE.sub("", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def emit_progress(state: dict[str, Any], message: str) -> None:
    """Push a sanitized progress message into the active stream queue, if present."""
    queue = _resolve_queue(state)
    if queue is None:
        return

    cleaned = sanitize_status(message)
    if cleaned:
        queue.put_nowait(cleaned)


def emit_event(state: dict[str, Any], event: str, payload: dict) -> None:
    """
    Push a structured event (e.g. `tool_telemetry`) into the active stream queue.
    The orchestrator forwards these dict packets verbatim to the SSE endpoint.
    """
    queue = _resolve_queue(state)
    if queue is None:
        return
    queue.put_nowait({"event": event, "data": payload})
