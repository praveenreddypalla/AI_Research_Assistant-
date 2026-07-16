import os
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from werkzeug.utils import secure_filename

from core_engine.orchestrator import ResearchOrchestrator
from core_engine.utilities.vector_db import ingest_pdf_to_chroma
from core_engine.utilities.progress import sanitize_status

# Initialize the FastAPI App
app = FastAPI(title="Agentic AI Research API")

# --- CORS CONFIGURATION ---
# This allows your local HTML/JS frontend to talk to this Python server 
# without getting blocked by the browser's security policies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Engine once when the server starts
orchestrator = ResearchOrchestrator()

# Valid Elicit command-bar workflow selections (TOOLS + WORKFLOWS).
ALLOWED_WORKFLOWS = {
    "find_papers",
    "chat_with_papers",
    "extract_data",
    "research_agent",
    "report",
    "systematic_review",
}

# --- SCHEMAS ---
class ResearchRequest(BaseModel):
    query: str
    mode: str  # Must be 'single' or 'deep'
    # Conversational continuity: the frontend generates a stable session_id per
    # chat and sends it with every turn so follow-ups reference prior reports.
    session_id: Optional[str] = None
    # Elicit command-bar selection: one of ALLOWED_WORKFLOWS, or None for the
    # legacy intent-classified behaviour.
    workflow_mode: Optional[str] = None

class PaperResearchRequest(BaseModel):
    """Isolated "Chat with Paper" request: answers come only from one ingested document."""
    query: str
    # The document ID returned by /api/upload-pdf; retrieval is filtered to it.
    target_doc_id: str
    session_id: Optional[str] = None
    workflow_mode: Optional[str] = None


def normalize_workflow_mode(workflow_mode: Optional[str]) -> Optional[str]:
    """Validate and normalize the command-bar workflow selection."""
    if not workflow_mode:
        return None
    normalized = workflow_mode.strip().lower()
    if normalized not in ALLOWED_WORKFLOWS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid workflow_mode. Use one of: {', '.join(sorted(ALLOWED_WORKFLOWS))}.",
        )
    return normalized

# --- 1. RESEARCH ENDPOINT ---
@app.post("/api/research")
async def generate_research(request: ResearchRequest):
    """Takes a query from the frontend and fires the LangGraph architecture."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
        
    workflow_mode = normalize_workflow_mode(request.workflow_mode)

    try:
        if workflow_mode or request.mode == "single":
            # Explicit workflows always run through the single worker graph;
            # the Intent Classifier maps workflow_mode to the right route.
            result = await orchestrator.run_single_research(
                request.query, session_id=request.session_id, workflow_mode=workflow_mode
            )
        elif request.mode == "deep":
            result = await orchestrator.run_deep_research(request.query, session_id=request.session_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'single' or 'deep'.")
            
        return {
            "status": "success",
            "report": result["report"],
            "extraction_matrix": result["extraction_matrix"],
            "kind": result["kind"],
        }
    
    except Exception as e:
        print(f"\u274c API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def format_sse_event(event: str, payload: dict) -> str:
    """Format a dictionary payload as a Server-Sent Event chunk."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


async def research_event_stream(
    query: str,
    mode: str,
    session_id: Optional[str] = None,
    target_doc_id: Optional[str] = None,
    workflow_mode: Optional[str] = None,
):
    """Yield assurance tokens, tool telemetry, progress updates, and the final report as SSE chunks."""
    try:
        async for update in orchestrator.stream_research(query, mode, session_id, target_doc_id, workflow_mode):
            event_type = update["event"]
            data = update["data"]

            if event_type == "assurance":
                # Raw token stream. Never run tokens through sanitize_status:
                # trimming would destroy the whitespace between streamed words.
                yield format_sse_event("assurance", {"token": data})

            elif event_type == "tool_telemetry":
                # Structured JSON packet describing the exact concurrent tool
                # queries selected by the Tool Router (rendered as an accordion).
                yield format_sse_event("tool_telemetry", data)

            elif event_type == "message":
                # Conversational route: standard chat text tokens streamed as-is.
                yield format_sse_event("message", data)

            elif event_type == "matrix":
                # The completed extraction matrix drops down at the end of the
                # execution as one unified payload for the Elicit grid.
                yield format_sse_event("matrix", {"extraction_matrix": data})

            elif event_type == "progress":
                yield format_sse_event("progress", {"message": sanitize_status(data)})
            
            elif event_type == "telemetry":
                # Final run statistics: parse the JSON string back to a dict
                # and safely route it to the frontend dashboard.
                yield format_sse_event("telemetry", json.loads(data))
                
            elif event_type == "complete":
                yield format_sse_event("complete", {"report": data})
                
            else:
                # Only trigger an error if the event type is completely unknown
                yield format_sse_event("error", {"message": sanitize_status(data)})
                return
                
    except Exception as e:
        yield format_sse_event("error", {"message": sanitize_status(str(e))})


@app.post("/api/research-stream")
async def stream_research(request: ResearchRequest):
    """Streams LangGraph progress updates and the final Markdown report via SSE."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    if request.mode not in {"single", "deep"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'single' or 'deep'.")

    workflow_mode = normalize_workflow_mode(request.workflow_mode)

    return StreamingResponse(
        research_event_stream(request.query, request.mode, request.session_id, None, workflow_mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- 1b. CHAT WITH PAPER ENDPOINT ---
@app.post("/api/research-paper-stream")
async def stream_paper_research(request: PaperResearchRequest):
    """
    Isolated "Chat with Paper" SSE stream.
    Feeds `target_doc_id` into the initial LangGraph state so the Tool Router
    bypasses all external search tools and answers exclusively from the local
    RAG store, filtered to the pinned document.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    if not request.target_doc_id.strip():
        raise HTTPException(status_code=400, detail="target_doc_id is required for paper-focused research.")

    workflow_mode = normalize_workflow_mode(request.workflow_mode)

    return StreamingResponse(
        research_event_stream(request.query, "single", request.session_id, request.target_doc_id, workflow_mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- 1c. SESSION TIMELINE ENDPOINTS ---
@app.get("/api/sessions")
async def list_sessions():
    """List recent chat session summaries for the sidebar."""
    return {"sessions": orchestrator.sessions.list_sessions()}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Return the full timeline (chat bubbles + extraction matrices) for one session."""
    timeline = orchestrator.sessions.get_timeline(session_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return timeline


# --- 2. RAG UPLOAD ENDPOINT ---
@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Receives a PDF from the frontend, saves it temporarily, and ingests it into ChromaDB."""
    original_filename = file.filename or ""
    safe_filename = secure_filename(original_filename)

    if not safe_filename or Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    doc_id = uuid.uuid4().hex
    unique_filename = f"{doc_id}_{safe_filename}"
    temp_dir = tempfile.mkdtemp(prefix="agentic_upload_")
    temp_file_path = os.path.join(temp_dir, unique_filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # Fire your custom Vector DB ingestion utility, tagging every chunk
        # with doc_id so "Chat with Paper" mode can filter retrieval.
        ingest_pdf_to_chroma(temp_file_path, doc_id=doc_id)
        
        return {
            "status": "success",
            "message": f"'{safe_filename}' successfully ingested into Vector Database!",
            "doc_id": doc_id,
            "filename": safe_filename,
        }
        
    except Exception as e:
        print(f"\u274c Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Ensure cleanup happens even if ingestion crashes
        shutil.rmtree(temp_dir, ignore_errors=True)

# To run this server, use the command: uvicorn api:app --reload
