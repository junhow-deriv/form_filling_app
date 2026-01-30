"""
FastAPI server for PDF form filling.

This is the main entry point. Run with:
    uvicorn main:app --reload

Endpoints:
    POST /analyze            - Upload PDF, get detected form fields
    POST /fill-agent         - Fill form fields (agent mode with tools) [RECOMMENDED]
    POST /fill-agent-stream  - Fill form fields with real-time streaming [RECOMMENDED]
    POST /fill               - Fill form fields (single-shot LLM mode) [LEGACY]
    GET  /                   - Serve the web UI

Note: The agent mode endpoints are recommended for production use. They provide
better accuracy, error recovery, and support for multi-turn conversations.
The single-shot /fill endpoint is maintained for backwards compatibility.
"""

import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_processor import detect_form_fields, edit_pdf_with_instructions, get_form_summary
from llm import map_instructions_to_fields
from agent import run_agent, run_agent_stream, AGENT_SDK_AVAILABLE, AGENT_SDK_ERROR, _session_manager
from parser import (
    parse_files_stream, needs_parsing, is_simple_text,
    DOCLING_AVAILABLE, DOCLING_ERROR, ParsedFile,
    estimate_file_chars
)
from services.query_generator import generate_field_queries
from services.embedding_service import (
    store_document,
    waterfall_search,
    assemble_waterfall_context
)


# ============================================================================
# Token Budget Constants
# ============================================================================

# Token budget for context files (60% of Claude's 200k context window)
MAX_CONTEXT_TOKENS = 120_000
CHARS_PER_TOKEN = 4  # Rough estimate for English text
MAX_CONTEXT_CHARS = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN  # 480,000


# ============================================================================
# App Setup
# ============================================================================

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="PDF Form Filler",
    description="Fill PDF forms using natural language instructions",
    version="0.1.0"
)


# Background task to cleanup old sessions periodically
import asyncio

async def periodic_session_cleanup():
    """
    Run form state cleanup every hour.
    
    Note: pg_cron handles automatic cleanup, but this provides a fallback
    and cleans up in-memory sessions.
    """
    while True:
        await asyncio.sleep(3600)  # 1 hour
        try:
            # Clean up form states older than 24 hours (pg_cron also does this)
            _session_manager.cleanup_old_form_states(max_age_seconds=86400)
        except Exception as e:
            print(f"[Cleanup] Error during periodic cleanup: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup."""
    asyncio.create_task(periodic_session_cleanup())
    print("[App] Started periodic form state cleanup task (every 1 hour, cleaning states older than 24 hours)")
    print("[App] Note: pg_cron also handles automatic cleanup in the database")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================

async def check_user_has_documents(user_id: str, session_id: str) -> bool:
    """
    Check if user has any documents (ephemeral or global KB).
    
    Returns True if:
    - User has ephemeral docs for this session_id, OR
    - User has documents in global knowledge base
    
    Args:
        user_id: User UUID
        session_id: Session UUID for ephemeral docs
        
    Returns:
        True if user has any documents, False otherwise
    """
    from database.supabase_client import get_client_for_user
    
    try:
        client = get_client_for_user(user_id)
        
        # Check ephemeral docs for this session
        ephemeral = client.table("documents").select("id").eq("user_id", user_id).eq("session_id", session_id).limit(1).execute()
        if ephemeral.data:
            print(f"[HasDocs] User {user_id} has ephemeral docs for session {session_id}")
            return True
        
        # Check global KB (session_id IS NULL)
        kb_docs = client.table("documents").select("id").eq("user_id", user_id).is_("session_id", "null").limit(1).execute()
        if kb_docs.data:
            print(f"[HasDocs] User {user_id} has global KB documents")
            return True
        
        print(f"[HasDocs] User {user_id} has no documents")
        return False
    except Exception as e:
        print(f"[HasDocs] Error checking documents: {e}")
        return False


# ============================================================================
# API Models
# ============================================================================

class FieldInfo(BaseModel):
    field_id: str
    field_type: str
    page: int
    label_context: str
    friendly_label: Optional[str] = None
    current_value: Optional[str] = None
    options: Optional[list[str]] = None


class AnalyzeResponse(BaseModel):
    success: bool
    message: str
    fields: list[FieldInfo]
    field_count: int


class FillRequest(BaseModel):
    instructions: str
    use_llm: bool = True  # Set to False to use simple keyword mapping


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/analyze")
async def analyze_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Analyze a PDF to detect fillable form fields (streaming version).
    
    Returns SSE stream with analysis progress and results.
    Creates or retrieves a session and saves system messages for persistence.
    
    Args:
        file: The PDF file to analyze
        session_id: Session ID for message persistence (required)
    """
    if not file.filename.lower().endswith('.pdf'):
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'error': 'File must be a PDF'})}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )
    
    pdf_bytes = await file.read()
    
    async def event_stream():
        from datetime import datetime
        
        user_id = "00000000-0000-0000-0000-000000000001"
        session = _session_manager.get_or_create_session(session_id, user_id)
        session.file_name = file.filename
        
        yield f"data: {json.dumps({'type': 'init', 'message': 'Analyzing PDF...'})}\n\n"
        
        try:
            fields = detect_form_fields(pdf_bytes)
            session.fields = fields
            
            field_infos = [
                {
                    'field_id': f.field_id,
                    'field_type': f.field_type.value,
                    'page': f.page,
                    'label_context': f.label_context,
                    'friendly_label': f.friendly_label,
                    'current_value': f.current_value,
                    'options': f.options
                }
                for f in fields
            ]
            
            yield f"data: {json.dumps({'type': 'fields_detected', 'fields': field_infos, 'field_count': len(fields)})}\n\n"
            
            if len(fields) > 0:
                system_message_content = f"Detected {len(fields)} fillable fields in the PDF"
            else:
                system_message_content = "No fillable form fields detected. Make sure this is a PDF with AcroForm fields."
            
            system_message = {
                'type': 'system_message',
                'role': 'system',
                'content': system_message_content,
                'timestamp': datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(system_message)}\n\n"
            
            session.messages = []
            session.messages.append({
                'role': 'system',
                'content': system_message_content,
                'timestamp': datetime.utcnow().isoformat()
            })
            _session_manager.save_session(session, user_id)
            print(f"[Analyze] Saved system message to session {session_id}")
            
            yield f"data: {json.dumps({'type': 'complete', 'success': True, 'field_count': len(fields)})}\n\n"
            
        except Exception as e:
            error_msg = f"Failed to analyze PDF: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
            
            if not hasattr(session, 'messages') or session.messages is None:
                session.messages = []
            session.messages.append({
                'role': 'system',
                'content': f"Error analyzing PDF: {str(e)}",
                'timestamp': datetime.utcnow().isoformat()
            })
            _session_manager.save_session(session, user_id)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/fill", deprecated=True)
async def fill_pdf(
    file: UploadFile = File(...),
    instructions: str = Form(...),
):
    """
    [LEGACY] Fill a PDF form using single-shot LLM mode.

    **DEPRECATED**: Use /fill-agent-stream for better accuracy and multi-turn support.

    This endpoint uses a single LLM call to map instructions to form fields.
    For complex forms or iterative refinement, use the agent endpoints instead.

    Args:
        file: The PDF file to fill
        instructions: Natural language description of what to fill
            e.g., "My name is John Doe, I live at 123 Main St,
                   my phone is 555-1234, and I agree to the terms"

    Returns:
        The filled PDF file as a download
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await file.read()
    
    # Step 1: Detect form fields
    try:
        fields = detect_form_fields(pdf_bytes)
    except Exception as e:
        raise HTTPException(500, f"Failed to analyze PDF: {str(e)}")
    
    if not fields:
        raise HTTPException(
            400, 
            "No fillable form fields found in this PDF. "
            "This endpoint only works with PDFs that have native AcroForm fields."
        )
    
    # Step 2: Map instructions to fields using LLM
    # Note: The simple keyword mapping (use_llm=False) is no longer supported.
    # Use the agent endpoints for better accuracy.
    try:
        edits = map_instructions_to_fields(instructions, fields)
    except ValueError as e:
        raise HTTPException(
            500,
            f"LLM error: {str(e)}. Make sure ANTHROPIC_API_KEY is set."
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to process instructions: {str(e)}")
    
    if not edits:
        raise HTTPException(
            400,
            "Could not determine which fields to fill from your instructions. "
            "Try being more specific, e.g., 'Name: John Doe, Email: john@example.com'"
        )
    
    # Step 3: Apply edits
    try:
        filled_pdf = edit_pdf_with_instructions(pdf_bytes, edits)
    except Exception as e:
        raise HTTPException(500, f"Failed to fill PDF: {str(e)}")
    
    # Return the filled PDF
    filename = file.filename.replace('.pdf', '_filled.pdf')
    
    return Response(
        content=filled_pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Fields-Filled": str(len(edits))
        }
    )


@app.post("/fill-preview", deprecated=True)
async def fill_pdf_preview(
    file: UploadFile = File(...),
    instructions: str = Form(...),
):
    """
    [LEGACY] Preview what fields would be filled without actually filling them.

    **DEPRECATED**: Use /fill-agent-stream for better accuracy.

    Useful for debugging and understanding how instructions are mapped in single-shot mode.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")

    pdf_bytes = await file.read()

    # Detect fields
    try:
        fields = detect_form_fields(pdf_bytes)
    except Exception as e:
        raise HTTPException(500, f"Failed to analyze PDF: {str(e)}")

    if not fields:
        return {
            "success": False,
            "message": "No fillable form fields found",
            "fields": [],
            "edits": []
        }

    # Map instructions using LLM
    try:
        edits = map_instructions_to_fields(instructions, fields)
    except ValueError as e:
        raise HTTPException(500, f"LLM error: {str(e)}")
    
    return {
        "success": True,
        "message": f"Would fill {len(edits)} of {len(fields)} fields",
        "fields": [f.to_dict() for f in fields],
        "edits": edits
    }


# ============================================================================
# Agent Mode Endpoint
# ============================================================================

@app.post("/fill-agent")
async def fill_pdf_agent(
    file: UploadFile = File(...),
    instructions: str = Form(...),
    max_iterations: int = Form(20),
):
    """
    Fill a PDF form using agent mode with tool calling (Claude Agent SDK).
    
    This mode uses an iterative agent that can:
    - Search and inspect fields
    - Validate values before setting
    - Review pending edits before committing
    - Recover from errors
    
    Requires Claude Code to be installed.
    
    Args:
        file: The PDF file to fill
        instructions: Natural language description of what to fill
        max_iterations: Maximum agent iterations (default 20)
    
    Returns:
        The filled PDF file as a download, plus agent execution summary
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await file.read()
    
    # Check for form fields first
    try:
        fields = detect_form_fields(pdf_bytes)
    except Exception as e:
        raise HTTPException(500, f"Failed to analyze PDF: {str(e)}")
    
    if not fields:
        raise HTTPException(
            400, 
            "No fillable form fields found in this PDF. "
            "This endpoint only works with PDFs that have native AcroForm fields."
        )
    
    # Run agent with Claude Agent SDK
    try:
        import tempfile
        import os as os_module
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        output_path = tmp_path.replace('.pdf', '_filled.pdf')
        
        try:
            # Use await since we're in an async context
            summary = await run_agent(tmp_path, instructions, output_path)
            
            if os_module.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    filled_pdf = f.read()
            else:
                raise HTTPException(500, "Agent did not produce output PDF")
        finally:
            if os_module.path.exists(tmp_path):
                os_module.unlink(tmp_path)
            if os_module.path.exists(output_path):
                os_module.unlink(output_path)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(500, f"Agent error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Agent failed: {str(e)}")
    
    # Handle different summary formats (SDK vs fallback)
    applied_count = summary.get("applied_count", 0)
    iterations = summary.get("iterations", summary.get("message_count", 0))
    
    if applied_count == 0:
        raise HTTPException(
            400,
            f"Agent could not fill any fields. Errors: {summary.get('errors', [])}"
        )
    
    # Return the filled PDF
    filename = file.filename.replace('.pdf', '_agent_filled.pdf')
    
    return Response(
        content=filled_pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Fields-Filled": str(applied_count),
            "X-Agent-Iterations": str(iterations),
        }
    )


@app.post("/fill-agent-preview")
async def fill_pdf_agent_preview(
    file: UploadFile = File(...),
    instructions: str = Form(...),
    max_iterations: int = Form(20),
):
    """
    Run agent mode and return execution summary without downloading the PDF.
    
    Useful for debugging and understanding how the agent processes the form.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await file.read()
    
    try:
        import tempfile
        import os as os_module
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        output_path = tmp_path.replace('.pdf', '_filled.pdf')
        
        try:
            # Use await since we're in an async context
            summary = await run_agent(tmp_path, instructions, output_path)
        finally:
            if os_module.path.exists(tmp_path):
                os_module.unlink(tmp_path)
            if os_module.path.exists(output_path):
                os_module.unlink(output_path)
                
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    
    return {
        "success": True,
        "message": f"Agent completed with {summary.get('message_count', 0)} messages",
        "result": summary.get("result", ""),
    }


# ============================================================================
# Streaming Agent Endpoint (SSE)
# ============================================================================

from fastapi.responses import StreamingResponse
import json
import asyncio

@app.post("/fill-agent-stream")
async def fill_pdf_agent_stream(
    file: UploadFile = File(...),
    instructions: str = Form(...),
    context_file_ids: Optional[str] = Form(None),  # JSON array of document IDs from /parse-files
    max_iterations: int = Form(20),
    is_continuation: bool = Form(False),
    previous_edits: Optional[str] = Form(None),  # JSON string of field_id -> value
    resume_session_id: Optional[str] = Form(None),  # Session ID from previous turn
    user_session_id: Optional[str] = Form(None),  # Unique ID for this user's form-filling session
    anthropic_api_key: Optional[str] = Form(None),  # User's Anthropic API key
):
    """
    Fill a PDF form using agent mode with real-time streaming.

    Returns Server-Sent Events (SSE) stream with agent messages.
    
    **Context Files**: Must be pre-uploaded via /parse-files endpoint.
    Pass the returned document IDs as a JSON array in context_file_ids parameter.

    Args:
        file: The PDF file to fill. For continuations, this should be the already-filled PDF.
        instructions: Natural language instructions for this turn
        context_file_ids: JSON array of document IDs from /parse-files (e.g., '["uuid1", "uuid2"]')
        is_continuation: Set to true for multi-turn conversations (subsequent messages)
        previous_edits: JSON string of {field_id: value} from previous turns
        resume_session_id: Session ID from previous turn to resume conversation context
        user_session_id: Unique ID for this user's form-filling session (for concurrent users)
        anthropic_api_key: User's Anthropic API key for Claude calls

    Event types:
    - init: Session initialized with field count
    - iteration: New iteration started
    - text: Agent thinking/response text
    - tool_start: Tool call started
    - tool_end: Tool call completed with result
    - complete: Agent finished (includes applied_edits, session_id, and user_session_id for tracking)
    - pdf_ready: Final summary with filled PDF (hex-encoded)
    - error: Error occurred
    """
    start_time = time.time()

    if not file.filename.lower().endswith('.pdf'):
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'error': 'File must be a PDF'})}\n\n"
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )
    
    # Check SDK availability early
    if not AGENT_SDK_AVAILABLE:
        async def sdk_error_stream():
            yield f"data: {json.dumps({'type': 'error', 'error': f'Claude Agent SDK not available: {AGENT_SDK_ERROR}. Install with: pip install claude-agent-sdk'})}\n\n"
        return StreamingResponse(
            sdk_error_stream(),
            media_type="text/event-stream"
        )
    
    pdf_bytes = await file.read()
    
    # Parse previous_edits JSON if provided
    parsed_previous_edits = None
    if previous_edits:
        try:
            parsed_previous_edits = json.loads(previous_edits)
        except json.JSONDecodeError:
            parsed_previous_edits = None

    async def event_stream():
        import tempfile
        import os as os_module

        tmp_path = None
        output_path = None

        # Send immediate acknowledgment
        cont_msg = " (continuation)" if is_continuation else ""
        yield f"data: {json.dumps({'type': 'init', 'message': f'Stream connected, initializing agent{cont_msg}...'})}\n\n"

        # Use default test user for development (TODO: get from auth)
        user_id = "00000000-0000-0000-0000-000000000001"

        try:
            pdf_to_use = pdf_bytes

            session = _session_manager.get_session(user_session_id, user_id)
            
            if is_continuation and user_session_id:
                # For continuations, load the previously filled PDF from session storage
                if session and session.current_pdf_bytes:
                    pdf_to_use = session.current_pdf_bytes
                    print(f"[FillAgentStream] ✅ Using stored filled PDF from session ({len(pdf_to_use)} bytes)")
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Loaded previously filled PDF from session...'})}\n\n"
                else:
                    print(f"[FillAgentStream] ⚠️ WARNING: Continuation requested but no stored PDF found, using uploaded PDF")
                    yield f"data: {json.dumps({'type': 'warning', 'message': 'No previous PDF found in session, using uploaded PDF'})}\n\n"
            else:
                print(f"[FillAgentStream] Using uploaded PDF ({len(pdf_to_use)} bytes)")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_to_use)
                tmp_path = tmp.name

            output_path = tmp_path.replace('.pdf', '_filled.pdf')

            # PRE-AGENT PROCESSING: Context generation
            intelligent_context = None
            
            if not is_continuation:
                # 1. Detect form fields BEFORE starting agent
                if session and session.fields:
                    fields = session.fields
                else:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Detecting form fields...'})}\n\n"
                    fields = detect_form_fields(pdf_bytes)
                    print(f"[FillAgentStream] Detected {len(fields)} form fields")
                
                fields_dict = [f.to_dict() for f in fields]
                yield f"data: {json.dumps({'type': 'fields_detected', 'fields': fields_dict, 'field_count': len(fields)})}\n\n"
                
                # 2. Check if user has any documents (ephemeral or KB)
                has_documents = await check_user_has_documents(user_id, user_session_id)
                
                # 3. Generate intelligent context if documents exist
                if has_documents:
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Generating search queries for form fields...'})}\n\n"
                    
                    # Parse context_file_ids if provided
                    has_uploaded_docs = False
                    if context_file_ids:
                        try:
                            doc_ids = json.loads(context_file_ids)
                            has_uploaded_docs = len(doc_ids) > 0
                            print(f"[FillAgentStream] Using {len(doc_ids)} pre-uploaded document(s)")
                        except json.JSONDecodeError:
                            print(f"[FillAgentStream] Failed to parse context_file_ids: {context_file_ids}")
                    
                    # Generate queries
                    field_queries = await generate_field_queries(
                        form_fields=fields,
                        user_instructions=instructions,
                        anthropic_api_key=anthropic_api_key,
                        has_uploaded_docs=has_uploaded_docs
                    )
                    
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Searching documents for relevant information...'})}\n\n"
                    
                    # Perform waterfall search
                    waterfall_results = await waterfall_search(
                        user_id=user_id,
                        session_id=user_session_id,
                        field_queries=field_queries,
                        similarity_threshold=0.7
                    )
                    
                    # Assemble context
                    intelligent_context = assemble_waterfall_context(waterfall_results)
                    
                    if intelligent_context:
                        print(f"[FillAgentStream] Generated intelligent context ({len(intelligent_context)} chars)")
                    else:
                        print(f"[FillAgentStream] No relevant context found from documents")

            yield f"data: {json.dumps({'type': 'status', 'message': f'Starting Claude Agent SDK...'})}\n\n"

            # Stream messages from Claude Agent SDK with continuation params
            # Pass original PDF bytes only for new sessions (not continuations)
            message_count = 0
            async for message in run_agent_stream(
                tmp_path,
                instructions,
                output_path,
                is_continuation=is_continuation,
                previous_edits=parsed_previous_edits,
                resume_session_id=resume_session_id,
                user_session_id=user_session_id,
                original_pdf_bytes=pdf_bytes if not is_continuation else None,
                intelligent_context=intelligent_context,
                anthropic_api_key=anthropic_api_key,
            ):
                message_count += 1
                # Convert message to JSON and send as SSE
                yield f"data: {json.dumps(message, default=str)}\n\n"
            
            if message_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Agent produced no messages - SDK may not be working'})}\n\n"
            
            # After streaming completes, check for output PDF
            if output_path and os_module.path.exists(output_path):
                # Read the filled PDF and include in final message
                with open(output_path, 'rb') as f:
                    pdf_hex = f.read().hex()
                yield f"data: {json.dumps({'type': 'pdf_ready', 'pdf_bytes': pdf_hex})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No output PDF generated'})}\n\n"
            
            end_time = time.time()
            yield f"data: {json.dumps({'type': 'status', 'message': f'Stream completed in {end_time - start_time:.2f} seconds'})}\n\n"
            print(f"[FillAgentStream] Stream completed in {end_time - start_time:.2f} seconds")
                
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Clean up temp files
            if tmp_path and os_module.path.exists(tmp_path):
                os_module.unlink(tmp_path)
            if output_path and os_module.path.exists(output_path):
                os_module.unlink(output_path)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# Static Files (Web UI)
# ============================================================================

# Serve the frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
async def serve_index():
    """Serve the main web UI."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "PDF Form Filler API. See /docs for API documentation."}


# Mount static files if frontend directory exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================================
# Context File Parsing
# ============================================================================

@app.post("/parse-files")
async def parse_context_files(
    files: list[UploadFile] = File(...),
    user_session_id: Optional[str] = Form(None),
    is_ephemeral: bool = Form(True),
):
    # NOTE: api_key parameter has been removed as we now use Docling (open source) instead of LlamaParse
    """
    Upload and process context files with extraction, chunking, and embedding.

    Ephemeral uploads (is_ephemeral=True):
    - Tagged with session_id for waterfall retrieval
    - Auto-cleanup after 24 hours via pg_cron
    - Used for uploaded context files specific to a form-filling session

    Persistent uploads (is_ephemeral=False):
    - Added to global knowledge base (session_id=NULL)
    - Never auto-deleted
    - Used for permanent reference documents

    Streams progress updates via SSE.

    Args:
        files: Up to 10 files to upload
        user_session_id: Session ID for ephemeral tagging (required if is_ephemeral=True)
        is_ephemeral: True = ephemeral (auto-cleanup), False = global KB

    Returns:
        SSE stream with progress updates and final results
    """
    # Validate file count
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file is required")

    # Check if Docling is available for files that need it
    files_needing_parse = [f for f in files if needs_parsing(f.filename or "")]
    if files_needing_parse and not DOCLING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Docling not available: {DOCLING_ERROR}. Cannot parse: {[f.filename for f in files_needing_parse]}"
        )

    # Read all file bytes
    file_data = []
    for f in files:
        content = await f.read()
        file_data.append((content, f.filename or "unknown"))

    # Pre-validation: estimate token usage before expensive Docling calls
    # Use 80% threshold to be conservative (Docling often extracts more than raw extraction)
    PRE_VALIDATION_THRESHOLD = int(MAX_CONTEXT_CHARS * 0.8)

    estimated_chars = 0
    file_estimates = []
    for file_bytes, filename in file_data:
        est = estimate_file_chars(file_bytes, filename)
        estimated_chars += est
        file_estimates.append((filename, est))

    estimated_tokens = estimated_chars // CHARS_PER_TOKEN

    if estimated_chars > PRE_VALIDATION_THRESHOLD:
        # Sort by size to show largest files first
        file_estimates.sort(key=lambda x: x[1], reverse=True)
        largest_files = ", ".join(f"{name} (~{chars//CHARS_PER_TOKEN:,} tokens)"
                                  for name, chars in file_estimates[:3])

        async def rejection_stream():
            yield f"data: {json.dumps({'type': 'error', 'error': f'Files likely exceed token limit before parsing. Estimated ~{estimated_tokens:,} tokens, limit is {MAX_CONTEXT_TOKENS:,} tokens. Largest files: {largest_files}. Please upload fewer or smaller files.'})}\n\n"

        return StreamingResponse(
            rejection_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    print(f"[Parse] Pre-validation passed: ~{estimated_tokens:,} estimated tokens for {len(file_data)} files")

    async def event_stream():
        from datetime import datetime
        
        yield f"data: {json.dumps({'type': 'init', 'message': f'Processing {len(file_data)} file(s)...'})}\n\n"

        try:
            results = []
            test_user_id = "00000000-0000-0000-0000-000000000001"
            session_id_to_use = user_session_id if is_ephemeral else None

            for idx, (file_bytes, filename) in enumerate(file_data):
                yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': len(file_data), 'filename': filename, 'status': 'processing'})}\n\n"
                
                try:
                    doc_id = await store_document(
                        user_id=test_user_id,
                        filename=filename,
                        file_bytes=file_bytes,
                        session_id=session_id_to_use,
                        metadata={"source": "parse_files_upload"}
                    )
                    
                    results.append({
                        "filename": filename,
                        "document_id": doc_id,
                        "success": True
                    })
                    
                    yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': len(file_data), 'filename': filename, 'status': 'complete', 'document_id': doc_id})}\n\n"
                    
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "success": False,
                        "error": str(e)
                    })
                    
                    error_message = f"Error parsing {filename}: {str(e)}"
                    yield f"data: {json.dumps({'type': 'error', 'filename': filename, 'error': str(e)})}\n\n"
                    
                    if user_session_id:
                        try:
                            session = _session_manager.get_session(user_session_id, test_user_id)
                            if session:
                                if not hasattr(session, 'messages') or session.messages is None:
                                    session.messages = []
                                session.messages.append({
                                    'role': 'system',
                                    'content': error_message,
                                    'timestamp': datetime.utcnow().isoformat()
                                })
                                _session_manager.save_session(session, test_user_id)
                        except Exception as save_error:
                            print(f"[ParseFiles] Error saving error message: {save_error}")

            storage_type = "ephemeral (auto-cleanup: 24h)" if is_ephemeral else "global knowledge base"
            success_count = sum(1 for r in results if r.get("success"))
            error_count = len(results) - success_count
            
            if success_count > 0 and error_count == 0:
                system_message_content = f"Successfully parsed {success_count} context file{'s' if success_count > 1 else ''}"
            elif success_count > 0 and error_count > 0:
                system_message_content = f"Parsed {success_count} of {len(results)} files ({error_count} failed)"
            else:
                system_message_content = f"Failed to parse all {len(results)} files"
            
            system_message = {
                'type': 'system_message',
                'role': 'system',
                'content': system_message_content,
                'timestamp': datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(system_message)}\n\n"
            
            if user_session_id:
                try:
                    session = _session_manager.get_session(user_session_id, test_user_id)
                    if session:
                        if not hasattr(session, 'messages') or session.messages is None:
                            session.messages = []
                        session.messages.append({
                            'role': 'system',
                            'content': system_message_content,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        _session_manager.save_session(session, test_user_id)
                        print(f"[ParseFiles] Saved system message to session {user_session_id}")
                except Exception as save_error:
                    print(f"[ParseFiles] Error saving system message: {save_error}")
            
            yield f"data: {json.dumps({'type': 'complete', 'results': results, 'count': len(results), 'success_count': success_count, 'storage_type': storage_type})}\n\n"
            
            print(f"[ParseFiles] Stored {success_count}/{len(results)} documents in {storage_type}")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            
            if user_session_id:
                try:
                    session = _session_manager.get_session(user_session_id, test_user_id)
                    if session:
                        if not hasattr(session, 'messages') or session.messages is None:
                            session.messages = []
                        session.messages.append({
                            'role': 'system',
                            'content': error_msg,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        _session_manager.save_session(session, test_user_id)
                except Exception as save_error:
                    print(f"[ParseFiles] Error saving error message: {save_error}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# Session Restoration Endpoints
# ============================================================================

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """
    Get session metadata for restoration after page reload.
    
    Returns session state including agent_session_id, applied_edits,
    fields, messages, pdf_filename, context_files, and PDF availability.
    """
    # Use default test user for development (TODO: get from auth)
    user_id = "00000000-0000-0000-0000-000000000001"
    
    session = _session_manager.get_session(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get full session data from database
    from database.supabase_client import get_client_for_user
    client = get_client_for_user(user_id)
    
    result = client.table("form_states")\
        .select("agent_session_id, pdf_filename, fields, messages")\
        .eq("session_id", session_id)\
        .eq("user_id", user_id)\
        .execute()
    
    session_data = result.data[0] if result.data else {}
    
    # Query context files from documents table (using session_id)
    docs_result = client.table("documents")\
        .select("id, filename")\
        .eq("session_id", session_id)\
        .eq("user_id", user_id)\
        .execute()
    
    context_files = [
        {"document_id": doc["id"], "filename": doc["filename"]}
        for doc in docs_result.data
    ] if docs_result.data else []
    
    return {
        "session_id": session.session_id,
        "user_session_id": session.session_id,
        "agent_session_id": session_data.get('agent_session_id'),
        "has_filled_pdf": session.current_pdf_bytes is not None,
        "has_original_pdf": session.original_pdf_bytes is not None,
        "applied_edits": session.applied_edits or {},
        "fields": session_data.get('fields', []),
        "messages": session_data.get('messages', []),
        "pdf_filename": session_data.get('pdf_filename'),
        "context_files": context_files,
    }


@app.get("/session/{session_id}/pdf")
async def get_session_pdf(session_id: str):
    """
    Retrieve the filled PDF for a session.
    
    Returns PDF bytes for displaying the current state of the form.
    """
    # Use default test user for development (TODO: get from auth)
    user_id = "00000000-0000-0000-0000-000000000001"
    
    pdf_bytes = _session_manager.get_session_pdf_bytes(session_id, user_id)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="Session not found or no PDF available")
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename=session_{session_id}.pdf"
        }
    )


@app.get("/session/{session_id}/original-pdf")
async def get_session_original_pdf(session_id: str):
    """
    Retrieve the original (unfilled) PDF for a session.
    
    Allows toggling between original and filled views.
    """
    # Use default test user for development (TODO: get from auth)
    user_id = "00000000-0000-0000-0000-000000000001"
    
    pdf_bytes = _session_manager.get_session_original_pdf_bytes(session_id, user_id)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="Session not found or no original PDF available")
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename=session_{session_id}_original.pdf"
        }
    )


@app.get("/parse-status")
async def get_parse_status():
    """Check if Docling is available."""
    return {
        "docling_available": DOCLING_AVAILABLE,
        "docling_error": DOCLING_ERROR if not DOCLING_AVAILABLE else None,
    }


@app.post("/validate-anthropic-key")
async def validate_anthropic_key(api_key: str = Form(...)):
    """
    Validate an Anthropic API key by making a test request.

    This endpoint is used to gate access to the application.
    Users must provide a valid Anthropic API key before using the app.
    """
    if not api_key or not api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required")

    api_key = api_key.strip()

    # Validate key format (Anthropic keys start with "sk-ant-")
    if not api_key.startswith("sk-ant-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid API key format. Anthropic API keys start with 'sk-ant-'"
        )

    # Test the key by making a request to Anthropic API
    try:
        import httpx

        # Use the Anthropic API models endpoint to validate the key
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=10.0,
            )

            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key. Please check your Anthropic API key."
                )
            elif response.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="API key does not have permission. Please check your Anthropic account."
                )
            elif response.status_code >= 400:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to validate API key: {response.text}"
                )

            return {"valid": True, "message": "API key is valid"}

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Timeout while validating API key. Please try again."
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Anthropic: {str(e)}"
        )

# ============================================================================
# Run directly for development
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("PDF Form Filler Server")
    print("="*60)
    print("\nRecommended Endpoints:")
    print("  POST /analyze            - Detect form fields in a PDF")
    print("  POST /fill-agent-stream  - Fill form (agent mode, SSE streaming)")
    print("  POST /fill-agent         - Fill form (agent mode)")
    print("\nLegacy Endpoints (deprecated):")
    print("  POST /fill               - Fill (single-shot LLM mode)")
    print("  POST /fill-preview       - Preview single-shot mode")
    print("\nOther:")
    print("  GET  /docs               - API documentation (Swagger UI)")
    print("\nWeb UI: http://localhost:8000")
    print("Next.js UI: http://localhost:3000 (run 'npm run dev' in web/)")
    print("\nTip: For auto-reload during development, run:")
    print("  uvicorn main:app --reload")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
