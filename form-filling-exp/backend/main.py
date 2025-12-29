"""
FastAPI server for PDF form filling.

This is the main entry point. Run with:
    uvicorn main:app --reload

Endpoints:
    POST /analyze      - Upload PDF, get detected form fields
    POST /fill         - Fill form fields (single-shot LLM mode)
    POST /fill-agent   - Fill form fields (agent mode with tools)
    GET  /             - Serve the web UI
"""

import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_processor import detect_form_fields, edit_pdf_with_instructions, get_form_summary
from llm import map_instructions_to_fields
from agent import run_agent, run_agent_stream, AGENT_SDK_AVAILABLE, AGENT_SDK_ERROR


# ============================================================================
# App Setup
# ============================================================================

app = FastAPI(
    title="PDF Form Filler",
    description="Fill PDF forms using natural language instructions",
    version="0.1.0"
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Models
# ============================================================================

class FieldInfo(BaseModel):
    field_id: str
    field_type: str
    page: int
    label_context: str
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

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze a PDF to detect fillable form fields.
    
    Returns information about each detected field including:
    - field_id: Unique identifier for the field
    - field_type: text, checkbox, dropdown, or radio
    - label_context: Nearby text that describes the field
    - current_value: Any existing value in the field
    - options: Available options for dropdown/radio fields
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    pdf_bytes = await file.read()
    
    try:
        fields = detect_form_fields(pdf_bytes)
    except Exception as e:
        raise HTTPException(500, f"Failed to analyze PDF: {str(e)}")
    
    if not fields:
        return AnalyzeResponse(
            success=True,
            message="No fillable form fields found in this PDF. This endpoint only works with PDFs that have native AcroForm fields.",
            fields=[],
            field_count=0
        )
    
    field_infos = [
        FieldInfo(
            field_id=f.field_id,
            field_type=f.field_type.value,
            page=f.page,
            label_context=f.label_context,
            current_value=f.current_value,
            options=f.options
        )
        for f in fields
    ]
    
    return AnalyzeResponse(
        success=True,
        message=f"Found {len(fields)} fillable form fields",
        fields=field_infos,
        field_count=len(fields)
    )


@app.post("/fill")
async def fill_pdf(
    file: UploadFile = File(...),
    instructions: str = Form(...),
    use_llm: bool = Form(True)
):
    """
    Fill a PDF form using natural language instructions.
    
    Args:
        file: The PDF file to fill
        instructions: Natural language description of what to fill
            e.g., "My name is John Doe, I live at 123 Main St, 
                   my phone is 555-1234, and I agree to the terms"
        use_llm: Whether to use LLM for mapping (default True)
                 Set to False to use simple keyword matching
    
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
    
    # Step 2: Map instructions to fields
    try:
        if use_llm:
            edits = map_instructions_to_fields(instructions, fields)
        else:
            from llm import simple_keyword_mapping
            edits = simple_keyword_mapping(instructions, fields)
    except ValueError as e:
        # LLM API key not set
        raise HTTPException(
            500,
            f"LLM error: {str(e)}. Set use_llm=false to use simple keyword matching."
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


@app.post("/fill-preview")
async def fill_pdf_preview(
    file: UploadFile = File(...),
    instructions: str = Form(...),
    use_llm: bool = Form(True)
):
    """
    Preview what fields would be filled without actually filling them.
    
    Useful for debugging and understanding how instructions are mapped.
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
    
    # Map instructions
    try:
        if use_llm:
            edits = map_instructions_to_fields(instructions, fields)
        else:
            from llm import simple_keyword_mapping
            edits = simple_keyword_mapping(instructions, fields)
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
    max_iterations: int = Form(20),
):
    """
    Fill a PDF form using agent mode with real-time streaming.
    
    Returns Server-Sent Events (SSE) stream with agent messages.
    
    Event types:
    - init: Session initialized with field count
    - iteration: New iteration started
    - text: Agent thinking/response text
    - tool_start: Tool call started
    - tool_end: Tool call completed with result
    - complete: Agent finished
    - summary: Final summary with filled PDF (hex-encoded)
    - error: Error occurred
    """
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
    
    async def event_stream():
        import tempfile
        import os as os_module
        
        tmp_path = None
        output_path = None
        
        # Send immediate acknowledgment
        yield f"data: {json.dumps({'type': 'init', 'message': 'Stream connected, initializing agent...'})}\n\n"
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            
            output_path = tmp_path.replace('.pdf', '_filled.pdf')
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'PDF saved, starting Claude Agent SDK...'})}\n\n"
            
            # Stream messages from Claude Agent SDK
            message_count = 0
            async for message in run_agent_stream(tmp_path, instructions, output_path):
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
# Run directly for development
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("PDF Form Filler Server")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /analyze            - Detect form fields in a PDF")
    print("  POST /fill               - Fill (single-shot LLM mode)")
    print("  POST /fill-agent         - Fill (agent mode with tools)")
    print("  POST /fill-agent-stream  - Fill (agent mode, SSE streaming)")
    print("  POST /fill-preview       - Preview single-shot mode")
    print("  POST /fill-agent-preview - Preview agent mode with log")
    print("  GET  /docs               - API documentation (Swagger UI)")
    print("\nWeb UI: http://localhost:8000")
    print("\nTip: For auto-reload during development, run:")
    print("  uvicorn main:app --reload")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

