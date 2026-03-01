"""
Audit Assistant Agent - AG-UI endpoint using Pydantic AI.

This creates an AG-UI compatible endpoint that CopilotKit can connect to
for bidirectional state synchronization and streaming agent responses.
Also provides a file upload endpoint for document processing.

uv run uvicorn main:app --reload --port 8001
"""

import os
import re
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Body  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, StreamingResponse  # noqa: E402

from agent import agent, AuditState  # noqa: E402
from pydantic_ai.ag_ui import StateDeps  # noqa: E402
from a2ui_generator import generate_audit_question_form  # noqa: E402

# Configuration
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
FORMS_DIR = os.path.join(os.path.dirname(__file__), "data", "forms")
os.makedirs(FORMS_DIR, exist_ok=True)


def pascal_to_screaming_snake(name: str) -> str:
    """Convert PascalCase to SCREAMING_SNAKE_CASE.

    Args:
        name: PascalCase event type name.

    Returns:
        SCREAMING_SNAKE_CASE version of the name.
    """
    result = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    return result.upper()


# Map of PascalCase to SCREAMING_SNAKE_CASE event types
EVENT_TYPE_MAP = {
    "RunStarted": "RUN_STARTED",
    "RunFinished": "RUN_FINISHED",
    "RunError": "RUN_ERROR",
    "StepStarted": "STEP_STARTED",
    "StepFinished": "STEP_FINISHED",
    "StateSnapshot": "STATE_SNAPSHOT",
    "StateDelta": "STATE_DELTA",
    "MessagesSnapshot": "MESSAGES_SNAPSHOT",
    "TextMessageStart": "TEXT_MESSAGE_START",
    "TextMessageContent": "TEXT_MESSAGE_CONTENT",
    "TextMessageEnd": "TEXT_MESSAGE_END",
    "TextMessageChunk": "TEXT_MESSAGE_CHUNK",
    "ToolCallStart": "TOOL_CALL_START",
    "ToolCallArgs": "TOOL_CALL_ARGS",
    "ToolCallEnd": "TOOL_CALL_END",
    "ToolCallChunk": "TOOL_CALL_CHUNK",
    "ToolCallResult": "TOOL_CALL_RESULT",
    "Raw": "RAW",
    "Custom": "CUSTOM",
}


def transform_event_type(event_data: str) -> str:
    """Transform event type in SSE data from PascalCase to SCREAMING_SNAKE_CASE.

    Args:
        event_data: JSON string containing an event with a 'type' field.

    Returns:
        JSON string with the 'type' field converted to SCREAMING_SNAKE_CASE.
    """
    try:
        data = json.loads(event_data)
        if "type" in data:
            original_type = data["type"]
            if original_type in EVENT_TYPE_MAP:
                data["type"] = EVENT_TYPE_MAP[original_type]
            elif not original_type.isupper():
                data["type"] = pascal_to_screaming_snake(original_type)
        return json.dumps(data)
    except json.JSONDecodeError:
        return event_data


async def transform_sse_stream(original_response):
    """Transform SSE stream to use SCREAMING_SNAKE_CASE event types.

    Wraps the original Pydantic AI SSE response and converts event names
    and data JSON 'type' fields for CopilotKit v2 compatibility.

    Args:
        original_response: The original StreamingResponse from pydantic-ai.

    Yields:
        Transformed SSE chunks with SCREAMING_SNAKE_CASE event types.
    """
    try:
        async for chunk in original_response.body_iterator:
            try:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode('utf-8')

                lines = chunk.split('\n')
                transformed_lines = []

                for line in lines:
                    if line.startswith('event: '):
                        event_name = line[7:]
                        if event_name in EVENT_TYPE_MAP:
                            transformed_lines.append(f'event: {EVENT_TYPE_MAP[event_name]}')
                        elif not event_name.isupper():
                            transformed_lines.append(
                                f'event: {pascal_to_screaming_snake(event_name)}'
                            )
                        else:
                            transformed_lines.append(line)
                    elif line.startswith('data: '):
                        data = line[6:]
                        transformed_data = transform_event_type(data)
                        transformed_lines.append(f'data: {transformed_data}')
                    else:
                        transformed_lines.append(line)

                yield '\n'.join(transformed_lines)
            except Exception as chunk_error:
                print(f"[SSE ERROR] Error processing chunk: {chunk_error}", flush=True)
                import traceback
                traceback.print_exc()
                raise
    except Exception as stream_error:
        print(f"[SSE ERROR] Stream error: {stream_error}", flush=True)
        import traceback
        traceback.print_exc()
        raise


# Shared in-memory state for AG-UI and persistence endpoints.
# This keeps UI edits and backend submit/restore operations in sync.
APP_DEPS = StateDeps(AuditState())

# Create the base AG-UI app using Pydantic AI's built-in integration.
_base_ag_ui_app = agent.to_ag_ui(deps=APP_DEPS)


# =========================================================================
# Request body models
# =========================================================================

class AuditFormRequestBody(BaseModel):
    """Flexible audit form request body for state sync and persistence endpoints.

    Accepts the form payload in three formats:
    1. Wrapped in an ``audit_form_result`` envelope.
    2. Wrapped in a ``form`` envelope.
    3. Flat/direct payload fields passed at the top level (captured via ``extra='allow'``).

    Example usage::

        # Wrapped format
        body = AuditFormRequestBody(audit_form_result={"peril": {...}, "questions": [...]})

        # Direct format (extra fields)
        body = AuditFormRequestBody(**{"peril": {...}, "questions": [...], "overall_outcome": "Pass"})

        payload = body.extract_form_payload()
    """

    model_config = ConfigDict(extra="allow")

    audit_form_result: dict[str, Any] | None = None
    form: dict[str, Any] | None = None
    current_form_id: str | None = None
    id: str | None = None
    title: str | None = None
    source_docs: list[Any] = []

    def extract_form_payload(self) -> dict[str, Any]:
        """Extract canonical audit form payload from the request body.

        Checks for an ``audit_form_result`` or ``form`` wrapper first, then
        falls back to any extra fields passed directly at the top level.

        Returns:
            Dict containing the audit form data, or an empty dict if none found.
        """
        if isinstance(self.audit_form_result, dict):
            return self.audit_form_result
        if isinstance(self.form, dict):
            return self.form
        # Direct payload passed as extra top-level fields
        return self.model_extra or {}


# =========================================================================
# Helper utilities
# =========================================================================

def _iso_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON atomically by writing to temp then replacing target file.

    Args:
        path: Destination JSON file path.
        payload: JSON-serializable dictionary payload.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _form_file_path(form_id: str) -> str:
    """Build a safe JSON file path for a form ID.

    Args:
        form_id: Form identifier.

    Returns:
        Full path to the form JSON file.
    """
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", form_id)
    return os.path.join(FORMS_DIR, f"{safe_id}.json")


def _validate_form_payload(payload: dict[str, Any]) -> str | None:
    """Validate required audit form fields for persistence and restore.

    Args:
        payload: Candidate audit form payload.

    Returns:
        None when valid, otherwise an error string.
    """
    required_fields = [
        "peril",
        "questions",
        "overall_outcome",
        "outcome_justification",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"

    if not isinstance(payload.get("questions"), list):
        return "Field 'questions' must be a list."

    if not isinstance(payload.get("peril"), dict):
        return "Field 'peril' must be an object."

    return None


def _build_form_title(form_payload: dict[str, Any]) -> str:
    """Build a human-friendly fallback title for a saved form.

    Args:
        form_payload: Canonical audit form payload.

    Returns:
        Generated title string.
    """
    peril = form_payload.get("peril", {}).get("peril", "Unknown")
    outcome = form_payload.get("overall_outcome", "Unknown")
    return f"{peril} - {outcome} - {_iso_now()[:10]}"


def _upsert_audit_form_component(state: AuditState, form_payload: dict[str, Any]) -> None:
    """Replace existing AuditQuestionForm component with latest payload.

    Args:
        state: Shared audit state.
        form_payload: Canonical audit form payload.
    """
    component = generate_audit_question_form(
        peril=form_payload["peril"],
        questions=form_payload["questions"],
        overall_outcome=form_payload["overall_outcome"],
        outcome_justification=form_payload["outcome_justification"],
        additional_analysis=form_payload.get("additional_analysis"),
        follow_ups=form_payload.get("follow_ups"),
    ).model_dump()

    # Keep only one active audit form component for predictable restore behavior.
    replaced = False
    for index, item in enumerate(state.components):
        if item.get("type") == "a2ui.AuditQuestionForm":
            state.components[index] = component
            replaced = True
            break
    if not replaced:
        state.components.append(component)


# =========================================================================
# App factory & lifespan
# =========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle handler."""
    print(f"[*] Audit Assistant Agent (AG-UI) starting on port {BACKEND_PORT}")
    print(f"[*] AG-UI endpoint: POST http://localhost:{BACKEND_PORT}/")
    print(f"[*] Upload endpoint: POST http://localhost:{BACKEND_PORT}/upload")
    print(f"[*] Form state sync endpoint: GET/PUT http://localhost:{BACKEND_PORT}/state/audit-form")
    print(f"[*] Form persistence endpoints: POST/GET http://localhost:{BACKEND_PORT}/forms")
    print(f"[*] Form restore endpoint: POST http://localhost:{BACKEND_PORT}/forms/{{form_id}}/restore")
    print(f"[*] Info endpoint: GET http://localhost:{BACKEND_PORT}/info")
    print(f"[*] Health endpoint: GET http://localhost:{BACKEND_PORT}/health")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[!] WARNING: OPENAI_API_KEY not set")
    else:
        print("[+] OpenAI API key configured")

    yield  # Application runs here; add shutdown logic after yield if needed


app = FastAPI(
    title="Audit Assistant Agent",
    description="Analyze documents and generate custom audit questionnaires",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# AG-UI endpoint
# =========================================================================

@app.post("/")
async def ag_ui_endpoint(request: Request):
    """AG-UI endpoint that wraps Pydantic AI's implementation
    and transforms event types to SCREAMING_SNAKE_CASE format.

    Uses raw ``Request`` access so the original body bytes can be
    forwarded unchanged into the pydantic-ai Starlette sub-app.
    """
    try:
        body = await request.body()
        print(f"[AG-UI] Received request: {len(body)} bytes", flush=True)

        # Reconstruct a fresh StarletteRequest with the buffered body so
        # the pydantic-ai handler can read it a second time.
        scope = dict(request.scope)

        async def receive():
            return {"type": "http.request", "body": body}

        new_request = Request(scope, receive)

        # Find the POST route handler in the pydantic-ai base app
        for route in _base_ag_ui_app.routes:
            if hasattr(route, 'methods') and 'POST' in route.methods:
                original_response = await route.endpoint(new_request)
                break
        else:
            return JSONResponse({"error": "AG-UI endpoint not found"}, status_code=500)

        # Wrap streaming responses with our event type transformer
        if isinstance(original_response, StreamingResponse):
            return StreamingResponse(
                transform_sse_stream(original_response),
                media_type="text/event-stream",
                headers=dict(original_response.headers),
            )

        return original_response
    except Exception as e:
        print(f"[AG-UI ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
async def root_get():
    """Return AG-UI endpoint info for GET requests."""
    return JSONResponse({
        "protocol": "ag-ui",
        "version": "1.0.0",
        "endpoints": {
            "run_agent": "POST /",
            "upload": "POST /upload",
            "audit_form_state": "GET|PUT /state/audit-form",
            "save_form": "POST /forms",
            "list_forms": "GET /forms",
            "get_form": "GET /forms/{form_id}",
            "restore_form": "POST /forms/{form_id}/restore",
            "info": "GET /info",
            "health": "GET /health",
        },
        "description": "Audit Assistant Agent - POST to / to run the agent",
    })


# =========================================================================
# Audit form state sync + persistence endpoints
# =========================================================================

@app.get("/state/audit-form")
async def get_audit_form_state():
    """Get the current audit form payload and form ID from shared state."""
    state = APP_DEPS.state
    return JSONResponse({
        "current_form_id": state.current_form_id,
        "audit_form_result": state.audit_form_result,
    })


@app.put("/state/audit-form")
async def put_audit_form_state(body: AuditFormRequestBody):
    """Update the editable audit form payload in shared state.

    Upserts payload into state and keeps the AuditQuestionForm component in sync.
    Accepts the payload wrapped in ``audit_form_result``/``form``, or flat at the
    top level.
    """
    state = APP_DEPS.state

    payload = body.extract_form_payload()
    validation_error = _validate_form_payload(payload)
    if validation_error:
        return JSONResponse({"error": validation_error}, status_code=400)

    state.audit_form_result = payload
    state.audit_questions = payload["questions"]
    if body.current_form_id:
        state.current_form_id = body.current_form_id
    _upsert_audit_form_component(state, payload)

    return JSONResponse({
        "message": "Audit form state synchronized.",
        "current_form_id": state.current_form_id,
        "audit_form_result": state.audit_form_result,
    })


@app.post("/forms")
async def save_form(body: AuditFormRequestBody | None = Body(default=None)):
    """Persist an audit form to local JSON storage under data/forms/.

    Request body can include either:
    - audit_form_result/form payload directly
    - optional id/title/source_docs metadata

    If no valid payload is provided in the request body, this endpoint falls
    back to the current in-memory state audit_form_result.
    """
    state = APP_DEPS.state

    payload = body.extract_form_payload() if body else {}
    if not payload:
        # Fall back to whatever is currently in memory
        payload = state.audit_form_result

    validation_error = _validate_form_payload(payload)
    if validation_error:
        return JSONResponse({"error": validation_error}, status_code=400)

    requested_id = body.id if body else None
    form_id = requested_id or state.current_form_id or str(uuid4())
    path = _form_file_path(form_id)

    # Preserve created_at when updating an existing saved form.
    existing_created_at = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                existing = json.load(file_obj)
                existing_created_at = existing.get("created_at")
        except Exception:
            existing_created_at = None

    record = {
        "id": form_id,
        "schema_version": "1.0",
        "created_at": existing_created_at or _iso_now(),
        "updated_at": _iso_now(),
        "title": (body.title if body and body.title else None) or _build_form_title(payload),
        "source_docs": body.source_docs if body else [],
        "peril": payload["peril"],
        "questions": payload["questions"],
        "overall_outcome": payload["overall_outcome"],
        "outcome_justification": payload["outcome_justification"],
        "additional_analysis": payload.get("additional_analysis"),
        "follow_ups": payload.get("follow_ups"),
    }

    _atomic_write_json(path, record)

    # Keep memory state aligned with saved payload.
    state.current_form_id = form_id
    state.audit_form_result = payload
    state.audit_questions = payload["questions"]
    _upsert_audit_form_component(state, payload)

    return JSONResponse({
        "message": "Form saved.",
        "form_id": form_id,
        "title": record["title"],
        "updated_at": record["updated_at"],
    })


@app.get("/forms")
async def list_forms():
    """List all saved forms from local JSON storage."""
    forms: list[dict[str, Any]] = []
    for name in os.listdir(FORMS_DIR):
        if not name.endswith(".json"):
            continue
        file_path = os.path.join(FORMS_DIR, name)
        try:
            with open(file_path, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
            forms.append({
                "id": data.get("id"),
                "title": data.get("title"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "peril": data.get("peril", {}).get("peril"),
                "overall_outcome": data.get("overall_outcome"),
                "question_count": len(data.get("questions", [])),
            })
        except Exception as exc:
            # Skip unreadable/corrupt files but expose useful debug signal.
            print(f"[FORMS] Failed reading {file_path}: {exc}", flush=True)

    forms.sort(key=lambda form: form.get("updated_at") or "", reverse=True)
    return JSONResponse({"forms": forms})


@app.get("/forms/{form_id}")
async def get_form(form_id: str):
    """Read one saved form record by ID."""
    path = _form_file_path(form_id)
    if not os.path.exists(path):
        return JSONResponse({"error": f"Form '{form_id}' not found."}, status_code=404)

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read form: {exc}"}, status_code=500)

    return JSONResponse(data)


@app.post("/forms/{form_id}/restore")
async def restore_form(form_id: str):
    """Load a saved form and restore it into shared agent state."""
    path = _form_file_path(form_id)
    if not os.path.exists(path):
        return JSONResponse({"error": f"Form '{form_id}' not found."}, status_code=404)

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            record = json.load(file_obj)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read form: {exc}"}, status_code=500)

    payload = {
        "peril": record.get("peril", {}),
        "questions": record.get("questions", []),
        "overall_outcome": record.get("overall_outcome", ""),
        "outcome_justification": record.get("outcome_justification", ""),
        "additional_analysis": record.get("additional_analysis"),
        "follow_ups": record.get("follow_ups"),
    }
    validation_error = _validate_form_payload(payload)
    if validation_error:
        return JSONResponse({"error": f"Saved form is invalid: {validation_error}"}, status_code=500)

    state = APP_DEPS.state
    state.current_form_id = record.get("id", form_id)
    state.audit_form_result = payload
    state.audit_questions = payload["questions"]
    state.status = "complete"
    state.current_step = f"Restored saved form {state.current_form_id}"
    _upsert_audit_form_component(state, payload)

    return JSONResponse({
        "message": "Form restored to state.",
        "form_id": state.current_form_id,
        "audit_form_result": state.audit_form_result,
    })


@app.delete("/forms/{form_id}")
async def delete_form(form_id: str):
    """Delete a saved form from local JSON storage.

    If the deleted form is the currently active form, the reference is cleared
    from shared state.

    Args:
        form_id: Form identifier from the URL path.

    Returns:
        JSON confirmation or 404 if the form does not exist.
    """
    path = _form_file_path(form_id)
    if not os.path.exists(path):
        return JSONResponse({"error": f"Form '{form_id}' not found."}, status_code=404)

    try:
        os.remove(path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to delete form: {exc}"}, status_code=500)

    # Clear the active reference if this was the current form
    state = APP_DEPS.state
    if state.current_form_id == form_id:
        state.current_form_id = None

    return JSONResponse({"message": "Form deleted.", "form_id": form_id})


# =========================================================================
# Text extraction helpers
# =========================================================================

def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """Extract text and page count from a PDF using PyMuPDF.

    Args:
        file_bytes: Raw PDF bytes.

    Returns:
        Tuple of (extracted_text, page_count).
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages), len(pages)


def extract_text_from_docx(file_bytes: bytes) -> tuple[str, int]:
    """Extract text and approximate page count from a DOCX file.

    Args:
        file_bytes: Raw DOCX bytes.

    Returns:
        Tuple of (extracted_text, estimated_page_count).
    """
    import io
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    # Rough page estimate: ~3000 chars per page
    page_estimate = max(1, len(text) // 3000)
    return text, page_estimate


def extract_text_from_xlsx(file_bytes: bytes) -> tuple[str, int]:
    """Extract text from an XLSX file by reading all sheets.

    Each row is joined with tabs, each sheet separated by a header line.

    Args:
        file_bytes: Raw XLSX bytes.

    Returns:
        Tuple of (extracted_text, sheet_count).
    """
    import io
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    parts: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows: list[str] = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                rows.append("\t".join(cells))
        if rows:
            parts.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(parts), len(wb.sheetnames)


EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".xlsx": extract_text_from_xlsx,
}


# =========================================================================
# File upload endpoint
# =========================================================================

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """Handle document file uploads with text extraction.

    Accepts a multipart ``file`` field. Supports .pdf, .docx, and .xlsx file
    types. Extracts text content so the agent can analyze the document.

    Args:
        file: The uploaded file provided via multipart form data.

    Returns:
        JSON with file metadata and extracted text content.
    """
    try:
        allowed_extensions = {".pdf", ".docx", ".xlsx"}
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in allowed_extensions:
            return JSONResponse(
                {"error": f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"},
                status_code=400,
            )

        file_bytes = await file.read()

        # Save to disk (useful for debugging / future re-processing)
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Extract text content
        extractor = EXTRACTORS.get(ext)
        if extractor:
            extracted_text, page_count = extractor(file_bytes)
        else:
            extracted_text, page_count = "", 0

        print(
            f"[UPLOAD] {filename}: {len(file_bytes)} bytes, "
            f"{page_count} pages, {len(extracted_text)} chars extracted",
            flush=True,
        )

        # Human-readable file size
        file_size = len(file_bytes)
        size_str = (
            f"{file_size / 1024 / 1024:.1f} MB"
            if file_size > 1024 * 1024
            else f"{file_size / 1024:.1f} KB"
        )

        return JSONResponse({
            "filename": filename,
            "file_type": ext.lstrip("."),
            "file_size": size_str,
            "file_size_bytes": file_size,
            "page_count": page_count,
            "content": extracted_text,
            "path": file_path,
        })
    except Exception as e:
        print(f"[UPLOAD ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================================================================
# Info / health endpoints
# =========================================================================

@app.get("/info")
async def info_endpoint():
    """Return agent information."""
    return JSONResponse({
        "name": "Audit Assistant Agent",
        "version": "1.0.0",
        "protocol": "ag-ui",
        "description": "Analyze documents and generate custom audit questionnaires",
    })


@app.get("/health")
async def health_endpoint():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "agent_ready": bool(os.getenv("OPENAI_API_KEY")),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=BACKEND_PORT, reload=True)
