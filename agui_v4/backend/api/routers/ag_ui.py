"""AG-UI, info, and health routers."""

import os

from fastapi import APIRouter
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.requests import Request
from starlette.responses import JSONResponse

from agent import agent
from dependencies import get_shared_state_deps

router = APIRouter()


@router.post("/")
async def ag_ui_endpoint(request: Request) -> JSONResponse:
    """Dispatch AG-UI requests through the shared pydantic-ai adapter."""
    try:
        body = await request.body()
        print(f"[AG-UI] Received request: {len(body)} bytes", flush=True)
        return await AGUIAdapter.dispatch_request(
            request=request,
            agent=agent,
            deps=get_shared_state_deps(),
        )
    except Exception as exc:
        print(f"[AG-UI ERROR] {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/")
async def root_get() -> JSONResponse:
    """Return AG-UI endpoint information for GET requests."""
    return JSONResponse(
        {
            "protocol": "ag-ui",
            "version": "1.0.0",
            "endpoints": {
                "run_agent": "POST /",
                "upload": "POST /upload",
                "summarize": "POST /summarize (SSE)",
                "audit_form_state": "GET|PUT /state/audit-form",
                "save_form": "POST /forms",
                "list_forms": "GET /forms",
                "get_form": "GET /forms/{form_id}",
                "restore_form": "POST /forms/{form_id}/restore",
                "info": "GET /info",
                "health": "GET /health",
            },
            "description": "Audit Assistant Agent - POST to / to run the agent",
        }
    )


@router.get("/info")
async def info_endpoint() -> JSONResponse:
    """Return backend metadata."""
    return JSONResponse(
        {
            "name": "Audit Assistant Agent",
            "version": "1.0.0",
            "protocol": "ag-ui",
            "description": "Analyze documents and generate custom audit questionnaires",
        }
    )


@router.get("/health")
async def health_endpoint() -> JSONResponse:
    """Return a basic health status payload."""
    return JSONResponse(
        {
            "status": "healthy",
            "agent_ready": bool(os.getenv("OPENAI_API_KEY")),
        }
    )
