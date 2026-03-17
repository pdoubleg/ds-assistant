"""AG-UI, info, and health routers."""

import os

from fastapi import APIRouter
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.requests import Request
from starlette.responses import JSONResponse

from agent import agent
from dependencies import get_shared_state_deps

router = APIRouter()


@router.post(
    "/",
    summary="Run the AG-UI agent",
    response_description="Streamed AG-UI protocol response.",
    responses={
        500: {"description": "Agent execution failed."},
    },
)
async def ag_ui_endpoint(request: Request) -> JSONResponse:
    """Dispatch an incoming AG-UI protocol request through the shared **pydantic-ai** adapter.

    The raw request body is forwarded to `AGUIAdapter.dispatch_request` which
    handles tool execution, streaming, and state management on behalf of the
    connected frontend.
    """
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


@router.get(
    "/",
    summary="AG-UI service directory",
    response_description="JSON map of available AG-UI endpoints.",
)
async def root_get() -> JSONResponse:
    """Return a machine-readable directory of every endpoint exposed by this service.

    Useful for AG-UI clients that need to discover available capabilities at
    runtime.
    """
    return JSONResponse(
        {
            "protocol": "ag-ui",
            "version": "1.0.0",
            "endpoints": {
                "run_agent": "POST /",
                "upload": "POST /upload",
                "summarize": "POST /summarize (SSE)",
                "audit_form_state": "GET|PUT /state/audit-form",
                "runtime_state": "GET /state/runtime",
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


@router.get(
    "/info",
    summary="Service metadata",
    response_description="Backend name, version, and protocol identifier.",
)
async def info_endpoint() -> JSONResponse:
    """Return static backend metadata including the service name, version, and
    AG-UI protocol identifier.
    """
    return JSONResponse(
        {
            "name": "Audit Assistant Agent",
            "version": "1.0.0",
            "protocol": "ag-ui",
            "description": "Analyze documents and generate custom audit questionnaires",
        }
    )


@router.get(
    "/health",
    summary="Health check",
    response_description="Current health status and readiness flags.",
)
async def health_endpoint() -> JSONResponse:
    """Return a lightweight health-check payload.

    The `agent_ready` flag indicates whether the `OPENAI_API_KEY` environment
    variable is set, which is a prerequisite for agent execution.
    """
    return JSONResponse(
        {
            "status": "healthy",
            "agent_ready": bool(os.getenv("OPENAI_API_KEY")),
        }
    )
