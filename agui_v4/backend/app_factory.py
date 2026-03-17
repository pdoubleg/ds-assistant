"""FastAPI application composition for the AGUI backend."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from api.routers.ag_ui import router as ag_ui_router
from api.routers.doc_lens import router as doc_lens_router
from api.routers.documents import router as documents_router
from api.routers.forms import router as forms_router
from api.routers.state import router as state_router
from api.routers.uploads import router as uploads_router
from dependencies import (
    get_backend_port,
    get_doc_lens_asset_mount_dir,
    get_runtime_documents_mount_dir,
    get_runtime_storage_service,
)
from services.doc_lens_factory import close_doc_lens_service

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle handler."""
    backend_port = get_backend_port()
    runtime_storage = get_runtime_storage_service()
    runtime_storage.ensure_dirs()

    print(f"[*] Audit Assistant Agent (AG-UI) starting on port {backend_port}")
    print(f"[*] AG-UI endpoint: POST http://localhost:{backend_port}/")
    print(f"[*] Upload endpoint: POST http://localhost:{backend_port}/upload")
    print(f"[*] Form state sync endpoint: GET/PUT http://localhost:{backend_port}/state/audit-form")
    print(
        f"[*] Claim session init endpoint: POST http://localhost:{backend_port}/state/claim-session/init"
    )
    print(f"[*] Runtime state endpoint: GET http://localhost:{backend_port}/state/runtime")
    print(f"[*] Form persistence endpoints: POST/GET http://localhost:{backend_port}/forms")
    print(
        f"[*] Form restore endpoint: POST http://localhost:{backend_port}/forms/{{form_id}}/restore"
    )
    print(f"[*] Summarize endpoint: POST http://localhost:{backend_port}/summarize (SSE)")
    print(f"[*] Info endpoint: GET http://localhost:{backend_port}/info")
    print(f"[*] Health endpoint: GET http://localhost:{backend_port}/health")

    if not os.getenv("OPENAI_API_KEY"):
        print("[!] WARNING: OPENAI_API_KEY not set")
    else:
        print("[+] OpenAI API key configured")

    try:
        yield
    finally:
        close_doc_lens_service()
        runtime_storage.clear_tmp_root()


OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "AG-UI",
        "description": "Core AG-UI protocol endpoint and service metadata.",
    },
    {
        "name": "Uploads",
        "description": "Document upload with automatic text extraction.",
    },
    {
        "name": "State",
        "description": "Shared in-memory audit state — form payload, runtime flags, and claim sessions.",
    },
    {
        "name": "Forms",
        "description": "CRUD operations for persisted audit forms in local JSON storage.",
    },
    {
        "name": "Documents",
        "description": "Document summarization, AI-powered search/sort, and auto-tagging workflows.",
    },
    {
        "name": "Doc Lens",
        "description": (
            "Visual document intelligence — ingest PDFs/images, extract assets, "
            "and run natural-language queries against document content."
        ),
    },
]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Fully wired application instance ready for ``uvicorn``.
    """
    app = FastAPI(
        title="Audit Assistant Agent",
        summary="AG-UI backend for document analysis and audit questionnaire generation.",
        description=(
            "Provides an AG-UI–compatible agent endpoint backed by **pydantic-ai**, "
            "along with supporting REST routes for document uploads, AI-powered "
            "summarization, search/sort, tagging, visual Doc Lens queries, "
            "and persisted audit-form management."
        ),
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ag_ui_router, tags=["AG-UI"])
    app.include_router(state_router, tags=["State"])
    app.include_router(forms_router, tags=["Forms"])
    app.include_router(documents_router, tags=["Documents"])
    app.include_router(uploads_router, tags=["Uploads"])
    app.include_router(doc_lens_router, tags=["Doc Lens"])

    app.mount(
        "/document-files",
        StaticFiles(directory=get_runtime_documents_mount_dir()),
        name="document-files",
    )
    app.mount(
        "/doc-lens-assets",
        StaticFiles(directory=get_doc_lens_asset_mount_dir()),
        name="doc-lens-assets",
    )
    return app
