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
    get_upload_dir,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle handler."""
    backend_port = get_backend_port()
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

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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

    app.include_router(ag_ui_router)
    app.include_router(state_router)
    app.include_router(forms_router)
    app.include_router(documents_router)
    app.include_router(uploads_router)
    app.include_router(doc_lens_router)

    app.mount("/uploads", StaticFiles(directory=get_upload_dir()), name="uploads")
    app.mount(
        "/doc-lens-assets",
        StaticFiles(directory=get_doc_lens_asset_mount_dir()),
        name="doc-lens-assets",
    )
    return app
