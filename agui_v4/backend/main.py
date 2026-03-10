"""Thin FastAPI composition entrypoint for the AGUI backend.

Local development:
    uv run uvicorn main:app --reload --port 8001
"""

from app_factory import create_app
from dependencies import get_backend_port

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=get_backend_port(), reload=True)
