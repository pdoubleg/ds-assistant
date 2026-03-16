FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    DOC_LENS_CACHE_ROOT=/app/tmp/doc_lens_cache \
    DOC_LENS_DUCKDB_PATH=/app/tmp/doc_lens_cache/doc_lens.duckdb \
    DOC_LENS_FASTEMBED_SEED_DIR=/app/model_artifacts/fastembed_models \
    DOC_LENS_FASTEMBED_RUNTIME_DIR=/app/tmp/fastembed_models

WORKDIR /app

# Runtime libs needed by OpenCV / ONNXRuntime-based dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /uvx /bin/

# Install dependency graph first for better Docker layer reuse.
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application code and seeded model-artifact directory.
COPY . .

# Ensure writable runtime storage exists while seeded artifacts remain in the image.
RUN mkdir -p /app/tmp /app/uploads /app/model_artifacts/fastembed_models

RUN uv sync --no-dev

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
