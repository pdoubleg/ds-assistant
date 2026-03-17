"""Request schemas for Doc Lens endpoints."""

from pydantic import BaseModel, ConfigDict, Field

from doc_lens.models import SearchMode


class DocLensFilePayload(BaseModel):
    """A single file descriptor sent by the frontend for Doc Lens ingestion."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_name": "policy-document.pdf",
                "mime_type": "application/pdf",
                "content_id": "abc123",
            }
        }
    )

    file_name: str = Field(..., description="Original file name including extension.")
    mime_type: str = Field(
        ..., description="MIME type of the file (e.g. `application/pdf`, `image/png`)."
    )
    content_id: str = Field(
        ...,
        description="Unique identifier assigned during upload staging, used to resolve the temp file path.",
    )


class DocLensSessionRequest(BaseModel):
    """Request body for ``POST /doc-lens/session``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "files": [
                    {
                        "file_name": "policy-document.pdf",
                        "mime_type": "application/pdf",
                        "content_id": "abc123",
                    }
                ]
            }
        }
    )

    files: list[DocLensFilePayload] = Field(
        ..., description="List of files to ingest into the new Doc Lens session."
    )


class DocLensQueryRequest(BaseModel):
    """Request body for ``POST /doc-lens/query``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "Find tables showing coverage limits",
                "search_mode": "image",
                "top_k": 5,
            }
        }
    )

    session_id: str = Field(..., description="Active Doc Lens session identifier.")
    query: str = Field(..., description="Natural-language search query.")
    search_mode: SearchMode = Field(
        "image", description="Search strategy — `image` for visual similarity, `text` for semantic text search."
    )
    top_k: int = Field(10, description="Maximum number of results to return.", ge=1, le=100)
    asset_types: list[str] | None = Field(
        None, description="Optional filter to restrict results to specific asset types (e.g. `table`, `figure`)."
    )
    document_ids: list[str] | None = Field(
        None, description="Optional filter to restrict results to specific ingested documents."
    )


class DocLensDocumentAssetsRequest(BaseModel):
    """Request body for ``POST /doc-lens/document-assets``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "document_id": "policy-document.pdf",
            }
        }
    )

    session_id: str = Field(..., description="Active Doc Lens session identifier.")
    document_id: str = Field(..., description="Document identifier within the session.")
