"""Request schemas for Doc Lens endpoints."""

from pydantic import BaseModel


class DocLensFilePayload(BaseModel):
    """A single file descriptor sent by the frontend for Doc Lens ingestion."""

    file_name: str
    mime_type: str


class DocLensSessionRequest(BaseModel):
    """Request body for `POST /doc-lens/session`."""

    files: list[DocLensFilePayload]


class DocLensQueryRequest(BaseModel):
    """Request body for `POST /doc-lens/query`."""

    session_id: str
    query: str
    top_k: int = 10
    asset_types: list[str] | None = None
    document_ids: list[str] | None = None


class DocLensDocumentAssetsRequest(BaseModel):
    """Request body for `POST /doc-lens/document-assets`."""

    session_id: str
    document_id: str
