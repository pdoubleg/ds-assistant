from typing import Literal

from pydantic import BaseModel, Field


AssetType = Literal["page", "photo"]
SearchMode = Literal["image", "text"]
ExtractionMethod = Literal[
    "full_page_render",
    "pdf_embedded_image",
    "page_segmentation",
    "standalone_image",
    "text_page",
]


class BBoxNorm(BaseModel):
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)


class IngestResponse(BaseModel):
    session_id: str
    document_id: str
    document_hash: str
    page_count: int
    num_new_assets: int
    num_reused_assets: int
    num_new_embeddings: int
    num_reused_embeddings: int


class QueryRequest(BaseModel):
    session_id: str
    query: str
    search_mode: SearchMode = "image"
    top_k: int = Field(default=10, ge=1, le=100)
    asset_types: list[AssetType] | None = None
    document_ids: list[str] | None = None


class QueryHit(BaseModel):
    rank: int
    score: float
    session_id: str
    document_id: str
    document_name: str
    page_number: int
    asset_hash: str
    asset_type: AssetType
    extraction_method: ExtractionMethod
    image_path: str
    bbox_norm: BBoxNorm | None = None
    page_text: str | None = None
    text_snippet: str | None = None


class QueryResponse(BaseModel):
    session_id: str
    query: str
    search_mode: SearchMode
    model_key: str
    top_k: int
    hits: list[QueryHit]


class SessionSummary(BaseModel):
    session_id: str
    document_count: int
    asset_count: int
    embedding_count: int
