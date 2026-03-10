"""Document summary and search/sort result models."""

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """Summarization output for a single document."""

    title: str = Field(..., description="Short title capturing the document's essence.")
    summary: str = Field(
        ...,
        description="Concise, well-structured document-type-agnostic summary of the document contents.",
    )
    label: str = Field(
        ...,
        description="Short (2 to 4 word) flavor-text label describing the document's character or purpose.",
    )


class DocSearchScore(BaseModel):
    """Search/sort score for a single document."""

    content_id: str = Field(..., description="Content ID of the scored document.")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score (0.0 = irrelevant, 1.0 = most relevant)."
    )
    label: str = Field(
        ...,
        description="Short (2 to 4 word) flavor-text label explaining the score.",
    )


class DocSearchResult(BaseModel):
    """Batch result from the search/sort agent."""

    scores: list[DocSearchScore] = Field(..., description="Per-document search/sort scores.")
