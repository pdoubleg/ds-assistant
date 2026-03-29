"""Request schemas for document-oriented endpoints."""

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.search import DocSearchScore
from models.tagging import ALL_DOC_TAGS, CUSTOM_FALLBACK_TAG_LABEL, TagSelectionMode


class SummarizeDocPayload(BaseModel):
    """A single document payload sent by the frontend for summarization."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_name": "policy-document.pdf",
                "content_id": "abc123",
                "content": "INSURANCE POLICY … (truncated)",
                "mime_type": "application/pdf",
                "content_url": "/document-files/abc123.pdf",
                "document_type": "Policy",
            }
        }
    )

    file_name: str = Field(..., description="Original file name including extension.")
    content_id: str = Field("", description="Optional staging identifier for the document.")
    content: str = Field(..., description="Extracted plain-text content of the document.")
    mime_type: str = Field("unknown", description="MIME type of the source file.")
    content_url: str = Field(
        "", description="Public URL for direct browser access to the staged file."
    )
    document_type: str = Field(
        "", description="Optional document category label (e.g. `Policy`, `Invoice`)."
    )
    document_description: str = Field("", description="Optional free-text document description.")


class SummarizeRequest(BaseModel):
    """Request body for ``POST /summarize``.

    Example:
        ```json
        {
            "documents": [{"file_name": "report.pdf", "content": "…"}],
            "additional_instructions": "Focus on financial figures."
        }
        ```
    """

    documents: list[SummarizeDocPayload] = Field(
        ..., description="One or more documents to summarize."
    )
    additional_instructions: str = Field(
        "",
        description="Optional free-text instructions to steer the summarization style or focus areas.",
    )


class SearchSortDocPayload(BaseModel):
    """A document payload with full metadata for the search/sort agent."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_name": "claim-report.pdf",
                "content_id": "abc123",
                "content": "Claim investigation findings …",
                "mime_type": "application/pdf",
                "domain": "claim",
                "document_type": "Claim Report",
            }
        }
    )

    file_name: str = Field(..., description="Original file name including extension.")
    content_id: str = Field(..., description="Unique staging identifier for the document.")
    content: str = Field("", description="Extracted plain-text content.")
    mime_type: str = Field("unknown", description="MIME type of the source file.")
    content_url: str = Field(
        "", description="Public URL for direct browser access to the staged file."
    )
    claim_number: str = Field("", description="Associated claim identifier, if any.")
    domain: str = Field("claim", description="Business domain context (default `claim`).")
    document_type: str = Field("", description="Primary document category.")
    document_sub_type: str = Field("", description="Secondary document category.")
    document_description: str = Field("", description="Free-text description of the document.")
    create_date: str = Field("", description="Document creation date (ISO-8601).")
    source_system: str = Field("", description="System of record the document originated from.")
    company_name: str = Field("", description="Company associated with the document.")


class SearchSortRequest(BaseModel):
    """Request body for ``POST /search-sort``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Find documents about roof damage",
                "documents": [
                    {
                        "file_name": "claim-report.pdf",
                        "content_id": "abc123",
                        "content": "Inspection revealed significant roof damage …",
                    }
                ],
            }
        }
    )

    query: str = Field(..., description="Natural-language search query to score documents against.")
    documents: list[SearchSortDocPayload] = Field(
        ..., description="Documents to rank against the query."
    )


class SearchSortResponse(BaseModel):
    """Response body for ``POST /search-sort``."""

    scores: list[DocSearchScore] = Field(
        default_factory=list,
        description="Subset of scored documents selected for display, ordered by relevance.",
    )
    content_id_to_file_name: dict[str, str] = Field(
        default_factory=dict,
        description="Lookup table mapping returned `content_id` values back to original file names.",
    )


class TagRequest(BaseModel):
    """Request body for ``POST /document-tags``.

    Example — custom tagging:
        ```json
        {
            "documents": [{"file_name": "report.pdf", "content": "…"}],
            "tag_mode": "custom",
            "selected_tags": ["Financial", "Legal"]
        }
        ```
    """

    model_config = ConfigDict(extra="forbid")

    documents: list[SummarizeDocPayload] = Field(
        ..., description="Documents to classify into tags."
    )
    tag_mode: TagSelectionMode = Field(
        "default",
        description="Tagging strategy — `default` uses the built-in taxonomy, `custom` uses `selected_tags`.",
    )
    selected_tags: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Custom tag labels (required when `tag_mode` is `custom`, max 20).",
    )

    @model_validator(mode="after")
    def validate_selected_tags(self) -> Self:
        """Normalize user-selected tags and enforce custom-mode constraints."""
        normalized_tags: list[str] = []
        seen_labels: dict[str, str] = {}

        for raw_tag in self.selected_tags:
            label = re.sub(r"\s+", " ", raw_tag).strip()
            if not label:
                raise ValueError("Tag labels cannot be empty.")

            normalized_key = label.casefold()
            if normalized_key in seen_labels:
                raise ValueError(
                    f"Duplicate tag labels are not allowed: '{seen_labels[normalized_key]}' and '{label}'."
                )

            seen_labels[normalized_key] = label
            normalized_tags.append(label)

        if len(normalized_tags) > 20:
            raise ValueError("A maximum of 20 selected tags is allowed.")

        if self.tag_mode == "custom" and not normalized_tags:
            raise ValueError("Custom tagging requires at least one selected tag.")

        self.selected_tags = normalized_tags
        return self

    def get_active_tags(self) -> list[str]:
        """Return the runtime tag vocabulary for this request.

        Returns:
            list[str]: Either the custom tag list (with a fallback label appended)
                or the full default taxonomy.
        """
        if self.tag_mode == "custom":
            return [*self.selected_tags, CUSTOM_FALLBACK_TAG_LABEL]
        return list(ALL_DOC_TAGS)
