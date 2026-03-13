"""Request schemas for document-oriented endpoints."""

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.search import DocSearchScore
from models.tagging import ALL_DOC_TAGS, CUSTOM_FALLBACK_TAG_LABEL, TagSelectionMode


class SummarizeDocPayload(BaseModel):
    """A single document payload sent by the frontend for summarization."""

    file_name: str
    content: str
    mime_type: str = "unknown"
    document_type: str = ""


class SummarizeRequest(BaseModel):
    """Request body for `POST /summarize`."""

    documents: list[SummarizeDocPayload]
    additional_instructions: str = ""


class SearchSortDocPayload(BaseModel):
    """A document payload with full metadata for the search/sort agent."""

    file_name: str
    content_id: str
    content: str = ""
    mime_type: str = "unknown"
    content_url: str = ""
    claim_number: str = ""
    domain: str = "claim"
    document_type: str = ""
    document_sub_type: str = ""
    document_description: str = ""
    create_date: str = ""
    source_system: str = ""
    company_name: str = ""


class SearchSortRequest(BaseModel):
    """Request body for `POST /search-sort`."""

    query: str
    documents: list[SearchSortDocPayload]


class SearchSortResponse(BaseModel):
    """Response body for `POST /search-sort`."""

    scores: list[DocSearchScore] = Field(
        default_factory=list,
        description="Subset of scored documents selected for display.",
    )
    content_id_to_file_name: dict[str, str] = Field(
        default_factory=dict,
        description="Lookup table for mapping returned content IDs back to file names.",
    )


class TagRequest(BaseModel):
    """Request body for `POST /document-tags`."""

    model_config = ConfigDict(extra="forbid")

    documents: list[SummarizeDocPayload]
    tag_mode: TagSelectionMode = "default"
    selected_tags: list[str] = Field(default_factory=list, max_length=20)

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
        """Return the runtime tag vocabulary for this request."""
        if self.tag_mode == "custom":
            return [*self.selected_tags, CUSTOM_FALLBACK_TAG_LABEL]
        return list(ALL_DOC_TAGS)
