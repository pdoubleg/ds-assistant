"""Document tagging models and runtime schema builders."""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Self, Sequence, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from pydantic_ai import ModelRetry


DocTag = Literal[
    "Insured",
    "Contractor",
    "Agent",
    "Vendor",
    "Attorney",
    "Contact/Status",
    "Estimate",
    "Supplement",
    "Demand",
    "Dwelling",
    "Contents",
    "ALE",
    "EMS",
    "Photos",
    "Damage Report",
    "Weather Report",
    "Attorney Demand",
    "Time Sensitive",
    "Compliance Issue",
    "Customer Complaint",
]

ALL_DOC_TAGS: list[str] = list(get_args(DocTag))
"""Flat list of every valid ``DocTag`` value, derived from the Literal."""

CUSTOM_FALLBACK_TAG_LABEL = "No Applicable Tags"
"""Special custom-mode tag used when no selected labels fit a document."""

TagSelectionMode = Literal["default", "custom"]
"""Frontend-selectable document tagging modes."""

TagIconName = Literal[
    "general",
    "insured",
    "contractor",
    "agent",
    "vendor",
    "attorney",
    "contact_status",
    "estimate",
    "supplement",
    "demand",
    "dwelling",
    "contents",
    "ale",
    "ems",
    "photos",
    "damage_report",
    "weather_report",
    "attorney_demand",
    "time_sensitive",
    "compliance_issue",
    "customer_complaint",
]
"""Allowed icon identifiers for document tags."""

ALL_TAG_ICON_NAMES: list[str] = list(get_args(TagIconName))
"""Flat list of every valid ``TagIconName`` value."""

DEFAULT_TAG_ICON_BY_LABEL: dict[str, TagIconName] = {
    "Insured": "insured",
    "Contractor": "contractor",
    "Agent": "agent",
    "Vendor": "vendor",
    "Attorney": "attorney",
    "Contact/Status": "contact_status",
    "Estimate": "estimate",
    "Supplement": "supplement",
    "Demand": "demand",
    "Dwelling": "dwelling",
    "Contents": "contents",
    "ALE": "ale",
    "EMS": "ems",
    "Photos": "photos",
    "Damage Report": "damage_report",
    "Weather Report": "weather_report",
    "Attorney Demand": "attorney_demand",
    "Time Sensitive": "time_sensitive",
    "Compliance Issue": "compliance_issue",
    "Customer Complaint": "customer_complaint",
}
"""Default icon mapping for the canonical document tag set."""


class DefaultDocumentTagAssignment(BaseModel):
    """Single tag/icon pair emitted by the default batch tagger."""

    label: DocTag = Field(..., description="Canonical tag label for the document.")
    icon: TagIconName = Field(..., description="Icon identifier paired with the tag.")


class DocumentTagResult(BaseModel):
    """Tags assigned to a single document by the batch tagger agent."""

    file_name: str = Field(..., description="Original file name of the document.")
    tags: list[DefaultDocumentTagAssignment] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="1-4 tag/icon assignments from the predefined DocTag vocabulary.",
    )


class BatchTagResult(BaseModel):
    """Output of the batch tagger agent for a single batch of documents."""

    results: list[DocumentTagResult] = Field(
        ..., description="Tag results for each document in the batch."
    )


@dataclass(frozen=True)
class RuntimeBatchTagSchema:
    """Dynamically generated schema bundle for one tagging run."""

    tag_literal: Any
    tag_assignment_model: type[BaseModel]
    document_result_model: type[BaseModel]
    batch_result_model: type[BaseModel]


def _build_literal(values: tuple[str, ...]) -> Any:
    """Create a runtime ``Literal`` type from string values."""
    return Literal.__getitem__(values)


@lru_cache(maxsize=32)
def _build_runtime_batch_tag_schema_cached(
    active_labels: tuple[str, ...],
) -> RuntimeBatchTagSchema:
    """Create cached runtime Pydantic models for a tag selection."""
    if not active_labels:
        raise ValueError("At least one active label is required to build a tag schema.")

    schema_suffix = hashlib.sha1("|".join(active_labels).encode("utf-8")).hexdigest()[:12]
    tag_literal = _build_literal(active_labels)
    icon_literal = _build_literal(tuple(ALL_TAG_ICON_NAMES))
    model_config = ConfigDict(extra="forbid")
    fallback_label = (
        CUSTOM_FALLBACK_TAG_LABEL if CUSTOM_FALLBACK_TAG_LABEL in active_labels else None
    )

    tag_assignment_model = create_model(
        f"DynamicDocumentTagAssignment_{schema_suffix}",
        __config__=model_config,
        label=(
            tag_literal,
            Field(..., description="Selected tag label for the document."),
        ),
        icon=(
            icon_literal,
            Field(..., description="Selected icon identifier for the tag."),
        ),
    )

    class _DynamicDocumentTagResultBase(BaseModel):
        """Shared validator enforcing runtime fallback-tag rules."""

        model_config = ConfigDict(extra="forbid")

        @model_validator(mode="after")
        def validate_fallback_tag_usage(self) -> Self:
            """Ensure the fallback tag is only used as a sole tag."""
            if fallback_label is None:
                return self

            labels = [getattr(tag, "label", None) for tag in self.tags]
            if fallback_label in labels and len(labels) != 1:
                raise ModelRetry(
                    "The fallback tag "
                    f"'{fallback_label}' may only be returned when no other tag applies. "
                    "If one or more real tags apply, remove the fallback tag. "
                    "If no tags apply, return only the fallback tag with the general icon."
                )
            return self

    document_result_model = create_model(
        f"DynamicDocumentTagResult_{schema_suffix}",
        __base__=_DynamicDocumentTagResultBase,
        file_name=(str, Field(..., description="Original file name of the document.")),
        tags=(
            list[tag_assignment_model],
            Field(
                ...,
                min_length=1,
                max_length=4,
                description="1-4 tag/icon assignments from the active tag vocabulary.",
            ),
        ),
    )
    batch_result_model = create_model(
        f"DynamicBatchTagResult_{schema_suffix}",
        __config__=model_config,
        results=(
            list[document_result_model],
            Field(..., description="Tagging results for the processed batch."),
        ),
    )

    return RuntimeBatchTagSchema(
        tag_literal=tag_literal,
        tag_assignment_model=tag_assignment_model,
        document_result_model=document_result_model,
        batch_result_model=batch_result_model,
    )


def build_runtime_batch_tag_schema(active_labels: Sequence[str]) -> RuntimeBatchTagSchema:
    """Build runtime-safe document tagging models for the active label set."""
    normalized = tuple(label.strip() for label in active_labels if label.strip())
    return _build_runtime_batch_tag_schema_cached(normalized)
