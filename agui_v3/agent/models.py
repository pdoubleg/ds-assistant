"""
Domain models for the TFR (Technical File Review) audit assistant.

Contains three schema groups:
    Document / Documents — Rich document representation with claim metadata,
        MIME types, content URLs, and helper methods for metadata retrieval.
    TFR Form — Structured output schema for the TFR analysis including
        peril determination, questions with sub-questions, and overall outcome.
    Analysis — Structured output for ``run_analysis``, producing timeline
        events, summary metrics, findings, tables, and charts that map
        directly to A2UI components.

Example usage:
    >>> from models import Document, Documents, TFRAnalysisResult, AnalysisResult
    >>> doc = Document(
    ...     claimNumber="CLM-001", contentId="cid-1",
    ...     mimeType="application/pdf", contentURL="/docs/report.pdf",
    ... )
    >>> print(doc.file_name)
    'report.pdf'
"""

import logging
import uuid
import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

logger = logging.getLogger(__name__)


# ============================================================================
# TFR Form Schema
# ============================================================================


class SubQuestion(BaseModel):
    """A single sub-question applicable to a TFR Question.

    Every TFR Question marked as "No" must have at least one associated
    SubQuestion identifying the specific driver / opportunity.

    Attributes:
        id: Unique identifier (e.g. 'Q1.1', 'Q1.2').
        text: The verbatim sub-question text from the TFR template.
        reasoning: Explanation of why this sub-question is an opportunity.
        citations: Specific citations to the evidence used in the reasoning.
    """

    id: str = Field(..., description="Unique identifier for the sub-question, e.g., 'Q1.1', 'Q1.2', etc.")
    text: str = Field(..., description="The verbatim text of the sub-question from the TFR template.")
    reasoning: str = Field(
        ..., description="An explanation of the reasoning behind this sub-question being selected as an opportunity."
    )
    citations: str = Field(..., description="A listing of specific citations to the evidence used in the reasoning.")
    answer: SkipJsonSchema[Literal["Yes", "No", "Insufficient information"]] = "No"
    comments: SkipJsonSchema[str | None] = Field(None, description="Optional comments on the sub-question.")

class TFRQuestion(BaseModel):
    """A single TFR Question.

    If marked as "No", the question must have at least one associated
    SubQuestion detailing the driver(s).

    Attributes:
        id: Unique identifier (e.g. 'Q1', 'Q2').
        text: The verbatim TFR question text.
        answer: Yes / No / Insufficient information.
        sub_questions: Driver sub-questions when answer is 'No'.
        missing_info: What information is missing when answer is 'Insufficient information'.
    """

    id: str = Field(..., description="Unique identifier for the TFR question, e.g., 'Q1', 'Q2', etc.")
    text: str = Field(..., description="The verbatim text of the TFR question from the TFR template.")
    answer: Literal["Yes", "No", "Insufficient information"] = Field(
        ..., description="Indicates whether this question is classified as an 'Opportunity' or 'Observation'."
    )
    sub_questions: list[SubQuestion] | None = Field(
        None, description="A list of one or more associated sub-questions if the answer is 'No'."
    )
    missing_info: str | None = Field(
        None,
        description="If the answer is 'Insufficient information', specifies what information is missing.",
    )

    @model_validator(mode="after")
    def validate_sub_questions(self) -> Self:
        """Questions marked 'No' must have at least one SubQuestion."""
        if self.answer == "No" and (not self.sub_questions or len(self.sub_questions) == 0):
            raise ValueError(
                "TFR Questions marked as 'No' (Opportunity) must have at least one associated SubQuestion."
            )
        return self

    @model_validator(mode="after")
    def validate_missing_info(self) -> Self:
        """Questions marked 'Insufficient information' must specify what is missing."""
        if self.answer == "Insufficient information" and (not self.missing_info or self.missing_info.strip() == ""):
            raise ValueError(
                "TFR Questions marked as 'Insufficient information' must specify what information is missing."
            )
        return self


class PerilDetermination(BaseModel):
    """Peril determination for the TFR analysis.

    Attributes:
        peril: Interior or Exterior peril classification.
        notes: Optional reasoning if the peril is unclear.
    """

    peril: Literal["Interior", "Exterior"] = Field(
        ..., description="The specific peril selected for this TFR analysis based on the claim information."
    )
    notes: str | None = Field(
        None,
        description="Optional notes or reasoning related to the peril determination.",
    )


class TFRAnalysisResult(BaseModel):
    """Overall TFR analysis result for a claim.

    Attributes:
        peril: The peril determination for this analysis.
        questions: All TFR Questions with answers and sub-questions.
        overall_outcome: Meets / Does Not Meet Expectations.
        outcome_justification: Concise justification for the outcome.
        additional_analysis: Optional Wind/Hail or Flooring/Cabinetry analysis.
        follow_ups: Optional recommended follow-up actions.
    """

    peril: PerilDetermination = Field(..., description="The peril determination for this TFR analysis.")
    questions: list[TFRQuestion] = Field(
        ...,
        description="All TFR Questions analyzed for the claim.",
    )
    overall_outcome: Literal["Meets", "Does Not Meet Expectations"] = Field(
        ...,
        description="The overall outcome of the TFR analysis.",
    )
    outcome_justification: str = Field(
        ...,
        description="A concise justification for the overall outcome.",
    )
    additional_analysis: str | None = Field(
        None,
        description="Optional additional analysis (e.g., Wind/Hail on EXTERIOR, Flooring/Cabinetry on INTERIOR).",
    )
    follow_ups: str | None = Field(
        None, description="Optional notes on recommended follow-up actions."
    )


# ============================================================================
# Analysis Schema (run_analysis structured output)
# ============================================================================


class TimelineEvent(BaseModel):
    """A single event in a claim's lifecycle.

    Used to populate the ``ClaimTimeline`` A2UI component.

    Attributes:
        date: Display date string (e.g. "2025-03-15" or "March 15, 2025").
        title: Short event headline.
        description: One-to-two sentence detail about what happened.
        category: Event classification for color-coding and grouping.
        status: Current state of this event.

    Example:
        >>> event = TimelineEvent(
        ...     date="2025-03-15",
        ...     title="Field Inspection Completed",
        ...     description="Adjuster conducted on-site inspection of roof damage.",
        ...     category="inspection",
        ...     status="completed",
        ... )
    """

    date: str = Field(
        ..., description="Display date for the event (e.g. '2025-03-15')."
    )
    title: str = Field(
        ..., description="Short headline for the event."
    )
    description: str = Field(
        ..., description="One-to-two sentence detail about the event."
    )
    category: Literal[
        "inspection", "estimate", "payment", "correspondence", "other"
    ] = Field(
        ...,
        description="Event classification: inspection, estimate, payment, correspondence, or other.",
    )
    status: Literal["completed", "pending", "flagged"] = Field(
        ...,
        description="Current state of the event: completed, pending, or flagged.",
    )


class SummaryMetric(BaseModel):
    """A single key-value metric for a claim summary card.

    Used to populate the ``SummaryCard`` A2UI component.

    Attributes:
        label: Metric name shown above the value.
        value: Formatted display value (e.g. "$12,450.00", "Open").
        icon: Optional icon hint for the frontend (e.g. "dollar", "calendar").
        trend: Optional directional trend indicator.

    Example:
        >>> metric = SummaryMetric(label="Total Estimate", value="$12,450.00", trend="up")
    """

    label: str = Field(..., description="Metric name displayed above the value.")
    value: str = Field(
        ..., description="Formatted display value (e.g. '$12,450.00', 'Open')."
    )
    icon: str | None = Field(
        None, description="Optional icon hint (e.g. 'dollar', 'calendar', 'user', 'shield')."
    )
    trend: Literal["up", "down", "stable"] | None = Field(
        None, description="Optional directional trend: up, down, or stable."
    )


class Finding(BaseModel):
    """An observation or flag surfaced by the analysis agent.

    Used to populate ``FindingCard`` A2UI components.

    Attributes:
        title: Short finding headline.
        content: Detailed explanation (supports markdown).
        severity: Visual urgency level.
        category: Optional grouping tag.

    Example:
        >>> finding = Finding(
        ...     title="Timeline Gap Detected",
        ...     content="There is a **14-day gap** between the inspection and the estimate.",
        ...     severity="warning",
        ...     category="timeline",
        ... )
    """

    title: str = Field(..., description="Short headline for the finding.")
    content: str = Field(
        ..., description="Detailed explanation of the finding (markdown supported)."
    )
    severity: Literal["info", "warning", "critical"] = Field(
        ..., description="Visual severity: info, warning, or critical."
    )
    category: str | None = Field(
        None,
        description="Optional grouping tag (e.g. 'timeline', 'coverage', 'estimate', 'resolution').",
    )


class TableSpec(BaseModel):
    """Specification for a data table to be rendered.

    Used to populate ``DataTable`` A2UI components.

    Attributes:
        caption: Table heading / description.
        headers: Column header labels.
        rows: 2-D list of cell values (strings or numbers).

    Example:
        >>> table = TableSpec(
        ...     caption="Estimate Line Items",
        ...     headers=["Item", "Category", "Amount"],
        ...     rows=[["Roof shingles", "Roofing", "$4,200"], ["Gutter replacement", "Exterior", "$1,800"]],
        ... )
    """

    caption: str = Field(..., description="Table heading / description.")
    headers: list[str] = Field(..., description="Column header labels.")
    rows: list[list[Any]] = Field(
        ..., description="2-D list of cell values (strings or numbers)."
    )


class ChartSpec(BaseModel):
    """Specification for a simple chart to be rendered.

    Used to populate ``SimpleChart`` A2UI components.

    Attributes:
        chart_type: Visualization type.
        title: Chart heading.
        labels: Data point labels (x-axis or pie segments).
        values: Numeric values corresponding to each label.
        colors: Optional hex color for each data point.

    Example:
        >>> chart = ChartSpec(
        ...     chart_type="bar",
        ...     title="Costs by Category",
        ...     labels=["Roofing", "Interior", "Exterior"],
        ...     values=[4200, 3100, 1800],
        ... )
    """

    chart_type: Literal["bar", "line", "pie"] = Field(
        ..., description="Chart type: bar, line, or pie."
    )
    title: str = Field(..., description="Chart heading.")
    labels: list[str] = Field(
        ..., description="Data point labels (x-axis categories or pie segments)."
    )
    values: list[float | int] = Field(
        ..., description="Numeric values corresponding to each label."
    )
    colors: list[str] | None = Field(
        None, description="Optional list of hex color strings for each data point."
    )


class AnalysisResult(BaseModel):
    """Structured output from the ``run_analysis`` sub-agent.

    Each optional section maps to one or more A2UI components. The
    sub-agent populates whichever sections are relevant given the input
    documents and the user's focus area.

    Attributes:
        title: Concise title for the summary text box.
        summary: Narrative summary rendered as a TextBox. Can be used for any general statements that should be passed to the front end and displayed.
        timeline_events: Optional Chronological events rendered as a ClaimTimeline.
        summary_metrics: Optional Key-value metrics rendered as a SummaryCard.
        findings: Optional Observations rendered as FindingCard(s).
        tables: Optional Tabular data rendered as DataTable(s).
        charts: Optional Visualizations rendered as SimpleChart(s).
    """
    title: str = Field(
        ...,
        description="A concise title for the summary text box."
    )
    summary: str = Field(
        ...,
        description=(
            "A concise narrative summary of the claim suitable for display "
            "in a summary text box. 2-4 sentences. Consider the focus; if just a general inquiry a full claim overview may not be necessary."
        ),
    )
    timeline_events: list[TimelineEvent] | None = Field(
        None,
        description=(
            "Optional chronological events in the claim lifecycle. Populate when the "
            "documents contain dated activities (inspections, estimates, "
            "payments, correspondence)."
        ),
    )
    summary_metrics: list[SummaryMetric] | None = Field(
        None,
        description=(
            "Optional key claim metrics at a glance (amounts, dates, statuses, parties). "
            "Populate when the documents contain quantifiable claim data."
        ),
    )
    findings: list[Finding] | None = Field(
        None,
        description=(
            "Optional observations, flags, or insights the auditor should be aware of. "
            "Populate when noteworthy patterns, gaps, or risks are identified."
        ),
    )
    tables: list[TableSpec] | None = Field(
        None,
        description=(
            "Optional structured tabular data such as estimate line items, coverage "
            "breakdowns, or payment history. Populate when the documents "
            "contain structured/tabular information."
        ),
    )
    charts: list[ChartSpec] | None = Field(
        None,
        description=(
            "Optional simple visualizations (bar, line, or pie) for numeric data. "
            "Populate when a visual breakdown adds clarity (e.g. costs by "
            "category, payment timeline)."
        ),
    )


# ============================================================================
# Document Schema
# ============================================================================


class DocBaseConfig(BaseModel):
    """Base model for documents with common configuration."""

    model_config = ConfigDict(
        validate_by_alias=True,
        validate_by_name=True,
        json_schema_extra={"additionalProperties": False},
        json_schema_serialization_defaults_required=True,
    )


class Document(DocBaseConfig):
    """A document related to a claim.

    Attributes:
        id: Auto-generated UUID.
        claim_number: The associated claim number.
        content_id: Unique content identifier.
        mime_type: MIME type (e.g. 'application/pdf').
        content_url: URL where the document can be accessed.
        presigned_url: Pre-signed download URL.
        domain: 'claim' or 'policy'.
        document_type: High-level type classification.
        upload_time: When the document was uploaded.
        source_system: Originating system.
        text: Extracted text content (not displayed in UI cards).
        content: Raw binary/string content (excluded from JSON schema).
        document_sub_type: Finer-grained type classification.
        document_description: Human-readable description.
        create_date: Creation timestamp.
        company_name: Associated company.
        list_of_contents: Table of contents (e.g. policy forms).

    Example:
        >>> doc = Document(
        ...     claimNumber="CLM-001", contentId="cid-1",
        ...     mimeType="application/pdf",
        ...     contentURL="/docs/report.pdf",
        ... )
        >>> doc.file_name
        'report.pdf'
    """

    id: SkipJsonSchema[uuid.UUID] = Field(default_factory=uuid.uuid4)
    claim_number: str = Field(alias="claimNumber")
    content_id: str = Field(alias="contentId")
    mime_type: str = Field(alias="mimeType")
    content_url: str = Field(alias="contentURL")
    presigned_url: str = Field(alias="presignedURL", default="")
    domain: Literal["claim", "policy"] = "claim"
    document_type: Optional[str] = Field(alias="documentType", default=None)
    upload_time: Optional[str] = Field(alias="uploadTime", default=None)
    source_system: Optional[str] = Field(alias="sourceSystem", default=None)
    text: Optional[str] = ""
    content: SkipJsonSchema[Optional[bytes | str]] = None
    document_sub_type: str | None = Field(alias="documentSubType", default=None)
    document_description: str | None = Field(alias="documentDescription", default=None)
    create_date: datetime.datetime = Field(alias="createDate", default=None)
    company_name: str | None = Field(alias="companyName", default=None)
    list_of_contents: list[dict[str, str]] | None = Field(alias="listOfContents", default=None)

    @field_serializer("create_date")
    def format_date(self, v: datetime.date) -> str:
        """Serialize create_date as a UTC string."""
        if v:
            return v.strftime("%Y-%m-%d %H:%M:%S %Z")

    @computed_field
    @property
    def file_name(self) -> str:
        """Derive the file name from content_url, falling back to content_id.

        Returns:
            The file name portion of the URL, or the content_id if unavailable.
        """
        return Path(self.content_url).name if self.content_url else self.content_id

    def as_string(self) -> str:
        """Format the document details as a human-readable string.

        Returns:
            Formatted string with file name, type, MIME type, and text content.
        """
        return (
            f"Document Name: {self.file_name}\n"
            f"Type: {self.document_type or 'N/A'}\n"
            f"MIME Type: {self.mime_type}\n"
            f"Text: {self.text or 'No text content available'}"
        )


class Documents(DocBaseConfig):
    """A collection of documents related to a claim.

    Attributes:
        documents: The list of Document objects.

    Example:
        >>> docs = Documents(documents=[doc1, doc2])
        >>> docs.valid_ids
        ['cid-1', 'cid-2']
    """

    documents: list[Document] = Field(default_factory=list)

    @property
    def valid_ids(self) -> list[str]:
        """Get content IDs for all documents that have one.

        Returns:
            List of non-empty content IDs.
        """
        return [doc.content_id for doc in self.documents if doc.content_id]

    def as_string(self) -> str:
        """Format all documents as a single string separated by dividers.

        Returns:
            Formatted string of all documents, or a fallback message.
        """
        return (
            "\n----------\n".join(document.as_string() for document in self.documents)
            if self.documents
            else "No documents available."
        )

    def as_summary_string(self, max_len: int = 50) -> str:
        """Return truncated text for each document.

        Args:
            max_len: Maximum text length per document before truncation.

        Returns:
            Formatted string with truncated text summaries.
        """
        summaries = []
        for doc in self.documents:
            truncated_text = (doc.text[:max_len] + "...") if (doc.text and len(doc.text) > max_len) else doc.text
            summaries.append(
                f"Document Name: {doc.file_name}\n"
                f"Type: {doc.document_type or 'N/A'}\n"
                f"MIME Type: {doc.mime_type}\n"
                f"Text (truncated): {truncated_text or 'No text content available'}"
            )
        return "\n----------\n".join(summaries) if summaries else "No documents available."

    def as_metadata_string(self) -> str:
        """Return metadata-only representation sorted by create_date (newest first).

        Returns:
            Formatted metadata string for all documents.
        """
        self.documents.sort(key=lambda doc: doc.create_date, reverse=True)
        metadata_list = []
        for doc in self.documents:
            meta_string = ""
            meta_string += f"CONTENT ID: {doc.content_id}\n"
            meta_string += f"Created: {doc.create_date.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            meta_string += f"MIME Type: {doc.mime_type}\n"
            meta_string += f"Type: {doc.document_type or 'N/A'}\n"
            if doc.document_sub_type:
                meta_string += f"Sub-Type: {doc.document_sub_type}\n"
            if doc.document_description:
                meta_string += f"Description: {doc.document_description}\n"
            meta_string += f"Source System: {doc.source_system or 'N/A'}\n"
            if doc.company_name:
                meta_string += f"Company Name: {doc.company_name}\n"
            # Policy packet form listing
            if doc.source_system == "GRMUC_ROOTS_PLCYPCKT" and doc.list_of_contents:
                meta_string += "Policy Forms:\n"
                for form in doc.list_of_contents:
                    form_id = form.get("formID", "N/A")
                    form_name = form.get("formName", "").strip()
                    if form_name:
                        meta_string += f"  • {form_id}: {form_name}\n"
                    else:
                        meta_string += f"  • {form_id}\n"
            metadata_list.append(meta_string)

        return "\n----------\n".join(metadata_list) if metadata_list else "No documents available."

    def get_doc_by_content_id(self, content_id: str) -> Optional[Document]:
        """Find a document by its content_id.

        Args:
            content_id: The content_id to search for.

        Returns:
            The matching Document, or None if not found.

        Raises:
            ValueError: If the content_id is not in the valid IDs list.
        """
        if content_id not in self.valid_ids:
            raise ValueError(f"Content ID {content_id} is not valid. Valid IDs are: {', '.join(self.valid_ids)}")
        return next((doc for doc in self.documents if doc.content_id == content_id), None)
