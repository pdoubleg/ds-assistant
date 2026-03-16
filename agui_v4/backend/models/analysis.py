"""Claim analysis model contracts."""

import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_serializer

from models.a2ui import A2UIComponent, A2UIConvertible


class TimelineEvent(BaseModel):
    """A single event in a claim's lifecycle."""

    date: datetime.date = Field(
        ...,
        description="Event date in ISO format (YYYY-MM-DD), e.g. '2025-03-15'.",
    )
    title: str = Field(..., description="Short headline for the event.")
    description: str = Field(..., description="One-to-two sentence detail about the event.")
    category: Literal[
        "fnol",
        "inspection",
        "estimate",
        "payment",
        "correspondence",
        "coverage_update",
        "settlement",
        "denial",
        "supplement",
        "reopen",
        "info_request",
        "info_receipt",
        "complaint",
        "demand",
        "attorney",
        "other",
    ] = Field(
        ...,
        description=(
            "Event classification. One of: fnol (First Notice of Loss — use for "
            "the initial report/claim filing), inspection, estimate, payment, "
            "correspondence, coverage_update, settlement, denial, supplement, "
            "reopen, info_request, info_receipt, complaint, demand, attorney, "
            "or other."
        ),
    )
    status: Literal["completed", "pending", "flagged", "closed"] = Field(
        ...,
        description=(
            "Current state of the event: completed, pending, flagged, or closed. "
            "Use 'closed' to mark claim closure events (e.g., settlement paid out, "
            "denial finalized). Can be paired with categories like settlement or denial."
        ),
    )

    @field_serializer("date")
    def _serialize_date(self, value: datetime.date) -> str:
        """Always emit an ISO-8601 string so downstream dicts and JSON are consistent."""
        return value.isoformat()


class TimelineEvents(A2UIConvertible):
    """A list of timeline events."""

    events: list[TimelineEvent] = Field(..., description="A list of timeline events.")

    def to_a2ui_component(self) -> A2UIComponent:
        """Convert the timeline events to an A2UI component."""
        from presenters.a2ui import timeline_events_to_component

        return timeline_events_to_component([event.model_dump() for event in self.events])


Icons = Literal[
    "dollar",
    "calendar",
    "user",
    "shield",
    "file",
    "alert",
    "home",
    "weather",
    "fire",
    "wind",
    "repair",
    "tree",
]
Trends = Literal["up", "down", "stable"]


class SummaryMetric(BaseModel):
    """A single key-value metric for a claim summary card."""

    label: str = Field(..., description="Metric name displayed above the value.")
    value: str = Field(..., description="Formatted display value (e.g. '$12,450.00', 'Open').")
    icon: Icons | None = Field(None, description="Optional icon hint.")
    trend: Trends | None = Field(
        None, description="Optional directional trend: up, down, or stable."
    )


class SummaryMetrics(A2UIConvertible):
    """A list of summary metrics."""

    metrics: list[SummaryMetric] = Field(..., description="A list of summary metrics.")

    def to_a2ui_component(self) -> A2UIComponent:
        """Convert the summary metrics to an A2UI component."""
        from presenters.a2ui import summary_metrics_to_component

        return summary_metrics_to_component([metric.model_dump() for metric in self.metrics])


FindingCategory = Literal[
    "coverage",
    "liability",
    "damages",
    "time_sensitive",
    "documentation",
    "compliance",
    "financial",
    "fraud",
    "medical",
    "subrogation",
    "vendor",
    "litigation",
    "customer_service",
    "general",
]

FindingSeverity = Literal[
    "tip",
    "info",
    "note",
    "warning",
    "critical",
    "urgent",
]


class Finding(A2UIConvertible):
    """An observation or flag surfaced by the analysis agent.

    Example usage::

        finding = Finding(
            title="Delayed Inspection",
            content="Inspection was scheduled 45 days after FNOL...",
            severity="warning",
            category="timeline",
        )
    """

    title: str = Field(..., description="Short headline for the finding.")
    content: str = Field(
        ..., description="Detailed explanation of the finding (markdown supported)."
    )
    severity: FindingSeverity = Field(
        ...,
        description=(
            "Visual severity level. One of: tip (helpful best-practice suggestion), "
            "info (neutral informational note), note (notable observation worth "
            "highlighting), warning (potential issue requiring attention), "
            "critical (significant problem), urgent (requires immediate action)."
        ),
    )
    category: FindingCategory | None = Field(
        None,
        description=(
            "Optional finding classification. One of: coverage (policy/coverage), "
            "liability (liability concerns), damages (damage assessment), time_sensitive "
            "(deadline or time-critical matters), documentation (missing/incomplete docs), "
            "compliance (regulatory/compliance), financial (monetary discrepancies), "
            "fraud (fraud indicators), medical (medical-related), subrogation "
            "(recovery opportunities), vendor (vendor/contractor issues), litigation "
            "(legal/litigation matters), customer_service (service quality), or general."
        ),
    )

    def to_a2ui_component(self) -> A2UIComponent:
        """Convert the finding to an A2UI component."""
        from presenters.a2ui import finding_to_component

        return finding_to_component(self.model_dump())


class TableSpec(A2UIConvertible):
    """Specification for a data table to be rendered."""

    caption: str = Field(..., description="Table heading / description.")
    headers: list[str] = Field(..., description="Column header labels.")
    rows: list[list[str | int | float]] = Field(
        ..., description="2-D list of cell values (strings or numbers)."
    )

    def to_a2ui_component(self) -> A2UIComponent:
        """Convert the table to an A2UI component."""
        from presenters.a2ui import table_spec_to_component

        return table_spec_to_component(
            caption=self.caption,
            headers=self.headers,
            rows=self.rows,
        )


class ChartSpec(A2UIConvertible):
    """Specification for a simple chart to be rendered."""

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

    def to_a2ui_component(self) -> A2UIComponent:
        """Convert the chart to an A2UI component."""
        from presenters.a2ui import chart_spec_to_component

        return chart_spec_to_component(
            chart_type=self.chart_type,
            title=self.title,
            labels=self.labels,
            values=self.values,
        )


class AnalysisResult(BaseModel):
    """Structured output from the ``run_analysis`` sub-agent."""

    title: str = Field(..., description="A concise title for the summary text box.")
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
