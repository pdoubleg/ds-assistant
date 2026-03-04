"""
A2UI Component Generator - Factory functions for creating audit UI components.

This module provides factory functions for the A2UI component catalog:

  Audit components (unchanged):
    - DocumentCard: Document display with metadata and checkbox
    - AuditQuestionForm: TFR audit questions with sub-questions, peril, and outcome

  Analysis / insight components:
    - TextBox: General-purpose text display
    - DataTable: Structured tabular data
    - SimpleChart: Simple bar/line/pie charts
    - ClaimTimeline: Vertical timeline of claim lifecycle events
    - SummaryCard: Grid of key-value metric pairs
    - FindingCard: Observation card with severity level

Example usage:
    >>> from a2ui_generator import generate_document_card, generate_text_box
    >>> card = generate_document_card(file_name="Policy.pdf", mime_type="application/pdf")
    >>> text = generate_text_box(title="Summary", content="Analysis results...")
"""

from typing import Any

from models import A2UIComponent


# ============================================================================
# DOCUMENT CARD
# ============================================================================


def generate_document_card(
    file_name: str,
    mime_type: str,
    content_id: str = "",
    claim_number: str = "",
    content_url: str = "",
    domain: str = "claim",
    document_type: str | None = None,
    document_sub_type: str | None = None,
    document_description: str | None = None,
    create_date: str = "",
    source_system: str | None = None,
    company_name: str | None = None,
    selected: bool = False,
) -> A2UIComponent:
    """Generate a DocumentCard component for displaying a claim document.

    Props are aligned with the Document model schema — no text field is
    included since it should not be displayed in the card UI.

    Args:
        file_name: Document filename (derived from content_url in the model).
        mime_type: MIME type (e.g. 'application/pdf').
        content_id: Unique content identifier.
        claim_number: Associated claim number.
        content_url: URL where the document can be accessed.
        domain: 'claim' or 'policy'.
        document_type: High-level type classification.
        document_sub_type: Finer-grained type classification.
        document_description: Human-readable description.
        create_date: ISO date string of creation time.
        source_system: Originating system name.
        company_name: Associated company name.
        selected: Whether the document is currently selected.

    Returns:
        A2UIComponent configured as a DocumentCard.

    Example:
        >>> card = generate_document_card(
        ...     file_name="Policy_v2.1.pdf",
        ...     mime_type="application/pdf",
        ...     document_type="Policy",
        ...     document_description="Corporate governance and compliance policy.",
        ...     domain="claim",
        ... )
    """
    return A2UIComponent(
        type="a2ui.DocumentCard",
        props={
            "file_name": file_name,
            "mime_type": mime_type,
            "content_id": content_id,
            "claim_number": claim_number,
            "content_url": content_url,
            "domain": domain,
            "document_type": document_type,
            "document_sub_type": document_sub_type,
            "document_description": document_description,
            "create_date": create_date,
            "source_system": source_system,
            "company_name": company_name,
            "selected": selected,
        },
        zone="documents",
    )


# ============================================================================
# AUDIT QUESTION FORM (TFR)
# ============================================================================


def generate_audit_question_form(
    peril: dict[str, Any],
    questions: list[dict[str, Any]],
    overall_outcome: str,
    outcome_justification: str,
    additional_analysis: str | None = None,
    follow_ups: str | None = None,
) -> A2UIComponent:
    """Generate an AuditQuestionForm component for TFR audit questions.

    Each question should conform to the TFRQuestion schema:
        {
            "id": "Q1",
            "text": "Was the estimate completed accurately?",
            "answer": "Yes" | "No" | "Insufficient information",
            "sub_questions": [
                {
                    "id": "Q1.1",
                    "text": "Specific sub-question text",
                    "reasoning": "Explanation of the finding",
                    "citations": "Evidence references"
                }
            ],
            "missing_info": "What info is missing (when answer is Insufficient information)"
        }

    The model pre-populates answers, reasoning, and citations. The UI
    renders them as editable fields so users can refine.

    Args:
        peril: PerilDetermination dict with 'peril' (Interior/Exterior) and optional 'notes'.
        questions: List of TFRQuestion dicts.
        overall_outcome: 'Meets' or 'Does Not Meet Expectations'.
        outcome_justification: Concise justification for the outcome.
        additional_analysis: Optional additional analysis text.
        follow_ups: Optional follow-up action recommendations.

    Returns:
        A2UIComponent configured as an AuditQuestionForm.

    Example:
        >>> form = generate_audit_question_form(
        ...     peril={"peril": "Exterior", "notes": None},
        ...     questions=[{
        ...         "id": "Q1",
        ...         "text": "Was the roof inspection documented?",
        ...         "answer": "No",
        ...         "sub_questions": [{
        ...             "id": "Q1.1",
        ...             "text": "What documentation is missing?",
        ...             "reasoning": "No photos of the damaged area were included.",
        ...             "citations": "Claim file section 3.2",
        ...         }],
        ...     }],
        ...     overall_outcome="Does Not Meet Expectations",
        ...     outcome_justification="Multiple documentation gaps identified.",
        ... )
    """
    return A2UIComponent(
        type="a2ui.AuditQuestionForm",
        props={
            "peril": peril,
            "questions": questions,
            "overall_outcome": overall_outcome,
            "outcome_justification": outcome_justification,
            "additional_analysis": additional_analysis,
            "follow_ups": follow_ups,
        },
        layout={"width": "full"},
        zone="output",
    )


# ============================================================================
# TEXT BOX
# ============================================================================


def generate_text_box(
    title: str,
    content: str,
    variant: str = "info",
) -> A2UIComponent:
    """Generate a TextBox component for displaying general text content.

    Args:
        title: Heading for the text box.
        content: Main text content (supports markdown-like formatting).
        variant: Visual variant - 'info', 'warning', 'success', or 'error'.

    Returns:
        A2UIComponent configured as a TextBox.

    Example:
        >>> box = generate_text_box(
        ...     title="Analysis Summary",
        ...     content="The document covers 5 key compliance areas...",
        ...     variant="info"
        ... )
    """
    return A2UIComponent(
        type="a2ui.TextBox",
        props={
            "title": title,
            "content": content,
            "variant": variant,
        },
        zone="output",
    )


# ============================================================================
# DATA TABLE
# ============================================================================


def generate_data_table(
    headers: list[str],
    rows: list[list[Any]],
    caption: str = "",
    sortable: bool = False,
) -> A2UIComponent:
    """Generate a DataTable component for structured tabular data.

    Args:
        headers: Column header strings.
        rows: 2D array of cell values.
        caption: Optional table caption.
        sortable: Whether columns are sortable by clicking headers.

    Returns:
        A2UIComponent configured as a DataTable.

    Example:
        >>> table = generate_data_table(
        ...     headers=["Risk Area", "Level", "Questions"],
        ...     rows=[
        ...         ["Access Control", "High", 5],
        ...         ["Data Protection", "Medium", 3],
        ...     ],
        ...     caption="Risk Distribution Summary"
        ... )
    """
    return A2UIComponent(
        type="a2ui.DataTable",
        props={
            "headers": headers,
            "rows": rows,
            "caption": caption,
            "sortable": sortable,
        },
        layout={"width": "full"},
        zone="output",
    )


# ============================================================================
# SIMPLE CHART
# ============================================================================


def generate_simple_chart(
    chart_type: str,
    title: str,
    labels: list[str],
    values: list[float | int],
    colors: list[str] | None = None,
) -> A2UIComponent:
    """Generate a SimpleChart component for data visualization.

    Args:
        chart_type: Chart type - 'bar', 'line', or 'pie'.
        title: Chart title.
        labels: Data labels for the x-axis or pie segments.
        values: Numeric values corresponding to each label.
        colors: Optional list of hex colors for each data point.

    Returns:
        A2UIComponent configured as a SimpleChart.

    Example:
        >>> chart = generate_simple_chart(
        ...     chart_type="bar",
        ...     title="Questions by Risk Level",
        ...     labels=["High", "Medium", "Low"],
        ...     values=[8, 12, 5],
        ...     colors=["#ef4444", "#f59e0b", "#22c55e"]
        ... )
    """
    return A2UIComponent(
        type="a2ui.SimpleChart",
        props={
            "chart_type": chart_type,
            "title": title,
            "labels": labels,
            "values": values,
            "colors": colors or [],
        },
        zone="output",
    )


# ============================================================================
# CLAIM TIMELINE
# ============================================================================


def generate_timeline(
    title: str,
    events: list[dict[str, Any]],
) -> A2UIComponent:
    """Generate a ClaimTimeline component showing claim lifecycle events.

    Renders a vertical timeline with date markers, event cards, category
    badges, and status indicators. Events are displayed in the order
    provided (the caller should pre-sort chronologically).

    Each event dict should have:
        - ``date`` (str): Display date (e.g. "2025-03-15").
        - ``title`` (str): Short event headline.
        - ``description`` (str): 1-2 sentence detail.
        - ``category`` (str): "inspection" | "estimate" | "payment" |
          "correspondence" | "other".
        - ``status`` (str): "completed" | "pending" | "flagged".

    Args:
        title: Timeline heading.
        events: List of event dicts conforming to the shape above.

    Returns:
        A2UIComponent configured as a ClaimTimeline.

    Example:
        >>> timeline = generate_claim_timeline(
        ...     title="Claim Timeline",
        ...     events=[
        ...         {
        ...             "date": "2025-03-10",
        ...             "title": "Claim Filed",
        ...             "description": "Homeowner reported wind damage to roof.",
        ...             "category": "correspondence",
        ...             "status": "completed",
        ...         },
        ...         {
        ...             "date": "2025-03-15",
        ...             "title": "Field Inspection",
        ...             "description": "Adjuster inspected the property.",
        ...             "category": "inspection",
        ...             "status": "completed",
        ...         },
        ...     ],
        ... )
    """
    return A2UIComponent(
        type="a2ui.ClaimTimeline",
        props={
            "title": title,
            "events": events,
        },
        layout={"width": "full"},
        zone="output",
    )


# ============================================================================
# SUMMARY CARD
# ============================================================================


def generate_summary_card(
    title: str,
    metrics: list[dict[str, Any]],
) -> A2UIComponent:
    """Generate a SummaryCard component displaying key claim metrics.

    Renders a responsive grid of metric tiles. Each tile shows a label,
    a prominent value, and optional trend/icon decorations.

    Each metric dict should have:
        - ``label`` (str): Metric name (e.g. "Total Estimate").
        - ``value`` (str): Formatted display value (e.g. "$12,450.00").
        - ``icon`` (str | None): Optional icon hint — "dollar", "calendar",
          "user", "shield", "file", "alert".
        - ``trend`` (str | None): Optional trend — "up", "down", "stable".

    Args:
        title: Card heading.
        metrics: List of metric dicts conforming to the shape above.

    Returns:
        A2UIComponent configured as a SummaryCard.

    Example:
        >>> card = generate_summary_card(
        ...     title="Claim Summary",
        ...     metrics=[
        ...         {"label": "Total Estimate", "value": "$12,450.00", "icon": "dollar", "trend": "up"},
        ...         {"label": "Deductible", "value": "$1,000.00", "icon": "dollar"},
        ...         {"label": "Status", "value": "Open", "icon": "alert"},
        ...     ],
        ... )
    """
    return A2UIComponent(
        type="a2ui.SummaryCard",
        props={
            "title": title,
            "metrics": metrics,
        },
        layout={"width": "full"},
        zone="output",
    )


# ============================================================================
# FINDING CARD
# ============================================================================


def generate_finding_card(
    title: str,
    content: str,
    severity: str = "info",
    category: str | None = None,
) -> A2UIComponent:
    """Generate a FindingCard component for an agent observation.

    Renders an alert-style card with a severity-colored left border
    (blue for info, amber for warning, red for critical), a title,
    markdown-rendered content, and an optional category badge.

    Args:
        title: Finding headline.
        content: Detailed explanation (supports markdown).
        severity: Visual urgency — "info", "warning", or "critical".
        category: Optional grouping tag (e.g. "timeline", "coverage",
            "estimate", "resolution", "documentation").

    Returns:
        A2UIComponent configured as a FindingCard.

    Example:
        >>> card = generate_finding_card(
        ...     title="Timeline Gap Detected",
        ...     content="There is a **14-day gap** between the inspection and the estimate.",
        ...     severity="warning",
        ...     category="timeline",
        ... )
    """
    return A2UIComponent(
        type="a2ui.FindingCard",
        props={
            "title": title,
            "content": content,
            "severity": severity,
            "category": category,
        },
        zone="output",
    )
