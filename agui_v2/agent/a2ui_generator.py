"""
A2UI Component Generator - Factory functions for creating audit-focused UI components.

This module provides factory functions for the simplified component catalog:
  - DocumentCard: Document display with metadata and checkbox
  - AuditQuestionForm: Audit questions with sub-questions
  - TextBox: General-purpose text display
  - DataTable: Structured tabular data
  - SimpleChart: Simple bar/line/pie charts

Example usage:
    >>> from a2ui_generator import generate_document_card, generate_text_box
    >>> card = generate_document_card(title="Policy.pdf", file_type="pdf", ...)
    >>> text = generate_text_box(title="Summary", content="Analysis results...")
"""

from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field


class A2UIComponent(BaseModel):
    """Represents a single A2UI component to be rendered on the frontend.

    Attributes:
        id: Unique component identifier.
        type: Component type string (e.g., 'a2ui.DocumentCard').
        props: Component-specific properties passed to the React renderer.
        layout: Optional layout hints (width, position, className).
        zone: Semantic zone for layout grouping.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    props: dict[str, Any] = Field(default_factory=dict)
    layout: dict[str, Any] | None = None
    zone: str | None = None


# ============================================================================
# DOCUMENT CARD
# ============================================================================

def generate_document_card(
    title: str,
    file_type: str = "pdf",
    file_size: str = "",
    page_count: int | None = None,
    upload_date: str = "",
    summary: str = "",
    selected: bool = False,
    tags: list[str] | None = None,
) -> A2UIComponent:
    """Generate a DocumentCard component for displaying an uploaded document.

    Args:
        title: Document filename or title.
        file_type: File extension (pdf, docx, xlsx).
        file_size: Human-readable file size (e.g., '2.4 MB').
        page_count: Number of pages in the document.
        upload_date: ISO date string of when the document was uploaded.
        summary: Brief summary of document contents.
        selected: Whether the document is currently selected.
        tags: Optional list of category tags.

    Returns:
        A2UIComponent configured as a DocumentCard.

    Example:
        >>> card = generate_document_card(
        ...     title="Corporate Policy v2.1.pdf",
        ...     file_type="pdf",
        ...     file_size="2.4 MB",
        ...     page_count=45,
        ...     summary="Corporate governance and compliance policy.",
        ...     tags=["governance", "compliance"]
        ... )
    """
    return A2UIComponent(
        type="a2ui.DocumentCard",
        props={
            "title": title,
            "file_type": file_type,
            "file_size": file_size,
            "page_count": page_count,
            "upload_date": upload_date,
            "summary": summary,
            "selected": selected,
            "tags": tags or [],
        },
        zone="documents",
    )


# ============================================================================
# AUDIT QUESTION FORM
# ============================================================================

def generate_audit_question_form(
    questions: list[dict[str, Any]],
) -> A2UIComponent:
    """Generate an AuditQuestionForm component for displaying audit questions.

    Each question should have the structure:
        {
            "id": "AQ-001",
            "question": "Does the organization have a documented...",
            "rating": "Yes" | "No" | "NA" | null,
            "comments": "",
            "sub_questions": [
                {
                    "id": "AQ-001-a",
                    "question": "What specific gap was identified?",
                    "rating": "Yes" | "No" | "NA" | null,
                    "comments": ""
                }
            ]
        }

    Top-level questions are rated Yes/No/NA by the user. When a question
    is rated "No", at least one sub-question must also be rated "No" to
    serve as the driver identifying the specific deficiency.

    Args:
        questions: List of audit question dictionaries.

    Returns:
        A2UIComponent configured as an AuditQuestionForm.

    Example:
        >>> form = generate_audit_question_form(questions=[
        ...     {
        ...         "id": "AQ-001",
        ...         "question": "Is there a documented access control policy?",
        ...         "rating": None,
        ...         "comments": "",
        ...         "sub_questions": [
        ...             {
        ...                 "id": "AQ-001-a",
        ...                 "question": "What gaps exist in the current policy?",
        ...                 "comments": ""
        ...             }
        ...         ]
        ...     }
        ... ])
    """
    return A2UIComponent(
        type="a2ui.AuditQuestionForm",
        props={"questions": questions},
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
