"""Presenter helpers for converting backend data into A2UI payloads."""

from typing import Any

from models.a2ui import A2UIComponent


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
    """Build a document card component payload."""
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


def generate_audit_question_form(
    peril: dict[str, Any],
    questions: list[dict[str, Any]],
    overall_outcome: str,
    outcome_justification: str,
    additional_analysis: str | None = None,
    follow_ups: str | None = None,
) -> A2UIComponent:
    """Build an audit question form component payload."""
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


def generate_text_box(title: str, content: str, variant: str = "info") -> A2UIComponent:
    """Build a markdown text-box component payload."""
    return A2UIComponent(
        type="a2ui.TextBox",
        props={
            "title": title,
            "content": content,
            "variant": variant,
        },
        zone="output",
    )


def generate_data_table(
    headers: list[str],
    rows: list[list[Any]],
    caption: str = "",
    sortable: bool = False,
) -> A2UIComponent:
    """Build a data-table component payload."""
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


def generate_simple_chart(
    chart_type: str,
    title: str,
    labels: list[str],
    values: list[float | int],
) -> A2UIComponent:
    """Build a simple-chart component payload."""
    return A2UIComponent(
        type="a2ui.SimpleChart",
        props={
            "chart_type": chart_type,
            "title": title,
            "labels": labels,
            "values": values,
        },
        zone="output",
    )


def generate_timeline(title: str, events: list[dict[str, Any]]) -> A2UIComponent:
    """Build a claim timeline component payload."""
    return A2UIComponent(
        type="a2ui.ClaimTimeline",
        props={
            "title": title,
            "events": events,
        },
        layout={"width": "full"},
        zone="output",
    )


def generate_summary_card(title: str, metrics: list[dict[str, Any]]) -> A2UIComponent:
    """Build a summary-card component payload."""
    return A2UIComponent(
        type="a2ui.SummaryCard",
        props={
            "title": title,
            "metrics": metrics,
        },
        layout={"width": "full"},
        zone="output",
    )


def generate_finding_card(
    title: str,
    content: str,
    severity: str = "info",
    category: str | None = None,
) -> A2UIComponent:
    """Build a finding-card component payload."""
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


def tfr_analysis_to_component(payload: dict[str, Any]) -> A2UIComponent:
    """Convert canonical TFR payload data into an A2UI component."""
    return generate_audit_question_form(
        peril=payload["peril"],
        questions=payload["questions"],
        overall_outcome=payload["overall_outcome"],
        outcome_justification=payload["outcome_justification"],
        additional_analysis=payload.get("additional_analysis"),
        follow_ups=payload.get("follow_ups"),
    )


def timeline_events_to_component(events: list[dict[str, Any]]) -> A2UIComponent:
    """Convert timeline event payloads into an A2UI component."""
    return generate_timeline(title="Timeline", events=events)


def summary_metrics_to_component(metrics: list[dict[str, Any]]) -> A2UIComponent:
    """Convert summary metric payloads into an A2UI component."""
    return generate_summary_card(title="Summary Metrics", metrics=metrics)


def finding_to_component(payload: dict[str, Any]) -> A2UIComponent:
    """Convert one finding payload into an A2UI component."""
    return generate_finding_card(
        title=payload["title"],
        content=payload["content"],
        severity=payload["severity"],
        category=payload.get("category"),
    )


def table_spec_to_component(
    caption: str, headers: list[str], rows: list[list[Any]]
) -> A2UIComponent:
    """Convert a table specification into an A2UI component."""
    return generate_data_table(headers=headers, rows=rows, caption=caption, sortable=True)


def chart_spec_to_component(
    chart_type: str,
    title: str,
    labels: list[str],
    values: list[float | int],
) -> A2UIComponent:
    """Convert a chart specification into an A2UI component."""
    return generate_simple_chart(
        chart_type=chart_type,
        title=title,
        labels=labels,
        values=values,
    )
