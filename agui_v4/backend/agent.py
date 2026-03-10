"""Composition root for the top-level AG-UI audit agent.

This module defines the shared `Agent(...)` instance and registers the user-facing
tools that bridge AG-UI state, workflow modules, and A2UI presenters.
"""

import logging
from typing import Literal

from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent

from agent_instructions import AUDIT_AGENT_INSTRUCTIONS
from domain.audit_state import AuditState
from model_config import get_agent_model
from presenters.a2ui import (
    chart_spec_to_component,
    finding_to_component,
    generate_text_box,
    summary_metrics_to_component,
    table_spec_to_component,
    tfr_analysis_to_component,
    timeline_events_to_component,
)
from services.agent_helpers import build_doc_payloads_from_state, log_tool_call
from services.document_mapper import DocumentMapper
from workflows.agent_factory import (
    charts_agent,
    findings_agent,
    summary_metrics_agent,
    tables_agent,
    timeline_event_agent,
)
from workflows.audit_form import generate_audit_questions

logger = logging.getLogger("audit_agent")


# =========================================================================
# Agent definition
# =========================================================================

agent = Agent(
    model=get_agent_model(),
    name="audit_agent",
    deps_type=StateDeps[AuditState],
    instructions=AUDIT_AGENT_INSTRUCTIONS,
    model_settings=OpenAIChatModelSettings(
        parallel_tool_calls=False,
    ),
)


# =========================================================================
# Tool: Context documents
# =========================================================================


@agent.tool
def get_documents_listing(ctx: RunContext[StateDeps[AuditState]]) -> ToolReturn:
    """Get a listing of current documents from the shared state and return it as a string.

    Returns:
        ToolReturn with a string containing the document listing and a state snapshot.
    """
    state = ctx.deps.state
    state.current_step = "Retrieving documents..."

    docs = state.documents
    logger.info(f"[TOOL] get_documents_listing: {len(docs)} documents")
    if not docs:
        return_value = "No documents uploaded yet."
    else:
        return_value = "\n".join(
            f"- {d.get('file_name', d.get('content_url', 'Untitled'))} ({d.get('mime_type', 'unknown')})"
            for d in docs
        )
    state.current_step = f"Synced {len(docs)} document(s)."
    log_tool_call(state, state.current_step, "completed", "get_documents_listing")
    return ToolReturn(
        return_value=return_value,
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=state,
            ),
        ],
    )


@agent.tool
def get_documents_content(ctx: RunContext[StateDeps[AuditState]]) -> ToolReturn:
    """Get the content of the current documents from the shared state and return it as a string.

    Returns:
        ToolReturn with a string containing the content of the documents and a state snapshot.
    """
    state = ctx.deps.state
    state.current_step = "Retrieving documents content..."
    log_tool_call(
        state,
        "Retrieving documents content from shared state",
        "in_progress",
        "get_documents_content",
    )
    mapper = DocumentMapper()
    doc_payloads = build_doc_payloads_from_state(state) or None  # None when empty
    combined_text: str | None = None
    doc_count = 0
    if doc_payloads:
        combined_text, doc_count = mapper.combine_documents(doc_payloads)
    else:
        combined_text = "No documents available."
    log_tool_call(
        state, f"Retrieved {doc_count} document(s) content.", "completed", "get_documents_content"
    )

    return ToolReturn(
        return_value=combined_text,
        metadata=[StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)],
    )


# =========================================================================
# Tools: Insight components
# =========================================================================


@agent.tool
async def generate_text_component(
    ctx: RunContext[StateDeps[AuditState]],
    title: str,
    content: str,
    variant: Literal["info", "warning", "success", "error"] = "info",
) -> ToolReturn:
    """Generate a markdown-formatted text component and render it in the output pane.

    Notes:
        - Useful for persisting arbitrary text content in the output pane.
        - Renders as rich react-markdown with github-flavored markdown support.
        - Supported markdown patterns include:
            - headings, emphasis, lists, links, tables, block quotes
            - fenced code blocks with language tags (for syntax highlighting), e.g., ```python
            - Mermaid diagrams via fenced mermaid code blocks, e.g., ```mermaid
            - math via inline/block delimiters, e.g., `$x^2$` and `$$x^2 + y^2 = z^2$$`
            - checklist items, e.g., `- [x] complete` and `- [ ] pending`
            - citations/footnotes, e.g., `Some claim[^1]` and `[^1]: citation text`
            - callout/admonition blocks, e.g., `[!WARNING] Validate coverage limits` or
              `> [!WARNING] Validate coverage limits`
            - simple sanitized inline HTML where needed, e.g.,
              `<table><tr><td><strong>Label</strong></td><td>Value</td></tr></table>`
        - Common use cases include:
            - Displaying a summary, e.g., a summary of an analysis or executive summary
            - Introducing subsequent components or sections
            - General output related to a user query that should be displayed in the output pane
        - Use this tool often to display summary information or context before or after
          calling other components or sections.

    Args:
        ctx: Pydantic AI run context carrying the shared ``AuditState``.
        title: The title of the text component.
        content: The content of the text component. This should be a string containing the markdown-formatted text.
        variant: The variant of the text component. Must be one of "info", "warning", "success", or "error". Defaults to "info".

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    state = ctx.deps.state
    state.current_step = f"Generating text component: {title}, {content[:100]}..., {variant}"
    log_tool_call(state, state.current_step, "in_progress", "generate_text_component")
    component = generate_text_box(title=title, content=content, variant=variant)
    state.components.append(component.model_dump())
    log_tool_call(state, state.current_step, "completed", "generate_text_component")
    return ToolReturn(
        return_value=f"Text component generated: {title}",
        metadata=[StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)],
    )


@agent.tool
async def generate_timeline_component(
    ctx: RunContext[StateDeps[AuditState]],
    input_spec: str,
) -> ToolReturn:
    """Spawn a sub-agent to generate a timeline component based on an input specification and render it in the output pane.

    Notes:
        Each event must include:
        - date (str)
        - title (str)
        - description (str)
        - category: "inspection" | "estimate" | "payment" | "correspondence" | "other"
        - status: "completed" | "pending" | "flagged"

    Args:
        ctx: Pydantic AI run context carrying the shared ``AuditState``.
        input_spec: The input specification for the timeline component. This should be a string containing the timeline event details, e.g., Chronological bullets with: date, title, description, category, status.


    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    state = ctx.deps.state
    state.current_step = f"Generating timeline component: {input_spec[:100]}..."
    log_tool_call(state, state.current_step, "in_progress", "generate_timeline_component")
    timeline_events_result = await timeline_event_agent.run(input_spec)
    timeline_events = timeline_events_result.output
    component = timeline_events_to_component(
        [event.model_dump() for event in timeline_events.events]
    )
    state.components.append(component.model_dump())
    log_tool_call(state, state.current_step, "completed", "generate_timeline_component")
    return ToolReturn(
        return_value=f"Timeline component generated with {len(timeline_events.events)} event(s).",
        metadata=[StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)],
    )


@agent.tool
async def generate_summary_metrics_component(
    ctx: RunContext[StateDeps[AuditState]],
    input_spec: str,
) -> ToolReturn:
    """Spawn a sub-agent to generate a summary metrics component based on an input specification and render it in the output pane.

    Notes:
        Each metric should include:
        - label (str): Metric name shown above the value.
        - value (str): Formatted display value (e.g. "$12,450.00", "Open").
        - icon (str | null): Optional icon hint for the frontend (e.g. "dollar", "calendar", "user", "shield", "file", "alert", "home", "weather", "fire", "wind", "repair", "tree").
        - trend (str | null): Optional directional trend indicator (e.g. "up", "down", "stable").

    Args:
        ctx: Pydantic AI run context carrying the shared ``AuditState``.
        input_spec: The input specification for the summary metrics component, e.g., Metric bullets with: label, value, icon(optional), trend(optional).

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    state = ctx.deps.state
    state.current_step = f"Generating summary metrics component: {input_spec[:100]}..."
    log_tool_call(state, state.current_step, "in_progress", "generate_summary_metrics_component")
    summary_metrics_result = await summary_metrics_agent.run(input_spec)
    summary_metrics = summary_metrics_result.output
    component = summary_metrics_to_component(
        [metric.model_dump() for metric in summary_metrics.metrics]
    )
    state.components.append(component.model_dump())
    log_tool_call(state, state.current_step, "completed", "generate_summary_metrics_component")
    metric_count = len(summary_metrics.metrics)
    if metric_count == 0:
        return_message = "Summary metrics component generated with 0 metric(s)."
    else:
        return_message = f"Summary metrics component generated with {metric_count} metric(s)."
    return ToolReturn(
        return_value=return_message,
        metadata=[StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)],
    )


@agent.tool
async def generate_findings_component(
    ctx: RunContext[StateDeps[AuditState]],
    input_spec: str,
) -> ToolReturn:
    """Spawn a sub-agent to generate one or more findings component(s) based on an input specification and render it in the output pane. Useful for calling attention to important items or areas of concern.

    Notes:
        Each finding should include:
        - title (str): Short finding headline.
        - content (str): Detailed explanation (supports markdown).
        - severity: "info" | "warning" | "critical"
        - category (str | null): Optional grouping tag (e.g. "timeline", "coverage", "estimate", "resolution").

    Args:
        ctx: Pydantic AI run context carrying the shared ``AuditState``.
        input_spec: The input specification for the findings component, e.g., Finding bullets with: title, content, severity(optional), category(optional). This should be a string containing the finding details.

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    state = ctx.deps.state
    state.current_step = f"Generating findings component: {input_spec[:100]}..."
    log_tool_call(state, state.current_step, "in_progress", "generate_findings_component")
    findings_result = await findings_agent.run(input_spec)
    findings = findings_result.output
    for finding in findings:
        state.components.append(finding_to_component(finding.model_dump()).model_dump())
    log_tool_call(state, state.current_step, "completed", "generate_findings_component")
    return ToolReturn(
        return_value=f"Findings component generated with {len(findings)} finding(s).",
        metadata=[StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)],
    )


@agent.tool
async def generate_table_component(
    ctx: RunContext[StateDeps[AuditState]],
    input_spec: str,
) -> ToolReturn:
    """Spawn a sub-agent to generate a table component based on an input specification and render it in the output pane. Useful for displaying structured data in a tabular format.

    Notes:
        Each table should include:
        - caption (str): Table heading / description.
        - headers (list[str]): Column header labels.
        - rows (list[list[str | int | float]]): 2-D list of cell values (strings or numbers).

    Args:
        ctx: Pydantic AI run context carrying the shared ``AuditState``.
        input_spec: The input specification for the table component, e.g., String with: caption, headers, rows. This should be a string containing the table details.

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    state = ctx.deps.state
    state.current_step = f"Generating table component: {input_spec[:100]}..."
    log_tool_call(state, state.current_step, "in_progress", "generate_table_component")
    table_result = await tables_agent.run(input_spec)
    table = table_result.output
    state.components.append(
        table_spec_to_component(
            caption=table.caption,
            headers=table.headers,
            rows=table.rows,
        ).model_dump()
    )
    n_rows = len(table.rows)
    log_tool_call(state, state.current_step, "completed", "generate_table_component")

    return ToolReturn(
        return_value=f"Table component generated with {n_rows} row(s).",
        metadata=[StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)],
    )


@agent.tool
async def generate_chart_component(
    ctx: RunContext[StateDeps[AuditState]],
    input_spec: str,
) -> ToolReturn:
    """Spawn a sub-agent to generate a chart component based on an input specification and render it in the output pane. Useful for displaying visual data in a chart format.

    Notes:
        For each chart, provide:
        - chart_type: one of "bar" | "line" | "pie"
        - title: short, auditor-friendly label
        - labels: list[str] (category names, periods, or segment names)
        - values: list[float | int] (raw numeric values only; no "$", "%", commas, or text)
        - colors (optional): list[str] of valid CSS color strings (hex preferred, e.g. "#003B6F")
        - labels and values MUST align one-to-one and have identical length.
        - Chart type expectations:
            - bar: compare magnitudes across discrete categories (e.g., costs by trade/category).
            - line: show ordered progression over time/sequences; labels should be chronologically or logically ordered.
            - pie: show parts of a whole at one point in time; values should be non-negative and represent a meaningful total breakdown.
            - Keep chart payloads compact and readable (typically 3-8 data points per chart).
            - If source data is ambiguous, do not force a chart.

    Args:
        ctx: Pydantic AI run context carrying the shared ``AuditState``.
        input_spec: The input specification for the chart component, e.g., String with: chart_type, title, labels, values, colors(optional). This should be a string containing the chart details.

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    state = ctx.deps.state
    state.current_step = f"Generating chart component: {input_spec[:100]}..."
    log_tool_call(state, state.current_step, "in_progress", "generate_chart_component")
    chart_result = await charts_agent.run(input_spec)
    chart = chart_result.output
    state.components.append(
        chart_spec_to_component(
            chart_type=chart.chart_type,
            title=chart.title,
            labels=chart.labels,
            values=chart.values,
            colors=chart.colors,
        ).model_dump()
    )
    n_values = len(chart.values)
    log_tool_call(state, state.current_step, "completed", "generate_chart_component")
    return ToolReturn(
        return_value=f"Chart component generated with {n_values} value(s).",
        metadata=[StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)],
    )


# =========================================================================
# Tool: generate audit form → TFR question set
# =========================================================================


@agent.tool
async def generate_audit_form(
    ctx: RunContext[StateDeps[AuditState]],
    additional_instructions: str = "",
) -> ToolReturn:
    """Generate a TFR audit questionnaire from the selected documents in the shared state.

    Reads raw document content from state, calls the TFR question
    sub-agent, and renders an AuditQuestionForm A2UI component with
    peril determination, questions, and overall outcome.

    Args:
        ctx: Pydantic AI run context carrying the shared ``AuditState``.
        additional_instructions: Optional additional instructions for the audit form sub-agent.
            (e.g. "initial handling only.").

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """

    state = ctx.deps.state
    doc_payloads = build_doc_payloads_from_state(state)

    state.status = "generating"
    state.progress = max(state.progress, 50)
    state.current_step = "Generating TFR audit questionnaire..."
    log_tool_call(state, state.current_step, "in_progress", "generate_audit_form")

    # Route the full TFR generation flow through the shared workflow layer so
    # routes and AG-UI use the same document mapping and prompt composition.
    tfr_result_dict = await generate_audit_questions(
        doc_payloads,
        additional_instructions=additional_instructions,
    )
    log_tool_call(state, state.current_step, "completed", "generate_audit_form")

    # ── Render form component ───────────────────────────────────────────
    form_component = tfr_analysis_to_component(tfr_result_dict)

    state.components.append(form_component.model_dump())
    log_tool_call(state, state.current_step, "completed", "generate_audit_form")
    state.audit_questions = tfr_result_dict["questions"]
    # Keep a canonical form payload in state so frontend edits can sync
    # and persistence endpoints can save without reconstructing fields.
    state.audit_form_result = {
        "peril": tfr_result_dict["peril"],
        "questions": tfr_result_dict["questions"],
        "overall_outcome": tfr_result_dict["overall_outcome"],
        "outcome_justification": tfr_result_dict["outcome_justification"],
        "additional_analysis": tfr_result_dict.get("additional_analysis"),
        "follow_ups": tfr_result_dict.get("follow_ups"),
    }
    # New generation = new distinct form; clear any previous form ID so the
    # next save creates a fresh record instead of overwriting the old one.
    state.current_form_id = None

    num_questions = len(tfr_result_dict["questions"])
    state.status = "complete"
    state.progress = 100
    state.current_step = f"Generated {num_questions} TFR questions"
    log_tool_call(state, state.current_step, "completed", "generate_audit_form")

    return ToolReturn(
        return_value=(
            f"TFR audit questionnaire generated with {num_questions} questions. "
            f"Peril: {tfr_result_dict['peril']['peril']}. "
            f"Outcome: {tfr_result_dict['overall_outcome']}. "
            f"The form has been rendered in the output pane."
        ),
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=state,
            ),
        ],
    )
