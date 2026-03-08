"""
Audit Assistant Agent - Pydantic AI agent with AG-UI protocol support.

This agent uses StateDeps for bidirectional state sync with the frontend
via the AG-UI protocol, enabling seamless CopilotKit integration.

"""

import logging
from typing import Any, Literal
from uuid import uuid4
from datetime import datetime
from textwrap import dedent

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent

from model_config import get_agent_model
from models import (
    A2UIComponent,
)

from llm_orchestrator import (
    _combine_documents,
    timeline_event_agent,
    summary_metrics_agent,
    findings_agent,
    tables_agent,
    charts_agent,
    audit_question_agent,
)
from prompts import format_audit_form_prompt

logger = logging.getLogger("audit_agent")


class AuditState(BaseModel):
    """Shared state between frontend and agent.

    Synchronized bidirectionally via AG-UI protocol. The frontend uses
    this state to render the three-pane layout:
    chat | documents | generative UI output.

    Attributes:
        documents: Uploaded document metadata conforming to the Document schema.
        components: Generated A2UI components for the output pane.
        audit_questions: Raw TFR question data.
        analysis_result: Raw analysis output from run_analysis.
        audit_form_result: Canonical editable audit form payload.
        current_form_id: ID of the currently active persisted form, if any.
        status: Current processing status.
        progress: Completion percentage (0-100).
        current_step: Human-readable description of current activity.
        activity_log: Timestamped log entries for UI display.
        error_message: Error details when status is 'error'.
    """

    documents: list[dict[str, Any]] = Field(default_factory=list)
    components: list[dict[str, Any]] = Field(default_factory=list)
    audit_questions: list[dict[str, Any]] = Field(default_factory=list)
    analysis_result: dict[str, Any] = Field(default_factory=dict)
    audit_form_result: dict[str, Any] = Field(default_factory=dict)
    current_form_id: str | None = None

    status: str = "idle"  # idle | analyzing | generating | complete | error
    progress: int = 0
    current_step: str = ""
    activity_log: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


# =========================================================================
# Helper: build document payloads from state
# =========================================================================


def _build_doc_payloads(state: AuditState) -> list[dict[str, Any]]:
    """Extract document payloads from shared state for LLM consumption.

    Maps from the Document schema shape used in state to the simple
    payload dicts consumed by the orchestrator functions.

    Args:
        state: The current ``AuditState`` with documents list.

    Returns:
        List of dicts with ``title``, ``content``, ``file_type``, and
        ``document_type`` suitable for the orchestrator functions.
    """
    payloads: list[dict[str, Any]] = []
    for doc in state.documents:
        payloads.append(
            {
                "title": doc.get("file_name", doc.get("content_url", "Untitled")),
                "content": doc.get("content", doc.get("text", "")),
                "file_type": doc.get("mime_type", "unknown"),
                "document_type": doc.get("document_type", ""),
            }
        )
    return payloads


def _log_tool_call(
    state: AuditState, message: str, status: str = "in_progress", tool_name: str = ""
) -> None:
    """Append or update an entry in the activity log.

    If the newest activity-log entry already represents the same tool/message,
    this function updates that existing row instead of appending a duplicate.
    This keeps the UI from showing duplicate lines when a tool transitions
    from ``in_progress`` to ``completed`` with the same message.

    Args:
        state: The current ``AuditState``.
        message: Human-readable log message.
        status: One of ``in_progress``, ``completed``, or ``error``.
        tool_name: The name of the tool that was called.
    """
    print(f"[TOOL] {tool_name}: {message}")
    timestamp = datetime.now().isoformat()

    # Prevent duplicate UI rows when status changes for the same message.
    if state.activity_log:
        last_entry = state.activity_log[-1]
        if last_entry.get("message") == message and last_entry.get("tool_name", "") == tool_name:
            last_entry["status"] = status
            last_entry["timestamp"] = timestamp
            return

    state.activity_log.append(
        {
            "id": str(uuid4()),
            "message": message,
            "timestamp": timestamp,
            "status": status,
            "tool_name": tool_name,
        }
    )


# =========================================================================
# Agent definition
# =========================================================================

agent = Agent(
    model=get_agent_model(),
    name="audit_agent",
    deps_type=StateDeps[AuditState],
    instructions=dedent("""
        You are Q-Bot, orchestrator, top-level agent, and general purpose assistant 
        for an AI-powered Quality Improvement (QI) workbench servicing the insurance 
        domain. Your task is to answer user queries using tools that render react-based 
        UI components to the user. Not every user query will require a tool call, but 
        you should always consider using tools to answer the user's query. For text only 
        outputs, consider calling the generate_text_component tool to render rich markdown 
        to the user.
        
        # TOOLS:
        Favor tools diversity over repetition.
        
        ## Document tools:
        Documents are part of a shared state with the frontend. The **current** documents 
        are ones that the user has selected or uploaded, which may change during the course 
        of the conversation. Use these tools only when the user references documents or 
        their content.
        
        • get_documents_listing: Get a metadata listing of **currently selected** documents from the 
          shared state, if any.
        • get_documents_content: Get the content of **currently selected** documents from the shared 
          state, if any. Note that each document's content will not change during the course of the conversation. Therefore if you 
          have already viewed the content of a document, you do not need to fetch it again. Use metadata to guide your use of this tool.

        ## Component tools:
        Components are react-based UI elements that are rendered in the output pane. This 
        is your primary mode of output generation.
        
        ## Text component tool:
        • generate_text_component: Generate a markdown-formatted text component and render 
          it in the output pane. Favor this tool to relay 'results' or summary 
          information to the user. Rich markdown rendering is available, including:
          headings, bullet/numbered lists, tables, block quotes, links, fenced code blocks 
          with language tags (for syntax highlighting), Mermaid diagrams via fenced 
          ```mermaid blocks, math expressions (`$...$` and `$$...$$`), GFM task checklists, 
          citations/footnotes (e.g., `[^1]`), and callouts via blockquote markers such as 
          `[!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]`, `[!CAUTION]` (with or without 
          the `>` blockquote prefix). A sanitized subset of inline HTML is also supported 
          for simple structures such as `<table>`, `<tr>`, `<td>`, `<strong>`, `<em>`. 
          Note that users are non-technical and may not ask for specific markdown formatting, or know what a mermaid diagram is. 
          Your role is to make the output pane engaging and informative for the user.
          
        ## Visual component tools:
        • generate_timeline_component: Generate a timeline component based on an input 
          specification and render it in the output pane. Favor this tool to communicate information with a temporal nature.
        • generate_summary_metrics_component: Generate a summary metrics component based 
          on an input specification and render it in the output pane. Favor this tool to communicate high level quantifiable information.
        • generate_findings_component: Generate a findings component based on an input 
          specification and render it in the output pane. Great for calling attention to important items or areas of concern.
        • generate_table_component: Generate a table component based on an input 
          specification and render it in the output pane. Favor this tool to communicate structured data.
        • generate_chart_component: Generate a chart component based on an input 
          specification and render it in the output pane. Great for supplementing an analysis with visual representations.
        
        ## Specialized tools:
        These tools trigger specialized workflows or processes. Context, e.g., documents, 
        will be loaded automatically based on the shared state.
        
        • generate_audit_form: Generate a **Targeted File Review (TFR)** audit questionnaire 
          from the **currently selected** documents in the shared state. Use this tool anytime users include "TFR" in their query.
        
        ## COMMON WORKFLOWS AND USE CASES:
        
        • When rendering components, favor the order: rich markdown text, metrics, timeline, findings, tables, charts. 
        
        • User asks to summarize a document or set of documents: check metadata listing; 
          get document content; call generate_text_component followed by a series of 
          components.
        
        • User asks to generate a timeline, summary metrics, findings, table, or chart: 
          check metadata listing; get document content if needed; call the appropriate component 
          tool(s) to generate the component(s).
        
        • User asks for a table or tables: check metadata listing; get document content if needed; 
          call generate_text_component to introduce the table(s) and then call 
          generate_table_component one or more times to generate the table(s).
        
        • User asks for a particular piece of context or citation(s): check metadata 
          listing; get document content; call generate_text_component to render GFM formatted citations. 
          Optionally create a table of citations.
          
        • User is interested in a process flow or series of events: check metadata and optionally fetch docs. 
        Generate a mermaid diagram and/or a timeline component to visualize the process flow.
        
        • User asks to generate an audit TFR form: SPECIAL USE CASE - call generate_audit_form; 
          NO need to check metadata listing or get document content.
    """).strip(),
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
    _log_tool_call(state, state.current_step, "completed", "get_documents_listing")
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
    _log_tool_call(
        state,
        "Retrieving documents content from shared state",
        "in_progress",
        "get_documents_content",
    )
    doc_payloads = _build_doc_payloads(state) or None  # None when empty
    combined_text: str | None = None
    doc_count = 0
    if doc_payloads:
        combined_text, doc_count = _combine_documents(doc_payloads)
    else:
        combined_text = "No documents available."
    _log_tool_call(
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
    _log_tool_call(state, state.current_step, "in_progress", "generate_text_component")
    component = A2UIComponent(
        type="a2ui.TextBox",
        props={
            "title": title,
            "content": content,
            "variant": variant,
        },
        zone="output",
    )
    state.components.append(component.model_dump())
    _log_tool_call(state, state.current_step, "completed", "generate_text_component")
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
    _log_tool_call(state, state.current_step, "in_progress", "generate_timeline_component")
    timeline_events_result = await timeline_event_agent.run(input_spec)
    timeline_events = timeline_events_result.output
    component = timeline_events.to_a2ui_component()
    state.components.append(component.model_dump())
    _log_tool_call(state, state.current_step, "completed", "generate_timeline_component")
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
    _log_tool_call(state, state.current_step, "in_progress", "generate_summary_metrics_component")
    summary_metrics_result = await summary_metrics_agent.run(input_spec)
    summary_metrics = summary_metrics_result.output
    component = summary_metrics.to_a2ui_component()
    state.components.append(component.model_dump())
    _log_tool_call(state, state.current_step, "completed", "generate_summary_metrics_component")
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
    _log_tool_call(state, state.current_step, "in_progress", "generate_findings_component")
    findings_result = await findings_agent.run(input_spec)
    findings = findings_result.output
    for finding in findings:
        state.components.append(finding.to_a2ui_component().model_dump())
    _log_tool_call(state, state.current_step, "completed", "generate_findings_component")
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
    _log_tool_call(state, state.current_step, "in_progress", "generate_table_component")
    table_result = await tables_agent.run(input_spec)
    table = table_result.output
    state.components.append(table.to_a2ui_component().model_dump())
    n_rows = len(table.rows)
    _log_tool_call(state, state.current_step, "completed", "generate_table_component")

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
    _log_tool_call(state, state.current_step, "in_progress", "generate_chart_component")
    chart_result = await charts_agent.run(input_spec)
    chart = chart_result.output
    state.components.append(chart.to_a2ui_component().model_dump())
    n_values = len(chart.values)
    _log_tool_call(state, state.current_step, "completed", "generate_chart_component")
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
    doc_payloads = _build_doc_payloads(state)
    combined_text, _ = _combine_documents(doc_payloads)

    state.status = "generating"
    state.progress = max(state.progress, 50)
    state.current_step = "Generating TFR audit questionnaire..."
    _log_tool_call(state, state.current_step, "in_progress", "generate_audit_form")

    # ── LLM question generation ─────────────────────────────────────────
    prompt = format_audit_form_prompt(
        combined_text, additional_instructions=additional_instructions
    )
    tfr_response = await audit_question_agent.run(prompt)
    tfr_result = tfr_response.output
    tfr_result_dict = tfr_result.model_dump()
    _log_tool_call(state, state.current_step, "completed", "generate_audit_form")

    # ── Render form component ───────────────────────────────────────────
    form_component = tfr_result.to_a2ui_component()

    state.components.append(form_component.model_dump())
    _log_tool_call(state, state.current_step, "completed", "generate_audit_form")
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
    _log_tool_call(state, state.current_step, "completed", "generate_audit_form")

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
