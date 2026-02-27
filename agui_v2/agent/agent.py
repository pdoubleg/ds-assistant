"""
Audit Assistant Agent - Pydantic AI agent with AG-UI protocol support.

This agent uses StateDeps for bidirectional state sync with the frontend
via the AG-UI protocol, enabling seamless CopilotKit integration.

Two independent tools are available:
    run_analysis – Analyzes claim documents and generates visual insight
                   components (timeline, summary card, findings, tables,
                   charts) to brief the auditor before review.
    generate_audit_form – Reads docs directly and generates the TFR
                          question set rendered as an AuditQuestionForm.

The main agent decides which tools to call (either, both, or neither)
based on the user query.
"""

from typing import Any
from uuid import uuid4
from datetime import datetime
from textwrap import dedent

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent

from model_config import get_agent_model
from a2ui_generator import (
    generate_audit_question_form,
    generate_claim_timeline,
    generate_data_table,
    generate_finding_card,
    generate_simple_chart,
    generate_summary_card,
    generate_text_box,
)


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
        payloads.append({
            "title": doc.get("file_name", doc.get("content_url", "Untitled")),
            "content": doc.get("content", doc.get("text", "")),
            "file_type": doc.get("mime_type", "unknown"),
            "document_type": doc.get("document_type", ""),
        })
    return payloads


def _log(state: AuditState, message: str, status: str = "in_progress") -> None:
    """Append an entry to the activity log.

    Args:
        state: The current ``AuditState``.
        message: Human-readable log message.
        status: One of ``in_progress``, ``completed``, or ``error``.
    """
    state.activity_log.append({
        "id": str(uuid4()),
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "status": status,
    })


# =========================================================================
# Agent definition
# =========================================================================

agent = Agent(
    model=get_agent_model(),
    name="audit_agent",
    deps_type=StateDeps[AuditState],
    instructions=dedent("""
        You are an insurance claim audit assistant. You help auditors
        understand a property claim before they begin their Technical File
        Review (TFR). You have two independent tools:

        • run_analysis — Analyze claim documents (estimates, notes, policy
          details, vendor reports, scope sheets) and generate visual insight
          components: a claim timeline, summary metrics, findings, data
          tables, and charts. Pass an optional *focus* string to steer
          the analysis (e.g. "timeline and damaged items"). The tool uses
          the same document context as the audit (whatever is selected
          and/or uploaded). Documents can also be None.
          If the user asks for an example, demo, or sample analysis
          (and no documents are uploaded), call run_analysis with a
          descriptive focus like "example property claim with wind damage"
          — the sub-agent will generate realistic fictional data.
        • generate_audit_form — Read uploaded documents and generate a
          structured TFR audit questionnaire with peril determination,
          questions, sub-questions with reasoning and citations, and
          an overall outcome assessment.

        Use whichever tool(s) make sense for the user's request. You can
        call one, both, or neither. They are completely independent.

    """).strip(),
)


# =========================================================================
# Tool: list documents
# =========================================================================

@agent.tool
def get_documents(ctx: RunContext[StateDeps[AuditState]]) -> ToolReturn:
    """Get the list of currently uploaded documents from state.

    Returns:
        ToolReturn with document listing and a state snapshot.
    """
    docs = ctx.deps.state.documents
    print(f"[TOOL] get_documents: {len(docs)} documents")
    if not docs:
        return_value = "No documents uploaded yet."
    else:
        return_value = "\n".join(
            f"- {d.get('file_name', d.get('content_url', 'Untitled'))} ({d.get('mime_type', 'unknown')})"
            for d in docs
        )
    return ToolReturn(
        return_value=return_value,
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=ctx.deps.state,
            ),
        ],
    )


# =========================================================================
# Tool: run_analysis → claim insight components
# =========================================================================

@agent.tool
async def run_analysis(
    ctx: RunContext[StateDeps[AuditState]],
    focus: str = "General claim review",
) -> ToolReturn:
    """Analyze claim documents and generate visual insight components.

    Calls the analysis sub-agent which returns an ``AnalysisResult`` with
    optional sections (timeline, metrics, findings, tables, charts). Each
    populated section is mapped to one or more A2UI components and appended
    to ``AuditState.components``.

    Args:
        ctx: Pydantic AI run context carrying the shared ``AuditState``.
        focus: User-supplied focus area forwarded to the sub-agent
            (e.g. "timeline and damaged items").

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    from llm_orchestrator import run_analysis as _run_analysis

    state = ctx.deps.state
    doc_payloads = _build_doc_payloads(state) or None  # None when empty

    print(f"[TOOL] run_analysis: {len(doc_payloads or [])} document(s), focus='{focus}'")

    state.status = "analyzing"
    state.progress = 10
    state.current_step = "Analyzing claim documents..."
    _log(state, f"Starting analysis — focus: {focus}")

    # ── LLM analysis ────────────────────────────────────────────────────
    analysis = await _run_analysis(doc_payloads, focus=focus)
    state.analysis_result = analysis.model_dump()
    state.progress = 50
    state.current_step = "Building insight components..."
    _log(state, "Analysis complete — building components", "completed")

    # ── Map AnalysisResult sections → A2UI components ───────────────────
    components_before = len(state.components)

    # 1. Overview text box (always present)
    state.components.append(
        generate_text_box(
            title="Claim Overview",
            content=analysis.claim_overview,
            variant="info",
        ).model_dump()
    )

    # 2. Summary metrics card
    if analysis.summary_metrics:
        state.components.append(
            generate_summary_card(
                title="Claim Summary",
                metrics=[m.model_dump() for m in analysis.summary_metrics],
            ).model_dump()
        )

    # 3. Timeline
    if analysis.timeline_events:
        state.components.append(
            generate_claim_timeline(
                title="Claim Timeline",
                events=[e.model_dump() for e in analysis.timeline_events],
            ).model_dump()
        )

    # 4. Findings (one card per finding)
    if analysis.findings:
        for finding in analysis.findings:
            state.components.append(
                generate_finding_card(
                    title=finding.title,
                    content=finding.content,
                    severity=finding.severity,
                    category=finding.category,
                ).model_dump()
            )

    # 5. Tables
    if analysis.tables:
        for table in analysis.tables:
            state.components.append(
                generate_data_table(
                    headers=table.headers,
                    rows=table.rows,
                    caption=table.caption,
                ).model_dump()
            )

    # 6. Charts
    if analysis.charts:
        for chart in analysis.charts:
            state.components.append(
                generate_simple_chart(
                    chart_type=chart.chart_type,
                    title=chart.title,
                    labels=chart.labels,
                    values=chart.values,
                    colors=chart.colors,
                ).model_dump()
            )

    new_count = len(state.components) - components_before
    state.progress = 100
    state.status = "complete"
    state.current_step = f"Generated {new_count} insight components"
    _log(state, f"Built {new_count} components", "completed")

    print(f"[TOOL] run_analysis: {new_count} components created")

    # Build a concise return value for the LLM
    sections = []
    if analysis.timeline_events:
        sections.append(f"{len(analysis.timeline_events)} timeline events")
    if analysis.summary_metrics:
        sections.append(f"{len(analysis.summary_metrics)} summary metrics")
    if analysis.findings:
        sections.append(f"{len(analysis.findings)} findings")
    if analysis.tables:
        sections.append(f"{len(analysis.tables)} tables")
    if analysis.charts:
        sections.append(f"{len(analysis.charts)} charts")

    return ToolReturn(
        return_value=(
            f"Claim analysis complete. Generated {new_count} insight components: "
            f"{', '.join(sections) if sections else 'overview only'}. "
            f"Components are now visible in the output pane."
        ),
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=state,
            ),
        ],
    )


# =========================================================================
# Tool: generate audit form → TFR question set
# =========================================================================

@agent.tool
async def generate_audit_form(ctx: RunContext[StateDeps[AuditState]]) -> ToolReturn:
    """Generate a TFR audit questionnaire from uploaded documents.

    Reads raw document content from state, calls the TFR question
    sub-agent, and renders an AuditQuestionForm A2UI component with
    peril determination, questions, and overall outcome.

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    from llm_orchestrator import generate_audit_questions

    state = ctx.deps.state
    doc_payloads = _build_doc_payloads(state)

    print(f"[TOOL] generate_audit_form: {len(doc_payloads)} document(s)")

    state.status = "generating"
    state.progress = max(state.progress, 50)
    state.current_step = "Generating TFR audit questionnaire..."
    _log(state, "Starting TFR question generation")

    # ── LLM question generation ─────────────────────────────────────────
    tfr_result = await generate_audit_questions(doc_payloads)

    # ── Render form component ───────────────────────────────────────────
    form_component = generate_audit_question_form(
        peril=tfr_result["peril"],
        questions=tfr_result["questions"],
        overall_outcome=tfr_result["overall_outcome"],
        outcome_justification=tfr_result["outcome_justification"],
        additional_analysis=tfr_result.get("additional_analysis"),
        follow_ups=tfr_result.get("follow_ups"),
    )
    state.components.append(form_component.model_dump())
    state.audit_questions = tfr_result["questions"]

    num_questions = len(tfr_result["questions"])
    state.status = "complete"
    state.progress = 100
    state.current_step = f"Generated {num_questions} TFR questions"
    _log(state, f"Created TFR questionnaire with {num_questions} questions", "completed")

    print(f"[TOOL] generate_audit_form: {num_questions} questions")

    return ToolReturn(
        return_value=(
            f"TFR audit questionnaire generated with {num_questions} questions. "
            f"Peril: {tfr_result['peril']['peril']}. "
            f"Outcome: {tfr_result['overall_outcome']}. "
            f"The form has been rendered in the output pane."
        ),
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=state,
            ),
        ],
    )
