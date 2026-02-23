"""
Audit Assistant Agent - Pydantic AI agent with AG-UI protocol support.

This agent uses StateDeps for bidirectional state sync with the frontend
via the AG-UI protocol, enabling seamless CopilotKit integration.

Two independent tools are available:
    analyze_documents – Reviews docs, produces stats, and generates UI
                        components (summary card, topics table, risk chart).
    generate_audit_form – Reads docs directly and generates the audit
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
    generate_text_box,
    generate_data_table,
    generate_simple_chart,
)


class AuditState(BaseModel):
    """Shared state between frontend and agent.

    Synchronized bidirectionally via AG-UI protocol. The frontend uses
    this state to render the three-pane layout:
    chat | documents | generative UI output.

    Attributes:
        documents: Uploaded document metadata (+ extracted content).
        components: Generated A2UI components for the output pane.
        audit_questions: Raw audit question data.
        document_review: Lightweight review stats from analyze_documents.
        status: Current processing status.
        progress: Completion percentage (0-100).
        current_step: Human-readable description of current activity.
        activity_log: Timestamped log entries for UI display.
        error_message: Error details when status is 'error'.
    """
    documents: list[dict[str, Any]] = Field(default_factory=list)
    components: list[dict[str, Any]] = Field(default_factory=list)
    audit_questions: list[dict[str, Any]] = Field(default_factory=list)
    document_review: dict[str, Any] = Field(default_factory=dict)

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

    Args:
        state: The current ``AuditState`` with documents list.

    Returns:
        List of dicts with ``title``, ``content``, ``file_type``, and
        ``page_count`` suitable for the orchestrator functions.
    """
    payloads: list[dict[str, Any]] = []
    for doc in state.documents:
        payloads.append({
            "title": doc.get("title", "Untitled"),
            "content": doc.get("content", doc.get("summary", "")),
            "file_type": doc.get("file_type", "unknown"),
            "page_count": doc.get("page_count") or 0,
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
        You are a helpful audit assistant. You have two independent tools:

        • analyze_documents — Summarize uploaded documents and generate
          visual insight components (summary card, topic table, risk chart).
        • generate_audit_form — Read uploaded documents and generate a
          structured audit questionnaire.

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
            f"- {d.get('title', 'Untitled')} ({d.get('file_type', 'unknown')})"
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
# Tool: analyze documents → stats + UI components
# =========================================================================

@agent.tool
async def analyze_documents(ctx: RunContext[StateDeps[AuditState]]) -> ToolReturn:
    """Review uploaded documents and generate visual insight components.

    Calls the document-review sub-agent to produce lightweight stats
    (key topics, risk areas, summary), then renders TextBox, DataTable,
    and SimpleChart A2UI components from those stats.

    Returns:
        ToolReturn with a summary string for the LLM and a
        StateSnapshotEvent to sync updated state with the frontend.
    """
    from llm_orchestrator import review_documents

    state = ctx.deps.state
    doc_payloads = _build_doc_payloads(state)

    print(f"[TOOL] analyze_documents: {len(doc_payloads)} document(s)")

    state.status = "analyzing"
    state.progress = 10
    state.current_step = "Reviewing documents..."
    _log(state, "Starting document review")

    # ── LLM review ──────────────────────────────────────────────────────
    review = await review_documents(doc_payloads)
    state.document_review = review.model_dump()
    state.progress = 50
    state.current_step = "Building insight components..."
    _log(state, f"Review done — {review.document_type}, {len(review.key_topics)} topics", "completed")

    # ── Component 1: Summary TextBox ────────────────────────────────────
    summary_text = (
        f"{review.summary}\n\n"
        f"Reviewed {review.total_pages} pages across {review.doc_count} document(s). "
        f"Found {len(review.key_topics)} key topics and {len(review.risk_areas)} risk areas."
    )
    state.components.append(
        generate_text_box(
            title="Document Review Summary",
            content=summary_text.strip(),
            variant="info",
        ).model_dump()
    )

    # ── Component 2: Key Topics Table ───────────────────────────────────
    if review.key_topics:
        state.components.append(
            generate_data_table(
                headers=["#", "Topic"],
                rows=[[str(i + 1), t] for i, t in enumerate(review.key_topics)],
                caption="Key Topics Identified",
                sortable=True,
            ).model_dump()
        )

    # ── Component 3: Risk Area Chart ────────────────────────────────────
    if review.risk_areas:
        palette = ["#ef4444", "#f97316", "#f59e0b", "#eab308",
                    "#84cc16", "#22c55e", "#14b8a6", "#06b6d4"]
        state.components.append(
            generate_simple_chart(
                chart_type="bar",
                title="Identified Risk Areas",
                labels=review.risk_areas[:8],
                values=[1] * min(len(review.risk_areas), 8),
                colors=palette[:min(len(review.risk_areas), 8)],
            ).model_dump()
        )

    state.progress = 60
    state.current_step = f"Generated {len(state.components)} insight components"
    _log(state, f"Built {len(state.components)} components", "completed")

    # Only mark complete if audit form isn't pending
    if state.status == "analyzing":
        state.status = "complete"
        state.progress = 100
        state.current_step = "Document review complete!"

    num_components = len(state.components)
    print(f"[TOOL] analyze_documents: {num_components} components created")

    return ToolReturn(
        return_value=(
            f"Document review complete. Reviewed {review.total_pages} pages "
            f"across {review.doc_count} document(s). Found "
            f"{len(review.key_topics)} key topics, "
            f"{len(review.risk_areas)} risk areas. "
            f"Generated {num_components} insight components."
        ),
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=state,
            ),
        ],
    )


# =========================================================================
# Tool: generate audit form → question set
# =========================================================================

@agent.tool
async def generate_audit_form(ctx: RunContext[StateDeps[AuditState]]) -> ToolReturn:
    """Generate an audit questionnaire from uploaded documents.

    Reads raw document content from state, calls the audit-question
    sub-agent, and renders an AuditQuestionForm A2UI component.
    Completely independent of ``analyze_documents``.

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
    state.current_step = "Generating audit questionnaire..."
    _log(state, "Starting audit question generation")

    # ── LLM question generation ─────────────────────────────────────────
    questions = await generate_audit_questions(doc_payloads)

    # ── Render form component ───────────────────────────────────────────
    form_component = generate_audit_question_form(questions=questions)
    state.components.append(form_component.model_dump())
    state.audit_questions = questions

    state.status = "complete"
    state.progress = 100
    state.current_step = f"Generated {len(questions)} audit questions"
    _log(state, f"Created questionnaire with {len(questions)} questions", "completed")

    num_questions = len(questions)
    print(f"[TOOL] generate_audit_form: {num_questions} questions")

    return ToolReturn(
        return_value=(
            f"Audit questionnaire generated with {num_questions} questions "
            f"across the uploaded documents. The form has been rendered "
            f"in the output pane."
        ),
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=state,
            ),
        ],
    )
