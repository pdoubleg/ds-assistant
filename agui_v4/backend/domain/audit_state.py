"""Domain models for shared audit session state."""

from typing import Any

from pydantic import BaseModel, Field


class AuditState(BaseModel):
    """Shared state synchronized between the frontend and AG-UI agent.

    The frontend uses this state to render the document pane, output pane, and
    editable audit form while the agent updates progress and activity metadata.

    Attributes:
        documents: Uploaded or selected document payloads.
        components: Generated A2UI component payloads for the output pane.
        audit_questions: Raw TFR question data.
        analysis_result: Structured analysis payload retained for reuse.
        audit_form_result: Canonical editable audit form payload.
        current_form_id: Persisted form identifier currently active in state.
        claim_number: Active claim number shown in the UI.
        effective_date: Effective date associated with the current claim.
        status: High-level workflow status.
        progress: Completion percentage from 0 to 100.
        current_step: Human-readable description of current work.
        activity_log: Timestamped tool and workflow log rows for the UI.
        error_message: Optional error details for failure states.
    """

    documents: list[dict[str, Any]] = Field(default_factory=list)
    components: list[dict[str, Any]] = Field(default_factory=list)
    audit_questions: list[dict[str, Any]] = Field(default_factory=list)
    analysis_result: dict[str, Any] = Field(default_factory=dict)
    audit_form_result: dict[str, Any] = Field(default_factory=dict)
    current_form_id: str | None = None
    claim_number: str = ""
    effective_date: str = ""

    status: str = "idle"
    progress: int = 0
    current_step: str = ""
    activity_log: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None
