"""Domain models for shared audit session state."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from models.documents import Document, Documents

logger = logging.getLogger(__name__)


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

    def get_documents(self) -> Documents:
        """Build a typed ``Documents`` collection from the raw state dicts.

        Converts each entry in ``self.documents`` into a ``Document`` model
        using ``model_validate``.  A default ``claim_number`` is injected
        from the state-level field when the individual dict lacks one.
        Malformed entries are silently skipped so a single bad payload
        does not break the entire collection.

        Returns:
            A ``Documents`` instance backed by all successfully parsed docs.

        Example usage::

            state = AuditState(claim_number="012345678", documents=[...])
            docs = state.get_documents()
            doc = docs.get_doc_by_content_id("abc123")
        """
        parsed: list[Document] = []
        for raw in self.documents:
            merged = {"claim_number": self.claim_number or "", **raw}
            try:
                parsed.append(Document.model_validate(merged))
            except Exception:
                logger.debug(
                    "Skipping unparseable document entry: %s",
                    raw.get("file_name", raw.get("content_id", "<unknown>")),
                )
        return Documents(documents=parsed)
