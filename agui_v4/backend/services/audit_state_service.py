"""Service helpers for mutating and synchronizing shared audit state."""

from domain.audit_state import AuditState
from presenters.a2ui import tfr_analysis_to_component


class AuditStateService:
    """Encapsulate form and component updates against the shared state object."""

    def __init__(self, state: AuditState) -> None:
        """Store the shared state reference.

        Args:
            state: Shared state used by AG-UI and REST endpoints.
        """
        self.state = state

    def get_audit_form_state(self) -> dict[str, object]:
        """Return the current audit-form state payload."""
        return {
            "current_form_id": self.state.current_form_id,
            "audit_form_result": self.state.audit_form_result,
        }

    def get_runtime_state(self) -> dict[str, object]:
        """Return the live runtime fields needed for incremental UI updates.

        Returns:
            A lightweight state payload containing only the fields that change
            during agent execution and are safe to poll frequently.
        """
        return {
            "status": self.state.status,
            "progress": self.state.progress,
            "current_step": self.state.current_step,
            "activity_log": self.state.activity_log,
            "error_message": self.state.error_message,
        }

    def upsert_audit_form_component(self, payload: dict[str, object]) -> None:
        """Replace or append the active audit form component.

        Args:
            payload: Canonical audit form payload.
        """
        component = tfr_analysis_to_component(payload).model_dump()
        replaced = False
        for index, item in enumerate(self.state.components):
            if item.get("type") == "a2ui.AuditQuestionForm":
                self.state.components[index] = component
                replaced = True
                break
        if not replaced:
            self.state.components.append(component)

    def sync_audit_form(
        self,
        payload: dict[str, object],
        current_form_id: str | None = None,
    ) -> dict[str, object]:
        """Persist the canonical audit form payload into shared state.

        Args:
            payload: Canonical audit form payload.
            current_form_id: Optional current form identifier to retain.

        Returns:
            Updated public audit-form state payload.
        """
        self.state.audit_form_result = payload
        self.state.audit_questions = payload["questions"]
        if current_form_id:
            self.state.current_form_id = current_form_id
        self.upsert_audit_form_component(payload)
        return self.get_audit_form_state()

    def mark_form_saved(self, form_id: str, payload: dict[str, object]) -> None:
        """Update state after a form has been saved.

        Args:
            form_id: Persisted form identifier.
            payload: Canonical audit form payload.
        """
        self.state.current_form_id = form_id
        self.state.audit_form_result = payload
        self.state.audit_questions = payload["questions"]
        self.upsert_audit_form_component(payload)

    def restore_form(self, form_id: str, payload: dict[str, object]) -> dict[str, object]:
        """Restore a saved form into active state.

        Args:
            form_id: Persisted form identifier.
            payload: Canonical audit form payload.

        Returns:
            Updated state payload returned by the restore endpoint.
        """
        self.state.current_form_id = form_id
        self.state.audit_form_result = payload
        self.state.audit_questions = payload["questions"]
        self.state.status = "complete"
        self.state.current_step = f"Restored saved form {form_id}"
        self.upsert_audit_form_component(payload)
        return {
            "message": "Form restored to state.",
            "form_id": self.state.current_form_id,
            "audit_form_result": self.state.audit_form_result,
        }

    def clear_current_form_reference(self, form_id: str) -> None:
        """Clear the active form reference when a saved form is deleted.

        Args:
            form_id: Deleted form identifier.
        """
        if self.state.current_form_id == form_id:
            self.state.current_form_id = None
