"""Request schemas for audit-form state and persistence endpoints."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class AuditFormRequestBody(BaseModel):
    """Flexible request schema for audit form sync and persistence endpoints.

    The payload may be wrapped in `audit_form_result`, wrapped in `form`, or
    passed flat at the top level.
    """

    model_config = ConfigDict(extra="allow")

    audit_form_result: dict[str, Any] | None = None
    form: dict[str, Any] | None = None
    current_form_id: str | None = None
    id: str | None = None
    title: str | None = None
    source_docs: list[Any] = []

    def extract_form_payload(self) -> dict[str, Any]:
        """Extract the canonical audit form payload from the request body."""
        if isinstance(self.audit_form_result, dict):
            return self.audit_form_result
        if isinstance(self.form, dict):
            return self.form
        return self.model_extra or {}
