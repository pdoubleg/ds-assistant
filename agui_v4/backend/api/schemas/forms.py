"""Request schemas for audit-form state and persistence endpoints."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuditFormRequestBody(BaseModel):
    """Flexible request schema for audit form sync and persistence endpoints.

    The payload may be wrapped in ``audit_form_result``, wrapped in ``form``,
    or passed flat at the top level.  The extraction logic in
    ``extract_form_payload`` resolves these variants into a single canonical
    dict.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "audit_form_result": {
                    "sections": [
                        {
                            "title": "Coverage Verification",
                            "questions": ["Is the policy effective?"],
                        }
                    ]
                },
                "title": "Q1 Audit Questionnaire",
            }
        },
    )

    audit_form_result: dict[str, Any] | None = Field(
        None,
        description="Primary payload wrapper — the full audit-form result dict.",
    )
    form: dict[str, Any] | None = Field(
        None,
        description="Alternate payload wrapper accepted for backward compatibility.",
    )
    current_form_id: str | None = Field(
        None,
        description="ID of the form currently loaded in the frontend, used by the PUT sync endpoint.",
    )
    id: str | None = Field(
        None,
        description="Explicit form ID to use when saving. Omit to auto-generate.",
    )
    title: str | None = Field(
        None,
        description="Human-readable form title for display in the forms list.",
    )
    source_docs: list[Any] = Field(
        default_factory=list,
        description="Optional list of source-document references associated with this form.",
    )

    def extract_form_payload(self) -> dict[str, Any]:
        """Extract the canonical audit form payload from the request body.

        Resolution order:

        1. ``audit_form_result`` if present.
        2. ``form`` if present.
        3. Any extra fields passed at the top level.

        Returns:
            dict[str, Any]: The resolved form payload dictionary.
        """
        if isinstance(self.audit_form_result, dict):
            return self.audit_form_result
        if isinstance(self.form, dict):
            return self.form
        return self.model_extra or {}
