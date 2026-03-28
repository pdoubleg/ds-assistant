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
                    "peril": {
                        "peril": "Exterior",
                        "notes": "Wind-driven exterior loss.",
                    },
                    "questions": [
                        {
                            "id": "Q1",
                            "text": "Was the estimate documented appropriately?",
                            "answer": "No",
                            "sub_questions": [
                                {
                                    "id": "Q1.1",
                                    "text": "Line-item pricing support is missing.",
                                    "reasoning": "The estimate lacks supporting detail for the proposed repair scope.",
                                    "citations": "Estimate p. 4; field notes p. 2",
                                }
                            ],
                            "missing_info": None,
                        }
                    ],
                    "overall_outcome": "Does Not Meet",
                    "outcome_justification": "At least one required area lacks sufficient documented support.",
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
        description="Alternate payload wrapper accepted by the endpoint.",
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
