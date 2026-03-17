"""Request schemas for claim-session initialization."""

from pydantic import BaseModel, ConfigDict, Field


class ClaimSessionInitRequestBody(BaseModel):
    """Request payload for claim-session initialization.

    When ``claim_number`` is provided the session enters claim-aware mode;
    otherwise static example documents are staged automatically.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "claim_number": "CLM-2026-00042",
                "effective_date": "2026-01-15",
            }
        }
    )

    claim_number: str | None = Field(
        "",
        description="Optional claim identifier. When set, enters claim-aware mode; when empty, stages example docs.",
    )
    effective_date: str | None = Field(
        "",
        description="Optional ISO-8601 date string associated with the session.",
    )

    def normalized_claim_number(self) -> str:
        """Return the normalized (stripped) claim number."""
        return (self.claim_number or "").strip()

    def normalized_effective_date(self) -> str:
        """Return the normalized (stripped) effective date."""
        return (self.effective_date or "").strip()
