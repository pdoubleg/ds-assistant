"""Request schemas for claim-session initialization."""

from pydantic import BaseModel


class ClaimSessionInitRequestBody(BaseModel):
    """Request payload for claim-session initialization.

    Attributes:
        claim_number: Optional claim identifier for claim-aware mode.
        effective_date: Optional ISO date string associated with the session.
    """

    claim_number: str | None = ""
    effective_date: str | None = ""

    def normalized_claim_number(self) -> str:
        """Return the normalized claim number."""
        return (self.claim_number or "").strip()

    def normalized_effective_date(self) -> str:
        """Return the normalized effective date."""
        return (self.effective_date or "").strip()
