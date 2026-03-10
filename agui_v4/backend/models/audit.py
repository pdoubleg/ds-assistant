"""Audit and TFR-related model contracts."""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from models.a2ui import A2UIComponent, A2UIConvertible


class SubQuestion(BaseModel):
    """A single sub-question applicable to a TFR Question.

    Every TFR Question marked as "No" must have at least one associated
    SubQuestion identifying the specific driver / opportunity.
    """

    id: str = Field(
        ..., description="Unique identifier for the sub-question, e.g., 'Q1.1', 'Q1.2', etc."
    )
    text: str = Field(
        ..., description="The verbatim text of the sub-question from the TFR template."
    )
    reasoning: str = Field(
        ...,
        description="An explanation of the reasoning behind this sub-question being selected as an opportunity.",
    )
    citations: str = Field(
        ..., description="A listing of specific citations to the evidence used in the reasoning."
    )
    answer: SkipJsonSchema[Literal["Yes", "No", "Insufficient information"]] = "No"
    comments: SkipJsonSchema[str | None] = Field(
        None, description="Optional comments on the sub-question."
    )


class TFRQuestion(BaseModel):
    """A single TFR Question."""

    id: str = Field(
        ..., description="Unique identifier for the TFR question, e.g., 'Q1', 'Q2', etc."
    )
    text: str = Field(
        ..., description="The verbatim text of the TFR question from the TFR template."
    )
    answer: Literal["Yes", "No", "Insufficient information"] = Field(
        ...,
        description="Indicates whether this question is classified as an 'Opportunity' or 'Observation'.",
    )
    sub_questions: list[SubQuestion] | None = Field(
        None, description="A list of one or more associated sub-questions if the answer is 'No'."
    )
    missing_info: str | None = Field(
        None,
        description="If the answer is 'Insufficient information', specifies what information is missing.",
    )

    @model_validator(mode="after")
    def validate_sub_questions(self) -> Self:
        """Questions marked 'No' must have at least one SubQuestion."""
        if self.answer == "No" and (not self.sub_questions or len(self.sub_questions) == 0):
            raise ValueError(
                "TFR Questions marked as 'No' (Opportunity) must have at least one associated SubQuestion."
            )
        return self

    @model_validator(mode="after")
    def validate_missing_info(self) -> Self:
        """Questions marked 'Insufficient information' must specify what is missing."""
        if self.answer == "Insufficient information" and (
            not self.missing_info or self.missing_info.strip() == ""
        ):
            raise ValueError(
                "TFR Questions marked as 'Insufficient information' must specify what information is missing."
            )
        return self


class PerilDetermination(BaseModel):
    """Peril determination for the TFR analysis."""

    peril: Literal["Interior", "Exterior"] = Field(
        ...,
        description="The specific peril selected for this TFR analysis based on the claim information.",
    )
    notes: str | None = Field(
        None,
        description="Optional notes or reasoning related to the peril determination.",
    )


class TFRAnalysisResult(A2UIConvertible):
    """Overall TFR analysis result for a claim."""

    peril: PerilDetermination = Field(
        ..., description="The peril determination for this TFR analysis."
    )
    questions: list[TFRQuestion] = Field(
        ...,
        description="All TFR Questions analyzed for the claim.",
    )
    overall_outcome: Literal["Meets", "Does Not Meet Expectations"] = Field(
        ...,
        description="The overall outcome of the TFR analysis.",
    )
    outcome_justification: str = Field(
        ...,
        description="A concise justification for the overall outcome.",
    )
    additional_analysis: str | None = Field(
        None,
        description="Optional additional analysis (e.g., Wind/Hail on EXTERIOR, Flooring/Cabinetry on INTERIOR).",
    )
    follow_ups: str | None = Field(
        None, description="Optional notes on recommended follow-up actions."
    )

    def to_a2ui_component(self) -> A2UIComponent:
        """Convert the TFR analysis result to an A2UI component."""
        from presenters.a2ui import tfr_analysis_to_component

        return tfr_analysis_to_component(self.model_dump())
