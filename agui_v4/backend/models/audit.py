"""Audit and TFR-related model contracts."""

import json
import os
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from models.a2ui import A2UIComponent, A2UIConvertible


class SubQuestion(BaseModel):
    """Represents a single sub-question applicable to a TFR question.

    Notes:
        - Every TFR question marked as an "Opportunity" must have at least one
          associated sub-question.
        - Sub-questions denote specific drivers related to the parent question.
        - Multiple sub-questions may exist when multiple distinct drivers apply.
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
    answer: SkipJsonSchema[bool] = False
    help_text: SkipJsonSchema[str | None] = None


class TFRQuestion(BaseModel):
    """Represents a single TFR question.

    Note:
        - If marked as "No", the question must include at least one associated
          sub-question that describes the specific opportunity or driver.
    """

    id: str = Field(
        ..., description="Unique identifier for the TFR question, e.g., 'Q1', 'Q2', etc."
    )
    text: str = Field(
        ...,
        description="The verbatim text of the TFR question from the TFR template, excluding any help text in parentheses.",
    )
    answer: Literal["Yes", "No", "Insufficient information"] = Field(
        ...,
        description="The answer for this question.",
    )
    sub_questions: list[SubQuestion] | None = Field(
        default_factory=list,
        description="A list of one or more associated sub-questions if the answer is 'No'.",
    )
    missing_info: str | None = Field(
        None,
        description="If the answer is 'Insufficient information', this field should specify what information is missing to make a determination.",
    )
    help_text: SkipJsonSchema[str | None] = None

    @model_validator(mode="after")
    def validate_sub_questions(self) -> Self:
        """Validate that ``No`` answers include at least one sub-question.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If a ``No`` answer omits sub-questions.
        """
        if self.answer == "No" and (not self.sub_questions or len(self.sub_questions) == 0):
            raise ValueError(
                "TFR Questions marked as 'No' (Opportunity) must have at least one associated SubQuestion."
            )
        return self

    @model_validator(mode="after")
    def validate_missing_info(self) -> Self:
        """Validate that insufficient-information answers explain what is missing.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If an insufficient-information answer omits details.
        """
        if self.answer == "Insufficient information" and (
            not self.missing_info or self.missing_info.strip() == ""
        ):
            raise ValueError(
                "TFR Questions marked as 'Insufficient information' must specify what information is missing in the 'missing_info' field."
            )
        return self


class PerilDetermination(BaseModel):
    """Represents the peril determination for the TFR analysis."""

    peril: str = Field(..., description="The specific peril for this TFR analysis.")
    notes: str | None = Field(
        None,
        description="Optional notes or reasoning related to the peril determination. For example if the peril is unclear.",
    )


class TFRAnalysisResult(A2UIConvertible):
    """Represents the overall TFR analysis result for a claim."""

    peril: PerilDetermination = Field(
        ..., description="The peril determination for this TFR analysis."
    )
    questions: list[TFRQuestion] = Field(
        ...,
        description="A list of all TFR Questions analyzed for the claim, including their answers and any associated SubQuestions.",
    )
    overall_outcome: Literal["Meets", "Does Not Meet"] = Field(
        ...,
        description="The overall outcome of the TFR analysis based on the classification of the individual questions.",
    )
    outcome_justification: str = Field(
        ...,
        description="A concise justification for the overall outcome, synthesizing the reasoning from all questions and sub-questions.",
    )
    id: SkipJsonSchema[str | None] = None
    cost: SkipJsonSchema[float | None] = None
    image_cost: SkipJsonSchema[float | None] = None
    latency: SkipJsonSchema[float | None] = None
    ground_truth: SkipJsonSchema[str | None] = None
    extras: SkipJsonSchema[str | None] = None

    def __str__(self) -> str:
        """Render the analysis as a compact markdown summary.

        Returns:
            A markdown string containing the peril, question hierarchy, and
            outcome details while omitting empty optional fields.
        """

        def append_optional_line(lines: list[str], label: str, value: str | None) -> None:
            """Append a markdown bullet when the value contains content.

            Args:
                lines: Collected markdown lines.
                label: Prefix label for the bullet.
                value: Optional value to render.
            """
            if value and value.strip():
                lines.append(f"- {label}: {value.strip()}")

        lines: list[str] = [
            "## Peril",
            f"- `{self.peril.peril}`",
        ]
        append_optional_line(lines, "Notes", self.peril.notes)

        lines.append("")
        lines.append("## Questions")

        for question in self.questions:
            lines.append(f"### `{question.id}` - {question.answer}")
            lines.append(question.text.strip())
            append_optional_line(lines, "Help text", question.help_text)
            append_optional_line(lines, "Missing info", question.missing_info)

            if question.sub_questions:
                for sub_question in question.sub_questions:
                    lines.append(f"#### `{sub_question.id}`")
                    lines.append(sub_question.text.strip())
                    append_optional_line(lines, "Help text", sub_question.help_text)
                    lines.append(f"- Reasoning: {sub_question.reasoning.strip()}")
                    lines.append(f"- Citations: {sub_question.citations.strip()}")

            lines.append("")

        lines.extend(
            [
                "## Outcome",
                f"- {self.overall_outcome}",
                f"- Justification: {self.outcome_justification.strip()}",
            ]
        )

        if lines[-1] == "":
            lines.pop()

        return "\n".join(lines)

    @property
    def has_insufficient_info(self) -> bool:
        """Determine whether any question is marked insufficient.

        Returns:
            ``True`` when at least one question has an insufficient-information answer.
        """
        return any(question.answer == "Insufficient information" for question in self.questions)

    def to_json(self, path: str) -> None:
        """Save the model to a JSON file with explicit flush semantics.

        Args:
            path: Destination file path.
        """
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(self.model_dump(), file_obj, indent=4)
            file_obj.flush()
            os.fsync(file_obj.fileno())

    @classmethod
    def from_json(cls, path: str) -> Self:
        """Load a validated TFR analysis result from JSON.

        Args:
            path: Source JSON file path.

        Returns:
            A validated ``TFRAnalysisResult`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed or validated.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON file not found: {path}")

            with open(path, "r", encoding="utf-8") as file_obj:
                model_dict = json.load(file_obj)

            return cls.model_validate(model_dict)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON format in file {path}: {exc}") from exc
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Error loading TFRAnalysisResult from JSON file {path}: {exc}"
            ) from exc

    def as_questionnaire_string(self) -> str:
        """Format the analysis as a questionnaire-like prompt string.

        Returns:
            A canonical questionnaire string suitable for prompting the LLM.

        Example:
            >>> result = TFRAnalysisResult(
            ...     peril=PerilDetermination(peril="Interior"),
            ...     questions=[
            ...         TFRQuestion(
            ...             id="Q1",
            ...             text="Was the estimate documented?",
            ...             answer="Yes",
            ...         )
            ...     ],
            ...     overall_outcome="Meets",
            ...     outcome_justification="All required documentation is present.",
            ... )
            >>> "TFR Questionnaire" in result.as_questionnaire_string()
            True
        """
        form_instructions = (
            "Please fill out the following TFR questionnaire based on the claim information provided. "
            "For any question marked as 'No', select at least one applicable sub-question that details the specific driver(s) related to the opportunity. "
            "Answer 'Insufficient information' only when there is a clear lack of specifically required information, and only after thorough review of the context."
        )
        output: list[str] = []
        output.append("\nTFR Questionnaire:")
        output.append(form_instructions)

        for question in self.questions:
            question_help_text = f" (help_text: {question.help_text})" if question.help_text else ""
            output.append(f"\n{question.id}: {question.text}{question_help_text}")

            if question.sub_questions:
                output.append("Sub-Questions:")
                for sub_question in question.sub_questions:
                    sub_help_text = (
                        f" (help_text: {sub_question.help_text})" if sub_question.help_text else ""
                    )
                    output.append(f"  {sub_question.id}: {sub_question.text}{sub_help_text}")

        overall_outcome_description = TFRAnalysisResult.model_fields["overall_outcome"].description
        outcome_options = TFRAnalysisResult.model_fields["overall_outcome"].annotation.__dict__[
            "__args__"
        ]
        output.append(
            f"\nOverall Outcome: {overall_outcome_description} Options: {', '.join(outcome_options)}"
        )
        return "\n".join(output)

    def to_a2ui_component(self) -> A2UIComponent:
        """Convert the TFR analysis result to an A2UI component."""
        from presenters.a2ui import tfr_analysis_to_component

        return tfr_analysis_to_component(self.model_dump())
