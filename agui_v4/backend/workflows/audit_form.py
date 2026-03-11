"""Audit-form generation workflows."""

from typing import Any

from prompts.audit_form import format_audit_form_prompt
from services.agent_helpers import NullStatusReporter, StatusReporter
from services.document_mapper import DocumentMapper
from workflows.agent_factory import audit_question_agent


async def generate_audit_questions(
    document_contents: list[dict[str, Any]],
    additional_instructions: str = "",
    mapper: DocumentMapper | None = None,
    reporter: StatusReporter | None = None,
) -> dict[str, Any]:
    """Generate a TFR analysis payload directly from document text.

    Args:
        document_contents: Lightweight prompt payloads.
        additional_instructions: Optional user instructions to guide the review.
        mapper: Optional shared document mapper.
        reporter: Optional nested progress reporter for UI-friendly status rows.

    Returns:
        Canonical TFR payload dictionary.
    """
    reporter = reporter or NullStatusReporter()
    mapper = mapper or DocumentMapper()

    reporter.in_progress("Combining uploaded documents...", progress=5)
    combined_text, _ = mapper.combine_documents(document_contents)
    reporter.completed("Combining uploaded documents...", progress=15)

    reporter.in_progress("Formatting audit questionnaire prompt...", progress=20)
    prompt = format_audit_form_prompt(
        combined_text,
        additional_instructions=additional_instructions,
    )
    reporter.completed("Formatting audit questionnaire prompt...", progress=35)

    print("[ORCHESTRATOR] Generating TFR audit questions from documents...", flush=True)
    active_step_message = "Running audit question agent..."
    try:
        reporter.in_progress(active_step_message, progress=40)
        result = await audit_question_agent.run(prompt)
        reporter.completed(active_step_message, progress=90)

        active_step_message = "Transforming audit question result..."
        reporter.in_progress(active_step_message, progress=92)
        tfr = result.output
        tfr_dict = tfr.model_dump()
        reporter.completed(active_step_message, progress=95)
        print(
            f"[ORCHESTRATOR] Generated {len(tfr.questions)} TFR questions, "
            f"peril={tfr.peril.peril}, outcome={tfr.overall_outcome}",
            flush=True,
        )
        return tfr_dict
    except Exception as exc:
        reporter.error(active_step_message, progress=90)
        print(f"[ORCHESTRATOR ERROR] TFR question generation failed: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return {
            "peril": {"peril": "Exterior", "notes": "Fallback — generation failed."},
            "questions": [
                {
                    "id": "Q1",
                    "text": "Could not generate questions — please retry or refine your documents.",
                    "answer": "Insufficient information",
                    "sub_questions": None,
                    "missing_info": "TFR question generation failed. Please retry.",
                }
            ],
            "overall_outcome": "Does Not Meet Expectations",
            "outcome_justification": "Analysis could not be completed due to an error.",
            "additional_analysis": None,
            "follow_ups": None,
        }
