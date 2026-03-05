"""LLM orchestration for claim analysis and TFR question generation."""

from typing import Any

from pydantic_ai import Agent

from model_config import get_orchestrator_model
from models import (
    AnalysisResult,
    TFRAnalysisResult,
    TimelineEvents,
    SummaryMetrics,
    Finding,
    TableSpec,
    ChartSpec,
    DocumentSummary,
)
from prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    AUDIT_FORM_SYSTEM_PROMPT,
    COMPONENT_SYSTEM_PROMPT,
    format_analysis_prompt,
    format_audit_form_prompt,
    format_component_prompt,
    TIMELINE_EVENT_SYSTEM_PROMPT,
    SUMMARY_METRICS_SYSTEM_PROMPT,
    FINDING_SYSTEM_PROMPT,
    TABLE_SYSTEM_PROMPT,
    CHART_SYSTEM_PROMPT,
    DOCUMENT_SUMMARY_SYSTEM_PROMPT,
)


# =============================================================================
# pydantic-ai sub-agents
# =============================================================================

# Step 1: Produces plain-text context brief from documents/focus
analysis_agent = Agent(
    model=get_orchestrator_model(),
    output_type=str,
    instructions=ANALYSIS_SYSTEM_PROMPT,
)

# Step 2: Produces structured AnalysisResult that maps to A2UI components
component_agent = Agent(
    model=get_orchestrator_model(),
    output_type=AnalysisResult,
    instructions=COMPONENT_SYSTEM_PROMPT,
)

# Produces structured TFRAnalysisResult for the audit questionnaire
audit_question_agent = Agent(
    model=get_orchestrator_model(),
    output_type=TFRAnalysisResult,
    instructions=AUDIT_FORM_SYSTEM_PROMPT,
)


timeline_event_agent = Agent(
    model=get_orchestrator_model(),
    output_type=TimelineEvents,
    instructions=TIMELINE_EVENT_SYSTEM_PROMPT,
)

summary_metrics_agent = Agent(
    model=get_orchestrator_model(),
    output_type=SummaryMetrics,
    instructions=SUMMARY_METRICS_SYSTEM_PROMPT,
)

findings_agent = Agent(
    model=get_orchestrator_model(),
    output_type=list[Finding],
    instructions=FINDING_SYSTEM_PROMPT,
)

tables_agent = Agent(
    model=get_orchestrator_model(),
    output_type=TableSpec,
    instructions=TABLE_SYSTEM_PROMPT,
)

charts_agent = Agent(
    model=get_orchestrator_model(),
    output_type=ChartSpec,
    instructions=CHART_SYSTEM_PROMPT,
)

# Summarizes a single document and assigns a relevance/importance rank.
document_summary_agent = Agent(
    model="gpt-4.1-mini",
    output_type=DocumentSummary,
    instructions=DOCUMENT_SUMMARY_SYSTEM_PROMPT,
)

# =============================================================================
# Helper: build combined document text from payloads
# =============================================================================


def _combine_documents(document_contents: list[dict[str, Any]]) -> tuple[str, int]:
    """Concatenate document payloads into a single string for LLM input.

    Args:
        document_contents: List of dicts with ``title``, ``content``,
            and optionally ``file_type``/``mime_type`` and ``document_type``.

    Returns:
        Tuple of (combined_text, doc_count).
    """
    parts: list[str] = []
    for doc in document_contents:
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        ftype = doc.get("file_type", doc.get("mime_type", "unknown"))
        doc_type = doc.get("document_type", "")
        header = f"--- Document: {title} ({ftype}"
        if doc_type:
            header += f", type={doc_type}"
        header += f") ---\n{content}"
        parts.append(header)
    return "\n\n".join(parts), len(document_contents)


# =============================================================================
# Independent orchestration functions
# =============================================================================


async def run_summary_analysis(
    document_contents: list[dict[str, Any]] | None,
    focus: str = "General claim review",
) -> AnalysisResult:
    """Run two-step claim analysis and return structured component data.

    This function first creates a context brief (analysis agent) and then uses
    that brief to generate structured UI components (component agent).

    Args:
        document_contents: List of dicts, each with at least ``title`` and
            ``content``. Can be *None* or empty when no documents are available.
        focus: User-supplied focus area (e.g. "timeline and damaged items").

    Returns:
        Validated ``AnalysisResult`` with populated sections.

    Example:
        >>> result = await run_analysis(
        ...     [{"title": "Estimate.pdf", "content": "Full text..."}],
        ...     focus="timeline and damaged items",
        ... )
        >>> print(result.claim_overview)
    """
    # Build combined document text (or None when no docs)
    combined_text: str | None = None
    doc_count = 0
    if document_contents:
        combined_text, doc_count = _combine_documents(document_contents)

    analysis_prompt = format_analysis_prompt(combined_text, focus=focus)

    print(
        f"[ORCHESTRATOR] Step 1/2 context analysis on {doc_count} document(s), "
        f"focus='{focus}', ~{len(combined_text or '')} chars",
        flush=True,
    )

    try:
        # Step 1: produce a context brief to separate reasoning from structure shaping.
        context_result = await analysis_agent.run(analysis_prompt)
        context_brief = context_result.output

        print(
            f"[ORCHESTRATOR] Context brief generated (~{len(context_brief)} chars). "
            "Step 2/2 structured component generation...",
            flush=True,
        )

        # Step 2: convert context brief into typed AnalysisResult.
        component_prompt = format_component_prompt(context_brief, focus=focus)
        structured_result = await component_agent.run(component_prompt)
        analysis = structured_result.output

        return analysis
    except Exception as exc:
        print(f"[ORCHESTRATOR ERROR] Analysis failed: {exc}", flush=True)
        import traceback

        traceback.print_exc()

        # Fallback with a minimal valid result
        return AnalysisResult(
            claim_overview="Analysis could not be completed. Please retry or refine your documents.",
        )


async def generate_audit_questions(
    document_contents: list[dict[str, Any]],
    additional_instructions: str = "",
) -> dict[str, Any]:
    """Generate a TFR analysis result directly from document text.

    Reads raw document content and produces the full TFR analysis including
    peril determination, questions with sub-questions, and overall outcome.

    Args:
        document_contents: List of dicts with ``title``, ``content``, etc.
        additional_instructions: User-supplied additional instructions for the audit form generation.
            (e.g. "timeline and damaged items").
    Returns:
        Dict with TFR result fields: ``peril``, ``questions``,
        ``overall_outcome``, ``outcome_justification``,
        ``additional_analysis``, ``follow_ups``.

    Example:
        >>> tfr = await generate_audit_questions([
        ...     {"title": "Estimate.pdf", "content": "Full text..."}
        ... ])
        >>> print(tfr["peril"], tfr["overall_outcome"])
    """
    combined_text, _ = _combine_documents(document_contents)
    prompt = format_audit_form_prompt(
        combined_text, additional_instructions=additional_instructions
    )

    print("[ORCHESTRATOR] Generating TFR audit questions from documents...", flush=True)

    try:
        result = await audit_question_agent.run(prompt)
        tfr = result.output
        tfr_dict = tfr.model_dump()
        print(
            f"[ORCHESTRATOR] Generated {len(tfr.questions)} TFR questions, "
            f"peril={tfr.peril.peril}, outcome={tfr.overall_outcome}",
            flush=True,
        )
        return tfr_dict
    except Exception as exc:
        print(f"[ORCHESTRATOR ERROR] TFR question generation failed: {exc}", flush=True)
        import traceback

        traceback.print_exc()

        # Fallback with minimal valid TFR structure
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
