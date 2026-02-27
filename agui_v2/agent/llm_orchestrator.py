"""
LLM Orchestrator - Independent sub-agents for claim analysis and TFR audit questions.

Two independent sub-agents that the main agent can call in any order or
combination depending on the user query:

    run_analysis:
        Analyzes claim documents (estimates, notes, policy details, vendor
        reports) and produces an ``AnalysisResult`` with optional sections
        that map directly to A2UI components (timeline, summary metrics,
        findings, tables, charts).

    generate_audit_questions:
        Reads raw document text and produces a structured TFR analysis
        matching the ``TFRAnalysisResult`` schema.

Example usage:
    >>> analysis = await run_analysis(doc_payloads, focus="timeline and damaged items")
    >>> tfr_result = await generate_audit_questions(doc_payloads)
"""

from typing import Any

from pydantic_ai import Agent

from model_config import get_orchestrator_model
from models import AnalysisResult, TFRAnalysisResult
from prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    AUDIT_FORM_SYSTEM_PROMPT,
    format_analysis_prompt,
    format_audit_form_prompt,
)


# =============================================================================
# pydantic-ai sub-agents
# =============================================================================

# Produces structured AnalysisResult that maps to A2UI components
analysis_agent = Agent(
    model=get_orchestrator_model(),
    output_type=AnalysisResult,
    instructions=ANALYSIS_SYSTEM_PROMPT,
)

# Produces structured TFRAnalysisResult for the audit questionnaire
audit_question_agent = Agent(
    model=get_orchestrator_model(),
    output_type=TFRAnalysisResult,
    instructions=AUDIT_FORM_SYSTEM_PROMPT,
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

async def run_analysis(
    document_contents: list[dict[str, Any]] | None,
    focus: str = "General claim review",
) -> AnalysisResult:
    """Analyze claim documents and return structured results for UI components.

    Calls the analysis sub-agent which returns an ``AnalysisResult`` with
    optional sections (timeline, metrics, findings, tables, charts). The
    caller maps non-None sections to A2UI components.

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

    prompt = format_analysis_prompt(combined_text, focus=focus)

    print(
        f"[ORCHESTRATOR] Running analysis on {doc_count} document(s), "
        f"focus='{focus}', ~{len(combined_text or '')} chars",
        flush=True,
    )

    try:
        result = await analysis_agent.run(prompt)
        analysis = result.output
        # Summarise what was populated for logging
        sections = []
        if analysis.timeline_events:
            sections.append(f"{len(analysis.timeline_events)} timeline events")
        if analysis.summary_metrics:
            sections.append(f"{len(analysis.summary_metrics)} metrics")
        if analysis.findings:
            sections.append(f"{len(analysis.findings)} findings")
        if analysis.tables:
            sections.append(f"{len(analysis.tables)} tables")
        if analysis.charts:
            sections.append(f"{len(analysis.charts)} charts")
        print(
            f"[ORCHESTRATOR] Analysis complete: {', '.join(sections) or 'overview only'}",
            flush=True,
        )
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
) -> dict[str, Any]:
    """Generate a TFR analysis result directly from document text.

    Reads raw document content and produces the full TFR analysis including
    peril determination, questions with sub-questions, and overall outcome.

    Args:
        document_contents: List of dicts with ``title``, ``content``, etc.

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
    prompt = format_audit_form_prompt(combined_text)

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
