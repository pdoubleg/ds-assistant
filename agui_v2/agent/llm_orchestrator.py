"""
LLM Orchestrator - Independent sub-agents for document review and audit questions.

Two independent sub-agents that the main agent can call in any order or
combination depending on the user query:

    review_documents:
        Quick statistical review of uploaded documents — produces a simple
        summary with key topics, risk areas, and page counts that the agent
        uses to generate UI components (charts, tables, cards).

    generate_audit_questions:
        Reads raw document text and produces a structured audit question set
        matching the schema:
            Question: { id, question, rating (Yes|No|NA|null), comments, sub_questions? }
            Sub-question: { id, question, rating (Yes|No|NA|null), comments }

Example usage:
    >>> # Either tool can be used independently
    >>> review = await review_documents(doc_payloads)
    >>> questions = await generate_audit_questions(doc_payloads)
"""

from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from model_config import get_orchestrator_model
from prompts import (
    DOCUMENT_ANALYSIS_SYSTEM_PROMPT,
    AUDIT_FORM_SYSTEM_PROMPT,
    format_document_analysis_prompt,
    format_audit_form_prompt,
)


# =============================================================================
# Structured output models
# =============================================================================

class DocumentReviewResult(BaseModel):
    """Lightweight stats from a document review.

    Designed to power UI summary components (TextBox, DataTable, SimpleChart)
    rather than feed into downstream agents.

    Attributes:
        title: Inferred title for the document set.
        document_type: Broad classification (policy, procedure, etc.).
        summary: One-paragraph narrative overview.
        key_topics: Primary compliance/risk topics discovered.
        risk_areas: Specific risk domains identified.
        total_pages: Approximate total pages reviewed.
        doc_count: Number of documents reviewed.
    """
    title: str = "Untitled"
    document_type: str = "general"
    summary: str = ""
    key_topics: list[str] = Field(default_factory=list)
    risk_areas: list[str] = Field(default_factory=list)
    total_pages: int = 0
    doc_count: int = 0


class AuditSubQuestion(BaseModel):
    """A sub-question (driver) for a No rating.

    Attributes:
        id: Unique identifier (e.g. 'AQ-001-a').
        question: Follow-up question text.
        comments: Initially empty; filled by auditor.
        rating: null until the auditor selects Yes, No, or NA.
    """
    id: str
    question: str
    comments: str = ""
    rating: Literal["Yes", "No", "NA"] | None = None


class AuditQuestion(BaseModel):
    """A top-level audit question.

    Attributes:
        id: Unique identifier (e.g. 'AQ-001').
        question: The audit question text.
        rating: null until the auditor selects Yes, No, or NA.
        comments: Optional auditor notes.
        sub_questions: Driver questions shown when rating is No.
    """
    id: str
    question: str
    rating: Literal["Yes", "No", "NA"] | None = None
    comments: str | None = None
    sub_questions: list[AuditSubQuestion] = Field(default_factory=list)


class AuditQuestionSetResult(BaseModel):
    """Structured result from the audit question generation sub-agent.

    Attributes:
        questions: The full set of audit questions with sub-questions.
    """
    questions: list[AuditQuestion]


# =============================================================================
# pydantic-ai sub-agents
# =============================================================================

document_review_agent = Agent(
    model=get_orchestrator_model(),
    output_type=DocumentReviewResult,
    instructions=DOCUMENT_ANALYSIS_SYSTEM_PROMPT,
)

audit_question_agent = Agent(
    model=get_orchestrator_model(),
    output_type=AuditQuestionSetResult,
    instructions=AUDIT_FORM_SYSTEM_PROMPT,
)


# =============================================================================
# Helper: build combined document text from payloads
# =============================================================================

def _combine_documents(document_contents: list[dict[str, Any]]) -> tuple[str, int]:
    """Concatenate document payloads into a single string for LLM input.

    Args:
        document_contents: List of dicts with ``title``, ``content``,
            and optionally ``file_type`` and ``page_count``.

    Returns:
        Tuple of (combined_text, total_pages).
    """
    parts: list[str] = []
    total_pages = 0
    for doc in document_contents:
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        ftype = doc.get("file_type", "unknown")
        pages = doc.get("page_count") or 0
        total_pages += pages
        parts.append(f"--- Document: {title} ({ftype}, {pages} pages) ---\n{content}")
    return "\n\n".join(parts), total_pages


# =============================================================================
# Independent orchestration functions
# =============================================================================

async def review_documents(
    document_contents: list[dict[str, Any]],
) -> DocumentReviewResult:
    """Review documents and return lightweight stats for UI components.

    Produces a simple summary with key topics, risk areas, and counts
    that the agent turns into TextBox / DataTable / SimpleChart components.

    Args:
        document_contents: List of dicts, each with at least:
            - ``title`` (str): Document filename/title.
            - ``content`` (str): Extracted text content.
            Optionally:
            - ``file_type`` (str): pdf, docx, xlsx.
            - ``page_count`` (int): Number of pages.

    Returns:
        Validated ``DocumentReviewResult`` with stats for UI rendering.

    Example:
        >>> result = await review_documents([
        ...     {"title": "Policy.pdf", "content": "Full text...", "page_count": 45}
        ... ])
        >>> print(result.key_topics, result.risk_areas)
    """
    combined_text, total_pages = _combine_documents(document_contents)
    prompt = format_document_analysis_prompt(combined_text)

    print(
        f"[ORCHESTRATOR] Reviewing {len(document_contents)} document(s), "
        f"~{len(combined_text)} chars",
        flush=True,
    )

    try:
        result = await document_review_agent.run(prompt)
        review = result.output
        # Patch in metadata the LLM may have missed
        if total_pages and not review.total_pages:
            review.total_pages = total_pages
        if not review.doc_count:
            review.doc_count = len(document_contents)
        print(
            f"[ORCHESTRATOR] Review complete: type={review.document_type}, "
            f"topics={len(review.key_topics)}, risks={len(review.risk_areas)}",
            flush=True,
        )
        return review
    except Exception as exc:
        print(f"[ORCHESTRATOR ERROR] Document review failed: {exc}", flush=True)
        import traceback
        traceback.print_exc()

        return DocumentReviewResult(
            title=(
                document_contents[0].get("title", "Untitled")
                if document_contents
                else "Untitled"
            ),
            document_type="general",
            summary="Document review failed; generated a minimal fallback.",
            key_topics=["General compliance"],
            risk_areas=["Unclassified risk"],
            total_pages=total_pages,
            doc_count=len(document_contents),
        )


async def generate_audit_questions(
    document_contents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate an audit question set directly from document text.

    Reads raw document content (not a prior review) and produces structured
    audit questions matching the frontend schema.

    Args:
        document_contents: Same shape as ``review_documents`` — list of dicts
            with ``title``, ``content``, etc.

    Returns:
        List of question dicts ready for ``generate_audit_question_form()``.

    Example:
        >>> questions = await generate_audit_questions([
        ...     {"title": "Policy.pdf", "content": "Full text..."}
        ... ])
        >>> print(questions[0]["id"], questions[0]["question"])
    """
    combined_text, _ = _combine_documents(document_contents)
    prompt = format_audit_form_prompt(combined_text)

    print("[ORCHESTRATOR] Generating audit questions from documents...", flush=True)

    try:
        result = await audit_question_agent.run(prompt)
        question_set = result.output
        questions = [q.model_dump() for q in question_set.questions]
        print(f"[ORCHESTRATOR] Generated {len(questions)} questions", flush=True)
        return questions
    except Exception as exc:
        print(f"[ORCHESTRATOR ERROR] Question generation failed: {exc}", flush=True)
        import traceback
        traceback.print_exc()

        return [
            {
                "id": "AQ-001",
                "question": "Could not generate questions — please retry or refine your documents.",
                "rating": None,
                "comments": "",
                "sub_questions": [],
            }
        ]
