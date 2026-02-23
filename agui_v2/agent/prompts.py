"""
Prompt Templates — System prompts and user-prompt formatters for pydantic-ai agents.

Each sub-agent has:
  - A *system prompt* constant (``*_SYSTEM_PROMPT``) providing role context.
  - A *user prompt* template (``*_PROMPT``) formatted at call-time.

pydantic-ai handles output formatting via the agent's ``output_type`` Pydantic
model, so prompts focus on analytical guidance rather than JSON structure.
"""

# ============================================================================
# DOCUMENT REVIEW (stats / summaries for UI components)
# ============================================================================

DOCUMENT_ANALYSIS_SYSTEM_PROMPT = (
    "You are a document analyst. Given uploaded documents, produce a concise "
    "statistical summary: classify the document type, list the main topics, "
    "identify risk areas, and write a short narrative overview. Keep it brief "
    "and factual — your output will power summary cards and charts, not a "
    "downstream agent."
)

DOCUMENT_ANALYSIS_PROMPT = """Summarize the following document content.

## Documents

{document_content}

## What to produce

1. **title** – A short title for the document set.
2. **document_type** – One word classification (policy, procedure, regulation, report, contract, technical, general).
3. **summary** – One paragraph narrative overview.
4. **key_topics** – List of primary topics / compliance areas found.
5. **risk_areas** – List of specific risk domains identified.
6. **total_pages** – Approximate page count from the input.
7. **doc_count** – Number of documents provided.

Be concise. Focus on facts, not interpretation."""


# ============================================================================
# AUDIT FORM GENERATION (works from raw document text)
# ============================================================================

AUDIT_FORM_SYSTEM_PROMPT = (
    "You are an expert audit questionnaire architect. Generate comprehensive "
    "audit questions with optional sub-questions (drivers) based on document "
    "content. Questions are rated Yes/No/NA. Sub-questions capture the "
    "drivers behind a 'No' rating."
)

AUDIT_FORM_PROMPT = """Generate an audit questionnaire based on the document content below.

## Document Content

{document_content}

## Questionnaire Requirements

### Question Structure
Each top-level question must have:
- **id**: Unique identifier (e.g., "AQ-001")
- **question**: Clear, specific audit question answerable with Yes, No, or NA
- **rating**: null (to be filled by the auditor as "Yes", "No", or "NA")
- **comments**: Empty string (optional auditor notes)
- **sub_questions**: Optional list of driver/follow-up questions. Each with:
  - **id**: Unique identifier (e.g., "AQ-001-a")
  - **question**: Specific follow-up question explaining a potential deficiency
  - **comments**: Empty string (auditor fills when parent is rated "No")

### Rating Logic
- Users rate each top-level question **Yes**, **No**, or **NA**
- When a question is rated **No**, at least one sub-question (driver) is required
- Sub-questions do NOT have ratings — they capture the reasoning behind a "No"

### Generation Rules
1. Generate 5-15 main questions depending on document complexity
2. Each main question should have 2-5 sub-questions (drivers)
3. Questions should cover all major document topics
4. Sub-questions should probe specific deficiencies or gaps
5. Questions should be answerable with evidence (documents, interviews, observations)"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_document_analysis_prompt(document_content: str) -> str:
    """Format the document analysis user prompt with actual document content.

    Args:
        document_content: Raw text content extracted from the uploaded documents.

    Returns:
        Formatted prompt string for the document review agent.

    Example:
        >>> prompt = format_document_analysis_prompt("Section 1: Overview ...")
        >>> assert "Section 1" in prompt
    """
    max_length = 30_000
    if len(document_content) > max_length:
        truncated = document_content[:max_length] + "\n\n[... content truncated for analysis ...]"
    else:
        truncated = document_content

    return DOCUMENT_ANALYSIS_PROMPT.format(document_content=truncated)


def format_audit_form_prompt(document_content: str) -> str:
    """Format the audit form generation prompt with raw document text.

    Args:
        document_content: Combined raw document text (same string that
            ``format_document_analysis_prompt`` would receive).

    Returns:
        Formatted prompt string for the audit question agent.

    Example:
        >>> prompt = format_audit_form_prompt("Policy text here ...")
        >>> assert "Policy text" in prompt
    """
    max_length = 30_000
    if len(document_content) > max_length:
        truncated = document_content[:max_length] + "\n\n[... content truncated for analysis ...]"
    else:
        truncated = document_content

    return AUDIT_FORM_PROMPT.format(document_content=truncated)
