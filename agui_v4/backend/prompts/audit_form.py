"""Prompt templates for TFR audit-form generation."""

AUDIT_FORM_SYSTEM_PROMPT = (
    "You are an expert Targeted File Review (TFR) auditor. Generate a comprehensive "
    "TFR analysis from optional claim documents. You must determine the peril (Interior or "
    "Exterior), evaluate each TFR question, provide sub-questions with detailed "
    "reasoning and citations for any 'No' answers, and produce an overall outcome "
    "assessment. Pre-populate all answers, reasoning, and citations based on the "
    "document evidence — the reviewer will refine your analysis. "
    "Users may specify a focus area for the review, which you should use to generate the questions. "
    "The focus may also be used when users are not providing documents and want to you to generate an example or demo of realistic fictional data."
)

AUDIT_FORM_PROMPT = """Generate a **Targeted File Review (TFR)** analysis based on the optional document content below, and \
optional additional instructions if provided.

## Additional Instructions (OPTIONAL)
{additional_instructions}

## Document Content (OPTIONAL) - If provided, use this to generate the TFR analysis.

{document_content}

## TFR Analysis Requirements

### Peril Determination
Determine whether this is an **Interior** or **Exterior** peril based on the claim documents.
Include optional notes if the peril classification is ambiguous.

### Question Structure
Each TFR question must have:
- **id**: Unique identifier (e.g., "Q1", "Q2")
- **text**: Clear, specific TFR question answerable with Yes, No, or Insufficient information
- **answer**: Your determination — "Yes", "No", or "Insufficient information"
  - **Yes**: The requirement is met / no issues found
  - **No**: An opportunity for improvement was identified (must have sub-questions)
  - **Insufficient information**: Cannot determine from available documents (must specify missing_info)
- **sub_questions**: Required when answer is "No". Each sub-question has:
  - **id**: Unique identifier (e.g., "Q1.1", "Q1.2")
  - **text**: Specific sub-question identifying the driver / opportunity
  - **reasoning**: Your detailed explanation of why this is an opportunity, based on evidence
  - **citations**: Specific references to the documents supporting your reasoning
- **missing_info**: Required when answer is "Insufficient information" — describe what information is needed

### Overall Assessment
- **overall_outcome**: "Meets" or "Does Not Meet Expectations"
- **outcome_justification**: Concise justification synthesizing all question findings
- **additional_analysis**: Optional Wind/Hail analysis (Exterior) or Flooring/Cabinetry analysis (Interior)
- **follow_ups**: Optional recommended follow-up actions

### Generation Rules
1. Generate 5-15 TFR questions depending on document complexity and peril type
2. Questions marked "No" must have 1-5 sub-questions with reasoning and citations
3. Pre-populate ALL answers based on your analysis of the documents
4. Sub-question reasoning should cite specific evidence from the documents
5. Citations should reference specific sections, pages, or content from the documents
6. Be thorough but concise in reasoning — reviewers will refine your analysis"""


def _truncate(text: str, max_length: int = 30_000) -> str:
    """Truncate text to a maximum length with an ellipsis marker."""
    if len(text) > max_length:
        return text[:max_length] + "\n\n[... content truncated for analysis ...]"
    return text


def format_audit_form_prompt(document_content: str, additional_instructions: str = "") -> str:
    """Format the audit-form generation prompt."""
    return AUDIT_FORM_PROMPT.format(
        document_content=_truncate(document_content),
        additional_instructions=additional_instructions,
    )
