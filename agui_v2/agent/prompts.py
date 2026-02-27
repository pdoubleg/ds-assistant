"""
Prompt Templates — System prompts and user-prompt formatters for pydantic-ai agents.

Each sub-agent has:
  - A *system prompt* constant (``*_SYSTEM_PROMPT``) providing role context.
  - A *user prompt* template (``*_PROMPT``) formatted at call-time.

pydantic-ai handles output formatting via the agent's ``output_type`` Pydantic
model, so prompts focus on analytical guidance rather than JSON structure.
"""

# ============================================================================
# CLAIM ANALYSIS (run_analysis → A2UI components)
# ============================================================================

ANALYSIS_SYSTEM_PROMPT = (
    "You are an insurance claim analyst. You review property claim documents "
    "(estimates, system notes, policy details, vendor reports, scope sheets) "
    "and produce structured analysis that helps auditors understand the claim "
    "before beginning their review.\n\n"
    "Your output maps directly to UI components. When real documents are "
    "provided, only populate sections supported by the evidence — be factual, "
    "cite evidence, and flag anything an auditor should investigate.\n\n"
    "When NO documents are provided and the user asks for an example, demo, "
    "or sample analysis, generate realistic but fictional insurance claim data "
    "that showcases every component type. Use plausible names, dollar amounts, "
    "dates, line items, and findings so the output looks like a real claim "
    "review. This is useful for testing and demonstrations."
)

ANALYSIS_PROMPT = """Analyze the following claim context and produce structured output.

## Focus
{focus}

## Documents
{document_content}

## Component Catalog

Your output fields map to UI components. Populate each section **only** when
the documents provide enough data. Leave a section as null when irrelevant.

### claim_overview (required → TextBox)
A 2-4 sentence narrative overview of the claim: what happened, key parties,
current status. Always provide this.

### timeline_events (optional → ClaimTimeline)
Chronological events in the claim lifecycle. Populate when documents contain
dated activities. Each event:
- **date** (str): Display date, e.g. "2025-03-15".
- **title** (str): Short headline.
- **description** (str): 1-2 sentence detail.
- **category**: One of "inspection", "estimate", "payment", "correspondence", "other".
- **status**: One of "completed", "pending", "flagged".

### summary_metrics (optional → SummaryCard)
Key claim metrics at a glance. Populate when documents contain quantifiable
data (dollar amounts, dates, counts, statuses). Each metric:
- **label** (str): Metric name (e.g. "Total Estimate", "Deductible").
- **value** (str): Formatted display value (e.g. "$12,450.00", "Open").
- **icon** (str | null): Optional hint — "dollar", "calendar", "user", "shield", "file", "alert".
- **trend** ("up" | "down" | "stable" | null): Optional directional indicator.

### findings (optional → FindingCard per item)
Observations, flags, or insights an auditor should know. Populate when you
identify patterns, gaps, risks, or notable details. Each finding:
- **title** (str): Short headline.
- **content** (str): Detailed explanation (markdown supported).
- **severity**: "info" (neutral observation), "warning" (needs attention), "critical" (major concern).
- **category** (str | null): Optional tag — "timeline", "coverage", "estimate", "resolution", "documentation".

### tables (optional → DataTable per item)
Structured tabular data (estimate line items, coverage breakdowns, payment
history). Populate when documents contain structured data worth tabulating.
Each table:
- **caption** (str): Table heading.
- **headers** (list[str]): Column labels.
- **rows** (list[list]): 2-D array of cell values (strings or numbers).

### charts (optional → SimpleChart per item)
Simple visualizations for numeric data. Populate when a visual breakdown
adds clarity. Each chart:
- **chart_type**: "bar", "line", or "pie".
- **title** (str): Chart heading.
- **labels** (list[str]): Data point labels.
- **values** (list[number]): Corresponding numeric values.
- **colors** (list[str] | null): Optional hex colors per data point.

## Guidelines
- **Real documents present**: Extract facts directly from the documents; do not
  fabricate data. Cite evidence where possible.
- **No documents / example requested**: Generate realistic fictional claim data
  that exercises all component types. Invent plausible claim numbers, parties,
  dollar amounts, dates, line items, timeline events, and findings. Make the
  output look like a genuine property claim review suitable for demos.
- Dollar amounts should use the format "$X,XXX.XX".
- Dates should use ISO format or a readable format like "March 15, 2025".
- For findings, prefer actionable observations over generic statements.
- If the user's focus mentions specific topics (e.g. "timeline", "damaged items"),
  prioritize those sections but still populate other relevant sections.
- When generating examples, populate **every** section (timeline, metrics,
  findings, at least one table, and at least one chart) to fully showcase the UI."""


# ============================================================================
# TFR AUDIT FORM GENERATION (works from raw document text — unchanged)
# ============================================================================

AUDIT_FORM_SYSTEM_PROMPT = (
    "You are an expert Technical File Review (TFR) auditor. Generate a comprehensive "
    "TFR analysis from claim documents. You must determine the peril (Interior or "
    "Exterior), evaluate each TFR question, provide sub-questions with detailed "
    "reasoning and citations for any 'No' answers, and produce an overall outcome "
    "assessment. Pre-populate all answers, reasoning, and citations based on the "
    "document evidence — the reviewer will refine your analysis."
)

AUDIT_FORM_PROMPT = """Generate a TFR (Technical File Review) analysis based on the document content below.

## Document Content

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _truncate(text: str, max_length: int = 30_000) -> str:
    """Truncate text to *max_length* characters with an ellipsis marker.

    Args:
        text: Raw text to truncate.
        max_length: Maximum allowed character count.

    Returns:
        The original text if within limits, otherwise truncated with a
        trailing marker.
    """
    if len(text) > max_length:
        return text[:max_length] + "\n\n[... content truncated for analysis ...]"
    return text


def format_analysis_prompt(
    document_content: str | None,
    focus: str = "General claim review",
) -> str:
    """Format the analysis user prompt for the ``run_analysis`` sub-agent.

    Args:
        document_content: Combined raw text from selected/uploaded documents,
            or *None* when no documents are available.
        focus: User-supplied focus area (e.g. "timeline and damaged items").

    Returns:
        Formatted prompt string for the analysis agent.

    Example:
        >>> prompt = format_analysis_prompt("Estimate text ...", focus="damaged items")
        >>> assert "damaged items" in prompt
    """
    doc_text = _truncate(document_content) if document_content else "(No documents provided.)"
    return ANALYSIS_PROMPT.format(document_content=doc_text, focus=focus)


def format_audit_form_prompt(document_content: str) -> str:
    """Format the TFR audit form generation prompt with raw document text.

    Args:
        document_content: Combined raw document text.

    Returns:
        Formatted prompt string for the TFR audit question agent.

    Example:
        >>> prompt = format_audit_form_prompt("Estimate text here ...")
        >>> assert "Estimate text" in prompt
    """
    return AUDIT_FORM_PROMPT.format(document_content=_truncate(document_content))
