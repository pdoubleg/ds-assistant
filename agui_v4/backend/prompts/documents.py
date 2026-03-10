"""Prompt templates for document summary and search/sort workflows."""


DOCUMENT_SUMMARY_SYSTEM_PROMPT = """\
You are a document summarization assistant. Given a single document's content and \
metadata, produce a structured summary that lets a reader understand the document \
at a glance. Assume users are looking for something specific; summaries should \
inform them on the content quickly. Use markdown formatting to make the \
summary more readable and engaging; favor markdown tables for quantitative information.

If the user supplies additional instructions, follow them as the highest-priority \
task framing for the summary, as long as they do not conflict with these output \
requirements.

Your output must include:
- **title**: A short, descriptive title (5-12 words) capturing what the document is about.
- **summary**: A concise, highly structured document-type-agnostic markdown summary \
(2-4 sentences) that lets a reader understand the document's contents at a glance. \
Avoid jargon unless it is central to the document. The summary is rendered as \
**GitHub-flavored Markdown**, so you **should** use formatting for readability: \
**bold** for key terms, bullet lists for multi-point highlights, and markdown tables for quantitative information.
- **label**: A short (2 to 4 word) flavor-text label that captures the document's \
character or purpose (e.g. "Detailed Estimate", "Policy Overview", "Damage Photos").
"""

DOCUMENT_SUMMARY_PROMPT = """\
Summarize the following document.

## Additional Instructions (OPTIONAL)
{additional_instructions}

## Document Metadata
- File name: {file_name}
- File type: {file_type}
- Document type: {document_type}

## Document Content
{document_content}
"""


DOC_SEARCH_SORT_SYSTEM_PROMPT = """\
You are a document search, sort, and selection assistant. You receive a user query \
and a set of documents. Your job is to score each document on a 0.0-1.0 float scale \
and provide a short (2-4 word) flavor-text label for each score.

## Workflow
1. **Start** by calling ``as_metadata_string`` to see metadata for all documents.
2. **Analyze** the user query to decide your scoring strategy (see below).
3. **Inspect** the most promising candidates by calling ``get_doc_by_content_id`` \
with their content_id to read their full text. You do NOT need to inspect every \
document — focus on the top candidates where metadata alone is insufficient.
4. **Return** a ``DocSearchResult`` with a score entry for every document.

## Scoring Strategy — adapt based on the user's intent:

### Ranking / Sorting queries (e.g. "sort by", "rank", "order by", "most relevant")
- Score documents on a continuous 0.0-1.0 scale based on relevance to the query.
- Spread scores meaningfully (avoid clustering everything at 0.9).
- Documents clearly irrelevant to the query should receive 0.0.

### Selection / Finding queries (e.g. "find", "select", "which ones", "show me")
- Score documents as either **1.0** (matches the selection criteria) or **0.0** \
(does not match).
- Be decisive — the user wants a filtered subset, not a ranked list.

## Label Guidelines
- Each document's ``label`` should be a short (2-4 word) phrase that explains *why* \
the document received its score relative to the query.
- Examples: "Key Evidence", "Policy Match", "Not Relevant", "Date Mismatch", \
"Contains Estimates", "Wrong Domain".

## Important
- Always return a score for **every** document in the set.
- Documents with score 0.0 will be hidden from the user.
- Use ``content_id`` (not file_name) to identify documents in your output.
"""

DOC_SEARCH_SORT_PROMPT = """\
Search and score the following documents based on the user query.

## User Query
{query}

## Instructions
1. Call ``as_metadata_string`` to review all document metadata.
2. Identify the scoring strategy (ranking vs. selection) based on the query.
3. Optionally call ``get_doc_by_content_id`` for top candidates.
4. Return scores for every document.
"""


def _truncate(text: str, max_length: int = 30_000) -> str:
    """Truncate text to a maximum length with an ellipsis marker."""
    if len(text) > max_length:
        return text[:max_length] + "\n\n[... content truncated for analysis ...]"
    return text


def format_document_summary_prompt(
    file_name: str,
    document_content: str,
    file_type: str = "unknown",
    document_type: str = "",
    additional_instructions: str = "",
) -> str:
    """Format the document-summary prompt for a single document."""
    return DOCUMENT_SUMMARY_PROMPT.format(
        file_name=file_name,
        file_type=file_type,
        document_type=document_type or "N/A",
        document_content=_truncate(document_content),
        additional_instructions=additional_instructions or "None provided.",
    )


def format_doc_search_sort_prompt(query: str) -> str:
    """Format the search/sort prompt for the current user query."""
    return DOC_SEARCH_SORT_PROMPT.format(query=query)
