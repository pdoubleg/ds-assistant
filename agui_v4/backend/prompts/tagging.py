"""Prompt templates for batch document tagging."""

from typing import Sequence

from models.tagging import CUSTOM_FALLBACK_TAG_LABEL


BATCH_TAGGER_ICON_GUIDE: dict[str, str] = {
    "general": "Broad fallback for ambiguous, mixed, or user-defined tags when nothing else fits safely.",
    "insured": "Use for documents centered on the insured or policyholder.",
    "contractor": "Use for contractor-authored or contractor-focused content.",
    "agent": "Use for insurance agent communications or agency-related materials.",
    "vendor": "Use for vendor, supplier, or service-provider documents.",
    "attorney": "Use for attorney-authored, attorney-facing, or legal-party content.",
    "contact_status": "Use for calls, status updates, routine correspondence, or communication logs.",
    "estimate": "Use for estimates, scoped pricing, or cost breakdowns.",
    "supplement": "Use for supplements, revisions, or additional estimate requests.",
    "demand": "Use for demands, requests for payment, or formal ask letters.",
    "dwelling": "Use for structure, dwelling, or building-related materials.",
    "contents": "Use for personal property, inventory, or contents-related materials.",
    "ale": "Use for additional living expense or temporary housing content.",
    "ems": "Use for emergency mitigation services or rapid response remediation work.",
    "photos": "Use for photos, image sets, or visual documentation.",
    "damage_report": "Use for inspections, assessments, or narrative damage findings.",
    "weather_report": "Use for weather data, storm reports, or meteorological evidence.",
    "attorney_demand": "Use for strong legal-demand signals or escalation threats.",
    "time_sensitive": "Use for deadlines, urgency, or required rapid follow-up.",
    "compliance_issue": "Use for compliance, documentation gaps, or procedural concerns.",
    "customer_complaint": "Use for dissatisfaction, complaints, or service concerns from the customer.",
}

BATCH_TAGGER_SYSTEM_PROMPT = """\
You are a document tagging assistant for insurance claim files.
"""

BATCH_TAGGER_PROMPT = """\
Tag each document in this batch using only the active tag vocabulary.

## Tag Mode
{tag_mode}

## Active Tags
{active_tags_block}

{custom_tags_block}

## Documents
{documents_block}
"""


def _truncate(text: str, max_length: int = 10_000) -> str:
    """Truncate text to a maximum length with an ellipsis marker."""
    if len(text) > max_length:
        return text[:max_length] + "\n\n[... content truncated for analysis ...]"
    return text


def build_batch_tagger_instructions(active_tags: Sequence[str]) -> str:
    """Build runtime system instructions for the batch tagger."""
    active_tag_lines = "\n".join(f"- {tag}" for tag in active_tags)
    icon_lines = "\n".join(
        f"- {icon}: {description}" for icon, description in BATCH_TAGGER_ICON_GUIDE.items()
    )
    has_fallback_tag = CUSTOM_FALLBACK_TAG_LABEL in active_tags
    fallback_guidance = (
        "\n## Fallback Tag\n"
        f"- `{CUSTOM_FALLBACK_TAG_LABEL}` is a special fallback label for this run.\n"
        f"- Use `{CUSTOM_FALLBACK_TAG_LABEL}` only when none of the other active tags apply.\n"
        f"- Never return `{CUSTOM_FALLBACK_TAG_LABEL}` alongside any other tag for the same document.\n"
        f"- When using `{CUSTOM_FALLBACK_TAG_LABEL}`, pair it with the `general` icon."
        if has_fallback_tag
        else ""
    )

    return f"""{BATCH_TAGGER_SYSTEM_PROMPT}

Assign **up to 4** tags to each document from the active runtime vocabulary.
For every selected tag, also choose the best matching icon from the allowed icon catalog.

## Active Tag Vocabulary
{active_tag_lines}

## Allowed Icons
{icon_lines}
{fallback_guidance}

## Rules
- Select **1 to 4** tags per document.
- Use the **exact** tag labels shown in the active vocabulary for this run.
- Do NOT invent new labels or icons.
- Choose the most specific and relevant tags supported by the document content.
- Pair each selected tag with the best icon.
- Use `general` when the tag is broad, user-defined, or no specific icon is a safe fit.
"""


def format_batch_tagger_prompt(
    documents: list[dict[str, str | bool]],
    active_tags: Sequence[str],
) -> str:
    """Format the batch-tagger prompt for a chunk of documents."""
    doc_parts: list[str] = []
    for document in documents:
        content = _truncate(str(document.get("content", "")))
        metadata_string = str(document.get("metadata_string", "")).strip()
        if document.get("is_image"):
            doc_parts.append(
                "\n".join(
                    [
                        f"### {document['file_name']} (image document)",
                        metadata_string or "No metadata available.",
                        "The image binary is attached immediately after the main prompt.",
                    ]
                )
            )
        else:
            doc_parts.append(
                f"### {document['file_name']} (type: {document.get('document_type', 'N/A')})\n{content}"
            )
    documents_block = "\n\n".join(doc_parts)
    active_tags_block = "\n".join(f"- {tag}" for tag in active_tags)
    fallback_note = (
        "## Fallback Tag\n"
        f"- `{CUSTOM_FALLBACK_TAG_LABEL}` is available for this run.\n"
        f"- Use it only when none of the other active tags apply to the document."
        if CUSTOM_FALLBACK_TAG_LABEL in active_tags
        else ""
    )
    return BATCH_TAGGER_PROMPT.format(
        tag_mode="Runtime",
        active_tags_block=active_tags_block,
        custom_tags_block=fallback_note,
        documents_block=documents_block,
    )
