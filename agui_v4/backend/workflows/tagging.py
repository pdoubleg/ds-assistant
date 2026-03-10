"""Document tagging workflow helpers."""

from typing import Any

from models.tagging import build_runtime_batch_tag_schema
from prompts.tagging import build_batch_tagger_instructions, format_batch_tagger_prompt
from services.document_mapper import DocumentMapper
from workflows.agent_factory import get_batch_tagger_agent

MAX_BATCH_DOCS = 10
MAX_BATCH_CHARS = 25_000


def batch_documents(documents: list[Any]) -> list[list[Any]]:
    """Split documents into batches by count and total characters.

    Args:
        documents: Tagging request payloads exposing a `content` field.

    Returns:
        Batches of document payloads.
    """
    batches: list[list[Any]] = []
    current: list[Any] = []
    chars = 0

    for document in documents:
        doc_chars = len(getattr(document, "content", ""))
        if current and (len(current) >= MAX_BATCH_DOCS or chars + doc_chars > MAX_BATCH_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(document)
        chars += doc_chars

    if current:
        batches.append(current)
    return batches


async def run_tag_batch(
    documents: list[Any],
    active_tags: list[str],
    mapper: DocumentMapper | None = None,
) -> list[dict[str, Any]]:
    """Tag one batch of documents.

    Args:
        documents: Tagging payloads for a single batch.
        active_tags: Runtime tag vocabulary.
        mapper: Optional shared document mapper.

    Returns:
        List of result dictionaries for the batch.
    """
    mapper = mapper or DocumentMapper()
    runtime_schema = build_runtime_batch_tag_schema(active_tags)
    instructions = build_batch_tagger_instructions(active_tags)
    prompt_documents = mapper.tagging_prompt_documents(documents)
    prompt = format_batch_tagger_prompt(prompt_documents, active_tags=active_tags)
    batch_tagger_agent = get_batch_tagger_agent()

    with batch_tagger_agent.override(instructions=instructions):
        result = await batch_tagger_agent.run(
            prompt,
            output_type=runtime_schema.batch_result_model,
        )
    return [entry.model_dump() for entry in result.output.results]
