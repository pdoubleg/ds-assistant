"""Document search and sort workflow helpers."""

from typing import Any

from api.schemas.documents import SearchSortResponse
from models.search import DocSearchScore
from prompts.documents import format_doc_search_sort_prompt
from services.document_mapper import DocumentMapper
from services.runtime_storage import RuntimeStorageService
from workflows.agent_factory import SearchSortDeps, search_sort_agent


def _normalize_doc_search_scores(
    scores: list[DocSearchScore],
) -> list[DocSearchScore]:
    """Normalize model output into a simple subset response.

    Args:
        scores: Raw scores returned by the search/sort agent.

    Returns:
        A de-duplicated list of scores that preserves the model's chosen order.

    Example:
        >>> _normalize_doc_search_scores([])
        []
    """
    normalized_scores: list[DocSearchScore] = []
    seen_content_ids: set[str] = set()

    for score in scores:
        if score.content_id in seen_content_ids:
            continue

        seen_content_ids.add(score.content_id)
        normalized_scores.append(score)

    return normalized_scores


async def run_search_sort(
    query: str,
    documents: list[Any],
    mapper: DocumentMapper | None = None,
    runtime_storage: RuntimeStorageService | None = None,
) -> SearchSortResponse:
    """Score documents against a user query.

    Args:
        query: User query to rank or filter documents.
        documents: Search/sort request payloads.
        mapper: Optional shared document mapper.
        runtime_storage: Optional runtime storage for staged image bytes.

    Returns:
        Normalized subset response containing `scores` and `content_id_to_file_name`.
    """
    if not documents:
        return SearchSortResponse()

    mapper = mapper or DocumentMapper()
    document_models, content_id_to_file_name = mapper.build_search_sort_documents(documents)
    deps = SearchSortDeps(
        documents=document_models,
        mapper=mapper,
        runtime_storage=runtime_storage,
    )

    prompt = format_doc_search_sort_prompt(query=query)
    result = await search_sort_agent.run(prompt, deps=deps)
    normalized_scores = _normalize_doc_search_scores(
        scores=result.output.scores,
    )
    return SearchSortResponse(
        scores=normalized_scores,
        content_id_to_file_name=content_id_to_file_name,
    )
