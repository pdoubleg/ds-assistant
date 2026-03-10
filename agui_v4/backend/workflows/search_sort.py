"""Document search and sort workflow helpers."""

from typing import Any

from prompts.documents import format_doc_search_sort_prompt
from services.document_mapper import DocumentMapper
from workflows.agent_factory import SearchSortDeps, search_sort_agent


async def run_search_sort(
    query: str,
    documents: list[Any],
    mapper: DocumentMapper | None = None,
) -> dict[str, Any]:
    """Score documents against a user query.

    Args:
        query: User query to rank or filter documents.
        documents: Search/sort request payloads.
        mapper: Optional shared document mapper.

    Returns:
        Dictionary containing `scores` and `content_id_to_file_name`.
    """
    if not documents:
        return {"scores": [], "content_id_to_file_name": {}}

    mapper = mapper or DocumentMapper()
    document_models, content_id_to_file_name = mapper.build_search_sort_documents(documents)
    deps = SearchSortDeps(documents=document_models)

    prompt = format_doc_search_sort_prompt(query=query)
    result = await search_sort_agent.run(prompt, deps=deps)
    return {
        "scores": [score.model_dump() for score in result.output.scores],
        "content_id_to_file_name": content_id_to_file_name,
    }
