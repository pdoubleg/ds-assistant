"""Summary-oriented LLM workflows."""

from typing import Any

from services.runtime_storage import RuntimeStorageService
from models.analysis import AnalysisResult
from prompts.analysis import format_analysis_prompt, format_component_prompt
from prompts.documents import format_document_summary_prompt
from services.document_mapper import DocumentMapper
from workflows.agent_factory import analysis_agent, component_agent, document_summary_agent


async def summarize_document(
    file_name: str,
    content: str,
    mime_type: str = "unknown",
    content_id: str = "",
    content_url: str = "",
    document_type: str = "",
    document_description: str = "",
    additional_instructions: str = "",
    mapper: DocumentMapper | None = None,
    runtime_storage: RuntimeStorageService | None = None,
) -> dict[str, Any]:
    """Summarize one document using the shared summary agent.

    Args:
        file_name: Document file name.
        content: Extracted text content.
        mime_type: MIME type or extension string.
        content_id: Optional staged content identifier.
        content_url: Optional staged content URL.
        document_type: High-level document type.
        document_description: Optional document description.
        additional_instructions: Optional user guidance.
        mapper: Optional shared document mapper.
        runtime_storage: Optional runtime storage for staged image bytes.

    Returns:
        JSON-serializable summary payload.
    """
    mapper = mapper or DocumentMapper()
    payload = {
        "file_name": file_name,
        "content_id": content_id,
        "content": content,
        "mime_type": mime_type,
        "content_url": content_url,
        "document_type": document_type,
        "document_description": document_description,
    }
    prompt = format_document_summary_prompt(
        document_metadata=mapper.build_document_metadata_string(payload),
        document_content=(
            content if not mapper.is_image_document(payload) else "The image is attached separately."
        ),
        additional_instructions=additional_instructions,
    )
    image_prompt_parts = mapper.build_image_prompt_parts([payload], runtime_storage)
    user_prompt: str | list[Any] = [prompt, *image_prompt_parts] if image_prompt_parts else prompt

    result = await document_summary_agent.run(user_prompt)
    summary_payload = result.output.model_dump()
    summary_payload["file_name"] = file_name
    return summary_payload


async def run_summary_analysis(
    document_contents: list[dict[str, Any]] | None,
    focus: str = "General claim review",
    mapper: DocumentMapper | None = None,
) -> AnalysisResult:
    """Run the two-step claim analysis workflow.

    Args:
        document_contents: Lightweight prompt payloads or `None`.
        focus: User-supplied focus area.
        mapper: Optional shared document mapper.

    Returns:
        Structured `AnalysisResult` output.
    """
    mapper = mapper or DocumentMapper()
    combined_text: str | None = None
    doc_count = 0
    if document_contents:
        combined_text, doc_count = mapper.combine_documents(document_contents)

    analysis_prompt = format_analysis_prompt(combined_text, focus=focus)
    print(
        f"[ORCHESTRATOR] Step 1/2 context analysis on {doc_count} document(s), "
        f"focus='{focus}', ~{len(combined_text or '')} chars",
        flush=True,
    )

    try:
        context_result = await analysis_agent.run(analysis_prompt)
        context_brief = context_result.output
        print(
            f"[ORCHESTRATOR] Context brief generated (~{len(context_brief)} chars). "
            "Step 2/2 structured component generation...",
            flush=True,
        )
        component_prompt = format_component_prompt(context_brief, focus=focus)
        structured_result = await component_agent.run(component_prompt)
        return structured_result.output
    except Exception as exc:
        print(f"[ORCHESTRATOR ERROR] Analysis failed: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return AnalysisResult(
            claim_overview="Analysis could not be completed. Please retry or refine your documents.",
        )
