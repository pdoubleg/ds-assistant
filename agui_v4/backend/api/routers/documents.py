"""Document summary, search/sort, tagging, and example-doc routes."""

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse, StreamingResponse

from api.schemas.documents import (
    SearchSortRequest,
    SearchSortResponse,
    SummarizeRequest,
    TagRequest,
)
from dependencies import (
    get_document_mapper,
    get_runtime_storage_service,
    get_text_extraction_service,
)
from services.document_mapper import DocumentMapper
from services.ndjson import NDJSON_HEADERS, encode_ndjson_line
from services.runtime_storage import RuntimeStorageService
from services.text_extraction import TextExtractionService
from workflows.search_sort import run_search_sort
from workflows.summary import summarize_document
from workflows.tagging import batch_documents, run_tag_batch

router = APIRouter()


@router.post(
    "/summarize",
    summary="Summarize documents",
    response_description="NDJSON stream — one summary object per document.",
    responses={
        500: {"description": "Summarization failed for one or more documents (reported inline)."},
    },
)
async def summarize_endpoint(
    body: SummarizeRequest,
    mapper: DocumentMapper = Depends(get_document_mapper),
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> StreamingResponse:
    """Summarize one or more documents, streaming results as **NDJSON**.

    Each line in the response is a JSON object keyed by `file_name` containing
    either the generated summary payload or an `error` field.  Documents with
    empty content are skipped with an inline error.

    Optionally include `additional_instructions` to steer summarization style
    or focus areas.
    """

    async def generate() -> str:
        for document in body.documents:
            if not document.content.strip() and not document.mime_type.startswith("image/"):
                yield encode_ndjson_line(
                    {
                        "file_name": document.file_name,
                        "error": "No extractable content.",
                    }
                )
                continue

            try:
                payload = await summarize_document(
                    file_name=document.file_name,
                    content=document.content,
                    content_id=document.content_id,
                    mime_type=document.mime_type,
                    content_url=document.content_url,
                    document_type=document.document_type,
                    document_description=document.document_description,
                    additional_instructions=body.additional_instructions,
                    mapper=mapper,
                    runtime_storage=runtime_storage,
                )
                yield encode_ndjson_line(payload)
            except Exception as exc:
                print(f"[SUMMARIZE ERROR] {document.file_name}: {exc}", flush=True)
                yield encode_ndjson_line({"file_name": document.file_name, "error": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers=NDJSON_HEADERS,
    )


@router.post(
    "/search-sort",
    summary="Search and sort documents",
    response_model=SearchSortResponse,
    response_description="Scored document list with content-ID lookup table.",
    responses={
        500: {"description": "Search/sort workflow failed — returns an empty response shell."},
    },
)
async def search_sort_endpoint(
    body: SearchSortRequest,
    mapper: DocumentMapper = Depends(get_document_mapper),
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> JSONResponse:
    """Score and rank documents against a natural-language query using the AI-powered
    search/sort workflow.

    Returns a subset of highest-scoring documents along with a `content_id` →
    `file_name` mapping for client-side resolution.
    """
    try:
        payload = await run_search_sort(
            body.query,
            body.documents,
            mapper=mapper,
            runtime_storage=runtime_storage,
        )
        return JSONResponse(payload.model_dump())
    except Exception as exc:
        print(f"[SEARCH-SORT ERROR] {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return JSONResponse(
            SearchSortResponse().model_dump(),
            status_code=500,
        )


@router.post(
    "/document-tags",
    summary="Auto-tag documents",
    response_description="NDJSON stream — one batch result per line, ending with a done sentinel.",
    responses={
        500: {"description": "Tagging failed for one or more batches (reported inline)."},
    },
)
async def document_tags_endpoint(
    body: TagRequest,
    mapper: DocumentMapper = Depends(get_document_mapper),
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> StreamingResponse:
    """Classify documents into tag categories using an AI tagging workflow,
    streaming **NDJSON** progress per batch.

    Documents are split into batches for parallel processing.  Each streamed
    line contains the batch index, total batches, and per-document tag results.
    The final line is a `done` sentinel with the canonical tag vocabulary used.

    Use `tag_mode="custom"` with `selected_tags` to override the default
    taxonomy.
    """
    active_tags = body.get_active_tags()

    async def generate() -> str:
        batches = batch_documents(body.documents)
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches):
            try:
                results = await run_tag_batch(
                    batch,
                    active_tags=active_tags,
                    mapper=mapper,
                    runtime_storage=runtime_storage,
                )
                yield encode_ndjson_line(
                    {
                        "batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "results": results,
                    }
                )
            except Exception as exc:
                print(f"[TAG BATCH ERROR] Batch {batch_idx + 1}: {exc}", flush=True)
                yield encode_ndjson_line(
                    {
                        "batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "error": str(exc),
                    }
                )

        yield encode_ndjson_line(
            {
                "done": True,
                "canonical_tags": active_tags,
                "tag_mode": body.tag_mode,
            }
        )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers=NDJSON_HEADERS,
    )


@router.get(
    "/example-docs",
    summary="List example documents",
    response_description="Array of pre-staged example document payloads.",
)
async def example_docs_endpoint(
    text_extraction_service: TextExtractionService = Depends(get_text_extraction_service),
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> JSONResponse:
    """Stage and return the built-in example documents from the static assets directory.

    Each document is automatically text-extracted and served with a `content_url`
    for direct browser access.  Useful for demos and local development.
    """
    staged_documents = runtime_storage.stage_static_examples(
        text_extraction_service.allowed_extensions
    )
    documents = [
        text_extraction_service.build_document_payload(
            file_name=staged_document.file_name,
            file_bytes=staged_document.file_path.read_bytes(),
            content_id=staged_document.content_id,
            content_url=staged_document.public_url,
        )
        for staged_document in staged_documents
    ]
    return JSONResponse({"documents": documents})
