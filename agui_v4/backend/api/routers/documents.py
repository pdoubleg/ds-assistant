"""Document summary, search/sort, tagging, and example-doc routes."""

import os

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse, StreamingResponse

from api.schemas.documents import SearchSortRequest, SummarizeRequest, TagRequest
from dependencies import get_document_mapper, get_text_extraction_service, get_upload_dir
from services.document_mapper import DocumentMapper
from services.ndjson import NDJSON_HEADERS, encode_ndjson_line
from services.text_extraction import TextExtractionService
from workflows.search_sort import run_search_sort
from workflows.summary import summarize_document
from workflows.tagging import batch_documents, run_tag_batch

router = APIRouter()


@router.post("/summarize")
async def summarize_endpoint(body: SummarizeRequest) -> StreamingResponse:
    """Summarize documents one at a time, streaming NDJSON results."""

    async def generate() -> str:
        for document in body.documents:
            if not document.content.strip():
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
                    mime_type=document.mime_type,
                    document_type=document.document_type,
                    additional_instructions=body.additional_instructions,
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


@router.post("/search-sort")
async def search_sort_endpoint(
    body: SearchSortRequest,
    mapper: DocumentMapper = Depends(get_document_mapper),
) -> JSONResponse:
    """Score documents against a user query using the search/sort workflow."""
    try:
        payload = await run_search_sort(body.query, body.documents, mapper=mapper)
        return JSONResponse(payload)
    except Exception as exc:
        print(f"[SEARCH-SORT ERROR] {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return JSONResponse(
            {"error": str(exc), "scores": [], "content_id_to_file_name": {}},
            status_code=500,
        )


@router.post("/document-tags")
async def document_tags_endpoint(
    body: TagRequest,
    mapper: DocumentMapper = Depends(get_document_mapper),
) -> StreamingResponse:
    """Tag documents in batches, streaming NDJSON progress."""
    active_tags = body.get_active_tags()

    async def generate() -> str:
        batches = batch_documents(body.documents)
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches):
            try:
                results = await run_tag_batch(batch, active_tags=active_tags, mapper=mapper)
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


@router.get("/example-docs")
async def example_docs_endpoint(
    text_extraction_service: TextExtractionService = Depends(get_text_extraction_service),
    upload_dir: str = Depends(get_upload_dir),
) -> JSONResponse:
    """List pre-loaded example documents from the uploads directory."""
    documents: list[dict[str, object]] = []
    try:
        for file_name in sorted(os.listdir(upload_dir)):
            extension = os.path.splitext(file_name)[1].lower()
            if extension not in text_extraction_service.allowed_extensions:
                continue

            file_path = os.path.join(upload_dir, file_name)
            if not os.path.isfile(file_path):
                continue

            with open(file_path, "rb") as file_obj:
                file_bytes = file_obj.read()
            documents.append(
                text_extraction_service.build_example_document_payload(file_name, file_bytes)
            )
    except Exception as exc:
        print(f"[EXAMPLE-DOCS] Error scanning uploads dir: {exc}", flush=True)

    return JSONResponse({"documents": documents})
