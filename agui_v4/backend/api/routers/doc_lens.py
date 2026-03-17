"""Doc Lens routes."""

from uuid import uuid4

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse, StreamingResponse

from api.schemas.doc_lens import (
    DocLensDocumentAssetsRequest,
    DocLensQueryRequest,
    DocLensSessionRequest,
)
from dependencies import get_doc_lens_service, get_runtime_storage_service
from services.doc_lens_factory import (
    DOC_LENS_IMAGE_MIMES,
    DOC_LENS_PDF_MIMES,
    reset_doc_lens_service_if_fatal,
)
from services.ndjson import NDJSON_HEADERS, encode_ndjson_line
from services.runtime_storage import RuntimeStorageService

router = APIRouter()


@router.post(
    "/doc-lens/session",
    summary="Create a Doc Lens session",
    response_description="NDJSON stream — ingestion progress per file, ending with a session-ready summary.",
    responses={
        500: {"description": "Session creation or file ingestion failed (reported inline)."},
    },
)
async def doc_lens_session_endpoint(
    body: DocLensSessionRequest,
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> StreamingResponse:
    """Create a new Doc Lens session and ingest the specified files, streaming
    **NDJSON** progress events.

    Streamed event types:

    | `type` | Meaning |
    |---|---|
    | `session_created` | Session ID assigned. |
    | `ingest_start` | File ingestion beginning. |
    | `ingest_complete` | File successfully ingested with extracted asset counts. |
    | `ingest_error` | File-level error (processing continues). |
    | `session_ready` | All files processed — includes session summary stats. |
    | `session_error` | Fatal error retrieving the session summary. |
    """
    session_id = str(uuid4())

    async def generate() -> str:
        yield encode_ndjson_line({"type": "session_created", "session_id": session_id})

        service = get_doc_lens_service()
        total = len(body.files)

        for idx, file_desc in enumerate(body.files):
            file_index = idx + 1
            yield encode_ndjson_line(
                {
                    "type": "ingest_start",
                    "file_name": file_desc.file_name,
                    "mime_type": file_desc.mime_type,
                    "file_index": file_index,
                    "total_files": total,
                }
            )

            try:
                file_path = runtime_storage.resolve_staged_document_path(
                    content_id=file_desc.content_id,
                    file_name=file_desc.file_name,
                )
                if file_path is None:
                    yield encode_ndjson_line(
                        {
                            "type": "ingest_error",
                            "file_name": file_desc.file_name,
                            "error": (
                                f"Temp document not found. content_id={file_desc.content_id}"
                            ),
                            "file_index": file_index,
                            "total_files": total,
                        }
                    )
                    continue

                if file_desc.mime_type in DOC_LENS_PDF_MIMES:
                    result = service.ingest_pdf(
                        session_id=session_id,
                        document_name=file_desc.file_name,
                        pdf_path=str(file_path),
                    )
                elif file_desc.mime_type in DOC_LENS_IMAGE_MIMES:
                    result = service.ingest_image(
                        session_id=session_id,
                        document_name=file_desc.file_name,
                        image_path=str(file_path),
                    )
                else:
                    yield encode_ndjson_line(
                        {
                            "type": "ingest_error",
                            "file_name": file_desc.file_name,
                            "error": f"Unsupported mime type: {file_desc.mime_type}",
                            "file_index": file_index,
                            "total_files": total,
                        }
                    )
                    continue

                yield encode_ndjson_line(
                    {
                        "type": "ingest_complete",
                        "file_name": file_desc.file_name,
                        "file_index": file_index,
                        "total_files": total,
                        **result.model_dump(),
                    }
                )
            except Exception as exc:
                print(f"[DOC-LENS INGEST ERROR] {file_desc.file_name}: {exc}", flush=True)
                if reset_doc_lens_service_if_fatal(exc):
                    print("[DOC-LENS] Service singleton reset after fatal DB error.", flush=True)
                yield encode_ndjson_line(
                    {
                        "type": "ingest_error",
                        "file_name": file_desc.file_name,
                        "error": str(exc),
                        "file_index": file_index,
                        "total_files": total,
                    }
                )

        try:
            summary = service.get_session_summary(session_id)
            yield encode_ndjson_line({"type": "session_ready", **summary.model_dump()})
        except Exception as exc:
            yield encode_ndjson_line(
                {
                    "type": "session_error",
                    "error": f"Failed to get session summary: {exc}",
                }
            )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers=NDJSON_HEADERS,
    )


@router.post(
    "/doc-lens/query",
    summary="Query a Doc Lens session",
    response_description="Ranked query results with matched asset metadata.",
    responses={
        500: {"description": "Query execution failed."},
    },
)
async def doc_lens_query_endpoint(body: DocLensQueryRequest) -> JSONResponse:
    """Run a natural-language query against an active Doc Lens session.

    Searches ingested document assets (images, tables, text blocks) using the
    specified `search_mode` and returns the top-*k* most relevant hits.
    Optionally filter by `asset_types` or `document_ids`.
    """
    try:
        service = get_doc_lens_service()
        response = service.query(
            session_id=body.session_id,
            query_text=body.query,
            search_mode=body.search_mode,
            top_k=body.top_k,
            asset_types=body.asset_types,
            document_ids=body.document_ids,
        )
        return JSONResponse(response.model_dump())
    except Exception as exc:
        print(f"[DOC-LENS QUERY ERROR] {exc}", flush=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post(
    "/doc-lens/document-assets",
    summary="List assets for a document",
    response_description="All extracted assets (images, tables, etc.) for the specified document.",
    responses={
        500: {"description": "Asset retrieval failed."},
    },
)
async def doc_lens_document_assets_endpoint(body: DocLensDocumentAssetsRequest) -> JSONResponse:
    """List every extracted asset for a single document within an active Doc Lens
    session.

    Returns the full set of ingested assets (images, tables, text blocks)
    without any query-based ranking — useful for browsing or inventorying
    document content.
    """
    try:
        service = get_doc_lens_service()
        hits = service.list_document_assets(
            session_id=body.session_id,
            document_id=body.document_id,
        )
        return JSONResponse(
            {
                "session_id": body.session_id,
                "document_id": body.document_id,
                "hits": [hit.model_dump() for hit in hits],
            }
        )
    except Exception as exc:
        print(f"[DOC-LENS DOCUMENT-ASSETS ERROR] {exc}", flush=True)
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get(
    "/doc-lens/session/{session_id}",
    summary="Get Doc Lens session summary",
    response_description="Session statistics including document and asset counts.",
    responses={
        500: {"description": "Session not found or summary retrieval failed."},
    },
)
async def doc_lens_session_summary(session_id: str) -> JSONResponse:
    """Return summary statistics for an existing Doc Lens session.

    Includes total document count, per-document asset counts, and ingestion
    status metadata.
    """
    try:
        service = get_doc_lens_service()
        summary = service.get_session_summary(session_id)
        return JSONResponse(summary.model_dump())
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.delete(
    "/doc-lens/session/{session_id}",
    summary="Clear a Doc Lens session",
    response_description="Confirmation that the session data was cleared.",
    responses={
        500: {"description": "Session not found or deletion failed."},
    },
)
async def doc_lens_clear_session(session_id: str) -> JSONResponse:
    """Permanently clear all ingested data for a Doc Lens session.

    After this call the `session_id` is no longer valid for queries or asset
    lookups.
    """
    try:
        service = get_doc_lens_service()
        service.clear_session(session_id)
        return JSONResponse({"message": "Session cleared.", "session_id": session_id})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
