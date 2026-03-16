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


@router.post("/doc-lens/session")
async def doc_lens_session_endpoint(
    body: DocLensSessionRequest,
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> StreamingResponse:
    """Create a Doc Lens session and ingest files, streaming NDJSON progress."""
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


@router.post("/doc-lens/query")
async def doc_lens_query_endpoint(body: DocLensQueryRequest) -> JSONResponse:
    """Run a natural-language image query against an active Doc Lens session."""
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


@router.post("/doc-lens/document-assets")
async def doc_lens_document_assets_endpoint(body: DocLensDocumentAssetsRequest) -> JSONResponse:
    """List all extracted assets for one document in an active session."""
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


@router.get("/doc-lens/session/{session_id}")
async def doc_lens_session_summary(session_id: str) -> JSONResponse:
    """Return summary stats for a Doc Lens session."""
    try:
        service = get_doc_lens_service()
        summary = service.get_session_summary(session_id)
        return JSONResponse(summary.model_dump())
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.delete("/doc-lens/session/{session_id}")
async def doc_lens_clear_session(session_id: str) -> JSONResponse:
    """Clear all data for a Doc Lens session."""
    try:
        service = get_doc_lens_service()
        service.clear_session(session_id)
        return JSONResponse({"message": "Session cleared.", "session_id": session_id})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
