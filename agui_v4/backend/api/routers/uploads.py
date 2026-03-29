"""File upload routes."""

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.responses import JSONResponse

from dependencies import get_runtime_storage_service, get_text_extraction_service
from services.runtime_storage import RuntimeStorageService
from services.text_extraction import TextExtractionService

router = APIRouter()


@router.post(
    "/upload",
    summary="Upload a document",
    response_description="Extracted text content, page count, and staging metadata.",
    responses={
        400: {"description": "Unsupported file type."},
        500: {"description": "Text extraction or staging failed."},
    },
)
async def upload_endpoint(
    file: UploadFile = File(..., description="The document file to upload."),
    text_extraction_service: TextExtractionService = Depends(get_text_extraction_service),
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> JSONResponse:
    """Upload a document, stage it to the runtime temp directory, and extract text content.

    Supported extensions are determined by the active `TextExtractionService`
    configuration.  The response includes the extracted plain-text content,
    total page count, a `content_id` for referencing the staged file, and a
    `content_url` for direct browser access.
    """
    try:
        filename = file.filename or "unknown"
        client_mime_type = file.content_type or None

        if not text_extraction_service.is_supported_upload(filename, client_mime_type):
            allowed = ", ".join(sorted(text_extraction_service.allowed_extensions))
            return JSONResponse(
                {
                    "error": (
                        f"Unsupported file type: {filename} ({client_mime_type or 'unknown MIME'}). "
                        f"Allowed extensions: {allowed}"
                    )
                },
                status_code=400,
            )

        file_bytes = await file.read()
        staged_document = runtime_storage.stage_bytes(filename, file_bytes)

        response_payload = text_extraction_service.build_upload_response(
            file_name=filename,
            file_bytes=file_bytes,
            content_id=staged_document.content_id,
            content_url=staged_document.public_url,
            mime_type=client_mime_type,
        )

        print(
            f"[UPLOAD] {filename}: {len(file_bytes)} bytes, "
            f"{response_payload['page_count']} pages, {len(response_payload['content'])} chars extracted",
            flush=True,
        )
        return JSONResponse(response_payload)
    except Exception as exc:
        print(f"[UPLOAD ERROR] {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)
