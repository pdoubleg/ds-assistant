"""File upload routes."""

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.responses import JSONResponse

from dependencies import get_runtime_storage_service, get_text_extraction_service
from services.runtime_storage import RuntimeStorageService
from services.text_extraction import TextExtractionService

router = APIRouter()


@router.post("/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    text_extraction_service: TextExtractionService = Depends(get_text_extraction_service),
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> JSONResponse:
    """Handle supported document uploads and extract text content."""
    try:
        filename = file.filename or "unknown"
        extension = filename.lower()[filename.rfind(".") :] if "." in filename else ""

        if extension not in text_extraction_service.allowed_extensions:
            allowed = ", ".join(sorted(text_extraction_service.allowed_extensions))
            return JSONResponse(
                {"error": f"Unsupported file type: {extension}. Allowed: {allowed}"},
                status_code=400,
            )

        file_bytes = await file.read()
        staged_document = runtime_storage.stage_bytes(filename, file_bytes)

        response_payload = text_extraction_service.build_upload_response(
            file_name=filename,
            file_bytes=file_bytes,
            content_id=staged_document.content_id,
            content_url=staged_document.public_url,
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
