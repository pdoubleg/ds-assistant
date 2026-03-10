"""File upload routes."""

import os

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.responses import JSONResponse

from dependencies import get_text_extraction_service, get_upload_dir
from services.text_extraction import TextExtractionService

router = APIRouter()


@router.post("/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    text_extraction_service: TextExtractionService = Depends(get_text_extraction_service),
    upload_dir: str = Depends(get_upload_dir),
) -> JSONResponse:
    """Handle supported document uploads and extract text content."""
    try:
        filename = file.filename or "unknown"
        extension = os.path.splitext(filename)[1].lower()

        if extension not in text_extraction_service.allowed_extensions:
            allowed = ", ".join(sorted(text_extraction_service.allowed_extensions))
            return JSONResponse(
                {"error": f"Unsupported file type: {extension}. Allowed: {allowed}"},
                status_code=400,
            )

        file_bytes = await file.read()
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, "wb") as file_obj:
            file_obj.write(file_bytes)

        response_payload = text_extraction_service.build_upload_response(filename, file_bytes)
        response_payload["path"] = file_path

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
