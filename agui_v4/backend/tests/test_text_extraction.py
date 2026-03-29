"""Focused tests for document extraction MIME support."""

import io
from email.message import EmailMessage

from openpyxl import Workbook
from pydantic_ai import BinaryContent

from services.document_mapper import DocumentMapper
from services.runtime_storage import RuntimeStorageService
from services.text_extraction import TextExtractionService


def test_build_upload_response_supports_html_mime_types() -> None:
    """HTML uploads should resolve MIME metadata and extract readable text."""
    service = TextExtractionService()

    response = service.build_upload_response(
        file_name="example.html",
        file_bytes=b"<html><body><h1>Hello</h1><p>World</p></body></html>",
        mime_type="text/html",
    )

    assert response["file_type"] == "html"
    assert response["mime_type"] == "text/html"
    assert "Hello" in response["content"]
    assert "World" in response["content"]


def test_build_upload_response_supports_rfc822_messages() -> None:
    """RFC822 emails should preserve key headers and body text."""
    service = TextExtractionService()
    message = EmailMessage()
    message["Subject"] = "Coverage review"
    message["From"] = "adjuster@example.com"
    message["To"] = "team@example.com"
    message.set_content("Please review the attached loss details.")

    response = service.build_upload_response(
        file_name="notice.eml",
        file_bytes=message.as_bytes(),
        mime_type="message/rfc822",
    )

    assert response["file_type"] == "eml"
    assert response["mime_type"] == "message/rfc822"
    assert "Subject: Coverage review" in response["content"]
    assert "Please review the attached loss details." in response["content"]


def test_build_upload_response_supports_xlsx_workbooks() -> None:
    """Spreadsheet uploads should extract row text from each sheet."""
    service = TextExtractionService()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Claim"
    sheet.append(["Column A", "Column B"])
    sheet.append(["Loss", 42])

    buffer = io.BytesIO()
    workbook.save(buffer)

    response = service.build_upload_response(
        file_name="loss-summary.xlsx",
        file_bytes=buffer.getvalue(),
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert response["file_type"] == "xlsx"
    assert response["mime_type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "Sheet: Claim" in response["content"]
    assert "Loss\t42" in response["content"]


def test_build_upload_response_supports_image_placeholder_text() -> None:
    """Image uploads should succeed even when OCR is intentionally disabled."""
    service = TextExtractionService()
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\x0f\x00\x01\x01\x01\x00"
        b"\x18\xdd\x8d\xb1\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    response = service.build_upload_response(
        file_name="photo.png",
        file_bytes=png_bytes,
        mime_type="image/png",
    )

    assert response["file_type"] == "png"
    assert response["mime_type"] == "image/png"
    assert "OCR extraction is not enabled" in response["content"]
    assert "MIME type: image/png" in response["content"]


def test_is_supported_upload_uses_extension_and_mime_fallbacks() -> None:
    """Support checks should accept the new extension and MIME combinations."""
    service = TextExtractionService()

    assert service.is_supported_upload("summary.txt", "text/plain")
    assert service.is_supported_upload("message.msg", "application/octet-stream")
    assert service.is_supported_upload("report.xls", "application/vnd.ms-excel")
    assert not service.is_supported_upload("archive.zip", "application/zip")


def test_document_mapper_builds_image_binary_prompt_parts(tmp_path) -> None:
    """Image prompt parts should include metadata text and staged bytes."""
    storage = RuntimeStorageService(tmp_path)
    mapper = DocumentMapper()
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\x0f\x00\x01\x01\x01\x00"
        b"\x18\xdd\x8d\xb1\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    staged = storage.stage_bytes("photo.png", png_bytes, content_id="image-1")
    payload = {
        "file_name": "photo.png",
        "content_id": staged.content_id,
        "mime_type": "image/png",
        "content_url": staged.public_url,
        "document_type": "Upload",
        "document_description": "Exterior damage photo",
    }

    prompt_parts = mapper.build_image_prompt_parts([payload], storage)

    assert len(prompt_parts) == 2
    assert "photo.png" in prompt_parts[0]
    assert "Exterior damage photo" in prompt_parts[0]
    assert isinstance(prompt_parts[1], BinaryContent)
    assert prompt_parts[1].media_type == "image/png"
    assert prompt_parts[1].identifier == "photo.png"
    assert prompt_parts[1].data == png_bytes
