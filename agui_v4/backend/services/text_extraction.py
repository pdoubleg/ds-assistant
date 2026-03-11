"""Document text extraction and upload catalog helpers."""

import io
import os
from typing import Callable


class TextExtractionService:
    """Extract text from supported document types and describe uploaded files."""

    def __init__(self) -> None:
        """Initialize supported extractors and MIME mappings."""
        self.extractors: dict[str, Callable[[bytes], tuple[str, int]]] = {
            ".pdf": self.extract_text_from_pdf,
            ".docx": self.extract_text_from_docx,
            ".xlsx": self.extract_text_from_xlsx,
        }
        self.ext_to_mime: dict[str, str] = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }

    @property
    def allowed_extensions(self) -> set[str]:
        """Return all supported file extensions."""
        return set(self.extractors.keys())

    def extract_text_from_pdf(self, file_bytes: bytes) -> tuple[str, int]:
        """Extract text and page count from a PDF using PyMuPDF.

        Args:
            file_bytes: Raw PDF bytes.

        Returns:
            Tuple containing extracted text and page count.
        """
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages), len(pages)

    def extract_text_from_docx(self, file_bytes: bytes) -> tuple[str, int]:
        """Extract text and an approximate page count from a DOCX file.

        Args:
            file_bytes: Raw DOCX bytes.

        Returns:
            Tuple containing extracted text and estimated page count.
        """
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        text = "\n\n".join(paragraphs)
        page_estimate = max(1, len(text) // 3000)
        return text, page_estimate

    def extract_text_from_xlsx(self, file_bytes: bytes) -> tuple[str, int]:
        """Extract text from all sheets in an XLSX workbook.

        Args:
            file_bytes: Raw XLSX bytes.

        Returns:
            Tuple containing extracted text and sheet count.
        """
        from openpyxl import load_workbook

        workbook = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
        parts: list[str] = []
        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]
            rows: list[str] = []
            for row in worksheet.iter_rows(values_only=True):
                cells = [str(cell) if cell is not None else "" for cell in row]
                if any(cells):
                    rows.append("\t".join(cells))
            if rows:
                parts.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows))
        workbook.close()
        return "\n\n".join(parts), len(workbook.sheetnames)

    def extract(self, extension: str, file_bytes: bytes) -> tuple[str, int]:
        """Extract text using the registered extractor for an extension.

        Args:
            extension: Lowercase file extension including the leading dot.
            file_bytes: Raw uploaded bytes.

        Returns:
            Tuple containing extracted text and page or sheet count.

        Raises:
            ValueError: If the extension is unsupported.
        """
        extractor = self.extractors.get(extension)
        if extractor is None:
            raise ValueError(f"Unsupported file type: {extension}")
        return extractor(file_bytes)

    def format_file_size(self, size_bytes: int) -> str:
        """Return a human-readable file size string.

        Args:
            size_bytes: Raw file size in bytes.

        Returns:
            Formatted kilobyte or megabyte string.
        """
        if size_bytes > 1024 * 1024:
            return f"{size_bytes / 1024 / 1024:.1f} MB"
        return f"{size_bytes / 1024:.1f} KB"

    def build_example_document_payload(
        self, file_name: str, file_bytes: bytes
    ) -> dict[str, object]:
        """Build the example-doc payload for one uploaded file.

        Args:
            file_name: File name on disk.
            file_bytes: File content bytes.

        Returns:
            JSON-serializable payload matching the existing frontend contract.
        """
        extension = os.path.splitext(file_name)[1].lower()
        text, pages = self.extract(extension, file_bytes)
        size = len(file_bytes)
        return {
            "file_name": file_name,
            "mime_type": self.ext_to_mime.get(extension, "application/octet-stream"),
            "content": text,
            "page_count": pages,
            "file_size": self.format_file_size(size),
            "file_size_bytes": size,
            "path": f"/uploads/{file_name}",
        }

    def build_upload_response(self, file_name: str, file_bytes: bytes) -> dict[str, object]:
        """Build the upload response payload after extraction.

        Args:
            file_name: Uploaded file name.
            file_bytes: Raw uploaded bytes.

        Returns:
            JSON-serializable payload for `POST /upload`.
        """
        extension = os.path.splitext(file_name)[1].lower()
        extracted_text, page_count = self.extract(extension, file_bytes)
        file_size = len(file_bytes)
        return {
            "filename": file_name,
            "file_type": extension.lstrip("."),
            "file_size": self.format_file_size(file_size),
            "file_size_bytes": file_size,
            "page_count": page_count,
            "content": extracted_text,
        }
