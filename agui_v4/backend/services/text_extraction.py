"""Document text extraction and upload catalog helpers."""

import io
import os
from typing import Callable


class TextExtractionService:
    """Extract text from supported document types and describe uploaded files."""

    _tiktoken_encoding = None

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

    def extract_segments(self, extension: str, file_bytes: bytes) -> tuple[list[str], int]:
        """Extract ordered text segments for a supported file type.

        This helper keeps page- or sheet-aware extraction centralized so other
        services do not need file-type-specific parsing logic.

        Args:
            extension: Lowercase file extension including the leading dot.
            file_bytes: Raw uploaded bytes.

        Returns:
            Tuple containing ordered text segments and the page or sheet count.

        Raises:
            ValueError: If the extension is unsupported.
        """
        extractor = self.extractors.get(extension)
        if extractor is None:
            raise ValueError(f"Unsupported file type: {extension}")

        if extension == ".pdf":
            return self.extract_pdf_segments(file_bytes)
        if extension == ".xlsx":
            return self.extract_xlsx_segments(file_bytes)

        text, page_count = extractor(file_bytes)
        return [text] if text else [""], page_count

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

    def extract_pdf_segments(self, file_bytes: bytes) -> tuple[list[str], int]:
        """Extract one text segment per PDF page.

        Args:
            file_bytes: Raw PDF bytes.

        Returns:
            Tuple containing per-page extracted text and the page count.
        """
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return pages, len(pages)

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

    def extract_xlsx_segments(self, file_bytes: bytes) -> tuple[list[str], int]:
        """Extract one text segment per XLSX sheet.

        Args:
            file_bytes: Raw XLSX bytes.

        Returns:
            Tuple containing per-sheet text and the sheet count.
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
            parts.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows) if rows else "")
        workbook.close()
        return parts, len(parts)

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
        segments, page_count = self.extract_segments(extension, file_bytes)
        return "\n\n".join(segment for segment in segments if segment), page_count

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using tiktoken cl100k_base.

        Uses a class-level cached encoder so the encoding is only loaded once
        across all calls and instances.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens in the text.

        Example usage::

            svc = TextExtractionService()
            svc.count_tokens("Hello, world!")  # => 4
        """
        if not text:
            return 0

        # Lazy-load the encoding once and cache on the class
        if TextExtractionService._tiktoken_encoding is None:
            import tiktoken

            TextExtractionService._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")

        return len(TextExtractionService._tiktoken_encoding.encode(text))

    def tag_segments(
        self,
        segments: list[str],
        file_name: str,
        content_id: str,
    ) -> list[str]:
        """Prepend a structured metadata tag to each text segment.

        Inserts a ``[DOC_META ...]`` line at the top of every non-empty
        segment so downstream LLMs can identify the source document and
        page/sheet number without relying on positional heuristics.

        Args:
            segments: Ordered text segments (one per page or sheet).
            file_name: Human-readable document file name.
            content_id: Stable backend identifier for the document.

        Returns:
            New list of segments with metadata tags prepended.

        Example usage::

            svc = TextExtractionService()
            tagged = svc.tag_segments(
                ["Page one text", "Page two text"],
                file_name="policy.pdf",
                content_id="abc123",
            )
            # tagged[0] starts with '[DOC_META content_id="abc123" page=1 ...]'
        """
        tagged: list[str] = []
        for idx, segment in enumerate(segments, start=1):
            if not segment:
                tagged.append(segment)
                continue
            tag = f'[DOC_META content_id="{content_id}" page={idx} doc_name="{file_name}"]'
            tagged.append(f"{tag}\n{segment}")
        return tagged

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

    def build_document_payload(
        self,
        file_name: str,
        file_bytes: bytes,
        content_id: str | None = None,
        content_url: str | None = None,
    ) -> dict[str, object]:
        """Build a frontend-facing document payload for one file.

        Each page/sheet segment is prefixed with a ``[DOC_META ...]`` tag so
        downstream LLMs can identify the source document and page number.

        Args:
            file_name: File name on disk.
            file_bytes: File content bytes.
            content_id: Optional stable backend document identifier.
            content_url: Optional public URL for preview/download access.

        Returns:
            JSON-serializable payload matching the existing frontend contract.
        """
        extension = os.path.splitext(file_name)[1].lower()
        safe_id = content_id or ""
        segments, page_count = self.extract_segments(extension, file_bytes)
        tagged = self.tag_segments(segments, file_name, safe_id)
        text = "\n\n".join(s for s in tagged if s)
        size = len(file_bytes)
        return {
            "file_name": file_name,
            "mime_type": self.ext_to_mime.get(extension, "application/octet-stream"),
            "content": text,
            "page_count": page_count,
            "file_size": self.format_file_size(size),
            "file_size_bytes": size,
            "token_count": self.count_tokens(text),
            "content_id": safe_id,
            "content_url": content_url or "",
            "path": content_url or "",
        }

    def build_example_documents_from_directory(self, upload_dir: str) -> list[dict[str, object]]:
        """Build example-document payloads for all supported files in a directory.

        Args:
            upload_dir: Directory containing the example source files.

        Returns:
            List of JSON-serializable payloads matching the frontend contract.
        """
        documents: list[dict[str, object]] = []
        try:
            for file_name in sorted(os.listdir(upload_dir)):
                extension = os.path.splitext(file_name)[1].lower()
                if extension not in self.allowed_extensions:
                    continue

                file_path = os.path.join(upload_dir, file_name)
                if not os.path.isfile(file_path):
                    continue

                with open(file_path, "rb") as file_obj:
                    file_bytes = file_obj.read()

                documents.append(self.build_document_payload(file_name, file_bytes))
        except Exception as exc:
            print(f"[EXAMPLE-DOCS] Error scanning uploads dir: {exc}", flush=True)

        return documents

    def build_upload_response(
        self,
        file_name: str,
        file_bytes: bytes,
        content_id: str | None = None,
        content_url: str | None = None,
    ) -> dict[str, object]:
        """Build the upload response payload after extraction.

        Each page/sheet segment is prefixed with a ``[DOC_META ...]`` tag so
        downstream LLMs can identify the source document and page number.

        Args:
            file_name: Uploaded file name.
            file_bytes: Raw uploaded bytes.
            content_id: Optional stable backend document identifier.
            content_url: Optional public URL for preview/download access.

        Returns:
            JSON-serializable payload for ``POST /upload``.
        """
        extension = os.path.splitext(file_name)[1].lower()
        safe_id = content_id or ""
        segments, page_count = self.extract_segments(extension, file_bytes)
        tagged = self.tag_segments(segments, file_name, safe_id)
        extracted_text = "\n\n".join(s for s in tagged if s)
        file_size = len(file_bytes)
        return {
            "filename": file_name,
            "file_type": extension.lstrip("."),
            "file_size": self.format_file_size(file_size),
            "file_size_bytes": file_size,
            "page_count": page_count,
            "content": extracted_text,
            "token_count": self.count_tokens(extracted_text),
            "content_id": safe_id,
            "content_url": content_url or "",
            "path": content_url or "",
        }
