"""Shared document normalization helpers for routes, workflows, and agent tools."""

from datetime import datetime, timezone
from typing import Any

from pydantic_ai import BinaryContent
from pydantic_ai.messages import UserContent

from domain.audit_state import AuditState
from models.documents import Document, Documents
from services.runtime_storage import RuntimeStorageService


class DocumentMapper:
    """Normalize document payloads across AG-UI, REST routes, and workflows."""

    IMAGE_MIME_TYPES: set[str] = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/webp",
    }

    def state_documents_to_prompt_payloads(self, state: AuditState) -> list[dict[str, Any]]:
        """Map shared-state documents into lightweight prompt payloads.

        Args:
            state: Current shared audit state.

        Returns:
            List of prompt-friendly dictionaries.
        """
        payloads: list[dict[str, Any]] = []
        for document in state.documents:
            payloads.append(
                {
                    "title": document.get("file_name", document.get("content_url", "Untitled")),
                    "content": document.get("content", document.get("text", "")),
                    "file_type": document.get("mime_type", "unknown"),
                    "document_type": document.get("document_type", ""),
                }
            )
        return payloads

    def combine_documents(self, document_contents: list[dict[str, Any]]) -> tuple[str, int]:
        """Concatenate normalized document payloads into one prompt string.

        Args:
            document_contents: Lightweight prompt payloads.

        Returns:
            Tuple of combined text and document count.
        """
        parts: list[str] = []
        for document in document_contents:
            title = document.get("title", "Untitled")
            content = document.get("content", "")
            file_type = document.get("file_type", document.get("mime_type", "unknown"))
            document_type = document.get("document_type", "")
            header = f"--- Document: {title} ({file_type}"
            if document_type:
                header += f", type={document_type}"
            header += f") ---\n{content}"
            parts.append(header)
        return "\n\n".join(parts), len(document_contents)

    def build_search_sort_documents(self, payloads: list[Any]) -> tuple[Documents, dict[str, str]]:
        """Map request payloads into the `Documents` domain model.

        Args:
            payloads: Request payload models or dict-like objects exposing the
                search/sort document fields.

        Returns:
            Tuple of `Documents` plus a `content_id -> file_name` mapping.
        """
        doc_models: list[Document] = []
        content_id_to_file_name: dict[str, str] = {}

        for payload in payloads:
            content_id = self._get_field(payload, "content_id")
            file_name = self._get_field(payload, "file_name")
            content_id_to_file_name[content_id] = file_name
            doc_models.append(
                Document(
                    claimNumber=self._get_field(payload, "claim_number", ""),
                    contentId=content_id,
                    mimeType=self._get_field(payload, "mime_type", "unknown"),
                    contentURL=self._get_field(payload, "content_url", "") or file_name,
                    domain=self._normalize_domain(self._get_field(payload, "domain", "claim")),
                    documentType=self._blank_to_none(self._get_field(payload, "document_type", "")),
                    documentSubType=self._blank_to_none(
                        self._get_field(payload, "document_sub_type", "")
                    ),
                    documentDescription=self._blank_to_none(
                        self._get_field(payload, "document_description", "")
                    ),
                    createDate=self._parse_or_now(self._get_field(payload, "create_date", "")),
                    sourceSystem=self._blank_to_none(self._get_field(payload, "source_system", "")),
                    companyName=self._blank_to_none(self._get_field(payload, "company_name", "")),
                    text=self._get_field(payload, "content", ""),
                )
            )

        return Documents(documents=doc_models), content_id_to_file_name

    def tagging_prompt_documents(self, payloads: list[Any]) -> list[dict[str, Any]]:
        """Map tagging request documents into prompt payload dicts.

        Args:
            payloads: Document payload models or dictionaries.

        Returns:
            List of prompt payload dictionaries.
        """
        return [
            {
                "file_name": self._get_field(payload, "file_name"),
                "content": self._get_field(payload, "content", ""),
                "document_type": self._get_field(payload, "document_type", ""),
                "metadata_string": self.build_document_metadata_string(payload),
                "is_image": self.is_image_document(payload),
            }
            for payload in payloads
        ]

    def is_image_document(self, payload: Any) -> bool:
        """Return whether the payload represents an image document."""
        mime_type = str(self._get_field(payload, "mime_type", "unknown")).lower()
        return mime_type in self.IMAGE_MIME_TYPES

    def build_document_metadata_string(self, payload: Any) -> str:
        """Build a compact metadata block for prompt use.

        Args:
            payload: Document payload model or dictionary.

        Returns:
            Multi-line metadata string suitable for prompt injection.
        """
        lines = [
            f"- File name: {self._get_field(payload, 'file_name', 'Untitled')}",
            f"- Content ID: {self._get_field(payload, 'content_id', 'N/A') or 'N/A'}",
            f"- MIME type: {self._get_field(payload, 'mime_type', 'unknown') or 'unknown'}",
            f"- Document type: {self._get_field(payload, 'document_type', '') or 'N/A'}",
        ]
        document_description = self._get_field(payload, "document_description", "")
        if document_description:
            lines.append(f"- Description: {document_description}")
        content_url = self._get_field(payload, "content_url", "")
        if content_url:
            lines.append(f"- Content URL: {content_url}")
        return "\n".join(lines)

    def build_image_prompt_parts(
        self,
        payloads: list[Any],
        runtime_storage: RuntimeStorageService | None,
    ) -> list[UserContent]:
        """Build prompt parts for image documents using metadata + BinaryContent.

        Args:
            payloads: Document payload models or dictionaries.
            runtime_storage: Service used to resolve staged image bytes.

        Returns:
            Ordered prompt parts suitable for `Agent.run(...)`.
        """
        if runtime_storage is None:
            return []

        prompt_parts: list[UserContent] = []
        for payload in payloads:
            binary_part = self.build_image_binary_content(payload, runtime_storage)
            if binary_part is None:
                continue
            prompt_parts.append(
                "\n".join(
                    [
                        "## Attached Image Document",
                        self.build_document_metadata_string(payload),
                    ]
                )
            )
            prompt_parts.append(binary_part)
        return prompt_parts

    def build_image_binary_content(
        self,
        payload: Any,
        runtime_storage: RuntimeStorageService,
    ) -> BinaryContent | None:
        """Resolve staged image bytes into a pydantic-ai `BinaryContent`.

        Args:
            payload: Document payload model or dictionary.
            runtime_storage: Runtime storage resolver.

        Returns:
            `BinaryContent` when the document is an image with staged bytes,
            otherwise `None`.
        """
        if not self.is_image_document(payload):
            return None

        content_id = self._get_field(payload, "content_id", "")
        file_name = self._get_field(payload, "file_name", "")
        mime_type = self._get_field(payload, "mime_type", "application/octet-stream")
        if not content_id or not file_name:
            return None

        staged_path = runtime_storage.resolve_staged_document_path(content_id, file_name)
        if staged_path is None:
            return None

        return BinaryContent(
            data=staged_path.read_bytes(),
            media_type=mime_type,
            identifier=file_name,
        )

    def _get_field(self, payload: Any, field_name: str, default: Any = None) -> Any:
        """Read a field from a pydantic model or dictionary."""
        if isinstance(payload, dict):
            return payload.get(field_name, default)
        return getattr(payload, field_name, default)

    def _blank_to_none(self, value: str | None) -> str | None:
        """Normalize blank strings to `None`."""
        return value or None

    def _normalize_domain(self, value: str) -> str:
        """Normalize the document domain for the `Document` model."""
        return value if value in ("claim", "policy") else "claim"

    def _parse_or_now(self, value: str) -> datetime:
        """Parse an ISO timestamp or fall back to the current UTC time."""
        if not value:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
