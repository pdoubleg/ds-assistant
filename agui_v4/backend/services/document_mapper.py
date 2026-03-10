"""Shared document normalization helpers for routes, workflows, and agent tools."""

from datetime import datetime, timezone
from typing import Any

from domain.audit_state import AuditState
from models.documents import Document, Documents


class DocumentMapper:
    """Normalize document payloads across AG-UI, REST routes, and workflows."""

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

    def tagging_prompt_documents(self, payloads: list[Any]) -> list[dict[str, str]]:
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
            }
            for payload in payloads
        ]

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
