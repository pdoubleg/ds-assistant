"""Helpers for static and temp-backed document storage."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5


@dataclass(frozen=True)
class StagedDocument:
    """Describe one document staged into the runtime temp area.

    Attributes:
        content_id: Stable backend-generated identifier for the staged file.
        file_name: Original display filename shown in the UI.
        file_path: Absolute filesystem path to the staged file.
        public_url: Frontend-consumable URL for preview/download access.
    """

    content_id: str
    file_name: str
    file_path: Path
    public_url: str


class RuntimeStorageService:
    """Manage static example files and temp runtime document storage.

    The backend keeps repository-managed sample files in ``uploads`` while
    placing session-scoped working files in ``tmp``. This helper centralizes
    the path layout so API routes do not need to know about directory details.

    Args:
        base_dir: Absolute path to the backend root directory.
    """

    static_upload_dir_name = "uploads"
    tmp_root_dir_name = "tmp"
    runtime_documents_dir_name = "documents"
    doc_lens_cache_dir_name = "doc_lens_cache"
    public_documents_mount = "/document-files"

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize runtime-storage paths from the backend base directory."""
        self.base_dir = Path(base_dir)
        self.static_upload_dir = self.base_dir / self.static_upload_dir_name
        self.tmp_root_dir = self.base_dir / self.tmp_root_dir_name
        self.runtime_documents_dir = self.tmp_root_dir / self.runtime_documents_dir_name
        self.doc_lens_cache_dir = self.tmp_root_dir / self.doc_lens_cache_dir_name
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create the expected storage directories if they do not exist."""
        self.static_upload_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_root_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_documents_dir.mkdir(parents=True, exist_ok=True)
        self.doc_lens_cache_dir.mkdir(parents=True, exist_ok=True)

    def build_temp_file_name(self, content_id: str, file_name: str) -> str:
        """Build the on-disk filename for a staged temp document.

        Args:
            content_id: Backend-generated stable identifier.
            file_name: Original display filename.

        Returns:
            Temp-safe filename using the original extension.
        """
        suffix = Path(file_name).suffix.lower()
        return f"{content_id}{suffix}"

    def build_public_document_url(self, content_id: str, file_name: str) -> str:
        """Return the public URL for a staged document.

        Args:
            content_id: Backend-generated stable identifier.
            file_name: Original display filename.

        Returns:
            URL mounted by FastAPI's static-files handler.
        """
        return f"{self.public_documents_mount}/{self.build_temp_file_name(content_id, file_name)}"

    def stage_bytes(
        self,
        file_name: str,
        file_bytes: bytes,
        content_id: str | None = None,
    ) -> StagedDocument:
        """Write raw bytes into the runtime temp document directory.

        Args:
            file_name: Original display filename from the client or static file.
            file_bytes: Raw file content to persist.
            content_id: Optional explicit document identifier.

        Returns:
            Metadata describing the staged file.
        """
        resolved_content_id = content_id or str(uuid4())
        out_path = self.runtime_documents_dir / self.build_temp_file_name(
            resolved_content_id, file_name
        )

        # Keep temp runtime documents isolated under one directory so the app
        # can clear them without touching static repository-managed samples.
        out_path.write_bytes(file_bytes)
        return StagedDocument(
            content_id=resolved_content_id,
            file_name=file_name,
            file_path=out_path,
            public_url=self.build_public_document_url(resolved_content_id, file_name),
        )

    def resolve_staged_document_path(
        self,
        content_id: str,
        file_name: str,
    ) -> Path | None:
        """Resolve the absolute path for a previously staged document.

        Args:
            content_id: Backend-generated document identifier.
            file_name: Original display filename.

        Returns:
            The staged document path when it exists, otherwise ``None``.
        """
        candidate_path = self.runtime_documents_dir / self.build_temp_file_name(
            content_id, file_name
        )
        return candidate_path if candidate_path.is_file() else None

    def stage_static_examples(
        self,
        allowed_extensions: set[str],
    ) -> list[StagedDocument]:
        """Copy supported static example files into the runtime temp area.

        Args:
            allowed_extensions: Supported file extensions, including the dot.

        Returns:
            Metadata for each staged example document.
        """
        staged_documents: list[StagedDocument] = []
        for static_path in sorted(self.static_upload_dir.iterdir(), key=lambda path: path.name):
            if not static_path.is_file():
                continue
            if static_path.suffix.lower() not in allowed_extensions:
                continue
            staged_documents.append(
                self.stage_bytes(
                    file_name=static_path.name,
                    file_bytes=static_path.read_bytes(),
                    content_id=str(uuid5(NAMESPACE_URL, f"example::{static_path.name}")),
                )
            )
        return staged_documents

    def clear_runtime_documents(self) -> None:
        """Delete all staged runtime documents while keeping the directory."""
        self._clear_directory_contents(self.runtime_documents_dir)
        self.ensure_dirs()

    def clear_tmp_root(self) -> None:
        """Delete all temp runtime contents while preserving the root layout."""
        self._clear_directory_contents(self.tmp_root_dir)
        self.ensure_dirs()

    def _clear_directory_contents(self, directory: Path) -> None:
        """Delete every file and directory within ``directory``.

        Args:
            directory: Directory whose contents should be removed.
        """
        if not directory.exists():
            return

        # Delete children in-place so any static mounts still point at the same
        # parent directory after cleanup.
        for child_path in directory.iterdir():
            if child_path.is_dir():
                shutil.rmtree(child_path, ignore_errors=True)
            else:
                child_path.unlink(missing_ok=True)
