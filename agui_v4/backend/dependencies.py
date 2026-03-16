"""Shared dependency providers for the backend application."""

from pydantic_ai.ui import StateDeps

from domain.audit_state import AuditState
from services.audit_state_service import AuditStateService
from services.doc_lens_factory import get_doc_lens_asset_dir, get_doc_lens_service
from services.document_mapper import DocumentMapper
from services.form_store import FormStore
from services.runtime_storage import RuntimeStorageService
from services.text_extraction import TextExtractionService


from os import getenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
BACKEND_PORT = int(getenv("BACKEND_PORT", "8001"))
UPLOAD_DIR = BASE_DIR / "uploads"
FORMS_DIR = BASE_DIR / "data" / "forms"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FORMS_DIR.mkdir(parents=True, exist_ok=True)

APP_DEPS = StateDeps(AuditState())

_runtime_storage = RuntimeStorageService(BASE_DIR)
_form_store = FormStore(str(FORMS_DIR))
_text_extraction_service = TextExtractionService()
_document_mapper = DocumentMapper()


def get_backend_port() -> int:
    """Return the configured backend port."""
    return BACKEND_PORT


def get_shared_state_deps() -> StateDeps[AuditState]:
    """Return the shared AG-UI state dependency container."""
    return APP_DEPS


def get_audit_state_service() -> AuditStateService:
    """Return a service wrapper over the shared audit state."""
    return AuditStateService(APP_DEPS.state)


def get_form_store() -> FormStore:
    """Return the shared form persistence service."""
    return _form_store


def get_runtime_storage_service() -> RuntimeStorageService:
    """Return the shared runtime storage service."""
    return _runtime_storage


def get_text_extraction_service() -> TextExtractionService:
    """Return the shared text extraction service."""
    return _text_extraction_service


def get_document_mapper() -> DocumentMapper:
    """Return the shared document mapper service."""
    return _document_mapper


def get_upload_dir() -> str:
    """Return the static example uploads directory path."""
    return str(UPLOAD_DIR)


def get_runtime_documents_dir() -> str:
    """Return the temp runtime documents directory path."""
    return str(_runtime_storage.runtime_documents_dir)


def get_runtime_documents_mount_dir() -> str:
    """Return the temp runtime documents static mount directory."""
    return str(_runtime_storage.runtime_documents_dir)


def get_forms_dir() -> str:
    """Return the forms directory path."""
    return str(FORMS_DIR)


def get_doc_lens_asset_mount_dir() -> str:
    """Return the public Doc Lens asset directory path."""
    return get_doc_lens_asset_dir(str(BASE_DIR), runtime_storage=_runtime_storage)


__all__ = [
    "APP_DEPS",
    "BACKEND_PORT",
    "BASE_DIR",
    "FORMS_DIR",
    "UPLOAD_DIR",
    "get_audit_state_service",
    "get_backend_port",
    "get_doc_lens_asset_mount_dir",
    "get_doc_lens_service",
    "get_document_mapper",
    "get_form_store",
    "get_forms_dir",
    "get_runtime_documents_dir",
    "get_runtime_documents_mount_dir",
    "get_runtime_storage_service",
    "get_shared_state_deps",
    "get_text_extraction_service",
    "get_upload_dir",
]
