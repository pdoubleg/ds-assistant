"""Shared dependency providers for the backend application."""

import os

from pydantic_ai.ui import StateDeps

from domain.audit_state import AuditState
from services.audit_state_service import AuditStateService
from services.doc_lens_factory import get_doc_lens_asset_dir, get_doc_lens_service
from services.document_mapper import DocumentMapper
from services.form_store import FormStore
from services.text_extraction import TextExtractionService


BASE_DIR = os.path.dirname(__file__)
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FORMS_DIR = os.path.join(BASE_DIR, "data", "forms")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FORMS_DIR, exist_ok=True)

APP_DEPS = StateDeps(AuditState())

_form_store = FormStore(FORMS_DIR)
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


def get_text_extraction_service() -> TextExtractionService:
    """Return the shared text extraction service."""
    return _text_extraction_service


def get_document_mapper() -> DocumentMapper:
    """Return the shared document mapper service."""
    return _document_mapper


def get_upload_dir() -> str:
    """Return the uploads directory path."""
    return UPLOAD_DIR


def get_forms_dir() -> str:
    """Return the forms directory path."""
    return FORMS_DIR


def get_doc_lens_asset_mount_dir() -> str:
    """Return the public Doc Lens asset directory path."""
    return get_doc_lens_asset_dir(BASE_DIR)


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
    "get_shared_state_deps",
    "get_text_extraction_service",
    "get_upload_dir",
]
