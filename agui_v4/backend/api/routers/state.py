"""Shared in-memory audit state routes."""

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from api.schemas.claim_session import ClaimSessionInitRequestBody
from api.schemas.forms import AuditFormRequestBody
from dependencies import (
    get_audit_state_service,
    get_form_store,
    get_runtime_storage_service,
    get_text_extraction_service,
)
from services.audit_state_service import AuditStateService
from services.form_store import FormStore
from services.runtime_storage import RuntimeStorageService
from services.text_extraction import TextExtractionService
from services.doc_lens_factory import reset_doc_lens_service

router = APIRouter()


@router.get("/state/audit-form")
async def get_audit_form_state(
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Get the current audit form payload and form ID from shared state."""
    return JSONResponse(audit_state_service.get_audit_form_state())


@router.get("/state/runtime")
async def get_runtime_state(
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Get live runtime status fields from shared state for polling clients."""
    return JSONResponse(audit_state_service.get_runtime_state())


@router.post("/state/claim-session/init")
async def initialize_claim_session(
    body: ClaimSessionInitRequestBody,
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
    text_extraction_service: TextExtractionService = Depends(get_text_extraction_service),
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> JSONResponse:
    """Initialize the shared claim session and replace the active documents list."""
    claim_number = body.normalized_claim_number()
    effective_date = body.normalized_effective_date()

    # Start every new claim/local session from a clean temp workspace so
    # session-scoped files never leak across claims.
    runtime_storage.clear_tmp_root()
    reset_doc_lens_service()

    if claim_number:
        # Claim-document fetch remains a WIP integration. For now we only clear
        # the previous session's temp state and start with an empty document set.
        documents: list[dict[str, object]] = []
    else:
        staged_documents = runtime_storage.stage_static_examples(
            text_extraction_service.allowed_extensions
        )
        documents = [
            text_extraction_service.build_document_payload(
                file_name=staged_document.file_name,
                file_bytes=staged_document.file_path.read_bytes(),
                content_id=staged_document.content_id,
                content_url=staged_document.public_url,
            )
            for staged_document in staged_documents
        ]

    return JSONResponse(
        audit_state_service.initialize_claim_session(
            claim_number=claim_number,
            effective_date=effective_date,
            documents=documents,
        )
    )


@router.put("/state/audit-form")
async def put_audit_form_state(
    body: AuditFormRequestBody,
    form_store: FormStore = Depends(get_form_store),
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Update the editable audit form payload in shared state."""
    payload = body.extract_form_payload()
    validation_error = form_store.validate_form_payload(payload)
    if validation_error:
        return JSONResponse({"error": validation_error}, status_code=400)

    state_payload = audit_state_service.sync_audit_form(
        payload, current_form_id=body.current_form_id
    )
    return JSONResponse(
        {
            "message": "Audit form state synchronized.",
            **state_payload,
        }
    )
