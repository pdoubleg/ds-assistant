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


@router.get(
    "/state/audit-form",
    summary="Get audit form state",
    response_description="Current audit form payload and associated form ID.",
)
async def get_audit_form_state(
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Return the current audit-form payload and its associated `form_id` from
    shared in-memory state.

    Clients poll this endpoint to stay synchronized with server-side form
    changes triggered by agent tool calls.
    """
    return JSONResponse(audit_state_service.get_audit_form_state())


@router.get(
    "/state/runtime",
    summary="Get runtime state",
    response_description="Live runtime status flags for polling clients.",
)
async def get_runtime_state(
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Return live runtime status fields from shared state.

    Includes transient flags such as agent-busy indicators and progress
    metadata that frontends use for real-time UI updates.
    """
    return JSONResponse(audit_state_service.get_runtime_state())


@router.post(
    "/state/claim-session/init",
    summary="Initialize a claim session",
    response_description="Session initialization result with resolved document list.",
    responses={
        500: {"description": "Session initialization or document staging failed."},
    },
)
async def initialize_claim_session(
    body: ClaimSessionInitRequestBody,
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
    text_extraction_service: TextExtractionService = Depends(get_text_extraction_service),
    runtime_storage: RuntimeStorageService = Depends(get_runtime_storage_service),
) -> JSONResponse:
    """Initialize (or reset) the shared claim session and replace the active document list.

    When a `claim_number` is provided the endpoint enters **claim-aware mode**
    (WIP integration — currently starts with an empty document set).  When
    omitted, static example documents are staged automatically.

    Side effects:

    * Clears the runtime temp directory to prevent cross-session file leaks.
    * Resets the Doc Lens service singleton.
    """
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


@router.put(
    "/state/audit-form",
    summary="Update audit form state",
    response_description="Confirmation message with the synchronized form payload.",
    responses={
        400: {"description": "Form payload failed validation."},
    },
)
async def put_audit_form_state(
    body: AuditFormRequestBody,
    form_store: FormStore = Depends(get_form_store),
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Replace the editable audit-form payload in shared state.

    The incoming payload is validated against `FormStore` rules before being
    accepted.  On success the updated state snapshot is returned so the caller
    can reconcile immediately.
    """
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
