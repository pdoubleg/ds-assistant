"""Audit-form state and persistence routes."""

from uuid import uuid4

from fastapi import APIRouter, Body, Depends
from starlette.responses import JSONResponse

from api.schemas.forms import AuditFormRequestBody
from services.audit_state_service import AuditStateService
from services.form_store import FormStore
from dependencies import get_audit_state_service, get_form_store

router = APIRouter()


@router.get("/state/audit-form")
async def get_audit_form_state(
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Get the current audit form payload and form ID from shared state."""
    return JSONResponse(audit_state_service.get_audit_form_state())


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

    state_payload = audit_state_service.sync_audit_form(payload, current_form_id=body.current_form_id)
    return JSONResponse(
        {
            "message": "Audit form state synchronized.",
            **state_payload,
        }
    )


@router.post("/forms")
async def save_form(
    body: AuditFormRequestBody | None = Body(default=None),
    form_store: FormStore = Depends(get_form_store),
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Persist an audit form to local JSON storage."""
    state = audit_state_service.state
    payload = body.extract_form_payload() if body else {}
    if not payload:
        payload = state.audit_form_result

    validation_error = form_store.validate_form_payload(payload)
    if validation_error:
        return JSONResponse({"error": validation_error}, status_code=400)

    requested_id = body.id if body else None
    form_id = requested_id or state.current_form_id or str(uuid4())
    record = form_store.save_form(
        form_id=form_id,
        payload=payload,
        title=(body.title if body and body.title else None),
        source_docs=(body.source_docs if body else []),
    )
    audit_state_service.mark_form_saved(form_id, payload)
    return JSONResponse(
        {
            "message": "Form saved.",
            "form_id": form_id,
            "title": record["title"],
            "updated_at": record["updated_at"],
        }
    )


@router.get("/forms")
async def list_forms(form_store: FormStore = Depends(get_form_store)) -> JSONResponse:
    """List all saved forms from local JSON storage."""
    return JSONResponse({"forms": form_store.list_forms()})


@router.get("/forms/all")
async def list_forms_full(form_store: FormStore = Depends(get_form_store)) -> JSONResponse:
    """Return all saved forms with full data for dashboard aggregation."""
    return JSONResponse({"forms": form_store.list_forms_full()})


@router.get("/forms/{form_id}")
async def get_form(
    form_id: str,
    form_store: FormStore = Depends(get_form_store),
) -> JSONResponse:
    """Read one saved form record by ID."""
    try:
        return JSONResponse(form_store.read_form(form_id))
    except FileNotFoundError:
        return JSONResponse({"error": f"Form '{form_id}' not found."}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read form: {exc}"}, status_code=500)


@router.post("/forms/{form_id}/restore")
async def restore_form(
    form_id: str,
    form_store: FormStore = Depends(get_form_store),
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Load a saved form and restore it into shared agent state."""
    try:
        record = form_store.read_form(form_id)
    except FileNotFoundError:
        return JSONResponse({"error": f"Form '{form_id}' not found."}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read form: {exc}"}, status_code=500)

    payload = form_store.to_form_payload(record)
    validation_error = form_store.validate_form_payload(payload)
    if validation_error:
        return JSONResponse({"error": f"Saved form is invalid: {validation_error}"}, status_code=500)

    resolved_form_id = record.get("id", form_id)
    return JSONResponse(audit_state_service.restore_form(resolved_form_id, payload))


@router.delete("/forms/{form_id}")
async def delete_form(
    form_id: str,
    form_store: FormStore = Depends(get_form_store),
    audit_state_service: AuditStateService = Depends(get_audit_state_service),
) -> JSONResponse:
    """Delete a saved form from local JSON storage."""
    try:
        form_store.delete_form(form_id)
    except FileNotFoundError:
        return JSONResponse({"error": f"Form '{form_id}' not found."}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to delete form: {exc}"}, status_code=500)

    audit_state_service.clear_current_form_reference(form_id)
    return JSONResponse({"message": "Form deleted.", "form_id": form_id})
