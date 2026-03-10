"""Reusable helpers for the top-level AG-UI agent."""

from datetime import datetime
from uuid import uuid4

from domain.audit_state import AuditState
from services.document_mapper import DocumentMapper


def build_doc_payloads_from_state(state: AuditState) -> list[dict[str, object]]:
    """Build lightweight prompt payloads from the shared document state.

    Args:
        state: Shared audit state that contains the current document list.

    Returns:
        Prompt-friendly document payloads with normalized title, content, and type
        fields suitable for workflow and tool consumption.
    """
    mapper = DocumentMapper()
    return mapper.state_documents_to_prompt_payloads(state)


def log_tool_call(
    state: AuditState,
    message: str,
    status: str = "in_progress",
    tool_name: str = "",
) -> None:
    """Append or update an activity-log entry for an AG-UI tool call.

    Args:
        state: Shared audit state to mutate.
        message: Human-readable message shown in the activity log.
        status: Current lifecycle status for the log row.
        tool_name: Tool name associated with the log event.
    """
    print(f"[TOOL] {tool_name}: {message}")
    timestamp = datetime.now().isoformat()

    # Reuse the newest row when the same tool/message transitions states.
    if state.activity_log:
        last_entry = state.activity_log[-1]
        if last_entry.get("message") == message and last_entry.get("tool_name", "") == tool_name:
            last_entry["status"] = status
            last_entry["timestamp"] = timestamp
            return

    state.activity_log.append(
        {
            "id": str(uuid4()),
            "message": message,
            "timestamp": timestamp,
            "status": status,
            "tool_name": tool_name,
        }
    )
