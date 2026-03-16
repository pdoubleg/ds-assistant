"""Reusable helpers for the top-level AG-UI agent."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import uuid4

from domain.audit_state import AuditState
from services.document_mapper import DocumentMapper

ActivityStatus = Literal["in_progress", "completed", "error"]


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


class StatusReporter(Protocol):
    """Protocol for reporting nested workflow progress into shared UI state.

    Example:
        reporter.in_progress("Running audit question agent...")
        reporter.completed("Running audit question agent...", progress=90)
    """

    def update(
        self,
        message: str,
        status: ActivityStatus = "in_progress",
        *,
        progress: int | None = None,
    ) -> None:
        """Publish a workflow status update.

        Args:
            message: Human-readable progress label shown in the UI.
            status: Lifecycle state for the progress row.
            progress: Optional progress percentage to merge into shared state.
        """

    def in_progress(self, message: str, *, progress: int | None = None) -> None:
        """Publish an in-progress workflow row."""

    def completed(self, message: str, *, progress: int | None = None) -> None:
        """Publish a completed workflow row."""

    def error(self, message: str, *, progress: int | None = None) -> None:
        """Publish an errored workflow row."""


class NullStatusReporter:
    """No-op reporter used when a workflow does not need UI progress updates."""

    def update(
        self,
        message: str,
        status: ActivityStatus = "in_progress",
        *,
        progress: int | None = None,
    ) -> None:
        """Ignore the status update."""
        del message, status, progress

    def in_progress(self, message: str, *, progress: int | None = None) -> None:
        """Ignore the in-progress update."""
        self.update(message, "in_progress", progress=progress)

    def completed(self, message: str, *, progress: int | None = None) -> None:
        """Ignore the completed update."""
        self.update(message, "completed", progress=progress)

    def error(self, message: str, *, progress: int | None = None) -> None:
        """Ignore the error update."""
        self.update(message, "error", progress=progress)


@dataclass(slots=True)
class StateStatusReporter:
    """Write nested workflow milestones into shared ``AuditState``.

    This reporter is intentionally scoped to internal workflow progress rather than
    top-level AG-UI tool lifecycle events, which already have native tool-call
    visibility in the frontend.

    Attributes:
        state: Shared audit state to mutate.
        source_name: Stable label used to group log rows for a workflow.
    """

    state: AuditState
    source_name: str

    def update(
        self,
        message: str,
        status: ActivityStatus = "in_progress",
        *,
        progress: int | None = None,
    ) -> None:
        """Persist a nested workflow status update into the shared state.

        Args:
            message: Human-readable progress label shown in the UI.
            status: Lifecycle state for the progress row.
            progress: Optional progress percentage to merge into shared state.
        """
        self.state.current_step = message
        if progress is not None:
            # Keep progress monotonic so nested workflows cannot accidentally
            # move the UI backward when callers already advanced the run.
            self.state.progress = max(self.state.progress, progress)
        log_tool_call(self.state, message, status, self.source_name)

    def in_progress(self, message: str, *, progress: int | None = None) -> None:
        """Publish an in-progress workflow row."""
        self.update(message, "in_progress", progress=progress)

    def completed(self, message: str, *, progress: int | None = None) -> None:
        """Publish a completed workflow row."""
        self.update(message, "completed", progress=progress)

    def error(self, message: str, *, progress: int | None = None) -> None:
        """Publish an errored workflow row."""
        self.update(message, "error", progress=progress)


def log_tool_call(
    state: AuditState,
    message: str,
    status: ActivityStatus = "in_progress",
    tool_name: str = "",
) -> None:
    """Append or update an activity-log entry for an AG-UI tool or workflow step.

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
