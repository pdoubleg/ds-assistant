"""Stateful hackathon Monty REPL service implementation."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .filesystem import (
    DEFAULT_HOST_WORKSPACE,
    HostWorkspaceOSAccess,
    VIRTUAL_WORKSPACE_ROOT,
)
from .execution import FreeformCodeError
from .interpreter import MontyReplInterpreter
from .base import SafeObjectStore
from .privacy import sanitize_exception, summarize_stdout
from .registry import build_default_registry


class HelpResponse(BaseModel):
    """Structured response for the minimal `help` tool."""

    session_id: str = Field(description="Session identifier for the active REPL.")
    view: Literal["overview", "collection", "tool", "not_found"] = Field(
        description="Active help payload variant."
    )
    name: str | None = Field(
        default=None,
        description="Optional collection or tool name used to build the response.",
    )
    workspace_root: str = Field(description="Sandbox workspace path.")
    collections: list[dict[str, Any]] | None = Field(default=None)
    collection: dict[str, Any] | None = Field(default=None)
    functions: list[dict[str, Any]] | None = Field(default=None)
    function: dict[str, Any] | None = Field(default=None)
    notes: list[str] | None = Field(default=None)
    workflow: list[str] | None = Field(default=None)
    limitations: list[str] | None = Field(default=None)
    error: str | None = Field(default=None)
    available_collections: list[str] | None = Field(default=None)
    available_functions: list[str] | None = Field(default=None)


class ExecutionRecord(BaseModel):
    """Structured record for one `execute` call."""

    session_id: str = Field(description="Session identifier for the active REPL.")
    execution_id: int = Field(description="Monotonic execution identifier.")
    executed_at: str = Field(description="UTC timestamp for the execution.")
    code: str = Field(description="Submitted code.")
    status: str = Field(description="Execution status: success or error.")
    summary: str = Field(description="User-facing execution summary.")
    stdout: dict[str, Any] = Field(description="Privacy-safe stdout summary.")
    persisted_variables: list[str] = Field(default_factory=list)
    persisted_value_summaries: dict[str, Any] = Field(default_factory=dict)
    last_expression_summary: Any | None = Field(default=None)
    persistence_failures: list[dict[str, str]] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = Field(default=None)


class MinimalMontyPythonREPL:
    """Stateful Monty REPL specialized for hackathon modeling workflows."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        type_check: bool = False,
    ) -> None:
        """Initialize the REPL service.

        Args:
            workspace_root: Optional host workspace root.
            type_check: Whether to enable Monty type checking.
        """
        self.workspace_root = (workspace_root or DEFAULT_HOST_WORKSPACE).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        self.os_access = HostWorkspaceOSAccess(self.workspace_root)
        self.object_store = SafeObjectStore()
        self.registry = build_default_registry(self.os_access, self.object_store)
        self.interpreter = MontyReplInterpreter(
            tools=self.registry.exported_tools(),
            type_check=type_check,
            os_access=self.os_access,
        )
        self.session_id = uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._execution_counter = 0
        self._pending_results: list[ExecutionRecord] = []

    def help(self, name: str | None = None) -> dict[str, Any]:
        """Describe available safe modeling helpers.

        Args:
            name: Optional collection or tool name.

        Returns:
            Structured help payload.
        """
        if name is None:
            return self._help_overview_payload()

        collection = self.registry.get_collection(name)
        if collection is not None:
            return self._help_collection_payload(name, collection.to_help_dict())

        function = self.registry.get(name)
        if function is not None:
            return self._help_tool_payload(name, function.to_help_dict(detailed=True))

        return self._help_not_found_payload(name)

    def _base_help_response(
        self,
        *,
        view: Literal["overview", "collection", "tool", "not_found"],
        name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a shared help response payload."""
        response = HelpResponse(
            session_id=self.session_id,
            view=view,
            name=name,
            workspace_root=str(VIRTUAL_WORKSPACE_ROOT),
            **kwargs,
        )
        return response.model_dump(exclude_none=True)

    def _help_overview_payload(self) -> dict[str, Any]:
        """Build the high-level collection overview payload."""
        return self._base_help_response(
            view="overview",
            collections=[item.to_help_dict() for item in self.registry.collections()],
            notes=[
                "This hackathon REPL never returns raw training rows or categorical examples.",
                "Use schema summaries, safe plots, feature screening, and model artifacts instead of row previews.",
                "Relative paths resolve under /workspace.",
                "Execute runs in a restricted Python runtime, so standard builtins, filesystem calls, compilation, and introspection behaviors may differ from a local interpreter.",
                "Execute returns privacy-safe helper summaries directly and suppresses raw stdout.",
                "For file creation, file reads, and persistence inside /workspace, prefer helper APIs such as write_workspace_text(...), write_workspace_json(...), read_workspace_text(...), and read_workspace_json(...).",
                "Use results() for buffered execution history when you want to review multiple prior steps together.",
            ],
            workflow=[
                "Call help() to discover safe modeling collections.",
                "Call help('<collection-name>') to inspect the tools in that collection.",
                "Call help('<tool-name>') before using an unfamiliar helper in execute(...).",
                "Prefer predefined helpers for inspection and modeling, and use freeform code mainly to create new dataframe handles from transformations or slices.",
                "Prefer workspace helpers for file IO and artifact persistence instead of direct open(...), Path.write_text(...), or similar stdlib file APIs inside execute(...).",
                "End execute(...) with a helper call when you want an immediate safe summary, or assign the result when you need to reuse it later.",
            ],
            limitations=[
                "Raw dataframe previews and raw workspace text reads are intentionally unavailable.",
                "All modeling uses LightGBM with native categorical handling.",
                "Freeform stdout and exception details are privacy-sanitized.",
                "Sandbox failures may surface as generic sanitized execution errors, so switch to a provided helper first when common Python operations fail unexpectedly.",
            ],
        )

    def _help_collection_payload(
        self,
        name: str,
        collection: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the collection-specific help payload."""
        return self._base_help_response(
            view="collection",
            name=name,
            collection=collection,
            functions=[
                entry.to_help_dict() for entry in self.registry.entries(collection=name)
            ],
            notes=[
                f"You are viewing the {name!r} collection.",
                "All tool returns are privacy-safe by design.",
                "Prefer these predefined helpers over freeform code whenever a tool already covers the task.",
            ]
            + (
                [
                    "Use the workspace collection for file reads and writes inside /workspace instead of direct open(...), Path.write_text(...), Path.read_text(...), or similar stdlib file operations."
                ]
                if name == "workspace"
                else []
            ),
        )

    def _help_tool_payload(self, name: str, function: dict[str, Any]) -> dict[str, Any]:
        """Build the detailed single-tool help payload."""
        return self._base_help_response(
            view="tool",
            name=name,
            function=function,
            notes=[
                f"Call `{name}(...)` directly inside execute(...) code.",
                "Leave a helper call as the final expression when you want execute(...) to surface its safe summary immediately.",
                "If a normal Python operation fails with a sanitized error, assume sandbox restrictions first and switch to a provided helper.",
            ]
            + (
                [
                    "For file IO, prefer this helper over direct open(...), Path.write_text(...), Path.read_text(...), or manual JSON/text persistence."
                ]
                if function.get("collection") == "workspace"
                else []
            ),
        )

    def _help_not_found_payload(self, name: str) -> dict[str, Any]:
        """Build the invalid-name help payload."""
        return self._base_help_response(
            view="not_found",
            name=name,
            error=(
                f"No collection or function named {name!r} is registered. "
                "Choose one of the valid names below."
            ),
            available_collections=[
                collection.name for collection in self.registry.collections()
            ],
            available_functions=[entry.name for entry in self.registry.entries()],
        )

    def _build_summary(
        self,
        *,
        success: bool,
        stdout_summary: dict[str, Any],
        artifacts: list[str],
        persisted_variables: list[str],
        persisted_value_summaries: dict[str, Any],
        last_expression_summary: Any | None,
        persistence_failures: list[dict[str, str]],
        error: dict[str, Any] | None,
    ) -> str:
        """Build a concise execution summary."""
        if not success and error is not None:
            return f"Execution failed with {error['error_type']}."

        parts = ["Execution succeeded"]
        if stdout_summary.get("suppressed"):
            parts.append(
                f"captured {stdout_summary.get('line_count', 0)} stdout line(s) and suppressed them for privacy"
            )
        if artifacts:
            parts.append(f"produced {len(artifacts)} artifact(s)")
        if persisted_variables:
            parts.append(f"persisted {len(persisted_variables)} variable(s)")
        if persisted_value_summaries:
            parts.append(
                f"surfaced {len(persisted_value_summaries)} persisted value "
                f"{'summary' if len(persisted_value_summaries) == 1 else 'summaries'}"
            )
        if last_expression_summary is not None:
            parts.append("surfaced the final expression summary")
        if persistence_failures:
            parts.append(f"failed to persist {len(persistence_failures)} variable(s)")
        return "; ".join(parts) + "."

    async def execute(self, code: str) -> dict[str, Any]:
        """Execute Monty code and buffer a privacy-safe result.

        Args:
            code: Python source to execute.

        Returns:
            Structured execute payload.
        """
        self.os_access.begin_artifact_tracking()
        raw_stdout = ""
        persisted_variables: list[str] = []
        persisted_value_summaries: dict[str, Any] = {}
        last_expression_summary: Any | None = None
        persistence_failures: list[dict[str, str]] = []
        error_payload: dict[str, Any] | None = None
        status = "success"

        try:
            interpreter_result = await self.interpreter.execute(code)
            raw_stdout = interpreter_result.stdout
            persisted_variables = interpreter_result.persisted_names
            persisted_value_summaries = interpreter_result.persisted_value_summaries
            last_expression_summary = interpreter_result.last_expression_summary
            persistence_failures = interpreter_result.persistence_failures
        except (
            SyntaxError,
            PermissionError,
            FileNotFoundError,
            FreeformCodeError,
        ) as exc:
            status = "error"
            error_payload = sanitize_exception(
                exc,
                traceback_text=traceback.format_exc(),
            )
        except Exception as exc:  # pragma: no cover
            status = "error"
            error_payload = sanitize_exception(
                exc,
                traceback_text=traceback.format_exc(),
            )

        stdout_summary = summarize_stdout(raw_stdout)
        artifacts = self.os_access.consume_tracked_artifacts()
        summary = self._build_summary(
            success=status == "success",
            stdout_summary=stdout_summary,
            artifacts=artifacts,
            persisted_variables=persisted_variables,
            persisted_value_summaries=persisted_value_summaries,
            last_expression_summary=last_expression_summary,
            persistence_failures=persistence_failures,
            error=error_payload,
        )

        self._execution_counter += 1
        record = ExecutionRecord(
            session_id=self.session_id,
            execution_id=self._execution_counter,
            executed_at=datetime.now(timezone.utc).isoformat(),
            code=code,
            status=status,
            summary=summary,
            stdout=stdout_summary,
            persisted_variables=persisted_variables,
            persisted_value_summaries=persisted_value_summaries,
            last_expression_summary=last_expression_summary,
            persistence_failures=persistence_failures,
            artifacts=artifacts,
            error=error_payload,
        )
        self._pending_results.append(record)

        return {
            "session_id": self.session_id,
            "execution_id": record.execution_id,
            "status": status,
            "summary": summary,
            "artifacts": artifacts,
            "persisted_variables": persisted_variables,
            "persisted_value_summaries": persisted_value_summaries,
            "last_expression_summary": last_expression_summary,
            "persistence_failures": persistence_failures,
            "pending_result_count": len(self._pending_results),
            "error": error_payload,
        }

    def results(self) -> dict[str, Any]:
        """Return and clear buffered privacy-safe execution output."""
        if not self._pending_results:
            return {
                "status": "empty",
                "summary": "No new execution output since the last results call.",
                "session_id": self.session_id,
                "started_at": self.started_at,
                "workspace_root": str(VIRTUAL_WORKSPACE_ROOT),
            }

        executions = [record.model_dump() for record in self._pending_results]
        chunks: list[str] = []
        for record in self._pending_results:
            chunk_lines = [f"Execution {record.execution_id} [{record.status}]"]
            if record.stdout.get("suppressed"):
                chunk_lines.append(
                    "STDOUT: suppressed for privacy "
                    f"({record.stdout.get('line_count', 0)} line(s))."
                )
            if record.last_expression_summary is not None:
                chunk_lines.append(
                    f"Last expression: {record.last_expression_summary!r}"
                )
            if record.persisted_value_summaries:
                chunk_lines.append(
                    f"Persisted values: {record.persisted_value_summaries!r}"
                )
            if record.error:
                chunk_lines.append(f"ERROR: {record.error['error_type']}")
            if record.persistence_failures:
                chunk_lines.append(
                    "Persistence warnings: "
                    + ", ".join(
                        f"{item['name']} ({item['error']})"
                        for item in record.persistence_failures
                    )
                )
            if record.artifacts:
                chunk_lines.append(f"Artifacts: {', '.join(record.artifacts)}")
            chunks.append("\n".join(chunk_lines).strip())

        response = {
            "status": "ok",
            "summary": f"Returned {len(self._pending_results)} buffered execution result(s).",
            "session_id": self.session_id,
            "started_at": self.started_at,
            "workspace_root": str(VIRTUAL_WORKSPACE_ROOT),
            "executions": executions,
            "combined_output": "\n\n".join(chunks),
        }
        self._pending_results = []
        return response


__all__ = ["MinimalMontyPythonREPL"]
