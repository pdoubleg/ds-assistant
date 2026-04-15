"""Stateful Monty REPL service implementation."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.rlm.types import CodeExecutionError

from .filesystem import (
    DEFAULT_HOST_WORKSPACE,
    HostWorkspaceOSAccess,
    VIRTUAL_WORKSPACE_ROOT,
)
from .interpreter import MontyReplInterpreter
from .registry import FunctionRegistry, ObjectStore, build_default_registry


class HelpResponse(BaseModel):
    """Structured response for the MCP ``help`` tool."""

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
    supported_native_imports: list[str] | None = Field(default=None)
    limitations: list[str] | None = Field(default=None)
    error: str | None = Field(default=None)
    available_collections: list[str] | None = Field(default=None)
    available_functions: list[str] | None = Field(default=None)


class ExecutionRecord(BaseModel):
    """Structured record for a single execute call."""

    session_id: str = Field(description="Session identifier for the active REPL.")
    execution_id: int = Field(description="Monotonic execution identifier.")
    executed_at: str = Field(description="UTC timestamp for the execution.")
    code: str = Field(description="Submitted code.")
    status: str = Field(description="success or error.")
    summary: str = Field(description="User-friendly execution summary.")
    stdout: str = Field(default="", description="Captured standard output.")
    persisted_variables: list[str] = Field(
        default_factory=list,
        description="Top-level variable names persisted for later executions.",
    )
    persistence_failures: list[dict[str, str]] = Field(
        default_factory=list,
        description="Top-level variables that could not be persisted and why.",
    )
    artifacts: list[str] = Field(
        default_factory=list,
        description="New or modified workspace files.",
    )
    error: str | None = Field(default=None, description="Error message, if any.")
    traceback: str | None = Field(
        default=None, description="Formatted traceback, if any."
    )


class MontyPythonREPL:
    """Stateful Monty REPL service."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        registry: FunctionRegistry | None = None,
        type_check: bool = False,
    ) -> None:
        """Initialize the REPL service."""
        self.workspace_root = (workspace_root or DEFAULT_HOST_WORKSPACE).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        self.os_access = HostWorkspaceOSAccess(self.workspace_root)
        self.object_store = ObjectStore()
        self.registry = registry or build_default_registry(
            self.os_access, self.object_store
        )
        self.interpreter = MontyReplInterpreter(
            tools=self.registry.exported_tools(),
            type_check=type_check,
            os_access=self.os_access,
        )
        self.session_id = uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._execution_counter = 0
        self._pending_results: list[ExecutionRecord] = []

    def help(
        self,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Describe available sandbox functions."""
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
        """Build a shared help response payload.

        Args:
            view: Active response variant.
            name: Optional collection or tool name being resolved.
            **kwargs: View-specific payload fields.

        Returns:
            dict[str, Any]: JSON-friendly help response with empty fields omitted.
        """
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
                "Use the collections list below to choose the right capability area before executing code.",
                "Relative paths resolve under /workspace.",
                "Native imports inside execute(...) are intentionally limited to a small built-in set.",
                "When you need broader dataframe-oriented DS library usage over a stored pandas dataframe, inspect and use `run_dataframe_code`.",
                "`run_dataframe_code` still returns a single final dataframe, so add print statements inside the submitted code when you want intermediate diagnostics.",
                "When you pass nested freeform source into helpers, prefer assigning the inner code to a named multiline variable first.",
                "Avoid escape-heavy inline strings such as `print(f'\\n...')`; prefer separate `print()` calls, or double-escape as `\\\\n` when a literal escape must survive outer parsing.",
                "The results tool returns and clears everything accumulated since the previous results call.",
            ],
            workflow=[
                "Call help() to discover task-focused collections.",
                "Call help('<collection-name>') to inspect the tools in that collection.",
                "Call help('<tool-name>') to inspect exact arguments and return details before writing execute(...) code.",
            ],
            supported_native_imports=["datetime", "json", "math", "re"],
            limitations=[
                "Monty supports expression-oriented REPL code and helper calls, but does not support defining classes.",
                "Keep all files inside /workspace.",
            ],
        )

    def _help_collection_payload(
        self,
        name: str,
        collection: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the collection-specific help payload.

        Args:
            name: Collection name supplied by the caller.
            collection: Serialized collection metadata.

        Returns:
            dict[str, Any]: Collection view payload.
        """
        return self._base_help_response(
            view="collection",
            name=name,
            collection=collection,
            functions=[
                entry.to_help_dict() for entry in self.registry.entries(collection=name)
            ],
            notes=[
                f"You are viewing the {name!r} collection.",
                "Call help('<tool-name>') for exact argument and return details before using an unfamiliar helper in execute(...).",
            ],
        )

    def _help_tool_payload(self, name: str, function: dict[str, Any]) -> dict[str, Any]:
        """Build the detailed single-tool help payload.

        Args:
            name: Tool name supplied by the caller.
            function: Detailed serialized tool metadata.

        Returns:
            dict[str, Any]: Tool view payload.
        """
        return self._base_help_response(
            view="tool",
            name=name,
            function=function,
            notes=[
                f"Call `{name}(...)` directly inside execute(...) code.",
            ],
        )

    def _help_not_found_payload(self, name: str) -> dict[str, Any]:
        """Build the invalid-name help payload.

        Args:
            name: Unknown lookup target supplied by the caller.

        Returns:
            dict[str, Any]: Not-found help payload with valid alternatives.
        """
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
        stdout: str,
        artifacts: list[str],
        persisted_variables: list[str],
        persistence_failures: list[dict[str, str]],
        error: str | None,
    ) -> str:
        """Build a concise user-facing execution summary."""
        if not success:
            return f"Execution failed: {error}"

        parts = ["Execution succeeded"]
        if stdout:
            parts.append(f"captured {len(stdout.splitlines())} stdout line(s)")
        if artifacts:
            parts.append(f"produced {len(artifacts)} artifact(s)")
        if persisted_variables:
            parts.append(f"persisted {len(persisted_variables)} variable(s)")
        if persistence_failures:
            parts.append(f"failed to persist {len(persistence_failures)} variable(s)")
        return "; ".join(parts) + "."

    async def execute(self, code: str) -> dict[str, Any]:
        """Execute Monty sandbox code and buffer the resulting output."""
        self.os_access.begin_artifact_tracking()
        stdout = ""
        persisted_variables: list[str] = []
        persistence_failures: list[dict[str, str]] = []
        error: str | None = None
        formatted_traceback: str | None = None
        status = "success"

        try:
            interpreter_result = await self.interpreter.execute(code)
            stdout = interpreter_result.stdout
            persisted_variables = interpreter_result.persisted_names
            persistence_failures = interpreter_result.persistence_failures
        except (
            CodeExecutionError,
            SyntaxError,
            PermissionError,
            FileNotFoundError,
        ) as exc:
            status = "error"
            error = str(exc)
            formatted_traceback = traceback.format_exc()
        except Exception as exc:  # pragma: no cover
            status = "error"
            error = f"Unexpected execution failure: {exc}"
            formatted_traceback = traceback.format_exc()

        artifacts = self.os_access.consume_tracked_artifacts()
        summary = self._build_summary(
            success=status == "success",
            stdout=stdout,
            artifacts=artifacts,
            persisted_variables=persisted_variables,
            persistence_failures=persistence_failures,
            error=error,
        )

        self._execution_counter += 1
        record = ExecutionRecord(
            session_id=self.session_id,
            execution_id=self._execution_counter,
            executed_at=datetime.now(timezone.utc).isoformat(),
            code=code,
            status=status,
            summary=summary,
            stdout=stdout,
            persisted_variables=persisted_variables,
            persistence_failures=persistence_failures,
            artifacts=artifacts,
            error=error,
            traceback=formatted_traceback,
        )
        self._pending_results.append(record)

        return {
            "session_id": self.session_id,
            "execution_id": record.execution_id,
            "status": status,
            "summary": summary,
            "artifacts": artifacts,
            "persisted_variables": persisted_variables,
            "persistence_failures": persistence_failures,
            "pending_result_count": len(self._pending_results),
            "error": error,
        }

    def results(self) -> dict[str, Any]:
        """Return and clear accumulated execution output."""
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
            if record.stdout:
                chunk_lines.append(record.stdout.rstrip())
            if record.error:
                chunk_lines.append(f"ERROR: {record.error}")
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
