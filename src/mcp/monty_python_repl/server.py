"""FastMCP server wiring for the Monty Python REPL."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from src.rlm.types import CodeExecutionError

from .filesystem import (
    DEFAULT_HOST_WORKSPACE,
    HostWorkspaceOSAccess,
    VIRTUAL_WORKSPACE_ROOT,
)
from .interpreter import MontyReplInterpreter
from .registry import FunctionRegistry, ObjectStore, build_default_registry

mcp = FastMCP(
    name="monty_python_repl",
    instructions=(
        "A Monty-sandboxed Python REPL. Use execute to run code in a persistent "
        "interpreter-like session, help to discover available collections and "
        "registered functions, and "
        "results to retrieve stdout and errors accumulated since the last "
        "results call. Keep files in /workspace."
    ),
)


class HelpResponse(BaseModel):
    """Structured response for the MCP ``help`` tool."""

    query: str | None = Field(
        default=None, description="Optional function name filter."
    )
    collection: str | None = Field(
        default=None,
        description="Optional collection filter for grouped tool discovery.",
    )
    workspace_root: str = Field(description="Sandbox workspace path.")
    collections: list[dict[str, Any]] = Field(default_factory=list)
    functions: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ExecutionRecord(BaseModel):
    """Structured record for a single execute call."""

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
    artifacts: list[str] = Field(
        default_factory=list,
        description="New or modified workspace files.",
    )
    error: str | None = Field(default=None, description="Error message, if any.")
    traceback: str | None = Field(
        default=None, description="Formatted traceback, if any."
    )


class MontyPythonREPL:
    """Stateful Monty REPL service used by the MCP server."""

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
        self._execution_counter = 0
        self._pending_results: list[ExecutionRecord] = []

    def help(
        self,
        name: str | None = None,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Describe available sandbox functions."""
        collections = [item.to_help_dict() for item in self.registry.collections()]
        functions: list[dict[str, Any]] = []
        notes = [
            "Relative paths resolve under /workspace.",
            "Top-level variable assignments persist automatically between execute calls when Monty can serialize them safely.",
            "The results tool returns and clears everything accumulated since the previous results call.",
            "Default EDA helpers return lightweight object handles like df_1 and fig_1.",
            "Use inspect_handle(handle) or dataframe_head(handle) to inspect stored objects.",
            "Use pathlib.Path inside sandbox code for file I/O when needed.",
        ]

        if collection is not None:
            if self.registry.get_collection(collection) is None:
                return HelpResponse(
                    query=name,
                    collection=collection,
                    workspace_root=str(VIRTUAL_WORKSPACE_ROOT),
                    collections=collections,
                    notes=[f"No collection named {collection!r} is registered."],
                ).model_dump()
            functions = [
                entry.to_help_dict()
                for entry in self.registry.entries(collection=collection)
            ]

        if name:
            entry = self.registry.get(name)
            if entry is None or (
                collection is not None and entry.collection != collection
            ):
                return HelpResponse(
                    query=name,
                    collection=collection,
                    workspace_root=str(VIRTUAL_WORKSPACE_ROOT),
                    collections=collections,
                    notes=[f"No function named {name!r} is registered."],
                ).model_dump()
            functions = [entry.to_help_dict()]

        if name is None and collection is None:
            notes.insert(
                0,
                "Call help(collection='<name>') to inspect a task-focused tool collection.",
            )
            notes.insert(
                1,
                "Use the collections list below to choose the right capability area before executing code.",
            )

        response = HelpResponse(
            query=name,
            collection=collection,
            workspace_root=str(VIRTUAL_WORKSPACE_ROOT),
            collections=collections,
            functions=functions,
            notes=notes,
        )
        return response.model_dump()

    def _snapshot_workspace(self) -> dict[str, tuple[int, int]]:
        """Capture file metadata for artifact detection."""
        snapshot: dict[str, tuple[int, int]] = {}
        if not self.workspace_root.exists():
            return snapshot

        for file_path in self.workspace_root.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(self.workspace_root).as_posix()
            snapshot[f"/workspace/{relative}"] = (
                file_path.stat().st_mtime_ns,
                file_path.stat().st_size,
            )
        return snapshot

    def _detect_artifacts(
        self,
        before: dict[str, tuple[int, int]],
        after: dict[str, tuple[int, int]],
    ) -> list[str]:
        """Return new or modified workspace files."""
        return sorted(
            path for path, metadata in after.items() if before.get(path) != metadata
        )

    def _build_summary(
        self,
        *,
        success: bool,
        stdout: str,
        artifacts: list[str],
        persisted_variables: list[str],
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
        return "; ".join(parts) + "."

    async def execute(self, code: str) -> dict[str, Any]:
        """Execute Monty sandbox code and buffer the resulting output."""
        before_snapshot = self._snapshot_workspace()
        stdout = ""
        persisted_variables: list[str] = []
        error: str | None = None
        formatted_traceback: str | None = None
        status = "success"

        try:
            interpreter_result = await self.interpreter.execute(code)
            stdout = interpreter_result.stdout
            persisted_variables = interpreter_result.persisted_names
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

        after_snapshot = self._snapshot_workspace()
        artifacts = self._detect_artifacts(before_snapshot, after_snapshot)
        summary = self._build_summary(
            success=status == "success",
            stdout=stdout,
            artifacts=artifacts,
            persisted_variables=persisted_variables,
            error=error,
        )

        self._execution_counter += 1
        record = ExecutionRecord(
            execution_id=self._execution_counter,
            executed_at=datetime.now(timezone.utc).isoformat(),
            code=code,
            status=status,
            summary=summary,
            stdout=stdout,
            persisted_variables=persisted_variables,
            artifacts=artifacts,
            error=error,
            traceback=formatted_traceback,
        )
        self._pending_results.append(record)

        return {
            "execution_id": record.execution_id,
            "status": status,
            "summary": summary,
            "artifacts": artifacts,
            "persisted_variables": persisted_variables,
            "pending_result_count": len(self._pending_results),
            "error": error,
        }

    def results(self) -> dict[str, Any]:
        """Return and clear accumulated execution output."""
        if not self._pending_results:
            return {
                "status": "empty",
                "summary": "No new execution output since the last results call.",
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
            if record.artifacts:
                chunk_lines.append(f"Artifacts: {', '.join(record.artifacts)}")
            chunks.append("\n".join(chunk_lines).strip())

        response = {
            "status": "ok",
            "summary": f"Returned {len(self._pending_results)} buffered execution result(s).",
            "workspace_root": str(VIRTUAL_WORKSPACE_ROOT),
            "executions": executions,
            "combined_output": "\n\n".join(chunks),
        }
        self._pending_results = []
        return response


_repl: MontyPythonREPL | None = None


def get_repl() -> MontyPythonREPL:
    """Return the lazily initialized module-level REPL service."""
    global _repl
    if _repl is None:
        _repl = MontyPythonREPL()
    return _repl


@mcp.tool(name="execute")
async def execute(
    code: Annotated[
        str,
        Field(
            description=(
                "Python code to execute inside the Monty sandbox. Top-level "
                "variable assignments persist automatically between execute "
                "calls when Monty can safely serialize them. Keep files under "
                "/workspace."
            )
        ),
    ],
) -> dict[str, Any]:
    """Execute Python code inside the Monty sandbox."""
    return await get_repl().execute(code)


@mcp.tool(name="help")
async def help(
    name: Annotated[
        str | None,
        Field(description="Optional sandbox function name to inspect in detail."),
    ] = None,
    collection: Annotated[
        str | None,
        Field(
            description=(
                "Optional collection name to inspect. When omitted, the help "
                "tool returns a summary of available collections."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Describe available sandbox functions."""
    return get_repl().help(name=name, collection=collection)


@mcp.tool(name="results")
async def results() -> dict[str, Any]:
    """Return and clear buffered execution output."""
    return get_repl().results()
