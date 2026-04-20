"""Stateful minimal Monty REPL service implementation."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .filesystem import (
    DEFAULT_HOST_WORKSPACE,
    HostWorkspaceOSAccess,
    VIRTUAL_WORKSPACE_ROOT,
)
from .interpreter import MontyReplInterpreter
from .base import SafeObjectStore
from .privacy import sanitize_exception, summarize_stdout
from .registry import build_default_registry


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
    """Stateful Monty REPL specialized for modeling workflows."""

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

    def help(self, name: str | None = None) -> str:
        """Describe available safe modeling helpers as formatted text.

        Args:
            name: Optional collection or tool name.

        Returns:
            Human-readable help text.
        """
        if name is None:
            return self._render_help_overview()

        collection = self.registry.get_collection(name)
        if collection is not None:
            return self._render_help_collection(name)

        function = self.registry.get(name)
        if function is not None:
            return self._render_help_tool(name)

        return self._render_help_not_found(name)

    def _join_help_sections(self, *sections: str) -> str:
        """Join non-empty help sections with a stable separator."""
        return "\n\n---\n\n".join(section for section in sections if section.strip())

    def _format_bullets(self, items: list[str] | tuple[str, ...]) -> str:
        """Render a flat bullet list for help output."""
        return "\n".join(f"- {item}" for item in items)

    def _format_numbered_steps(self, items: list[str] | tuple[str, ...]) -> str:
        """Render a numbered step list for help output."""
        return "\n".join(
            f"{index}. {item}" for index, item in enumerate(items, start=1)
        )

    def _render_help_overview(self) -> str:
        """Render the high-level collection overview as formatted text."""
        collections_section = "\n\n".join(
            self._render_collection_overview_block(collection.name)
            for collection in self.registry.collections()
        )
        workflow = (
            "Call help() to discover safe modeling collections.",
            "Call help('<collection-name>') to inspect the tools in that collection.",
            "Call help('<tool-name>') before using an unfamiliar helper in execute(...).",
            "Keep execute(...) short and orchestration-focused: compose registered helpers, assign handles, and then call results() when you want buffered details.",
            "Prefer workspace helpers for file IO and artifact persistence instead of direct open(...), Path.write_text(...), or similar stdlib file APIs inside execute(...).",
            "End execute(...) with a helper call or compact dict/list expression when you want results() to surface a specific safe summary shape.",
            "For model diagnostics, score a dataframe with score_model_dataframe(...), summarize ranked slices with summarize_top_p_predictions(...), and inspect aggregate false-positive patterns with analyze_top_p_false_positives(...).",
        )
        notes = (
            "This REPL never returns raw training rows or categorical examples.",
            "Use schema summaries, safe plots, feature screening, and model artifacts instead of row previews.",
            "Relative paths resolve under /workspace.",
            "Execute runs in a restricted Python runtime, so standard builtins, filesystem calls, compilation, and introspection behaviors may differ from a local interpreter.",
            "Call registered helpers directly inside execute(...) instead of treating them like dataframe or collection methods.",
            "Arguments ending in `_handle` expect stored handle strings returned by earlier Monty steps.",
            "Execute returns a compact status payload and suppresses raw stdout.",
            "Prefer a compact bare final expression over print(...) when you want results() to expose a specific value immediately after execution.",
            "Residual-analysis helpers return only aggregate statistics, column names, handles, and report summaries; they never expose raw rows or category examples.",
            "For file creation, file reads, and persistence inside /workspace, prefer helper APIs such as write_workspace_text(...), write_workspace_json(...), read_workspace_text(...), and read_workspace_json(...).",
            "Use results() as the detailed output channel for buffered helper summaries, created handles, warnings, and execution history.",
        )
        limitations = (
            "Raw dataframe previews and raw workspace text reads are intentionally unavailable.",
            "All modeling uses LightGBM with native categorical handling.",
            "Stdout and exception details are privacy-sanitized.",
            "Sandbox failures may surface as generic sanitized execution errors, so switch to a provided helper first when common Python operations fail unexpectedly.",
        )
        return self._join_help_sections(
            "Monty Minimal Help\n\nPurpose:\nDiscover safe modeling collections and inspect helpers before using them inside execute(...).",
            f"Collections:\n\n{collections_section}",
            f"Workflow:\n\n{self._format_numbered_steps(workflow)}",
            f"Key Notes:\n\n{self._format_bullets(notes)}",
            f"Limitations:\n\n{self._format_bullets(limitations)}",
        )

    def _render_collection_overview_block(self, collection_name: str) -> str:
        """Render one collection summary block for the overview page."""
        collection = self.registry.get_collection(collection_name)
        if collection is None:  # pragma: no cover - defensive guard
            return collection_name

        tool_names = collection.sorted_tool_names()
        return "\n".join(
            (
                f"[{collection.name}] ({len(tool_names)} tools)",
                collection.description,
                "Tools: " + ", ".join(tool_names),
            )
        )

    def _render_help_collection(self, name: str) -> str:
        """Render a collection-specific help page."""
        collection = self.registry.get_collection(name)
        if collection is None:  # pragma: no cover - defensive guard
            return self._render_help_not_found(name)

        functions = self.registry.entries(collection=name)
        next_steps = [
            'Call help("<tool-name>") before using an unfamiliar helper.',
            "Prefer these predefined helpers over ad hoc sandbox logic whenever a tool already covers the task.",
            "All tool returns are privacy-safe by design.",
        ]
        if name == "workspace":
            next_steps.append(
                "Use the workspace collection for file reads and writes inside /workspace instead of direct open(...), Path.write_text(...), Path.read_text(...), or similar stdlib file operations."
            )

        return self._join_help_sections(
            f"Collection: {name}\n\nPurpose:\n{collection.description}",
            "Available Tools:\n\n"
            + "\n\n".join(
                self._render_collection_tool_block(function.name)
                for function in functions
            ),
            f"Next Steps:\n\n{self._format_bullets(next_steps)}",
        )

    def _render_collection_tool_block(self, function_name: str) -> str:
        """Render one compact tool block for collection help."""
        function = self.registry.get(function_name)
        if function is None:  # pragma: no cover - defensive guard
            return function_name

        lines = [
            function.render_signature(multiline=False),
            f"  {function.description}",
        ]
        compact_returns = self._render_return_value(function, compact=True)
        if compact_returns:
            lines.append(compact_returns)
        return "\n".join(lines)

    def _render_help_tool(self, name: str) -> str:
        """Render a single-tool help page."""
        function = self.registry.get(name)
        if function is None:  # pragma: no cover - defensive guard
            return self._render_help_not_found(name)

        intro_lines = [
            f"Tool: {function.name}",
            f"Collection: {function.collection or 'ungrouped'}",
            f"Purpose: {function.description}",
        ]
        if (
            function.detailed_description
            and function.detailed_description != function.description
        ):
            intro_lines.extend(("", function.detailed_description))

        sections = [
            "\n".join(intro_lines),
            "Signature:\n" + function.render_signature(multiline=True),
            "Arguments:\n" + self._render_arguments_section(function.name),
        ]

        returns_section = self._render_return_value(function)
        if returns_section:
            sections.append("Returns:\n" + returns_section)
        if function.usage_example:
            sections.append("Usage example:\n" + function.usage_example)
        return self._join_help_sections(*sections)

    def _render_arguments_section(self, function_name: str) -> str:
        """Render the argument list for a tool help page."""
        function = self.registry.get(function_name)
        if function is None or not function.arguments:
            return "- None"
        return "\n".join(
            argument.render_argument_help() for argument in function.arguments
        )

    def _render_return_value(
        self,
        function: Any,
        *,
        compact: bool = False,
    ) -> str:
        """Render return annotation and description for help output."""
        lines: list[str] = []
        if function.return_annotation and function.return_description:
            lines.append(
                f"- {function.return_annotation}: {function.return_description}"
            )
        elif function.return_annotation:
            lines.append(f"- {function.return_annotation}")
        elif function.return_description:
            lines.append(f"- {function.return_description}")

        if compact and lines:
            return "\n".join(f"  {line}" for line in lines)
        return "\n".join(lines)

    def _render_help_not_found(self, name: str) -> str:
        """Render an invalid-name help page."""
        available_collections = ", ".join(
            collection.name for collection in self.registry.collections()
        )
        available_functions = ", ".join(entry.name for entry in self.registry.entries())
        return self._join_help_sections(
            (
                f"No collection or function named {name!r} is registered.\n\n"
                "Choose one of the valid names below."
            ),
            f"Collections:\n{available_collections}",
            f"Tools:\n{available_functions}",
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

    def _build_execute_response_summary(
        self,
        *,
        success: bool,
        error: dict[str, Any] | None,
    ) -> str:
        """Build the compact execute-response summary string.

        Args:
            success: Whether execution completed successfully.
            error: Sanitized execution error payload, if any.

        Returns:
            Compact execute response summary for immediate tool output.
        """
        if not success and error is not None:
            return (
                f"Execution failed with {error['error_type']}. "
                "Call results() for buffered details."
            )
        return "Execution succeeded. Call results() for buffered details."

    def _extract_handles(self, value: Any) -> list[str]:
        """Return handle-like strings embedded in a structured payload."""

        handles: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_handle") and isinstance(item, str):
                    handles.append(item)
                else:
                    handles.extend(self._extract_handles(item))
        elif isinstance(value, list):
            for item in value:
                handles.extend(self._extract_handles(item))

        deduped_handles: list[str] = []
        for handle in handles:
            if handle not in deduped_handles:
                deduped_handles.append(handle)
        return deduped_handles

    def _render_structured_value_summary(self, name: str, value: Any) -> list[str]:
        """Render compact results output for a structured helper payload."""

        rendered_lines: list[str] = []
        if isinstance(value, dict):
            summary = value.get("summary")
            if isinstance(summary, str) and summary:
                rendered_lines.append(f"{name}: {summary}")
            handles = self._extract_handles(value)
            if handles:
                rendered_lines.append(f"{name} handles: {', '.join(handles)}")
            warnings = value.get("warnings")
            if isinstance(warnings, list) and warnings:
                rendered_lines.append(
                    f"{name} warnings: {', '.join(str(item) for item in warnings)}"
                )
        return rendered_lines

    def _find_nested_duplicate_path(
        self,
        container: Any,
        target: Any,
        *,
        path: str = "",
    ) -> str | None:
        """Return the first nested path whose value matches the target.

        Args:
            container: Candidate container to search.
            target: Value that may already be represented in the container.
            path: Recursive dotted path prefix.

        Returns:
            Dotted path to the duplicate value, or ``None`` when not found.
        """
        if container == target:
            return path

        if isinstance(container, dict):
            for key, item in container.items():
                child_path = f"{path}.{key}" if path else str(key)
                duplicate_path = self._find_nested_duplicate_path(
                    item,
                    target,
                    path=child_path,
                )
                if duplicate_path is not None:
                    return duplicate_path

        if isinstance(container, list):
            for index, item in enumerate(container):
                child_path = f"{path}[{index}]"
                duplicate_path = self._find_nested_duplicate_path(
                    item,
                    target,
                    path=child_path,
                )
                if duplicate_path is not None:
                    return duplicate_path

        return None

    def _dedupe_execution_views(
        self,
        *,
        persisted_value_summaries: dict[str, Any],
        last_expression_summary: Any | None,
    ) -> tuple[dict[str, Any], Any | None]:
        """Drop redundant final-expression payloads already surfaced elsewhere.

        Args:
            persisted_value_summaries: Persisted variable summaries from execution.
            last_expression_summary: Summary produced by the final bare expression.

        Returns:
            Possibly adjusted persisted summaries and final-expression summary.
        """
        if last_expression_summary is None:
            return persisted_value_summaries, last_expression_summary

        # When the final expression simply re-surfaces a payload already present in
        # an assigned helper result, keep the persisted view and omit the duplicate.
        for value in persisted_value_summaries.values():
            duplicate_path = self._find_nested_duplicate_path(
                value,
                last_expression_summary,
            )
            if duplicate_path is not None:
                return persisted_value_summaries, None

        return persisted_value_summaries, last_expression_summary

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
        persisted_value_summaries, last_expression_summary = (
            self._dedupe_execution_views(
                persisted_value_summaries=persisted_value_summaries,
                last_expression_summary=last_expression_summary,
            )
        )
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
        execute_response_summary = self._build_execute_response_summary(
            success=status == "success",
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
            "summary": execute_response_summary,
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
                structured_lines = self._render_structured_value_summary(
                    "Last expression",
                    record.last_expression_summary,
                )
                if structured_lines:
                    chunk_lines.extend(structured_lines)
                else:
                    chunk_lines.append(
                        f"Last expression: {record.last_expression_summary!r}"
                    )
            if record.persisted_value_summaries:
                rendered_persisted = False
                for name, value in record.persisted_value_summaries.items():
                    structured_lines = self._render_structured_value_summary(
                        name, value
                    )
                    if structured_lines:
                        rendered_persisted = True
                        chunk_lines.extend(structured_lines)
                if not rendered_persisted:
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
