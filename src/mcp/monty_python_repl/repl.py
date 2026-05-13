"""Stateful Monty REPL service implementation."""

from __future__ import annotations

import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.rlm.types import CodeExecutionError

from .filesystem import (
    DEFAULT_HOST_WORKSPACE,
    HostWorkspaceOSAccess,
)
from .help_content import (
    OVERVIEW_KEY_NOTES,
    OVERVIEW_LIMITATIONS,
    OVERVIEW_PURPOSE,
    OVERVIEW_TITLE,
    OVERVIEW_WORKFLOW,
    SUPPORTED_NATIVE_IMPORTS,
    CollectionWorkflowStep,
    get_collection_help_content,
)
from .interpreter import MontyReplInterpreter
from .registry import FunctionRegistry, ObjectStore, build_default_registry


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
            tool_entries={entry.name: entry for entry in self.registry.entries()},
            type_check=type_check,
            os_access=self.os_access,
        )
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._execution_counter = 0

    def help(
        self,
        name: str | None = None,
    ) -> str:
        """Describe available sandbox functions as formatted text."""
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
        return self._join_help_sections(
            f"{OVERVIEW_TITLE}\n\nPurpose:\n{OVERVIEW_PURPOSE}",
            f"Collections:\n\n{collections_section}",
            f"Workflow:\n\n{self._format_numbered_steps(OVERVIEW_WORKFLOW)}",
            f"Key Notes:\n\n{self._format_bullets(OVERVIEW_KEY_NOTES)}",
            "Supported native imports:\n" + ", ".join(SUPPORTED_NATIVE_IMPORTS),
            f"Limitations:\n\n{self._format_bullets(OVERVIEW_LIMITATIONS)}",
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

        content = get_collection_help_content(name)
        functions = self.registry.entries(collection=name)
        sections = [
            f"Collection: {name}\n\nPurpose:\n{content.purpose or collection.description}"
        ]

        if content.when_to_use:
            sections.append(
                f"When to use:\n\n{self._format_bullets(content.when_to_use)}"
            )
        if content.workflow:
            sections.append(
                "Typical Workflow:\n\n"
                + self._render_collection_workflow(content.workflow)
            )

        sections.append(
            "Available Tools:\n\n"
            + "\n\n".join(
                self._render_collection_tool_block(function.name)
                for function in functions
            )
        )

        if content.key_concepts:
            sections.append(
                "Key Concepts:\n\n" + self._render_key_concepts(content.key_concepts)
            )
        if content.common_patterns:
            sections.append(
                f"Common Patterns:\n\n{self._format_bullets(content.common_patterns)}"
            )
        if content.common_mistakes:
            sections.append(
                f"Common Mistakes:\n\n{self._format_bullets(content.common_mistakes)}"
            )

        next_steps = list(content.next_steps)
        if not next_steps:
            next_steps.append(
                'Call help("<tool-name>") before using an unfamiliar helper.'
            )
            if functions:
                next_steps.append(
                    f'Call help("{functions[0].name}") to inspect a representative tool in this collection.'
                )
        sections.append(f"Next Steps:\n\n{self._format_bullets(next_steps)}")
        return self._join_help_sections(*sections)

    def _render_collection_workflow(
        self,
        steps: tuple[CollectionWorkflowStep, ...],
    ) -> str:
        """Render the typical workflow section for a collection."""

        blocks: list[str] = []
        for index, step in enumerate(steps, start=1):
            lines = [f"{index}. {step.title}"]
            if step.tools:
                lines.extend(f"   -> {tool}(...)" for tool in step.tools)
            if step.detail:
                lines.append(f"   {step.detail}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

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

    def _render_key_concepts(self, concepts: dict[str, str]) -> str:
        """Render key concepts as definition-style bullets."""

        return "\n\n".join(
            f"- {name}:\n  {description}" for name, description in concepts.items()
        )

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
            intro_lines.append("")
            intro_lines.append(function.detailed_description)

        sections = [
            "\n".join(intro_lines),
            "Signature:\n" + function.render_signature(multiline=True),
            "Arguments:\n" + self._render_arguments_section(name),
        ]

        returns_section = self._render_return_value(function)
        if returns_section:
            sections.append("Returns:\n" + returns_section)
        if function.usage_example:
            sections.append("Usage example:\n" + function.usage_example)
        sections.append(
            "Guidance:\n" + self._format_bullets(tuple(function.usage_guidance))
        )
        return self._join_help_sections(*sections)

    def _render_arguments_section(self, function_name: str) -> str:
        """Render the argument list for a tool help page."""

        function = self.registry.get(function_name)
        if function is None or not function.arguments:
            return "- None"
        return "\n".join(
            argument.render_argument_help() for argument in function.arguments
        )

    def _render_return_value(self, function: Any, *, compact: bool = False) -> str:
        """Render return annotation and key hints for help output."""

        lines: list[str] = []
        normalized_description = self._normalize_return_description(
            function.return_annotation,
            function.return_description,
        )
        if function.return_annotation and function.return_description:
            lines.append(f"- {function.return_annotation}: {normalized_description}")
        elif function.return_annotation:
            lines.append(f"- {function.return_annotation}")
        elif normalized_description:
            lines.append(f"- {normalized_description}")

        if function.return_annotation and function.return_annotation.startswith(
            "dict["
        ):
            return_keys = self._extract_return_keys(function.usage_example or "")
            if return_keys:
                key_label = "Keys:" if not compact else "  Keys:"
                lines.append(key_label)
                prefix = "  - " if not compact else "    - "
                lines.extend(f"{prefix}{key}" for key in return_keys)

        if compact and lines:
            return "\n".join(f"  {line}" for line in lines)
        return "\n".join(lines)

    def _extract_return_keys(self, usage_example: str) -> list[str]:
        """Extract representative dict keys from an example block."""

        lines = usage_example.splitlines()
        collecting = False
        seen: set[str] = set()
        keys: list[str] = []
        for line in lines:
            stripped = line.strip()
            if "Returns:" in stripped:
                collecting = True
                continue
            if not collecting:
                continue
            if not stripped.startswith("#"):
                break
            for key in re.findall(r'"([^"]+)":', stripped):
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
        return keys if 0 < len(keys) <= 4 else []

    def _normalize_return_description(
        self,
        annotation: str | None,
        description: str | None,
    ) -> str | None:
        """Strip duplicated type prefixes from parsed return descriptions."""

        if not description:
            return None
        if annotation:
            prefixed = f"{annotation}: "
            if description.startswith(prefixed):
                return description.removeprefix(prefixed)
        return description

    def _render_help_not_found(self, name: str) -> str:
        """Render a readable not-found page with valid alternatives."""

        available_collections = ", ".join(
            collection.name for collection in self.registry.collections()
        )
        available_functions = ", ".join(entry.name for entry in self.registry.entries())
        return self._join_help_sections(
            f"No collection or function named {name!r} is registered.\n\nChoose one of the valid names below.",
            "Available collections:\n" + available_collections,
            "Available tools:\n" + available_functions,
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
        """Execute Monty sandbox code and return the full execution record."""
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
        return record.model_dump()
