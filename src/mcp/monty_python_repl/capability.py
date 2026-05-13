"""Monty Python REPL capability implementation. Useful as an alternative to MCP Server."""

from __future__ import annotations

from typing import Any, Annotated

from dataclasses import dataclass
from pydantic import Field
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset

from rich.console import Console

from .help_content import (
    build_execute_tool_description,
    build_help_tool_description,
    build_shared_runtime_guidance,
)
from .repl import MontyPythonREPL
from src.tools.logging import LoggingToolset


CAPABILITY_SYSTEM_PROMPT = build_shared_runtime_guidance()
HELP_TOOL_DESCRIPTION = build_help_tool_description()
EXECUTE_TOOL_DESCRIPTION = build_execute_tool_description()


def _create_toolset(
    repl: MontyPythonREPL,
    console: Console | None = None,
) -> FunctionToolset[Any]:
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool_plain(name="help", description=HELP_TOOL_DESCRIPTION)
    async def help(
        name: Annotated[
            str | None,
            Field(
                description=(
                    "Optional collection or tool name to inspect. When omitted, the "
                    "help tool returns a formatted high-level summary of available collections."
                )
            ),
        ] = None,
    ) -> str:
        """Describe available sandbox functions."""
        return repl.help(name=name)

    @toolset.tool_plain(name="execute", description=EXECUTE_TOOL_DESCRIPTION)
    async def execute(
        code: Annotated[
            str,
            Field(description="The code to execute inside the Monty sandbox."),
        ],
    ) -> dict[str, Any]:
        """Execute Python code inside the Monty sandbox."""
        return await repl.execute(code)

    if console is not None:
        toolset = LoggingToolset(wrapped=toolset, console=console)

    return toolset


@dataclass
class MontyPythonCapability(AbstractCapability[Any]):
    """A Monty-sandboxed Python REPL capability."""

    console: Console | None = None
    _repl: MontyPythonREPL | None = None
    _toolset: FunctionToolset[Any] | None = None

    def __post_init__(self) -> None:
        self._repl = MontyPythonREPL()
        self._toolset = _create_toolset(self._repl, console=self.console)

    @classmethod
    def get_serialization_name(cls) -> str:
        return "MontyPythonCapability"

    def get_toolset(self) -> AbstractToolset[Any] | None:
        """Return the toolset with all registered tools."""
        return self._toolset

    def get_instructions(self) -> str:
        """Return the system prompt for the capability."""
        return CAPABILITY_SYSTEM_PROMPT
