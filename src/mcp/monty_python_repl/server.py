"""FastMCP server wiring for the Monty Python REPL."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .help_content import (
    build_execute_tool_description,
    build_help_tool_description,
    build_shared_runtime_guidance,
)
from .repl import MontyPythonREPL

mcp = FastMCP(
    name="monty_python_repl",
    instructions=build_shared_runtime_guidance(),
)
_repl: MontyPythonREPL | None = None

RESULTS_TOOL_DESCRIPTION = (
    "Retrieve stdout, warnings, and errors accumulated since the last results call. "
    "The results will be returned to you. "
    "Keep all files and outputs in `/workspace`. "
)


def get_repl() -> MontyPythonREPL:
    """Return the lazily initialized module-level REPL service."""
    global _repl
    if _repl is None:
        _repl = MontyPythonREPL()
    return _repl


@mcp.tool(name="execute", description=build_execute_tool_description())
async def execute(
    code: Annotated[
        str,
        Field(description="The code to execute inside the Monty sandbox."),
    ],
) -> dict[str, Any]:
    return await get_repl().execute(code)


@mcp.tool(name="help", description=build_help_tool_description())
async def help(
    name: Annotated[
        str | None,
        Field(
            description=(
                "Optional collection or tool name to inspect. When omitted, the "
                "help tool returns a formatted high-level summary of available collections."
            ),
        ),
    ] = None,
) -> str:
    """Describe available sandbox functions."""
    return get_repl().help(name=name)


@mcp.tool(name="results", description=RESULTS_TOOL_DESCRIPTION)
async def results() -> dict[str, Any]:
    """Return and clear buffered execution output."""
    return get_repl().results()
