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
_repl = MontyPythonREPL()


def get_repl() -> MontyPythonREPL:
    """Return the persistent REPL service for this agent process."""
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
