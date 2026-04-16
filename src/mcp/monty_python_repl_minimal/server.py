"""FastMCP server wiring for the hackathon Monty Python REPL."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .repl import MinimalMontyPythonREPL

mcp = FastMCP(
    name="monty_python_repl_minimal",
    instructions=(
        "A minimal Monty-sandboxed Python REPL for tabular modeling. "
        "Use `help` to discover safe collections, `execute` to run code, and "
        "`results` to retrieve buffered execution history. `execute` surfaces "
        "privacy-safe helper summaries directly, while raw stdout stays "
        "suppressed. The runtime is restricted, so direct stdlib filesystem, "
        "compilation, and introspection operations may be limited. Prefer "
        "`write_workspace_text`, `write_workspace_json`, `read_workspace_text`, "
        "and `read_workspace_json` for workspace file IO, and prefer model/report "
        "helpers for persistence. Sanitized errors may hide the exact underlying "
        "operation. Prefer predefined helpers for inspection, feature "
        "screening, and modeling, and use freeform code mainly to create new "
        "dataframe handles from transformations or slices. This server never "
        "returns raw training rows, categorical examples, raw workspace text, "
        "or embedded row-level chart payloads. Prefer schema summaries, safe "
        "plots, feature screening, LightGBM native categorical handling, and "
        "Optuna tuning for PPV at 5%."
    ),
)
_repl: MinimalMontyPythonREPL | None = None


def get_repl() -> MinimalMontyPythonREPL:
    """Return the lazily initialized module-level REPL service."""
    global _repl
    if _repl is None:
        _repl = MinimalMontyPythonREPL()
    return _repl


@mcp.tool(name="execute")
async def execute(
    code: Annotated[
        str,
        Field(
            description=(
                "Python code to execute inside the persistent Monty session. "
                "Use safe helper functions for data access, schema summaries, "
                "plotting, feature workbench steps, and LightGBM modeling. "
                "This is a restricted Python runtime, so direct `open(...)`, "
                "`Path.write_text(...)`, `Path.read_text(...)`, `compile(...)`, "
                "and similar stdlib operations may be limited. Prefer workspace "
                "helpers and model/report save helpers instead. "
                "Execute returns privacy-safe helper summaries directly, while "
                "raw stdout and exception details are sanitized for privacy."
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
        Field(
            description=(
                "Optional collection or tool name to inspect. When omitted, the "
                "help tool returns a high-level summary of available collections."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Describe available safe modeling helpers."""
    return get_repl().help(name=name)


@mcp.tool(name="results")
async def results() -> dict[str, Any]:
    """Return and clear buffered privacy-safe execution history."""
    return get_repl().results()
