"""FastMCP server wiring for the Monty Python REPL."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .repl import MontyPythonREPL

mcp = FastMCP(
    name="monty_python_repl",
    instructions=(
        "A Monty-sandboxed Python REPL. Use `execute` to run code in a persistent "
        "interpreter-like session, `help` to discover collections or inspect a "
        "specific collection/tool by name, and `results` to retrieve stdout, "
        "warnings, and errors accumulated since the last results call. Keep files "
        "in /workspace. Monty supports only a small set of native imports inside "
        "`execute(...)`, such as datetime, re, json, and math, and does not "
        "support defining classes. When you need broader dataframe-oriented data "
        "science library usage over a stored pandas dataframe, inspect and use "
        "the `run_dataframe_code` freeform helper. For reusable pipeline-safe "
        "logic, inspect the `fit_freeform_transformer(...)` helpers in the same "
        "collection. Those tools keep the same single-`df` contract while "
        "exposing broader DS libraries such as Optuna, LightGBM, joblib, and "
        "more sklearn pipeline utilities. Inside freeform code, convert "
        "`/workspace/...` paths with `workspace_path(...)` or "
        "`resolve_workspace_path(...)` before passing them to pandas, joblib, "
        "or other host-side file APIs. When submitting code strings to "
        "`run_dataframe_code(...)` or `fit_freeform_transformer(...)`, prefer "
        "assigning the inner code to a named multiline variable first, avoid "
        "nested escape-heavy strings like `print(f'\\n...')`, use separate "
        "`print()` calls for blank lines or diagnostics, and only use `\\\\n` "
        "when the inner code truly needs a literal backslash escape to survive "
        "outer parsing. The preferred modeling flow is reusable "
        "freeform, then declarative feature engineering, then preprocessing."
    ),
)
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
                "calls when Monty can safely serialize them. Native imports are "
                "intentionally limited inside execute(...). Keep files under "
                "/workspace`, and use `run_dataframe_code(...)` when you need "
                "broader host-side dataframe-oriented DS library access. For "
                "reusable sklearn-style pipeline stages, use the "
                "`fit_freeform_transformer(...)` helpers. "
                "`run_dataframe_code(...)` still returns a single final dataframe, "
                "so print intermediate diagnostics inside your code when needed. "
                "Inside freeform code, do not pass raw `/workspace/...` strings "
                "to pandas, joblib, or similar libraries; convert them with "
                "`workspace_path(...)` or `resolve_workspace_path(...)` first. "
                "When passing freeform source to helpers, prefer a named multiline "
                "variable, keep the inner code simple, and avoid nested escapes "
                "like `print(f'\\n...')` unless the literal escape is double-escaped as `\\\\n`. "
                "Favor the fixed modeling flow of freeform -> declarative feature engineering -> preprocessing when possible."
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
    """Describe available sandbox functions."""
    return get_repl().help(name=name)


@mcp.tool(name="results")
async def results() -> dict[str, Any]:
    """Return and clear buffered execution output."""
    return get_repl().results()
