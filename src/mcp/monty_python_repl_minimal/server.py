"""FastMCP server wiring for the minimal Monty Python REPL."""

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
        "`results` to retrieve buffered execution history. `execute` returns "
        "a compact status payload, while stdout, including `print(...)`, stays "
        "suppressed. `results` is the detailed output channel and returns "
        "buffered execution records whose visible computed values come from the "
        "last expression or helper summary rather than printed output. Build the "
        "final expression deliberately when you want `results` to expose a "
        "specific value shape. "
        "The runtime is restricted, so direct stdlib filesystem, "
        "compilation, and introspection operations may be limited. Only a small "
        "supported subset of imports is available to caller code, such as "
        "`typing`, `json`, `math`, `re`, and `datetime`. "
        "If registered helpers rely on external packages like `pandas` or "
        "`sklearn`, the helper handles those imports internally, so never "
        "execute code that starts with `import pandas as pd`. Prefer "
        "`write_workspace_text`, `write_workspace_json`, `read_workspace_text`, "
        "and `read_workspace_json` for workspace file IO, and prefer model/report "
        "helpers for persistence. Sanitized errors may hide the exact underlying "
        "operation. Prefer predefined helpers for EDA, feature selection, "
        "feature pipelines, scoring, and modeling, and keep `execute(...)` "
        "focused on orchestrating those helpers. Score dataframes with "
        "`score_model_dataframe(...)`, summarize ranked slices with "
        "`summarize_top_p_predictions(...)`, and inspect aggregate false-positive "
        "patterns with `analyze_top_p_false_positives(...)`. This server never "
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
                "EDA, feature selection, feature pipelines, and LightGBM modeling. "
                "This is a restricted Python runtime, so direct `open(...)`, "
                "`Path.write_text(...)`, `Path.read_text(...)`, `compile(...)`, "
                "and similar stdlib operations may be limited. Only a small "
                "supported subset of imports is available to caller code, such as "
                "`typing`, `json`, `math`, `re`, and `datetime`; "
                "if a helper relies on packages like `pandas` or `sklearn`, it "
                "handles those imports internally, so do not execute "
                "`import pandas as pd`. Prefer workspace "
                "helpers and model/report save helpers instead. "
                "Execute returns only compact execution status, while stdout, "
                "including `print(...)`, is suppressed and exception details are "
                "sanitized for privacy. Call `results()` for accumulated helper "
                "summaries, handles, and buffered execution details. Use a "
                "deliberate final bare expression when you want `results` to "
                "expose a specific value shape. Prefer helper composition over "
                "nested code strings."
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
) -> str:
    """Describe available safe modeling helpers."""
    return get_repl().help(name=name)


@mcp.tool(name="results")
async def results() -> dict[str, Any]:
    """Return and clear buffered detailed execution records."""
    return get_repl().results()
