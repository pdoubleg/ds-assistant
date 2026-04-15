"""Monty Python REPL capability implementation. Useful as an alternative to MCP Server."""

from __future__ import annotations

from typing import Any, Annotated

from dataclasses import dataclass
from pydantic import Field
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset

from rich.console import Console

from .repl import MontyPythonREPL
from src.tools.logging import LoggingToolset


CAPABILITY_SYSTEM_PROMPT = (
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
)

HELP_TOOL_DESCRIPTION = (
    "Discover registered sandbox helper functions and capability groups. "
    "Call `help()` to discover task-focused collections, "
    "`help('<collection-name>')` to list tools in a collection, and "
    "`help('<tool-name>')` right before using an unfamiliar helper in `execute(...)`. "
    "Monty's native execute imports are intentionally limited, so use the "
    "`run_dataframe_code` helper when you need broader dataframe-oriented DS "
    "library access such as Optuna, LightGBM, joblib, and sklearn pipelines. "
    "When the transform should be reused inside a tunable pipeline, inspect the "
    "`fit_freeform_transformer(...)` helpers in the same collection. Inside "
    "freeform code, convert `/workspace/...` paths with `workspace_path(...)` "
    "or `resolve_workspace_path(...)` before using pandas or joblib. Favor the "
    "pattern `freeform_code = '''...'''` over deeply nested inline strings, and "
    "avoid embedded newline escapes like `print(f'\\n...')` unless you double-escape them as `\\\\n`. "
    "For diagnostics, prefer separate `print()` calls. "
    "fixed modeling flow of freeform -> declarative feature engineering -> preprocessing when possible. "
)

EXECUTE_TOOL_DESCRIPTION = (
    "Run Python code in a persistent interpreter-like session. "
    "Use this to execute code that you want to run. "
    "The code will be executed in a sandboxed environment, and the results will be returned to you. "
    "Keep all files and outputs in `/workspace`. "
    "Native imports are intentionally limited inside `execute(...)`. "
    "When you need broader host-side dataframe library access over a pandas dataframe handle, "
    "prefer the `run_dataframe_code(...)` helper. "
    "If the transform should become part of a reusable sklearn-style pipeline, "
    "use the `fit_freeform_transformer(...)` helpers instead. "
    "The ad hoc helper still returns a single final dataframe, so add print statements inside your code when you want intermediate diagnostics. "
    "Inside freeform code, do not pass raw `/workspace/...` strings to pandas, joblib, or similar libraries; "
    "convert them with `workspace_path(...)` or `resolve_workspace_path(...)` first. "
    "For nested code strings, prefer assigning the freeform source to a named multiline variable before calling the helper. "
    "Avoid escape-heavy inline snippets like `print(f'\\n...')`; use separate `print()` calls or `\\\\n` when a literal slash escape must survive multiple parsing layers. "
    "Prefer the registered helper functions for dataframe loading, EDA, Plotly charts, report generation, export helpers, and tuned pipeline artifacts when useful. "
    "When you finish a main step, summarize the outcome for the user, cite important metrics or artifacts, and propose the next decision instead of silently continuing. "
    "Be explicit in your code, use descriptive variable names, and inspect outputs before making strong claims. "
)

RESULTS_TOOL_DESCRIPTION = (
    "Retrieve stdout, warnings, and errors accumulated since the last results call. "
    "The results will be returned to you. "
    "Keep all files and outputs in `/workspace`. "
)


_repl: MontyPythonREPL | None = None


def get_repl() -> MontyPythonREPL:
    """Return the lazily initialized module-level REPL service."""
    global _repl
    if _repl is None:
        _repl = MontyPythonREPL()
    return _repl


def _create_toolset(console: Console | None = None) -> FunctionToolset[Any]:
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool_plain(name="help", description=HELP_TOOL_DESCRIPTION)
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

    @toolset.tool_plain(name="execute", description=EXECUTE_TOOL_DESCRIPTION)
    async def execute(
        code: Annotated[
            str,
            Field(
                description=(
                    "Python code to execute inside the Monty sandbox. Top-level "
                    "variable assignments persist automatically between execute "
                    "calls when Monty can safely serialize them. Native imports are "
                    "intentionally limited inside execute(...). Keep files under "
                    "/workspace, and use `run_dataframe_code(...)` when you need "
                    "broader host-side dataframe library access. When passing "
                    "freeform code strings to helpers, prefer storing the inner "
                    "source in a named multiline variable, keep the code simple, "
                    "and avoid nested escapes like `\\n` unless they are double-escaped."
                )
            ),
        ],
    ) -> dict[str, Any]:
        """Execute Python code inside the Monty sandbox."""
        return await get_repl().execute(code)

    @toolset.tool_plain(name="results", description=RESULTS_TOOL_DESCRIPTION)
    async def results() -> dict[str, Any]:
        """Return and clear buffered execution output."""
        return get_repl().results()

    if console is not None:
        toolset = LoggingToolset(wrapped=toolset, console=console)

    return toolset


@dataclass
class MontyPythonCapability(AbstractCapability[Any]):
    """A Monty-sandboxed Python REPL capability."""

    console: Console | None = None
    _toolset: FunctionToolset[Any] | None = None

    def __post_init__(self) -> None:
        self._toolset = _create_toolset(console=self.console)

    @classmethod
    def get_serialization_name(cls) -> str:
        return "MontyPythonCapability"

    def get_toolset(self) -> AbstractToolset[Any] | None:
        """Return the toolset with all registered tools."""
        return self._toolset

    def get_instructions(self) -> str:
        """Return the system prompt for the capability."""
        return CAPABILITY_SYSTEM_PROMPT
