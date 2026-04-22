"""Minimal Monty Python REPL capability implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from rich.console import Console

from .repl import MinimalMontyPythonREPL
from .logging import LoggingToolset

CAPABILITY_SYSTEM_PROMPT = """\
A minimal modeling REPL. Use `help` to discover safe collections, `execute` to run code in a persistent \
session, and `results` when you need buffered execution history. Prefer predefined helpers for data \
inspection, wide-table feature batching, feature selection, feature-engineering pipelines, scoring, and \
modeling. Keep `execute(...)` short and orchestration-focused. The runtime is restricted, so direct stdlib \
file IO, compilation, and introspection operations may be limited. Only a small supported subset of imports \
is available to caller code, such as `typing`, `json`, `math`, `re`, and `datetime`. If registered helpers \
rely on external packages like `pandas` or `sklearn`, the helper handles those imports internally, so never \
execute code that starts with `import pandas as pd`. Use workspace helpers for reads/writes and model/report \
helpers for persistence. Never ask for or display raw training rows, categorical examples, raw file contents, \
or row-level chart payloads. Stdout, including `print(...)`, is suppressed. `execute(...)` returns only \
compact execution status; use `results()` for buffered helper summaries, handles, and detailed execution \
output.

When you want a specific visible result, end `execute(...)` with a \
compact bare helper call or dict/list expression instead of `print(...)`, because `results()` exposes the \
last expression or helper summary rather than printed output. Prefer local CSV or partial parquet reads, \
data-view summaries, aggregate plots, feature screening, LightGBM native categoricals, and Optuna tuning \
for PPV@5.
"""

_repl: MinimalMontyPythonREPL | None = None


def get_repl() -> MinimalMontyPythonREPL:
    """Return the lazily initialized module-level REPL service."""
    global _repl
    if _repl is None:
        _repl = MinimalMontyPythonREPL()
    return _repl


def _create_toolset(console: Console | None = None) -> FunctionToolset[Any]:
    """Build the minimal capability toolset.

    Args:
        console: Optional rich console for logging tool calls.

    Returns:
        Function toolset exposing help, execute, and results.
    """
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool_plain(
        name="help",
        description=(
            "Discover safe modeling collections and inspect a specific collection "
            "or helper before using it."
        ),
    )
    async def help(
        name: Annotated[
            str | None,
            Field(
                description=(
                    "Optional collection or tool name to inspect. When omitted, "
                    "returns a high-level overview."
                )
            ),
        ] = None,
    ) -> str:
        """Describe available safe modeling helpers."""
        return get_repl().help(name=name)

    @toolset.tool_plain(
        name="execute",
        description=(
            "Run Python code in the persistent minimal Monty session. Use safe "
            "helpers for data access, data inspection, feature selection, feature pipelines, "
            "and LightGBM modeling."
        ),
    )
    async def execute(
        code: Annotated[
            str,
            Field(
                description=(
                    "Python code to execute inside the persistent Monty session. "
                    "This is a restricted Python runtime, so direct `open(...)`, "
                    "`Path.write_text(...)`, `Path.read_text(...)`, `compile(...)`, "
                    "and similar stdlib operations may be limited. Only a small "
                    "supported subset of imports is available to caller code, such "
                    "as `typing`, `json`, `math`, `re`, and "
                    "`datetime`; if a helper relies on packages like `pandas` or "
                    "`sklearn`, it handles those imports internally, so do not "
                    "execute `import pandas as pd`. Prefer workspace helpers for "
                    "file IO and supported save helpers for persistence. "
                    "Execute returns only compact execution status, while stdout, "
                    "including `print(...)`, remains suppressed. Call `results()` "
                    "for accumulated helper summaries, handles, and buffered "
                    "execution details. Prefer a compact bare final expression "
                    "over `print(...)` when you want a specific visible result, "
                    "and use that final expression deliberately because `results` "
                    "exposes last-expression/helper summaries rather than printed "
                    "output. Prefer helper composition over nested code strings."
                )
            ),
        ],
    ) -> dict[str, Any]:
        """Execute Python code inside the Monty sandbox."""
        return await get_repl().execute(code)

    @toolset.tool_plain(
        name="results",
        description=(
            "Retrieve and clear buffered detailed execution records whose visible "
            "computed values come from last-expression and helper summaries."
        ),
    )
    async def results() -> dict[str, Any]:
        """Return and clear buffered detailed execution records."""
        return get_repl().results()

    if console is not None:
        toolset = LoggingToolset(wrapped=toolset, console=console)

    return toolset


@dataclass
class MinimalMontyPythonCapability(AbstractCapability[Any]):
    """Minimal Monty Python REPL capability."""

    console: Console | None = None
    _toolset: FunctionToolset[Any] | None = None

    def __post_init__(self) -> None:
        """Initialize the wrapped toolset."""
        self._toolset = _create_toolset(console=self.console)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the capability serialization name."""
        return "MinimalMontyPythonCapability"

    def get_toolset(self) -> AbstractToolset[Any] | None:
        """Return the toolset with all registered tools."""
        return self._toolset

    def get_instructions(self) -> str:
        """Return the system prompt for the capability."""
        return CAPABILITY_SYSTEM_PROMPT


__all__ = ["MinimalMontyPythonCapability"]
