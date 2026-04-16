"""Top-level interactive minimal Monty agent entrypoint."""

from __future__ import annotations

import asyncio
import os
from textwrap import dedent

from rich import pretty
from rich.console import Console
from rich.traceback import install

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

from src.mcp.monty_python_repl_minimal.capability import MinimalMontyPythonCapability
from src.clai import run_chat


os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

console = Console()
pretty.install(console=console)
install(show_locals=True, console=console)

model = OpenAIResponsesModel("gpt-5.4")
model_settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort="medium",
    openai_reasoning_summary="auto",
    parallel_tool_calls=True,
)
agent = Agent(
    model=model,
    model_settings=model_settings,
    output_type=str,
    retries=5,
    capabilities=[MinimalMontyPythonCapability(console=console)],
    deps_type=None,
    system_prompt=(
        "You are a world class tabular modeling assistant for a minimal workflow. "
        "Use the minimal Monty REPL tools to inspect schema safely, generate "
        "aggregate plots, run feature engineering, and fit or tune LightGBM models "
        "for PPV at 5%. Never request or reveal raw training rows."
    ),
)

@agent.instructions
async def get_minimal_monty_agent_system_prompt() -> str:
    """Return additional workflow instructions for the minimal agent."""
    return dedent(
        """\
        <instructions>
            <instruction>Start with `help()` to discover collections, then inspect a collection or tool before using it.</instruction>
            <instruction>Never surface raw training rows, raw categorical examples, or raw workspace text.</instruction>
            <instruction>Use schema summaries and aggregate plots first, then narrow feature sets before tuning.</instruction>
            <instruction>Prefer local CSV or partial parquet reads over loading more data than necessary.</instruction>
            <instruction>Always use LightGBM with native categorical handling for modeling steps.</instruction>
            <instruction>Treat PPV@5 as the primary objective, and use recall@5, lift@5, and base rate for context.</instruction>
            <instruction>Call `results` after each meaningful `execute` batch before making claims.</instruction>
            <instruction>Keep all files and artifacts inside `/workspace`.</instruction>
        </instructions>
        """
    )

async def main() -> None:
    """Run the interactive Monty chat loop."""
    deps = None

    try:
        await run_chat(
            stream=True,
            agent=agent,
            deps=deps,
            console=console,
            code_theme="github-dark",
            prog_name="monty-minimal-bot",
            message_history=[],
        )
    except* Exception as eg:
        for exc in eg.exceptions:
            console.print(f"[red]Exception in task: {type(exc).__name__}: {exc}[/red]")
            console.print_exception()


if __name__ == "__main__":
    asyncio.run(main())