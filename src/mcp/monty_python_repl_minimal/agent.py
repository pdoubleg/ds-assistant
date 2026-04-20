"""Standalone pydantic-ai agent helpers for the hackathon Monty package."""

from __future__ import annotations

import asyncio
from textwrap import dedent
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from rich.console import Console

from .capability import MinimalMontyPythonCapability


def build_agent(console: Console | None = None) -> Agent[None, str]:
    """Build the standalone minimal Monty agent.

    Args:
        console: Optional rich console forwarded to the capability.

    Returns:
        Configured pydantic-ai agent.
    """
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
                <instruction>Prefer predefined helpers for EDA, feature selection, feature pipelines, and modeling. Keep `execute(...)` focused on composing those helpers rather than embedding nested code strings.</instruction>
                <instruction>Treat `execute(...)` as a restricted Python runtime rather than a full local interpreter. Direct stdlib file IO, compilation, and introspection operations may be limited.</instruction>
                <instruction>For file creation, file reads, and persistence inside `/workspace`, prefer `write_workspace_text(...)`, `write_workspace_json(...)`, `read_workspace_text(...)`, `read_workspace_json(...)`, and model/report save helpers instead of direct `open(...)` or `Path.*` file APIs.</instruction>
                <instruction>If a common Python operation fails with a sanitized execution error, assume sandbox restrictions first and switch to a provided helper.</instruction>
                <instruction>Use a bare final helper call when you want `execute(...)` to surface a safe summary immediately, and assign results only when you plan to reuse them.</instruction>
                <instruction>Call `results` when you need buffered execution history before making claims, not as the primary way to inspect helper returns.</instruction>
                <instruction>Keep all files and artifacts inside `/workspace`.</instruction>
            </instructions>
            """
        )

    return agent


async def run_console_chat(console: Console | None = None) -> None:
    """Run a minimal standalone console chat loop for the hackathon agent.

    Args:
        console: Optional rich console used for IO.
    """
    console = console or Console()
    agent = build_agent(console=console)
    message_history: list[Any] = []

    while True:
        prompt = await asyncio.to_thread(input, "monty-hackathon> ")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        if not prompt.strip():
            continue

        result = await agent.run(prompt, message_history=message_history)
        console.print(result.output)
        try:
            message_history = result.all_messages()
        except Exception:
            message_history = []


__all__ = ["build_agent", "run_console_chat"]
