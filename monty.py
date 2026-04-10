import asyncio
import os
from textwrap import dedent

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from rich import pretty
from rich.console import Console
from rich.traceback import install

from src.clai import run_chat
from src.tools.logging import LoggingToolset

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

console = Console()
pretty.install(console=console)
install(show_locals=True, console=console)

monty_python_repl = MCPServerStdio(
    "python",
    ["-m", "src.mcp.monty_python_repl"],
    timeout=60,
    max_retries=5,
)

monty_python_repl = LoggingToolset(wrapped=monty_python_repl, console=console)

agent = Agent(
    model="openai:gpt-5.4",
    output_type=str,
    retries=5,
    toolsets=[monty_python_repl],
    deps_type=None,
    system_prompt=(
        "You are a world class Python analysis assistant. Use the Monty Python "
        "REPL MCP tools to help the user explore data, run code, and create "
        "artifacts in /workspace."
    ),
)


@agent.instructions
async def get_monty_agent_system_prompt() -> str:
    prompt = dedent(
        """\
        <instructions>
            <instruction>Use the `help` tool early to discover the available functions and how the REPL behaves.</instruction>
            <instruction>Use the `execute` tool to run Python code in the persistent Monty REPL session.</instruction>
            <instruction>You may call `execute` multiple times when that improves the workflow.</instruction>
            <instruction>Use the `results` tool to retrieve accumulated stdout, errors, and artifact information after one or more execute calls.</instruction>
            <instruction>Keep all files and outputs in `/workspace`.</instruction>
            <instruction>Prefer the registered helper functions for dataframe loading, EDA, Plotly charts, and Excel exports when useful.</instruction>
            <instruction>Be explicit in your code, use descriptive variable names, and inspect outputs before making strong claims.</instruction>
        </instructions>
        """
    )
    return prompt


async def main() -> None:
    deps = None

    try:
        await run_chat(
            stream=True,
            agent=agent,
            deps=deps,
            console=console,
            code_theme="github-dark",
            prog_name="monty-bot",
            message_history=[],
        )
    except* Exception as eg:
        for exc in eg.exceptions:
            console.print(f"[red]Exception in task: {type(exc).__name__}: {exc}[/red]")
            console.print_exception()


if __name__ == "__main__":
    asyncio.run(main())
