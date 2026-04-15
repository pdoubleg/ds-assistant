"""Top-level interactive Monty agent entrypoint."""

import asyncio
import os
from textwrap import dedent

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

from rich import pretty
from rich.console import Console
from rich.traceback import install

from src.clai import run_chat
from src.mcp.monty_python_repl.capability import MontyPythonCapability
from src.message_history.processor import create_summarization_processor, count_tokens_tiktoken

os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

DEFAULT_MODEL = os.getenv("MONTY_MODEL", "openai:gpt-5.4")
DEFAULT_MCP_TIMEOUT_SECONDS = int(os.getenv("MONTY_MCP_TIMEOUT_SECONDS", "1800"))

console = Console()
pretty.install(console=console)
install(show_locals=True, console=console)

# monty_python_repl = MCPServerStdio(
#     "python",
#     ["-m", "src.mcp.monty_python_repl"],
#     timeout=DEFAULT_MCP_TIMEOUT_SECONDS,
#     max_retries=5,
# )

# monty_python_repl = LoggingToolset(wrapped=monty_python_repl, console=console)

history_processor = create_summarization_processor(
    model="openai:gpt-5.4",
    trigger=("fraction", 0.8),
    keep=("messages", 10),
    max_input_tokens=272_000,
    token_counter=count_tokens_tiktoken,
)

model = OpenAIResponsesModel('gpt-5.4')
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
    capabilities=[MontyPythonCapability(console=console)],
    history_processors=[history_processor],
    deps_type=None,
    system_prompt=(
        "You are a world class Python analysis assistant. Use the Monty Python "
        "REPL tools to help the user explore data, run code, create "
        "artifacts in /workspace, and leave behind reproducible modeling outputs. "
        "Remember that native imports inside `execute(...)` are intentionally "
        "limited; when broader dataframe-oriented library usage is needed over a "
        "stored pandas dataframe, prefer the `run_dataframe_code` freeform helper."
    ),
)


@agent.instructions
async def get_monty_agent_system_prompt() -> str:
    """Return additional workflow instructions for the modeling agent."""
    prompt = dedent(
        """\
        <instructions>
            <instruction>Start with `help()` to discover collections, then use `help('<collection-name>')` to inspect a collection or `help('<tool-name>')` for the specific function needed right now.</instruction>
            <instruction>Monty's native imports inside `execute(...)` are intentionally limited. When you need broader dataframe-oriented library usage over a pandas dataframe handle, inspect and use `run_dataframe_code`.</instruction>
            <instruction>Treat the session as a guided workflow. Before each major phase, explain the next step, ask for user confirmation when the direction could materially change the model, and only then execute code.</instruction>
            <instruction>Major phases usually include data inspection, preprocessing design, feature engineering, feature selection, tuning, final model packaging, and report/export generation.</instruction>
            <instruction>Use the `execute` tool to run Python code in the persistent Monty REPL session. Inside Monty code, call registered helpers directly by name.</instruction>
            <instruction>After every meaningful `execute` batch, call `results` before making claims so you inspect stdout, warnings, errors, and newly created artifacts.</instruction>
            <instruction>Keep all files and outputs in `/workspace`.</instruction>
            <instruction>Prefer the registered helper functions for dataframe loading, EDA, Plotly charts, report generation, export helpers, and tuned pipeline artifacts when useful.</instruction>
            <instruction>When you finish a main step, summarize the outcome for the user, cite important metrics or artifacts, and propose the next decision instead of silently continuing.</instruction>
            <instruction>Be explicit in your code, use descriptive variable names, and inspect outputs before making strong claims.</instruction>
        </instructions>
        """
    )
    return prompt


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
            prog_name="monty-bot",
            message_history=[],
        )
    except* Exception as eg:
        for exc in eg.exceptions:
            console.print(f"[red]Exception in task: {type(exc).__name__}: {exc}[/red]")
            console.print_exception()


if __name__ == "__main__":
    asyncio.run(main())
