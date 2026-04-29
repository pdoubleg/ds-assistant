import asyncio
import os

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from rich import pretty
from rich.console import Console
from rich.traceback import install

from src.clai import run_chat
from src.tools.data import AnalystAgentDeps, data_tools
from src.tools.file import file_tools
from src.tools.image_gen import image_tools
from src.tools.logging import LoggingToolset

DATA_DIRECTORY = "data"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

console = Console()
pretty.install(console=console)
install(show_locals=True, console=console)

_internet_search = MCPServerStdio(command="uvx", args=["duckduckgo-mcp-server"])
internet_search = LoggingToolset(wrapped=_internet_search, console=console)

run_python = MCPServerStdio(
    "python",
    ["src/mcp/python_repl.py"],
    max_retries=5,
)

get_weather = MCPServerStdio(
    "python",
    ["src/mcp/weather.py"],
    max_retries=5,
)


run_python = LoggingToolset(wrapped=run_python, console=console)
data_toolset = LoggingToolset(wrapped=data_tools, console=console)
image_toolset = LoggingToolset(wrapped=image_tools, console=console)
file_toolset = LoggingToolset(wrapped=file_tools, console=console)
weather_toolset = LoggingToolset(wrapped=get_weather, console=console)

agent = Agent(
    model="openai:gpt-4.1-mini",
    output_type=str,
    retries=5,
    toolsets=[
        data_toolset,
        file_toolset,
        run_python,
        internet_search,
        image_toolset,
        weather_toolset,
    ],
    deps_type=AnalystAgentDeps,
)


@agent.instructions
async def get_analyst_agent_system_prompt(ctx: RunContext[AnalystAgentDeps]):
    prompt = """
    You are a world class assistant AI. Use the available tools to help the user with their query.
    """
    return prompt


async def main():
    deps = AnalystAgentDeps(data_directory=DATA_DIRECTORY)

    try:
        await run_chat(
            stream=True,
            agent=agent,
            deps=deps,
            console=console,
            code_theme="github-dark",
            prog_name="clai-bot",
            message_history=[],
        )
    except* Exception as eg:  # Use exception group handling
        for exc in eg.exceptions:
            console.print(f"[red]Exception in task: {type(exc).__name__}: {exc}[/red]")
            console.print_exception()


if __name__ == "__main__":
    asyncio.run(main())
