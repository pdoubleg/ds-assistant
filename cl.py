import asyncio
import os

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from rich import pretty
from rich.console import Console
from rich.traceback import install

from src.clai import run_chat
from src.tools.logging import LoggingToolset

DATA_DIRECTORY = "data"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

console = Console()
pretty.install(console=console)
install(show_locals=True, console=console)

court_listener = MCPServerStdio(
    "python",
    ["src/mcp/court_listener.py"],
    max_retries=5,
)

# cl_query_writer = MCPServerStdio(
#     "python",
#     ["src/mcp/cl_query_writer.py"],
#     max_retries=5,
# )

court_listener = LoggingToolset(wrapped=court_listener, console=console)
# cl_query_writer = LoggingToolset(wrapped=cl_query_writer, console=console)

agent = Agent(
    model="openai:gpt-4.1",
    output_type=str,
    retries=5,
    toolsets=[
        court_listener,
        # cl_query_writer,
    ],
    deps_type=None,
)


@agent.instructions
async def get_analyst_agent_system_prompt(ctx: RunContext[None]):
    prompt = """
    You are a world class legal research assistant AI. Use the available tools to help the user with their query.
    """
    return prompt


async def main():
    deps = None
    
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
