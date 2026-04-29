import asyncio
import os
from textwrap import dedent

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
    timeout=60,
    max_retries=5,
)

cl_query_writer = MCPServerStdio(
    "python",
    ["src/mcp/cl_query_writer.py"],
    timeout=60,
    max_retries=5,
)

court_listener = LoggingToolset(wrapped=court_listener, console=console)
cl_query_writer = LoggingToolset(wrapped=cl_query_writer, console=console)

agent = Agent(
    model="openai:gpt-5-mini",
    output_type=str,
    retries=5,
    toolsets=[
        court_listener,
        cl_query_writer,
    ],
    deps_type=None,
    system_prompt="You are a world class legal assistant AI. Use the available tools to help the user with their query.",
)


@agent.instructions
async def get_query_agent_system_prompt(ctx: RunContext[None]):
    prompt = dedent("""\
        <instructions>
            <instruction>Use the provided tools to explore the CourtListener database.</instruction>
            <instruction>Start by using **get_court_listener_query** tool to construct a query or series of queries using an AI agent to address the user's request.</instruction>
            <instruction>Then use **execute_court_listener_search_query** tool to execute the query or series of queries and return the results as a string.</instruction>
            <instruction>Be thorough but efficient with tool usage.</instruction>
            <instruction>Always aim to call get_court_listener_query tool first and only ONE time, then use the results to call the execute_court_listener_search_query tool.</instruction>
            <instruction>Use the IDs as input to the "get tools", e.g., get_opinion, get_opinion_excerpt, get_opinion_excerpt_by_citation, get_person, get_docket, get_attorney, get_oral_argument tools to retrieve the data.</instruction>
            <instruction>Think step by step about what information you need.</instruction>
        </instructions>
    """)
    return prompt


# @agent.instructions
# async def get_legal_assistant_agent_system_prompt(ctx: RunContext[None]):
#     prompt = dedent("""\
#         <instructions>
#             <instruction>Use the provided tools to explore the CourtListener database.</instruction>
#             <instruction>Start by using **search** tools to understand what's available.</instruction>
#             <instruction>Then using ID information from the search results, use **get** tools to retrieve data.</instruction>
#             <instruction>Optionally, use **fetch** tools to fetch additional data that is not directly available from the search or get tools.</instruction>
#             <instruction>Be thorough but efficient with tool usage.</instruction>
#             <instruction>Prefer targeted opinion get tools, e.g., query string or citation, over getting the full text of the opinion.</instruction>
#             <instruction>If you find your tool call returns an error or won't satisfy the user request, try to fix the query or try a different query.</instruction>
#             <instruction>Think step by step about what information you need.</instruction>
#             <instruction>Be sure to specify every parameter for each tool call.</instruction>
#         </instructions>
#     """)
#     return prompt


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
