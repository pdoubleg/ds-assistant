import os
import aiosqlite  # type: ignore
import logging
import asyncio
import requests
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import argparse
from datetime import date
from typing import Annotated, Any, List, Optional, Union
import pandas as pd
from typing_extensions import TypeAlias
from annotated_types import MinLen
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import Usage
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)


# Initialize rich console
console = Console()


# --- OpenAI Client ---


def get_async_openai_client_cortex() -> AsyncOpenAI:
    """Returns the async client for interacting with the OpenAI API.

    Returns:
        AsyncOpenAI: The configured OpenAI client instance.
    """
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    console.log("[green]OpenAI Client created successfully[/green]")
    return client


# --- Database ---


@asynccontextmanager
async def database_connect(db_path: str) -> AsyncGenerator[Any, None]:
    """Async context manager for database connections.
    
    Args:
        db_path (str): Path to the SQLite database file.
        
    Yields:
        aiosqlite.Connection: The database connection.
    """
    conn = await aiosqlite.connect(db_path)
    try:
        async with conn.execute("PRAGMA foreign_keys = ON;"):
            pass
        await conn.commit()
        yield conn
    finally:
        await conn.close()


async def run_sql_query(sql_query: str, conn: aiosqlite.Connection) -> pd.DataFrame:
    """Executes an SQL query and returns a pandas DataFrame.
    
    Args:
        sql_query (str): The SQL query to execute.
        conn (aiosqlite.Connection): The database connection.
        
    Returns:
        pd.DataFrame: The query results as a DataFrame.
        
    Raises:
        Exception: If the query execution fails.
    """
    try:
        async with conn.execute(sql_query) as cursor:
            # Get column names from cursor description
            columns = [description[0] for description in cursor.description]
            # Fetch all rows
            rows = await cursor.fetchall()

        # Convert rows to pandas DataFrame
        df = pd.DataFrame(rows, columns=columns)
        console.log(Panel(f"[green] SQL Execution:[/green]\n\n{sql_query}\n\n"))
        return df

    except Exception as e:
        console.log(f"[bold red] Error running final query:[/bold red] {str(e)}")
        return str(e)


# --- Agent Dependencies ---


@dataclass
class SQLDependencies:
    """Dependencies for the SQL agent.
    
    Attributes:
        conn (aiosqlite.Connection): The database connection.
        df (Optional[pd.DataFrame]): Optional DataFrame for storing results.
    """
    conn: aiosqlite.Connection
    df: Optional[pd.DataFrame] = None


# --- Result Models ---


class FinalSQLQuery(BaseModel):
    """Response when the ideal SQL could be successfully generated.
    
    Attributes:
        sql_query (str): The validated SQL query.
        explanation (str): Explanation of the SQL query in markdown format.
        file_name (str): Filename for saving query results (snake_case, no extension).
    """
    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field(
        ...,
        description="Explanation of the SQL query, as markdown",
    )
    file_name: str = Field(
        ...,
        description="A filename to save the query results of the query to. Do not include the file extension. Should be snake_case.",
    )


class FollowUp(BaseModel):
    """Response when the ideal SQL could not be generated.
    
    Attributes:
        next_step (str): Next step to take if the query is not perfect.
        follow_up_question (str): A follow-up question for the user.
    """
    next_step: str = Field(
        ...,
        description="Next step to take if the query is not perfect",
    )
    follow_up_question: str = Field(
        ...,
        description="A follow-up question that the user can answer to provide more information",
    )


Response: TypeAlias = Union[FinalSQLQuery, FollowUp]


# --- Agent ---


SYSTEM_PROMPT = f"""\
You are a world-class expert at crafting precise SQLite SQL queries.

Your goal is to return a validated final SQL query that satisfies the user request.

<instructions>
    <instruction>Use the provided tools to explore the database and construct the perfect query.</instruction>
    <instruction>Start by listing tables to understand what's available.</instruction>
    <instruction>Describe tables to understand their schema and columns.</instruction>
    <instruction>Sample tables to see actual data patterns.</instruction>
    <instruction>Only call FinalSQLQuery when you're confident the query is perfect.</instruction>
    <instruction>Be thorough but efficient with tool usage.</instruction>
    <instruction>If you find your FinalSQLQuery tool call returns an error or won't satisfy the user request, try to fix the query or try a different query.</instruction>
    <instruction>Think step by step about what information you need.</instruction>
    <instruction>Be sure to specify every parameter for each tool call.</instruction>
    <instruction>Every tool call should have a reasoning parameter which gives you a place to explain why you are calling the tool.</instruction>
</instructions>

today's date = {date.today()}
"""


def get_agent(model: str, retries: int) -> Agent:
    """Creates and configures the SQL agent with tools and validators.
    
    Args:
        model (str): The OpenAI model to use.
        retries (int): Number of retries for each API call.
        
    Returns:
        Agent: The configured SQL agent.
    """
    client = get_async_openai_client_cortex()
    model = OpenAIModel(model, openai_client=client)

    sqlite_agent = Agent(
        model,
        result_type=Response,
        deps_type=SQLDependencies,
        retries=retries,
        system_prompt=SYSTEM_PROMPT,
    )

    @sqlite_agent.tool
    async def list_tables(
        ctx: RunContext[SQLDependencies], reasoning: str
    ) -> List[str]:
        """Returns a list of available tables in the database.

        Args:
            reasoning (str): The reasoning for calling this tool.

        Returns:
            List[str]: A list of table names in the database.
        """
        console.print(f"\n[green]🤖 Assistant: [/green]{reasoning}\n")

        async with ctx.deps.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ) as cursor:
            rows = await cursor.fetchall()
            tables = [row[0] for row in rows]
            console.log(f"[bold blue] List Tables Tool[/bold blue] - Tables: {tables}")

        return tables

    @sqlite_agent.tool
    async def describe_table(
        ctx: RunContext[SQLDependencies], reasoning: str, table_name: str
    ) -> str:
        """Returns schema information for a given table name.

        Args:
            reasoning (str): The reasoning for calling this tool.
            table_name (str): The name of the table to describe.

        Returns:
            str: A string containing the schema information for the table.
        """
        console.print(f"\n[green]🤖 Assistant: [/green]{reasoning}\n")

        try:
            async with ctx.deps.conn.execute(
                f"PRAGMA table_info('{table_name}');"
            ) as cursor:
                rows = await cursor.fetchall()

            output = "\n".join([str(row) for row in rows])
            console.log(
                f"[bold blue] Describe Table Tool[/bold blue] - Table: {table_name}"
            )
            console.log(output)
            return output

        except Exception as e:
            console.log(f"[bold red] Error describing table:[/bold red] {str(e)}")
            return ""

    @sqlite_agent.tool
    async def sample_table(
        ctx: RunContext[SQLDependencies],
        reasoning: str,
        table_name: str,
        row_sample_size: int,
    ) -> str:
        """Returns a sample of rows from the specified table.

        Args:
            reasoning (str): The reasoning for calling this tool.
            table_name (str): The name of the table to sample.
            row_sample_size (int): The number of rows to sample from the table.

        Returns:
            str: A string containing the sample rows from the table.
        """
        def truncate_string(value: any, max_length: int) -> any:
            """Truncates string values to a maximum length."""
            if isinstance(value, str) and len(value) > max_length:
                return value[:max_length] + "..."
            return value

        console.print(f"\n[green]🤖 Assistant: [/green]{reasoning}\n")

        try:
            async with ctx.deps.conn.execute(
                f"SELECT * FROM {table_name} LIMIT {row_sample_size};"
            ) as cursor:
                rows = await cursor.fetchall()

            truncated_rows = [
                tuple(truncate_string(value, 50) for value in row) for row in rows
            ]

            output = "\n".join([str(row) for row in truncated_rows])
            console.log(
                f"[bold blue] Sample Table Tool[/bold blue] - Table: {table_name} - Rows: {row_sample_size}"
            )
            console.log(output)
            return output

        except Exception as e:
            console.log(f"[bold red] Error sampling table:[/bold red] {str(e)}")
            return ""

    @sqlite_agent.result_validator
    async def validate_result(
        ctx: RunContext[SQLDependencies], result: Response
    ) -> Response:
        """Validates the agent's response and SQL query.
        
        Args:
            ctx (RunContext[SQLDependencies]): The runtime context.
            result (Response): The agent's response to validate.
            
        Returns:
            Response: The validated response.
            
        Raises:
            ModelRetry: If the query is invalid or not a SELECT statement.
        """
        if isinstance(result, FollowUp):
            console.print(
                f"\n[bold green]🤖Assistant Follow Up Question:[/bold green] {result.follow_up_question}"
            )
            return result

        # Remove extraneous backslashes (if any)
        result.sql_query = result.sql_query.replace("\\", "")

        if not result.sql_query.upper().startswith("SELECT"):
            console.print(
                "\n[red]Response Validation:[/red] Only SELECT queries are allowed. Please try again."
            )
            raise ModelRetry("Please create a SELECT query")

        try:
            console.print(
                f"\n[yellow]Response Validation Tool: [/yellow]Input SQL:\n{result.sql_query}"
            )

            # Validate the SQL query using SQLite's EXPLAIN QUERY PLAN
            async with ctx.deps.conn.execute(
                f"EXPLAIN QUERY PLAN {result.sql_query}"
            ) as cursor:
                await cursor.fetchall()

            console.print(
                "\n[yellow]Response Validation: [/yellow][bold green]✅ Success![/bold green]"
            )

        except Exception as e:
            console.print(f"\n[bold red]Response Validation: Error:[/bold red] {e}")
            raise ModelRetry(f"Invalid query: {e}") from e

        else:
            return result

    return sqlite_agent


# --- Helpers ---


def get_cost_from_tokens(usages: List[Usage], pricing_dict: dict) -> float:
    """Calculates token cost from given prompt and completion tokens.

    Args:
        usages (List[Usage]): List of usage objects containing token counts.
        pricing_dict (dict): Pricing dictionary containing prompt and completion costs for the deployment.

    Returns:
        float: Total token cost.
    """
    prompt_tokens = sum([usage.request_tokens for usage in usages])
    completion_tokens = sum([usage.response_tokens for usage in usages])
    prompt_token_cost = prompt_tokens * pricing_dict["prompt_cost"]
    completion_token_cost = completion_tokens * pricing_dict["completion_cost"]
    return prompt_token_cost + completion_token_cost


# --- Main CLI Application ---


async def main() -> str:
    """Main entry point for the SQLite Agent CLI application.

    This function handles:
    1. Command line argument parsing
    2. Database initialization
    3. Agent execution loop for SQL query generation

    The conversation continues until:
    - User types 'exit' or 'quit'
    - Maximum compute iterations are reached

    Returns:
        str: The results of the final SQL query execution

    Raises:
        Exception: If maximum compute iterations reached without result
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="SQLite Agent using OpenAI API")
    parser.add_argument(
        "-m", "--model", default="gpt-4o-mini-2024-07-18", help="OpenAI model to use"
    )
    parser.add_argument("-p", "--prompt", required=True, help="The user's request")
    parser.add_argument(
        "-r",
        "--retries",
        default=5,
        help="Number of retries for the agent for each api call",
    )
    parser.add_argument(
        "-db",
        "--database",
        default="chinook.db",
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "-out", "--output", default="output", help="Path to save the output Excel file"
    )
    parser.add_argument(
        "-c",
        "--compute",
        type=int,
        default=5,
        help="Maximum number of agent loops (default: 5)",
    )

    args = parser.parse_args()

    # Try to get pricing info
    pricing_dict = {}

    try:
        RESOURCE_URL: str = (
            "https://lm-cf-openai-api-prd01.apim.lmig.com/libertygpt/models"
        )

        model_json = requests.get(RESOURCE_URL).json()

        for model_versions in model_json.values():
            # Traverse through version keys
            for versions in model_versions.values():
                # Each version contains a list of dictionaries
                for deployment in versions:
                    # Check if the deployment_name matches
                    if deployment.get("deployment_name") == args.model:
                        # Get the token_cost if a match is found
                        pricing_dict = deployment.get("token_cost")
                        console.print(
                            f"\n[green]Pricing info retrieved for model {args.model}[/green]"
                        )

    except Exception as _:
        console.print("[red]Model pricing info could not be retrieved[/red]")

    # Initialize agent
    sqlite_agent = get_agent(model=args.model, retries=args.retries)

    async with database_connect(args.database) as conn:
        usages = []
        deps = SQLDependencies(conn=conn)
        compute_iterations = 0
        message_history: list[ModelMessage] = []

        console.print(f"\n[blue]👤 User:[/blue]\n{args.prompt}")
        prompt = args.prompt

        while True:
            console.rule(
                f"\n[bold yellow]Agent Loop {compute_iterations + 1}/{args.compute}[/bold yellow]"
            )

            if compute_iterations >= args.compute:
                console.print(
                    "\n[bold red]Maximum compute iterations reached. Exiting...[/bold red]"
                )
                break

            compute_iterations += 1

            try:
                response = await sqlite_agent.run(
                    prompt,
                    deps=deps,
                    message_history=message_history,
                )

                message_history.extend(response.new_messages())
                usages.append(response.usage())

                if isinstance(response.data, FinalSQLQuery):
                    console.print("\n[green]🤖 Assistant:[/green]")
                    console.print(f"{response.data.explanation}\n")

                    df = await run_sql_query(
                        sql_query=response.data.sql_query,
                        conn=conn,
                    )

                    preview_df = df.head(10)

                    # Create table with DataFrame styling
                    table = Table(
                        title=f"Query Results (showing first {len(preview_df)} of {len(df)} rows)",
                        show_header=True,
                        header_style="bold magenta",
                        show_lines=True,
                        title_style="bold magenta",
                    )

                    # Add columns
                    for column in preview_df.columns:
                        table.add_column(str(column))

                    # Add rows
                    for row in preview_df.values:
                        table.add_row(*[str(cell) for cell in row])

                    console.print(table)

                    # Show basic DataFrame info
                    console.print("\n[blue]DataFrame Info:[/blue]")
                    console.print(f"Total Rows: {len(df)}")
                    console.print(f"Columns: {', '.join(df.columns.tolist())}")

                    # Create data directory if it doesn't exist
                    os.makedirs(args.output, exist_ok=True)

                    # Save DataFrame to Excel file using filename from response
                    excel_path = f"{args.output}/{response.data.file_name}.xlsx"
                    df.to_excel(excel_path, index=False)
                    console.print(f"\n[green]Data saved to:[/green] {excel_path}")

                    if pricing_dict:
                        total_cost = get_cost_from_tokens(usages, pricing_dict)
                        usages = []
                        console.print(
                            f"\n[green]Total cost:[/green] [bold green]${total_cost:.4f}[/bold green]"
                        )

                    # After displaying results, ask if user wants to continue
                    console.print("\n[blue]🤖...[/blue]")
                    prompt = Prompt.ask(
                        "Enter your next query (or 'exit'/'quit' to end)"
                    )

                    if prompt.lower() in ["exit", "quit"]:
                        console.print("[blue]Exiting agent loop[/blue]")
                        break

                else:
                    # Handle follow-up questions
                    console.print(
                        f"\n[bold green]🤖 Assistant:\n[/bold green] {response.data.follow_up_question}"
                    )

                    console.print("\n[blue]👤 User:[/blue]")
                    prompt = Prompt.ask("Your response")

                    if prompt.lower() in ["exit", "quit"]:
                        console.print(
                            "\n[green]Conversation ended by user. Goodbye![/green]"
                        )
                        break

            except Exception as e:
                console.print(f"\n[bold red]Error in agent loop:[/bold red] {str(e)}")
                console.print("\n[blue]👤 User:[/blue]")
                prompt = Prompt.ask(
                    "Error occurred. Enter new query or 'exit'/'quit' to end"
                )

                if prompt.lower() in ["exit", "quit"]:
                    break

                continue

        return "Conversation ended"


if __name__ == "__main__":
    asyncio.run(main())
