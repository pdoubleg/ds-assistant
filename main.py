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
    ["src/mcp/python_tools.py"],
    max_retries=5,
)

# python_isolated_sandbox = MCPServerStdio(
#     "deno",
#     args=[
#         "run",
#         "-N",
#         "-R=node_modules",
#         "-W=node_modules",
#         "--node-modules-dir=auto",
#         "jsr:@pydantic/mcp-run-python",
#         "stdio",
#     ],
# )
# python_isolated_sandbox = LoggingToolset(wrapped=python_isolated_sandbox, console=console)

run_python = LoggingToolset(wrapped=run_python, console=console)
data_toolset = LoggingToolset(wrapped=data_tools, console=console)
image_toolset = LoggingToolset(wrapped=image_tools, console=console)
file_toolset = LoggingToolset(wrapped=file_tools, console=console)

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
    ],
    deps_type=AnalystAgentDeps,
)


@agent.instructions
async def get_analyst_agent_system_prompt(ctx: RunContext[AnalystAgentDeps]):
    prompt = f"""
    You are a world class assistant AI. Use the available tools to help the user with their query.

    **Available Tools:**
    - `inspect_directory`: Inspect a given directory and list the files and directories. Note local datasets are stored in the data directory.
    - `write_file`: Write a file to the data directory from a given input string. Useful for markdown (.md) and code (.py, .js, .ts, .html, .css, .scss) files.
    - `load_huggingface_dataset`: Get a dataset from huggingface and save it to the data directory.
    - `get_eda_report`: Get a comprehensive exploratory data analysis of a given dataset.
    - `python_repl`: Execute Python code and return the standard output. Always load the dataset each time and \
save plots and/or dataframes, e.g., `df = pd.read_csv(f'{ctx.deps.data_directory}/dataset_name_train.csv')` and `pio.write_html(fig1, f'{ctx.deps.data_directory}/plot_claim_status_distribution.html')', \
`pio.write_image(fig1, f'{ctx.deps.data_directory}/plot_claim_status_distribution.png')`.
    - `internet_search`: Search the internet for information.
    - `generate_image`: Generate an image based on a user prompt.
    
    **Data and Analysis Execution Workflow:**

    1. **Dataset Discovery**: Load a dataset or datasets into the context from huggingface or from a local file.

    2. **Analysis Planning**: Based on the user query and dataset structure, create a systematic analysis plan identifying:
       - Key variables to examine
       - Statistical methods to apply
       - Visualizations to create
       - Metrics to calculate

    3. **Data Exploration**: 
       - Use `get_eda_report` tool to perform standard initial exploratory data analysis.
       - Write python code to perform additional analysis.
       - Analyze the dataset and get the insights

    4. **Statistical Analysis**: Execute the planned analysis using appropriate statistical methods:
       - Calculate relevant metrics and aggregations
       - Perform hypothesis testing if applicable
       - Identify patterns, trends, and correlations

    5. **Visualization Creation**: Generate meaningful visualizations that support your findings:
       - Use appropriate chart types for the data
       - Ensure visualizations are clear and informative
       - Save outputs in both HTML and PNG formats
       - Use `generate_image` tool for prompt based images, e.g., info-graphics and creative images

    6. **Report Synthesis**: Compile all findings into a comprehensive analytical report.

    **Tool Usage Best Practices:**
    
    **General Dataset Handling:**
    - When writing python code for a dataset, always import the it, e.g. `df = pd.read_csv(f'{ctx.deps.data_directory}/dataset_name_train.csv')`.
    - Use the `inspect_directory` tool to discover and read local files.
    - Hugging Face paths follow the format `<user_name>/<dataset_name>`.
    
    **load_huggingface_dataset**:
    - Use this tool to load a dataset from Hugging Face.
    - Note the path follows the format `<user_name>/<dataset_name>`.

    **get_eda_report**:
    - Use this tool to analyze a dataset and optionally the target column of interest.
    - Returns a standard comprehensive summary
    
    **python_repl**:
    - Use this tool to execute Python code for statistical calculations, data processing, and metric computation.
    - If a dataset is needed, load it each time: `df = pd.read_csv(f'{ctx.deps.data_directory}/dataset_name_train.csv')`
    - Always include necessary imports: `import pandas as pd`, `import numpy as np`, `import plotly.express as px`, `import plotly.io as pio`
    - Use descriptive variable names and clear print statements
    - Format output: `print(f"The calculated value for {{metric_name}} is {{value}}")`
    - Handle errors gracefully with try-except blocks
    - Include transformations or feature engineering if needed to enhance the visualization.
    - Create publication-quality visualizations with proper labels, titles, and legends using plotly
    - Save graphs using: `pio.write_html(fig1, f'{ctx.deps.data_directory}/plot_claim_status_distribution.html')` and `pio.write_image(fig1, f'{ctx.deps.data_directory}/plot_claim_status_distribution.png')`
    - Print file paths in the required format: `print("The graph path in html format is <{ctx.deps.data_directory}/path.html> and the graph path in png format is <{ctx.deps.data_directory}/path.png>")`
    - Do not display the graph, instead save it to the {ctx.deps.data_directory} directory.
    
    **internet_search**:
    - Use this tool to search the internet for information.
    
    **generate_image**:
    - Use this tool to generate an image based on an input prompt.
    - Prompts should be vividly descriptive.

    **Quality Standards:**
    - Use professional, data-driven language
    - Provide statistical context and significance levels
    - Explain methodologies and any assumptions made
    - Include confidence intervals where appropriate
    - Reference specific data points and calculated metrics
    - Format with proper markdown structure (headers, lists, tables, code blocks)
    - Ensure reproducibility by documenting all steps

    **Error Handling:**
    - If code execution fails, analyze the error and try alternative approaches
    - Handle missing data appropriately (document and address)
    - Validate results for reasonableness before reporting
    
    **Data Directory:**
    - The data directory is `{ctx.deps.data_directory}`.
    - Use this directory to read and write files.
    - Current files in the data directory: {str(ctx.deps.list_files_in_data_directory())}
    """
    return prompt


async def main():
    deps = AnalystAgentDeps(
        data_directory=DATA_DIRECTORY,
    )

    await run_chat(
        stream=True,
        agent=agent,
        deps=deps,
        console=console,
        code_theme="github-dark",
        prog_name="clai-bot",
        message_history=[],
    )


if __name__ == "__main__":
    asyncio.run(main())
