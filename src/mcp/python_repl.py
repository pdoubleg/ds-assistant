import os
import sys
import traceback
from io import StringIO
from typing import Annotated
from pydantic import Field

from mcp.server.fastmcp import FastMCP

DATA_DIRECTORY = "data"

mcp = FastMCP(
    name="python_repl",
    instructions="A Python REPL that executes code and returns the standard output.",
)


class PythonREPL:
    """A Python REPL that executes code and returns the standard output."""

    def __init__(self):
        """Initialize the REPL with a persistent namespace."""

        self.namespace = {"__builtins__": __builtins__}

    def run(self, code: str) -> str:
        """Execute Python code and return the standard output.

        Args:
            code: The Python code to execute

        Returns:
            The standard output of the Python code
        """
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        # Ensure the data directory exists
        os.makedirs(DATA_DIRECTORY, exist_ok=True)

        try:
            # Execute code in persistent namespace to maintain state between calls
            exec(code, self.namespace)
            sys.stdout = old_stdout
            output = redirected_output.getvalue()

            return output or "No output"
        except Exception as e:
            sys.stdout = old_stdout
            error_output = f"Error: {str(e)}\n{traceback.format_exc()}"

            return error_output or "No output"


repl = PythonREPL()


@mcp.tool(
    name="python_repl",
    description=(
        "Execute Python code and return the standard output.\n"
        "Include any necessary imports, e.g., `import pandas as pd`, `import numpy as np`, `import plotly.express as px`, `import plotly.io as pio`.\n"
        "Always load the dataset each time e.g., `df = pd.read_csv(f'{DATA_DIRECTORY}/dataset_name_train.csv')`\n"
        "Always use plotly. Never display plots, only save them."
        "Save plots and/or dataframes, e.g., `df = pd.to_csv(f'{DATA_DIRECTORY}/dataset_name_train.csv')` and for plots, "
        "`pio.write_html(fig1, f'{DATA_DIRECTORY}/plot_claim_status_distribution.html')', "
        "`pio.write_image(fig1, f'{DATA_DIRECTORY}/plot_claim_status_distribution.png')`. Plots should not be displayed, only saved. Favor plotly when possible.\n"
        "Use descriptive variable names and clear print statements, e.g., `print(f'The calculated value for {metric_name} is {value}')`. "
        "Handle errors gracefully with try-except blocks, e.g., `try: ... except Exception as e: ...`."
    ),
)
async def python_repl(
    code: Annotated[str, Field(description="The python code to execute")],
) -> str:
    """Execute Python code and return the standard output.

    This tool maintains state between executions, so variables and imports
    persist across multiple calls. Files can be saved to the `{DATA_DIRECTORY}` directory.

    Args:
        code: The python code to execute

    Returns:
        The standard output of the python code

    """
    
    result = repl.run(code)
    
    return result or "No output"


if __name__ == "__main__":
    mcp.run()
