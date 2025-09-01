import base64
import io
import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Annotated

import matplotlib.pyplot as plt

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("python_tools")


class PythonREPL:
    """A Python REPL that executes code and returns the standard output."""
    
    def run(self, code):
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        try:
            exec(code, globals())
            sys.stdout = old_stdout
            return redirected_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {str(e)}\n{traceback.format_exc()}"


repl = PythonREPL()


@mcp.tool()
async def python_repl(code: str) -> str:
    """Execute Python code and return the standard output."""
    return repl.run(code)


@mcp.tool()
async def matplotlib_visualization(code: Annotated[str, "The python code to execute to generate visualization using matplotlib"]) -> str:
    """Use this tool to generate graphs and visualizations using python code and matplotlib library.
    
    - Always include necessary imports and dataset loading, e.g. `df = pd.read_csv('dataframe_1.csv')`
    - Use matplotlib library to make the graph interactive
    - Create publication-quality visualizations with proper labels, titles, and legends
    - Save graphs using: `plt.savefig('graph.png', dpi=300, bbox_inches='tight')` and HTML equivalent
    - Print file paths in the required format: `print("The graph path in html format is <path.html> and the graph path in png format is <path.png>")`
    """
    try:
        repl.run(code)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()  # Close the figure to free memory
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return f"Error creating chart: {str(e)}"


async def plotly_visualization(
    code: Annotated[str, "The python code to execute to generate visualization using plotly"],
) -> str:
    """
    Use this tool to generate graphs and visualizations using python code.

    - Always include necessary imports and dataset loading, e.g. `df = pd.read_csv('dataframe_1.csv')`
    - Use plotly express library to make the graph interactive
    - Create publication-quality visualizations with proper labels, titles, and legends
    - Save graphs using: `plt.savefig('graph.png', dpi=300, bbox_inches='tight')` and HTML equivalent
    - Print file paths in the required format: `print("The graph path in html format is <path.html> and the graph path in png format is <path.png>")`
    """

    catcher = StringIO()

    try:
        with redirect_stdout(catcher):
            # The compile step can catch syntax errors early
            compiled_code = compile(code, "<string>", "exec")
            exec(compiled_code, globals(), globals())

            return f"The graph path is \n\n{catcher.getvalue()}"

    except Exception as e:
        return f"Failed to run code. Error: {repr(e)}, try a different approach"


if __name__ == "__main__":
    mcp.run()
