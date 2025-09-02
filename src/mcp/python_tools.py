import base64
import io
import os
import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Annotated

import matplotlib.pyplot as plt

from mcp.server.fastmcp import FastMCP

DATA_DIRECTORY = "data"

mcp = FastMCP("python_tools")


class PythonREPL:
    """A Python REPL that executes code and returns the standard output."""
    
    def __init__(self):
        """Initialize the REPL with a persistent namespace."""
        self.namespace = {'__builtins__': __builtins__}
        
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
            
            # Close any open matplotlib figures to prevent memory leaks
            plt.close('all')
            
            return output
        except Exception as e:
            sys.stdout = old_stdout
            error_output = f"Error: {str(e)}\n{traceback.format_exc()}"
            
            # Close any open matplotlib figures even on error
            plt.close('all')
            
            return error_output


repl = PythonREPL()


@mcp.tool(
    name="python_repl",
    description=(
        "Execute Python code and return the standard output.\n"
        "This tool maintains state between executions, so variables and imports "
        f"persist across multiple calls. Files can be saved and read from the `{DATA_DIRECTORY}` directory."
    ),
)
async def python_repl(
    code: Annotated[str, "The python code to execute"],
    ) -> str:
    """Execute Python code and return the standard output.
    
    This tool maintains state between executions, so variables and imports
    persist across multiple calls. Files can be saved to the `{DATA_DIRECTORY}` directory.
    
    Args:
        code: The python code to execute

    Returns:
        The standard output of the python code
    
    """
    return repl.run(code)


if __name__ == "__main__":
    mcp.run()
