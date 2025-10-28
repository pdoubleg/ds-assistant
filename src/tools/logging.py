from dataclasses import dataclass
from typing_extensions import Any
from rich.console import Console
from pydantic_ai import RunContext
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset

# from src.clai import Markdown
from pydantic_ai._cli import Markdown


@dataclass
class LoggingToolset(WrapperToolset):
    """A toolset wrapper that logs tool calls using Rich console."""
    
    console: Console | None = None
    
    def __post_init__(self):
        """Initialize console after dataclass initialization."""
        if self.console is None:
            self.console = Console()
    
    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool) -> Any:
        """Log tool calls and delegate to wrapped toolset."""
        # Check if any key contains "code" to handle keys like python_code, code_snippet, etc.
        code_key = next((key for key in tool_args.keys() if "code" in key.lower()), None)
        
        if code_key and tool_args.get(code_key):
            code_string = f"```python\n{tool_args[code_key]}\n```"
            self.console.print(f'Calling code tool {name!r}')
            self.console.print(Markdown(code_string, code_theme="github-dark"))
        else:
            # Truncate long arguments for display
            truncated_args = self._truncate_args(tool_args)
            self.console.print(f'Calling tool {name!r} with args: {truncated_args}')
            
        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
            self.console.print(f'Finished calling tool {name!r}')
            return result
        except Exception as e:
            self.console.print(f'Error calling tool {name!r}: {e}')
            raise
        
    def _truncate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Truncate long argument values for display purposes.
        
        Args:
            args: Dictionary of tool arguments
            
        Returns:
            Dictionary with truncated string representations of arguments
        """
        MAX_ARG_LENGTH = 10000
        truncated = {}
        for key, value in args.items():
            # Convert value to string representation
            str_value = repr(value)
            
            # Truncate if too long
            if len(str_value) > MAX_ARG_LENGTH:
                truncated[key] = str_value[:MAX_ARG_LENGTH - 3] + "..."
            else:
                truncated[key] = str_value
                
        return truncated
        
        