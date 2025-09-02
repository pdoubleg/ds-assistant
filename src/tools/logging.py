from dataclasses import dataclass
from typing_extensions import Any
from rich.console import Console
from pydantic_ai import RunContext
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset

from src.clai import Markdown


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
        if tool_args.get("code", False):
            code_string = f"```python\n{tool_args['code']}\n```"
            self.console.print(f'Calling code tool {name!r}')
            self.console.print(Markdown(code_string, code_theme="github-dark"))
        else:
            self.console.print(f'Calling tool {name!r} with args: {tool_args!r}')
        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
            self.console.print(f'Finished calling tool {name!r}')
            return result
        except Exception as e:
            self.console.print(f'Error calling tool {name!r}: {e}')
            raise
        
        