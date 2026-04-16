"""Rich logging wrapper for MCP tool calls."""

from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset
from rich.console import Console
from rich.console import ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Heading, Markdown
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text


class SimpleCodeBlock(CodeBlock):
    """Customized code blocks in markdown.

    This avoids a background color which messes up copy-pasting and sets the language name as dim prefix and suffix.
    """

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        yield Text(self.lexer_name, style="dim")
        yield Syntax(code, self.lexer_name, theme=self.theme, word_wrap=True)
        yield Text(f"/{self.lexer_name}", style="dim")


class LeftHeading(Heading):
    """Customized headings in markdown to stop centering and prepend markdown style hashes."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # note we use `Style(bold=True)` not `self.style_name` here to disable underlining which is ugly IMHO
        yield Text(
            f"{'#' * int(self.tag[1:])} {self.text.plain}", style=Style(bold=True)
        )


Markdown.elements.update(
    fence=SimpleCodeBlock,
    heading_open=LeftHeading,
)


@dataclass
class LoggingToolset(WrapperToolset):
    """A toolset wrapper that logs tool calls using Rich console."""

    console: Console | None = None

    def __post_init__(self) -> None:
        """Initialize console after dataclass initialization."""
        if self.console is None:
            self.console = Console()

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        """Log tool calls and delegate to wrapped toolset."""
        # Check if any key contains "code" to handle keys like python_code, code_snippet, etc.
        code_key = next(
            (key for key in tool_args.keys() if "code" in key.lower()), None
        )

        if code_key and tool_args.get(code_key):
            code_string = f"```python\n{tool_args[code_key]}\n```"
            self.console.print(f"Calling code tool {name!r}")
            self.console.print(Markdown(code_string, code_theme="github-dark"))
        else:
            # Truncate long arguments for display
            truncated_args = self._truncate_args(tool_args)
            self.console.print(f"Calling tool {name!r} with args: {truncated_args}")

        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
            self.console.print(f"Finished calling tool {name!r}")
            return result
        except Exception as e:
            self.console.print(f"Error calling tool {name!r}: {e}")
            raise

    def _truncate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Truncate long argument values for display purposes.

        Args:
            args: Dictionary of tool arguments.

        Returns:
            dict[str, Any]: Dictionary with truncated string representations.
        """
        max_arg_length = 10000
        truncated: dict[str, Any] = {}
        for key, value in args.items():
            # Convert value to string representation
            str_value = repr(value)

            # Truncate if too long
            if len(str_value) > max_arg_length:
                truncated[key] = str_value[: max_arg_length - 3] + "..."
            else:
                truncated[key] = str_value

        return truncated
