"""Module entrypoint for the Monty Python REPL MCP server."""

from .server import mcp


if __name__ == "__main__":
    mcp.run(transport="stdio")
