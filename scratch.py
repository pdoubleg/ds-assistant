import asyncio
import os
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.mcp import MCPServerStdio


instructions = """
You are a helpful assistant. Use the provided tools to explore the codebase and answer the user's question.
"""


run_python = MCPServerStdio(
    "deno",
    args=[
        "run",
        "-N",
        "-R=node_modules",
        "-W=node_modules",
        "--node-modules-dir=auto",
        "jsr:@pydantic/mcp-run-python",
        "stdio",
    ],
)

internet_search = MCPServerStdio(command="uvx", args=["duckduckgo-mcp-server"])

code_reasoning = MCPServerStdio(
    command="npx",
    args=["-y", "@mettamatt/code-reasoning"],
    tool_prefix="code_reasoning",
)


agent = Agent(
    instructions=instructions,
    model="openai:gpt-4.1",
    output_type=str,
    retries=5,
    toolsets=[
        run_python,
        internet_search,
        code_reasoning,
    ],
)

@agent.tool_plain
def discover_files(directory: str = ".") -> str:
    """
    Discover files and folders in the specified directory with detailed information. Call this tool 
    as many times as needed to explore the codebase or find a particular file of interest.
    
    Args:
        directory (str): The directory path to explore. Defaults to current directory.
        
    Returns:
        str: A formatted string containing the directory structure with file details and exact paths.
        
    """
    try:
        import os
        from pathlib import Path
        
        # Convert to Path object for better handling
        path = Path(directory).resolve()
        
        if not path.exists():
            return f"Error: Directory '{directory}' does not exist."
        
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        result = [f"Directory: {path}/"]
        
        # Get all items and sort them (directories first, then files)
        try:
            items = list(path.iterdir())
            # Sort: directories first, then files, both alphabetically
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))
            
            for item in items:
                # Skip hidden files and common build/cache directories
                if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules', '.git']:
                    continue
                
                # Get the exact path for downstream lookup
                exact_path = str(item)
                
                if item.is_dir():
                    # Count items in subdirectory for context
                    try:
                        subitem_count = len([x for x in item.iterdir() if not x.name.startswith('.')])
                        result.append(f"📁 {item.name}/ ({subitem_count} items) - Path: {exact_path}")
                    except PermissionError:
                        result.append(f"📁 {item.name}/ (access denied) - Path: {exact_path}")
                else:
                    # Get file size in human-readable format
                    try:
                        size = item.stat().st_size
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f} KB"
                        else:
                            size_str = f"{size / (1024 * 1024):.1f} MB"
                        
                        # Add file type emoji based on extension
                        ext = item.suffix.lower()
                        if ext in ['.py']:
                            emoji = '🐍'
                        elif ext in ['.js', '.ts']:
                            emoji = '📜'
                        elif ext in ['.json', '.yaml', '.yml']:
                            emoji = '⚙️'
                        elif ext in ['.md', '.txt']:
                            emoji = '📝'
                        elif ext in ['.css', '.scss']:
                            emoji = '🎨'
                        elif ext in ['.html']:
                            emoji = '🌐'
                        else:
                            emoji = '📄'
                        
                        result.append(f"{emoji} {item.name} ({size_str}) - Path: {exact_path}")
                    except Exception as e:
                        ModelRetry(f"Error getting file size: {str(e)}")
            
            if len(result) == 1:  # Only the directory header
                result.append("empty directory")
                
        except PermissionError as e:
            return f"Error: Permission denied accessing '{directory}'"
        
        return "\n".join(result)
        
    except Exception as e:
        ModelRetry(f"Error discovering files: {str(e)}")


@agent.tool_plain
def read_file(file_path: str) -> str:
    """
    Read the contents of a file and return them as a string.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        str: Contents of the file
        
    Raises:
        ModelRetry: If file cannot be read after trying multiple encodings
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue  # Try next encoding
        except Exception as e:
            # For non-encoding related errors like permissions or file not found
            ModelRetry(f"Error reading file: {str(e)}")
            
    # If we've tried all encodings and none worked
    ModelRetry(f"Error reading file: Could not decode file with any of these encodings: {encodings}")


async def main():
    await agent.to_cli()


if __name__ == "__main__":
    asyncio.run(main())