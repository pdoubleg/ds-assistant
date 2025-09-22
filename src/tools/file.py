
import re
from pathlib import Path
from typing import Union

from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai import ModelRetry



def resolve_path(path_input: Union[str, Path]) -> Path:
    """Resolve a path to an absolute, normalized Path."""
    
    p = Path(path_input)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def format_path_for_display(path: Path, relative_to_cwd: bool = True) -> str:
    """
    Format a path for display to the user.

    Args:
        path: An absolute Path object
        relative_to_cwd: If True, show relative to cwd when possible

    Returns:
        A string representation of the path
    """
    if relative_to_cwd:
        try:
            # Try to make it relative to cwd for cleaner display
            cwd = Path.cwd()
            rel_path = path.relative_to(cwd)
            # If it's in the current directory, use ./ prefix for clarity
            if str(rel_path) == ".":
                return "./"
            elif not str(rel_path).startswith(".."):
                return str(rel_path)
        except ValueError:
            # Path is not relative to cwd, use absolute
            pass

    return str(path)


def read_file(file_path: str) -> str:
    """
    Read the contents of a file and return them as a string.

    Args:
        file_path: Path to the file to read

    Returns:
        str: Contents of the file, truncated if necessary

    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "ascii"]

    # Resolve to absolute path
    resolved_path = resolve_path(file_path)

    for encoding in encodings:
        try:
            with open(resolved_path, "r", encoding=encoding) as file:
                contents = file.read()
                if len(contents) > 1_000_000:
                    return contents[:1_000_000] + "..."
                return contents
        except UnicodeDecodeError:
            continue  # Try next encoding
        except Exception as e:
            # For non-encoding related errors like permissions or file not found
            return f"Error reading file: {str(e)}. Path: {file_path}, Resolved Path: {resolved_path}"

    # If we've tried all encodings and none worked
    return f"Error reading file: Could not decode file with any of these encodings: {encodings}. Path: {file_path}, Resolved Path: {resolved_path}"


def inspect_directory(directory: str = ".") -> str:
    """
    Discover files and folders in a given directory with detailed information. Call this tool
    as many times as needed to explore the current directory or find a particular file of interest.

    Args:
        directory (str): The directory path to explore. Defaults to current directory.

    Returns:
        str: A formatted string containing the directory structure with file details and paths.

    """
    try:
        # Convert to Path object for better handling
        path = resolve_path(directory)

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
                if item.name.startswith(".") or item.name in [
                    "__pycache__",
                    "node_modules",
                    ".git",
                ]:
                    continue

                # Get the exact path for downstream lookup
                exact_path = str(item)

                if item.is_dir():
                    # Count items in subdirectory for context
                    try:
                        subitem_count = len(
                            [x for x in item.iterdir() if not x.name.startswith(".")]
                        )
                        result.append(
                            f"📁 {item.name}/ ({subitem_count} items) - Path: {exact_path}"
                        )
                    except PermissionError:
                        result.append(
                            f"📁 {item.name}/ (access denied) - Path: {exact_path}"
                        )
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
                        if ext in [".py"]:
                            emoji = "🐍"
                        elif ext in [".js", ".ts"]:
                            emoji = "📜"
                        elif ext in [".json", ".yaml", ".yml"]:
                            emoji = "⚙️"
                        elif ext in [".md", ".txt"]:
                            emoji = "📝"
                        elif ext in [".css", ".scss"]:
                            emoji = "🎨"
                        elif ext in [".html"]:
                            emoji = "🌐"
                        else:
                            emoji = "📄"

                        result.append(
                            f"{emoji} {item.name} ({size_str}) - Path: {exact_path}"
                        )
                    except Exception as _:
                        result.append(
                            f"{item.name} - Path: {exact_path}"
                        )

            if len(result) == 1:  # Only the directory header
                result.append("empty directory")

        except PermissionError as _:
            return f"Error: Permission denied accessing '{directory}'"

        return "\n".join(result)

    except Exception as e:
        ModelRetry(f"Error discovering files: {str(e)}")


def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file. Useful for markdown (.md) and code (.py, .js, .ts, .html, .css, .scss) files.

    Args:
        file_path: Path where the file should be written (relative or absolute)
        content: Content to write to the file

    Returns:
        Success message or error
    """
    try:
        # Resolve to absolute path
        path = resolve_path(file_path)

        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        size = len(content)
        result = {
            "success": True,
            "file_path": str(path.absolute()),
            "bytes_written": size,
        }
        return result
    except Exception as e:
        error_msg = f"Error writing file {file_path}: {str(e)}"
        raise ModelRetry(error_msg)


def edit_file(file_path: str, old_str: str, new_str: str) -> str:
    """
    Edit a file by replacing exact text matches.

    Args:
        file_path: Path to the file to edit (relative or absolute)
        old_str: The exact text to find and replace (must match exactly including whitespace)
        new_str: The new text to insert in place of old_str

    Returns:
        Success message or detailed error message
    """
    try:
        # Resolve to absolute path
        path = resolve_path(file_path)

        # Check if file exists
        if not path.exists():
            raise ModelRetry(f"File not found: {file_path}")

        # Check if it's a file (not directory)
        if not path.is_file():
            raise ModelRetry(f"Path is not a file: {file_path}")

        # Read the current content
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError as e:
            raise ModelRetry(f"Cannot read file (encoding issue): {str(e)}")

        # Check if old_str exists in the file
        if old_str not in content:
            # Provide helpful feedback
            lines = old_str.split("\n")
            if len(lines) > 1:
                # Multi-line search - check if any lines exist
                found_lines = []
                for line in lines:
                    if line.strip() and line.strip() in content:
                        found_lines.append(line.strip())

                if found_lines:
                    raise ModelRetry(
                        f"Exact text not found in file. "
                        f"Found similar lines but not exact match. "
                        f"Check whitespace and indentation. "
                        f"Found: {found_lines[:3]}"
                    )  # Show first 3 matches
                else:
                    raise ModelRetry(
                        "Text not found in file. None of the lines exist in the file."
                    )
            else:
                # Single line search - provide more context
                stripped = old_str.strip()
                if stripped and stripped in content:
                    raise ModelRetry(
                        f"Found similar text but not exact match. "
                        f"Check whitespace and indentation around: '{stripped[:50]}...'"
                    )
                else:
                    raise ModelRetry(f"Text not found in file: '{old_str[:100]}...'")

        # Check for multiple occurrences
        occurrences = content.count(old_str)
        if occurrences > 1:
            raise ModelRetry(
                f"Found {occurrences} occurrences of the text. "
                f"Please provide more context to make the match unique, "
                f"or use a different tool to replace all occurrences."
            )

        # Perform the replacement
        new_content = content.replace(
            old_str, new_str, 1
        )  # Replace only first occurrence

        # Write the updated content back
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            raise ModelRetry(f"Failed to write file: {str(e)}")

        result = {
            "success": True,
            "file_path": str(path.absolute()),
            "bytes_written": len(new_content),
        }
        return result

    except PermissionError:
        raise ModelRetry(f"Permission denied when accessing file: {file_path}")
    except Exception as e:
        error_msg = f"Error editing file {file_path}: {str(e)}"
        raise ModelRetry(error_msg)


def grep_a(file_path: str, pattern: str, after_lines: int = 20, before_lines: int = 0) -> str:
    """
    Grep a file for a pattern and return the lines before and after the match.

    Args:
        file_path: Path to the file to search
        pattern: Regular expression pattern to search for
        after_lines: Number of lines to include after each match
        before_lines: Number of lines to include before each match

    Returns:
        List of matched lines with context
    """
    # Resolve to absolute path
    file_path = resolve_path(file_path)
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise ModelRetry(f"File not found: {file_path}")
    except PermissionError:
        return f"Permission denied when accessing file: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

    match_lines = []
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            # Get lines before match, starting at 0
            start = max(0, i - before_lines)
            # Get lines after match, up to end of file
            end = min(len(lines), i + 1 + after_lines)
            match_lines.extend(lines[start:end])

    return "\n".join(match_lines)

file_tools = FunctionToolset(tools=[inspect_directory, write_file, read_file], max_retries=5)
