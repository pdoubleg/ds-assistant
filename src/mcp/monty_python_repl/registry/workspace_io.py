"""General workspace, dataframe IO, and handle inspection collections."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from ..core.registry import WorkspaceToolCollection, tool

_TEXT_FILE_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".py",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".conf",
        ".cfg",
        ".csv",
        ".tsv",
        ".sql",
        ".sh",
    }
)


class WorkspaceFileCollection(WorkspaceToolCollection):
    """Workspace text file helpers for common authoring and inspection tasks."""

    name = "workspace"
    description = (
        "Read and write common text files inside /workspace and inspect available "
        "workspace files.\n"
        "Supported file extensions: " + ", ".join(sorted(_TEXT_FILE_SUFFIXES))
    )

    def _validate_text_path(self, path: str) -> PurePosixPath:
        """Validate that a path targets a supported workspace text file.

        Args:
            path (str): Relative or `/workspace`-scoped file path.

        Returns:
            PurePosixPath: Normalized virtual-style path object.

        Raises:
            ValueError: If the file extension is not supported.
        """
        virtual_path = PurePosixPath(path)
        suffix = virtual_path.suffix.lower()
        if suffix not in _TEXT_FILE_SUFFIXES:
            supported = ", ".join(sorted(_TEXT_FILE_SUFFIXES))
            raise ValueError(
                f"Unsupported text file type `{suffix or '[no extension]'}`. "
                f"Use one of: {supported}."
            )
        return virtual_path

    @tool
    def list_workspace_files(self, subdir: str = ".") -> list[str]:
        """List files currently available under `/workspace`.

        Args:
            subdir (str): Optional workspace subdirectory to search from.

        Returns:
            list[str]: Virtual workspace file paths.

        Examples:
            print(list_workspace_files("docs"))
        """
        host_root = self._resolve_host_path(subdir)
        if not host_root.exists():
            return []

        files: list[str] = []
        for child in sorted(host_root.rglob("*")):
            if child.is_file():
                files.append(str(self._os_access.virtualize_host_path(child)))
        return files

    @tool
    def read_workspace_text(
        self,
        path: str,
        *,
        max_chars: int = 200_000_000,
    ) -> dict[str, Any]:
        """Read a supported text file from `/workspace`.

        Args:
            path (str): Relative or `/workspace`-scoped text file path.
            max_chars (int): Maximum number of characters to return before truncating.

        Returns:
            dict[str, Any]: File contents and summary metadata.

        Examples:
            result = read_workspace_text('/workspace/docs/notes.md')
            # Returns:
            # {
            #     "path": "/workspace/docs/notes.md",
            #     "content": "# Notes\\nReady to go",
            #     "character_count": 18,
            #     "truncated": False
            # }

            result = read_workspace_text('/workspace/large_file.txt', max_chars=100)
            # Returns (if file is larger than 100 chars):
            # {
            #     "path": "/workspace/large_file.txt",
            #     "content": "First 100 characters...\\n... [truncated]",
            #     "character_count": 5000,
            #     "truncated": True
            # }
        """
        if max_chars <= 0:
            raise ValueError("`max_chars` must be greater than zero.")

        self._validate_text_path(path)
        host_path = self._resolve_host_path(path)
        text = host_path.read_text(encoding="utf-8")
        truncated = len(text) > max_chars
        content = text[:max_chars]
        if truncated:
            content += "\n... [truncated]"
        return {
            "path": str(self._os_access.virtualize_host_path(host_path)),
            "content": content,
            "character_count": len(text),
            "truncated": truncated,
        }

    @tool
    def write_workspace_text(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Write a supported text file inside `/workspace`.

        Args:
            path (str): Relative or `/workspace`-scoped text file path.
            content (str): UTF-8 text to persist.
            overwrite (bool): Whether an existing file may be replaced.

        Returns:
            dict[str, Any]: Saved path and write summary metadata.

        Examples:
            result = write_workspace_text("/workspace/docs/notes.md", "Ready to go")
            # Returns:
            # {
            #     "path": "/workspace/docs/notes.md",
            #     "character_count": 11
            # }
        """
        self._validate_text_path(path)
        host_path = self._resolve_host_path(path)
        if host_path.exists() and host_path.is_dir():
            raise IsADirectoryError(f"{path!r} points to a directory.")
        if host_path.exists() and not overwrite:
            raise FileExistsError(f"{path!r} already exists.")
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_text(content, encoding="utf-8")
        self._record_artifact(host_path)
        return {
            "path": str(self._os_access.virtualize_host_path(host_path)),
            "character_count": len(content),
        }

    @tool
    def read_workspace_json(self, path: str) -> dict[str, Any]:
        """Read and parse a JSON file from `/workspace`.

        Args:
            path (str): Relative or `/workspace`-scoped JSON file path.

        Returns:
            dict[str, Any]: Parsed JSON payload and file path metadata.

        Examples:
            payload = read_workspace_json("/workspace/config/settings.json")
            # Returns:
            # {
            #     "path": "/workspace/config/settings.json",
            #     "data": {
            #         "mode": "demo",
            #         "retries": 2
            #     }
            # }
        """
        if PurePosixPath(path).suffix.lower() != ".json":
            raise ValueError("`read_workspace_json` only supports `.json` files.")
        host_path = self._resolve_host_path(path)
        return {
            "path": str(self._os_access.virtualize_host_path(host_path)),
            "data": json.loads(host_path.read_text(encoding="utf-8")),
        }

    @tool
    def write_workspace_json(
        self,
        path: str,
        data: Any,
        *,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Serialize JSON data into a workspace file.

        Args:
            path (str): Relative or `/workspace`-scoped JSON file path.
            data (Any): JSON-serializable object to persist.
            overwrite (bool): Whether an existing file may be replaced.

        Returns:
            dict[str, Any]: Saved path and write summary metadata.

        Examples:
            result = write_workspace_json(
                "/workspace/config/settings.json",
                {"mode": "demo", "retries": 2, "features": {"beta": True}},
            )
            # Returns:
            # {
            #     "path": "/workspace/config/settings.json",
            #     "character_count": 82
            # }
        """
        if PurePosixPath(path).suffix.lower() != ".json":
            raise ValueError("`write_workspace_json` only supports `.json` files.")
        rendered = json.dumps(data, indent=2, ensure_ascii=True, sort_keys=True)
        return self.write_workspace_text(path, rendered, overwrite=overwrite)


class DataIOCollection(WorkspaceToolCollection):
    """General dataframe file loading and export helpers."""

    name = "data_io"
    description = "Load and save pandas dataframes as CSV and Excel files."

    @tool
    def load_csv(self, path: str, *, nrows: int | None = None) -> str:
        """Load a CSV file from `/workspace` and return a dataframe handle.

        Args:
            path (str): Relative or `/workspace`-scoped path to the CSV file.
            nrows (int | None): Optional maximum row count to load.

        Returns:
            str: Handle for the stored dataframe.

        Examples:
            df_handle = load_csv("/workspace/input/data.csv")
        """
        dataframe = pd.read_csv(self._resolve_host_path(path), nrows=nrows)
        return self._object_store.put(dataframe, prefix="df")

    @tool
    def load_excel(
        self,
        path: str,
        *,
        sheet_name: str | int = 0,
        nrows: int | None = None,
    ) -> str:
        """Load an Excel worksheet from `/workspace` and return a dataframe handle.

        Args:
            path (str): Relative or `/workspace`-scoped path to the workbook.
            sheet_name (str | int): Sheet name or zero-based sheet index to load.
            nrows (int | None): Optional maximum row count to load.

        Returns:
            str: Handle for the stored dataframe.

        Examples:
            df_handle = load_excel("/workspace/input/report.xlsx", sheet_name="claims")
        """
        dataframe = pd.read_excel(
            self._resolve_host_path(path),
            sheet_name=sheet_name,
            nrows=nrows,
        )
        return self._object_store.put(dataframe, prefix="df")

    @tool
    def save_csv(self, dataframe_handle: str, path: str, *, index: bool = False) -> str:
        """Save a stored dataframe handle to CSV inside `/workspace`.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            path (str): Relative or `/workspace`-scoped CSV destination.
            index (bool): Whether to persist the dataframe index.

        Returns:
            str: Virtual path to the saved CSV file.

        Examples:
            save_csv(df_handle, "/workspace/output/clean.csv")
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        self._get_dataframe(dataframe_handle).to_csv(host_path, index=index)
        self._record_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool
    def save_excel(
        self,
        dataframes: dict[str, str],
        path: str,
        *,
        index: bool = False,
    ) -> str:
        """Save one or more stored dataframes to an Excel workbook.

        Args:
            dataframes (dict[str, str]): Mapping of sheet names to dataframe handles.
            path (str): Relative or `/workspace` workbook destination.
            index (bool): Whether to include dataframe indices in each sheet.

        Returns:
            str: Virtual path to the saved workbook.

        Examples:
            save_excel(
                {"raw_claims": df_handle, "segment_summary": summary_handle},
                "/workspace/output/report.xlsx",
            )
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(host_path) as writer:
            for sheet_name, dataframe_handle in dataframes.items():
                self._get_dataframe(dataframe_handle).to_excel(
                    writer,
                    sheet_name=str(sheet_name)[:31],
                    index=index,
                )
        self._record_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))


class HandleInspectionCollection(WorkspaceToolCollection):
    """Handle inspection helpers for stored in-memory artifacts."""

    name = "handles"
    description = "Inspect active dataframe and figure handles stored by Monty."

    @tool
    def list_object_handles(self) -> list[str]:
        """List the dataframe and figure handles currently stored in memory.

        Returns:
            list[str]: Active object handles in insertion order.

        Examples:
            print(list_object_handles())
        """
        return self._object_store.list_handles()

    @tool
    def inspect_handle(self, handle: str) -> dict[str, Any]:
        """Return a summary of a stored host-side object.

        Args:
            handle (str): Dataframe or figure handle to inspect.

        Returns:
            dict[str, Any]: JSON-friendly handle summary.

        Examples:
            print(inspect_handle(df_handle))
            # Returns:
            # {
            #     "handle": "df_1",
            #     "value": {
            #         "type": "DataFrame",
            #         "shape": [100, 12],
            #         "columns": ["target", "score"]
            #     }
            # }
        """
        return self._object_store.summary(handle)


__all__ = [
    "DataIOCollection",
    "HandleInspectionCollection",
    "WorkspaceFileCollection",
]
