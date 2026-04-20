"""Data loading tools for the minimal registry package."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from ..base import WorkspaceToolCollection
from ..core.registry import tool
from ..privacy import summarize_dataframe
from .base import DataReadConfig


class DataAccessCollection(WorkspaceToolCollection):
    """Safe data-access helpers for local CSV and partial parquet reads."""

    name = "data_access"
    description = (
        "Load local CSV files or partial parquet slices into dataframe handles "
        "without returning raw row previews to the model."
    )

    @tool
    def load_csv(
        self,
        path: str,
        *,
        nrows: int | None = None,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Load a CSV file from `/workspace` into a reusable dataframe handle.

        Use this when a workflow begins from a local CSV and you want a privacy-safe
        dataframe summary instead of raw row previews.

        Args:
            path (str): Relative or `/workspace` path to the CSV file.
            nrows (int | None): Optional maximum number of rows to load.
            columns (list[str] | None): Optional subset of columns to read.

        Returns:
            dict[str, Any]: Dataframe handle plus a lightweight safe dataframe
            overview.

        Examples:
            ```python
            dataset = load_csv("/workspace/train.csv", nrows=5000)
            df_handle = dataset["dataframe_handle"]
            # Returns
            # {
            #     "dataframe_handle": "df_abc123",
            #     "summary": {
            #         "type": "DataFrame",
            #         "shape": [5000, 10],
            #         "columns": ["customer_id", "balance", "target"],
            #         "column_type_counts": {
            #             "numeric": 2,
            #             "datetime": 0,
            #             "categorical": 1,
            #             "other": 0,
            #         },
            #         "usage_hint": "Use summarize_dataframe_columns(...) ...",
            #     }
            # }
            ```
        """

        dataframe = pd.read_csv(
            self._resolve_host_path(path),
            nrows=nrows,
            usecols=columns,
        )
        handle = self._object_store.put(dataframe, prefix="df")
        return {
            "dataframe_handle": handle,
            "summary": summarize_dataframe(dataframe),
        }

    @tool
    def load_parquet_slice(
        self,
        s3_uri: str,
        *,
        label_col: str,
        id_cols: list[str] | None = None,
        candidate_cols: list[str] | None = None,
        partition_filters: dict[str, list[Any] | Any] | None = None,
        sample_n_rows: int | None = None,
        max_files: int | None = None,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Load a sampled parquet slice from S3 or another parquet dataset URI.

        This tool is useful for hackathon-scale exploration when the full dataset is
        too large to read. It keeps the label column, optional ID columns, and either
        a requested feature subset or the full schema.

        Args:
            s3_uri (str): S3 URI or workspace path pointing to a parquet dataset.
            label_col (str): Target column that must be included in the result.
            id_cols (list[str] | None): Optional identifier columns to retain.
            candidate_cols (list[str] | None): Optional explicit feature subset.
            partition_filters (dict[str, list[Any] | Any] | None): Optional equality
                or membership filters applied to partition columns.
            sample_n_rows (int | None): Optional maximum sampled row count.
            max_files (int | None): Optional maximum number of fragments to scan.
            random_seed (int): Seed used for fragment and row sampling.

        Returns:
            dict[str, Any]: Dataframe handle plus a lightweight safe dataframe
            overview.

        Examples:
            ```python
            parquet_sample = load_parquet_slice(
                "s3://bucket/train_dataset",
                label_col="target",
                id_cols=["customer_id"],
                candidate_cols=["segment", "balance", "utilization"],
                partition_filters={"snapshot_date": "2026-04-01", "country": ["US", "CA"]},
                sample_n_rows=10000,
            )
            # Returns
            # {
            #     "dataframe_handle": "df_abc123",
            #     "summary": {
            #         "type": "DataFrame",
            #         "shape": [10000, 10],
            #         "columns": ["customer_id", "balance", "target"],
            #         "column_type_counts": {
            ```
        """

        try:
            import pyarrow as pa
            import pyarrow.dataset as ds
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "pyarrow is required for parquet loading in the hackathon Monty REPL."
            ) from exc

        config = DataReadConfig(
            s3_uri=s3_uri,
            label_col=label_col,
            id_cols=list(id_cols or []),
            candidate_cols=list(candidate_cols) if candidate_cols else None,
            partition_filters=partition_filters,
            sample_n_rows=sample_n_rows,
            max_files=max_files,
            random_seed=random_seed,
        )
        dataset_uri: str | Path
        if config.s3_uri.startswith("s3://"):
            dataset_uri = config.s3_uri
        else:
            dataset_uri = self._resolve_host_path(config.s3_uri)

        dataset = ds.dataset(dataset_uri, format="parquet")
        filter_expression = None
        if config.partition_filters:
            expressions = []
            for key, raw_value in config.partition_filters.items():
                field = ds.field(str(key))
                if isinstance(raw_value, list):
                    expressions.append(field.isin(list(raw_value)))
                else:
                    expressions.append(field == raw_value)
            for expression in expressions:
                filter_expression = (
                    expression
                    if filter_expression is None
                    else filter_expression & expression
                )

        fragments = list(dataset.get_fragments(filter=filter_expression))
        if config.max_files is not None and len(fragments) > config.max_files:
            rng = random.Random(config.random_seed)
            fragments = rng.sample(fragments, config.max_files)

        schema_names = set(dataset.schema.names)
        columns = [config.label_col] + list(config.id_cols)
        if config.candidate_cols is None:
            columns.extend(
                [name for name in dataset.schema.names if name not in columns]
            )
        else:
            columns.extend(config.candidate_cols)

        resolved_columns: list[str] = []
        seen_columns: set[str] = set()
        for column in columns:
            if column in schema_names and column not in seen_columns:
                resolved_columns.append(column)
                seen_columns.add(column)

        tables: list[Any] = []
        rows_collected = 0
        for fragment in fragments:
            scanner = ds.Scanner.from_fragment(
                fragment,
                columns=resolved_columns,
                filter=filter_expression,
                use_threads=True,
            )
            table = scanner.to_table()
            if table.num_rows == 0:
                continue
            tables.append(table)
            rows_collected += int(table.num_rows)
            if (
                config.sample_n_rows is not None
                and rows_collected >= config.sample_n_rows
            ):
                break

        if not tables:
            raise ValueError("No rows were read from the parquet dataset.")

        combined = pa.concat_tables(tables, promote_options="default")
        if (
            config.sample_n_rows is not None
            and combined.num_rows > config.sample_n_rows
        ):
            rng = np.random.default_rng(config.random_seed)
            take_idx = rng.choice(
                combined.num_rows,
                size=config.sample_n_rows,
                replace=False,
            )
            combined = combined.take(pa.array(np.sort(take_idx)))

        dataframe = combined.to_pandas(types_mapper=None)
        handle = self._object_store.put(dataframe, prefix="df")
        return {
            "dataframe_handle": handle,
            "summary": summarize_dataframe(dataframe),
        }

    @tool
    def select_columns(
        self,
        dataframe_handle: str,
        columns: list[str],
    ) -> dict[str, Any]:
        """Create a new dataframe handle with only the requested columns.

        Use this to reduce memory footprint or pass a smaller feature set into a
        later screening or modeling workflow.

        Args:
            dataframe_handle (str): Source dataframe handle.
            columns (list[str]): Columns to retain in the output dataframe.

        Returns:
            dict[str, Any]: New dataframe handle plus a lightweight safe dataframe
            overview.

        Examples:
            ```python
            dataset = load_csv("/workspace/train.csv")
            slim = select_columns(
                dataset["dataframe_handle"],
                ["customer_id", "balance", "target"],
            )
            # Returns
            # {
            #     "dataframe_handle": "df_abc123",
            #     "summary": {
            #         "type": "DataFrame",
            #         "shape": [10000, 3],
            #         "columns": ["customer_id", "balance", "target"],
            #         "column_type_counts": {
            #             "numeric": 2,
            #             "datetime": 0,
            #             "categorical": 1,
            #             "other": 0,
            #         },
            #     }
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle)
        missing_columns = [
            column for column in columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")
        result = dataframe[columns].copy()
        handle = self._object_store.put(result, prefix="df")
        return {
            "dataframe_handle": handle,
            "summary": summarize_dataframe(result),
        }


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
        "workspace files. Prefer these helpers over direct `open(...)`, "
        "`Path.write_text(...)`, `Path.read_text(...)`, or similar stdlib file "
        "operations inside the sandbox.\n"
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
            ```python
            workspace_files = list_workspace_files("docs")
            # Returns
            # {
            #     "files": ["/workspace/docs/notes.md", "/workspace/docs/README.md"]
            # }
            ```
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

        Use this helper instead of direct `open(...)` or `Path.read_text(...)`
        calls when you need workspace file contents inside the sandbox. Standard
        Python file APIs may be restricted and can fail with privacy-sanitized
        errors.

        Args:
            path (str): Relative or `/workspace`-scoped text file path.
            max_chars (int): Maximum number of characters to return before truncating.

        Returns:
            dict[str, Any]: File contents and summary metadata.

        Examples:
            ```python
            notes = read_workspace_text("/workspace/docs/notes.md")
            # Returns
            # {
            #     "path": "/workspace/docs/notes.md",
            #     "content": "# Notes\\nReady to go",
            #     "character_count": 18,
            #     "truncated": False
            # }
            preview = read_workspace_text("/workspace/large_file.txt", max_chars=100)
            # Returns
            # {
            #     "path": "/workspace/large_file.txt",
            #     "content": "First 100 characters...\\n... [truncated]",
            #     "character_count": 5000,
            #     "truncated": True
            # }
            ```
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

        Use this helper to create scripts, notes, config fragments, prompts, and
        other text assets under `/workspace`. Prefer it over direct `open(...)`,
        `Path.write_text(...)`, or other stdlib file-write patterns inside the
        sandbox because those operations may be restricted or unavailable.

        Args:
            path (str): Relative or `/workspace`-scoped text file path.
            content (str): UTF-8 text to persist.
            overwrite (bool): Whether an existing file may be replaced.

        Returns:
            dict[str, Any]: Saved path and write summary metadata.

        Examples:
            ```python
            write_result = write_workspace_text(
                "/workspace/scripts/train.py",
                "print('ready')\\n",
            )
            # Returns
            # {
            #     "path": "/workspace/scripts/train.py",
            #     "character_count": 11
            # }
            saved_script = read_workspace_text(write_result["path"])
            # Returns
            # {
            #     "path": "/workspace/scripts/train.py",
            #     "content": "print('ready')\\n",
            #     "character_count": 11,
            #     "truncated": False
            # }
            ```
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

        Prefer this helper over manual `json.loads(...)` plus direct file reads in
        the sandbox when you need a JSON payload from `/workspace`.

        Args:
            path (str): Relative or `/workspace`-scoped JSON file path.

        Returns:
            dict[str, Any]: Parsed JSON payload and file path metadata.

        Examples:
            ```python
            payload = read_workspace_json("/workspace/config/settings.json")
            # Returns
            # {
            #     "path": "/workspace/config/settings.json",
            #     "data": {"mode": "demo", "retries": 2}
            # }
            ```
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

        Prefer this helper over manual `json.dumps(...)` plus direct file writes
        in the sandbox. It is the supported path for persisting JSON under
        `/workspace`.

        Args:
            path (str): Relative or `/workspace`-scoped JSON file path.
            data (Any): JSON-serializable object to persist.
            overwrite (bool): Whether an existing file may be replaced.

        Returns:
            dict[str, Any]: Saved path and write summary metadata.

        Examples:
            ```python
            result = write_workspace_json(
                "/workspace/config/settings.json",
                {"mode": "demo", "retries": 2, "features": {"beta": True}},
            )
            # Returns
            # {
            #     "path": "/workspace/config/settings.json",
            #     "character_count": 82
            # }
            saved_config = read_workspace_json(result["path"])
            # Returns
            # {
            #     "path": "/workspace/config/settings.json",
            #     "data": {"mode": "demo", "retries": 2, "features": {"beta": True}}
            # }
            ```
        """
        if PurePosixPath(path).suffix.lower() != ".json":
            raise ValueError("`write_workspace_json` only supports `.json` files.")
        rendered = json.dumps(data, indent=2, ensure_ascii=True, sort_keys=True)
        return self.write_workspace_text(path, rendered, overwrite=overwrite)


__all__ = ["DataAccessCollection", "WorkspaceFileCollection"]
