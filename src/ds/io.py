"""Data access helpers for the standalone ds package."""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .config import DataReadConfig, ParquetReadMetadata


def read_csv(
    path: str | Path,
    *,
    nrows: int | None = None,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file into a dataframe.

    Args:
        path: Local CSV path.
        nrows: Optional maximum number of rows to read.
        columns: Optional subset of columns to project.
        **kwargs: Additional keyword arguments forwarded to ``pd.read_csv``.

    Returns:
        Loaded dataframe.
    """

    return pd.read_csv(Path(path), nrows=nrows, usecols=columns, **kwargs)


def infer_parquet_feature_columns(
    uri: str | Path, *, target_column: str, id_columns: list[str] | None = None
) -> list[str]:
    excluded = {target_column, *list(id_columns or [])}
    schema_names = list(ds.dataset(uri, format="parquet").schema.names)
    return [col for col in schema_names if col not in excluded]


def read_parquet_fragment(
    uri: str | Path,
    *,
    columns: list[str] | None = None,
    partition_filters: dict[str, list[Any] | Any] | None = None,
    sample_n_rows: int | None = None,
    max_fragments: int | None = None,
    batch_size: int = 65_536,
    sample_strategy: Literal["head", "reservoir"] = "head",
    random_seed: int = 42,
    include_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, ParquetReadMetadata]:
    """Read a sampled fragment of a parquet dataset.

    Args:
        uri: Local parquet path or remote URI such as ``s3://bucket/dataset``.
        columns: Optional projected column subset.
        partition_filters: Optional equality or membership filters applied before
            scanning fragments.
        sample_n_rows: Optional maximum sampled row count.
        max_fragments: Optional maximum number of parquet fragments to scan.
        batch_size: Maximum Arrow record-batch size read from each fragment.
        sample_strategy: Sampling strategy used when ``sample_n_rows`` is set.
            Use ``"head"`` to stop after the first matching rows, or
            ``"reservoir"`` to scan selected fragments and keep a deterministic
            random sample without materializing the full dataset.
        random_seed: Random seed used for deterministic fragment and row sampling.
        include_metadata: Whether to also return fragment read metadata.

    Returns:
        A dataframe, or ``(dataframe, metadata)`` when ``include_metadata=True``.

    Example:
        >>> df = read_parquet_fragment("train.parquet", sample_n_rows=10_000)
        >>> sample = read_parquet_fragment(
        ...     "train.parquet",
        ...     sample_n_rows=10_000,
        ...     sample_strategy="reservoir",
        ... )
    """

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("pyarrow is required for parquet loading.") from exc

    config = DataReadConfig(
        uri=str(uri),
        columns=list(columns) if columns else None,
        partition_filters=partition_filters,
        sample_n_rows=sample_n_rows,
        max_fragments=max_fragments,
        batch_size=batch_size,
        sample_strategy=sample_strategy,
        random_seed=random_seed,
    )
    if config.sample_n_rows is not None and config.sample_n_rows <= 0:
        raise ValueError("sample_n_rows must be positive when provided.")
    if config.max_fragments is not None and config.max_fragments <= 0:
        raise ValueError("max_fragments must be positive when provided.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.sample_strategy not in {"head", "reservoir"}:
        raise ValueError("sample_strategy must be either 'head' or 'reservoir'.")

    dataset_uri: str | Path = (
        config.uri if str(config.uri).startswith("s3://") else Path(config.uri)
    )
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
    available_fragment_count = len(fragments)
    if config.max_fragments is not None and len(fragments) > config.max_fragments:
        rng = random.Random(config.random_seed)
        fragments = rng.sample(fragments, config.max_fragments)

    resolved_columns: list[str]
    if config.columns is None:
        resolved_columns = list(dataset.schema.names)
    else:
        schema_names = set(dataset.schema.names)
        resolved_columns = [
            column for column in config.columns if column in schema_names
        ]

    tables: list[Any] = []
    rows_collected = 0
    scanned_fragment_count = 0
    rng = np.random.default_rng(config.random_seed)
    for fragment in fragments:
        scanned_fragment_count += 1
        scanner = ds.Scanner.from_fragment(
            fragment,
            columns=resolved_columns or None,
            filter=filter_expression,
            use_threads=True,
            batch_size=config.batch_size,
        )

        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue

            if config.sample_n_rows is None:
                table = pa.Table.from_batches([batch])
            elif config.sample_strategy == "head":
                remaining_rows = config.sample_n_rows - rows_collected
                if remaining_rows <= 0:
                    break
                table = pa.Table.from_batches([batch.slice(0, remaining_rows)])
            else:
                # Keep only the smallest random keys seen so far. This gives a
                # deterministic random sample while holding at most sample rows
                # plus one Arrow batch in memory.
                table = pa.Table.from_batches([batch]).append_column(
                    "__ds_sample_key", pa.array(rng.random(batch.num_rows))
                )

            if table.num_rows == 0:
                continue

            tables.append(table)
            rows_collected += int(table.num_rows)

            if (
                config.sample_n_rows is not None
                and config.sample_strategy == "reservoir"
                and rows_collected > config.sample_n_rows
            ):
                combined_candidates = pa.concat_tables(
                    tables, promote_options="default"
                )
                sample_keys = combined_candidates["__ds_sample_key"].to_numpy()
                take_idx = np.argpartition(sample_keys, config.sample_n_rows - 1)[
                    : config.sample_n_rows
                ]
                take_idx = take_idx[np.argsort(sample_keys[take_idx])]
                tables = [combined_candidates.take(pa.array(take_idx))]
                rows_collected = config.sample_n_rows

            if (
                config.sample_n_rows is not None
                and config.sample_strategy == "head"
                and rows_collected >= config.sample_n_rows
            ):
                break

        if (
            config.sample_n_rows is not None
            and config.sample_strategy == "head"
            and rows_collected >= config.sample_n_rows
        ):
            break

    if not tables:
        raise ValueError("No rows were read from the parquet dataset.")

    combined = pa.concat_tables(tables, promote_options="default")
    if "__ds_sample_key" in combined.column_names:
        combined = combined.drop_columns(["__ds_sample_key"])

    dataframe = combined.to_pandas(types_mapper=None)
    metadata = ParquetReadMetadata(
        source_uri=str(uri),
        resolved_columns=list(dataframe.columns),
        available_fragment_count=available_fragment_count,
        scanned_fragment_count=scanned_fragment_count,
        row_count=int(len(dataframe)),
    )
    if include_metadata:
        return dataframe, metadata
    return dataframe


def select_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of a dataframe with only the requested columns.

    Args:
        dataframe: Source dataframe.
        columns: Ordered columns to retain.

    Returns:
        Reduced dataframe.
    """

    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")
    return dataframe[columns].copy()


def estimate_dataframe_memory(dataframe: pd.DataFrame, *, deep: bool = True) -> int:
    """Estimate dataframe memory usage in bytes.

    Args:
        dataframe: Dataframe to inspect.
        deep: Whether to include deep object memory usage.

    Returns:
        Estimated memory usage in bytes.
    """

    return int(dataframe.memory_usage(deep=deep).sum())


def cleanup_memory(*objects: Any, aggressive: bool = False) -> None:
    """Run explicit memory cleanup after dropping large intermediates.

    Args:
        *objects: Optional objects whose in-place contents should be cleared before
            garbage collection.
        aggressive: Whether to clear dataframe contents in place before running
            ``gc.collect()``.

    Returns:
        None

    Example:
        >>> frame = pd.DataFrame({"x": [1, 2, 3]})
        >>> cleanup_memory(frame, aggressive=True)
    """

    if aggressive:
        # Clearing frames in place can help notebooks release large intermediates
        # once the caller has decided they are no longer needed.
        for obj in objects:
            if isinstance(obj, pd.DataFrame):
                obj.drop(index=obj.index, inplace=True)
                obj.drop(columns=list(obj.columns), inplace=True, errors="ignore")
    gc.collect()


__all__ = [
    "cleanup_memory",
    "estimate_dataframe_memory",
    "read_csv",
    "read_parquet_fragment",
    "select_columns",
]
