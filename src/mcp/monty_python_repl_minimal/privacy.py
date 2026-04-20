"""Privacy and redaction helpers for the minimal Monty REPL."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

_DEFAULT_MAX_ITEMS = 200
_DEFAULT_MAX_CHARS = 500
_SANDBOX_LINE_PATTERN = re.compile(r"<[^>]+>.*?line (\d+)")


def _truncate_string(value: str, *, max_chars: int) -> str:
    """Return a safely truncated string value.

    Args:
        value: Raw text value.
        max_chars: Maximum retained character count.

    Returns:
        Truncated string with an explicit marker when needed.
    """
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "... [truncated]"


def summarize_stdout(text: str) -> dict[str, Any]:
    """Summarize captured stdout without returning raw contents.

    Args:
        text: Raw stdout captured during execution.

    Returns:
        Compact metadata describing the captured output.
    """
    line_count = len([line for line in text.splitlines() if line.strip()])
    return {
        "suppressed": bool(text),
        "line_count": line_count,
        "character_count": len(text),
        "message": (
            "Captured stdout was suppressed for privacy."
            if text
            else "No stdout was captured."
        ),
    }


def sanitize_exception(
    error: Exception,
    *,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    """Return a privacy-safe exception summary.

    Args:
        error: Original exception instance.
        traceback_text: Optional formatted traceback string.

    Returns:
        Sanitized error metadata that omits runtime values.
    """
    line_number = None
    if traceback_text:
        match = _SANDBOX_LINE_PATTERN.search(traceback_text)
        if match:
            line_number = int(match.group(1))

    return {
        "error_type": type(error).__name__,
        "line_number": line_number,
        "message": (
            f"{type(error).__name__} raised during execution. "
            "Raw exception details were suppressed for privacy."
        ),
    }


def summarize_series(series: pd.Series) -> dict[str, Any]:
    """Return a privacy-safe summary for a pandas series.

    Args:
        series: Series to summarize.

    Returns:
        Aggregate series metadata without row-level values.
    """
    non_null = series.dropna()
    summary: dict[str, Any] = {
        "type": "Series",
        "name": str(series.name),
        "dtype": str(series.dtype),
        "length": int(len(series)),
        "missing_count": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()) if len(series) else 0.0,
        "unique_count": int(non_null.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        described = non_null.astype(float).describe(percentiles=[0.25, 0.5, 0.75])
        summary["numeric_summary"] = {
            "count": float(described.get("count", 0.0)),
            "mean": float(described.get("mean", 0.0)),
            "std": float(described.get("std", 0.0))
            if not np.isnan(described.get("std", np.nan))
            else None,
            "min": float(described.get("min", 0.0)),
            "p25": float(described.get("25%", 0.0)),
            "p50": float(described.get("50%", 0.0)),
            "p75": float(described.get("75%", 0.0)),
            "max": float(described.get("max", 0.0)),
        }
    return summary


def _summarize_numeric_non_null(non_null: pd.Series) -> dict[str, Any]:
    """Return aggregate numeric statistics for a non-null numeric series.

    Args:
        non_null: Numeric series with nulls already dropped.

    Returns:
        Aggregate numeric statistics.
    """
    if non_null.empty:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
        }

    described = non_null.astype(float).describe(percentiles=[0.25, 0.5, 0.75])
    return {
        "mean": float(described.get("mean", 0.0)),
        "std": float(described.get("std", 0.0))
        if not np.isnan(described.get("std", np.nan))
        else None,
        "min": float(described.get("min", 0.0)),
        "p25": float(described.get("25%", 0.0)),
        "p50": float(described.get("50%", 0.0)),
        "p75": float(described.get("75%", 0.0)),
        "max": float(described.get("max", 0.0)),
    }


def summarize_dataframe_column(
    dataframe: pd.DataFrame,
    column_name: str,
) -> dict[str, Any]:
    """Return a privacy-safe summary for one dataframe column.

    Args:
        dataframe: Dataframe containing the column.
        column_name: Column to summarize.

    Returns:
        Aggregate metadata for the requested column.
    """
    series = dataframe[column_name]
    non_null = series.dropna()
    column_summary: dict[str, Any] = {
        "column": column_name,
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()) if len(series) else 0.0,
        "non_null_count": int(non_null.shape[0]),
        "unique_count": int(non_null.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        column_summary["numeric_summary"] = _summarize_numeric_non_null(non_null)
    elif pd.api.types.is_datetime64_any_dtype(series):
        if not non_null.empty:
            column_summary["datetime_summary"] = {
                "min": str(non_null.min()),
                "max": str(non_null.max()),
            }
    else:
        # Categorical/object columns only expose aggregate structure.
        top_frequency = (
            float(non_null.value_counts(normalize=True, dropna=True).iloc[0])
            if not non_null.empty
            else 0.0
        )
        column_summary["categorical_summary"] = {
            "cardinality": int(non_null.nunique(dropna=True)),
            "top_frequency": top_frequency,
        }
    return column_summary


def summarize_dataframe(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Return a lightweight privacy-safe dataframe overview.

    Args:
        dataframe: Dataframe to summarize.

    Returns:
        Aggregate dataframe metadata with schema-level counts and column names.
    """
    column_names = [str(column) for column in dataframe.columns]
    missing_counts = dataframe.isna().sum()
    column_type_counts = {
        "numeric": 0,
        "datetime": 0,
        "categorical": 0,
        "other": 0,
    }
    for column_name in column_names:
        series = dataframe[column_name]
        if pd.api.types.is_numeric_dtype(series):
            column_type_counts["numeric"] += 1
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_type_counts["datetime"] += 1
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(
            series
        ):
            column_type_counts["categorical"] += 1
        else:
            column_type_counts["other"] += 1

    return {
        "type": "DataFrame",
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": column_names,
        "column_type_counts": column_type_counts,
        "missingness": {
            "total_missing_cells": int(missing_counts.sum()),
            "columns_with_missing_values": int((missing_counts > 0).sum()),
            "fully_non_null_columns": int((missing_counts == 0).sum()),
            "max_column_missing_rate": (
                float((missing_counts / len(dataframe)).max())
                if len(dataframe)
                else 0.0
            ),
        },
        "usage_hint": (
            "Use summarize_dataframe_columns(...) with a focused column list "
            "when you need per-column detail."
        ),
    }


def summarize_dataframe_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict[str, Any]:
    """Return privacy-safe details for a focused subset of dataframe columns.

    Args:
        dataframe: Dataframe to summarize.
        columns: Requested columns to inspect in detail.

    Returns:
        Targeted per-column summary payload for the requested columns.
    """
    unique_columns: list[str] = []
    for column in columns:
        if column not in unique_columns:
            unique_columns.append(column)

    return {
        "type": "DataFrameColumnDetails",
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "requested_columns": unique_columns,
        "column_summaries": [
            summarize_dataframe_column(dataframe, column_name)
            for column_name in unique_columns
        ],
    }


def safe_json_value(
    value: Any,
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> Any:
    """Convert runtime values into privacy-safe JSON-friendly summaries.

    Args:
        value: Runtime value to summarize.
        max_items: Maximum list-like item count to retain.
        max_chars: Maximum retained characters for free-form strings.

    Returns:
        JSON-friendly, privacy-safe representation of the input.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_string(value, max_chars=max_chars)

    to_json_summary = getattr(value, "to_json_summary", None)
    if callable(to_json_summary):
        return to_json_summary(max_items=max_items, max_chars=max_chars)

    if isinstance(value, pd.DataFrame):
        return summarize_dataframe(value)

    if isinstance(value, pd.Series):
        return summarize_series(value)

    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype),
        }

    if isinstance(value, PurePosixPath | Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): safe_json_value(
                item,
                max_items=max_items,
                max_chars=max_chars,
            )
            for key, item in list(value.items())[:max_items]
        }

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            safe_json_value(item, max_items=max_items, max_chars=max_chars)
            for item in list(value)[:max_items]
        ]

    return {"type": type(value).__name__}


__all__ = [
    "safe_json_value",
    "sanitize_exception",
    "summarize_dataframe",
    "summarize_dataframe_column",
    "summarize_dataframe_columns",
    "summarize_series",
    "summarize_stdout",
]
