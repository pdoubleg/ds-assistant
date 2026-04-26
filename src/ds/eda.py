"""EDA helpers for notebook-first tabular workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a compact dataframe summary.

    Args:
        dataframe: Dataframe to summarize.

    Returns:
        Aggregate dataframe metadata suitable for quick notebook inspection.
    """

    numeric_count = int(
        sum(pd.api.types.is_numeric_dtype(dtype) for dtype in dataframe.dtypes)
    )
    datetime_count = int(
        sum(pd.api.types.is_datetime64_any_dtype(dtype) for dtype in dataframe.dtypes)
    )
    categorical_count = int(
        sum(
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            for dtype in dataframe.dtypes
        )
    )
    other_count = int(
        dataframe.shape[1] - numeric_count - datetime_count - categorical_count
    )
    total_cells = int(dataframe.shape[0] * dataframe.shape[1])
    total_missing = int(dataframe.isna().sum().sum())
    return {
        "shape": tuple(int(value) for value in dataframe.shape),
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": [str(column) for column in dataframe.columns],
        "column_type_counts": {
            "numeric": numeric_count,
            "datetime": datetime_count,
            "categorical": categorical_count,
            "other": other_count,
        },
        "missingness": {
            "total_missing_cells": total_missing,
            "missing_cell_rate": float(total_missing / total_cells)
            if total_cells
            else 0.0,
        },
    }


def summarize_dataframe_columns(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return per-column dataframe diagnostics.

    Args:
        dataframe: Dataframe to inspect.
        columns: Optional subset of columns to summarize.

    Returns:
        A dataframe with one diagnostic row per requested column.
    """

    selected_columns = list(columns or dataframe.columns)
    missing_columns = [
        column for column in selected_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    rows: list[dict[str, Any]] = []
    for column in selected_columns:
        series = dataframe[column]
        non_null = series.dropna()
        row: dict[str, Any] = {
            "column": str(column),
            "dtype": str(series.dtype),
            "row_count": int(len(series)),
            "non_null_count": int(non_null.shape[0]),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()) if len(series) else 0.0,
            "unique_count": int(non_null.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            described = pd.to_numeric(non_null, errors="coerce").describe(
                percentiles=[0.25, 0.5, 0.75]
            )
            row.update(
                {
                    "mean": float(described.get("mean", 0.0))
                    if len(non_null)
                    else None,
                    "std": float(described.get("std", 0.0))
                    if len(non_null) and not np.isnan(described.get("std", np.nan))
                    else None,
                    "min": float(described.get("min", 0.0)) if len(non_null) else None,
                    "p25": float(described.get("25%", 0.0)) if len(non_null) else None,
                    "p50": float(described.get("50%", 0.0)) if len(non_null) else None,
                    "p75": float(described.get("75%", 0.0)) if len(non_null) else None,
                    "max": float(described.get("max", 0.0)) if len(non_null) else None,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_target(dataframe: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Return compact diagnostics for a target column.

    Args:
        dataframe: Source dataframe.
        target_column: Target column to inspect.

    Returns:
        Aggregate target diagnostics.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    target = dataframe[target_column]
    non_null = target.dropna()
    summary: dict[str, Any] = {
        "target_column": target_column,
        "row_count": int(len(target)),
        "non_null_count": int(non_null.shape[0]),
        "missing_count": int(target.isna().sum()),
        "unique_count": int(non_null.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(target):
        target_numeric = pd.to_numeric(non_null, errors="coerce")
        summary["base_rate"] = (
            float(target_numeric.mean()) if target_numeric.notna().any() else None
        )
    return summary


def plan_feature_subsets(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None = None,
    id_columns: list[str] | None = None,
    batch_size: int = 50,
) -> list[list[str]]:
    """Split candidate features into deterministic subsets.

    Args:
        dataframe: Source dataframe.
        target_column: Optional target column excluded from the subsets.
        id_columns: Optional identifier columns excluded from the subsets.
        batch_size: Maximum number of features per subset.

    Returns:
        Ordered feature subsets in dataframe order.
    """

    if batch_size <= 0:
        raise ValueError("`batch_size` must be greater than zero.")

    excluded = set(id_columns or [])
    if target_column is not None:
        excluded.add(target_column)
    candidate_columns = [
        str(column) for column in dataframe.columns if str(column) not in excluded
    ]
    return [
        candidate_columns[index : index + batch_size]
        for index in range(0, len(candidate_columns), batch_size)
    ]


__all__ = [
    "plan_feature_subsets",
    "summarize_dataframe",
    "summarize_dataframe_columns",
    "summarize_target",
]
