"""Shared helper utilities for Monty registry modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import plotly.graph_objects as go


def safe_json_value(value: Any, *, max_items: int = 20, max_chars: int = 500) -> Any:
    """Convert runtime values into JSON-friendly summaries.

    Args:
        value (Any): Runtime value to summarize.
        max_items (int): Maximum item count to retain for container previews.
        max_chars (int): Maximum character count for long string-like values.

    Returns:
        Any: JSON-friendly representation of the input value.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > max_chars:
            return value[:max_chars] + "... [truncated]"
        return value

    if isinstance(value, PurePosixPath | Path):
        return str(value)

    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(column) for column in value.columns[:max_items]],
            "preview": value.head(max_items).to_dict(orient="records"),
        }

    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "name": str(value.name),
            "length": int(len(value)),
            "preview": value.head(max_items).tolist(),
        }

    if isinstance(value, go.Figure):
        return {
            "type": "PlotlyFigure",
            "trace_count": len(value.data),
            "layout_title": getattr(value.layout.title, "text", None),
        }

    if isinstance(value, Mapping):
        items = list(value.items())[:max_items]
        return {
            str(key): safe_json_value(item, max_items=max_items, max_chars=max_chars)
            for key, item in items
        }

    if isinstance(value, (list, tuple, set)):
        return [
            safe_json_value(item, max_items=max_items, max_chars=max_chars)
            for item in list(value)[:max_items]
        ]

    rendered = repr(value)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "... [truncated]"
    return rendered


def flatten_columns(columns: Iterable[Any]) -> list[str]:
    """Convert possibly nested column labels into flat strings.

    Args:
        columns (Iterable[Any]): Dataframe column labels to normalize.

    Returns:
        list[str]: Flattened column labels.
    """
    flattened: list[str] = []
    for column in columns:
        if isinstance(column, tuple):
            flattened.append("__".join(str(part) for part in column))
        else:
            flattened.append(str(column))
    return flattened


def coerce_group_keys(by: str | list[str]) -> list[str]:
    """Normalize group-by keys into a list.

    Args:
        by (str | list[str]): Raw group-by input.

    Returns:
        list[str]: Normalized grouping keys.
    """
    if isinstance(by, str):
        return [by]
    return [str(item) for item in by]


__all__ = ["coerce_group_keys", "flatten_columns", "safe_json_value"]
