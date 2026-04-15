"""Shared helper utilities for Monty registry modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import plotly.graph_objects as go


def _is_scalar_json_like(value: Any) -> bool:
    """Return whether a value is already small and JSON-friendly.

    Args:
        value (Any): Runtime value to classify.

    Returns:
        bool: ``True`` when the value is a scalar-like JSON primitive.
    """
    return value is None or isinstance(value, (bool, int, float, str))


def _truncate_preview_value(value: Any, *, max_chars: int, max_items: int) -> Any:
    """Summarize dataframe-like preview cells without bloating payloads.

    Args:
        value (Any): Cell or preview value.
        max_chars (int): Maximum characters to retain for long preview strings.
        max_items (int): Maximum nested preview items to retain.

    Returns:
        Any: JSON-friendly preview representation.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + "... [truncated]"
        return value
    return safe_json_value(value, max_items=max_items, max_chars=max_chars)


def safe_json_value(value: Any, *, max_items: int = 20, max_chars: int = 500) -> Any:
    """Convert runtime values into JSON-friendly summaries.

    Args:
        value (Any): Runtime value to summarize.
        max_items (int): Maximum item count to retain for container previews.
        max_chars (int): Maximum character count for long string-like values.

    Returns:
        Any: JSON-friendly representation of the input value.
    """
    if _is_scalar_json_like(value):
        return value

    # Let rich host-side artifacts provide their own compact summaries without
    # forcing this module to import every registry-specific runtime type.
    to_json_summary = getattr(value, "to_json_summary", None)
    if callable(to_json_summary):
        return to_json_summary(max_items=max_items, max_chars=max_chars)

    if isinstance(value, PurePosixPath | Path):
        return str(value)

    if isinstance(value, pd.DataFrame):
        preview_records = value.head(max_items).to_dict(orient="records")
        return {
            "type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            # Preserve the full schema because callers often need the exact
            # fitted/evaluation column names for debugging.
            "columns": [str(column) for column in value.columns],
            "preview": [
                {
                    str(key): _truncate_preview_value(
                        item,
                        max_chars=max_chars,
                        max_items=max_items,
                    )
                    for key, item in row.items()
                }
                for row in preview_records
            ],
        }

    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "name": str(value.name),
            "length": int(len(value)),
            "preview": [
                _truncate_preview_value(
                    item,
                    max_chars=max_chars,
                    max_items=max_items,
                )
                for item in value.head(max_items).tolist()
            ],
        }

    if isinstance(value, go.Figure):
        return {
            "type": "PlotlyFigure",
            "trace_count": len(value.data),
            "layout_title": getattr(value.layout.title, "text", None),
        }

    if isinstance(value, Mapping):
        return {
            str(key): safe_json_value(item, max_items=max_items, max_chars=max_chars)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if all(_is_scalar_json_like(item) for item in items):
            return [
                safe_json_value(item, max_items=max_items, max_chars=max_chars)
                for item in items
            ]
        return [
            safe_json_value(item, max_items=max_items, max_chars=max_chars)
            for item in items[:max_items]
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
