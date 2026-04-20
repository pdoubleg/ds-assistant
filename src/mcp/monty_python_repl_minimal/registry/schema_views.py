"""Schema inspection tools for the minimal registry package."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..base import WorkspaceToolCollection
from ..core.registry import tool
from ..privacy import summarize_dataframe, summarize_dataframe_columns
from .utils import _is_numeric_dtype


class SchemaViewCollection(WorkspaceToolCollection):
    """Schema and summary-statistics helpers."""

    name = "schema_views"
    description = (
        "Inspect dataframe schema, dtypes, missingness, and targeted column "
        "statistics without exposing raw row values or category examples."
    )

    @tool
    def dataframe_shape(self, dataframe_handle: str) -> dict[str, int]:
        """Return row and column counts for a stored dataframe.

        Args:
            dataframe_handle (str): Stored dataframe handle.

        Returns:
            dict[str, int]: Row and column counts for the dataframe.

        Examples:
            ```python
            shape = dataframe_shape(df_handle)
            # Returns
            # {
            #     "rows": 1000,
            #     "columns": 10,
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle)
        return {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])}

    @tool
    def dataframe_columns(self, dataframe_handle: str) -> list[str]:
        """Return column names for a stored dataframe.

        Args:
            dataframe_handle (str): Stored dataframe handle.

        Returns:
            list[str]: Column names in dataframe order.

        Examples:
            ```python
            columns = dataframe_columns(df_handle)
            # Returns
            # ["customer_id", "balance", "target", "income", "education"]
            ```
        """

        return [str(column) for column in self._get_dataframe(dataframe_handle).columns]

    @tool
    def dataframe_dtypes(self, dataframe_handle: str) -> dict[str, str]:
        """Return dtypes keyed by column name.

        Args:
            dataframe_handle (str): Stored dataframe handle.

        Returns:
            dict[str, str]: Mapping from column name to pandas dtype string.

        Examples:
            ```python
            dtypes = dataframe_dtypes(df_handle)
            # Returns
            # {
            #     "customer_id": "object",
            #     "balance": "float64",
            #     "target": "int64",
            #     "income": "float64",
            #     "education": "object",
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle)
        return {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}

    @tool
    def summarize_dataframe(self, dataframe_handle: str) -> dict[str, Any]:
        """Return a lightweight privacy-safe dataframe overview.

        Use this first to inspect dataset size, column names, type mix, and
        missingness without paying for per-column statistics on wide tables.

        Args:
            dataframe_handle (str): Stored dataframe handle.

        Returns:
            dict[str, Any]: Aggregate schema metadata and a usage hint for
            targeted follow-up inspection.

        Examples:
            ```python
            summary = summarize_dataframe(df_handle)
            # Returns
            # {
            #     "type": "DataFrame",
            #     "shape": [1000, 10],
            #     "row_count": 1000,
            #     "column_count": 10,
            #     "columns": ["customer_id", "balance", "target", "income", "education"],
            #     "column_type_counts": {"numeric": 3, "datetime": 0, "categorical": 2, "other": 0},
            #     "missingness": {"total_missing_cells": 0, ...},
            # }
            ```
        """

        return summarize_dataframe(self._get_dataframe(dataframe_handle))

    @tool
    def summarize_dataframe_columns(
        self,
        dataframe_handle: str,
        columns: list[str],
    ) -> dict[str, Any]:
        """Return detailed summaries for a focused list of dataframe columns.

        Call this after `summarize_dataframe(...)` once you know which columns
        deserve deeper inspection.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            columns (list[str]): Column names to summarize in detail.

        Returns:
            dict[str, Any]: Targeted per-column summaries for the requested columns.

        Examples:
            ```python
            details = summarize_dataframe_columns(
                df_handle,
                ["balance", "target", "education"],
            )
            # Returns
            # {
            #     "type": "DataFrameColumnDetails",
            #     "requested_columns": ["balance", "target", "education"],
            #     "column_summaries": [...],
            # }
            ```
        """
        if not columns:
            raise ValueError("At least one column name must be provided.")

        dataframe = self._get_dataframe(dataframe_handle)
        missing_columns = [
            column for column in columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

        return summarize_dataframe_columns(dataframe, columns)

    @tool
    def summarize_target(
        self,
        dataframe_handle: str,
        target_column: str,
    ) -> dict[str, Any]:
        """Return aggregate target statistics for a single column.

        Use this to confirm class balance, missingness, and uniqueness before
        training or feature screening.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            target_column (str): Target column to summarize.

        Returns:
            dict[str, Any]: Aggregate target metadata, including base rate when the
            target is numeric.

        Examples:
            ```python
            target_summary = summarize_target(df_handle, "target")
            # Returns
            # {
            #     "target_column": "target",
            #     "row_count": 1000,
            #     "non_null_count": 1000,
            #     "missing_count": 0,
            #     "unique_count": 2,
            #     "base_rate": 0.5,
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle)
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
        if _is_numeric_dtype(target):
            target_numeric = pd.to_numeric(non_null, errors="coerce")
            summary["base_rate"] = (
                float(target_numeric.mean()) if target_numeric.notna().any() else None
            )
        return summary


__all__ = ["SchemaViewCollection"]
