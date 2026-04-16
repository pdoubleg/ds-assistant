"""Schema inspection tools for the minimal registry package."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..base import WorkspaceToolCollection
from ..core.registry import tool
from ..privacy import summarize_dataframe
from .utils import _is_numeric_dtype


class SchemaViewCollection(WorkspaceToolCollection):
    """Schema and summary-statistics helpers."""

    name = "schema_views"
    description = (
        "Inspect dataframe schema, dtypes, missingness, and summary statistics "
        "without exposing raw row values or category examples."
    )

    @tool
    def dataframe_shape(self, dataframe_handle: str) -> dict[str, int]:
        """Return row and column counts for a stored dataframe.

        Args:
            dataframe_handle (str): Stored dataframe handle.

        Returns:
            dict[str, int]: Row and column counts for the dataframe.

        Examples:
            shape = dataframe_shape(df_handle)
            # Returns:
            # {"rows": 10000, "columns": 42}
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
            columns = dataframe_columns(df_handle)
            # Returns:
            # ["customer_id", "balance", "target"]
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
            dtypes = dataframe_dtypes(df_handle)
            # Returns:
            # {"balance": "float64", "target": "int64"}
        """

        dataframe = self._get_dataframe(dataframe_handle)
        return {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}

    @tool
    def summarize_dataframe(self, dataframe_handle: str) -> dict[str, Any]:
        """Return a privacy-safe dataframe summary.

        This is the quickest way to inspect schema shape, missingness, and summary
        statistics without leaking raw rows back to the model.

        Args:
            dataframe_handle (str): Stored dataframe handle.

        Returns:
            dict[str, Any]: Aggregate schema and summary statistics.

        Examples:
            summary = summarize_dataframe(df_handle)
            # Returns:
            # {
            #     "row_count": 10000,
            #     "column_count": 42,
            #     "columns": ["customer_id", "segment", "balance", "target"],
            #     "numeric_columns": ["balance", "utilization", "target"]
            # }
        """

        return summarize_dataframe(self._get_dataframe(dataframe_handle))

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
            target_summary = summarize_target(df_handle, "target")
            # Returns:
            # {
            #     "target_column": "target",
            #     "row_count": 10000,
            #     "non_null_count": 10000,
            #     "missing_count": 0,
            #     "unique_count": 2,
            #     "base_rate": 0.17
            # }
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
