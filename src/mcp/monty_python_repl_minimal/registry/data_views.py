"""Unified dataframe inspection helpers for the minimal registry package."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..base import WorkspaceToolCollection
from ..core.registry import tool
from ..privacy import summarize_dataframe, summarize_dataframe_columns
from .base import StoredDataframeReport
from .utils import _is_numeric_dtype


def _split_feature_columns(columns: list[str], *, batch_size: int) -> list[list[str]]:
    """Split feature columns into deterministic batches.

    Args:
        columns: Ordered feature columns to partition.
        batch_size: Maximum number of columns per batch.

    Returns:
        Deterministic column batches in original dataframe order.

    Examples:
        >>> _split_feature_columns(["a", "b", "c"], batch_size=2)
        [['a', 'b'], ['c']]
    """
    if batch_size <= 0:
        raise ValueError("`batch_size` must be greater than zero.")
    return [
        columns[index : index + batch_size]
        for index in range(0, len(columns), batch_size)
    ]


class DataViewCollection(WorkspaceToolCollection):
    """Handle-based dataframe inspection helpers."""

    name = "data_views"
    description = (
        "Inspect dataframe handles with lightweight shape/schema helpers, "
        "privacy-safe summaries, target diagnostics, and deterministic feature "
        "subset planning for wide tables."
    )

    def _resolve_candidate_columns(
        self,
        dataframe: pd.DataFrame,
        *,
        target_column: str | None,
        id_columns: list[str] | None,
    ) -> list[str]:
        """Return candidate feature columns in dataframe order.

        Args:
            dataframe: Source dataframe stored behind a handle.
            target_column: Optional target column to exclude.
            id_columns: Optional identifier columns to exclude.

        Returns:
            Ordered candidate feature column names.
        """
        excluded = set(id_columns or [])
        if target_column is not None:
            excluded.add(target_column)
        return [
            str(column) for column in dataframe.columns if str(column) not in excluded
        ]

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

        Use this first after `load_csv(...)`, `load_parquet_slice(...)`, or
        `select_columns(...)` to inspect the stored handle before choosing more
        targeted follow-up helpers.

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

        Call this after `summarize_dataframe(...)` once you know which stored
        columns deserve deeper inspection.

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

        Use this after a high-level dataframe summary when you need a compact view
        of class balance, missingness, and uniqueness before screening or modeling.

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

        # Keep the returned payload compact and aggregate-only.
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

    @tool
    def plan_feature_subsets(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        id_columns: list[str] | None = None,
        batch_size: int = 50,
        max_inline_subsets: int = 12,
    ) -> dict[str, Any]:
        """Plan deterministic feature subsets for a wide dataframe handle.

        Use this when the stored dataframe has many candidate features and later
        screening or ranking steps should work through stable batches instead of a
        single full-width pass.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            target_column (str | None): Optional target column excluded from subsets.
            id_columns (list[str] | None): Optional identifier columns excluded from
                subsets.
            batch_size (int): Maximum feature count per subset.
            max_inline_subsets (int): Maximum subset definitions returned inline.

        Returns:
            dict[str, Any]: Report handle plus deterministic feature subsets.

        Examples:
            ```python
            subset_plan = plan_feature_subsets(
                df_handle,
                target_column="target",
                id_columns=["customer_id"],
                batch_size=25,
            )
            # Returns
            # {
            #     "report_handle": "report_abc123",
            #     "summary": "Planned 2 feature subsets from 10 candidate columns.",
            #     "subset_count": 2,
            #     "feature_subsets": [["feature_1", "feature_2"], ["feature_3", "feature_4"]],
            #     "subsets_truncated": False,
            # }
            ```
        """
        dataframe = self._get_dataframe(dataframe_handle)
        candidate_columns = self._resolve_candidate_columns(
            dataframe,
            target_column=target_column,
            id_columns=id_columns,
        )
        feature_subsets = _split_feature_columns(
            candidate_columns,
            batch_size=batch_size,
        )
        summary = (
            f"Planned {len(feature_subsets)} feature subsets from "
            f"{len(candidate_columns)} candidate columns."
        )
        report = StoredDataframeReport(
            report_type="feature_subset_plan",
            title="Feature subset plan",
            summary=summary,
            details={
                "target_column": target_column,
                "id_columns": list(id_columns or []),
                "batch_size": batch_size,
                "subset_count": len(feature_subsets),
            },
            metadata={"feature_subsets": feature_subsets},
        )
        report_handle = self._object_store.put(report, prefix="report")
        return {
            "report_handle": report_handle,
            "summary": summary,
            "subset_count": len(feature_subsets),
            "feature_subsets": feature_subsets[:max_inline_subsets],
            "subsets_truncated": len(feature_subsets) > max_inline_subsets,
        }


__all__ = ["DataViewCollection"]
