"""Privacy-safe EDA helpers for the minimal registry package."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ..base import WorkspaceToolCollection
from ..core.registry import tool
from ..privacy import summarize_dataframe_columns
from .base import StoredDataframeReport


def _split_feature_columns(columns: list[str], *, batch_size: int) -> list[list[str]]:
    """Split feature columns into deterministic batches."""

    if batch_size <= 0:
        raise ValueError("`batch_size` must be greater than zero.")
    return [
        columns[index : index + batch_size]
        for index in range(0, len(columns), batch_size)
    ]


class EDACollection(WorkspaceToolCollection):
    """High-level EDA helpers that avoid exposing raw data values."""

    name = "eda"
    description = (
        "Triage dataset size, plan feature subsets for wide tables, and summarize "
        "feature batches without returning raw values."
    )

    def _resolve_candidate_columns(
        self,
        dataframe: pd.DataFrame,
        *,
        target_column: str | None,
        id_columns: list[str] | None,
    ) -> list[str]:
        """Return candidate feature columns in dataframe order."""

        excluded = set(id_columns or [])
        if target_column is not None:
            excluded.add(target_column)
        return [
            str(column) for column in dataframe.columns if str(column) not in excluded
        ]

    def _dtype_mix(self, dataframe: pd.DataFrame, columns: list[str]) -> dict[str, int]:
        """Return aggregate dtype counts for a column subset."""

        numeric_count = 0
        categorical_count = 0
        datetime_count = 0
        other_count = 0

        for column in columns:
            series = dataframe[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric_count += 1
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetime_count += 1
            elif (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_categorical_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            ):
                categorical_count += 1
            else:
                other_count += 1

        return {
            "numeric": numeric_count,
            "categorical": categorical_count,
            "datetime": datetime_count,
            "other": other_count,
        }

    @tool
    def triage_dataframe(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        id_columns: list[str] | None = None,
        max_columns_before_batching: int = 150,
        recommended_batch_size: int = 50,
    ) -> dict[str, Any]:
        """Triage dataframe size and recommend a privacy-safe EDA workflow.

        Use this before deeper EDA or modeling so the agent can decide whether to
        work on the full feature space or break a wide table into deterministic
        feature subsets.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            target_column (str | None): Optional target column retained outside the
                candidate feature set.
            id_columns (list[str] | None): Optional identifier columns excluded from
                the candidate feature set.
            max_columns_before_batching (int): Feature-count threshold that marks a
                dataframe as wide.
            recommended_batch_size (int): Suggested feature count per batch when the
                dataframe is wide.

        Returns:
            dict[str, Any]: Report handle, compact summary, and batching guidance.

        Examples:
            ```python
            triage = triage_dataframe(
                df_handle,
                target_column="target",
                id_columns=["customer_id"],
            )
            # Returns
            # {
            #     "report_handle": "report_abc123",
            #     "summary": "Wide dataframe with 10 candidate features; analyze in 2 feature subsets of about 5 columns.",
            #     "wide_table": True,
            #     "candidate_feature_count": 10,
            #     "subset_count": 2,
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle)
        candidate_columns = self._resolve_candidate_columns(
            dataframe,
            target_column=target_column,
            id_columns=id_columns,
        )
        dtype_mix = self._dtype_mix(dataframe, candidate_columns)
        wide_table = len(candidate_columns) > max_columns_before_batching
        subset_count = (
            int(math.ceil(len(candidate_columns) / recommended_batch_size))
            if candidate_columns and wide_table
            else 1
        )
        summary = (
            f"Wide dataframe with {len(candidate_columns)} candidate features; "
            f"analyze in {subset_count} feature subsets of about "
            f"{recommended_batch_size} columns."
            if wide_table
            else f"Manageable dataframe with {len(candidate_columns)} candidate "
            f"features; full-frame EDA is reasonable."
        )
        details = {
            "dataframe_shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
            "target_column": target_column,
            "id_columns": list(id_columns or []),
            "candidate_feature_count": len(candidate_columns),
            "dtype_mix": dtype_mix,
            "wide_table": wide_table,
            "recommended_batch_size": recommended_batch_size,
            "subset_count": subset_count,
        }
        report = StoredDataframeReport(
            report_type="dataframe_triage",
            title="Dataframe triage",
            summary=summary,
            details=details,
            metadata={"candidate_columns": candidate_columns},
        )
        report_handle = self._object_store.put(report, prefix="report")
        return {
            "report_handle": report_handle,
            "summary": summary,
            "wide_table": wide_table,
            "candidate_feature_count": len(candidate_columns),
            "subset_count": subset_count,
        }

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
        """Plan deterministic feature subsets for a wide dataframe.

        This helper returns column batches in stable dataframe order so the agent
        can iterate through wide-table EDA and screening without writing custom
        batching code in the sandbox.

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
            #     "feature_subsets": ["feature_1", "feature_2", "feature_3"],
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
            candidate_columns, batch_size=batch_size
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

    @tool
    def summarize_feature_subset(
        self,
        dataframe_handle: str,
        columns: list[str],
        *,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """Summarize a deterministic subset of feature columns.

        Use this after `plan_feature_subsets(...)` so the agent can inspect one
        feature batch at a time while keeping outputs aggregate-only.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            columns (list[str]): Feature columns included in the subset.
            target_column (str | None): Optional target column summarized separately.

        Returns:
            dict[str, Any]: Report handle and compact subset summary.

        Examples:
            ```python
            subset_summary = summarize_feature_subset(
                df_handle,
                subset_plan["feature_subsets"][0],
                target_column="target",
            )
            # Returns
            # {
            #     "report_handle": "report_abc123",
            #     "summary": "Summarized 5 feature columns for privacy-safe EDA.",
            #     "column_count": 5,
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle)
        missing_columns = [
            column for column in columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

        subset_frame = dataframe[columns].copy()
        details: dict[str, Any] = {
            "column_count": len(columns),
            "dtype_mix": self._dtype_mix(dataframe, columns),
            # Feature subsets are already targeted, so detailed per-column stats
            # are appropriate here and stay bounded by the requested batch size.
            "subset_summary": summarize_dataframe_columns(subset_frame, columns),
        }
        if target_column is not None:
            if target_column not in dataframe.columns:
                raise ValueError(f"Target column {target_column!r} was not found.")
            target = dataframe[target_column]
            details["target_summary"] = {
                "target_column": target_column,
                "non_null_count": int(target.notna().sum()),
                "missing_count": int(target.isna().sum()),
                "unique_count": int(target.dropna().nunique(dropna=True)),
            }

        summary = f"Summarized {len(columns)} feature columns for privacy-safe EDA."
        report = StoredDataframeReport(
            report_type="feature_subset_summary",
            title="Feature subset summary",
            summary=summary,
            details=details,
            metadata={"columns": list(columns)},
        )
        report_handle = self._object_store.put(report, prefix="report")
        return {
            "report_handle": report_handle,
            "summary": summary,
            "column_count": len(columns),
        }


__all__ = ["EDACollection"]
