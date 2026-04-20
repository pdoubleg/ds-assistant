"""Dataframe EDA helpers for the Monty Python REPL."""

from __future__ import annotations

from typing import Any

from ..core.registry import (
    WorkspaceToolCollection,
    coerce_group_keys,
    flatten_columns,
    safe_json_value,
    tool,
)


class DataframeEDACollection(WorkspaceToolCollection):
    """Dataframe inspection, summary, and transformation helpers."""

    name = "dataframe"
    description = (
        "Inspect dataframe handles, summarize tabular data, and create derived "
        "dataframe handles."
    )

    @tool
    def dataframe_shape(self, dataframe_handle: str) -> dict[str, int]:
        """Return row and column counts for a stored dataframe.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.

        Returns:
            dict[str, int]: Mapping with `rows` and `columns` counts.

        Examples:
            print(dataframe_shape(df_handle))
            # Returns: {"rows": 1000, "columns": 24}
        """
        dataframe = self._get_dataframe(dataframe_handle)
        return {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])}

    @tool
    def dataframe_head(
        self, dataframe_handle: str, *, rows: int = 5
    ) -> list[dict[str, Any]]:
        """Return a JSON-friendly preview of a stored dataframe.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            rows (int): Number of rows to preview.

        Returns:
            list[dict[str, Any]]: Record-oriented preview rows.

        Examples:
            print(dataframe_head(df_handle, rows=5))
        """
        return (
            self._get_dataframe(dataframe_handle).head(rows).to_dict(orient="records")
        )

    @tool
    def dataframe_columns(self, dataframe_handle: str) -> list[str]:
        """Return column names for a stored dataframe.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.

        Returns:
            list[str]: Column labels converted to strings.

        Examples:
            print(dataframe_columns(df_handle))
        """
        return [str(column) for column in self._get_dataframe(dataframe_handle).columns]

    @tool
    def dataframe_dtypes(self, dataframe_handle: str) -> dict[str, str]:
        """Return dataframe dtypes keyed by column name.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.

        Returns:
            dict[str, str]: Column dtype mapping.

        Examples:
            print(dataframe_dtypes(df_handle))
            # Returns:
            # {
            #     "claim_amount": "float64",
            #     "segment": "object"
            # }
        """
        dataframe = self._get_dataframe(dataframe_handle)
        return {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}

    @tool
    def dataframe_missing_summary(self, dataframe_handle: str) -> list[dict[str, Any]]:
        """Summarize missing values for each dataframe column.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.

        Returns:
            list[dict[str, Any]]: Per-column missing-count summaries.

        Examples:
            print(dataframe_missing_summary(df_handle))
        """
        dataframe = self._get_dataframe(dataframe_handle)
        total_rows = max(len(dataframe), 1)
        missing_counts = dataframe.isna().sum()
        summary: list[dict[str, Any]] = []
        for column in dataframe.columns:
            missing = int(missing_counts[column])
            summary.append(
                {
                    "column": str(column),
                    "missing_count": missing,
                    "missing_rate": round(missing / total_rows, 6),
                }
            )
        return summary

    @tool
    def dataframe_describe(
        self,
        dataframe_handle: str,
        *,
        include_all: bool = False,
        max_rows: int = 20,
    ) -> list[dict[str, Any]]:
        """Return `DataFrame.describe()` output in record format.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            include_all (bool): Whether to include non-numeric columns.
            max_rows (int): Maximum describe rows to return.

        Returns:
            list[dict[str, Any]]: Record-oriented describe output.

        Examples:
            print(dataframe_describe(df_handle, include_all=True))
        """
        dataframe = self._get_dataframe(dataframe_handle)
        described = dataframe.describe(
            include="all" if include_all else None
        ).reset_index()
        return described.head(max_rows).to_dict(orient="records")

    @tool
    def value_counts(
        self,
        dataframe_handle: str,
        column: str,
        *,
        normalize: bool = False,
        dropna: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Compute top value counts for a single dataframe column.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            column (str): Column name to summarize.
            normalize (bool): Whether to return proportions instead of counts.
            dropna (bool): Whether to exclude missing values.
            limit (int): Maximum number of values to return.

        Returns:
            list[dict[str, Any]]: Top values with counts or normalized shares.

        Examples:
            print(value_counts(df_handle, "segment", limit=10))
        """
        dataframe = self._get_dataframe(dataframe_handle)
        series = (
            dataframe[column]
            .value_counts(normalize=normalize, dropna=dropna)
            .head(limit)
        )
        return [
            {
                "value": safe_json_value(index),
                "count": float(value) if normalize else int(value),
            }
            for index, value in series.items()
        ]

    @tool
    def filter_dataframe(self, dataframe_handle: str, query: str) -> str:
        """Filter rows with a pandas query expression.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            query (str): Pandas query expression evaluated against the dataframe.

        Returns:
            str: Handle for the filtered dataframe.

        Examples:
            high_value_handle = filter_dataframe(df_handle, "premium > 1000")
        """
        filtered = self._get_dataframe(dataframe_handle).query(query).copy()
        return self._object_store.put(filtered, prefix="df")

    @tool
    def groupby_aggregate(
        self,
        dataframe_handle: str,
        by: str | list[str],
        aggregations: dict[str, str | list[str]],
    ) -> str:
        """Group a dataframe and apply pandas aggregations.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            by (str | list[str]): Column or columns used for grouping.
            aggregations (dict[str, str | list[str]]): Aggregations keyed by
                source column name.

        Returns:
            str: Handle for the aggregated dataframe.

        Examples:
            summary_handle = groupby_aggregate(
                df_handle,
                ["segment"],
                {"loss": ["mean", "sum"], "premium": "mean"},
            )
        """
        dataframe = self._get_dataframe(dataframe_handle)
        grouped = (
            dataframe.groupby(coerce_group_keys(by), dropna=False)
            .agg(aggregations)
            .reset_index()
        )
        grouped.columns = flatten_columns(grouped.columns)
        return self._object_store.put(grouped, prefix="df")


__all__ = ["DataframeEDACollection"]
