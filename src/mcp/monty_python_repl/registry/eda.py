"""Dataframe EDA and Plotly collections for the Monty Python REPL."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.io as pio

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


class PlotlyCollection(WorkspaceToolCollection):
    """Plotly chart creation and export helpers."""

    name = "plotly"
    description = (
        "Create Plotly figures from stored dataframes and export chart artifacts."
    )

    @tool
    def create_scatter_plot(
        self,
        dataframe_handle: str,
        x: str,
        y: str,
        *,
        color: str | None = None,
        title: str | None = None,
    ) -> str:
        """Create a Plotly scatter plot from a stored dataframe.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            x (str): Column used for the x-axis.
            y (str): Column used for the y-axis.
            color (str | None): Optional color grouping column.
            title (str | None): Optional chart title.

        Returns:
            str: Handle for the stored Plotly figure.

        Examples:
            fig_handle = create_scatter_plot(
                df_handle,
                "age",
                "claim_amount",
                color="state",
            )
        """
        figure = px.scatter(
            self._get_dataframe(dataframe_handle),
            x=x,
            y=y,
            color=color,
            title=title,
        )
        return self._object_store.put(figure, prefix="fig")

    @tool
    def create_bar_chart(
        self,
        dataframe_handle: str,
        x: str,
        y: str,
        *,
        color: str | None = None,
        title: str | None = None,
    ) -> str:
        """Create a Plotly bar chart from a stored dataframe.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            x (str): Column used for the x-axis.
            y (str): Column used for the y-axis.
            color (str | None): Optional color grouping column.
            title (str | None): Optional chart title.

        Returns:
            str: Handle for the stored Plotly figure.

        Examples:
            fig_handle = create_bar_chart(summary_handle, "segment", "loss__sum")
        """
        figure = px.bar(
            self._get_dataframe(dataframe_handle),
            x=x,
            y=y,
            color=color,
            title=title,
        )
        return self._object_store.put(figure, prefix="fig")

    @tool
    def create_histogram(
        self,
        dataframe_handle: str,
        column: str,
        *,
        color: str | None = None,
        title: str | None = None,
        nbins: int | None = None,
    ) -> str:
        """Create a Plotly histogram from a stored dataframe column.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            column (str): Column used for the histogram x-axis.
            color (str | None): Optional color grouping column.
            title (str | None): Optional chart title.
            nbins (int | None): Optional number of histogram bins.

        Returns:
            str: Handle for the stored Plotly figure.

        Examples:
            fig_handle = create_histogram(df_handle, "claim_amount", nbins=30)
        """
        figure = px.histogram(
            self._get_dataframe(dataframe_handle),
            x=column,
            color=color,
            title=title,
            nbins=nbins,
        )
        return self._object_store.put(figure, prefix="fig")

    @tool
    def save_plotly_figure(
        self,
        figure_handle: str,
        html_path: str,
        *,
        image_path: str | None = None,
    ) -> list[str]:
        """Save a stored Plotly figure as HTML and optionally as an image.

        Args:
            figure_handle (str): Handle pointing to a stored Plotly figure.
            html_path (str): Relative or `/workspace` HTML destination.
            image_path (str | None): Optional static image destination.

        Returns:
            list[str]: Saved virtual paths in write order.

        Examples:
            save_plotly_figure(fig_handle, "/workspace/output/chart.html")
        """
        figure = self._get_figure(figure_handle)
        host_html_path = self._resolve_host_path(html_path)
        host_html_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(figure, host_html_path, auto_open=False, include_plotlyjs="cdn")
        self._record_artifact(host_html_path)

        saved_paths = [str(self._os_access.virtualize_host_path(host_html_path))]
        if image_path:
            host_image_path = self._resolve_host_path(image_path)
            host_image_path.parent.mkdir(parents=True, exist_ok=True)
            pio.write_image(figure, host_image_path)
            self._record_artifact(host_image_path)
            saved_paths.append(
                str(self._os_access.virtualize_host_path(host_image_path))
            )
        return saved_paths


__all__ = [
    "DataframeEDACollection",
    "PlotlyCollection",
]
