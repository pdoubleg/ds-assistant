"""Built-in data analysis tool collections for the Monty Python REPL."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from ..filesystem import HostWorkspaceOSAccess
from .base import (
    ObjectStore,
    ToolCollection,
    tool,
)
from .utils import (
    coerce_group_keys,
    flatten_columns,
    safe_json_value,
)


class _WorkspaceToolCollection(ToolCollection):
    """Shared workspace-aware helpers for built-in data tool collections."""

    def __init__(
        self, os_access: HostWorkspaceOSAccess, object_store: ObjectStore
    ) -> None:
        """Initialize the helper collection.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (ObjectStore): Shared handle store for dataframes and figures.
        """
        self._os_access = os_access
        self._object_store = object_store

    def _resolve_host_path(self, path: str) -> Path:
        """Resolve a virtual or relative path to the host workspace."""
        return self._os_access._to_host_path(PurePosixPath(path))

    def _get_dataframe(self, dataframe_handle: str) -> pd.DataFrame:
        """Fetch a dataframe from the object store."""
        return self._object_store.get(dataframe_handle, expected_type=pd.DataFrame)

    def _get_figure(self, figure_handle: str) -> go.Figure:
        """Fetch a Plotly figure from the object store."""
        return self._object_store.get(figure_handle, expected_type=go.Figure)


class DataIOCollection(_WorkspaceToolCollection):
    """Workspace file loading and export helpers."""

    name = "data_io"
    description = "Data loading, file export, and workspace file discovery helpers."

    @tool(
        categories=("io", "pandas"),
        usage_example="df_handle = load_csv('/workspace/input/data.csv')",
    )
    def load_csv(self, path: str, *, nrows: int | None = None) -> str:
        """Load a CSV file from `/workspace` and return a dataframe handle.

        Args:
            path (str): Relative or `/workspace`-scoped path to the CSV file.
            nrows (int | None): Optional maximum row count to load.

        Returns:
            str: Handle for the stored dataframe.
        """
        dataframe = pd.read_csv(self._resolve_host_path(path), nrows=nrows)
        return self._object_store.put(dataframe, prefix="df")

    @tool(
        categories=("io", "export"),
        usage_example="save_csv(df_handle, '/workspace/output/clean.csv')",
    )
    def save_csv(self, dataframe_handle: str, path: str, *, index: bool = False) -> str:
        """Save a stored dataframe handle to CSV inside `/workspace`.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            path (str): Relative or `/workspace`-scoped CSV destination.
            index (bool): Whether to persist the dataframe index.

        Returns:
            str: Virtual path to the saved CSV file.
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        self._get_dataframe(dataframe_handle).to_csv(host_path, index=index)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool(
        categories=("io", "workspace"),
        usage_example="print(list_workspace_files())",
    )
    def list_workspace_files(self, subdir: str = ".") -> list[str]:
        """List files currently available under `/workspace`.

        Args:
            subdir (str): Optional workspace subdirectory to search from.

        Returns:
            list[str]: Virtual workspace file paths.
        """
        host_root = self._resolve_host_path(subdir)
        if not host_root.exists():
            return []

        files: list[str] = []
        for child in sorted(host_root.rglob("*")):
            if child.is_file():
                files.append(str(self._os_access.virtualize_host_path(child)))
        return files

    @tool(
        categories=("export", "excel"),
        usage_example=(
            "save_excel("
            "{'raw': df_handle, 'summary': summary_handle}, "
            "'/workspace/output/report.xlsx')"
        ),
    )
    def save_excel(
        self,
        dataframes: dict[str, str],
        path: str,
        *,
        index: bool = False,
    ) -> str:
        """Save one or more stored dataframes to an Excel workbook.

        Args:
            dataframes (dict[str, str]): Mapping of sheet names to dataframe handles.
            path (str): Relative or `/workspace` workbook destination.
            index (bool): Whether to include dataframe indices in each sheet.

        Returns:
            str: Virtual path to the saved workbook.
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(host_path) as writer:
            for sheet_name, dataframe_handle in dataframes.items():
                self._get_dataframe(dataframe_handle).to_excel(
                    writer,
                    sheet_name=str(sheet_name)[:31],
                    index=index,
                )
        return str(self._os_access.virtualize_host_path(host_path))


class HandleInspectionCollection(_WorkspaceToolCollection):
    """Handle inspection helpers for stored in-memory artifacts."""

    name = "handles"
    description = "Inspect active dataframe and figure handles stored by Monty."

    @tool(
        categories=("state", "workspace"),
        usage_example="print(list_object_handles())",
    )
    def list_object_handles(self) -> list[str]:
        """List the dataframe and figure handles currently stored in memory.

        Returns:
            list[str]: Active object handles in insertion order.
        """
        return self._object_store.list_handles()

    @tool(
        categories=("state", "summary"),
        usage_example="print(inspect_handle(df_handle))",
    )
    def inspect_handle(self, handle: str) -> dict[str, Any]:
        """Return a summary of a stored host-side object.

        Args:
            handle (str): Dataframe or figure handle to inspect.

        Returns:
            dict[str, Any]: JSON-friendly handle summary.
        """
        return self._object_store.summary(handle)


class DataframeEDACollection(_WorkspaceToolCollection):
    """Dataframe inspection, summary, and transformation helpers."""

    name = "dataframe"
    description = (
        "Inspect dataframe handles, summarize tabular data, and create derived "
        "dataframe handles."
    )

    @tool(
        categories=("eda", "summary"),
        usage_example="print(dataframe_shape(df_handle))",
    )
    def dataframe_shape(self, dataframe_handle: str) -> dict[str, int]:
        """Return row and column counts for a stored dataframe.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.

        Returns:
            dict[str, int]: Mapping with `rows` and `columns` counts.
        """
        dataframe = self._get_dataframe(dataframe_handle)
        return {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])}

    @tool(
        categories=("eda", "summary"),
        usage_example="print(dataframe_head(df_handle, rows=5))",
    )
    def dataframe_head(
        self, dataframe_handle: str, *, rows: int = 5
    ) -> list[dict[str, Any]]:
        """Return a JSON-friendly preview of a stored dataframe.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            rows (int): Number of rows to preview.

        Returns:
            list[dict[str, Any]]: Record-oriented preview rows.
        """
        return (
            self._get_dataframe(dataframe_handle).head(rows).to_dict(orient="records")
        )

    @tool(
        categories=("eda", "summary"),
        usage_example="print(dataframe_columns(df_handle))",
    )
    def dataframe_columns(self, dataframe_handle: str) -> list[str]:
        """Return column names for a stored dataframe.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.

        Returns:
            list[str]: Column labels converted to strings.
        """
        return [str(column) for column in self._get_dataframe(dataframe_handle).columns]

    @tool(
        categories=("eda", "summary"),
        usage_example="print(dataframe_dtypes(df_handle))",
    )
    def dataframe_dtypes(self, dataframe_handle: str) -> dict[str, str]:
        """Return dataframe dtypes keyed by column name.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.

        Returns:
            dict[str, str]: Column dtype mapping.
        """
        dataframe = self._get_dataframe(dataframe_handle)
        return {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()}

    @tool(
        categories=("eda", "quality"),
        usage_example="print(dataframe_missing_summary(df_handle))",
    )
    def dataframe_missing_summary(self, dataframe_handle: str) -> list[dict[str, Any]]:
        """Summarize missing values for each dataframe column.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.

        Returns:
            list[dict[str, Any]]: Per-column missing-count summaries.
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

    @tool(
        categories=("eda", "summary"),
        usage_example="print(dataframe_describe(df_handle, include_all=True))",
    )
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
        """
        dataframe = self._get_dataframe(dataframe_handle)
        described = dataframe.describe(
            include="all" if include_all else None
        ).reset_index()
        return described.head(max_rows).to_dict(orient="records")

    @tool(
        categories=("eda", "summary"),
        usage_example="print(value_counts(df_handle, 'segment', limit=10))",
    )
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

    @tool(
        categories=("eda", "transform"),
        usage_example="high_value_handle = filter_dataframe(df_handle, 'premium > 1000')",
    )
    def filter_dataframe(self, dataframe_handle: str, query: str) -> str:
        """Filter rows with a pandas query expression.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            query (str): Pandas query expression evaluated against the dataframe.

        Returns:
            str: Handle for the filtered dataframe.
        """
        filtered = self._get_dataframe(dataframe_handle).query(query).copy()
        return self._object_store.put(filtered, prefix="df")

    @tool(
        categories=("eda", "transform"),
        usage_example=(
            "summary_handle = groupby_aggregate("
            "df_handle, ['segment'], {'loss': ['mean', 'sum']})"
        ),
    )
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
        """
        dataframe = self._get_dataframe(dataframe_handle)
        grouped = (
            dataframe.groupby(coerce_group_keys(by), dropna=False)
            .agg(aggregations)
            .reset_index()
        )
        grouped.columns = flatten_columns(grouped.columns)
        return self._object_store.put(grouped, prefix="df")


class PlotlyCollection(_WorkspaceToolCollection):
    """Plotly chart creation and export helpers."""

    name = "plotly"
    description = (
        "Create Plotly figures from stored dataframes and export chart artifacts."
    )

    @tool(
        categories=("plotly", "visualization"),
        usage_example=(
            "fig_handle = create_scatter_plot("
            "df_handle, 'age', 'claim_amount', color='state')"
        ),
    )
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
        """
        figure = px.scatter(
            self._get_dataframe(dataframe_handle),
            x=x,
            y=y,
            color=color,
            title=title,
        )
        return self._object_store.put(figure, prefix="fig")

    @tool(
        categories=("plotly", "visualization"),
        usage_example="fig_handle = create_bar_chart(summary_handle, 'segment', 'loss__sum')",
    )
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
        """
        figure = px.bar(
            self._get_dataframe(dataframe_handle),
            x=x,
            y=y,
            color=color,
            title=title,
        )
        return self._object_store.put(figure, prefix="fig")

    @tool(
        categories=("plotly", "visualization"),
        usage_example="fig_handle = create_histogram(df_handle, 'claim_amount', nbins=30)",
    )
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
        """
        figure = px.histogram(
            self._get_dataframe(dataframe_handle),
            x=column,
            color=color,
            title=title,
            nbins=nbins,
        )
        return self._object_store.put(figure, prefix="fig")

    @tool(
        categories=("plotly", "export"),
        usage_example="save_plotly_figure(fig_handle, '/workspace/output/chart.html')",
    )
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
        """
        figure = self._get_figure(figure_handle)
        host_html_path = self._resolve_host_path(html_path)
        host_html_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(figure, host_html_path, auto_open=False, include_plotlyjs="cdn")

        saved_paths = [str(self._os_access.virtualize_host_path(host_html_path))]
        if image_path:
            host_image_path = self._resolve_host_path(image_path)
            host_image_path.parent.mkdir(parents=True, exist_ok=True)
            pio.write_image(figure, host_image_path)
            saved_paths.append(
                str(self._os_access.virtualize_host_path(host_image_path))
            )
        return saved_paths


__all__ = [
    "DataIOCollection",
    "DataframeEDACollection",
    "HandleInspectionCollection",
    "PlotlyCollection",
]
