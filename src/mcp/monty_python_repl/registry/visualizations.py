"""Visualization helpers for the Monty Python REPL."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns

from ..core.registry import WorkspaceToolCollection, tool
from .feature_selection import StoredFeatureSelectionReport
from .hpo import StoredTunedPipeline

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class VisualizationCollection(WorkspaceToolCollection):
    """Aggregate chart helpers for stored dataframe and modeling artifacts."""

    name = "visualizations"
    description = (
        "Create Plotly figures plus file-based matplotlib visualizations for "
        "stored dataframe, report, and tuned-model handles."
    )

    def _save_figure(self, figure: matplotlib.figure.Figure, path: str) -> str:
        """Save a matplotlib figure under `/workspace`.

        Args:
            figure (matplotlib.figure.Figure): Figure to save.
            path (str): Relative or virtual workspace output path.

        Returns:
            str: Virtual workspace path to the saved artifact.
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(host_path, dpi=150)
        plt.close(figure)
        self._record_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    def _extract_report_importance_rows(
        self,
        artifact_handle: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Normalize feature-selection report findings into importance rows.

        Args:
            artifact_handle (str): Handle pointing to a stored feature-selection report.

        Returns:
            tuple[list[dict[str, Any]], str]: Importance rows and the importance kind.

        Raises:
            ValueError: If the report is not an importance report or has no usable rows.
        """
        report = self._object_store.get(
            artifact_handle,
            expected_type=StoredFeatureSelectionReport,
        )
        if report.report_type != "importance":
            raise ValueError(
                "Feature-importance plots require a StoredFeatureSelectionReport "
                "with report_type='importance'."
            )

        rows: list[dict[str, Any]] = []
        # The feature-selection report already stores ranked rows; keep only
        # entries that still have a numeric importance payload.
        for finding in report.findings:
            feature_name = finding.get("feature")
            importance_value = finding.get("importance")
            if feature_name is None or importance_value is None:
                continue
            rows.append(
                {
                    "feature": str(feature_name),
                    "importance": float(importance_value),
                }
            )
        if not rows:
            raise ValueError(
                "The stored feature-selection report does not contain any "
                "numeric importance rows to plot."
            )
        return rows, str(report.method)

    def _extract_tuned_pipeline_importance_rows(
        self,
        artifact_handle: str,
    ) -> tuple[list[dict[str, Any]], str, str]:
        """Extract feature-importance rows from a stored tuned pipeline.

        Args:
            artifact_handle (str): Handle pointing to a stored tuned pipeline.

        Returns:
            tuple[list[dict[str, Any]], str, str]:
                Ranked importance rows, importance kind, and model class name.

        Raises:
            ValueError: If the estimator does not expose a usable importance view.
        """
        artifact = self._object_store.get(
            artifact_handle,
            expected_type=StoredTunedPipeline,
        )
        fitted_model = artifact.fitted_model
        model_class = type(fitted_model).__name__

        feature_names = [str(column) for column in artifact.model_feature_columns]
        importance_kind = "feature_importances"

        booster = getattr(fitted_model, "booster_", None)
        if booster is not None and hasattr(booster, "feature_importance"):
            importance_values = list(booster.feature_importance(importance_type="gain"))
            booster_feature_names = getattr(booster, "feature_name", lambda: None)()
            if booster_feature_names:
                feature_names = [str(column) for column in booster_feature_names]
            importance_kind = "gain"
        elif hasattr(fitted_model, "feature_importances_"):
            importance_values = list(getattr(fitted_model, "feature_importances_"))
        elif hasattr(fitted_model, "coef_"):
            coefficient_array = np.asarray(getattr(fitted_model, "coef_"))
            importance_values = list(np.abs(np.ravel(coefficient_array)))
            importance_kind = "abs_coef"
        else:
            raise ValueError(
                "The tuned pipeline's fitted estimator does not expose "
                "`feature_importances_`, `coef_`, or a LightGBM booster-backed "
                "importance view."
            )

        if len(feature_names) != len(importance_values):
            raise ValueError(
                "The tuned pipeline's feature names do not align with the fitted "
                "estimator importance vector."
            )

        rows = sorted(
            [
                {"feature": feature_name, "importance": float(importance_value)}
                for feature_name, importance_value in zip(
                    feature_names,
                    importance_values,
                )
            ],
            key=lambda row: row["importance"],
            reverse=True,
        )
        if not rows:
            raise ValueError(
                "The tuned pipeline does not contain any feature importances to plot."
            )
        return rows, importance_kind, model_class

    def _resolve_feature_importance_rows(
        self,
        artifact_handle: str,
        *,
        top_n: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Resolve a report or tuned-pipeline handle into plot-ready rows.

        Args:
            artifact_handle (str): Stored feature-selection report or tuned-pipeline
                handle.
            top_n (int): Maximum number of features to include.

        Returns:
            tuple[pd.DataFrame, dict[str, Any]]: Plot dataframe and metadata payload.

        Raises:
            TypeError: If the handle is neither a feature-selection report nor a
                tuned-pipeline artifact.
        """
        artifact = self._object_store.get(artifact_handle)

        if isinstance(artifact, StoredFeatureSelectionReport):
            rows, importance_kind = self._extract_report_importance_rows(
                artifact_handle
            )
            source_metadata = {
                "source_type": "feature_selection_report",
                "importance_kind": importance_kind,
                "source_method": artifact.method,
            }
        elif isinstance(artifact, StoredTunedPipeline):
            rows, importance_kind, model_class = (
                self._extract_tuned_pipeline_importance_rows(artifact_handle)
            )
            source_metadata = {
                "source_type": "tuned_pipeline",
                "importance_kind": importance_kind,
                "model_class": model_class,
            }
        else:
            raise TypeError(
                "Feature-importance plots require a StoredFeatureSelectionReport "
                "or StoredTunedPipeline handle."
            )

        importance = pd.DataFrame(rows[:top_n], columns=["feature", "importance"])
        if importance.empty:
            raise ValueError("No feature importances are available to plot.")
        return importance, source_metadata

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

    @tool
    def plot_missingness(
        self,
        dataframe_handle: str,
        path: str,
        *,
        top_n: int = 30,
    ) -> dict[str, Any]:
        """Save a missingness bar chart for the most-missing columns.

        Use this when you want a quick visual scan of data quality problems
        before feature engineering or modeling.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            path (str): Output image path under `/workspace`.
            top_n (int): Maximum number of columns shown in the chart.

        Returns:
            dict[str, Any]: Saved artifact path and chart metadata.

        Examples:
            chart = plot_missingness(df_handle, "/workspace/reports/missingness.png")
            # Returns:
            # {
            #     "path": "/workspace/reports/missingness.png",
            #     "plot_type": "missingness_bar",
            #     "column_count": 30
            # }
        """
        dataframe = self._get_dataframe(dataframe_handle)
        missing = (
            dataframe.isna()
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        missing.columns = ["column", "missing_rate"]
        figure, axis = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=missing,
            x="missing_rate",
            y="column",
            ax=axis,
            color="#4C72B0",
        )
        axis.set_title("Top Missingness Rates")
        axis.set_xlabel("Missing rate")
        axis.set_ylabel("Column")
        saved_path = self._save_figure(figure, path)
        return {
            "path": saved_path,
            "plot_type": "missingness_bar",
            "column_count": int(len(missing)),
        }

    @tool
    def plot_numeric_histogram(
        self,
        dataframe_handle: str,
        column: str,
        path: str,
        *,
        bins: int = 30,
    ) -> dict[str, Any]:
        """Save a histogram for a numeric feature.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            column (str): Numeric column to visualize.
            path (str): Output image path under `/workspace`.
            bins (int): Number of histogram bins.

        Returns:
            dict[str, Any]: Saved artifact path and histogram metadata.

        Examples:
            hist = plot_numeric_histogram(
                df_handle,
                "balance",
                "/workspace/balance_hist.png",
            )
            # Returns:
            # {
            #     "path": "/workspace/balance_hist.png",
            #     "plot_type": "numeric_histogram",
            #     "column": "balance",
            #     "row_count": 10000,
            #     "bins": 30
            # }
        """
        dataframe = self._get_dataframe(dataframe_handle)
        if column not in dataframe.columns:
            raise ValueError(f"Column {column!r} was not found.")
        series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if series.empty:
            raise ValueError("The requested column has no numeric values to plot.")

        figure, axis = plt.subplots(figsize=(8, 5))
        sns.histplot(series, bins=bins, ax=axis, color="#55A868")
        axis.set_title(f"Histogram: {column}")
        axis.set_xlabel(column)
        saved_path = self._save_figure(figure, path)
        return {
            "path": saved_path,
            "plot_type": "numeric_histogram",
            "column": column,
            "row_count": int(series.shape[0]),
            "bins": int(bins),
        }

    @tool
    def plot_target_rate_by_numeric_bin(
        self,
        dataframe_handle: str,
        feature_column: str,
        target_column: str,
        path: str,
        *,
        bins: int = 10,
    ) -> dict[str, Any]:
        """Save a target-rate-by-bin chart for a numeric feature.

        This chart is useful for checking whether a candidate numeric feature has a
        monotonic or otherwise interpretable relationship with the target.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            feature_column (str): Numeric feature column to bucket.
            target_column (str): Numeric or binary target column.
            path (str): Output image path under `/workspace`.
            bins (int): Maximum number of quantile buckets.

        Returns:
            dict[str, Any]: Saved artifact path and binning metadata.

        Examples:
            result = plot_target_rate_by_numeric_bin(
                df_handle,
                "score_signal",
                "target",
                "/workspace/target_rate.png",
            )
            # Returns:
            # {
            #     "path": "/workspace/target_rate.png",
            #     "plot_type": "target_rate_by_bin",
            #     "feature_column": "score_signal",
            #     "target_column": "target",
            #     "bin_count": 10
            # }
        """
        dataframe = self._get_dataframe(dataframe_handle)
        if (
            feature_column not in dataframe.columns
            or target_column not in dataframe.columns
        ):
            raise ValueError("Both the feature and target columns must exist.")

        temp = dataframe[[feature_column, target_column]].copy()
        temp[feature_column] = pd.to_numeric(temp[feature_column], errors="coerce")
        temp[target_column] = pd.to_numeric(temp[target_column], errors="coerce")
        temp = temp.dropna()
        if temp.empty:
            raise ValueError(
                "No numeric rows are available after dropping missing values."
            )

        temp["bin"] = pd.qcut(
            temp[feature_column],
            q=min(bins, temp[feature_column].nunique()),
            duplicates="drop",
        )
        binned = (
            temp.groupby("bin", observed=True)
            .agg(
                row_count=(target_column, "size"),
                target_rate=(target_column, "mean"),
            )
            .reset_index()
        )
        binned["bin_label"] = [f"bin_{index + 1}" for index in range(len(binned))]

        figure, axis = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=binned, x="bin_label", y="target_rate", marker="o", ax=axis)
        axis.set_title(f"Target Rate by Quantile Bin: {feature_column}")
        axis.set_xlabel("Quantile bin")
        axis.set_ylabel("Target rate")
        saved_path = self._save_figure(figure, path)
        return {
            "path": saved_path,
            "plot_type": "target_rate_by_bin",
            "feature_column": feature_column,
            "target_column": target_column,
            "bin_count": int(len(binned)),
        }

    @tool
    def plot_prediction_diagnostics(
        self,
        dataframe_handle: str,
        target_column: str,
        score_column: str,
        path: str,
        *,
        bins: int = 10,
    ) -> dict[str, Any]:
        """Save an aggregate prediction-diagnostic chart using score buckets.

        Use this after scoring a dataframe to compare average prediction score
        against observed target rate across ordered score buckets.

        Args:
            dataframe_handle (str): Stored dataframe handle.
            target_column (str): Actual target column.
            score_column (str): Prediction or score column to bucket.
            path (str): Output image path under `/workspace`.
            bins (int): Maximum number of quantile buckets.

        Returns:
            dict[str, Any]: Saved artifact path and bucket metadata.

        Examples:
            diagnostics = plot_prediction_diagnostics(
                scored_df_handle,
                "target",
                "pred_score",
                "/workspace/diagnostics.png",
            )
            # Returns:
            # {
            #     "path": "/workspace/diagnostics.png",
            #     "plot_type": "prediction_diagnostics",
            #     "target_column": "target",
            #     "score_column": "pred_score",
            #     "bucket_count": 10
            # }
        """
        dataframe = self._get_dataframe(dataframe_handle)
        if (
            target_column not in dataframe.columns
            or score_column not in dataframe.columns
        ):
            raise ValueError("Both the target and score columns must exist.")

        temp = dataframe[[target_column, score_column]].copy()
        temp[target_column] = pd.to_numeric(temp[target_column], errors="coerce")
        temp[score_column] = pd.to_numeric(temp[score_column], errors="coerce")
        temp = temp.dropna()
        if temp.empty:
            raise ValueError("No rows are available after dropping missing values.")

        temp["bucket"] = pd.qcut(
            temp[score_column],
            q=min(bins, temp[score_column].nunique()),
            duplicates="drop",
        )
        bucketed = (
            temp.groupby("bucket", observed=True)
            .agg(
                avg_score=(score_column, "mean"),
                actual_rate=(target_column, "mean"),
                row_count=(target_column, "size"),
            )
            .reset_index(drop=True)
        )
        bucketed["bucket_label"] = [
            f"bucket_{index + 1}" for index in range(len(bucketed))
        ]

        figure, axis = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=bucketed,
            x="bucket_label",
            y="avg_score",
            marker="o",
            ax=axis,
            label="avg_score",
        )
        sns.lineplot(
            data=bucketed,
            x="bucket_label",
            y="actual_rate",
            marker="o",
            ax=axis,
            label="actual_rate",
        )
        axis.set_title("Prediction Diagnostics by Score Bucket")
        axis.set_xlabel("Score bucket")
        axis.set_ylabel("Rate")
        saved_path = self._save_figure(figure, path)
        return {
            "path": saved_path,
            "plot_type": "prediction_diagnostics",
            "target_column": target_column,
            "score_column": score_column,
            "bucket_count": int(len(bucketed)),
        }

    @tool
    def plot_feature_importance(
        self,
        artifact_handle: str,
        path: str,
        *,
        top_n: int = 30,
    ) -> dict[str, Any]:
        """Save a feature-importance chart for a report or tuned model artifact.

        Args:
            artifact_handle (str): Stored feature-selection report handle or tuned
                pipeline handle.
            path (str): Output image path under `/workspace`.
            top_n (int): Maximum number of features shown in the plot.

        Returns:
            dict[str, Any]: Saved artifact path and feature-importance metadata.

        Examples:
            chart = plot_feature_importance(
                importance_handle,
                "/workspace/feature_importance.png",
            )
            # Returns:
            # {
            #     "path": "/workspace/feature_importance.png",
            #     "plot_type": "feature_importance",
            #     "feature_count": 30
            # }
        """
        importance, source_metadata = self._resolve_feature_importance_rows(
            artifact_handle,
            top_n=top_n,
        )
        figure, axis = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=importance,
            x="importance",
            y="feature",
            ax=axis,
            color="#C44E52",
        )
        axis.set_title("Feature Importance")
        axis.set_xlabel(source_metadata["importance_kind"])
        axis.set_ylabel("Feature")
        saved_path = self._save_figure(figure, path)
        return {
            "path": saved_path,
            "plot_type": "feature_importance",
            "feature_count": int(len(importance)),
            **source_metadata,
        }


__all__ = ["VisualizationCollection"]
