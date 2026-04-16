"""Visualization tools for the minimal registry package."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..base import WorkspaceToolCollection
from ..core.registry import tool
from .base import StoredLightGBMModelArtifact


class VisualizationCollection(WorkspaceToolCollection):
    """Aggregate plotting helpers that emit safe image artifacts."""

    name = "visualizations"
    description = (
        "Create aggregate plots saved to `/workspace` without returning row-level "
        "chart payloads."
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
            data=missing, x="missing_rate", y="column", ax=axis, color="#4C72B0"
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
            hist = plot_numeric_histogram(df_handle, "balance", "/workspace/balance_hist.png")
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
        model_handle: str,
        path: str,
        *,
        top_n: int = 30,
    ) -> dict[str, Any]:
        """Save a feature-importance chart for a fitted model artifact.

        Args:
            model_handle (str): Stored fitted-model handle.
            path (str): Output image path under `/workspace`.
            top_n (int): Maximum number of features shown in the plot.

        Returns:
            dict[str, Any]: Saved artifact path and feature-importance metadata.

        Examples:
            chart = plot_feature_importance(model_handle, "/workspace/feature_importance.png")
            # Returns:
            # {
            #     "path": "/workspace/feature_importance.png",
            #     "plot_type": "feature_importance",
            #     "feature_count": 30
            # }
        """

        artifact = self._object_store.get(
            model_handle,
            expected_type=StoredLightGBMModelArtifact,
        )
        rows = sorted(
            artifact.feature_importance_gain.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        importance = pd.DataFrame(rows, columns=["feature", "gain"])
        figure, axis = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance, x="gain", y="feature", ax=axis, color="#C44E52")
        axis.set_title("LightGBM Feature Importance (Gain)")
        saved_path = self._save_figure(figure, path)
        return {
            "path": saved_path,
            "plot_type": "feature_importance",
            "feature_count": int(len(importance)),
        }


__all__ = ["VisualizationCollection"]
