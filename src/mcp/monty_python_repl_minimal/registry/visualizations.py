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

    def _prepare_prediction_frame(
        self,
        dataframe: pd.DataFrame,
        *,
        target_column: str,
        score_column: str,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a clean target-score-feature frame for prediction plots."""
        requested_columns = [target_column, score_column] + list(feature_columns or [])
        missing_columns = [
            column for column in requested_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

        temp = dataframe[requested_columns].copy()
        temp[target_column] = pd.to_numeric(temp[target_column], errors="coerce")
        temp[score_column] = pd.to_numeric(temp[score_column], errors="coerce")
        temp = temp.dropna(subset=[target_column, score_column])
        if temp.empty:
            raise ValueError(
                "No rows are available after dropping missing target/score values."
            )
        return temp

    def _build_quantile_buckets(
        self,
        series: pd.Series,
        *,
        max_bins: int,
    ) -> pd.Series:
        """Return stable generic quantile bucket labels."""
        numeric_series = pd.to_numeric(series, errors="coerce")
        valid = numeric_series.dropna()
        if valid.empty:
            raise ValueError(
                "No numeric rows are available after dropping missing values."
            )
        bucket_count = max(1, min(max_bins, int(valid.nunique())))
        if bucket_count == 1:
            labels = pd.Series(index=valid.index, data="bin_1", dtype="object")
        else:
            bucket_codes = pd.qcut(
                valid,
                q=bucket_count,
                labels=False,
                duplicates="drop",
            )
            labels = bucket_codes.map(lambda code: f"bin_{int(code) + 1}").astype(
                "object"
            )
        return labels.reindex(series.index)

    def _build_categorical_groups(
        self,
        series: pd.Series,
        *,
        top_n_categories: int,
    ) -> pd.Series:
        """Return generic grouped labels for a categorical series."""
        normalized = series.astype("object").where(series.notna(), "__missing__")
        counts = normalized.value_counts(dropna=False)
        kept_values = counts.head(max(1, top_n_categories)).index.tolist()
        mapping = {
            value: f"group_{index + 1}" for index, value in enumerate(kept_values)
        }
        grouped = normalized.map(lambda value: mapping.get(value, "other"))
        return grouped.astype("object")

    def _summarize_prediction_profile(
        self,
        dataframe: pd.DataFrame,
        *,
        target_column: str,
        score_column: str,
        bucket_labels: pd.Series,
    ) -> pd.DataFrame:
        """Aggregate prediction-vs-actual metrics for plotting."""
        temp = dataframe[[target_column, score_column]].copy()
        temp["bucket_label"] = bucket_labels
        temp = temp.dropna(subset=["bucket_label"])
        if temp.empty:
            raise ValueError("No grouped rows are available for plotting.")

        summary = (
            temp.groupby("bucket_label", observed=True)
            .agg(
                avg_score=(score_column, "mean"),
                actual_rate=(target_column, "mean"),
                row_count=(target_column, "size"),
            )
            .reset_index()
        )

        def _bucket_sort_key(label: str) -> tuple[int, str]:
            if label.startswith("bin_"):
                return (0, label)
            if label.startswith("group_"):
                return (1, label)
            if label == "other":
                return (2, label)
            return (3, label)

        return summary.sort_values(
            by="bucket_label",
            key=lambda values: values.map(_bucket_sort_key),
        ).reset_index(drop=True)

    def _plot_prediction_profile_panel(
        self,
        axis: matplotlib.axes.Axes,
        summary: pd.DataFrame,
        *,
        title: str,
        x_label: str,
    ) -> None:
        """Plot average prediction, actual rate, and row counts for one panel."""
        counts_axis = axis.twinx()
        sns.barplot(
            data=summary,
            x="bucket_label",
            y="row_count",
            ax=counts_axis,
            color="#DDDDDD",
            alpha=0.7,
        )
        counts_axis.set_ylabel("Row count")

        sns.lineplot(
            data=summary,
            x="bucket_label",
            y="avg_score",
            marker="o",
            ax=axis,
            label="avg_score",
            color="#4C72B0",
        )
        sns.lineplot(
            data=summary,
            x="bucket_label",
            y="actual_rate",
            marker="o",
            ax=axis,
            label="actual_rate",
            color="#C44E52",
        )
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Rate")
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylim(0.0, 1.0)

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
            ```python
            chart = plot_missingness(
                df_handle,
                "/workspace/reports/missingness.png",
            )
            # Returns
            # {
            #     "path": "/workspace/reports/missingness.png",
            #     "plot_type": "missingness_bar",
            #     "column_count": 30,
            # }
            ```
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
            ```python
            hist = plot_numeric_histogram(
                df_handle,
                "balance",
                "/workspace/balance_hist.png",
            )
            # Returns
            # {
            #     "path": "/workspace/balance_hist.png",
            #     "plot_type": "numeric_histogram",
            #     "column": "balance",
            #     "row_count": 10000,
            #     "bins": 30,
            # }
            ```
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
            ```python
            result = plot_target_rate_by_numeric_bin(
                df_handle,
                "score_signal",
                "target",
                "/workspace/target_rate.png",
            )
            # Returns
            # {
            #     "path": "/workspace/target_rate.png",
            #     "plot_type": "target_rate_by_bin",
            #     "feature_column": "score_signal",
            #     "target_column": "target",
            #     "bin_count": 10,
            # }
            ```
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
            ```python
            diagnostics = plot_prediction_diagnostics(
                scored_df_handle,
                "target",
                "pred_score",
                "/workspace/diagnostics.png",
            )
            # Returns
            # {
            #     "path": "/workspace/diagnostics.png",
            #     "plot_type": "prediction_diagnostics",
            #     "target_column": "target",
            #     "score_column": "pred_score",
            #     "bucket_count": 10,
            # }
            ```
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
    def plot_prediction_vs_actual_slices(
        self,
        dataframe_handle: str,
        target_column: str,
        score_column: str,
        path: str,
        *,
        feature_columns: list[str] | None = None,
        bins: int = 10,
        top_n_categories: int = 8,
        max_features: int = 6,
    ) -> dict[str, Any]:
        """Save a global and feature-sliced prediction-vs-actual diagnostic figure.

        This helper creates a global score-bucket panel plus one panel per requested
        feature. Numeric features are quantile-binned with generic labels such as
        `bin_1`, while categorical features are capped to the most-common groups and
        relabeled as `group_1`, `group_2`, and `other` so raw category examples are
        never surfaced in plot metadata.

        Args:
            dataframe_handle: Stored scored dataframe handle.
            target_column: Actual binary target column.
            score_column: Prediction or score column.
            path: Output image path under `/workspace`.
            feature_columns: Optional feature subset to visualize alongside the
                global score-bucket panel.
            bins: Maximum number of quantile buckets for numeric panels.
            top_n_categories: Maximum grouped categories retained for categorical
                panels before collapsing the remainder into `other`.
            max_features: Maximum feature panels included in the figure.

        Returns:
            Saved artifact path and panel metadata.

        Examples:
            ```python
            chart = plot_prediction_vs_actual_slices(
                scored_df_handle,
                "target",
                "pred_score",
                "/workspace/output/pred_vs_actual.png",
                feature_columns=["vehicle_age", "segment"],
            )
            # Returns
            # {
            #     "path": "/workspace/output/pred_vs_actual.png",
            #     "plot_type": "prediction_vs_actual_slices",
            #     "panel_count": 3,
            #     "feature_count": 2,
            # }
            ```
        """
        requested_features = list(feature_columns or [])[:max_features]
        dataframe = self._get_dataframe(dataframe_handle)
        temp = self._prepare_prediction_frame(
            dataframe,
            target_column=target_column,
            score_column=score_column,
            feature_columns=requested_features,
        )

        panel_specs: list[dict[str, Any]] = []
        global_buckets = self._build_quantile_buckets(temp[score_column], max_bins=bins)
        global_summary = self._summarize_prediction_profile(
            temp,
            target_column=target_column,
            score_column=score_column,
            bucket_labels=global_buckets,
        )
        panel_specs.append(
            {
                "title": "Global Prediction Diagnostics",
                "x_label": "Score bucket",
                "summary": global_summary,
                "metadata": {
                    "feature_column": None,
                    "analysis_type": "global",
                    "bucket_count": int(len(global_summary)),
                },
            }
        )

        for feature_column in requested_features:
            feature_series = temp[feature_column]
            if pd.api.types.is_numeric_dtype(feature_series):
                bucket_labels = self._build_quantile_buckets(
                    feature_series,
                    max_bins=bins,
                )
                analysis_type = "numeric"
            else:
                bucket_labels = self._build_categorical_groups(
                    feature_series,
                    top_n_categories=top_n_categories,
                )
                analysis_type = "categorical"

            feature_summary = self._summarize_prediction_profile(
                temp,
                target_column=target_column,
                score_column=score_column,
                bucket_labels=bucket_labels,
            )
            panel_specs.append(
                {
                    "title": f"Prediction vs Actual: {feature_column}",
                    "x_label": feature_column,
                    "summary": feature_summary,
                    "metadata": {
                        "feature_column": feature_column,
                        "analysis_type": analysis_type,
                        "bucket_count": int(len(feature_summary)),
                    },
                }
            )

        panel_count = len(panel_specs)
        figure, axes = plt.subplots(
            panel_count,
            1,
            figsize=(10, max(4 * panel_count, 5)),
            squeeze=False,
        )
        for axis, panel in zip(axes.flatten(), panel_specs):
            self._plot_prediction_profile_panel(
                axis,
                panel["summary"],
                title=panel["title"],
                x_label=panel["x_label"],
            )

        saved_path = self._save_figure(figure, path)
        return {
            "path": saved_path,
            "plot_type": "prediction_vs_actual_slices",
            "target_column": target_column,
            "score_column": score_column,
            "panel_count": int(panel_count),
            "feature_count": int(len(requested_features)),
            "panels": [panel["metadata"] for panel in panel_specs],
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
            ```python
            baseline = train_lightgbm_baseline(
                df_handle,
                "target",
                id_columns=["customer_id"],
                num_threads=1,
            )
            chart = plot_feature_importance(
                baseline["model_handle"],
                "/workspace/feature_importance.png",
            )
            # Returns
            # {
            #     "path": "/workspace/feature_importance.png",
            #     "plot_type": "feature_importance",
            #     "feature_count": 30,
            # }
            ```
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
