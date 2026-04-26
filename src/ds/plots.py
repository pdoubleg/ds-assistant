"""Notebook-friendly plotting helpers for the standalone ds package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import LightGBMTrainingResult, SavedFigure


def _build_quantile_buckets(series: pd.Series, *, max_bins: int) -> pd.Series:
    """Return stable quantile bucket labels for a numeric series.

    Args:
        series: Numeric series to bucket.
        max_bins: Maximum bucket count.

    Returns:
        Generic quantile bucket labels.
    """

    numeric_series = pd.to_numeric(series, errors="coerce")
    valid = numeric_series.dropna()
    if valid.empty:
        raise ValueError("No numeric rows are available after dropping missing values.")
    bucket_count = max(1, min(max_bins, int(valid.nunique())))
    if bucket_count == 1:
        labels = pd.Series(index=valid.index, data="bin_1", dtype="object")
    else:
        bucket_codes = pd.qcut(valid, q=bucket_count, labels=False, duplicates="drop")
        labels = bucket_codes.map(lambda code: f"bin_{int(code) + 1}").astype("object")
    return labels.reindex(series.index)


def _build_categorical_groups(series: pd.Series, *, top_n_categories: int) -> pd.Series:
    """Return grouped labels for a categorical series.

    Args:
        series: Categorical series to group.
        top_n_categories: Maximum retained explicit groups.

    Returns:
        Generic grouped labels.
    """

    normalized = series.astype("object").where(series.notna(), "__missing__")
    counts = normalized.value_counts(dropna=False)
    kept_values = counts.head(max(1, top_n_categories)).index.tolist()
    mapping = {value: f"group_{index + 1}" for index, value in enumerate(kept_values)}
    grouped = normalized.map(lambda value: mapping.get(value, "other"))
    return grouped.astype("object")


def _summarize_prediction_profile(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    score_column: str,
    bucket_labels: pd.Series,
) -> pd.DataFrame:
    """Aggregate prediction-vs-actual metrics for plotting.

    Args:
        dataframe: Source dataframe.
        target_column: Actual target column.
        score_column: Prediction score column.
        bucket_labels: Bucket labels aligned to the dataframe index.

    Returns:
        Aggregated summary dataframe.
    """

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
    return summary.sort_values("bucket_label").reset_index(drop=True)


def _plot_prediction_profile_panel(
    axis: matplotlib.axes.Axes,
    summary: pd.DataFrame,
    *,
    title: str,
    x_label: str,
) -> None:
    """Plot one prediction profile panel.

    Args:
        axis: Axis to draw on.
        summary: Aggregated summary dataframe.
        title: Panel title.
        x_label: Panel x-axis label.

    Returns:
        None
    """

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


def save_figure(
    figure: matplotlib.figure.Figure,
    path: str | Path,
    *,
    figure_type: str,
    metadata: dict[str, Any] | None = None,
) -> SavedFigure:
    """Save a matplotlib figure to disk.

    Args:
        figure: Figure to save.
        path: Output image path.
        figure_type: Logical plot type name.
        metadata: Optional plot metadata.

    Returns:
        Saved figure metadata.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return SavedFigure(
        path=output_path.resolve(),
        figure_type=figure_type,
        metadata=dict(metadata or {}),
    )


def plot_missingness(
    dataframe: pd.DataFrame,
    *,
    top_n: int = 30,
    save_path: str | Path | None = None,
) -> tuple[matplotlib.figure.Figure, dict[str, Any]]:
    """Plot missingness rates for the most-missing columns.

    Args:
        dataframe: Source dataframe.
        top_n: Maximum number of columns shown in the chart.
        save_path: Optional output path for saving the figure.

    Returns:
        Tuple of matplotlib figure and chart metadata.
    """

    missing = (
        dataframe.isna().mean().sort_values(ascending=False).head(top_n).reset_index()
    )
    missing.columns = ["column", "missing_rate"]
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(data=missing, x="missing_rate", y="column", ax=axis, color="#4C72B0")
    axis.set_title("Top Missingness Rates")
    axis.set_xlabel("Missing rate")
    axis.set_ylabel("Column")
    metadata = {"plot_type": "missingness_bar", "column_count": int(len(missing))}
    if save_path is not None:
        metadata["saved_figure"] = save_figure(
            figure,
            save_path,
            figure_type="missingness_bar",
            metadata=metadata,
        )
    return figure, metadata


def plot_numeric_histogram(
    dataframe: pd.DataFrame,
    column: str,
    *,
    bins: int = 30,
    save_path: str | Path | None = None,
) -> tuple[matplotlib.figure.Figure, dict[str, Any]]:
    """Plot a histogram for a numeric feature.

    Args:
        dataframe: Source dataframe.
        column: Numeric column to visualize.
        bins: Number of histogram bins.
        save_path: Optional output path for saving the figure.

    Returns:
        Tuple of matplotlib figure and chart metadata.
    """

    if column not in dataframe.columns:
        raise ValueError(f"Column {column!r} was not found.")
    series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if series.empty:
        raise ValueError("The requested column has no numeric values to plot.")
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(series, bins=bins, ax=axis, color="#55A868")
    axis.set_title(f"Histogram: {column}")
    axis.set_xlabel(column)
    metadata = {
        "plot_type": "numeric_histogram",
        "column": column,
        "row_count": int(series.shape[0]),
        "bins": int(bins),
    }
    if save_path is not None:
        metadata["saved_figure"] = save_figure(
            figure,
            save_path,
            figure_type="numeric_histogram",
            metadata=metadata,
        )
    return figure, metadata


def plot_target_rate_by_numeric_bin(
    dataframe: pd.DataFrame,
    *,
    feature_column: str,
    target_column: str,
    bins: int = 10,
    save_path: str | Path | None = None,
) -> tuple[matplotlib.figure.Figure, dict[str, Any]]:
    """Plot target rate by quantile bin for a numeric feature.

    Args:
        dataframe: Source dataframe.
        feature_column: Numeric feature column to bucket.
        target_column: Numeric or binary target column.
        bins: Maximum number of quantile buckets.
        save_path: Optional output path for saving the figure.

    Returns:
        Tuple of matplotlib figure and chart metadata.
    """

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
        raise ValueError("No numeric rows are available after dropping missing values.")
    temp["bin"] = pd.qcut(
        temp[feature_column],
        q=min(bins, temp[feature_column].nunique()),
        duplicates="drop",
    )
    binned = (
        temp.groupby("bin", observed=True)
        .agg(row_count=(target_column, "size"), target_rate=(target_column, "mean"))
        .reset_index()
    )
    binned["bin_label"] = [f"bin_{index + 1}" for index in range(len(binned))]
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=binned, x="bin_label", y="target_rate", marker="o", ax=axis)
    axis.set_title(f"Target Rate by Quantile Bin: {feature_column}")
    axis.set_xlabel("Quantile bin")
    axis.set_ylabel("Target rate")
    metadata = {
        "plot_type": "target_rate_by_bin",
        "feature_column": feature_column,
        "target_column": target_column,
        "bin_count": int(len(binned)),
    }
    if save_path is not None:
        metadata["saved_figure"] = save_figure(
            figure,
            save_path,
            figure_type="target_rate_by_bin",
            metadata=metadata,
        )
    return figure, metadata


def plot_prediction_diagnostics(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    score_column: str,
    bins: int = 10,
    save_path: str | Path | None = None,
) -> tuple[matplotlib.figure.Figure, dict[str, Any]]:
    """Plot aggregate prediction-vs-actual diagnostics using score buckets.

    Args:
        dataframe: Scored dataframe.
        target_column: Actual binary target column.
        score_column: Prediction or score column.
        bins: Maximum number of quantile buckets.
        save_path: Optional output path for saving the figure.

    Returns:
        Tuple of matplotlib figure and chart metadata.
    """

    if target_column not in dataframe.columns or score_column not in dataframe.columns:
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
    bucketed["bucket_label"] = [f"bucket_{index + 1}" for index in range(len(bucketed))]
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
    metadata = {
        "plot_type": "prediction_diagnostics",
        "target_column": target_column,
        "score_column": score_column,
        "bucket_count": int(len(bucketed)),
    }
    if save_path is not None:
        metadata["saved_figure"] = save_figure(
            figure,
            save_path,
            figure_type="prediction_diagnostics",
            metadata=metadata,
        )
    return figure, metadata


def plot_feature_importance(
    model_result: LightGBMTrainingResult,
    *,
    top_n: int = 30,
    save_path: str | Path | None = None,
) -> tuple[matplotlib.figure.Figure, dict[str, Any]]:
    """Plot LightGBM feature importances.

    Args:
        model_result: Fitted LightGBM model result.
        top_n: Maximum number of features shown in the plot.
        save_path: Optional output path for saving the figure.

    Returns:
        Tuple of matplotlib figure and chart metadata.
    """

    rows = sorted(
        model_result.feature_importance_gain.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    importance = pd.DataFrame(rows, columns=["feature", "gain"])
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance, x="gain", y="feature", ax=axis, color="#C44E52")
    axis.set_title("LightGBM Feature Importance (Gain)")
    metadata = {
        "plot_type": "feature_importance",
        "feature_count": int(len(importance)),
    }
    if save_path is not None:
        metadata["saved_figure"] = save_figure(
            figure,
            save_path,
            figure_type="feature_importance",
            metadata=metadata,
        )
    return figure, metadata


def plot_prediction_vs_actual_slices(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    score_column: str,
    feature_columns: list[str] | None = None,
    bins: int = 10,
    top_n_categories: int = 8,
    max_features: int = 6,
    save_path: str | Path | None = None,
) -> tuple[matplotlib.figure.Figure, dict[str, Any]]:
    """Plot global and feature-sliced prediction-vs-actual diagnostics.

    Args:
        dataframe: Scored dataframe.
        target_column: Actual binary target column.
        score_column: Prediction or score column.
        feature_columns: Optional feature subset to visualize.
        bins: Maximum number of quantile buckets for numeric panels.
        top_n_categories: Maximum grouped categories retained for categorical
            panels.
        max_features: Maximum feature panels included in the figure.
        save_path: Optional output path for saving the figure.

    Returns:
        Tuple of matplotlib figure and chart metadata.
    """

    requested_features = list(feature_columns or [])[:max_features]
    required_columns = [target_column, score_column, *requested_features]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    temp = dataframe[required_columns].copy()
    temp[target_column] = pd.to_numeric(temp[target_column], errors="coerce")
    temp[score_column] = pd.to_numeric(temp[score_column], errors="coerce")
    temp = temp.dropna(subset=[target_column, score_column])
    if temp.empty:
        raise ValueError(
            "No rows are available after dropping missing target/score values."
        )

    panel_specs: list[dict[str, Any]] = []
    global_buckets = _build_quantile_buckets(temp[score_column], max_bins=bins)
    global_summary = _summarize_prediction_profile(
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
            bucket_labels = _build_quantile_buckets(feature_series, max_bins=bins)
            analysis_type = "numeric"
        else:
            bucket_labels = _build_categorical_groups(
                feature_series, top_n_categories=top_n_categories
            )
            analysis_type = "categorical"
        feature_summary = _summarize_prediction_profile(
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
        _plot_prediction_profile_panel(
            axis,
            panel["summary"],
            title=panel["title"],
            x_label=panel["x_label"],
        )

    metadata = {
        "plot_type": "prediction_vs_actual_slices",
        "target_column": target_column,
        "score_column": score_column,
        "panel_count": int(panel_count),
        "feature_count": int(len(requested_features)),
        "panels": [panel["metadata"] for panel in panel_specs],
    }
    if save_path is not None:
        metadata["saved_figure"] = save_figure(
            figure,
            save_path,
            figure_type="prediction_vs_actual_slices",
            metadata=metadata,
        )
    return figure, metadata


__all__ = [
    "plot_feature_importance",
    "plot_missingness",
    "plot_numeric_histogram",
    "plot_prediction_diagnostics",
    "plot_prediction_vs_actual_slices",
    "plot_target_rate_by_numeric_bin",
    "save_figure",
]
