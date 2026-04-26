"""PPV-focused error analysis helpers for the standalone ds package."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import ErrorAnalysisResult
from .metrics import summarize_top_p_predictions, top_p_indices
from .modeling import _is_numeric_dtype


def _prepare_target_and_score_frame(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    score_column: str,
) -> pd.DataFrame:
    """Return a clean target-and-score frame for diagnostics.

    Args:
        dataframe: Source dataframe.
        target_column: Binary target column.
        score_column: Numeric score column.

    Returns:
        Cleaned dataframe with numeric target and score columns.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    if score_column not in dataframe.columns:
        raise ValueError(f"Score column {score_column!r} was not found.")

    temp = dataframe[[target_column, score_column]].copy()
    temp[target_column] = pd.to_numeric(temp[target_column], errors="coerce")
    temp[score_column] = pd.to_numeric(temp[score_column], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        raise ValueError(
            "No rows are available after dropping missing target/score values."
        )
    unique_targets = set(temp[target_column].astype(int).unique().tolist())
    if not unique_targets.issubset({0, 1}):
        raise ValueError("Target column must be binary and encoded as 0/1.")
    temp[target_column] = temp[target_column].astype(int)
    return temp


def _resolve_analysis_feature_columns(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    score_column: str,
    feature_columns: list[str] | None,
    id_columns: list[str] | None,
) -> list[str]:
    """Resolve feature columns for aggregate prediction error analysis.

    Args:
        dataframe: Source dataframe.
        target_column: Binary target column.
        score_column: Numeric score column.
        feature_columns: Optional explicit feature subset.
        id_columns: Optional identifier columns excluded from analysis.

    Returns:
        Ordered analysis feature columns.
    """

    excluded = {target_column, score_column, *(id_columns or [])}
    if feature_columns is not None:
        missing_columns = [
            column for column in feature_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")
        resolved = [column for column in feature_columns if column not in excluded]
    else:
        resolved = [
            str(column) for column in dataframe.columns if str(column) not in excluded
        ]
    if not resolved:
        raise ValueError("No feature columns remain after exclusions.")
    return resolved


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return a lightweight two-sample KS distance.

    Args:
        left: First sample.
        right: Second sample.

    Returns:
        Two-sample KS distance.
    """

    if len(left) == 0 or len(right) == 0:
        return 0.0
    left_sorted = np.sort(left.astype(float))
    right_sorted = np.sort(right.astype(float))
    combined = np.sort(np.unique(np.concatenate([left_sorted, right_sorted])))
    left_cdf = np.searchsorted(left_sorted, combined, side="right") / len(left_sorted)
    right_cdf = np.searchsorted(right_sorted, combined, side="right") / len(
        right_sorted
    )
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _entropy(probabilities: np.ndarray) -> float:
    """Return entropy for a probability vector.

    Args:
        probabilities: Probability vector.

    Returns:
        Entropy value.
    """

    positive = probabilities[probabilities > 0]
    if len(positive) == 0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Jensen-Shannon divergence between two distributions.

    Args:
        left: First probability vector.
        right: Second probability vector.

    Returns:
        Jensen-Shannon divergence.
    """

    midpoint = 0.5 * (left + right)

    def _kl_divergence(base: np.ndarray, ref: np.ndarray) -> float:
        valid = (base > 0) & (ref > 0)
        if not np.any(valid):
            return 0.0
        return float(np.sum(base[valid] * np.log2(base[valid] / ref[valid])))

    return 0.5 * _kl_divergence(left, midpoint) + 0.5 * _kl_divergence(right, midpoint)


def _analyze_numeric_false_positives(
    dataframe: pd.DataFrame,
    *,
    column: str,
    false_positive_mask: pd.Series,
    true_positive_mask: pd.Series,
) -> dict[str, Any]:
    """Return aggregate numeric FP-vs-TP diagnostics for one column."""

    fp_series = pd.to_numeric(
        dataframe.loc[false_positive_mask, column], errors="coerce"
    )
    tp_series = pd.to_numeric(
        dataframe.loc[true_positive_mask, column], errors="coerce"
    )
    fp_non_null = fp_series.dropna()
    tp_non_null = tp_series.dropna()

    fp_missing_rate = float(fp_series.isna().mean()) if len(fp_series) else 0.0
    tp_missing_rate = float(tp_series.isna().mean()) if len(tp_series) else 0.0
    pooled_std = (
        float(np.nanstd(np.concatenate([fp_non_null.values, tp_non_null.values])))
        if len(fp_non_null) + len(tp_non_null) > 0
        else 0.0
    )
    mean_shift = (
        float(fp_non_null.mean()) - float(tp_non_null.mean())
        if len(fp_non_null) and len(tp_non_null)
        else 0.0
    )
    effect_size = mean_shift / pooled_std if pooled_std > 0 else 0.0
    ks_distance = _ks_distance(fp_non_null.values, tp_non_null.values)
    pattern_score = max(
        abs(effect_size), abs(fp_missing_rate - tp_missing_rate), ks_distance
    )

    return {
        "column": column,
        "analysis_type": "numeric",
        "dtype": str(dataframe[column].dtype),
        "pattern_score": float(pattern_score),
        "false_positive_count": int(len(fp_series)),
        "true_positive_count": int(len(tp_series)),
        "false_positive_missing_rate": fp_missing_rate,
        "true_positive_missing_rate": tp_missing_rate,
        "missing_rate_diff": float(fp_missing_rate - tp_missing_rate),
        "false_positive_mean": float(fp_non_null.mean()) if len(fp_non_null) else None,
        "true_positive_mean": float(tp_non_null.mean()) if len(tp_non_null) else None,
        "false_positive_median": float(fp_non_null.median())
        if len(fp_non_null)
        else None,
        "true_positive_median": float(tp_non_null.median())
        if len(tp_non_null)
        else None,
        "standardized_mean_shift": float(effect_size),
        "ks_distance": float(ks_distance),
    }


def _analyze_categorical_false_positives(
    dataframe: pd.DataFrame,
    *,
    column: str,
    false_positive_mask: pd.Series,
    true_positive_mask: pd.Series,
) -> dict[str, Any]:
    """Return aggregate categorical FP-vs-TP diagnostics for one column."""

    fp_series = dataframe.loc[false_positive_mask, column]
    tp_series = dataframe.loc[true_positive_mask, column]
    fp_missing_rate = float(fp_series.isna().mean()) if len(fp_series) else 0.0
    tp_missing_rate = float(tp_series.isna().mean()) if len(tp_series) else 0.0
    fp_non_null = fp_series.dropna().astype(str)
    tp_non_null = tp_series.dropna().astype(str)

    categories = sorted(
        set(fp_non_null.unique().tolist()) | set(tp_non_null.unique().tolist())
    )
    if categories:
        fp_probs = (
            fp_non_null.value_counts(normalize=True)
            .reindex(categories, fill_value=0.0)
            .values
        )
        tp_probs = (
            tp_non_null.value_counts(normalize=True)
            .reindex(categories, fill_value=0.0)
            .values
        )
        fp_entropy = _entropy(fp_probs)
        tp_entropy = _entropy(tp_probs)
        normalization = math.log2(len(categories)) if len(categories) > 1 else 1.0
        fp_entropy_norm = fp_entropy / normalization if normalization > 0 else 0.0
        tp_entropy_norm = tp_entropy / normalization if normalization > 0 else 0.0
        fp_concentration = float(np.max(fp_probs)) if len(fp_probs) else 0.0
        tp_concentration = float(np.max(tp_probs)) if len(tp_probs) else 0.0
        js_divergence = _js_divergence(fp_probs, tp_probs)
    else:
        fp_entropy_norm = 0.0
        tp_entropy_norm = 0.0
        fp_concentration = 0.0
        tp_concentration = 0.0
        js_divergence = 0.0

    pattern_score = max(
        abs(fp_missing_rate - tp_missing_rate),
        abs(fp_concentration - tp_concentration),
        abs(fp_entropy_norm - tp_entropy_norm),
        js_divergence,
    )
    return {
        "column": column,
        "analysis_type": "categorical",
        "dtype": str(dataframe[column].dtype),
        "pattern_score": float(pattern_score),
        "false_positive_count": int(len(fp_series)),
        "true_positive_count": int(len(tp_series)),
        "false_positive_missing_rate": fp_missing_rate,
        "true_positive_missing_rate": tp_missing_rate,
        "missing_rate_diff": float(fp_missing_rate - tp_missing_rate),
        "false_positive_cardinality": int(fp_non_null.nunique(dropna=True)),
        "true_positive_cardinality": int(tp_non_null.nunique(dropna=True)),
        "false_positive_concentration": float(fp_concentration),
        "true_positive_concentration": float(tp_concentration),
        "false_positive_normalized_entropy": float(fp_entropy_norm),
        "true_positive_normalized_entropy": float(tp_entropy_norm),
        "distribution_divergence": float(js_divergence),
    }


def analyze_top_p_false_positives(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    score_column: str,
    top_p: float = 0.05,
    feature_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
) -> ErrorAnalysisResult:
    """Analyze false-positive patterns inside the highest-ranked predictions.

    Args:
        dataframe: Scored dataframe.
        target_column: Binary target column.
        score_column: Numeric score column.
        top_p: Fraction retained in the top-ranked slice.
        feature_columns: Optional explicit feature subset to analyze.
        id_columns: Optional identifier columns excluded from analysis.

    Returns:
        Structured false-positive analysis result.
    """

    temp = _prepare_target_and_score_frame(
        dataframe,
        target_column=target_column,
        score_column=score_column,
    )
    analysis_columns = _resolve_analysis_feature_columns(
        dataframe,
        target_column=target_column,
        score_column=score_column,
        feature_columns=feature_columns,
        id_columns=id_columns,
    )
    aligned = dataframe.loc[temp.index, analysis_columns].copy()
    slice_indices = top_p_indices(temp[score_column].values, p=top_p)
    top_slice = aligned.iloc[slice_indices].copy()
    top_slice_targets = temp.iloc[slice_indices][target_column].astype(int)
    false_positive_mask = top_slice_targets == 0
    true_positive_mask = top_slice_targets == 1
    if int(false_positive_mask.sum()) == 0 or int(true_positive_mask.sum()) == 0:
        raise ValueError(
            "Top-p slice must contain both false positives and true positives for comparative analysis."
        )

    numeric_findings: list[dict[str, Any]] = []
    categorical_findings: list[dict[str, Any]] = []
    for column in analysis_columns:
        if _is_numeric_dtype(top_slice[column]):
            numeric_findings.append(
                _analyze_numeric_false_positives(
                    top_slice,
                    column=column,
                    false_positive_mask=false_positive_mask,
                    true_positive_mask=true_positive_mask,
                )
            )
        else:
            categorical_findings.append(
                _analyze_categorical_false_positives(
                    top_slice,
                    column=column,
                    false_positive_mask=false_positive_mask,
                    true_positive_mask=true_positive_mask,
                )
            )

    numeric_findings.sort(key=lambda item: item["pattern_score"], reverse=True)
    categorical_findings.sort(key=lambda item: item["pattern_score"], reverse=True)
    top_p_summary = summarize_top_p_predictions(
        temp[target_column].values,
        temp[score_column].values,
        p=top_p,
    )
    summary = f"Analyzed top-{top_p * 100:.1f}% false positives across {len(analysis_columns)} columns."
    return ErrorAnalysisResult(
        summary=summary,
        top_p_summary=top_p_summary,
        analyzed_columns=analysis_columns,
        false_positive_count=int(false_positive_mask.sum()),
        true_positive_count=int(true_positive_mask.sum()),
        numeric_findings=numeric_findings,
        categorical_findings=categorical_findings,
    )


__all__ = ["analyze_top_p_false_positives"]
