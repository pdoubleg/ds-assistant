"""Ranking metrics used by the standalone ds modeling workflow."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def ppv_at_top_p(
    y_true: np.ndarray | list[int] | list[float],
    y_score: np.ndarray | list[float],
    *,
    p: float = 0.05,
) -> float:
    """Compute PPV within the highest-ranked slice.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Model scores used for ranking.
        p: Fraction of rows retained in the top slice.

    Returns:
        Precision observed within the retained slice.
    """

    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score).astype(float)
    if len(labels) == 0:
        return 0.0
    n_top = max(1, int(math.ceil(len(labels) * p)))
    top_idx = np.argsort(-scores, kind="mergesort")[:n_top]
    return float(np.mean(labels[top_idx]))


def recall_at_top_p(
    y_true: np.ndarray | list[int] | list[float],
    y_score: np.ndarray | list[float],
    *,
    p: float = 0.05,
) -> float:
    """Compute recall captured by the highest-ranked slice.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Model scores used for ranking.
        p: Fraction of rows retained in the top slice.

    Returns:
        Recall captured by the retained rows.
    """

    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score).astype(float)
    positives = float(np.sum(labels))
    if positives == 0:
        return 0.0
    n_top = max(1, int(math.ceil(len(labels) * p)))
    top_idx = np.argsort(-scores, kind="mergesort")[:n_top]
    return float(np.sum(labels[top_idx]) / positives)


def lift_at_top_p(
    y_true: np.ndarray | list[int] | list[float],
    y_score: np.ndarray | list[float],
    *,
    p: float = 0.05,
) -> float:
    """Compute lift relative to the overall base rate.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Model scores used for ranking.
        p: Fraction of rows retained in the top slice.

    Returns:
        Lift of the retained slice over the full-sample base rate.
    """

    labels = np.asarray(y_true).astype(int)
    base_rate = float(np.mean(labels)) if len(labels) else 0.0
    if base_rate == 0.0:
        return 0.0
    return float(ppv_at_top_p(labels, y_score, p=p) / base_rate)


def top_p_indices(
    y_score: np.ndarray | list[float],
    *,
    p: float = 0.05,
) -> np.ndarray:
    """Return row indices retained by the top-p cutoff.

    Args:
        y_score: Model scores used for ranking.
        p: Fraction of rows retained in the top slice.

    Returns:
        Ranked row indices in descending score order.
    """

    scores = np.asarray(y_score).astype(float)
    if len(scores) == 0:
        return np.array([], dtype=int)
    n_top = max(1, int(math.ceil(len(scores) * p)))
    return np.argsort(-scores, kind="mergesort")[:n_top]


def summarize_top_p_predictions(
    y_true: np.ndarray | list[int] | list[float],
    y_score: np.ndarray | list[float],
    *,
    p: float = 0.05,
) -> dict[str, Any]:
    """Summarize aggregate prediction quality in the top-p slice.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Model scores used for ranking.
        p: Fraction of rows retained in the top slice.

    Returns:
        Aggregate top-p ranking metrics.
    """

    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score).astype(float)
    if len(labels) != len(scores):
        raise ValueError("`y_true` and `y_score` must have the same length.")
    if len(labels) == 0:
        return {
            "top_p": float(p),
            "row_count": 0,
            "top_p_row_count": 0,
            "score_threshold": None,
            "true_positive_count": 0,
            "false_positive_count": 0,
            "ppv_at_p": 0.0,
            "recall_at_p": 0.0,
            "lift_at_p": 0.0,
            "base_rate": 0.0,
        }

    selected = top_p_indices(scores, p=p)
    selected_labels = labels[selected]
    selected_scores = scores[selected]
    return {
        "top_p": float(p),
        "row_count": int(len(labels)),
        "top_p_row_count": int(len(selected)),
        "score_threshold": float(np.min(selected_scores))
        if len(selected_scores)
        else None,
        "true_positive_count": int(np.sum(selected_labels == 1)),
        "false_positive_count": int(np.sum(selected_labels == 0)),
        "ppv_at_p": float(ppv_at_top_p(labels, scores, p=p)),
        "recall_at_p": float(recall_at_top_p(labels, scores, p=p)),
        "lift_at_p": float(lift_at_top_p(labels, scores, p=p)),
        "base_rate": float(np.mean(labels)),
    }


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for tied values.

    Args:
        values: Values to rank.

    Returns:
        Average rank per value.
    """

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        right = index + 1
        while right < len(values) and values[order[right]] == values[order[index]]:
            right += 1
        avg_rank = 0.5 * (index + right - 1) + 1.0
        ranks[order[index:right]] = avg_rank
        index = right
    return ranks


def fast_auc_score(
    y_true: np.ndarray | list[int] | list[float],
    y_score: np.ndarray | list[float],
) -> float:
    """Compute a fast binary AUC estimate from average ranks.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Scores to evaluate.

    Returns:
        Approximate AUC value.
    """

    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score).astype(float)
    positive_mask = labels == 1
    negative_mask = labels == 0
    n_pos = int(positive_mask.sum())
    n_neg = int(negative_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata_average(scores)
    positive_rank_sum = ranks[positive_mask].sum()
    auc = (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


__all__ = [
    "fast_auc_score",
    "lift_at_top_p",
    "ppv_at_top_p",
    "recall_at_top_p",
    "summarize_top_p_predictions",
    "top_p_indices",
]
