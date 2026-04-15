"""Metrics and ranking helpers for Monty modeling workflows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    chi2,
    f_classif,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold


def infer_task_type(target: pd.Series) -> str:
    """Infer whether a target looks like classification or regression.

    Args:
        target (pd.Series): Target values used for model evaluation.

    Returns:
        str: Either ``classification`` or ``regression``.
    """
    non_null = target.dropna()
    if non_null.empty:
        return "classification"

    if non_null.dtype.kind in {"b", "O", "U", "S"}:
        return "classification"
    if pd.api.types.is_integer_dtype(non_null) and non_null.nunique() <= 20:
        return "classification"
    if non_null.nunique() <= min(20, max(2, int(len(non_null) * 0.1))):
        return "classification"
    return "regression"


def _align_categorical_frame(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Encode categorical columns into LightGBM-safe numeric values.

    Args:
        train_frame (pd.DataFrame): Training feature frame.
        validation_frame (pd.DataFrame | None): Optional validation frame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: Encoded train and validation
            feature frames.
    """
    train_encoded = train_frame.copy()
    validation_encoded = (
        validation_frame.copy() if validation_frame is not None else None
    )

    for column in train_encoded.columns:
        train_series = train_encoded[column]
        if pd.api.types.is_numeric_dtype(train_series):
            train_encoded[column] = pd.to_numeric(train_series, errors="coerce")
            if validation_encoded is not None:
                validation_encoded[column] = pd.to_numeric(
                    validation_encoded[column],
                    errors="coerce",
                )
            continue

        categories = pd.Index(
            pd.Series(train_series).astype("string").dropna().unique()
        )
        train_categorical = pd.Categorical(
            train_series.astype("string"),
            categories=categories,
        )
        train_encoded[column] = pd.Series(
            train_categorical.codes, index=train_encoded.index
        )
        train_encoded[column] = train_encoded[column].replace(-1, np.nan)

        if validation_encoded is not None:
            validation_categorical = pd.Categorical(
                validation_encoded[column].astype("string"),
                categories=categories,
            )
            validation_encoded[column] = pd.Series(
                validation_categorical.codes,
                index=validation_encoded.index,
            )
            validation_encoded[column] = validation_encoded[column].replace(-1, -1.0)

    return train_encoded, validation_encoded


def _validate_matching_feature_columns(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
) -> None:
    """Raise a clear error when validation columns do not match training columns.

    Args:
        train_frame (pd.DataFrame): Training feature frame.
        validation_frame (pd.DataFrame | None): Optional validation feature frame.

    Raises:
        ValueError: If the validation columns do not exactly match the training
            columns.
    """
    if validation_frame is None:
        return

    train_columns = [str(column) for column in train_frame.columns]
    validation_columns = [str(column) for column in validation_frame.columns]
    if train_columns == validation_columns:
        return

    train_set = set(train_columns)
    validation_set = set(validation_columns)
    missing_columns = [
        column for column in train_columns if column not in validation_set
    ]
    unexpected_columns = [
        column for column in validation_columns if column not in train_set
    ]

    message_parts = [
        "Validation feature columns do not match the training feature columns after pipeline materialization."
    ]
    if missing_columns:
        message_parts.append(
            f"Missing validation columns: {', '.join(missing_columns)}."
        )
    if unexpected_columns:
        message_parts.append(
            f"Unexpected validation columns: {', '.join(unexpected_columns)}."
        )
    if not missing_columns and not unexpected_columns:
        message_parts.append(
            "The column sets match, but the column order differs. "
            f"Training order: {train_columns}. Validation order: {validation_columns}."
        )
    raise ValueError(" ".join(message_parts))


def prepare_model_frames(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Prepare frames for model fitting by coercing objects to numeric codes.

    Args:
        train_frame (pd.DataFrame): Training feature frame.
        validation_frame (pd.DataFrame | None): Optional validation frame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: Model-ready feature frames.
    """
    train_ready = train_frame.replace([np.inf, -np.inf], np.nan)
    validation_ready = (
        validation_frame.replace([np.inf, -np.inf], np.nan)
        if validation_frame is not None
        else None
    )
    _validate_matching_feature_columns(train_ready, validation_ready)
    return _align_categorical_frame(train_ready, validation_ready)


def prepare_targets(
    train_target: pd.Series,
    validation_target: pd.Series | None = None,
    *,
    task_type: str,
) -> tuple[pd.Series, pd.Series | None]:
    """Prepare targets for LightGBM fitting.

    Args:
        train_target (pd.Series): Training target values.
        validation_target (pd.Series | None): Optional validation target values.
        task_type (str): Either ``classification`` or ``regression``.

    Returns:
        tuple[pd.Series, pd.Series | None]: Model-ready target vectors.
    """
    if task_type == "regression":
        return pd.to_numeric(train_target, errors="coerce"), (
            pd.to_numeric(validation_target, errors="coerce")
            if validation_target is not None
            else None
        )

    classes = pd.Index(pd.Series(train_target).dropna().astype("string").unique())
    train_categorical = pd.Categorical(
        train_target.astype("string"), categories=classes
    )
    train_encoded = pd.Series(train_categorical.codes, index=train_target.index)

    validation_encoded = None
    if validation_target is not None:
        validation_categorical = pd.Categorical(
            validation_target.astype("string"),
            categories=classes,
        )
        validation_encoded = pd.Series(
            validation_categorical.codes,
            index=validation_target.index,
        )

    return train_encoded, validation_encoded


def build_lightgbm_estimator(
    *,
    task_type: str,
    class_count: int | None = None,
    random_state: int = 0,
) -> lgb.LGBMModel:
    """Build a small LightGBM estimator for diagnostics.

    Args:
        task_type (str): Either ``classification`` or ``regression``.
        class_count (int | None): Number of classes for multiclass classification.
        random_state (int): Random seed for reproducibility.

    Returns:
        lgb.LGBMModel: LightGBM estimator instance.
    """
    common_args = {
        "random_state": random_state,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "n_jobs": 1,
        "verbosity": -1,
        "force_row_wise": True,
    }
    if task_type == "classification":
        if class_count is not None and class_count > 2:
            return lgb.LGBMClassifier(
                objective="multiclass",
                num_class=int(class_count),
                **common_args,
            )
        return lgb.LGBMClassifier(
            objective="binary",
            **common_args,
        )
    return lgb.LGBMRegressor(
        objective="regression",
        **common_args,
    )


def summarize_cv_metrics(metrics_by_fold: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Aggregate per-fold metric dictionaries into a compact summary.

    Args:
        metrics_by_fold (Sequence[dict[str, float]]): Fold metrics in evaluation order.

    Returns:
        dict[str, Any]: Mean/std summaries plus the original fold metrics.
    """
    if not metrics_by_fold:
        return {"folds": [], "mean_metrics": {}, "std_metrics": {}}

    metric_names = sorted({name for fold in metrics_by_fold for name in fold})
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in metrics_by_fold if metric_name in fold]
        if not values:
            continue
        mean_metrics[metric_name] = float(np.mean(values))
        std_metrics[metric_name] = float(np.std(values))
    return {
        "folds": list(metrics_by_fold),
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }


def _normalize_metric_name(metric_name: str, *, task_type: str) -> str:
    """Normalize user-facing metric aliases into canonical registry keys.

    Args:
        metric_name (str): Raw metric name or alias.
        task_type (str): Either ``classification`` or ``regression``.

    Returns:
        str: Canonical metric name.
    """
    normalized = str(metric_name).strip().lower()
    alias_map = {
        "auc": "roc_auc",
        "roc": "roc_auc",
        "precision_at_k": "ppv",
        "ppv_at_k": "ppv",
        "ppv@k": "ppv",
        "precision_at_top_k": "ppv",
        "positive_predictive_value": "ppv",
        "mean_squared_error": "mse",
        "mean_absolute_error": "mae",
        "root_mean_squared_error": "rmse",
    }
    canonical = alias_map.get(normalized, normalized)
    if task_type == "classification" and canonical in {"mse", "mae", "rmse", "r2"}:
        raise ValueError(
            f"Metric {metric_name!r} is not valid for classification tasks."
        )
    if task_type == "regression" and canonical in {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "ppv",
    }:
        raise ValueError(f"Metric {metric_name!r} is not valid for regression tasks.")
    return canonical


def _coerce_probability_array(
    y_pred_proba: pd.Series | np.ndarray | Sequence[Any] | None,
) -> np.ndarray | None:
    """Convert probability-like input into a NumPy array when provided."""
    if y_pred_proba is None:
        return None
    return np.asarray(y_pred_proba)


def _positive_class_probabilities(
    y_pred_proba: pd.Series | np.ndarray | Sequence[Any] | None,
) -> np.ndarray:
    """Return the positive-class probability vector used by PPV-style metrics.

    Args:
        y_pred_proba (pd.Series | np.ndarray | Sequence[Any] | None): Predicted
            probabilities or scores.

    Returns:
        np.ndarray: One-dimensional score vector.

    Raises:
        ValueError: If the probability payload cannot support binary PPV scoring.
    """
    probabilities = _coerce_probability_array(y_pred_proba)
    if probabilities is None:
        raise ValueError("PPV scoring requires predicted probabilities or scores.")
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        return probabilities[:, 1].astype(float)
    raise ValueError(
        "PPV scoring currently supports only binary probability vectors or "
        "two-column probability arrays."
    )


def metric_ppv(
    y_true: pd.Series | np.ndarray | Sequence[Any],
    y_pred_proba: pd.Series | np.ndarray | Sequence[Any],
    *,
    top_p: float | None = None,
    top_k: int | None = None,
    positive_label: int | str = 1,
) -> float:
    """Compute positive predictive value among the highest-ranked predictions.

    Args:
        y_true (pd.Series | np.ndarray | Sequence[Any]): Ground-truth binary labels.
        y_pred_proba (pd.Series | np.ndarray | Sequence[Any]): Predicted probabilities
            or binary confidence scores.
        top_p (float | None): Fraction of rows to keep, between 0 and 1.
        top_k (int | None): Fixed number of highest-ranked rows to keep.
        positive_label (int | str): Label treated as the positive class.

    Returns:
        float: Positive predictive value among the retained rows.

    Raises:
        ValueError: If ``top_p`` / ``top_k`` is invalid or the targets are not binary.
    """
    if (top_p is None and top_k is None) or (top_p is not None and top_k is not None):
        raise ValueError("Provide exactly one of top_p or top_k for PPV scoring.")
    labels = pd.Series(y_true)
    if labels.nunique(dropna=True) > 2:
        raise ValueError("PPV scoring currently supports only binary classification.")

    probabilities = _positive_class_probabilities(y_pred_proba)
    if len(labels) != len(probabilities):
        raise ValueError("y_true and y_pred_proba must have the same length.")

    if top_p is not None:
        if not (0 < float(top_p) <= 1):
            raise ValueError("top_p must be between 0 and 1.")
        top_n = max(1, math.ceil(len(labels) * float(top_p)))
    else:
        if int(top_k or 0) <= 0:
            raise ValueError("top_k must be a positive integer.")
        top_n = min(len(labels), int(top_k))

    ranked = pd.DataFrame(
        {
            "label": labels.values,
            "predicted_prob": probabilities,
        }
    )
    top_ranked = ranked.sort_values("predicted_prob", ascending=False).head(top_n)
    return float(
        top_ranked["label"].value_counts(normalize=True).get(positive_label, 0.0)
    )


def compute_metric_score(
    y_true: pd.Series | np.ndarray | Sequence[Any],
    y_pred: pd.Series | np.ndarray | Sequence[Any],
    *,
    metric_name: str,
    task_type: str,
    y_pred_proba: pd.Series | np.ndarray | Sequence[Any] | None = None,
    average: str | None = None,
    pos_label: int | str = 1,
    top_p: float | None = None,
    top_k: int | None = None,
    greater_is_better: bool = True,
    metric_kwargs: dict[str, Any] | None = None,
) -> float:
    """Compute one metric score with scorer-style options.

    Args:
        y_true (pd.Series | np.ndarray | Sequence[Any]): Ground-truth values.
        y_pred (pd.Series | np.ndarray | Sequence[Any]): Predicted labels or values.
        metric_name (str): Metric name or alias.
        task_type (str): Either ``classification`` or ``regression``.
        y_pred_proba (pd.Series | np.ndarray | Sequence[Any] | None): Optional
            probabilities used by AUC, log-loss, and PPV scoring.
        average (str | None): Optional averaging mode for precision/recall/F1.
        pos_label (int | str): Positive class label for binary metrics.
        top_p (float | None): Optional top fraction for PPV scoring.
        top_k (int | None): Optional top-k cutoff for PPV scoring.
        greater_is_better (bool): Whether higher values should remain positive.
        metric_kwargs (dict[str, Any] | None): Additional metric-specific kwargs.

    Returns:
        float: Computed metric score.
    """
    kwargs = dict(metric_kwargs or {})
    canonical_name = _normalize_metric_name(metric_name, task_type=task_type)
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    probabilities = _coerce_probability_array(y_pred_proba)

    if task_type == "classification":
        resolved_average = average or (
            "binary" if len(np.unique(truth)) == 2 else "weighted"
        )
        if canonical_name == "accuracy":
            score = float(accuracy_score(truth, predictions))
        elif canonical_name == "balanced_accuracy":
            score = float(balanced_accuracy_score(truth, predictions))
        elif canonical_name == "precision":
            score = float(
                precision_score(
                    truth,
                    predictions,
                    average=resolved_average,
                    pos_label=pos_label,
                    zero_division=0,
                    **kwargs,
                )
            )
        elif canonical_name == "recall":
            score = float(
                recall_score(
                    truth,
                    predictions,
                    average=resolved_average,
                    pos_label=pos_label,
                    zero_division=0,
                    **kwargs,
                )
            )
        elif canonical_name == "f1":
            score = float(
                f1_score(
                    truth,
                    predictions,
                    average=resolved_average,
                    pos_label=pos_label,
                    zero_division=0,
                    **kwargs,
                )
            )
        elif canonical_name == "roc_auc":
            if probabilities is None:
                raise ValueError("roc_auc scoring requires y_pred_proba.")
            if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                if len(np.unique(truth)) == 2:
                    score = float(roc_auc_score(truth, probabilities[:, 1], **kwargs))
                else:
                    score = float(
                        roc_auc_score(
                            truth,
                            probabilities,
                            multi_class=kwargs.pop("multi_class", "ovr"),
                            average=kwargs.pop("average", "weighted"),
                            **kwargs,
                        )
                    )
            else:
                score = float(roc_auc_score(truth, probabilities, **kwargs))
        elif canonical_name == "log_loss":
            if probabilities is None:
                raise ValueError("log_loss scoring requires y_pred_proba.")
            score = float(log_loss(truth, probabilities, **kwargs))
        elif canonical_name == "ppv":
            score = metric_ppv(
                truth,
                probabilities,
                top_p=top_p,
                top_k=top_k,
                positive_label=pos_label,
            )
        else:  # pragma: no cover - guarded by normalization
            raise ValueError(f"Unsupported classification metric: {metric_name!r}.")
    else:
        if canonical_name == "r2":
            score = float(r2_score(truth, predictions, **kwargs))
        elif canonical_name == "mae":
            score = float(mean_absolute_error(truth, predictions, **kwargs))
        elif canonical_name == "mse":
            score = float(mean_squared_error(truth, predictions, **kwargs))
        elif canonical_name == "rmse":
            score = float(np.sqrt(mean_squared_error(truth, predictions, **kwargs)))
        else:  # pragma: no cover - guarded by normalization
            raise ValueError(f"Unsupported regression metric: {metric_name!r}.")

    return score if greater_is_better else -score


def _merge_scorer_metric(
    metrics: dict[str, float],
    *,
    scorer: Any | None,
    y_true: pd.Series | np.ndarray | Sequence[Any],
    y_pred: pd.Series | np.ndarray | Sequence[Any],
    task_type: str,
    y_pred_proba: pd.Series | np.ndarray | Sequence[Any] | None = None,
) -> dict[str, float]:
    """Merge an optional scorer-handle score into a metrics dictionary."""
    if scorer is None:
        return metrics
    metric_name = _normalize_metric_name(
        getattr(scorer, "metric_name"),
        task_type=task_type,
    )
    merged = dict(metrics)
    merged[metric_name] = compute_metric_score(
        y_true,
        y_pred,
        metric_name=metric_name,
        task_type=task_type,
        y_pred_proba=y_pred_proba,
        average=getattr(scorer, "average", None),
        pos_label=getattr(scorer, "pos_label", 1),
        top_p=getattr(scorer, "top_p", None),
        top_k=getattr(scorer, "top_k", None),
        greater_is_better=True,
        metric_kwargs=getattr(scorer, "metric_kwargs", {}),
    )
    return merged


def build_default_splitter(
    *,
    task_type: str,
    cv_folds: int,
    random_state: int,
) -> Any:
    """Build the default splitter used when no stored splitter handle is supplied."""
    return (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        if task_type == "classification"
        else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    )


def _build_split_iterator(
    *,
    task_type: str,
    cv_folds: int,
    random_state: int,
    splitter: Any | None = None,
    groups: pd.Series | np.ndarray | Sequence[Any] | None = None,
) -> tuple[Any, Any]:
    """Resolve the active CV splitter and split arguments."""
    if splitter is None:
        splitter_instance = build_default_splitter(
            task_type=task_type,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        return splitter_instance, None

    from ..registry.splitting import materialize_splitter

    splitter_instance = materialize_splitter(splitter)
    if getattr(splitter, "requires_groups", False) and groups is None:
        raise ValueError(
            "The requested splitter requires group values. Supply a group column."
        )
    return splitter_instance, (np.asarray(groups) if groups is not None else None)


def compute_prediction_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    task_type: str,
    y_pred_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a JSON-friendly set of prediction metrics.

    Args:
        y_true (pd.Series | np.ndarray): Ground-truth target values.
        y_pred (pd.Series | np.ndarray): Predicted labels or regression outputs.
        task_type (str): Either ``classification`` or ``regression``.
        y_pred_proba (np.ndarray | None): Optional predicted probabilities.

    Returns:
        dict[str, float]: Computed metric values.
    """
    metrics: dict[str, float] = {}
    if task_type == "classification":
        average_method = "binary" if len(np.unique(y_true)) == 2 else "weighted"
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average_method, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=average_method, zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average=average_method, zero_division=0)
        )
        if y_pred_proba is not None:
            try:
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
                    if len(np.unique(y_true)) == 2:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_true, y_pred_proba[:, 1])
                        )
                    else:
                        metrics["roc_auc"] = float(
                            roc_auc_score(
                                y_true,
                                y_pred_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                        )
                metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
            except (ValueError, IndexError):
                pass
        return metrics

    metrics["r2"] = float(r2_score(y_true, y_pred))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    return metrics


def rank_feature_target_metrics(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    *,
    method: str,
    random_state: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rank features against a target using a univariate scoring method.

    Args:
        feature_frame (pd.DataFrame): Candidate feature frame.
        target (pd.Series): Target vector.
        method (str): One of ``mutual_info``, ``f_score``, or ``chi2``.
        random_state (int): Random seed for stochastic MI variants.

    Returns:
        tuple[list[dict[str, Any]], list[str]]: Ranking rows and warnings.
    """
    warnings: list[str] = [
        "Target-aware rankings can be optimistic when computed on the full dataset. Prefer train-only data when possible."
    ]
    feature_ready, _ = prepare_model_frames(feature_frame)
    task_type = infer_task_type(target)
    target_ready, _ = prepare_targets(target, task_type=task_type)

    ranking_rows: list[dict[str, Any]] = []
    if feature_ready.empty:
        return ranking_rows, warnings

    if method == "mutual_info":
        if task_type == "classification":
            scores = mutual_info_classif(
                feature_ready.fillna(-1.0),
                target_ready,
                random_state=random_state,
            )
        else:
            scores = mutual_info_regression(
                feature_ready.fillna(-1.0),
                target_ready,
                random_state=random_state,
            )
    elif method == "f_score":
        if task_type != "classification":
            raise ValueError("'f_score' is only supported for classification targets.")
        scores, pvalues = f_classif(feature_ready.fillna(0.0), target_ready)
        for column, score, pvalue in zip(feature_ready.columns, scores, pvalues):
            ranking_rows.append(
                {
                    "feature": str(column),
                    "score": float(score) if np.isfinite(score) else None,
                    "p_value": float(pvalue) if np.isfinite(pvalue) else None,
                }
            )
        return sorted(
            ranking_rows,
            key=lambda row: row["score"] if row["score"] is not None else float("-inf"),
            reverse=True,
        ), warnings
    elif method == "chi2":
        if task_type != "classification":
            raise ValueError("'chi2' is only supported for classification targets.")
        non_negative_columns = [
            str(column)
            for column in feature_ready.columns
            if feature_ready[column].min(skipna=True) >= 0
        ]
        skipped_columns = [
            str(column)
            for column in feature_ready.columns
            if column not in non_negative_columns
        ]
        if skipped_columns:
            skipped_text = ", ".join(skipped_columns[:10])
            warnings.append(
                "Chi-squared requires non-negative features; skipped columns: "
                f"{skipped_text}."
            )
        if not non_negative_columns:
            return [], warnings
        scores, pvalues = chi2(
            feature_ready[non_negative_columns].fillna(0.0),
            target_ready,
        )
        for column, score, pvalue in zip(non_negative_columns, scores, pvalues):
            ranking_rows.append(
                {
                    "feature": str(column),
                    "score": float(score) if np.isfinite(score) else None,
                    "p_value": float(pvalue) if np.isfinite(pvalue) else None,
                }
            )
        return sorted(
            ranking_rows,
            key=lambda row: row["score"] if row["score"] is not None else float("-inf"),
            reverse=True,
        ), warnings
    else:
        raise ValueError(f"Unsupported target metric method: {method!r}.")

    for column, score in zip(feature_ready.columns, scores):
        ranking_rows.append(
            {
                "feature": str(column),
                "score": float(score) if np.isfinite(score) else None,
            }
        )
    return sorted(
        ranking_rows,
        key=lambda row: row["score"] if row["score"] is not None else float("-inf"),
        reverse=True,
    ), warnings


def rank_lightgbm_importance(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    *,
    validation_features: pd.DataFrame | None = None,
    validation_target: pd.Series | None = None,
    random_state: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    """Train a small LightGBM model and return feature importances.

    Args:
        train_features (pd.DataFrame): Training feature frame.
        train_target (pd.Series): Training target values.
        validation_features (pd.DataFrame | None): Optional validation frame.
        validation_target (pd.Series | None): Optional validation target values.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[list[dict[str, Any]], dict[str, float], list[str]]:
            Ranked importances, evaluation metrics, and warnings.
    """
    warnings = [
        "Model-based feature importances are diagnostic only; use train-only or train/validation data to avoid optimistic bias."
    ]
    task_type = infer_task_type(train_target)
    class_count = (
        int(pd.Series(train_target).dropna().nunique())
        if task_type == "classification"
        else None
    )
    train_ready, validation_ready = prepare_model_frames(
        train_features,
        validation_features,
    )
    train_target_ready, validation_target_ready = prepare_targets(
        train_target,
        validation_target,
        task_type=task_type,
    )
    estimator = build_lightgbm_estimator(
        task_type=task_type,
        class_count=class_count,
        random_state=random_state,
    )
    estimator.fit(train_ready, train_target_ready)

    importances = list(
        getattr(estimator, "feature_importances_", np.zeros(train_ready.shape[1]))
    )
    ranking_rows = sorted(
        [
            {"feature": str(column), "importance": float(importance)}
            for column, importance in zip(train_ready.columns, importances)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )

    metrics: dict[str, float] = {}
    if validation_ready is not None and validation_target_ready is not None:
        predictions = estimator.predict(validation_ready)
        prediction_probabilities = (
            estimator.predict_proba(validation_ready)
            if hasattr(estimator, "predict_proba")
            else None
        )
        metrics = compute_prediction_metrics(
            validation_target_ready,
            predictions,
            task_type=task_type,
            y_pred_proba=prediction_probabilities,
        )
    return ranking_rows, metrics, warnings


def evaluate_feature_subset(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    *,
    validation_features: pd.DataFrame | None = None,
    validation_target: pd.Series | None = None,
    cv_folds: int = 5,
    random_state: int = 0,
    model_params: dict[str, Any] | None = None,
    scorer: Any | None = None,
    splitter: Any | None = None,
    groups: pd.Series | np.ndarray | Sequence[Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate a user-provided feature subset with LightGBM.

    Args:
        train_features (pd.DataFrame): Training feature frame.
        train_target (pd.Series): Training target values.
        validation_features (pd.DataFrame | None): Optional validation frame.
        validation_target (pd.Series | None): Optional validation target values.
        cv_folds (int): Number of folds when no validation frame is provided.
        random_state (int): Random seed for reproducibility.
        model_params (dict[str, Any] | None): Optional LightGBM parameter overrides.
        scorer (Any | None): Optional stored scorer definition used to compute an
            additional objective-ready metric.
        splitter (Any | None): Optional stored splitter definition for cross-validation.
        groups (pd.Series | np.ndarray | Sequence[Any] | None): Optional group values
            aligned to ``train_features`` when the splitter requires groups.

    Returns:
        tuple[dict[str, Any], list[str]]: Evaluation summary and warnings.
    """
    task_type = infer_task_type(train_target)
    warnings: list[str] = []
    class_count = (
        int(pd.Series(train_target).dropna().nunique())
        if task_type == "classification"
        else None
    )

    if validation_features is not None and validation_target is not None:
        warnings.append(
            "Metrics were computed on the provided validation dataset. Do not use an unseen test set to drive feature selection."
        )
        train_ready, validation_ready = prepare_model_frames(
            train_features,
            validation_features,
        )
        train_target_ready, validation_target_ready = prepare_targets(
            train_target,
            validation_target,
            task_type=task_type,
        )
        estimator = build_lightgbm_estimator(
            task_type=task_type,
            class_count=class_count,
            random_state=random_state,
        )
        if model_params:
            estimator.set_params(**model_params)
        estimator.fit(train_ready, train_target_ready)
        predictions = estimator.predict(validation_ready)
        probabilities = (
            estimator.predict_proba(validation_ready)
            if hasattr(estimator, "predict_proba")
            else None
        )
        metrics = compute_prediction_metrics(
            validation_target_ready,
            predictions,
            task_type=task_type,
            y_pred_proba=probabilities,
        )
        metrics = _merge_scorer_metric(
            metrics,
            scorer=scorer,
            y_true=validation_target_ready,
            y_pred=predictions,
            task_type=task_type,
            y_pred_proba=probabilities,
        )
        return {"mode": "validation", "metrics": metrics}, warnings

    warnings.append(
        "No validation dataset was supplied. Reported metrics are cross-validation estimates on the provided dataframe."
    )
    train_ready, _ = prepare_model_frames(train_features)
    train_target_ready, _ = prepare_targets(train_target, task_type=task_type)
    splitter_instance, group_values = _build_split_iterator(
        task_type=task_type,
        cv_folds=cv_folds,
        random_state=random_state,
        splitter=splitter,
        groups=groups,
    )
    fold_metrics: list[dict[str, float]] = []
    if group_values is not None:
        split_iterator = splitter_instance.split(
            train_ready, train_target_ready, group_values
        )
    else:
        split_iterator = splitter_instance.split(train_ready, train_target_ready)
    for train_idx, valid_idx in split_iterator:
        fold_train = train_ready.iloc[train_idx].copy()
        fold_valid = train_ready.iloc[valid_idx].copy()
        fold_target_train = train_target_ready.iloc[train_idx].copy()
        fold_target_valid = train_target_ready.iloc[valid_idx].copy()

        estimator = build_lightgbm_estimator(
            task_type=task_type,
            class_count=class_count,
            random_state=random_state,
        )
        if model_params:
            estimator.set_params(**model_params)
        estimator.fit(fold_train, fold_target_train)
        predictions = estimator.predict(fold_valid)
        probabilities = (
            estimator.predict_proba(fold_valid)
            if hasattr(estimator, "predict_proba")
            else None
        )
        fold_metrics.append(
            _merge_scorer_metric(
                compute_prediction_metrics(
                    fold_target_valid,
                    predictions,
                    task_type=task_type,
                    y_pred_proba=probabilities,
                ),
                scorer=scorer,
                y_true=fold_target_valid,
                y_pred=predictions,
                task_type=task_type,
                y_pred_proba=probabilities,
            )
        )

    return {
        "mode": "cross_validation",
        "cv_folds": int(cv_folds),
        "summary": summarize_cv_metrics(fold_metrics),
    }, warnings


__all__ = [
    "build_lightgbm_estimator",
    "compute_prediction_metrics",
    "evaluate_feature_subset",
    "infer_task_type",
    "prepare_model_frames",
    "prepare_targets",
    "rank_feature_target_metrics",
    "rank_lightgbm_importance",
    "summarize_cv_metrics",
]
