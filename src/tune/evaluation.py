import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)

from typing import Callable, Union
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold


def get_cv(
    task_type: str, cv_folds: int, n_repeats: int, stratify: bool, random_state: int
) -> RepeatedStratifiedKFold | RepeatedKFold:
    if task_type == "classification" and stratify:
        return RepeatedStratifiedKFold(
            n_splits=cv_folds,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    else:
        return RepeatedKFold(
            n_splits=cv_folds,
            n_repeats=n_repeats,
            random_state=random_state,
        )


def get_scorer_from_string(metric_name: str, task_type: str) -> Callable:
    """Get sklearn scorer from string name."""
    try:
        return get_scorer(metric_name)
    except ValueError:
        # Handle custom metrics or common aliases
        if task_type == "classification":
            if metric_name in ["accuracy"]:
                return make_scorer(accuracy_score)
            elif metric_name in ["precision"]:
                return make_scorer(precision_score, average="weighted")
            elif metric_name in ["recall"]:
                return make_scorer(recall_score, average="weighted")
            elif metric_name in ["f1"]:
                return make_scorer(f1_score, average="weighted")
            elif metric_name in ["roc_auc", "auc"]:
                return make_scorer(
                    roc_auc_score,
                    needs_proba=True,
                    multi_class="ovr",
                    average="weighted",
                )
        else:  # regression
            if metric_name in ["mse", "mean_squared_error"]:
                return make_scorer(mean_squared_error, greater_is_better=False)
            elif metric_name in ["mae", "mean_absolute_error"]:
                return make_scorer(mean_absolute_error, greater_is_better=False)
            elif metric_name in ["r2", "r2_score"]:
                return make_scorer(r2_score)

        raise ValueError(f"Unsupported metric: {metric_name}")


def get_default_metric(task_type: str) -> str:
    """Get default metric for a task type.

    Args:
        task_type (str): Type of ML task

    Returns:
        str: Default metric name
    """
    if task_type == "classification":
        return "f1"  # More informative than accuracy for most cases
    else:  # regression
        return "r2"


def get_scorer_smart(metric: Union[str, Callable, None], task_type: str) -> Callable:
    """Get scorer function with 'smart' defaults.

    Args:
        metric: User-specified metric (string name, callable, or None)
        task_type: Type of ML task

    Returns:
        Callable: Scoring function
    """
    if callable(metric):
        return metric
    elif isinstance(metric, str):
        return get_scorer_from_string(metric, task_type)
    else:
        # Use better defaults than just accuracy/r2
        default_metric = get_default_metric(task_type)
        return get_scorer_from_string(default_metric, task_type)


def get_comprehensive_metrics(
    y_true, y_pred, y_pred_proba=None, task_type="classification"
) -> dict[str, float]:
    """
    Get comprehensive evaluation metrics based on task type.

    Args:
        y_true: True target values
        y_pred: Predicted values
        y_pred_proba: Predicted probabilities (for classification only)

    Returns:
        dict: Dictionary of metric names and values
    """
    metrics = {}

    if task_type == "classification":
        # Core classification metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

        # Handle multiclass vs binary
        average_method = "binary" if len(np.unique(y_true)) == 2 else "weighted"

        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average_method, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=average_method, zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average=average_method, zero_division=0)
        )

        # Probability-based metrics (if available)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_pred_proba[:, 1])
                    )
                    metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                else:
                    # Multiclass
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
                # Skip if probabilities are not compatible
                pass

    else:  # regression
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

        # Avoid division by zero for MAPE
        if not np.any(y_true == 0):
            metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))

    return metrics
