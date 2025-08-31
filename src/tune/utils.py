from typing import Callable, Union

import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Heading, Markdown
from rich.style import Style
from rich.text import Text
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor


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


def get_model_pipeline(
    model_hyperparameters: dict | None = None,
    model_type: str = "xgboost",
    task_type: str = "classification",
    custom_estimator: BaseEstimator | None = None,
):
    """
    Return a model for classification or regression.

    Args:
        model_hyperparameters (dict, optional): Hyperparameters to set on the model.
        model_type (str): Model framework - "xgboost" or "lightgbm".
        task_type (str): Task type - either "classification" or "regression".
        custom_estimator (BaseEstimator, optional): Custom sklearn estimator.

    Returns:
        BaseEstimator: Configured model (XGB or LightGBM).

    """
    # Handle custom estimator case first
    if custom_estimator is not None:
        if model_hyperparameters is not None:
            return custom_estimator.set_params(**model_hyperparameters)
        return custom_estimator

    # Determine model class based on model_type and task_type
    if model_type.lower() == "lightgbm":
        if task_type == "regression":
            model_class = LGBMRegressor
            default_params = {
                "random_state": 42,
                "objective": "regression",
                "metric": "rmse",
                "n_jobs": -1,
                "verbosity": -1,  # Suppress warnings
                "force_col_wise": True,  # Avoid LightGBM warnings
            }
        else:  # classification
            model_class = LGBMClassifier
            default_params = {
                "random_state": 42,
                "objective": "binary",
                "metric": "binary_logloss",
                "n_jobs": -1,
                "verbosity": -1,  # Suppress warnings
                "force_col_wise": True,  # Avoid LightGBM warnings
            }
    else:  # Default to XGBoost
        if task_type == "regression":
            model_class = XGBRegressor
            default_params = {
                "random_state": 42,
                "objective": "reg:squarederror",
                "n_jobs": -1,
                "verbosity": 0,
            }
        else:  # classification
            model_class = XGBClassifier
            default_params = {
                "random_state": 42,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_jobs": -1,
                "use_label_encoder": False,
                "verbosity": 0,
            }

    # Create model with default parameters
    model = model_class(**default_params)

    # Apply custom hyperparameters if provided
    if model_hyperparameters is not None:
        model.set_params(**model_hyperparameters)

    return model


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


def extract_logs_from_study(study: optuna.Study, top_n: int = 5) -> tuple[str, float]:
    """Summarize the top trials and return (summary, best_value)."""
    trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=study.direction == optuna.study.StudyDirection.MAXIMIZE,
    )[:top_n]

    lines = []
    for i, t in enumerate(trials, start=1):
        param_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        lines.append(f"Trial {i} (value={t.value:.4f}): {param_str}")

    best_value = (
        trials[0].value
        if trials
        else (
            float("-inf")
            if study.direction == optuna.study.StudyDirection.MAXIMIZE
            else float("inf")
        )
    )
    return "\n".join(lines), best_value


class LeftHeading(Heading):
    """Customized headings in markdown to stop centering and prepend markdown style hashes."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # note we use `Style(bold=True)` not `self.style_name` here to disable underlining which is ugly IMHO
        yield Text(
            f"{'#' * int(self.tag[1:])} {self.text.plain}", style=Style(bold=True)
        )


Markdown.elements.update(
    heading_open=LeftHeading,
)
