from dataclasses import asdict
from typing import Any

import numpy as np
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Heading, Markdown
from rich.style import Style
from rich.text import Text
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier, XGBRegressor


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


def make_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON/YAML serializable format.
    
    Args:
        obj: The object to make serializable
        
    Returns:
        A JSON/YAML serializable version of the object
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Handle dataclass objects and other objects with __dict__
        if hasattr(obj, "__dataclass_fields__"):
            # It's a dataclass
            return make_serializable(asdict(obj))
        else:
            # Regular object with __dict__
            return make_serializable(obj.__dict__)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Fallback to string representation for unknown types
        return str(obj)
