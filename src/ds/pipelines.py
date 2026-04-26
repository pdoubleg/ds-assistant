"""Sklearn-native feature pipeline helpers for the standalone ds package."""

from __future__ import annotations

from typing import Any, Sequence

import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from .config import PipelineSearchSpace
from .transformers import CarSpecParser, YesNoMapper


# ---------------------------------------------------------------------
# Feature setup
# ---------------------------------------------------------------------

target_column = "claim_status"

yes_no_columns = [
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_parking_camera",
    "is_brake_assist",
]

numeric_columns = [
    "subscription_length",
    "vehicle_age",
    "customer_age",
    "displacement",
    "region_density",
    "width",
    "length",
    "cylinder",
    "max_torque_nm",
    "max_torque_rpm",
    "max_power_bhp",
    "max_power_rpm",
    *yes_no_columns,
]

categorical_columns = [
    "region_code",
    "model",
    "engine_type",
    "steering_type",
    "segment",
    "transmission_type",
]


def _identity_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe copy for identity-style pipelines.

    Args:
        dataframe: Input dataframe.

    Returns:
        Dataframe copy.
    """

    return dataframe.copy()


def build_feature_pipeline(
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    numeric_transformers: list[tuple[str, Any]] | None = None,
    categorical_transformers: list[tuple[str, Any]] | None = None,
    remainder: str = "drop",
) -> Pipeline:
    """Build a dataframe-preserving sklearn feature pipeline.

    Args:
        numeric_features: Optional numeric columns. When omitted, the package's
            car-insurance numeric feature manifest is used.
        categorical_features: Optional categorical columns. When omitted, the
            package's car-insurance categorical feature manifest is used.
        numeric_transformers: Optional transformers for numeric columns.
        categorical_transformers: Optional transformers for categorical columns.
        remainder: ColumnTransformer remainder strategy.

    Returns:
        Pipeline that emits pandas dataframes.

    Example:
        >>> pipeline = build_feature_pipeline(
        ...     numeric_features=["age"],
        ...     categorical_features=["segment"],
        ... )
    """

    resolved_numeric_features = list(numeric_features or numeric_columns)
    resolved_categorical_features = list(categorical_features or categorical_columns)
    numeric_steps = numeric_transformers or [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    categorical_steps = categorical_transformers or [
        ("impute", SimpleImputer(strategy="most_frequent")),
    ]
    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=numeric_steps,
                ),
                resolved_numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=categorical_steps,
                ),
                resolved_categorical_features,
            ),
        ],
        remainder=remainder,
        verbose_feature_names_out=False,
    )

    steps: list[tuple[str, Any]]
    if numeric_features is None and categorical_features is None:
        steps = [
            ("parse_specs", CarSpecParser(drop_original=True)),
            ("map_yes_no", YesNoMapper(columns=yes_no_columns)),
            ("preprocess", preprocess),
            ("variance", VarianceThreshold(threshold=0.0)),
        ]
    else:
        steps = [("preprocess", preprocess)]

    feature_pipeline = Pipeline(steps=steps)
    feature_pipeline.set_output(transform="pandas")

    return feature_pipeline


def list_pipeline_params(
    pipeline: Pipeline,
    *,
    contains: Sequence[str] | None = None,
    exclude_estimator_objects: bool = True,
) -> pd.DataFrame:
    """List pipeline params for search-space authoring.

    Args:
        pipeline: Sklearn pipeline.
        contains: Optional substrings. If provided, only params containing at
            least one substring are returned.
        exclude_estimator_objects: Whether to suppress params whose values are
            estimator-like objects.

    Returns:
        Dataframe with parameter names, value types, and current values.
    """

    params = pipeline.get_params(deep=True)
    contains_values = [value.lower() for value in contains or []]

    rows: list[dict[str, Any]] = []

    for name, value in sorted(params.items()):
        lower_name = name.lower()
        if contains_values and not any(
            token in lower_name for token in contains_values
        ):
            continue

        if exclude_estimator_objects and hasattr(value, "get_params"):
            continue

        rows.append(
            {
                "param": name,
                "value_type": type(value).__name__,
                "value": value,
            }
        )

    return pd.DataFrame(rows)


def validate_pipeline_search_space(
    pipeline: Pipeline,
    search_space: list[PipelineSearchSpace],
) -> None:
    available_params = pipeline.get_params(deep=True)
    invalid_params = [
        entry.estimator_param
        for entry in search_space
        if entry.estimator_param not in available_params
    ]

    if invalid_params:
        invalid_preview = "\n".join(f"  - {param}" for param in invalid_params)
        raise ValueError(
            "Invalid pipeline search-space params:\n"
            f"{invalid_preview}\n\n"
            "Call `pipeline.get_params(deep=True).keys()` to inspect valid paths."
        )


def get_pipeline_search_space(
    search_space: list[PipelineSearchSpace] | None = None,
) -> list[PipelineSearchSpace]:
    """Return the normalized pipeline search space manifest.

    Args:
        search_space: Optional caller-provided search space definition.

    Returns:
        Pipeline search-space entries.
    """

    return list(search_space or [])


def suggest_pipeline_params(
    trial: optuna.Trial,
    search_space: list[PipelineSearchSpace] | None = None,
) -> dict[str, Any]:
    """Suggest sklearn pipeline parameters for an Optuna trial.

    Args:
        trial: Active Optuna trial.
        search_space: Parameter manifest returned by ``get_pipeline_search_space``.

    Returns:
        Flattened sklearn ``set_params`` mapping.
    """

    suggested_params: dict[str, Any] = {}
    for entry in get_pipeline_search_space(search_space):
        if entry.suggestion_kind == "categorical":
            if not entry.choices:
                raise ValueError(
                    f"{entry.estimator_param!r} requires categorical choices."
                )
            suggested_params[entry.estimator_param] = trial.suggest_categorical(
                entry.estimator_param,
                entry.choices,
            )
            continue

        if entry.suggestion_kind == "bool":
            suggested_params[entry.estimator_param] = trial.suggest_categorical(
                entry.estimator_param,
                entry.choices or [False, True],
            )
            continue

        if entry.low is None or entry.high is None:
            raise ValueError(
                f"{entry.estimator_param!r} requires `low` and `high` bounds."
            )

        if entry.suggestion_kind == "int":
            suggested_params[entry.estimator_param] = trial.suggest_int(
                entry.estimator_param,
                int(entry.low),
                int(entry.high),
                step=int(entry.step) if entry.step is not None else 1,
                log=entry.log,
            )
            continue

        if entry.suggestion_kind == "float":
            suggested_params[entry.estimator_param] = trial.suggest_float(
                entry.estimator_param,
                float(entry.low),
                float(entry.high),
                step=float(entry.step) if entry.step is not None else None,
                log=entry.log,
            )
            continue

        raise ValueError(
            f"Unsupported pipeline suggestion kind: {entry.suggestion_kind!r}."
        )

    return suggested_params


def clone_pipeline_with_params(
    pipeline: Pipeline,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Clone a sklearn pipeline and apply a parameter mapping.

    Args:
        pipeline: Base sklearn pipeline.
        params: Optional flattened ``set_params`` mapping.

    Returns:
        Cloned pipeline ready for fitting.
    """

    cloned = clone(pipeline)
    if hasattr(cloned, "set_output"):
        cloned.set_output(transform="pandas")
    if params:
        cloned.set_params(**params)
    return cloned


def _ensure_dataframe(value: Any) -> pd.DataFrame:
    """Normalize sklearn pipeline output to a dataframe.

    Args:
        value: Pipeline output.

    Returns:
        Dataframe representation of the pipeline output.
    """

    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value).copy()


def fit_transform_features(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    *,
    valid_df: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Pipeline]:
    """Fit a pipeline and transform train and validation features.

    Args:
        pipeline: Sklearn pipeline to fit.
        train_df: Training dataframe.
        valid_df: Optional validation dataframe.
        feature_columns: Optional explicit feature subset.

    Returns:
        Tuple of transformed train features, transformed validation features, and
        the fitted pipeline.
    """

    feature_columns = list(feature_columns or train_df.columns.tolist())
    fitted_pipeline = clone_pipeline_with_params(pipeline)
    transformed_train = _ensure_dataframe(
        fitted_pipeline.fit_transform(train_df[feature_columns])
    )
    transformed_train.index = train_df.index

    transformed_valid: pd.DataFrame | None = None
    if valid_df is not None:
        transformed_valid = _ensure_dataframe(
            fitted_pipeline.transform(valid_df[feature_columns])
        )
        transformed_valid.index = valid_df.index
    return transformed_train, transformed_valid, fitted_pipeline


__all__ = [
    "build_feature_pipeline",
    "clone_pipeline_with_params",
    "fit_transform_features",
    "get_pipeline_search_space",
    "suggest_pipeline_params",
    "list_pipeline_params",
    "validate_pipeline_search_space",
]
