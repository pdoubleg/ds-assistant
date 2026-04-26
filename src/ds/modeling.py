"""LightGBM modeling helpers for the standalone ds package."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LightGBMTrainingResult, SplitConfig, TrainConfig, TrainValidSplit
from .metrics import lift_at_top_p, ppv_at_top_p, recall_at_top_p


def _safe_series_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to datetimes while coercing invalid values.

    Args:
        series: Series to convert.

    Returns:
        Datetime-like series with invalid values set to ``NaT``.
    """

    return pd.to_datetime(series, errors="coerce", utc=False)


def _is_numeric_dtype(series: pd.Series) -> bool:
    """Return whether a series is numeric.

    Args:
        series: Series to inspect.

    Returns:
        Whether the series has a numeric dtype.
    """

    return bool(pd.api.types.is_numeric_dtype(series))


def _is_datetime_dtype(series: pd.Series) -> bool:
    """Return whether a series is datetime-like.

    Args:
        series: Series to inspect.

    Returns:
        Whether the series has a datetime-like dtype.
    """

    return bool(pd.api.types.is_datetime64_any_dtype(series))


def infer_categorical_columns(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """Infer which feature columns should use native LightGBM categoricals.

    Args:
        dataframe: Source dataframe.
        feature_columns: Candidate feature columns.

    Returns:
        Feature columns treated as categorical.
    """

    return [
        column for column in feature_columns if not _is_numeric_dtype(dataframe[column])
    ]


def resolve_feature_columns(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
) -> list[str]:
    """Resolve modeling feature columns in dataframe order.

    Args:
        dataframe: Source dataframe.
        target_column: Optional target column excluded from the returned feature list.
        feature_columns: Optional explicit feature subset.
        id_columns: Optional identifier columns excluded from automatic selection.

    Returns:
        Final feature column list.
    """

    id_columns = list(id_columns or [])
    if feature_columns is not None:
        missing_columns = [
            column for column in feature_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")
        return list(feature_columns)

    excluded_columns = [*id_columns]
    if target_column is not None:
        excluded_columns.append(target_column)

    return [
        str(column)
        for column in dataframe.columns
        if str(column) not in excluded_columns
    ]


def _validate_binary_target(target: pd.Series, *, target_column: str) -> pd.Series:
    """Validate and normalize a binary target series.

    Args:
        target: Raw target series.
        target_column: Target column name for error messages.

    Returns:
        Integer-encoded binary target series.
    """

    normalized = pd.to_numeric(target, errors="coerce")
    if normalized.isna().any():
        raise ValueError(
            f"Target column {target_column!r} contains non-numeric values."
        )
    unique_values = set(normalized.astype(int).unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Target column {target_column!r} must be binary and encoded as 0/1."
        )
    return normalized.astype(int)


def prepare_lightgbm_frame(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    categorical_columns: list[str] | None = None,
    feature_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare a dataframe for LightGBM with native categorical handling.

    Args:
        dataframe: Source dataframe.
        target_column: Target column name.
        categorical_columns: Preferred categorical columns.
        feature_columns: Optional explicit feature subset.
        drop_columns: Optional columns removed before fitting.

    Returns:
        Tuple of prepared features, normalized target, and resolved categorical
        columns.
    """

    categorical_columns = list(categorical_columns or [])
    drop_columns = list(drop_columns or [])
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    if feature_columns is None:
        feature_columns = [
            str(column)
            for column in dataframe.columns
            if str(column) not in [target_column, *drop_columns]
        ]
    else:
        missing_columns = [
            column for column in feature_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    target = _validate_binary_target(
        dataframe[target_column], target_column=target_column
    )
    features = dataframe[feature_columns].copy()

    final_categorical_columns: list[str] = []
    for column in list(features.columns):
        series = features[column]
        if _is_datetime_dtype(series):
            datetime_values = _safe_series_to_datetime(series)
            features[column] = (
                datetime_values.astype("int64", copy=False) / 10**9
            ).replace(
                -9223372036854775808 / 10**9,
                np.nan,
            )
            continue

        if column in categorical_columns:
            features[column] = series.astype("category")
            final_categorical_columns.append(column)
            continue

        if _is_numeric_dtype(series):
            continue

        # Keep all remaining non-numeric columns as native LightGBM categoricals.
        features[column] = series.astype("category")
        final_categorical_columns.append(column)

    return features, target, final_categorical_columns


def prepare_lightgbm_features(
    dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a feature-only dataframe for LightGBM scoring.

    Args:
        dataframe: Source dataframe containing the feature columns.
        feature_columns: Exact feature columns expected by the fitted model.
        categorical_columns: Preferred categorical columns.

    Returns:
        Tuple of prepared features and resolved categorical columns.
    """

    categorical_columns = list(categorical_columns or [])
    missing_columns = [
        column for column in feature_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    features = dataframe[feature_columns].copy()
    final_categorical_columns: list[str] = []
    for column in list(features.columns):
        series = features[column]
        if _is_datetime_dtype(series):
            datetime_values = _safe_series_to_datetime(series)
            features[column] = (
                datetime_values.astype("int64", copy=False) / 10**9
            ).replace(
                -9223372036854775808 / 10**9,
                np.nan,
            )
            continue

        if column in categorical_columns:
            features[column] = series.astype("category")
            final_categorical_columns.append(column)
            continue

        if _is_numeric_dtype(series):
            continue

        features[column] = series.astype("category")
        final_categorical_columns.append(column)

    return features, final_categorical_columns


def make_train_valid_split(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    config: SplitConfig,
) -> TrainValidSplit:
    """Create a train and validation split with optional stratification.

    Args:
        dataframe: Source dataframe to split.
        target_column: Binary target column used for stratification.
        config: Split strategy configuration.

    Returns:
        Explicit train/validation split payload.
    """

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    stratify_values: pd.Series | None = None
    if config.stratify:
        normalized_target = _validate_binary_target(
            dataframe[target_column],
            target_column=target_column,
        )
        class_counts = normalized_target.value_counts()
        # Sklearn requires at least two rows per class to stratify.
        if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
            stratify_values = normalized_target

    train_df, valid_df = train_test_split(
        dataframe,
        test_size=config.valid_frac,
        random_state=config.random_seed,
        shuffle=True,
        stratify=stratify_values,
    )
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    return TrainValidSplit(
        train_df=train_df,
        valid_df=valid_df,
        train_indices=train_df.index.tolist(),
        valid_indices=valid_df.index.tolist(),
        split_config=config,
    )


def build_train_valid_frames(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    validation_df: pd.DataFrame | None = None,
    split_config: SplitConfig | None = None,
) -> TrainValidSplit:
    """Build train and validation frames for modeling.

    Args:
        dataframe: Source modeling dataframe.
        target_column: Target column name.
        feature_columns: Feature columns used for modeling.
        validation_df: Optional explicit validation dataframe.
        split_config: Split configuration used when ``validation_df`` is omitted.

    Returns:
        Explicit train/validation split payload.
    """

    required_columns = [target_column, *feature_columns]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    train_base = dataframe[required_columns].copy()
    if validation_df is not None:
        missing_validation_columns = [
            column for column in required_columns if column not in validation_df.columns
        ]
        if missing_validation_columns:
            raise ValueError(
                "Validation dataframe is missing required columns: "
                f"{', '.join(missing_validation_columns)}."
            )
        valid_base = validation_df[required_columns].copy()
        default_split = split_config or SplitConfig()
        return TrainValidSplit(
            train_df=train_base,
            valid_df=valid_base,
            train_indices=train_base.index.tolist(),
            valid_indices=valid_base.index.tolist(),
            split_config=default_split,
        )

    return make_train_valid_split(
        train_base,
        target_column=target_column,
        config=split_config or SplitConfig(),
    )


def _align_validation_categories(
    train_features: pd.DataFrame,
    valid_features: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Align validation categorical vocabularies to the training frame.

    Args:
        train_features: Training feature frame.
        valid_features: Validation feature frame.
        categorical_columns: Columns treated as categorical.

    Returns:
        Validation frame with aligned categorical vocabularies.
    """

    aligned = valid_features.reindex(columns=train_features.columns)
    for column in categorical_columns:
        if column in aligned.columns:
            aligned[column] = pd.Categorical(
                aligned[column],
                categories=train_features[column].cat.categories,
            )
    return aligned


def prepare_lightgbm_train_valid_frames(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Prepare paired train and validation frames for native LightGBM categoricals.

    Args:
        train_df: Training dataframe containing the target column.
        valid_df: Validation dataframe containing the target column.
        target_column: Binary target column name.
        feature_columns: Optional explicit feature subset.
        categorical_columns: Preferred categorical columns.

    Returns:
        Tuple of prepared training features, training target, prepared validation
        features, validation target, and resolved categorical columns.
    """

    resolved_feature_columns = list(feature_columns or [])
    if not resolved_feature_columns:
        resolved_feature_columns = [
            str(column) for column in train_df.columns if str(column) != target_column
        ]

    X_train, y_train, final_cats = prepare_lightgbm_frame(
        train_df,
        target_column=target_column,
        categorical_columns=categorical_columns,
        feature_columns=resolved_feature_columns,
    )
    X_valid, y_valid, _ = prepare_lightgbm_frame(
        valid_df,
        target_column=target_column,
        categorical_columns=final_cats,
        feature_columns=resolved_feature_columns,
    )
    # Validation must reuse the train-side category vocabulary so LightGBM sees
    # a stable categorical encoding across both datasets.
    X_valid = _align_validation_categories(X_train, X_valid, final_cats)
    return X_train, y_train, X_valid, y_valid, final_cats


def _fit_lightgbm_prepared(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    target_column: str,
    categorical_columns: list[str],
    params: dict[str, Any] | None = None,
    train_config: TrainConfig | None = None,
    top_p: float = 0.05,
) -> LightGBMTrainingResult:
    """Fit LightGBM from already prepared train and validation matrices.

    Args:
        X_train: Prepared training feature matrix.
        y_train: Normalized binary training target.
        X_valid: Prepared validation feature matrix.
        y_valid: Normalized binary validation target.
        target_column: Target column name retained in the result payload.
        categorical_columns: Feature columns treated as categorical.
        params: Additional LightGBM parameters.
        train_config: Training configuration override.
        top_p: Fraction retained for PPV-style validation metrics.

    Returns:
        Compact fitted-model result payload.
    """

    train_config = train_config or TrainConfig()
    params = dict(params or {})

    default_params = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 200,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": train_config.seed,
        "num_threads": train_config.num_threads,
    }
    default_params.update(params)
    if default_params.get("boosting_type") == "goss":
        # GOSS uses its own sampling strategy and cannot be combined with bagging.
        default_params.pop("bagging_fraction", None)
        default_params.pop("bagging_freq", None)
    else:
        default_params.pop("top_rate", None)
        default_params.pop("other_rate", None)

    # Build Dataset handles only after params are finalized because some LightGBM
    # settings are bound when the Dataset object is created.
    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_columns,
        params=default_params,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical_columns,
        params=default_params,
        reference=dtrain,
        free_raw_data=False,
    )
    booster = lgb.train(
        params=default_params,
        train_set=dtrain,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=make_lgb_ppv_eval(p=top_p),
        num_boost_round=train_config.num_boost_round,
        callbacks=[
            lgb.early_stopping(train_config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    valid_pred = booster.predict(X_valid, num_iteration=booster.best_iteration)
    importance_gain = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=X_train.columns,
    ).sort_values(ascending=False)
    evaluation_summary = {
        "ppv_at_p": float(ppv_at_top_p(y_valid.values, valid_pred, p=top_p)),
        "recall_at_p": float(recall_at_top_p(y_valid.values, valid_pred, p=top_p)),
        "lift_at_p": float(lift_at_top_p(y_valid.values, valid_pred, p=top_p)),
        "base_rate": float(np.mean(y_valid)),
        "top_p": float(top_p),
    }
    return LightGBMTrainingResult(
        booster=booster,
        target_column=target_column,
        feature_columns=list(X_train.columns),
        categorical_columns=list(categorical_columns),
        best_params=default_params,
        best_iteration=int(booster.best_iteration),
        evaluation_summary=evaluation_summary,
        feature_importance_gain={
            str(feature): float(gain)
            for feature, gain in importance_gain.to_dict().items()
        },
        top_p=float(top_p),
    )


def make_lgb_ppv_eval(*, p: float = 0.05) -> Any:
    """Build a LightGBM evaluation callback for PPV at top-p.

    Args:
        p: Fraction of rows retained when computing PPV.

    Returns:
        LightGBM-compatible evaluation callback.
    """

    def _feval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
        y_true = dataset.get_label()
        score = ppv_at_top_p(y_true=y_true, y_score=preds, p=p)
        return f"ppv_at_{int(p * 100)}", score, True

    return _feval


def fit_lightgbm_binary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    params: dict[str, Any] | None = None,
    train_config: TrainConfig | None = None,
    top_p: float = 0.05,
) -> LightGBMTrainingResult:
    """Fit one LightGBM binary classifier and evaluate it on validation data.

    Args:
        train_df: Training dataframe.
        valid_df: Validation dataframe.
        target_column: Target column name.
        feature_columns: Optional explicit feature subset.
        categorical_columns: Preferred categorical columns.
        params: Additional LightGBM parameters.
        train_config: Training configuration override.
        top_p: Fraction retained for PPV-style validation metrics.

    Returns:
        Compact fitted-model result payload.
    """

    X_train, y_train, X_valid, y_valid, final_cats = (
        prepare_lightgbm_train_valid_frames(
            train_df,
            valid_df,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
        )
    )
    return _fit_lightgbm_prepared(
        X_train,
        y_train,
        X_valid,
        y_valid,
        target_column=target_column,
        categorical_columns=final_cats,
        params=params,
        train_config=train_config,
        top_p=top_p,
    )


def score_lightgbm_dataframe(
    dataframe: pd.DataFrame,
    *,
    booster: lgb.Booster,
    feature_columns: list[str],
    categorical_columns: list[str],
    best_iteration: int,
) -> np.ndarray:
    """Score a dataframe with a fitted LightGBM booster.

    Args:
        dataframe: Source dataframe to score.
        booster: Fitted LightGBM booster.
        feature_columns: Exact feature columns expected by the booster.
        categorical_columns: Columns treated as categorical at fit time.
        best_iteration: Best iteration retained by the fitted booster.

    Returns:
        Prediction scores aligned with the dataframe rows.
    """

    features, final_cats = prepare_lightgbm_features(
        dataframe,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    aligned_features = features.reindex(columns=feature_columns)
    for column in final_cats:
        if column in aligned_features.columns:
            aligned_features[column] = aligned_features[
                column
            ].cat.remove_unused_categories()
    return booster.predict(aligned_features, num_iteration=best_iteration)


def score_dataframe(
    dataframe: pd.DataFrame,
    model_result: LightGBMTrainingResult,
    *,
    score_column_name: str = "pred_score",
) -> pd.DataFrame:
    """Append LightGBM scores to a dataframe.

    Args:
        dataframe: Dataframe to score.
        model_result: Fitted model result.
        score_column_name: Name of the added numeric score column.

    Returns:
        Scored dataframe copy.
    """

    if score_column_name in dataframe.columns:
        raise ValueError(
            f"Score column {score_column_name!r} already exists on the dataframe."
        )
    scores = score_lightgbm_dataframe(
        dataframe,
        booster=model_result.booster,
        feature_columns=model_result.feature_columns,
        categorical_columns=model_result.categorical_columns,
        best_iteration=model_result.best_iteration,
    )
    scored = dataframe.copy()
    scored[score_column_name] = scores
    return scored


__all__ = [
    "_is_datetime_dtype",
    "_is_numeric_dtype",
    "build_train_valid_frames",
    "fit_lightgbm_binary",
    "infer_categorical_columns",
    "make_lgb_ppv_eval",
    "make_train_valid_split",
    "prepare_lightgbm_features",
    "prepare_lightgbm_frame",
    "prepare_lightgbm_train_valid_frames",
    "resolve_feature_columns",
    "score_dataframe",
    "score_lightgbm_dataframe",
]
