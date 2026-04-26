"""Optuna orchestration helpers for the standalone ds package."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd

from .config import (
    HpoResult,
    OptunaConfig,
    PipelineSearchSpace,
    SplitConfig,
    TrainConfig,
)
from .modeling import (
    _fit_lightgbm_prepared,
    build_train_valid_frames,
    prepare_lightgbm_train_valid_frames,
    resolve_feature_columns,
)
from .pipelines import (
    build_feature_pipeline,
    clone_pipeline_with_params,
    fit_transform_features,
    suggest_pipeline_params,
)


def suggest_lgbm_params(
    trial: optuna.Trial,
    *,
    num_threads: int,
    seed: int,
    positive_class_ratio: float | None = None,
) -> dict[str, Any]:
    """Suggest a LightGBM parameter set for Optuna.

    Args:
        trial: Active Optuna trial.
        num_threads: Number of LightGBM worker threads.
        seed: Random seed propagated to LightGBM.
        positive_class_ratio: Optional positive-class prevalence used to anchor
            the ``scale_pos_weight`` search range.

    Returns:
        Candidate LightGBM parameter set.
    """

    boosting_type = trial.suggest_categorical("lgbm__boosting_type", ["gbdt", "goss"])
    max_depth = trial.suggest_categorical(
        "lgbm__max_depth", [-1, 4, 5, 6, 7, 8, 10, 12]
    )
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": boosting_type,
        "verbosity": -1,
        "seed": seed,
        "num_threads": num_threads,
        "learning_rate": trial.suggest_float(
            "lgbm__learning_rate", 0.03, 0.15, log=True
        ),
        "num_leaves": trial.suggest_int("lgbm__num_leaves", 16, 255, log=True),
        "min_child_samples": trial.suggest_int(
            "lgbm__min_child_samples",
            50,
            1_000,
            log=True,
        ),
        "min_child_weight": trial.suggest_float(
            "lgbm__min_child_weight",
            1e-4,
            10.0,
            log=True,
        ),
        "feature_fraction": trial.suggest_float("lgbm__feature_fraction", 0.4, 0.95),
        "feature_fraction_bynode": trial.suggest_float(
            "lgbm__feature_fraction_bynode",
            0.4,
            1.0,
        ),
        "lambda_l1": trial.suggest_float("lgbm__lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lgbm__lambda_l2", 1e-4, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("lgbm__min_gain_to_split", 0.0, 2.0),
        "max_depth": max_depth,
        "extra_trees": trial.suggest_categorical("lgbm__extra_trees", [False, True]),
        "linear_tree": trial.suggest_categorical("lgbm__linear_tree", [False, True]),
        "force_row_wise": trial.suggest_categorical(
            "lgbm__force_row_wise", [False, True]
        ),
        "cat_smooth": trial.suggest_float("lgbm__cat_smooth", 0.0, 100.0),
        "cat_l2": trial.suggest_float("lgbm__cat_l2", 0.0, 100.0),
        "max_cat_to_onehot": trial.suggest_int("lgbm__max_cat_to_onehot", 1, 64),
        "min_data_per_group": trial.suggest_int(
            "lgbm__min_data_per_group",
            10,
            500,
            log=True,
        ),
        "max_cat_threshold": trial.suggest_int(
            "lgbm__max_cat_threshold",
            8,
            256,
            log=True,
        ),
    }
    if boosting_type == "gbdt":
        params["bagging_fraction"] = trial.suggest_float(
            "lgbm__bagging_fraction", 0.5, 0.95
        )
        params["bagging_freq"] = trial.suggest_int("lgbm__bagging_freq", 1, 7)
    if boosting_type == "goss":
        top_rate = trial.suggest_float("lgbm__top_rate", 0.1, 0.5)
        max_other_rate = min(0.4, 0.99 - top_rate)
        params["top_rate"] = top_rate
        params["other_rate"] = trial.suggest_float(
            "lgbm__other_rate", 0.05, max_other_rate
        )

    class_weight_mode = trial.suggest_categorical(
        "lgbm__class_weight_mode",
        ["none", "scale_pos_weight"],
    )
    if class_weight_mode == "scale_pos_weight" and positive_class_ratio is not None:
        positive_class_ratio = float(positive_class_ratio)
        if 0.0 < positive_class_ratio < 1.0:
            baseline_ratio = (1.0 - positive_class_ratio) / positive_class_ratio
            low = max(1e-3, baseline_ratio * 0.5)
            high = max(low * 1.01, baseline_ratio * 2.0)
            params["scale_pos_weight"] = trial.suggest_float(
                "lgbm__scale_pos_weight",
                low,
                high,
                log=True,
            )
    return params


def _combine_model_frame(
    features: pd.DataFrame, target: pd.Series, *, target_column: str
) -> pd.DataFrame:
    """Combine transformed features and target into one modeling frame.

    Args:
        features: Feature dataframe.
        target: Target series.
        target_column: Target column name to append.

    Returns:
        Modeling dataframe containing the target column.
    """

    frame = features.copy()
    frame[target_column] = target.values
    return frame


def tune_lightgbm_pipeline(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    pipeline: Any | None = None,
    pipeline_search_space: list[PipelineSearchSpace] | None = None,
    feature_columns: list[str] | None = None,
    validation_df: pd.DataFrame | None = None,
    id_columns: list[str] | None = None,
    split_config: SplitConfig | None = None,
    train_config: TrainConfig | None = None,
    optuna_config: OptunaConfig | None = None,
    top_p: float = 0.05,
) -> HpoResult:
    """Tune sklearn pipeline and LightGBM parameters against PPV@top-p.

    Args:
        dataframe: Source modeling dataframe.
        target_column: Binary target column.
        pipeline: Base sklearn pipeline. When omitted, an identity pipeline is used.
        pipeline_search_space: Tunable sklearn pipeline parameter manifest.
        feature_columns: Optional explicit feature subset.
        validation_df: Optional held-out validation dataframe.
        id_columns: Optional identifier columns excluded from automatic feature
            resolution.
        split_config: Split configuration used when ``validation_df`` is omitted.
        train_config: Training configuration override.
        optuna_config: Optuna tuning configuration override.
        top_p: Fraction retained for PPV-style validation metrics.

    Returns:
        Structured HPO result containing the best pipeline and model fit.
    """

    train_config = train_config or TrainConfig()
    optuna_config = optuna_config or OptunaConfig()
    feature_columns = resolve_feature_columns(
        dataframe,
        target_column=target_column,
        feature_columns=feature_columns,
        id_columns=id_columns,
    )
    split = build_train_valid_frames(
        dataframe,
        target_column=target_column,
        feature_columns=feature_columns,
        validation_df=validation_df,
        split_config=split_config,
    )
    base_pipeline = pipeline or build_feature_pipeline()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=optuna_config.random_seed),
    )

    def objective(trial: optuna.Trial) -> float:
        pipeline_params = suggest_pipeline_params(trial, pipeline_search_space)
        lgbm_params = suggest_lgbm_params(
            trial,
            num_threads=train_config.num_threads,
            seed=train_config.seed,
            positive_class_ratio=float(split.train_df[target_column].mean()),
        )
        trial_pipeline = clone_pipeline_with_params(base_pipeline, pipeline_params)
        transformed_train, transformed_valid, _ = fit_transform_features(
            trial_pipeline,
            split.train_df,
            valid_df=split.valid_df,
            feature_columns=feature_columns,
        )
        if transformed_valid is None:
            raise ValueError("Validation data is required for HPO scoring.")

        model_train_df = _combine_model_frame(
            transformed_train,
            split.train_df[target_column],
            target_column=target_column,
        )
        model_valid_df = _combine_model_frame(
            transformed_valid,
            split.valid_df[target_column],
            target_column=target_column,
        )
        feature_names = [str(column) for column in transformed_train.columns]
        prepared = prepare_lightgbm_train_valid_frames(
            model_train_df,
            model_valid_df,
            target_column=target_column,
            feature_columns=feature_names,
        )
        X_train, y_train, X_valid, y_valid, categorical_columns = prepared
        result = _fit_lightgbm_prepared(
            X_train,
            y_train,
            X_valid,
            y_valid,
            target_column=target_column,
            categorical_columns=categorical_columns,
            params=lgbm_params,
            train_config=train_config,
            top_p=top_p,
        )
        trial.set_user_attr("ppv_at_p", float(result.evaluation_summary["ppv_at_p"]))
        trial.set_user_attr(
            "recall_at_p", float(result.evaluation_summary["recall_at_p"])
        )
        trial.set_user_attr("lift_at_p", float(result.evaluation_summary["lift_at_p"]))
        return float(result.evaluation_summary["ppv_at_p"])

    study.optimize(
        objective,
        n_trials=optuna_config.n_trials,
        timeout=optuna_config.timeout,
    )

    best_pipeline_params = {
        key: value
        for key, value in study.best_trial.params.items()
        if not key.startswith("lgbm__")
    }
    best_lightgbm_params = {
        key.replace("lgbm__", "", 1): value
        for key, value in study.best_trial.params.items()
        if key.startswith("lgbm__")
    }
    transformed_train, transformed_valid, best_pipeline = fit_transform_features(
        clone_pipeline_with_params(base_pipeline, best_pipeline_params),
        split.train_df,
        valid_df=split.valid_df,
        feature_columns=feature_columns,
    )
    if transformed_valid is None:
        raise ValueError("Validation data is required for HPO scoring.")

    best_feature_names = [str(column) for column in transformed_train.columns]
    best_prepared = prepare_lightgbm_train_valid_frames(
        _combine_model_frame(
            transformed_train,
            split.train_df[target_column],
            target_column=target_column,
        ),
        _combine_model_frame(
            transformed_valid,
            split.valid_df[target_column],
            target_column=target_column,
        ),
        target_column=target_column,
        feature_columns=best_feature_names,
    )
    best_X_train, best_y_train, best_X_valid, best_y_valid, best_categorical_columns = (
        best_prepared
    )
    best_result = _fit_lightgbm_prepared(
        best_X_train,
        best_y_train,
        best_X_valid,
        best_y_valid,
        target_column=target_column,
        categorical_columns=best_categorical_columns,
        params=best_lightgbm_params,
        train_config=train_config,
        top_p=top_p,
    )
    trial_rows = [
        {
            "number": int(trial.number),
            "value": float(trial.value) if trial.value is not None else None,
            "params": dict(trial.params),
            "ppv_at_p": trial.user_attrs.get("ppv_at_p"),
            "recall_at_p": trial.user_attrs.get("recall_at_p"),
            "lift_at_p": trial.user_attrs.get("lift_at_p"),
        }
        for trial in study.trials
    ]
    return HpoResult(
        study=study,
        best_params=dict(study.best_trial.params),
        best_value=float(study.best_value),
        best_pipeline_params=best_pipeline_params,
        best_lightgbm_params=best_lightgbm_params,
        trial_rows=trial_rows,
        split_config=split.split_config,
        split_indices={
            "train": list(split.train_indices),
            "valid": list(split.valid_indices),
        },
        best_pipeline=best_pipeline,
        best_result=best_result,
    )


def refit_best_lightgbm_pipeline(
    dataframe: pd.DataFrame,
    hpo_result: HpoResult,
    *,
    target_column: str,
    pipeline: Any | None = None,
    feature_columns: list[str] | None = None,
    validation_df: pd.DataFrame | None = None,
    id_columns: list[str] | None = None,
    train_config: TrainConfig | None = None,
    top_p: float | None = None,
) -> tuple[Any, Any]:
    """Refit the best pipeline and LightGBM parameters from an HPO run.

    Args:
        dataframe: Source modeling dataframe.
        hpo_result: Prior HPO result containing the best parameter mapping.
        target_column: Binary target column.
        pipeline: Base sklearn pipeline used during tuning.
        feature_columns: Optional explicit feature subset.
        validation_df: Optional held-out validation dataframe.
        id_columns: Optional identifier columns excluded from automatic feature
            resolution.
        train_config: Training configuration override.
        top_p: Optional override for the PPV cutoff.

    Returns:
        Tuple of fitted pipeline and fitted LightGBM result.
    """

    train_config = train_config or TrainConfig()
    feature_columns = resolve_feature_columns(
        dataframe,
        target_column=target_column,
        feature_columns=feature_columns,
        id_columns=id_columns,
    )
    split_config = hpo_result.split_config or SplitConfig()
    split = build_train_valid_frames(
        dataframe,
        target_column=target_column,
        feature_columns=feature_columns,
        validation_df=validation_df,
        split_config=split_config,
    )
    base_pipeline = pipeline or build_feature_pipeline()
    transformed_train, transformed_valid, fitted_pipeline = fit_transform_features(
        clone_pipeline_with_params(base_pipeline, hpo_result.best_pipeline_params),
        split.train_df,
        valid_df=split.valid_df,
        feature_columns=feature_columns,
    )
    if transformed_valid is None:
        raise ValueError("Validation data is required for refitting.")

    fitted_feature_names = [str(column) for column in transformed_train.columns]
    fitted_prepared = prepare_lightgbm_train_valid_frames(
        _combine_model_frame(
            transformed_train,
            split.train_df[target_column],
            target_column=target_column,
        ),
        _combine_model_frame(
            transformed_valid,
            split.valid_df[target_column],
            target_column=target_column,
        ),
        target_column=target_column,
        feature_columns=fitted_feature_names,
    )
    (
        fitted_X_train,
        fitted_y_train,
        fitted_X_valid,
        fitted_y_valid,
        fitted_categorical_columns,
    ) = fitted_prepared
    fitted_result = _fit_lightgbm_prepared(
        fitted_X_train,
        fitted_y_train,
        fitted_X_valid,
        fitted_y_valid,
        target_column=target_column,
        categorical_columns=fitted_categorical_columns,
        params=hpo_result.best_lightgbm_params,
        train_config=train_config,
        top_p=float(top_p if top_p is not None else hpo_result.best_result.top_p),
    )
    return fitted_pipeline, fitted_result


__all__ = [
    "refit_best_lightgbm_pipeline",
    "suggest_lgbm_params",
    "tune_lightgbm_pipeline",
]
