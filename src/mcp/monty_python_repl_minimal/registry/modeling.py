"""Modeling tools for the minimal registry package."""

from __future__ import annotations

from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from ..base import WorkspaceToolCollection
from ..core.registry import tool
from ..privacy import safe_json_value
from .base import (
    OptunaConfig,
    SplitConfig,
    StoredLightGBMModelArtifact,
    StoredLightGBMStudy,
    TrainConfig,
)
from .utils import (
    _align_validation_categories,
    _is_numeric_dtype,
    lift_at_top_p,
    make_lgb_ppv_eval,
    make_train_valid_split,
    ppv_at_top_p,
    prepare_lightgbm_frame,
    recall_at_top_p,
    suggest_lgbm_params,
    train_lightgbm_once,
)


class ModelingCollection(WorkspaceToolCollection):
    """Minimal LightGBM and Optuna workflow helpers."""

    name = "modeling"
    description = (
        "Train LightGBM baselines, tune with Optuna on PPV@5, inspect studies, "
        "and persist fitted model artifacts."
    )

    def _resolve_feature_columns(
        self,
        dataframe: pd.DataFrame,
        *,
        target_column: str,
        feature_columns: list[str] | None,
        id_columns: list[str] | None = None,
    ) -> list[str]:
        """Resolve feature columns for model fitting.

        Args:
            dataframe (pd.DataFrame): Source dataframe.
            target_column (str): Target column name.
            feature_columns (list[str] | None): Optional explicit feature subset.
            id_columns (list[str] | None): Optional identifier columns to exclude.

        Returns:
            list[str]: Final feature column list used for modeling.
        """

        id_columns = list(id_columns or [])
        if feature_columns is not None:
            return list(feature_columns)
        return [
            str(column)
            for column in dataframe.columns
            if column not in [target_column] + id_columns
        ]

    def _build_train_valid_frames(
        self,
        *,
        dataframe_handle: str,
        target_column: str,
        feature_columns: list[str],
        validation_handle: str | None,
        split_method: str,
        valid_frac: float,
        random_seed: int,
        time_column: str | None,
        group_column: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build train and validation frames for model fitting.

        Args:
            dataframe_handle (str): Source training dataframe handle.
            target_column (str): Target column name.
            feature_columns (list[str]): Feature columns used for modeling.
            validation_handle (str | None): Optional explicit validation handle.
            split_method (str): Split strategy when validation data is not provided.
            valid_frac (float): Validation fraction used for generated splits.
            random_seed (int): Random seed for deterministic splitting.
            time_column (str | None): Time column used for time-based splits.
            group_column (str | None): Group column used for grouped splits.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Training and validation frames.
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        required_columns = [target_column] + feature_columns
        train_base = dataframe[required_columns].copy()
        if validation_handle is not None:
            valid_frame = self._get_dataframe(validation_handle).copy()
            return train_base, valid_frame[required_columns].copy()
        split_config = SplitConfig(
            method=split_method,
            valid_frac=valid_frac,
            random_seed=random_seed,
            time_col=time_column,
            group_col=group_column,
        )
        return make_train_valid_split(train_base, config=split_config)

    def _infer_categorical_columns(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
    ) -> list[str]:
        """Infer categorical columns for native LightGBM handling.

        Args:
            dataframe (pd.DataFrame): Modeling dataframe.
            feature_columns (list[str]): Candidate feature columns.

        Returns:
            list[str]: Feature columns treated as categorical.
        """

        return [
            column
            for column in feature_columns
            if not _is_numeric_dtype(dataframe[column])
        ]

    @tool
    def train_lightgbm_baseline(
        self,
        dataframe_handle: str,
        target_column: str,
        *,
        feature_columns: list[str] | None = None,
        validation_handle: str | None = None,
        id_columns: list[str] | None = None,
        split_method: str = "random",
        valid_frac: float = 0.2,
        random_seed: int = 42,
        time_column: str | None = None,
        group_column: str | None = None,
        top_p: float = 0.05,
        num_threads: int = 4,
    ) -> dict[str, Any]:
        """Train one baseline LightGBM model optimized for PPV@5.

        Use this as the fastest modeling pass when you want a reasonable baseline
        model and feature-importance signal before running Optuna tuning.

        Args:
            dataframe_handle (str): Source training dataframe handle.
            target_column (str): Binary target column.
            feature_columns (list[str] | None): Optional explicit feature subset.
            validation_handle (str | None): Optional held-out validation dataframe.
            id_columns (list[str] | None): Optional identifier columns excluded from
                automatic feature resolution.
            split_method (str): Split strategy when validation data is not provided.
            valid_frac (float): Validation fraction used for generated splits.
            random_seed (int): Random seed for splitting and LightGBM training.
            time_column (str | None): Time column used for time-based splits.
            group_column (str | None): Group column used for grouped splits.
            top_p (float): Fraction retained for PPV-style evaluation metrics.
            num_threads (int): Number of LightGBM worker threads.

        Returns:
            dict[str, Any]: Model handle, validation metrics, and feature metadata.

        Examples:
            baseline = train_lightgbm_baseline(
                df_handle,
                "target",
                id_columns=["customer_id"],
                num_threads=1,
            )
            # Returns:
            # {
            #     "model_handle": "model_123",
            #     "evaluation_summary": {
            #         "valid_ppv_at_5": 0.31,
            #         "valid_recall_at_5": 0.09,
            #         "valid_lift_at_5": 2.4,
            #         "base_rate": 0.13
            #     },
            #     "feature_columns": ["score_signal", "balance", "segment"],
            #     "categorical_columns": ["segment"]
            # }
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column {target_column!r} was not found.")
        feature_columns = self._resolve_feature_columns(
            dataframe,
            target_column=target_column,
            feature_columns=feature_columns,
            id_columns=id_columns,
        )
        train_df, valid_df = self._build_train_valid_frames(
            dataframe_handle=dataframe_handle,
            target_column=target_column,
            feature_columns=feature_columns,
            validation_handle=validation_handle,
            split_method=split_method,
            valid_frac=valid_frac,
            random_seed=random_seed,
            time_column=time_column,
            group_column=group_column,
        )
        categorical_columns = self._infer_categorical_columns(train_df, feature_columns)
        result = train_lightgbm_once(
            train_df,
            valid_df,
            label_col=target_column,
            categorical_cols=categorical_columns,
            train_config=TrainConfig(seed=random_seed, num_threads=num_threads),
            top_p=top_p,
        )
        artifact = StoredLightGBMModelArtifact(
            booster=result["booster"],
            target_column=target_column,
            feature_columns=result["feature_columns"],
            categorical_columns=result["categorical_columns"],
            best_params={},
            best_iteration=result["best_iteration"],
            evaluation_summary={
                "valid_ppv_at_5": result["valid_ppv_at_5"],
                "valid_recall_at_5": result["valid_recall_at_5"],
                "valid_lift_at_5": result["valid_lift_at_5"],
                "base_rate": result["base_rate"],
            },
            feature_importance_gain=result["feature_importance_gain"],
        )
        model_handle = self._object_store.put(artifact, prefix="model")
        return {
            "model_handle": model_handle,
            "evaluation_summary": artifact.evaluation_summary,
            "feature_columns": artifact.feature_columns,
            "categorical_columns": artifact.categorical_columns,
        }

    @tool
    def tune_lightgbm(
        self,
        dataframe_handle: str,
        target_column: str,
        *,
        feature_columns: list[str] | None = None,
        validation_handle: str | None = None,
        id_columns: list[str] | None = None,
        split_method: str = "random",
        valid_frac: float = 0.2,
        random_seed: int = 42,
        time_column: str | None = None,
        group_column: str | None = None,
        top_p: float = 0.05,
        n_trials: int = 30,
        timeout: int | None = None,
        num_threads: int = 4,
    ) -> dict[str, Any]:
        """Tune a LightGBM model with Optuna against PPV@5.

        This workflow searches a stronger parameter space than the baseline model
        and stores the best study for later inspection or final fitting.

        Args:
            dataframe_handle (str): Source training dataframe handle.
            target_column (str): Binary target column.
            feature_columns (list[str] | None): Optional explicit feature subset.
            validation_handle (str | None): Optional held-out validation dataframe.
            id_columns (list[str] | None): Optional identifier columns excluded from
                automatic feature resolution.
            split_method (str): Split strategy when validation data is not provided.
            valid_frac (float): Validation fraction used for generated splits.
            random_seed (int): Random seed for splitting, Optuna, and LightGBM.
            time_column (str | None): Time column used for time-based splits.
            group_column (str | None): Group column used for grouped splits.
            top_p (float): Fraction retained for PPV-style evaluation metrics.
            n_trials (int): Maximum number of Optuna trials.
            timeout (int | None): Optional optimization timeout in seconds.
            num_threads (int): Number of LightGBM worker threads.

        Returns:
            dict[str, Any]: Study handle, best objective value, and best params.

        Examples:
            study = tune_lightgbm(
                df_handle,
                "target",
                n_trials=20,
                num_threads=1,
            )
            # Returns:
            # {
            #     "study_handle": "study_123",
            #     "best_value": 0.31,
            #     "best_params": {
            #         "objective": "binary",
            #         "learning_rate": 0.05,
            #         "num_leaves": 63,
            #         "feature_fraction": 0.7
            #     },
            #     "objective_metric": "ppv_at_5"
            # }
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column {target_column!r} was not found.")
        feature_columns = self._resolve_feature_columns(
            dataframe,
            target_column=target_column,
            feature_columns=feature_columns,
            id_columns=id_columns,
        )
        train_df, valid_df = self._build_train_valid_frames(
            dataframe_handle=dataframe_handle,
            target_column=target_column,
            feature_columns=feature_columns,
            validation_handle=validation_handle,
            split_method=split_method,
            valid_frac=valid_frac,
            random_seed=random_seed,
            time_column=time_column,
            group_column=group_column,
        )
        categorical_columns = self._infer_categorical_columns(train_df, feature_columns)
        train_config = TrainConfig(seed=random_seed, num_threads=num_threads)
        optuna_config = OptunaConfig(
            n_trials=n_trials,
            timeout=timeout,
            random_seed=random_seed,
        )

        X_train, y_train, final_cats = prepare_lightgbm_frame(
            train_df,
            label_col=target_column,
            categorical_cols=categorical_columns,
        )
        X_valid, y_valid, _ = prepare_lightgbm_frame(
            valid_df,
            label_col=target_column,
            categorical_cols=final_cats,
        )
        X_valid = _align_validation_categories(X_train, X_valid, final_cats)

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=final_cats,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=final_cats,
            reference=dtrain,
            free_raw_data=False,
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=optuna_config.random_seed),
        )

        trial_rows: list[dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            params = suggest_lgbm_params(
                trial,
                num_threads=train_config.num_threads,
                seed=train_config.seed,
            )
            booster = lgb.train(
                params=params,
                train_set=dtrain,
                valid_sets=[dvalid],
                valid_names=["valid"],
                feval=make_lgb_ppv_eval(p=top_p),
                num_boost_round=train_config.num_boost_round,
                callbacks=[
                    lgb.early_stopping(
                        train_config.early_stopping_rounds,
                        verbose=False,
                    ),
                    lgb.log_evaluation(0),
                ],
            )
            valid_pred = booster.predict(X_valid, num_iteration=booster.best_iteration)
            ppv5 = ppv_at_top_p(y_valid.values, valid_pred, p=top_p)
            rec5 = recall_at_top_p(y_valid.values, valid_pred, p=top_p)
            lift5 = lift_at_top_p(y_valid.values, valid_pred, p=top_p)
            trial.set_user_attr("best_iteration", int(booster.best_iteration))
            trial.set_user_attr("valid_ppv_at_5", float(ppv5))
            trial.set_user_attr("valid_recall_at_5", float(rec5))
            trial.set_user_attr("valid_lift_at_5", float(lift5))
            return float(ppv5)

        study.optimize(
            objective,
            n_trials=optuna_config.n_trials,
            timeout=optuna_config.timeout,
            show_progress_bar=False,
        )

        for trial in sorted(
            study.trials,
            key=lambda item: item.value if item.value is not None else -np.inf,
            reverse=True,
        ):
            trial_rows.append(
                {
                    "trial": int(trial.number),
                    "value": float(trial.value) if trial.value is not None else None,
                    "best_iteration": trial.user_attrs.get("best_iteration"),
                    "valid_ppv_at_5": trial.user_attrs.get("valid_ppv_at_5"),
                    "valid_recall_at_5": trial.user_attrs.get("valid_recall_at_5"),
                    "valid_lift_at_5": trial.user_attrs.get("valid_lift_at_5"),
                    "params": safe_json_value(trial.params),
                }
            )

        best_params = dict(study.best_params)
        best_params.update(
            {
                "objective": "binary",
                "metric": "None",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "seed": train_config.seed,
                "num_threads": train_config.num_threads,
            }
        )
        artifact = StoredLightGBMStudy(
            train_handle=dataframe_handle,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=final_cats,
            validation_handle=validation_handle,
            best_params=best_params,
            top_p=top_p,
            study=study,
            trial_rows=trial_rows,
        )
        handle = self._object_store.put(artifact, prefix="study")
        return {
            "study_handle": handle,
            "best_value": float(study.best_value),
            "best_params": safe_json_value(best_params),
            "objective_metric": "ppv_at_5",
        }

    @tool
    def inspect_hpo_study(self, study_handle: str) -> dict[str, Any]:
        """Return a safe Optuna study summary.

        Args:
            study_handle (str): Stored Optuna study handle.

        Returns:
            dict[str, Any]: Compact study summary including top trials.

        Examples:
            study_summary = inspect_hpo_study(study_handle)
            # Returns:
            # {
            #     "type": "StoredLightGBMStudy",
            #     "target_column": "target",
            #     "trial_count": 20,
            #     "best_value": 0.31,
            #     "best_params": {"learning_rate": 0.05, "num_leaves": 63}
            # }
        """

        artifact = self._object_store.get(
            study_handle,
            expected_type=StoredLightGBMStudy,
        )
        return artifact.to_json_summary()

    @tool
    def fit_best_lightgbm(self, study_handle: str) -> dict[str, Any]:
        """Fit the best LightGBM configuration from a stored Optuna study.

        Args:
            study_handle (str): Stored Optuna study handle.

        Returns:
            dict[str, Any]: Model handle plus aggregate evaluation summary.

        Examples:
            best_model = fit_best_lightgbm(study_handle)
            # Returns:
            # {
            #     "model_handle": "model_456",
            #     "evaluation_summary": {
            #         "valid_ppv_at_5": 0.34,
            #         "valid_recall_at_5": 0.11,
            #         "valid_lift_at_5": 2.6,
            #         "base_rate": 0.13
            #     }
            # }
        """

        study_artifact = self._object_store.get(
            study_handle,
            expected_type=StoredLightGBMStudy,
        )
        train_df = self._get_dataframe(study_artifact.train_handle).copy()
        required_columns = [
            study_artifact.target_column
        ] + study_artifact.feature_columns
        train_df = train_df[required_columns].copy()
        if study_artifact.validation_handle is None:
            train_part, valid_part = make_train_valid_split(
                train_df,
                config=SplitConfig(random_seed=42),
            )
        else:
            train_part = train_df
            valid_part = self._get_dataframe(study_artifact.validation_handle)[
                required_columns
            ].copy()
        result = train_lightgbm_once(
            train_part,
            valid_part,
            label_col=study_artifact.target_column,
            categorical_cols=study_artifact.categorical_columns,
            params=study_artifact.best_params,
            top_p=study_artifact.top_p,
        )
        artifact = StoredLightGBMModelArtifact(
            booster=result["booster"],
            target_column=study_artifact.target_column,
            feature_columns=result["feature_columns"],
            categorical_columns=result["categorical_columns"],
            best_params=study_artifact.best_params,
            best_iteration=result["best_iteration"],
            evaluation_summary={
                "valid_ppv_at_5": result["valid_ppv_at_5"],
                "valid_recall_at_5": result["valid_recall_at_5"],
                "valid_lift_at_5": result["valid_lift_at_5"],
                "base_rate": result["base_rate"],
            },
            feature_importance_gain=result["feature_importance_gain"],
        )
        model_handle = self._object_store.put(artifact, prefix="model")
        return {
            "model_handle": model_handle,
            "evaluation_summary": artifact.evaluation_summary,
        }

    @tool
    def inspect_model(self, model_handle: str) -> dict[str, Any]:
        """Return a safe fitted-model summary.

        Args:
            model_handle (str): Stored fitted-model handle.

        Returns:
            dict[str, Any]: Compact fitted-model summary.

        Examples:
            model_summary = inspect_model(model_handle)
            # Returns:
            # {
            #     "type": "StoredLightGBMModelArtifact",
            #     "target_column": "target",
            #     "best_iteration": 87,
            #     "evaluation_summary": {"valid_ppv_at_5": 0.34, "base_rate": 0.13},
            #     "top_feature_importances": [{"feature": "score_signal", "gain": 123.4}]
            # }
        """

        artifact = self._object_store.get(
            model_handle,
            expected_type=StoredLightGBMModelArtifact,
        )
        return artifact.to_json_summary()

    @tool
    def save_model_artifact(self, model_handle: str, path: str) -> str:
        """Persist a fitted model artifact to `/workspace`.

        Args:
            model_handle (str): Stored fitted-model handle.
            path (str): Output artifact path under `/workspace`.

        Returns:
            str: Virtual workspace path to the saved artifact.

        Examples:
            saved_path = save_model_artifact(model_handle, "/workspace/models/baseline.joblib")
            # Returns:
            # "/workspace/models/baseline.joblib"
        """

        artifact = self._object_store.get(
            model_handle,
            expected_type=StoredLightGBMModelArtifact,
        )
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, host_path)
        self._record_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool
    def load_model_artifact(self, path: str) -> str:
        """Load a fitted model artifact from `/workspace`.

        Args:
            path (str): Workspace path to a previously saved model artifact.

        Returns:
            str: Handle for the loaded fitted-model artifact.

        Examples:
            model_handle = load_model_artifact("/workspace/models/baseline.joblib")
            # Returns:
            # "model_789"
        """

        artifact = joblib.load(self._resolve_host_path(path))
        if not isinstance(artifact, StoredLightGBMModelArtifact):
            raise TypeError("Loaded artifact is not a StoredLightGBMModelArtifact.")
        return self._object_store.put(artifact, prefix="model")


__all__ = ["ModelingCollection"]
