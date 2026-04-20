"""Modeling tools for the minimal registry package."""

from __future__ import annotations

from typing import Any

import math
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
    StoredDataframeReport,
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
    prepare_lightgbm_features,
    prepare_lightgbm_frame,
    recall_at_top_p,
    score_lightgbm_dataframe,
    suggest_lgbm_params,
    summarize_top_p_predictions as summarize_top_p_prediction_metrics,
    top_p_indices,
    train_lightgbm_once,
)

_PPV_TOP_P = 0.05


def _lightgbm_tunable_param_catalog() -> list[dict[str, Any]]:
    """Return the default Optuna search space for LightGBM tuning."""

    return [
        {
            "name": "boosting_type",
            "type": "categorical",
            "default": "gbdt",
            "choices": ["gbdt", "goss"],
            "description": (
                "Boosting strategy. Choose `goss` to enable gradient-based one-side "
                "sampling; if you do, also tune `top_rate` and `other_rate`, and note "
                "that bagging knobs are typically ignored."
            ),
        },
        {
            "name": "learning_rate",
            "type": "float",
            "default": 0.05,
            "range": {"low": 0.03, "high": 0.15, "log": True},
            "description": "Step size used by each boosting round.",
        },
        {
            "name": "num_leaves",
            "type": "int",
            "default": 63,
            "range": {"low": 16, "high": 255, "log": True},
            "description": "Maximum leaf count per tree.",
        },
        {
            "name": "min_child_samples",
            "type": "int",
            "default": 200,
            "range": {"low": 50, "high": 1000, "log": True},
            "description": "Minimum row count allowed in a leaf.",
        },
        {
            "name": "min_child_weight",
            "type": "float",
            "default": 1e-3,
            "range": {"low": 1e-4, "high": 10.0, "log": True},
            "description": "Minimum Hessian weight allowed in a leaf.",
        },
        {
            "name": "feature_fraction",
            "type": "float",
            "default": 0.7,
            "range": {"low": 0.4, "high": 0.95, "log": False},
            "description": "Fraction of features sampled per tree.",
        },
        {
            "name": "feature_fraction_bynode",
            "type": "float",
            "default": 1.0,
            "range": {"low": 0.4, "high": 1.0, "log": False},
            "description": "Fraction of features sampled at each split node.",
        },
        {
            "name": "bagging_fraction",
            "type": "float",
            "default": 0.8,
            "range": {"low": 0.5, "high": 0.95, "log": False},
            "condition": {"boosting_type": "gbdt"},
            "description": (
                "Fraction of rows sampled during bagging. Only relevant when "
                "`boosting_type='gbdt'`."
            ),
        },
        {
            "name": "bagging_freq",
            "type": "int",
            "default": 1,
            "range": {"low": 1, "high": 7, "log": False},
            "condition": {"boosting_type": "gbdt"},
            "description": (
                "Bagging frequency in boosting rounds. Only relevant when "
                "`boosting_type='gbdt'`."
            ),
        },
        {
            "name": "lambda_l1",
            "type": "float",
            "default": 0.1,
            "range": {"low": 1e-4, "high": 10.0, "log": True},
            "description": "L1 regularization term.",
        },
        {
            "name": "lambda_l2",
            "type": "float",
            "default": 1.0,
            "range": {"low": 1e-4, "high": 10.0, "log": True},
            "description": "L2 regularization term.",
        },
        {
            "name": "min_gain_to_split",
            "type": "float",
            "default": 0.0,
            "range": {"low": 0.0, "high": 2.0, "log": False},
            "description": "Minimum gain required to split a node.",
        },
        {
            "name": "max_depth",
            "type": "categorical",
            "default": -1,
            "choices": [-1, 4, 5, 6, 7, 8, 10, 12],
            "description": "Optional hard cap on tree depth.",
        },
        {
            "name": "extra_trees",
            "type": "categorical",
            "default": False,
            "choices": [False, True],
            "description": "Use extremely randomized splits to inject additional randomness.",
        },
        {
            "name": "linear_tree",
            "type": "categorical",
            "default": False,
            "choices": [False, True],
            "description": (
                "Enable piecewise-linear leaves. `linear_trees` is an alias; turning "
                "this on increases memory usage and is best left with the default "
                "serial CPU learner."
            ),
        },
        {
            "name": "force_row_wise",
            "type": "categorical",
            "default": False,
            "choices": [False, True],
            "description": (
                "Force row-wise histogram building on CPU. Useful for some wide or "
                "high-thread workloads; do not combine with `force_col_wise`."
            ),
        },
        {
            "name": "top_rate",
            "type": "float",
            "default": 0.2,
            "range": {"low": 0.1, "high": 0.5, "log": False},
            "condition": {"boosting_type": "goss"},
            "description": (
                "Share of large-gradient rows kept by GOSS. Only relevant when "
                "`boosting_type='goss'`; tune together with `other_rate`."
            ),
        },
        {
            "name": "other_rate",
            "type": "float",
            "default": 0.1,
            "range": {"low": 0.05, "high": 0.4, "log": False},
            "condition": {"boosting_type": "goss"},
            "description": (
                "Share of small-gradient rows kept by GOSS. Only relevant when "
                "`boosting_type='goss'`; keep `top_rate + other_rate < 1.0`."
            ),
        },
        {
            "name": "cat_smooth",
            "type": "float",
            "default": 10.0,
            "range": {"low": 0.0, "high": 100.0, "log": False},
            "description": (
                "Smoothing for categorical target statistics. Only matters when "
                "native categorical features are present."
            ),
        },
        {
            "name": "cat_l2",
            "type": "float",
            "default": 10.0,
            "range": {"low": 0.0, "high": 100.0, "log": False},
            "description": (
                "L2 regularization applied to categorical splits. Only matters when "
                "native categorical features are present."
            ),
        },
        {
            "name": "max_cat_to_onehot",
            "type": "int",
            "default": 4,
            "range": {"low": 1, "high": 64, "log": False},
            "description": (
                "Maximum category count before LightGBM stops using one-vs-other "
                "categorical splitting. Only matters when native categorical features "
                "are present."
            ),
        },
        {
            "name": "min_data_per_group",
            "type": "int",
            "default": 100,
            "range": {"low": 10, "high": 500, "log": True},
            "description": (
                "Minimum row count per categorical group. Only matters when native "
                "categorical features are present."
            ),
        },
        {
            "name": "max_cat_threshold",
            "type": "int",
            "default": 32,
            "range": {"low": 8, "high": 256, "log": True},
            "description": (
                "Upper bound on the number of split points evaluated for categorical "
                "features. Only matters when native categorical features are present."
            ),
        },
    ]


class ModelingCollection(WorkspaceToolCollection):
    """Minimal LightGBM and Optuna workflow helpers."""

    name = "modeling"
    description = (
        "Train LightGBM baselines, score dataframes, analyze top-p prediction "
        "quality, tune with Optuna on PPV@5, inspect studies, and persist fitted "
        "model artifacts."
    )

    @tool
    def list_lightgbm_tunable_params(self) -> dict[str, Any]:
        """Describe the tunable LightGBM parameters used by Optuna studies.

        Call this before `tune_lightgbm(...)` so the agent can see the supported
        search space and explain which knobs are available for PPV@5 optimization.

        Returns:
            dict[str, Any]: Tunable-parameter catalog and optimization objective.

        Examples:
            ```python
            tunables = list_lightgbm_tunable_params()
            # Returns
            # {
            #     "objective_metric": "ppv_at_5",
            #     "native_categorical_handling": True,
            #     "params": [...],
            # }
            ```
        """

        return {
            "objective_metric": "ppv_at_5",
            "native_categorical_handling": True,
            "params": _lightgbm_tunable_param_catalog(),
        }

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

    def _prepare_target_and_score_frame(
        self,
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
        self,
        dataframe: pd.DataFrame,
        *,
        target_column: str,
        score_column: str,
        feature_columns: list[str] | None,
        id_columns: list[str] | None,
    ) -> list[str]:
        """Resolve feature columns for aggregate prediction error analysis."""
        excluded = {target_column, score_column, *(id_columns or [])}
        if feature_columns is not None:
            missing = [
                column for column in feature_columns if column not in dataframe.columns
            ]
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)}.")
            resolved = [column for column in feature_columns if column not in excluded]
        else:
            resolved = [
                str(column)
                for column in dataframe.columns
                if str(column) not in excluded
            ]
        if not resolved:
            raise ValueError("No feature columns remain after exclusions.")
        return resolved

    def _ks_distance(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return a lightweight two-sample KS distance."""
        if len(left) == 0 or len(right) == 0:
            return 0.0
        left_sorted = np.sort(left.astype(float))
        right_sorted = np.sort(right.astype(float))
        combined = np.sort(np.unique(np.concatenate([left_sorted, right_sorted])))
        left_cdf = np.searchsorted(left_sorted, combined, side="right") / len(
            left_sorted
        )
        right_cdf = np.searchsorted(right_sorted, combined, side="right") / len(
            right_sorted
        )
        return float(np.max(np.abs(left_cdf - right_cdf)))

    def _entropy(self, probabilities: np.ndarray) -> float:
        """Return entropy for a probability vector."""
        positive = probabilities[probabilities > 0]
        if len(positive) == 0:
            return 0.0
        return float(-np.sum(positive * np.log2(positive)))

    def _js_divergence(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return the Jensen-Shannon divergence between two distributions."""
        midpoint = 0.5 * (left + right)

        def _kl_divergence(base: np.ndarray, ref: np.ndarray) -> float:
            valid = (base > 0) & (ref > 0)
            if not np.any(valid):
                return 0.0
            return float(np.sum(base[valid] * np.log2(base[valid] / ref[valid])))

        return 0.5 * _kl_divergence(left, midpoint) + 0.5 * _kl_divergence(
            right, midpoint
        )

    def _analyze_numeric_false_positives(
        self,
        dataframe: pd.DataFrame,
        *,
        column: str,
        false_positive_mask: pd.Series,
        true_positive_mask: pd.Series,
    ) -> dict[str, Any]:
        """Return aggregate numeric FP-vs-TP diagnostics for one column."""
        fp_series = pd.to_numeric(
            dataframe.loc[false_positive_mask, column],
            errors="coerce",
        )
        tp_series = pd.to_numeric(
            dataframe.loc[true_positive_mask, column],
            errors="coerce",
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
        ks_distance = self._ks_distance(fp_non_null.values, tp_non_null.values)
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
            "false_positive_mean": float(fp_non_null.mean())
            if len(fp_non_null)
            else None,
            "true_positive_mean": float(tp_non_null.mean())
            if len(tp_non_null)
            else None,
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
        self,
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
            fp_entropy = self._entropy(fp_probs)
            tp_entropy = self._entropy(tp_probs)
            normalization = math.log2(len(categories)) if len(categories) > 1 else 1.0
            fp_entropy_norm = fp_entropy / normalization if normalization > 0 else 0.0
            tp_entropy_norm = tp_entropy / normalization if normalization > 0 else 0.0
            fp_concentration = float(np.max(fp_probs)) if len(fp_probs) else 0.0
            tp_concentration = float(np.max(tp_probs)) if len(tp_probs) else 0.0
            js_divergence = self._js_divergence(fp_probs, tp_probs)
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
            num_threads (int): Number of LightGBM worker threads.

        Returns:
            dict[str, Any]: Model handle, validation metrics, and feature metadata.

        Examples:
            baseline = train_lightgbm_baseline(
                df_handle,
                "target",
                id_columns=["customer_id"],
                feature_columns=["balance", "income"],
                validation_handle="df_abc123",
                split_method="random",
                valid_frac=0.2,
                num_threads=1,
            )
            # Returns
            # {
            #     "model_handle": "model_abc123",
            #     "summary": "Trained baseline LightGBM model on 2 features using native categorical handling.",
            #     "evaluation_summary": {
            #         "valid_ppv_at_5": 0.75,
            #         "valid_recall_at_5": 0.75,
            #         "valid_lift_at_5": 1.5,
            #         "base_rate": 0.5,
            #     },
            #     "feature_columns": ["balance", "income"],
            #     "categorical_columns": ["income"],
            # }
            model_summary = inspect_model(baseline["model_handle"])
            # Returns
            # {
            #     "type": "StoredLightGBMModelArtifact",
            #     "target_column": "target",
            #     "feature_columns": ["balance", "income"],
            #     "categorical_columns": ["income"],
            #     "best_params": {},
            #     "best_iteration": 100,
            #     "evaluation_summary": {...},
            #     "feature_importance_gain": {...},
            # }
            ```
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
            top_p=_PPV_TOP_P,
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
        summary = (
            f"Trained baseline LightGBM model on {len(artifact.feature_columns)} "
            "features using native categorical handling."
        )
        return {
            "model_handle": model_handle,
            "summary": summary,
            "evaluation_summary": artifact.evaluation_summary,
            "feature_columns": artifact.feature_columns,
            "categorical_columns": artifact.categorical_columns,
        }

    @tool
    def score_model_dataframe(
        self,
        model_handle: str,
        dataframe_handle: str,
        *,
        score_column_name: str = "pred_score",
    ) -> dict[str, Any]:
        """Score a dataframe with a fitted LightGBM model.

        Use this after training or loading a fitted model when you want a scored
        dataframe handle for aggregate diagnostics, top-p summaries, or plots.

        Args:
            model_handle: Stored fitted-model handle.
            dataframe_handle: Dataframe handle to score.
            score_column_name: Name of the added numeric score column.

        Returns:
            Compact scoring summary plus the scored dataframe handle.

        Examples:
            ```python
            scored = score_model_dataframe(
                model_handle,
                df_handle,
                score_column_name="pred_score",
            )
            # Returns
            # {
            #     "dataframe_handle": "df_abc123",
            #     "score_column": "pred_score",
            #     "summary": "Scored 1000 rows with the fitted LightGBM model.",
            #     "row_count": 1000,
            #     "score_mean": 0.42,
            # }
            ```
        """
        artifact = self._object_store.get(
            model_handle,
            expected_type=StoredLightGBMModelArtifact,
        )
        dataframe = self._get_dataframe(dataframe_handle).copy()
        if score_column_name in dataframe.columns:
            raise ValueError(
                f"Score column {score_column_name!r} already exists on the dataframe."
            )
        missing_columns = [
            column
            for column in artifact.feature_columns
            if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

        scores = score_lightgbm_dataframe(
            dataframe,
            booster=artifact.booster,
            feature_columns=artifact.feature_columns,
            categorical_columns=artifact.categorical_columns,
            best_iteration=artifact.best_iteration,
        )
        scored = dataframe.copy()
        scored[score_column_name] = scores
        scored_handle = self._object_store.put(scored, prefix="df")
        summary = f"Scored {len(scored)} rows with the fitted LightGBM model."
        return {
            "dataframe_handle": scored_handle,
            "score_column": score_column_name,
            "summary": summary,
            "row_count": int(len(scored)),
            "score_mean": float(np.mean(scores)) if len(scores) else None,
            "score_std": float(np.std(scores)) if len(scores) else None,
            "score_min": float(np.min(scores)) if len(scores) else None,
            "score_max": float(np.max(scores)) if len(scores) else None,
        }

    @tool
    def summarize_top_p_predictions(
        self,
        dataframe_handle: str,
        target_column: str,
        score_column: str,
        *,
        top_p: float = _PPV_TOP_P,
    ) -> dict[str, Any]:
        """Summarize aggregate prediction quality inside the top-p slice.

        Use this on a scored dataframe handle to measure how many true and false
        positives land in the highest-ranked prediction slice.

        Args:
            dataframe_handle: Scored dataframe handle.
            target_column: Binary target column.
            score_column: Numeric prediction score column.
            top_p: Fraction retained in the top-ranked slice.

        Returns:
            Aggregate top-p prediction metrics.

        Examples:
            ```python
            top_p_summary = summarize_top_p_predictions(
                scored_df_handle,
                "target",
                "pred_score",
                top_p=0.05,
            )
            # Returns
            # {
            #     "summary": "Top 5.0% slice contains 50 rows with PPV 0.32.",
            #     "true_positive_count": 16,
            #     "false_positive_count": 34,
            # }
            ```
        """
        dataframe = self._get_dataframe(dataframe_handle)
        temp = self._prepare_target_and_score_frame(
            dataframe,
            target_column=target_column,
            score_column=score_column,
        )
        metrics = summarize_top_p_prediction_metrics(
            temp[target_column].values,
            temp[score_column].values,
            p=top_p,
        )
        metrics["summary"] = (
            f"Top {top_p * 100:.1f}% slice contains {metrics['top_p_row_count']} rows "
            f"with PPV {metrics['ppv_at_p']:.4f}."
        )
        metrics["target_column"] = target_column
        metrics["score_column"] = score_column
        return metrics

    @tool
    def analyze_top_p_false_positives(
        self,
        dataframe_handle: str,
        target_column: str,
        score_column: str,
        *,
        top_p: float = _PPV_TOP_P,
        feature_columns: list[str] | None = None,
        id_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Analyze false-positive patterns inside the highest-ranked predictions.

        This tool compares false positives against true positives within the top-p
        slice and stores only aggregate column-level diagnostics in a report handle.

        Args:
            dataframe_handle: Scored dataframe handle.
            target_column: Binary target column.
            score_column: Numeric prediction score column.
            top_p: Fraction retained in the top-ranked slice.
            feature_columns: Optional explicit feature subset to analyze.
            id_columns: Optional identifier columns excluded from analysis.

        Returns:
            Report handle plus a compact analysis summary.

        Examples:
            ```python
            fp_report = analyze_top_p_false_positives(
                scored_df_handle,
                "target",
                "pred_score",
                top_p=0.05,
                id_columns=["customer_id"],
            )
            # Returns
            # {
            #     "report_handle": "report_abc123",
            #     "summary": "Analyzed top-p false positives across 12 columns.",
            #     "top_p": 0.05,
            #     "analyzed_column_count": 12,
            # }
            ```
        """
        dataframe = self._get_dataframe(dataframe_handle).copy()
        temp = self._prepare_target_and_score_frame(
            dataframe,
            target_column=target_column,
            score_column=score_column,
        )
        analysis_columns = self._resolve_analysis_feature_columns(
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
                "Top-p slice must contain both false positives and true positives "
                "for comparative analysis."
            )

        numeric_findings: list[dict[str, Any]] = []
        categorical_findings: list[dict[str, Any]] = []
        for column in analysis_columns:
            if _is_numeric_dtype(top_slice[column]):
                numeric_findings.append(
                    self._analyze_numeric_false_positives(
                        top_slice,
                        column=column,
                        false_positive_mask=false_positive_mask,
                        true_positive_mask=true_positive_mask,
                    )
                )
            else:
                categorical_findings.append(
                    self._analyze_categorical_false_positives(
                        top_slice,
                        column=column,
                        false_positive_mask=false_positive_mask,
                        true_positive_mask=true_positive_mask,
                    )
                )

        numeric_findings.sort(key=lambda item: item["pattern_score"], reverse=True)
        categorical_findings.sort(key=lambda item: item["pattern_score"], reverse=True)
        top_p_summary = summarize_top_p_prediction_metrics(
            temp[target_column].values,
            temp[score_column].values,
            p=top_p,
        )
        summary = (
            f"Analyzed top-{top_p * 100:.1f}% false positives across "
            f"{len(analysis_columns)} columns."
        )
        report = StoredDataframeReport(
            report_type="top_p_false_positive_analysis",
            title="Top-p false positive analysis",
            summary=summary,
            details={
                "target_column": target_column,
                "score_column": score_column,
                "top_p_summary": top_p_summary,
                "false_positive_count": int(false_positive_mask.sum()),
                "true_positive_count": int(true_positive_mask.sum()),
                "numeric_findings": numeric_findings,
                "categorical_findings": categorical_findings,
            },
            metadata={
                "analyzed_columns": analysis_columns,
                "id_columns": list(id_columns or []),
            },
        )
        report_handle = self._object_store.put(report, prefix="report")
        return {
            "report_handle": report_handle,
            "summary": summary,
            "top_p": float(top_p),
            "analyzed_column_count": int(len(analysis_columns)),
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
            n_trials (int): Maximum number of Optuna trials.
            timeout (int | None): Optional optimization timeout in seconds.
            num_threads (int): Number of LightGBM worker threads.

        Returns:
            dict[str, Any]: Study handle, best objective value, and best params.

        Examples:
            ```python
            study = tune_lightgbm(
                df_handle,
                "target",
                n_trials=20,
                num_threads=1,
            )
            # Returns
            # {
            #     "study_handle": "study_abc123",
            #     "summary": "Completed 20 Optuna trial(s) optimizing ppv_at_5 over 2 features.",
            #     "best_value": 0.75,
            #     "best_params": {...},
            #     "objective_metric": "ppv_at_5",
            # }
            study_summary = inspect_hpo_study(study["study_handle"])
            # Returns
            # {
            #     "type": "StoredLightGBMStudy",
            #     "target_column": "target",
            #     "feature_columns": ["balance", "income"],
            #     "categorical_columns": ["income"],
            #     "validation_handle": "df_abc123",
            #     "best_params": {...},
            #     "top_p": 0.05,
            #     "study": {...},
            #     "trial_rows": [...],
            # }
            ```
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
            # Rebuild Dataset handles per trial so Dataset-time LightGBM params
            # like `linear_tree` are applied during construction.
            dtrain = lgb.Dataset(
                X_train,
                label=y_train,
                categorical_feature=final_cats,
                params=params,
                free_raw_data=False,
            )
            dvalid = lgb.Dataset(
                X_valid,
                label=y_valid,
                categorical_feature=final_cats,
                params=params,
                reference=dtrain,
                free_raw_data=False,
            )
            booster = lgb.train(
                params=params,
                train_set=dtrain,
                valid_sets=[dvalid],
                valid_names=["valid"],
                feval=make_lgb_ppv_eval(p=_PPV_TOP_P),
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
            ppv5 = ppv_at_top_p(y_valid.values, valid_pred, p=_PPV_TOP_P)
            rec5 = recall_at_top_p(y_valid.values, valid_pred, p=_PPV_TOP_P)
            lift5 = lift_at_top_p(y_valid.values, valid_pred, p=_PPV_TOP_P)
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
            top_p=_PPV_TOP_P,
            study=study,
            trial_rows=trial_rows,
        )
        handle = self._object_store.put(artifact, prefix="study")
        summary = (
            f"Completed {len(study.trials)} Optuna trial(s) optimizing ppv_at_5 "
            f"over {len(feature_columns)} features."
        )
        return {
            "study_handle": handle,
            "summary": summary,
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
            ```python
            study_summary = inspect_hpo_study(study_handle)
            # Returns
            # {
            #     "type": "StoredLightGBMStudy",
            #     "target_column": "target",
            #     "feature_columns": ["balance", "income"],
            #     "categorical_columns": ["income"],
            #     "validation_handle": "df_abc123",
            #     "best_params": {...},
            #     "top_p": 0.05,
            #     "study": {...},
            #     "trial_rows": [...],
            # }
            ```
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
            ```python
            best_model = fit_best_lightgbm(study_handle)
            # Returns
            # {
            #     "model_handle": "model_abc123",
            #     "summary": "Fit the best LightGBM configuration from the stored Optuna study.",
            #     "evaluation_summary": {
            #         "valid_ppv_at_5": 0.75,
            #         "valid_recall_at_5": 0.75,
            #         "valid_lift_at_5": 1.5,
            #         "base_rate": 0.5,
            #     },
            # }
            model_summary = inspect_model(best_model["model_handle"])
            # Returns
            # {
            #     "type": "StoredLightGBMModelArtifact",
            #     "target_column": "target",
            #     "feature_columns": ["balance", "income"],
            #     "categorical_columns": ["income"],
            #     "best_params": {...},
            #     "best_iteration": 100,
            #     "evaluation_summary": {...},
            #     "feature_importance_gain": {...},
            # }
            ```
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
        summary = "Fit the best LightGBM configuration from the stored Optuna study."
        return {
            "model_handle": model_handle,
            "summary": summary,
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
            ```python
            model_summary = inspect_model(model_handle)
            # Returns
            # {
            #     "type": "StoredLightGBMModelArtifact",
            #     "target_column": "target",
            #     "feature_columns": ["balance", "income"],
            #     "categorical_columns": ["income"],
            #     "best_params": {...},
            #     "best_iteration": 100,
            #     "evaluation_summary": {...},
            #     "feature_importance_gain": {...},
            # }
            ```
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
            ```python
            saved_path = save_model_artifact(
                model_handle,
                "/workspace/models/baseline.joblib",
            )
            # Returns
            # "/workspace/models/baseline.joblib"
            ```
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
            ```python
            model_handle = load_model_artifact("/workspace/models/baseline.joblib")
            # Returns
            # "model_abc123"
            ```
        """

        artifact = joblib.load(self._resolve_host_path(path))
        if not isinstance(artifact, StoredLightGBMModelArtifact):
            raise TypeError("Loaded artifact is not a StoredLightGBMModelArtifact.")
        return self._object_store.put(artifact, prefix="model")


__all__ = ["ModelingCollection"]
