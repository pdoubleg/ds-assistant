"""Shared statistics and modeling helpers for the minimal registry package."""

from __future__ import annotations

import math
import random
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from .base import SplitConfig, TrainConfig


def ppv_at_top_p(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    p: float = 0.05,
) -> float:
    """Compute PPV among the highest-ranked predictions.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_score (np.ndarray): Model scores used for ranking.
        p (float): Fraction of rows retained in the top slice.

    Returns:
        float: Precision within the retained top-ranked rows.
    """

    if len(y_true) == 0:
        return 0.0
    n_top = max(1, int(math.ceil(len(y_true) * p)))
    order = np.argsort(-y_score)
    top_idx = order[:n_top]
    return float(np.mean(y_true[top_idx]))


def recall_at_top_p(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    p: float = 0.05,
) -> float:
    """Compute recall among the highest-ranked predictions.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_score (np.ndarray): Model scores used for ranking.
        p (float): Fraction of rows retained in the top slice.

    Returns:
        float: Recall captured by the retained top-ranked rows.
    """

    positives = float(np.sum(y_true))
    if positives == 0:
        return 0.0
    n_top = max(1, int(math.ceil(len(y_true) * p)))
    order = np.argsort(-y_score)
    top_idx = order[:n_top]
    captured = float(np.sum(y_true[top_idx]))
    return captured / positives


def lift_at_top_p(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    p: float = 0.05,
) -> float:
    """Compute lift relative to the base rate.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_score (np.ndarray): Model scores used for ranking.
        p (float): Fraction of rows retained in the top slice.

    Returns:
        float: Lift of the top slice relative to the full-sample base rate.
    """

    base_rate = float(np.mean(y_true)) if len(y_true) else 0.0
    if base_rate == 0.0:
        return 0.0
    return ppv_at_top_p(y_true, y_score, p=p) / base_rate


def make_lgb_ppv_eval(
    *,
    p: float = 0.05,
) -> Any:
    """Build the custom LightGBM evaluation callback.

    Args:
        p (float): Fraction of rows retained when computing PPV.

    Returns:
        Any: LightGBM-compatible evaluation callback.
    """

    def _feval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
        y_true = dataset.get_label()
        score = ppv_at_top_p(y_true=y_true, y_score=preds, p=p)
        return f"ppv_at_{int(p * 100)}", score, True

    return _feval


def _safe_series_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to datetimes with invalid values coerced.

    Args:
        series (pd.Series): Source series to coerce.

    Returns:
        pd.Series: Datetime-like series with invalid values set to ``NaT``.
    """

    return pd.to_datetime(series, errors="coerce", utc=False)


def _is_numeric_dtype(series: pd.Series) -> bool:
    """Return whether a series is numeric.

    Args:
        series (pd.Series): Series to inspect.

    Returns:
        bool: Whether the series has a numeric dtype.
    """

    return pd.api.types.is_numeric_dtype(series)


def _is_datetime_dtype(series: pd.Series) -> bool:
    """Return whether a series is datetime-like.

    Args:
        series (pd.Series): Series to inspect.

    Returns:
        bool: Whether the series has a datetime-like dtype.
    """

    return pd.api.types.is_datetime64_any_dtype(series)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for tied values.

    Args:
        values (np.ndarray): Values to rank.

    Returns:
        np.ndarray: Average rank per value.
    """

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def fast_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute a fast binary AUC estimate.

    Args:
        y_true (np.ndarray): Binary ground-truth labels.
        y_score (np.ndarray): Scores to evaluate.

    Returns:
        float: Approximate AUC computed from average ranks.
    """

    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score).astype(float)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata_average(scores)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def univariate_numeric_score(feature: pd.Series, target: pd.Series) -> float:
    """Return a univariate numeric screening score.

    Args:
        feature (pd.Series): Candidate numeric feature.
        target (pd.Series): Binary target series.

    Returns:
        float: Absolute AUC lift above random performance.
    """

    valid = feature.notna() & target.notna()
    if valid.sum() < 100:
        return 0.0
    feature_values = feature[valid].astype(float).values
    target_values = target[valid].astype(int).values
    if np.nanstd(feature_values) == 0:
        return 0.0
    try:
        auc = fast_auc_score(target_values, feature_values)
        return float(abs(auc - 0.5))
    except Exception:
        return 0.0


def univariate_categorical_score(
    feature: pd.Series,
    target: pd.Series,
    *,
    min_count: int = 25,
) -> float:
    """Return a categorical screening score without exposing raw categories.

    Args:
        feature (pd.Series): Candidate categorical feature.
        target (pd.Series): Binary target series.
        min_count (int): Minimum category count retained in the score.

    Returns:
        float: Weighted absolute lift relative to the overall target mean.
    """

    valid = feature.notna() & target.notna()
    if valid.sum() < 100:
        return 0.0
    grouped = (
        pd.DataFrame(
            {
                "x": feature[valid].astype(str),
                "y": target[valid].astype(int),
            }
        )
        .groupby("x", observed=True)
        .agg(
            n=("y", "size"),
            rate=("y", "mean"),
        )
    )
    grouped = grouped[grouped["n"] >= min_count]
    if grouped.empty:
        return 0.0
    overall = float(target[valid].mean())
    if overall <= 0:
        return 0.0
    weighted_abs_lift = float(
        ((grouped["rate"] - overall).abs() * grouped["n"]).sum() / grouped["n"].sum()
    )
    return weighted_abs_lift


def prepare_lightgbm_frame(
    dataframe: pd.DataFrame,
    *,
    label_col: str,
    categorical_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare a dataframe for LightGBM with native categorical handling.

    Args:
        dataframe (pd.DataFrame): Source dataframe.
        label_col (str): Target column name.
        categorical_cols (list[str] | None): Preferred categorical columns.
        drop_cols (list[str] | None): Optional columns removed before fitting.

    Returns:
        tuple[pd.DataFrame, pd.Series, list[str]]: Feature dataframe, target
        series, and resolved categorical column list.
    """

    drop_cols = list(drop_cols or [])
    categorical_cols = list(categorical_cols or [])

    target = dataframe[label_col].astype(int)
    features = dataframe.drop(
        columns=[label_col] + [column for column in drop_cols if column in dataframe],
        errors="ignore",
    ).copy()

    final_cats: list[str] = []
    for column in list(features.columns):
        series = features[column]
        if _is_datetime_dtype(series):
            datetime_values = _safe_series_to_datetime(series)
            features[column] = (datetime_values.view("int64") / 10**9).replace(
                -9223372036854775808 / 10**9, np.nan
            )
            continue

        if column in categorical_cols:
            features[column] = series.astype("category")
            final_cats.append(column)
            continue

        if _is_numeric_dtype(series):
            continue

        # Use native categorical handling for any remaining non-numeric columns.
        features[column] = series.astype("category")
        final_cats.append(column)

    return features, target, final_cats


def prepare_lightgbm_features(
    dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a feature-only dataframe for LightGBM scoring.

    Args:
        dataframe: Source dataframe containing the feature columns.
        feature_columns: Exact feature columns expected by the fitted model.
        categorical_cols: Preferred categorical columns.

    Returns:
        Tuple of prepared features and resolved categorical columns.
    """
    categorical_cols = list(categorical_cols or [])
    features = dataframe[feature_columns].copy()

    final_cats: list[str] = []
    for column in list(features.columns):
        series = features[column]
        if _is_datetime_dtype(series):
            datetime_values = _safe_series_to_datetime(series)
            features[column] = (datetime_values.view("int64") / 10**9).replace(
                -9223372036854775808 / 10**9, np.nan
            )
            continue

        if column in categorical_cols:
            features[column] = series.astype("category")
            final_cats.append(column)
            continue

        if _is_numeric_dtype(series):
            continue

        features[column] = series.astype("category")
        final_cats.append(column)

    return features, final_cats


def make_train_valid_split(
    dataframe: pd.DataFrame,
    *,
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a train/validation split.

    Args:
        dataframe (pd.DataFrame): Source dataframe to split.
        config (SplitConfig): Split strategy configuration.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
    """

    if config.method == "time":
        if not config.time_col or config.time_col not in dataframe.columns:
            raise ValueError("`time_col` must be provided for time-based split.")
        temp = dataframe.copy()
        temp[config.time_col] = _safe_series_to_datetime(temp[config.time_col])
        temp = temp.sort_values(config.time_col)
        cutoff = int((1.0 - config.valid_frac) * len(temp))
        return temp.iloc[:cutoff].copy(), temp.iloc[cutoff:].copy()

    if config.method == "group":
        if not config.group_col or config.group_col not in dataframe.columns:
            raise ValueError("`group_col` must be provided for group-based split.")
        groups = dataframe[config.group_col].dropna().unique().tolist()
        rng = random.Random(config.random_seed)
        rng.shuffle(groups)
        n_valid_groups = max(1, int(len(groups) * config.valid_frac))
        valid_groups = set(groups[:n_valid_groups])
        valid_mask = dataframe[config.group_col].isin(valid_groups)
        return dataframe.loc[~valid_mask].copy(), dataframe.loc[valid_mask].copy()

    rng = np.random.default_rng(config.random_seed)
    idx = np.arange(len(dataframe))
    rng.shuffle(idx)
    n_valid = max(1, int(len(dataframe) * config.valid_frac))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return dataframe.iloc[train_idx].copy(), dataframe.iloc[valid_idx].copy()


def _align_validation_categories(
    train_features: pd.DataFrame,
    valid_features: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Align validation categorical columns to training categories.

    Args:
        train_features (pd.DataFrame): Training feature frame.
        valid_features (pd.DataFrame): Validation feature frame.
        categorical_columns (list[str]): Categorical columns to align.

    Returns:
        pd.DataFrame: Validation frame with aligned category vocabularies.
    """

    aligned = valid_features.reindex(columns=train_features.columns)
    for column in categorical_columns:
        if column in aligned.columns:
            aligned[column] = pd.Categorical(
                aligned[column],
                categories=train_features[column].cat.categories,
            )
    return aligned


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
        categorical_cols=categorical_columns,
    )
    aligned_features = features.reindex(columns=feature_columns)
    for column in final_cats:
        if column in aligned_features.columns:
            aligned_features[column] = aligned_features[
                column
            ].cat.remove_unused_categories()
    return booster.predict(aligned_features, num_iteration=best_iteration)


def top_p_indices(
    y_score: np.ndarray,
    *,
    p: float = 0.05,
) -> np.ndarray:
    """Return ranked indices for the retained top-p slice.

    Args:
        y_score: Model scores used for ranking.
        p: Fraction of rows retained in the top slice.

    Returns:
        Numpy array of retained row indices in descending score order.
    """
    if len(y_score) == 0:
        return np.array([], dtype=int)
    n_top = max(1, int(math.ceil(len(y_score) * p)))
    order = np.argsort(-y_score, kind="mergesort")
    return order[:n_top]


def summarize_top_p_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    p: float = 0.05,
) -> dict[str, Any]:
    """Return aggregate outcome metrics for the highest-ranked predictions.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Model scores used for ranking.
        p: Fraction of rows retained in the top slice.

    Returns:
        Aggregate top-p prediction metrics.
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
    true_positive_count = int(np.sum(selected_labels == 1))
    false_positive_count = int(np.sum(selected_labels == 0))
    base_rate = float(np.mean(labels))
    return {
        "top_p": float(p),
        "row_count": int(len(labels)),
        "top_p_row_count": int(len(selected)),
        "score_threshold": float(np.min(selected_scores))
        if len(selected_scores)
        else None,
        "true_positive_count": true_positive_count,
        "false_positive_count": false_positive_count,
        "ppv_at_p": float(ppv_at_top_p(labels, scores, p=p)),
        "recall_at_p": float(recall_at_top_p(labels, scores, p=p)),
        "lift_at_p": float(lift_at_top_p(labels, scores, p=p)),
        "base_rate": base_rate,
    }


def train_lightgbm_once(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    label_col: str,
    categorical_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
    params: dict[str, Any] | None = None,
    train_config: TrainConfig | None = None,
    top_p: float = 0.05,
) -> dict[str, Any]:
    """Fit one LightGBM model and evaluate it on a validation frame.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        valid_df (pd.DataFrame): Validation dataframe.
        label_col (str): Target column name.
        categorical_cols (list[str] | None): Preferred categorical columns.
        drop_cols (list[str] | None): Optional columns removed before fitting.
        params (dict[str, Any] | None): Additional LightGBM parameters.
        train_config (TrainConfig | None): Training configuration override.
        top_p (float): Fraction retained for PPV-style validation metrics.

    Returns:
        dict[str, Any]: Booster, feature metadata, and validation metrics.
    """

    train_config = train_config or TrainConfig()
    params = dict(params or {})

    X_train, y_train, final_cats = prepare_lightgbm_frame(
        train_df,
        label_col=label_col,
        categorical_cols=categorical_cols,
        drop_cols=drop_cols,
    )
    X_valid, y_valid, _ = prepare_lightgbm_frame(
        valid_df,
        label_col=label_col,
        categorical_cols=final_cats,
        drop_cols=drop_cols,
    )
    X_valid = _align_validation_categories(X_train, X_valid, final_cats)

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

    # Some LightGBM options, such as `linear_tree`, are bound when the Dataset
    # handle is constructed, so build datasets only after params are finalized.
    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=final_cats,
        params=default_params,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=final_cats,
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

    return {
        "booster": booster,
        "feature_columns": list(X_train.columns),
        "categorical_columns": final_cats,
        "best_iteration": int(booster.best_iteration),
        "valid_ppv_at_5": float(ppv_at_top_p(y_valid.values, valid_pred, p=top_p)),
        "valid_recall_at_5": float(
            recall_at_top_p(y_valid.values, valid_pred, p=top_p)
        ),
        "valid_lift_at_5": float(lift_at_top_p(y_valid.values, valid_pred, p=top_p)),
        "base_rate": float(np.mean(y_valid)),
        "feature_importance_gain": {
            str(feature): float(gain)
            for feature, gain in importance_gain.to_dict().items()
        },
    }


def suggest_lgbm_params(
    trial: optuna.Trial,
    *,
    num_threads: int,
    seed: int,
) -> dict[str, Any]:
    """Suggest a LightGBM parameter set for Optuna.

    Args:
        trial (optuna.Trial): Active Optuna trial.
        num_threads (int): Number of LightGBM worker threads.
        seed (int): Random seed propagated to LightGBM.

    Returns:
        dict[str, Any]: Candidate LightGBM parameter set.
    """

    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "goss"])
    max_depth = trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 7, 8, 10, 12])
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": boosting_type,
        "verbosity": -1,
        "seed": seed,
        "num_threads": num_threads,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 255, log=True),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 50, 1_000, log=True
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-4, 10.0, log=True
        ),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.95),
        "feature_fraction_bynode": trial.suggest_float(
            "feature_fraction_bynode", 0.4, 1.0
        ),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
        "max_depth": max_depth,
        "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
        "linear_tree": trial.suggest_categorical("linear_tree", [False, True]),
        "force_row_wise": trial.suggest_categorical("force_row_wise", [False, True]),
        "cat_smooth": trial.suggest_float("cat_smooth", 0.0, 100.0),
        "cat_l2": trial.suggest_float("cat_l2", 0.0, 100.0),
        "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 64),
        "min_data_per_group": trial.suggest_int(
            "min_data_per_group", 10, 500, log=True
        ),
        "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 256, log=True),
    }
    if boosting_type == "gbdt":
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.5, 0.95)
        params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)
    if boosting_type == "goss":
        top_rate = trial.suggest_float("top_rate", 0.1, 0.5)
        max_other_rate = min(0.4, 0.99 - top_rate)
        params["top_rate"] = top_rate
        params["other_rate"] = trial.suggest_float(
            "other_rate",
            0.05,
            max_other_rate,
        )
    return params


__all__ = [
    "_align_validation_categories",
    "_is_datetime_dtype",
    "_is_numeric_dtype",
    "fast_auc_score",
    "lift_at_top_p",
    "make_lgb_ppv_eval",
    "make_train_valid_split",
    "ppv_at_top_p",
    "prepare_lightgbm_features",
    "prepare_lightgbm_frame",
    "recall_at_top_p",
    "score_lightgbm_dataframe",
    "suggest_lgbm_params",
    "summarize_top_p_predictions",
    "top_p_indices",
    "train_lightgbm_once",
    "univariate_categorical_score",
    "univariate_numeric_score",
]
