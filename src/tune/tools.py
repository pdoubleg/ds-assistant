from typing import Any, Callable, Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, RegressorMixin

from .schema import HPOProfile


def generate_search_space_from_code(code: str) -> Callable[[optuna.trial.Trial], dict]:
    """Execute LLM code and return the define_search_space function."""
    local_ns: dict[str, Any] = {"optuna": optuna, "np": np}
    exec(code, local_ns)
    return local_ns["define_search_space"]


def render_estimator_params(estimator: BaseEstimator, deep: bool = True) -> str:
    """
    Render a scikit-learn estimator's tunable parameters into a compact,
    LLM-friendly string. Includes model name and task type.

    Args:
        estimator: Any sklearn BaseEstimator subclass instance
        deep: If True, include nested parameters (e.g., pipeline__step__param)

    Returns:
        str: Human-readable description of model and parameters.
    """
    if not hasattr(estimator, "get_params"):
        raise TypeError("Object is not a scikit-learn estimator.")

    # Detect model type
    task_type = None
    if isinstance(estimator, ClassifierMixin):
        task_type = "classification"
    elif isinstance(estimator, RegressorMixin):
        task_type = "regression"
    elif isinstance(estimator, ClusterMixin):
        task_type = "clustering"
    else:
        task_type = "unspecified"

    model_name = estimator.__class__.__name__

    params = estimator.get_params(deep=deep)
    lines = [f"{name} = {repr(val)}" for name, val in sorted(params.items())]

    header = f"Model: {model_name} ({task_type})"
    return header + "\n" + "\n".join(lines)


def hpo_profile_from_dataframe(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    task: Literal["binary", "multiclass", "regression", "unknown"] | None = None,
    categorical_like: list[str] | None = None,
    mode: Literal["fast", "thorough"] = "fast",
    corr_sample_cap: int = 200,
    top_k_columns: int = 10,
    random_state: int = 0,
) -> HPOProfile:
    """
    Build a HPO profile from a dataframe (and optional target).

    Parameters
    ----------
    X : DataFrame of features
    y : Optional target
    task : If None, inferred as:
        - binary if 2 unique values
        - multiclass if integer with <= 10 unique values
        - regression otherwise
    categorical_like : columns to force-treat as categorical
    mode : 'fast' (default) or 'thorough' (adds top correlated pairs)
    corr_sample_cap : max numeric columns used for correlation snapshot
    top_k_columns : how many columns to show in top-missing / top-zero lists
    random_state : RNG seed
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    # Dtype buckets
    cat_cols = _infer_categorical_like(X, categorical_like)
    num_cols = _safe_num_cols(X, cat_cols)
    bool_cols = [
        c
        for c in X.columns
        if X[c].dtype.kind == "b" and c not in cat_cols and c not in num_cols
    ]
    dt_cols = [
        c
        for c in X.columns
        if isinstance(X[c].dtype, np.dtype) and np.issubdtype(X[c].dtype, np.datetime64)
    ]

    # Basic counts
    num_numeric = len(num_cols)
    num_categorical = len(cat_cols)
    num_boolean = len(bool_cols)
    num_datetime = len(dt_cols)

    # Missingness
    frac_missing_overall = float(X.isna().sum().sum() / max(1, n_samples * n_features))
    miss_by_col = _missing_by_col(X)
    top_missing = sorted(miss_by_col.items(), key=lambda kv: kv[1], reverse=True)[
        :top_k_columns
    ]

    # Zero-inflation (numeric)
    zero_frac_col = _numeric_zero_fraction(X, num_cols)
    zero_overall = float(np.nanmean(list(zero_frac_col.values()))) if num_cols else None
    top_zero = sorted(
        zero_frac_col.items(),
        key=lambda kv: (kv[1] if kv[1] is not None else -1),
        reverse=True,
    )[:top_k_columns]

    # Categorical cardinality & rarity
    cat_card = _categorical_cardinality(X, cat_cols)
    avg_cat_card = float(np.mean(list(cat_card.values()))) if cat_card else None
    high_card_cols = [
        c for c, k in cat_card.items() if k > min(100, int(0.1 * n_samples))
    ]
    rare_rate = _rare_category_rate(X, cat_cols, cutoff=0.01)

    # Sparsity proxy: combine numeric density & rare-category rate
    if num_cols:
        nz_density = []
        for c in num_cols:
            s = pd.to_numeric(X[c], errors="coerce")
            valid = s.notna()
            denom = valid.sum()
            if denom == 0:
                continue
            nz = (s[valid] != 0).sum()
            nz_density.append(nz / denom)
        numeric_sparsity = 1.0 - float(np.mean(nz_density)) if nz_density else 0.0
    else:
        numeric_sparsity = 0.0
    sparsity_proxy = float(np.clip(0.7 * numeric_sparsity + 0.3 * rare_rate, 0.0, 1.0))

    # Correlation snapshot
    corr_info = _numeric_corr_snapshot(
        X, num_cols, sample_cap=corr_sample_cap, mode=mode, rng=rng
    )

    # Feature scale dispersion
    feature_scale_cov = _feature_scale_dispersion(X, num_cols)

    # Target facts
    inferred_task: Literal["binary", "multiclass", "regression", "unknown"] = "unknown"
    target_clf: dict[str, Any] | None = None
    target_reg: dict[str, Any] | None = None
    if y is not None:
        ys = pd.Series(y)
        uniq = ys.dropna().unique()
        if task is not None:
            inferred_task = task
        else:
            if len(uniq) == 2:
                inferred_task = "binary"
            elif len(uniq) <= 10:
                inferred_task = "multiclass"
            else:
                inferred_task = "regression"
        if inferred_task in {"binary", "multiclass"}:
            target_clf = _class_info(ys)
        elif inferred_task == "regression":
            target_reg = _regression_moments(ys)

    # Assemble
    facts = HPOProfile(
        n_samples=int(n_samples),
        n_features=int(n_features),
        num_numeric=int(num_numeric),
        num_categorical=int(num_categorical),
        num_boolean=int(num_boolean),
        num_datetime=int(num_datetime),
        frac_missing_overall=float(frac_missing_overall),
        frac_zero_overall_numeric=(
            float(zero_overall) if zero_overall is not None else None
        ),
        sparsity_proxy=float(sparsity_proxy),
        rare_category_rate_mean=float(rare_rate),
        corr_num_used=int(corr_info["num_numeric_used"]),
        corr_median_abs=(
            corr_info["median_abs_corr"]
            if corr_info["median_abs_corr"] is not None
            else None
        ),
        corr_q90_abs=(
            corr_info["q90_abs_corr"] if corr_info["q90_abs_corr"] is not None else None
        ),
        corr_max_abs=(
            corr_info["max_abs_corr"] if corr_info["max_abs_corr"] is not None else None
        ),
        corr_top_pairs=corr_info["top_pairs"],
        feature_scale_cov=float(feature_scale_cov),
        categorical_cardinality=cat_card,
        avg_categorical_cardinality=(
            float(avg_cat_card) if avg_cat_card is not None else None
        ),
        high_cardinality_columns=high_card_cols,
        top_missing_columns=top_missing,
        top_zero_fraction_numeric=top_zero,
        task=inferred_task if task is None else task,
        target_binary_multiclass=target_clf,
        target_regression=target_reg,
    )
    return facts


def _safe_num_cols(X: pd.DataFrame, cat_cols: list[str]) -> list[str]:
    return [c for c in X.columns if c not in set(cat_cols)]


def _infer_categorical_like(
    X: pd.DataFrame, categorical_like: list[str] | None
) -> list[str]:
    cats = set(categorical_like or [])
    for c in X.columns:
        dt = X[c].dtype
        if getattr(dt, "name", str(dt)) in {"object", "category", "bool"}:
            cats.add(c)
    return [c for c in X.columns if c in cats]


def _class_info(y: pd.Series) -> dict[str, Any]:
    vc = y.value_counts(dropna=False)
    total = int(vc.sum())
    probs = (vc / total).to_dict()
    minority_frac = float(min(probs.values())) if len(probs) else None
    entropy = (
        float(-sum(p * np.log2(p) for p in probs.values() if p > 0))
        if len(probs)
        else None
    )
    return {
        "num_classes": int(len(probs)),
        "class_frequencies": {str(k): int(v) for k, v in vc.to_dict().items()},
        "class_probs": {str(k): float(v) for k, v in (vc / total).to_dict().items()},
        "minority_class_fraction": minority_frac,
        "target_entropy_bits": entropy,
    }


def _regression_moments(y: pd.Series) -> dict[str, Any]:
    yv = y.dropna().astype(float)
    n = int(yv.shape[0])
    mean = float(yv.mean()) if n else None
    std = float(yv.std()) if n else None
    skew = float(yv.skew()) if n else None
    kurt = float(yv.kurt()) if n else None
    out3 = float((np.abs((yv - mean) / (std + 1e-12)) > 3).mean()) if n else None
    p = np.percentile(yv, [1, 5, 50, 95, 99]) if n else [None] * 5
    return {
        "count_non_null": n,
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurt,
        "outlier_rate_std_gt_3": out3,
        "p01": float(p[0]) if n else None,
        "p05": float(p[1]) if n else None,
        "p50": float(p[2]) if n else None,
        "p95": float(p[3]) if n else None,
        "p99": float(p[4]) if n else None,
    }


def _numeric_zero_fraction(X: pd.DataFrame, num_cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        valid = s.notna()
        denom = valid.sum()
        if denom == 0:
            out[c] = np.nan
        else:
            out[c] = float(((s[valid] == 0).sum()) / denom)
    return out


def _missing_by_col(X: pd.DataFrame) -> dict[str, float]:
    n = X.shape[0]
    return {c: float(X[c].isna().sum() / max(1, n)) for c in X.columns}


def _categorical_cardinality(X: pd.DataFrame, cat_cols: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cat_cols:
        out[c] = int(X[c].astype("category").cat.categories.size)
    return out


def _rare_category_rate(
    X: pd.DataFrame, cat_cols: list[str], cutoff: float = 0.01
) -> float:
    rates = []
    for c in cat_cols:
        s = X[c].astype("category")
        vc = s.value_counts(dropna=True)
        if vc.sum() == 0:
            continue
        rates.append(float((vc / vc.sum() < cutoff).mean()))
    return float(np.mean(rates)) if rates else 0.0


def _numeric_corr_snapshot(
    X: pd.DataFrame,
    num_cols: list[str],
    sample_cap: int,
    mode: Literal["fast", "thorough"],
    rng: np.random.Generator,
) -> dict[str, Any]:
    cols = num_cols
    if len(cols) == 0:
        return {
            "num_numeric_used": 0,
            "median_abs_corr": None,
            "q90_abs_corr": None,
            "max_abs_corr": None,
            "top_pairs": [],
        }
    if len(cols) > sample_cap:
        cols = list(
            pd.Index(cols)[rng.choice(len(cols), size=sample_cap, replace=False)]
        )
    if len(cols) < 2:
        return {
            "num_numeric_used": len(cols),
            "median_abs_corr": 0.0,
            "q90_abs_corr": 0.0,
            "max_abs_corr": 0.0,
            "top_pairs": [],
        }

    corr = (
        X[cols]
        .astype(float)
        .corr()
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    # exclude diagonal
    mask = ~np.eye(len(cols), dtype=bool)
    vals = corr.where(mask).stack().values
    med = float(np.median(vals))
    q90 = float(np.quantile(vals, 0.90))
    mx = float(np.max(vals))
    result = {
        "num_numeric_used": int(len(cols)),
        "median_abs_corr": med,
        "q90_abs_corr": q90,
        "max_abs_corr": mx,
        "top_pairs": [],
    }
    if mode == "thorough":
        # top 10 correlated pairs (names & value)
        tri = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
        top = tri.stack().sort_values(ascending=False).head(10)
        result["top_pairs"] = [
            {"feature_a": a, "feature_b": b, "abs_corr": float(v)}
            for (a, b), v in top.items()
        ]
    return result


def _feature_scale_dispersion(X: pd.DataFrame, num_cols: list[str]) -> float:
    stds = []
    for c in num_cols:
        s = pd.to_numeric(X[c], errors="coerce")
        sd = float(s.std(skipna=True)) if s.notna().any() else np.nan
        if np.isfinite(sd):
            stds.append(sd)
    if not stds:
        return 0.0
    stds = np.array(stds)
    return float(np.std(stds) / (np.mean(stds) + 1e-12))
