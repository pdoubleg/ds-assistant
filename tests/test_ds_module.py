"""Focused tests for the standalone ds package."""

from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ds.config import (
    BatchedFeatureScreenConfig,
    FeatureScreenConfig,
    OptunaConfig,
    PipelineSearchSpace,
    SplitConfig,
    TrainConfig,
)
from src.ds.error_analysis import analyze_top_p_false_positives
from src.ds.hpo import (
    refit_best_lightgbm_pipeline,
    suggest_lgbm_params,
    tune_lightgbm_pipeline,
)
from src.ds.io import read_parquet_fragment
from src.ds.main import run_binary_experiment
from src.ds.metrics import ppv_at_top_p, summarize_top_p_predictions
from src.ds.modeling import (
    build_train_valid_frames,
    fit_lightgbm_binary,
    make_train_valid_split,
    prepare_lightgbm_train_valid_frames,
    resolve_feature_columns,
    score_dataframe,
)
from src.ds.pipelines import (
    build_feature_pipeline,
    fit_transform_features,
    suggest_pipeline_params,
)
from src.ds.selection import (
    analyze_feature_correlation,
    rank_features_by_lightgbm,
    rank_features_by_shap,
    screen_feature_batches,
    screen_features,
    screen_parquet_feature_batches,
    univariate_categorical_score,
)
from src.ds.nn.model import RealMLP, make_binary_dataloader, predict_proba
from src.ds.nn.preprocess import TabularTorchPreprocessor


def make_classification_frame(rows: int = 600) -> pd.DataFrame:
    """Return a small synthetic binary-classification dataframe.

    Args:
        rows: Number of rows to generate.

    Returns:
        Synthetic dataframe with numeric, categorical, and datetime features.
    """

    rng = np.random.default_rng(42)
    signal = rng.normal(loc=0.0, scale=1.0, size=rows)
    balance = 100.0 + signal * 25.0 + rng.normal(scale=5.0, size=rows)
    utilization = rng.uniform(0.0, 1.0, size=rows)
    segment = np.where(
        signal > 0.4, "vip", np.where(signal > -0.2, "standard", "watch")
    )
    noise = rng.normal(scale=0.35, size=rows)
    target = ((signal + 0.5 * utilization + noise) > 0.35).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(rows),
            "snapshot_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "signal": signal,
            "balance": balance,
            "utilization": utilization,
            "segment": pd.Series(segment, dtype="category"),
            "target": target,
        }
    )
    frame.loc[frame.index[::19], "balance"] = np.nan
    frame.loc[frame.index[::23], "segment"] = None
    return frame


def test_metrics_and_split_helpers() -> None:
    """Ranking metrics and split helpers should return coherent summaries."""

    frame = make_classification_frame()
    scores = frame["signal"].rank(pct=True).to_numpy()
    summary = summarize_top_p_predictions(frame["target"], scores, p=0.1)
    split = make_train_valid_split(
        frame,
        target_column="target",
        config=SplitConfig(valid_frac=0.2, random_seed=42, stratify=True),
    )

    assert ppv_at_top_p(frame["target"], scores, p=0.1) >= 0.0
    assert summary["top_p_row_count"] > 0
    assert len(split.train_df) + len(split.valid_df) == len(frame)
    assert abs(split.train_df["target"].mean() - split.valid_df["target"].mean()) < 0.03


def test_local_parquet_fragment_reader(tmp_path: Path) -> None:
    """Parquet fragment reads should work for local datasets when pyarrow exists."""

    pytest.importorskip("pyarrow")
    frame = make_classification_frame()
    parquet_path = tmp_path / "train.parquet"
    frame.to_parquet(parquet_path, index=False)

    sampled, metadata = read_parquet_fragment(
        parquet_path,
        columns=["target", "customer_id", "signal", "segment"],
        sample_n_rows=80,
        max_fragments=1,
        batch_size=17,
        include_metadata=True,
    )

    assert len(sampled) <= 80
    assert set(sampled.columns) == {"target", "customer_id", "signal", "segment"}
    assert sampled["customer_id"].tolist() == list(range(80))
    assert metadata.row_count == len(sampled)
    assert metadata.scanned_fragment_count == 1


def test_local_parquet_fragment_reader_streaming_random_sample(tmp_path: Path) -> None:
    """Reservoir parquet sampling should be deterministic and memory bounded."""

    pytest.importorskip("pyarrow")
    frame = make_classification_frame()
    parquet_path = tmp_path / "train.parquet"
    frame.to_parquet(parquet_path, index=False)

    first_sample = read_parquet_fragment(
        parquet_path,
        columns=["customer_id", "target"],
        sample_n_rows=80,
        batch_size=17,
        sample_strategy="reservoir",
        random_seed=7,
    )
    second_sample = read_parquet_fragment(
        parquet_path,
        columns=["customer_id", "target"],
        sample_n_rows=80,
        batch_size=17,
        sample_strategy="reservoir",
        random_seed=7,
    )

    assert len(first_sample) == 80
    assert first_sample.equals(second_sample)
    assert first_sample["customer_id"].max() > 79


def test_pipeline_param_unpacking_and_screening() -> None:
    """Pipeline search specs and feature screening should both be usable."""

    frame = make_classification_frame()
    pipeline = build_feature_pipeline(
        numeric_features=["signal", "balance", "utilization"],
        categorical_features=["segment"],
        numeric_transformers=[("impute", SimpleImputer(strategy="median"))],
        remainder="drop",
    )
    trial = optuna.trial.FixedTrial({"preprocess__numeric__impute__strategy": "mean"})
    pipeline_params = suggest_pipeline_params(
        trial,
        [
            PipelineSearchSpace(
                estimator_param="preprocess__numeric__impute__strategy",
                suggestion_kind="categorical",
                default="median",
                choices=["mean", "median"],
            )
        ],
    )
    report = screen_features(
        frame,
        target_column="target",
        id_columns=["customer_id"],
        config=FeatureScreenConfig(top_k_univariate=3, min_non_null=20),
    )

    assert pipeline.named_steps["preprocess"] is not None
    assert pipeline_params["preprocess__numeric__impute__strategy"] == "mean"
    assert "signal" in report.selected_columns
    assert report.filtered_df.shape[1] >= 2
    assert set(report.selected_columns).isdisjoint(report.dropped_columns)
    assert set(report.selected_columns) | set(report.dropped_columns) == {
        "snapshot_date",
        "signal",
        "balance",
        "utilization",
        "segment",
    }
    assert "segment" in report.selected_columns


def test_oof_categorical_score_prefers_informative_feature() -> None:
    """OOF categorical screening should beat an uninformative categorical feature."""

    frame = make_classification_frame()
    rng = np.random.default_rng(7)
    noise_feature = pd.Series(
        rng.choice(["a", "b", "c", "d"], size=len(frame)),
        index=frame.index,
        dtype="category",
    )

    informative_score = univariate_categorical_score(frame["segment"], frame["target"])
    noise_score = univariate_categorical_score(noise_feature, frame["target"])

    assert informative_score > 0.0
    assert informative_score > noise_score


def test_lgbm_trial_helper_can_add_scale_pos_weight() -> None:
    """The LightGBM Optuna helper should support prevalence-aware class weights."""

    trial = optuna.trial.FixedTrial(
        {
            "lgbm__boosting_type": "gbdt",
            "lgbm__max_depth": -1,
            "lgbm__learning_rate": 0.05,
            "lgbm__num_leaves": 64,
            "lgbm__min_child_samples": 100,
            "lgbm__min_child_weight": 0.01,
            "lgbm__feature_fraction": 0.8,
            "lgbm__feature_fraction_bynode": 0.8,
            "lgbm__lambda_l1": 0.1,
            "lgbm__lambda_l2": 1.0,
            "lgbm__min_gain_to_split": 0.0,
            "lgbm__extra_trees": False,
            "lgbm__linear_tree": False,
            "lgbm__force_row_wise": False,
            "lgbm__cat_smooth": 10.0,
            "lgbm__cat_l2": 10.0,
            "lgbm__max_cat_to_onehot": 16,
            "lgbm__min_data_per_group": 20,
            "lgbm__max_cat_threshold": 64,
            "lgbm__bagging_fraction": 0.8,
            "lgbm__bagging_freq": 1,
            "lgbm__class_weight_mode": "scale_pos_weight",
            "lgbm__scale_pos_weight": 9.0,
        }
    )

    params = suggest_lgbm_params(
        trial,
        num_threads=1,
        seed=42,
        positive_class_ratio=0.1,
    )

    assert params["scale_pos_weight"] == 9.0
    assert "is_unbalance" not in params


def test_prepare_lightgbm_train_valid_frames_aligns_categories() -> None:
    """Prepared train and validation frames should share train-side categories."""

    train_df = pd.DataFrame(
        {
            "segment": pd.Series(["vip", "standard", "vip"], dtype="category"),
            "signal": [1.0, 0.2, -0.3],
            "target": [1, 0, 1],
        }
    )
    valid_df = pd.DataFrame(
        {
            "segment": ["standard", "watch"],
            "signal": [0.1, -0.4],
            "target": [0, 1],
        }
    )

    X_train, y_train, X_valid, y_valid, categorical_columns = (
        prepare_lightgbm_train_valid_frames(
            train_df,
            valid_df,
            target_column="target",
            feature_columns=["segment", "signal"],
        )
    )

    assert categorical_columns == ["segment"]
    assert list(X_train["segment"].cat.categories) == ["standard", "vip"]
    assert list(X_valid["segment"].cat.categories) == ["standard", "vip"]
    assert pd.isna(X_valid.loc[1, "segment"])
    assert y_train.tolist() == [1, 0, 1]
    assert y_valid.tolist() == [0, 1]


def test_hpo_refit_path_preserves_native_categorical_handling() -> None:
    """HPO and refit should share the same prepared categorical columns."""

    frame = make_classification_frame(rows=200)
    pipeline = Pipeline(
        [("identity", FunctionTransformer(feature_names_out="one-to-one"))]
    )
    pipeline.set_output(transform="pandas")

    hpo_result = tune_lightgbm_pipeline(
        frame,
        target_column="target",
        pipeline=pipeline,
        id_columns=["customer_id"],
        split_config=SplitConfig(random_seed=42),
        train_config=TrainConfig(
            num_threads=1, num_boost_round=20, early_stopping_rounds=5
        ),
        optuna_config=OptunaConfig(n_trials=1, random_seed=42),
        top_p=0.2,
    )
    _, refit_result = refit_best_lightgbm_pipeline(
        frame,
        hpo_result,
        target_column="target",
        pipeline=pipeline,
        id_columns=["customer_id"],
        train_config=TrainConfig(
            num_threads=1, num_boost_round=20, early_stopping_rounds=5
        ),
    )

    assert hpo_result.best_result.categorical_columns
    assert "segment" in hpo_result.best_result.categorical_columns
    assert (
        refit_result.categorical_columns == hpo_result.best_result.categorical_columns
    )
    assert refit_result.evaluation_summary["ppv_at_p"] >= 0.0


def test_correlation_pruning_keeps_stronger_target_signal() -> None:
    """Correlation pruning should keep the stronger feature when a target exists."""

    rng = np.random.default_rng(17)
    rows = 600
    latent = rng.normal(size=rows)
    target = (latent + rng.normal(scale=0.3, size=rows) > 0.0).astype(int)
    strong_signal = latent + rng.normal(scale=0.05, size=rows)
    weak_signal = strong_signal + rng.normal(scale=0.35, size=rows)
    frame = pd.DataFrame(
        {
            "strong_signal": strong_signal,
            "weak_signal": weak_signal,
            "target": target,
        }
    )

    result = analyze_feature_correlation(
        frame,
        target_column="target",
        feature_columns=["strong_signal", "weak_signal"],
        threshold=0.9,
    )

    assert "strong_signal" in result.selected_columns
    assert "weak_signal" in result.dropped_columns
    assert result.pair_rows[0]["kept_feature"] == "strong_signal"
    assert result.pair_rows[0]["dropped_feature"] == "weak_signal"
    assert result.pair_rows[0]["drop_reason"] == "lower_target_score"


def test_batched_feature_screening_decorrelates_across_batches(tmp_path: Path) -> None:
    """Batched screening should prune correlated numeric features across batches."""

    rng = np.random.default_rng(123)
    rows = 600
    latent = rng.normal(size=rows)
    target = (latent + rng.normal(scale=0.25, size=rows) > 0.0).astype(int)
    strong_signal = latent + rng.normal(scale=0.03, size=rows)
    weak_signal = strong_signal + rng.normal(scale=0.04, size=rows)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(rows),
            "strong_signal": strong_signal,
            "noise_a": rng.normal(size=rows),
            "constant_feature": 1.0,
            "weak_signal": weak_signal,
            "mostly_missing": np.nan,
            "segment": np.where(strong_signal > 0.0, "high", "low"),
            "target": target,
        }
    )
    frame.loc[:20, "mostly_missing"] = rng.normal(size=21)
    feature_columns = [
        "strong_signal",
        "noise_a",
        "constant_feature",
        "weak_signal",
        "mostly_missing",
        "segment",
    ]
    requested_columns: list[list[str]] = []

    def load_columns(columns: list[str]) -> pd.DataFrame:
        requested_columns.append(list(columns))
        return frame[columns].copy()

    result = screen_feature_batches(
        feature_columns=feature_columns,
        target_column="target",
        id_columns=["row_id"],
        load_columns=load_columns,
        config=BatchedFeatureScreenConfig(
            batch_size=3,
            correlation_batch_size=1,
            correlation_threshold=0.9,
            output_dir=tmp_path,
            top_k_univariate=10,
            screen_config=FeatureScreenConfig(
                top_k_univariate=10,
                min_non_null=20,
                max_missing_frac=0.9,
            ),
        ),
    )
    full_correlation = analyze_feature_correlation(
        frame,
        target_column="target",
        feature_columns=["strong_signal", "weak_signal"],
        threshold=0.9,
    )

    assert "strong_signal" in result.selected_columns
    assert "weak_signal" in result.dropped_columns
    assert full_correlation.dropped_columns == ["weak_signal"]
    assert any(
        row["dropped_feature"] == "weak_signal" and row["decision"] == "dropped"
        for row in result.correlation_pair_rows
    )
    assert any(
        row["feature"] == "constant_feature" and "zero_variance" in row["reasons"]
        for row in result.finding_rows
    )
    assert any(
        row["feature"] == "mostly_missing" and "too_missing" in row["reasons"]
        for row in result.finding_rows
    )
    assert len(result.finding_paths) == 2
    assert all(path.exists() for path in result.finding_paths)
    assert (
        max(
            len([column for column in columns if column not in {"target", "row_id"}])
            for columns in requested_columns
        )
        <= 3
    )


def test_parquet_batched_feature_screening_uses_projected_reads(tmp_path: Path) -> None:
    """Parquet wrapper should run batched screening from projected fragments."""

    pytest.importorskip("pyarrow")
    frame = make_classification_frame(rows=300)
    frame["balance_copy"] = frame["balance"] * 1.0
    parquet_path = tmp_path / "wide.parquet"
    frame.to_parquet(parquet_path, index=False)

    result = screen_parquet_feature_batches(
        parquet_path,
        target_column="target",
        id_columns=["customer_id"],
        feature_columns=["signal", "balance", "balance_copy", "segment"],
        config=BatchedFeatureScreenConfig(
            batch_size=2,
            correlation_batch_size=1,
            correlation_threshold=0.99,
            output_dir=tmp_path / "findings",
            top_k_univariate=4,
            screen_config=FeatureScreenConfig(top_k_univariate=4, min_non_null=20),
        ),
    )

    assert "signal" in result.selected_columns
    assert len(result.finding_paths) == 2
    assert any(path.exists() for path in result.finding_paths)
    assert (
        "balance" in result.dropped_columns or "balance_copy" in result.dropped_columns
    )


def test_torch_preprocessor_feeds_categorical_embeddings() -> None:
    """Torch helpers should pass encoded categoricals through embeddings."""

    frame = make_classification_frame(rows=64)
    preprocessor = TabularTorchPreprocessor(
        numeric_columns=["signal", "balance", "utilization"],
        categorical_columns=["segment"],
    )
    features = preprocessor.fit_transform(frame)
    dataloader = make_binary_dataloader(
        features,
        frame["target"],
        batch_size=16,
        shuffle=False,
    )
    model = RealMLP(
        input_dim=features["x_num"].shape[1],
        category_cardinalities=preprocessor.get_category_cardinalities(),
        hidden_dim=16,
        num_blocks=1,
    )

    x_num_batch, x_cat_batch, y_batch = next(iter(dataloader))
    logits = model(x_num_batch, x_cat_batch)
    probs = predict_proba(model, features)

    assert logits.shape == y_batch.shape
    assert probs.shape == (len(frame),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_lightgbm_training_hpo_and_error_analysis() -> None:
    """The standalone workflow should support training, tuning, and FP analysis."""

    frame = make_classification_frame()
    feature_columns = resolve_feature_columns(
        frame,
        target_column="target",
        id_columns=["customer_id"],
    )
    split = build_train_valid_frames(
        frame,
        target_column="target",
        feature_columns=feature_columns,
        split_config=SplitConfig(valid_frac=0.2, random_seed=42),
    )
    pipeline = build_feature_pipeline(
        numeric_features=["signal", "balance", "utilization"],
        categorical_features=["segment", "snapshot_date"],
        numeric_transformers=[("impute", SimpleImputer(strategy="median"))],
        remainder="drop",
    )
    transformed_train, transformed_valid, _ = fit_transform_features(
        pipeline,
        split.train_df,
        valid_df=split.valid_df,
        feature_columns=feature_columns,
    )
    assert transformed_valid is not None

    train_frame = transformed_train.copy()
    train_frame["target"] = split.train_df["target"].values
    valid_frame = transformed_valid.copy()
    valid_frame["target"] = split.valid_df["target"].values
    baseline_result = fit_lightgbm_binary(
        train_frame,
        valid_frame,
        target_column="target",
        feature_columns=[str(column) for column in transformed_train.columns],
        train_config=TrainConfig(
            num_threads=1, num_boost_round=60, early_stopping_rounds=10
        ),
        top_p=0.2,
    )
    scored_valid = score_dataframe(
        valid_frame, baseline_result, score_column_name="pred_score"
    )
    error_report = analyze_top_p_false_positives(
        scored_valid,
        target_column="target",
        score_column="pred_score",
        top_p=0.2,
    )
    ranking = rank_features_by_lightgbm(
        frame,
        target_column="target",
        feature_columns=feature_columns,
        keep_top_k=4,
        split_config=SplitConfig(random_seed=42),
        train_config=TrainConfig(
            num_threads=1, num_boost_round=40, early_stopping_rounds=10
        ),
        top_p=0.2,
    )
    hpo_result = tune_lightgbm_pipeline(
        frame,
        target_column="target",
        pipeline=pipeline,
        pipeline_search_space=[
            PipelineSearchSpace(
                estimator_param="preprocess__numeric__impute__strategy",
                suggestion_kind="categorical",
                default="median",
                choices=["mean", "median"],
            )
        ],
        id_columns=["customer_id"],
        split_config=SplitConfig(random_seed=42),
        train_config=TrainConfig(
            num_threads=1, num_boost_round=50, early_stopping_rounds=10
        ),
        optuna_config=OptunaConfig(n_trials=2, random_seed=42),
        top_p=0.2,
    )
    fitted_pipeline, refit_result = refit_best_lightgbm_pipeline(
        frame,
        hpo_result,
        target_column="target",
        pipeline=pipeline,
        id_columns=["customer_id"],
        train_config=TrainConfig(
            num_threads=1, num_boost_round=50, early_stopping_rounds=10
        ),
    )

    assert baseline_result.evaluation_summary["ppv_at_p"] >= 0.0
    assert error_report.analyzed_columns
    assert ranking.selected_columns
    assert hpo_result.best_result.evaluation_summary["ppv_at_p"] >= 0.0
    assert hpo_result.best_pipeline_params["preprocess__numeric__impute__strategy"] in {
        "mean",
        "median",
    }
    assert hasattr(fitted_pipeline, "fit")
    assert refit_result.evaluation_summary["ppv_at_p"] >= 0.0


def test_shap_feature_ranking_can_filter_against_random_feature() -> None:
    """SHAP ranking should support optional top-k and random-baseline filtering."""

    frame = make_classification_frame(rows=300)
    feature_columns = ["signal", "balance", "utilization", "segment"]

    ranking = rank_features_by_shap(
        frame,
        target_column="target",
        feature_columns=feature_columns,
        keep_top_k=None,
        filter_below_random=True,
        split_config=SplitConfig(random_seed=42),
        train_config=TrainConfig(
            num_threads=1,
            num_boost_round=30,
            early_stopping_rounds=10,
        ),
        top_p=0.2,
    )

    assert ranking.selected_columns
    assert "__random_feature_baseline__" not in ranking.selected_columns
    assert len(ranking.selected_columns) <= len(feature_columns)
    assert set(ranking.categorical_columns).issubset(ranking.selected_columns)
    assert ranking.evaluation_summary["importance_method"] == "shap_mean_abs"
    assert ranking.evaluation_summary["random_feature_mean_abs_shap"] is not None
    assert all(row["passed_random_filter"] for row in ranking.importance_rows)


def test_run_binary_experiment_returns_summary_payload() -> None:
    """The example entrypoint should compose the standalone workflow."""

    frame = make_classification_frame()
    result = run_binary_experiment(
        frame,
        target_column="target",
        id_columns=["customer_id"],
        split_config=SplitConfig(random_seed=42),
        train_config=TrainConfig(
            num_threads=1, num_boost_round=40, early_stopping_rounds=10
        ),
        top_p=0.2,
    )

    assert result["mode"] == "baseline"
    assert result["evaluation_summary"]["ppv_at_p"] >= 0.0
    assert result["top_p_summary"]["top_p_row_count"] > 0
