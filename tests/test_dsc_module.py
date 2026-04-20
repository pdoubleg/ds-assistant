"""Focused tests for the standalone DSC module."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from src.dsc.artifacts import RunArtifactStore
from src.dsc.cli import main
from src.dsc.config import RunConfig
from src.dsc.features import screen_features
from src.dsc.io import (
    build_training_frame,
    inspect_dataset,
    load_csv_sample,
    load_parquet_sample,
)
from src.dsc.modeling import ppv_at_top_p, summarize_binary_predictions, tune_pipeline
from src.dsc.modeling.pipeline_runner import fit_pipeline
from src.dsc.pipeline import (
    PipelineSpec,
    PipelineStageSpec,
    StageFactory,
    TunableParam,
    apply_trial_params,
    build_default_binary_pipeline_spec,
    describe_pipeline_params,
    register_stage_factory,
    resolve_tunable_params,
)
from src.dsc.split import make_holdout_split


def make_classification_frame(rows: int = 240) -> pd.DataFrame:
    """Return a small synthetic binary-classification dataframe."""

    rng = np.random.default_rng(42)
    numeric_signal = rng.normal(loc=0.0, scale=1.0, size=rows)
    balance = rng.normal(loc=100.0, scale=20.0, size=rows)
    segment = np.where(numeric_signal > 0, "A", "B")
    noise = rng.normal(size=rows)
    target = ((numeric_signal + noise * 0.25) > 0.15).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(rows),
            "signal": numeric_signal,
            "balance": balance,
            "segment": pd.Series(segment, dtype="category"),
            "target": target,
        }
    )
    frame.loc[frame.index[::11], "balance"] = np.nan
    frame.loc[frame.index[::13], "segment"] = None
    return frame


def make_run_config_dict(input_path: str, artifact_root: str) -> dict[str, object]:
    """Return a small JSON config payload for CLI tests."""

    return {
        "run_name": "cli_smoke",
        "data": {
            "input_path": input_path,
            "file_format": "csv",
            "target_column": "target",
            "id_columns": ["customer_id"],
        },
        "feature_screen": {
            "top_k_univariate": 3,
            "min_non_null": 10,
        },
        "optuna": {"n_trials": 2, "random_seed": 42},
        "training": {"top_p": 0.05},
        "artifacts": {"root_dir": artifact_root, "save_predictions": True},
        "pipeline": build_default_binary_pipeline_spec().to_dict(),
    }


def test_csv_sampling_and_inspection(tmp_path: Path) -> None:
    """CSV inspection and sampling should preserve the requested columns."""

    frame = make_classification_frame()
    csv_path = tmp_path / "train.csv"
    frame.to_csv(csv_path, index=False)

    inspection = inspect_dataset(str(csv_path), file_format="csv", sample_rows=20)
    sampled = load_csv_sample(str(csv_path), nrows=15, usecols=["target", "signal"])

    assert inspection.file_format == "csv"
    assert inspection.column_count == frame.shape[1]
    assert sampled.shape == (15, 2)
    assert list(sampled.columns) == ["target", "signal"]


def test_parquet_sampling_supports_local_paths(tmp_path: Path) -> None:
    """Parquet sampling should work for local datasets when pyarrow is available."""

    pytest.importorskip("pyarrow")
    frame = make_classification_frame()
    parquet_path = tmp_path / "train.parquet"
    frame.to_parquet(parquet_path, index=False)

    sampled = load_parquet_sample(
        str(parquet_path),
        label_col="target",
        id_cols=["customer_id"],
        candidate_cols=["signal", "segment"],
        sample_n_rows=40,
        max_files=1,
    )

    assert len(sampled) <= 40
    assert set(sampled.columns) == {"target", "customer_id", "signal", "segment"}


def test_parameter_manifest_and_custom_stage_registration() -> None:
    """New stage factories should expose tunable params in the shared manifest."""

    class MultiplySignalTransformer(BaseEstimator, TransformerMixin):
        """Multiply the signal column by a configurable factor."""

        def __init__(self, factor: float = 1.0) -> None:
            self.factor = factor

        def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MultiplySignalTransformer":
            del y
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            updated = X.copy()
            updated["signal_scaled"] = updated["signal"] * self.factor
            return updated

    register_stage_factory(
        StageFactory(
            kind="multiply_signal",
            build_estimator=lambda config: MultiplySignalTransformer(
                factor=float(config.get("factor", 1.0))
            ),
            build_tunable_params=lambda stage_name, config: [
                TunableParam(
                    path="factor",
                    estimator_param=f"{stage_name}__factor",
                    suggestion_kind="float",
                    description="Scale factor for the synthetic test transformer.",
                    default=config.get("factor", 1.0),
                    low=0.5,
                    high=2.0,
                    step=0.25,
                )
            ],
        )
    )
    stage = PipelineStageSpec(
        name="scale_signal",
        kind="multiply_signal",
        config={"factor": 1.0},
    )
    stage.tunable_params = resolve_tunable_params(stage)
    pipeline_spec = PipelineSpec(stages=[stage])
    manifest = describe_pipeline_params(pipeline_spec)

    assert manifest[0]["path"] == "factor"
    assert manifest[0]["estimator_param"] == "scale_signal__factor"


def test_ppv_metric_and_feature_screening() -> None:
    """PPV summaries and feature screening should produce actionable outputs."""

    frame = make_classification_frame()
    scores = frame["signal"].rank(pct=True).to_numpy()
    summary = summarize_binary_predictions(frame["target"], scores, top_p=0.05)
    report = screen_features(
        frame,
        target_column="target",
        config=RunConfig.from_dict(
            {
                "run_name": "test",
                "data": {
                    "input_path": "unused.csv",
                    "target_column": "target",
                },
                "pipeline": build_default_binary_pipeline_spec().to_dict(),
            }
        ).feature_screen,
        id_columns=["customer_id"],
    )

    assert ppv_at_top_p(frame["target"], scores, p=0.05) >= 0.0
    assert summary["ppv_at_5"] >= 0.0
    assert report.selected_columns
    assert "signal" in report.selected_columns


def test_lightgbm_pipeline_fit_and_optuna_tuning(tmp_path: Path) -> None:
    """Baseline fitting and Optuna tuning should both complete on categorical data."""

    frame = make_classification_frame()
    modeling_frame = build_training_frame(
        frame,
        target_column="target",
        id_columns=["customer_id"],
    )
    train_frame, valid_frame, _test_frame, manifest = make_holdout_split(
        modeling_frame,
        target_column="target",
        split_config=RunConfig.from_dict(
            {
                "run_name": "test",
                "data": {
                    "input_path": "unused.csv",
                    "target_column": "target",
                },
                "pipeline": build_default_binary_pipeline_spec().to_dict(),
            }
        ).split,
    )
    pipeline_spec = build_default_binary_pipeline_spec(random_state=42, n_jobs=1)
    baseline_result = fit_pipeline(
        pipeline_spec,
        train_frame,
        valid_frame,
        target_column="target",
        id_columns=["customer_id"],
        top_p=0.05,
    )
    hpo_result = tune_pipeline(
        build_default_binary_pipeline_spec(random_state=42, n_jobs=1),
        train_frame,
        valid_frame,
        target_column="target",
        id_columns=["customer_id"],
        top_p=0.05,
        n_trials=2,
        random_seed=42,
    )

    assert manifest.row_counts["train"] > 0
    assert baseline_result.metrics["ppv_at_5"] >= 0.0
    assert baseline_result.parameter_manifest
    assert hpo_result.best_params
    assert hpo_result.best_result.metrics["ppv_at_5"] >= 0.0

    store = RunArtifactStore(tmp_path / "artifacts", "training")
    model_path = store.save_joblib("model.joblib", baseline_result.pipeline)
    loaded = joblib.load(model_path)
    assert hasattr(loaded, "predict")


def test_apply_trial_params_updates_nested_configs() -> None:
    """Optuna parameter mapping should update nested stage config paths."""

    import optuna

    stage = PipelineStageSpec(
        name="model",
        kind="lightgbm_classifier",
        config={"learning_rate": 0.05},
        tunable_params=[
            TunableParam(
                path="learning_rate",
                estimator_param="model__learning_rate",
                suggestion_kind="float",
                description="Learning rate.",
                default=0.05,
                low=0.01,
                high=0.1,
            )
        ],
    )
    pipeline_spec = PipelineSpec(stages=[stage])

    def objective(trial: optuna.Trial) -> float:
        updated, params = apply_trial_params(pipeline_spec, trial)
        assert "model__learning_rate" in params
        assert updated.stages[0].config["learning_rate"] == params["model__learning_rate"]
        return float(params["model__learning_rate"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)


def test_cli_train_baseline_smoke(tmp_path: Path) -> None:
    """The CLI should run an end-to-end baseline workflow and persist artifacts."""

    frame = make_classification_frame()
    csv_path = tmp_path / "train.csv"
    config_path = tmp_path / "config.json"
    frame.to_csv(csv_path, index=False)
    config_path.write_text(
        json.dumps(
            make_run_config_dict(str(csv_path), str(tmp_path / "artifacts")),
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = main(["train-baseline", "--config", str(config_path)])

    assert exit_code == 0
    artifact_root = tmp_path / "artifacts"
    run_dirs = sorted(path for path in artifact_root.iterdir() if path.is_dir())
    latest = run_dirs[-1]
    assert (latest / "model.joblib").exists()
    assert (latest / "metrics.json").exists()
    assert (latest / "pipeline_params.json").exists()

