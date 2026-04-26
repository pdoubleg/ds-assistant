"""Small script-style entrypoint for standalone ds experiments."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from .config import OptunaConfig, SplitConfig, TrainConfig
from .eda import summarize_dataframe, summarize_target
from .hpo import tune_lightgbm_pipeline
from .io import read_csv, read_parquet_fragment
from .metrics import summarize_top_p_predictions
from .modeling import (
    build_train_valid_frames,
    fit_lightgbm_binary,
    resolve_feature_columns,
    score_dataframe,
)
from .pipelines import build_feature_pipeline, fit_transform_features


def _build_default_pipeline(
    dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> Any:
    """Build a simple default feature pipeline for experiments.

    Args:
        dataframe: Source dataframe used to infer numeric and categorical columns.
        feature_columns: Feature columns included in the experiment.

    Returns:
        Sklearn feature pipeline.
    """

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]
    # Keep the default entrypoint intentionally simple so notebook users can swap
    # in their own pipeline object without fighting hidden behavior.
    return build_feature_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        remainder="drop",
    )


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


def run_binary_experiment(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    id_columns: list[str] | None = None,
    perform_hpo: bool = False,
    split_config: SplitConfig | None = None,
    train_config: TrainConfig | None = None,
    optuna_config: OptunaConfig | None = None,
    top_p: float = 0.05,
) -> dict[str, Any]:
    """Run a simple baseline or HPO-backed binary classification experiment.

    Args:
        dataframe: Modeling dataframe.
        target_column: Binary target column.
        id_columns: Optional identifier columns excluded from automatic feature
            resolution.
        perform_hpo: Whether to run Optuna instead of a baseline fit.
        split_config: Split configuration used when building train/validation frames.
        train_config: Training configuration override.
        optuna_config: Optuna tuning configuration override.
        top_p: Fraction retained for PPV-style evaluation.

    Returns:
        Compact experiment summary payload.
    """

    feature_columns = resolve_feature_columns(
        dataframe,
        target_column=target_column,
        id_columns=id_columns,
    )
    split = build_train_valid_frames(
        dataframe,
        target_column=target_column,
        feature_columns=feature_columns,
        split_config=split_config,
    )
    pipeline = _build_default_pipeline(dataframe, feature_columns=feature_columns)

    if perform_hpo:
        hpo_result = tune_lightgbm_pipeline(
            dataframe,
            target_column=target_column,
            pipeline=pipeline,
            feature_columns=feature_columns,
            id_columns=id_columns,
            split_config=split_config,
            train_config=train_config,
            optuna_config=optuna_config,
            top_p=top_p,
        )
        transformed_valid = _ensure_dataframe(
            hpo_result.best_pipeline.transform(split.valid_df[feature_columns])
        )
        transformed_valid.index = split.valid_df.index
        valid_frame = transformed_valid.copy()
        valid_frame[target_column] = split.valid_df[target_column].values
        scored_valid = score_dataframe(
            valid_frame, hpo_result.best_result, score_column_name="pred_score"
        )
        top_summary = summarize_top_p_predictions(
            scored_valid[target_column].values,
            scored_valid["pred_score"].values,
            p=top_p,
        )
        return {
            "mode": "hpo",
            "dataframe_summary": summarize_dataframe(dataframe),
            "target_summary": summarize_target(dataframe, target_column),
            "feature_columns": feature_columns,
            "best_value": hpo_result.best_value,
            "best_pipeline_params": hpo_result.best_pipeline_params,
            "best_lightgbm_params": hpo_result.best_lightgbm_params,
            "evaluation_summary": hpo_result.best_result.evaluation_summary,
            "top_p_summary": top_summary,
            "trial_count": len(hpo_result.trial_rows),
        }

    transformed_train, transformed_valid, _ = fit_transform_features(
        pipeline,
        split.train_df,
        valid_df=split.valid_df,
        feature_columns=feature_columns,
    )
    if transformed_valid is None:
        raise ValueError("Validation data is required for scoring.")

    train_frame = transformed_train.copy()
    train_frame[target_column] = split.train_df[target_column].values
    valid_frame = transformed_valid.copy()
    valid_frame[target_column] = split.valid_df[target_column].values

    baseline_result = fit_lightgbm_binary(
        train_frame,
        valid_frame,
        target_column=target_column,
        feature_columns=[str(column) for column in transformed_train.columns],
        train_config=train_config,
        top_p=top_p,
    )
    scored_valid = score_dataframe(
        valid_frame, baseline_result, score_column_name="pred_score"
    )
    top_summary = summarize_top_p_predictions(
        scored_valid[target_column].values,
        scored_valid["pred_score"].values,
        p=top_p,
    )
    return {
        "mode": "baseline",
        "dataframe_summary": summarize_dataframe(dataframe),
        "target_summary": summarize_target(dataframe, target_column),
        "feature_columns": feature_columns,
        "evaluation_summary": baseline_result.evaluation_summary,
        "top_p_summary": top_summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the example entrypoint.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Run a standalone ds experiment.")
    parser.add_argument("path", help="CSV or parquet path to read.")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--id-columns", nargs="*", default=[])
    parser.add_argument("--top-p", type=float, default=0.05)
    parser.add_argument("--run-hpo", action="store_true")
    parser.add_argument("--n-trials", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the example standalone experiment entrypoint.

    Args:
        argv: Optional explicit CLI argv.

    Returns:
        Process exit code.
    """

    args = _build_parser().parse_args(argv)
    if args.format == "csv":
        dataframe = read_csv(args.path)
    else:
        dataframe = read_parquet_fragment(args.path)

    result = run_binary_experiment(
        dataframe,
        target_column=args.target_column,
        id_columns=list(args.id_columns),
        perform_hpo=bool(args.run_hpo),
        optuna_config=OptunaConfig(n_trials=int(args.n_trials)),
        top_p=float(args.top_p),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
