"""Shared configuration and result types for the standalone ds package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import optuna
import pandas as pd


PipelineParamKind = Literal["categorical", "float", "int", "bool"]


@dataclass(slots=True)
class DataReadConfig:
    """Configuration for reading a fragment of a parquet dataset.

    Args:
        uri: Local path or remote URI such as ``s3://bucket/path``.
        columns: Optional projected column subset.
        partition_filters: Optional equality or membership filters.
        sample_n_rows: Optional maximum row count to retain.
        max_fragments: Optional maximum number of parquet fragments to scan.
        random_seed: Random seed used for deterministic fragment and row sampling.
    """

    uri: str
    columns: list[str] | None = None
    partition_filters: dict[str, list[Any] | Any] | None = None
    sample_n_rows: int | None = None
    max_fragments: int | None = None
    random_seed: int = 42


@dataclass(slots=True)
class FeatureScreenConfig:
    """Configuration controlling the initial feature screening pass.

    Args:
        max_missing_frac: Maximum tolerated missing-value rate.
        near_constant_thresh: Threshold for flagging near-constant columns.
        min_non_null: Minimum required non-null row count.
        top_k_univariate: Maximum retained feature count after univariate ranking.
    """

    max_missing_frac: float = 0.98
    near_constant_thresh: float = 0.995
    min_non_null: int = 100
    top_k_univariate: int = 200


@dataclass(slots=True)
class BatchedFeatureScreenConfig:
    """Configuration for memory-bounded feature screening.

    Args:
        batch_size: Maximum number of feature columns loaded for each screening
            batch.
        output_dir: Optional directory used to persist intermediate batch
            findings.
        persisted_file_format: File format used for persisted batch findings.
        correlation_threshold: Absolute Pearson correlation threshold used for
            numeric de-correlation.
        correlation_batch_size: Maximum number of numeric columns loaded in each
            block during cross-batch correlation analysis.
        cleanup_batch_frames: Whether to run explicit cleanup after each loaded
            batch frame.
        random_seed: Seed propagated to deterministic readers when applicable.
        top_k_univariate: Maximum retained feature count after merging all
            batch-level screening rows.
        screen_config: Per-batch descriptive and univariate screening settings.

    Example:
        >>> config = BatchedFeatureScreenConfig(batch_size=10, correlation_threshold=0.9)
    """

    batch_size: int = 50
    output_dir: Path | None = None
    persisted_file_format: Literal["json", "csv"] = "json"
    correlation_threshold: float = 0.95
    correlation_batch_size: int = 50
    cleanup_batch_frames: bool = True
    random_seed: int = 42
    top_k_univariate: int | None = None
    screen_config: FeatureScreenConfig = field(default_factory=FeatureScreenConfig)


@dataclass(slots=True)
class SplitConfig:
    """Configuration for creating a train and validation split.

    Args:
        valid_frac: Fraction of rows assigned to validation.
        random_seed: Random seed used for deterministic splitting.
        stratify: Whether to stratify the split on the binary target column when
            possible.
    """

    valid_frac: float = 0.2
    random_seed: int = 42
    stratify: bool = True


@dataclass(slots=True)
class TrainConfig:
    """Configuration for one LightGBM training run.

    Args:
        early_stopping_rounds: Early stopping patience measured in boosting rounds.
        num_boost_round: Maximum number of boosting rounds.
        num_threads: Number of LightGBM worker threads.
        seed: Random seed propagated to LightGBM.
    """

    early_stopping_rounds: int = 100
    num_boost_round: int = 2_000
    num_threads: int = 4
    seed: int = 42


@dataclass(slots=True)
class OptunaConfig:
    """Configuration for Optuna tuning.

    Args:
        n_trials: Maximum number of optimization trials.
        timeout: Optional wall-clock timeout in seconds.
        random_seed: Random seed used by the Optuna sampler.
    """

    n_trials: int = 30
    timeout: int | None = None
    random_seed: int = 42


@dataclass(slots=True)
class ParquetReadMetadata:
    """Metadata describing a parquet fragment read.

    Args:
        source_uri: Dataset URI that was read.
        resolved_columns: Ordered columns projected into the output frame.
        available_fragment_count: Total fragment count after applying filters.
        scanned_fragment_count: Number of fragments actually scanned.
        row_count: Final dataframe row count.
    """

    source_uri: str
    resolved_columns: list[str]
    available_fragment_count: int
    scanned_fragment_count: int
    row_count: int


@dataclass(slots=True)
class TrainValidSplit:
    """Explicit train and validation split payload.

    Args:
        train_df: Training dataframe.
        valid_df: Validation dataframe.
        train_indices: Original dataframe indices assigned to training.
        valid_indices: Original dataframe indices assigned to validation.
        split_config: Split configuration used to build the split.
    """

    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_indices: list[Any]
    valid_indices: list[Any]
    split_config: SplitConfig


@dataclass(slots=True)
class FeatureScreenResult:
    """Result payload for the descriptive screening stage.

    Args:
        filtered_df: Reduced dataframe containing the selected feature set.
        selected_columns: Retained feature columns after screening.
        dropped_columns: Dropped feature columns after screening.
        categorical_columns: Selected columns treated as categorical.
        findings: Per-column screening diagnostics.
        metrics: Aggregate workflow metrics.
        warnings: Non-fatal warnings emitted during screening.
    """

    filtered_df: pd.DataFrame
    selected_columns: list[str]
    dropped_columns: list[str]
    categorical_columns: list[str]
    findings: list[dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BatchedFeatureScreenResult:
    """Result payload for memory-bounded feature screening.

    Args:
        selected_columns: Final retained feature columns after global screening
            and numeric de-correlation.
        dropped_columns: Final dropped feature columns.
        categorical_columns: Selected columns inferred as categorical.
        batch_rows: Compact diagnostics for each processed feature batch.
        finding_rows: In-memory per-feature screening findings.
        finding_paths: Persisted per-batch finding files.
        correlation_pair_rows: Correlated numeric-pair diagnostics.
        metrics: Aggregate workflow metrics.
        warnings: Non-fatal warnings emitted during screening.
    """

    selected_columns: list[str]
    dropped_columns: list[str]
    categorical_columns: list[str]
    batch_rows: list[dict[str, Any]]
    finding_rows: list[dict[str, Any]]
    finding_paths: list[Path] = field(default_factory=list)
    correlation_pair_rows: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CorrelationAnalysisResult:
    """Result payload for the correlation analysis stage.

    Args:
        filtered_df: Reduced dataframe after deterministic drops.
        selected_columns: Feature columns retained after correlation analysis.
        dropped_columns: Feature columns proposed for removal.
        pair_rows: Correlated feature-pair diagnostics.
        warnings: Non-fatal warnings emitted during analysis.
    """

    filtered_df: pd.DataFrame
    selected_columns: list[str]
    dropped_columns: list[str]
    pair_rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeatureRankingResult:
    """Result payload for LightGBM-based feature ranking.

    Args:
        selected_columns: Retained top-ranked feature columns.
        categorical_columns: Columns treated as categorical during ranking.
        importance_rows: Ordered LightGBM importance rows.
        evaluation_summary: Validation metrics for the ranking run.
    """

    selected_columns: list[str]
    categorical_columns: list[str]
    importance_rows: list[dict[str, Any]]
    evaluation_summary: dict[str, Any]


@dataclass(slots=True)
class FeatureSubsetRankingResult:
    """Result payload for batched subset ranking.

    Args:
        filtered_df: Reduced dataframe built from the selected union of columns.
        selected_columns: Union of the retained subset winners.
        categorical_columns: Selected columns treated as categorical.
        subset_rows: Ordered subset-level ranking diagnostics.
        warnings: Non-fatal warnings emitted during ranking.
    """

    filtered_df: pd.DataFrame
    selected_columns: list[str]
    categorical_columns: list[str]
    subset_rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LightGBMTrainingResult:
    """Compact result payload for one fitted LightGBM model.

    Args:
        booster: Fitted LightGBM booster.
        target_column: Target column used for training.
        feature_columns: Final feature columns seen by the booster.
        categorical_columns: Feature columns treated as categorical.
        best_params: Final LightGBM parameters used for the fit.
        best_iteration: Best iteration retained after early stopping.
        evaluation_summary: Validation metrics for the fitted model.
        feature_importance_gain: Gain-based feature importance mapping.
        top_p: Top-percent cutoff used for PPV-style evaluation.
    """

    booster: lgb.Booster
    target_column: str
    feature_columns: list[str]
    categorical_columns: list[str]
    best_params: dict[str, Any]
    best_iteration: int
    evaluation_summary: dict[str, Any]
    feature_importance_gain: dict[str, float]
    top_p: float


@dataclass(slots=True)
class PipelineSearchSpace:
    """Definition of a tunable sklearn pipeline parameter.

    Args:
        estimator_param: Fully qualified sklearn parameter name.
        suggestion_kind: Optuna suggestion kind used for the parameter.
        default: Default value used when the parameter is not tuned.
        choices: Categorical choices when ``suggestion_kind`` is categorical/bool.
        low: Lower bound for numeric suggestions.
        high: Upper bound for numeric suggestions.
        step: Optional additive step for integer/float suggestions.
        log: Whether to sample on a log scale for numeric suggestions.
        description: Human-readable description for notebooks and manifests.
    """

    estimator_param: str
    suggestion_kind: PipelineParamKind
    default: Any
    choices: list[Any] | None = None
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    description: str = ""


@dataclass(slots=True)
class HpoResult:
    """Result payload for Optuna tuning over a pipeline and LightGBM.

    Args:
        study: Backing Optuna study object.
        best_params: Flattened best parameter mapping across pipeline and model.
        best_value: Best objective value observed during tuning.
        best_pipeline_params: Best sklearn pipeline parameter mapping.
        best_lightgbm_params: Best LightGBM parameter mapping.
        trial_rows: Compact per-trial summary rows.
        split_config: Split configuration used when an internal split was built.
        split_indices: Explicit train and validation indices for reproducibility.
        best_pipeline: Fitted sklearn pipeline for the best trial.
        best_result: Training result refit for the best trial on the retained split.
    """

    study: optuna.study.Study
    best_params: dict[str, Any]
    best_value: float
    best_pipeline_params: dict[str, Any]
    best_lightgbm_params: dict[str, Any]
    trial_rows: list[dict[str, Any]]
    split_config: SplitConfig | None
    split_indices: dict[str, list[Any]]
    best_pipeline: Any
    best_result: LightGBMTrainingResult


@dataclass(slots=True)
class ErrorAnalysisResult:
    """Result payload for PPV-focused false-positive analysis.

    Args:
        summary: Human-readable analysis summary.
        top_p_summary: Aggregate top-p prediction metrics.
        analyzed_columns: Feature columns included in the analysis.
        false_positive_count: Number of false positives in the retained top-p slice.
        true_positive_count: Number of true positives in the retained top-p slice.
        numeric_findings: Numeric column diagnostics sorted by pattern strength.
        categorical_findings: Categorical column diagnostics sorted by pattern
            strength.
    """

    summary: str
    top_p_summary: dict[str, Any]
    analyzed_columns: list[str]
    false_positive_count: int
    true_positive_count: int
    numeric_findings: list[dict[str, Any]]
    categorical_findings: list[dict[str, Any]]


@dataclass(slots=True)
class SavedFigure:
    """Metadata describing a saved matplotlib figure.

    Args:
        path: Absolute path to the saved figure.
        figure_type: Logical plot name.
        metadata: Additional chart metadata.
    """

    path: Path
    figure_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BatchedFeatureScreenConfig",
    "BatchedFeatureScreenResult",
    "CorrelationAnalysisResult",
    "DataReadConfig",
    "ErrorAnalysisResult",
    "FeatureRankingResult",
    "FeatureScreenConfig",
    "FeatureScreenResult",
    "FeatureSubsetRankingResult",
    "HpoResult",
    "LightGBMTrainingResult",
    "OptunaConfig",
    "ParquetReadMetadata",
    "PipelineSearchSpace",
    "SavedFigure",
    "SplitConfig",
    "TrainConfig",
    "TrainValidSplit",
]
