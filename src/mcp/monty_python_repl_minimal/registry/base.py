"""Shared configs and stored artifacts for the minimal registry package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import optuna

from ..privacy import safe_json_value


@dataclass(slots=True)
class DataReadConfig:
    """Configuration for loading a partial parquet slice.

    Args:
        s3_uri (str): Source parquet dataset URI or workspace path.
        label_col (str): Target column that must be included in the result.
        id_cols (list[str]): Identifier columns to retain alongside features.
        candidate_cols (list[str] | None): Optional explicit feature subset.
        partition_filters (dict[str, list[Any] | Any] | None): Optional parquet
            partition filters applied before scanning fragments.
        sample_n_rows (int | None): Optional maximum sampled row count.
        max_files (int | None): Optional maximum number of fragments to scan.
        random_seed (int): Seed used for fragment and row sampling.
    """

    s3_uri: str
    label_col: str
    id_cols: list[str] = field(default_factory=list)
    candidate_cols: list[str] | None = None
    partition_filters: dict[str, list[Any] | Any] | None = None
    sample_n_rows: int | None = None
    max_files: int | None = None
    random_seed: int = 42


@dataclass(slots=True)
class FeatureScreenConfig:
    """Configuration controlling quick feature screening.

    Args:
        max_missing_frac (float): Maximum allowed missing-value fraction.
        near_constant_thresh (float): Maximum allowed dominant-value rate.
        min_non_null (int): Minimum non-null row count required to keep a feature.
        top_k_univariate (int): Maximum number of features kept after ranking.
    """

    max_missing_frac: float = 0.98
    near_constant_thresh: float = 0.995
    min_non_null: int = 100
    top_k_univariate: int = 200


@dataclass(slots=True)
class SplitConfig:
    """Configuration for building a train/validation split.

    Args:
        method (str): Split strategy: ``random``, ``time``, or ``group``.
        valid_frac (float): Fraction of rows assigned to validation.
        random_seed (int): Seed used for deterministic splitting.
        time_col (str | None): Timestamp column used for time-based splits.
        group_col (str | None): Group column used for grouped splits.
    """

    method: str = "random"
    valid_frac: float = 0.2
    random_seed: int = 42
    time_col: str | None = None
    group_col: str | None = None


@dataclass(slots=True)
class TrainConfig:
    """Configuration for one LightGBM training run.

    Args:
        early_stopping_rounds (int): Early stopping patience on the validation set.
        num_boost_round (int): Maximum boosting iterations.
        num_threads (int): Number of LightGBM worker threads.
        seed (int): Random seed propagated to LightGBM.
    """

    early_stopping_rounds: int = 100
    num_boost_round: int = 2_000
    num_threads: int = 4
    seed: int = 42


@dataclass(slots=True)
class OptunaConfig:
    """Configuration for Optuna tuning.

    Args:
        n_trials (int): Maximum number of optimization trials.
        timeout (int | None): Optional wall-clock timeout in seconds.
        random_seed (int): Random seed used by the Optuna sampler.
    """

    n_trials: int = 30
    timeout: int | None = None
    random_seed: int = 42


@dataclass(slots=True)
class StoredFeatureSelectionReport:
    """Persisted feature-selection report artifact.

    Args:
        report_type (str): Name of the feature-selection workflow that produced it.
        target_column (str): Target column used while ranking or screening features.
        selected_columns (list[str]): Final selected feature list.
        categorical_columns (list[str]): Selected categorical feature subset.
        findings (list[dict[str, Any]]): Safe row-level feature findings.
        metrics (dict[str, Any]): Aggregate workflow metrics.
        warnings (list[str]): Non-fatal warnings emitted during selection.
        metadata (dict[str, Any]): Extra workflow metadata for later inspection.
    """

    report_type: str
    target_column: str
    selected_columns: list[str]
    categorical_columns: list[str]
    findings: list[dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1_000,
    ) -> dict[str, Any]:
        """Render a privacy-safe report summary for handle inspection.

        Args:
            max_items (int): Maximum items retained in preview lists.
            max_chars (int): Maximum retained characters for nested strings.

        Returns:
            dict[str, Any]: Compact JSON-friendly report summary payload.
        """

        return {
            "type": "StoredFeatureSelectionReport",
            "report_type": self.report_type,
            "target_column": self.target_column,
            "selected_columns": self.selected_columns[:max_items],
            "categorical_columns": self.categorical_columns[:max_items],
            "findings": safe_json_value(
                self.findings[:max_items],
                max_items=max_items,
                max_chars=max_chars,
            ),
            "metrics": safe_json_value(
                self.metrics,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "warnings": self.warnings[:max_items],
            "metadata": safe_json_value(
                self.metadata,
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


@dataclass(slots=True)
class StoredDataframeReport:
    """Persisted aggregate report for privacy-safe EDA and planning.

    Args:
        report_type (str): Report category identifier.
        title (str): Short human-readable report title.
        summary (str): Compact user-facing summary sentence.
        details (dict[str, Any]): Structured aggregate report payload.
        warnings (list[str]): Non-fatal warnings emitted during report creation.
        metadata (dict[str, Any]): Additional context for later inspection.
    """

    report_type: str
    title: str
    summary: str
    details: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1_000,
    ) -> dict[str, Any]:
        """Render a compact structured report summary.

        Args:
            max_items (int): Maximum items retained in preview lists.
            max_chars (int): Maximum retained characters for nested strings.

        Returns:
            dict[str, Any]: Safe summary suitable for ``inspect_handle``.
        """

        return {
            "type": "StoredDataframeReport",
            "report_type": self.report_type,
            "title": self.title,
            "summary": self.summary,
            "details": safe_json_value(
                self.details,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "warnings": self.warnings[:max_items],
            "metadata": safe_json_value(
                self.metadata,
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


@dataclass(slots=True)
class StoredFeatureEngineeringPipeline:
    """Persisted deterministic feature-engineering pipeline.

    Args:
        target_column (str | None): Optional target excluded during fitting.
        steps (list[dict[str, Any]]): Resolved pipeline steps and learned params.
        input_columns (list[str]): Input feature columns seen during fitting.
        output_columns (list[str]): Output columns produced after applying steps.
        warnings (list[str]): Non-fatal warnings emitted during fitting.
        metadata (dict[str, Any]): Additional pipeline metadata for inspection.
    """

    target_column: str | None
    steps: list[dict[str, Any]]
    input_columns: list[str]
    output_columns: list[str]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1_000,
    ) -> dict[str, Any]:
        """Render a compact fitted pipeline summary."""

        return {
            "type": "StoredFeatureEngineeringPipeline",
            "target_column": self.target_column,
            "input_columns": self.input_columns[:max_items],
            "output_columns": self.output_columns[:max_items],
            "steps": safe_json_value(
                self.steps[:max_items],
                max_items=max_items,
                max_chars=max_chars,
            ),
            "warnings": self.warnings[:max_items],
            "metadata": safe_json_value(
                self.metadata,
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


@dataclass(slots=True)
class StoredLightGBMStudy:
    """Persisted Optuna study artifact.

    Args:
        train_handle (str): Training dataframe handle used during tuning.
        target_column (str): Target column name optimized during tuning.
        feature_columns (list[str]): Feature columns included in the study.
        categorical_columns (list[str]): Feature columns treated as categorical.
        validation_handle (str | None): Optional explicit validation dataframe handle.
        best_params (dict[str, Any]): Best LightGBM parameter set found by Optuna.
        top_p (float): Top-percent cutoff used for PPV-style evaluation.
        study (optuna.study.Study): Backing Optuna study object.
        trial_rows (list[dict[str, Any]]): Safe trial summaries ordered by quality.
    """

    train_handle: str
    target_column: str
    feature_columns: list[str]
    categorical_columns: list[str]
    validation_handle: str | None
    best_params: dict[str, Any]
    top_p: float
    study: optuna.study.Study
    trial_rows: list[dict[str, Any]]

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1_000,
    ) -> dict[str, Any]:
        """Render a compact Optuna study summary.

        Args:
            max_items (int): Maximum items retained in preview lists.
            max_chars (int): Maximum retained characters for nested strings.

        Returns:
            dict[str, Any]: Safe study summary for handle inspection.
        """

        best_value = float(self.study.best_value) if self.study.best_trial else None
        best_iteration = None
        if self.study.best_trial:
            best_iteration = self.study.best_trial.user_attrs.get("best_iteration")
        return {
            "type": "StoredLightGBMStudy",
            "target_column": self.target_column,
            "feature_count": len(self.feature_columns),
            "categorical_count": len(self.categorical_columns),
            "trial_count": len(self.study.trials),
            "best_value": best_value,
            "best_iteration": best_iteration,
            "top_p": self.top_p,
            "best_params": safe_json_value(
                self.best_params,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "top_trials": safe_json_value(
                self.trial_rows[: min(max_items, 5)],
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


@dataclass(slots=True)
class StoredLightGBMModelArtifact:
    """Persisted fitted LightGBM model artifact.

    Args:
        booster (lgb.Booster): Fitted LightGBM booster.
        target_column (str): Target column used while fitting the model.
        feature_columns (list[str]): Final feature columns passed to LightGBM.
        categorical_columns (list[str]): Feature columns treated as categorical.
        best_params (dict[str, Any]): Best parameter set associated with the model.
        best_iteration (int): Best boosting iteration retained after early stopping.
        evaluation_summary (dict[str, Any]): Aggregate validation metrics.
        feature_importance_gain (dict[str, float]): Gain-based feature importance map.
    """

    booster: lgb.Booster
    target_column: str
    feature_columns: list[str]
    categorical_columns: list[str]
    best_params: dict[str, Any]
    best_iteration: int
    evaluation_summary: dict[str, Any]
    feature_importance_gain: dict[str, float]

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1_000,
    ) -> dict[str, Any]:
        """Render a compact fitted-model summary.

        Args:
            max_items (int): Maximum items retained in preview lists.
            max_chars (int): Maximum retained characters for nested strings.

        Returns:
            dict[str, Any]: Safe model summary for handle inspection.
        """

        top_importances = sorted(
            self.feature_importance_gain.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max_items]
        return {
            "type": "StoredLightGBMModelArtifact",
            "target_column": self.target_column,
            "feature_columns": self.feature_columns[:max_items],
            "categorical_columns": self.categorical_columns[:max_items],
            "best_iteration": self.best_iteration,
            "best_params": safe_json_value(
                self.best_params,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "evaluation_summary": safe_json_value(
                self.evaluation_summary,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "top_feature_importances": safe_json_value(
                [
                    {"feature": feature, "gain": gain}
                    for feature, gain in top_importances
                ],
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


__all__ = [
    "DataReadConfig",
    "FeatureScreenConfig",
    "OptunaConfig",
    "SplitConfig",
    "StoredDataframeReport",
    "StoredFeatureEngineeringPipeline",
    "StoredFeatureSelectionReport",
    "StoredLightGBMModelArtifact",
    "StoredLightGBMStudy",
    "TrainConfig",
]
