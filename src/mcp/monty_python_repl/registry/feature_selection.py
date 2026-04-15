"""Diagnostic feature selection helpers for the Monty Python REPL."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..filesystem import HostWorkspaceOSAccess
from ..core.registry import ObjectStore, ToolCollection, tool
from ..support.metrics import (
    evaluate_feature_subset as evaluate_subset_metrics,
    rank_feature_target_metrics,
    rank_lightgbm_importance,
)
from ..core.registry import safe_json_value

_SUPPORTED_TARGET_METHODS = {"mutual_info", "f_score", "chi2"}
_SUPPORTED_REDUNDANCY_METHODS = {"correlation"}
_SUPPORTED_IMPORTANCE_METHODS = {"lightgbm"}


@dataclass(slots=True)
class StoredFeatureSelectionReport:
    """Persisted feature-selection analysis artifact for Monty.

    Args:
        report_type (str): High-level analysis family such as `summary` or
            `target_metrics`.
        method (str): Specific method used to produce the findings.
        feature_columns (list[str]): Candidate feature columns considered.
        target_column (str | None): Optional target column used by the analysis.
        findings (list[dict[str, Any]]): Main report rows shown to the caller.
        metrics (dict[str, Any]): Optional evaluation metrics payload.
        warnings (list[str]): Caveats that the caller should consider.
        metadata (dict[str, Any]): Additional analysis metadata.
    """

    report_type: str
    method: str
    feature_columns: list[str]
    target_column: str | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection.

        Args:
            max_items (int): Maximum finding rows to preview.
            max_chars (int): Maximum nested string length to retain.

        Returns:
            dict[str, Any]: Summary payload suitable for `inspect_handle`.
        """
        return {
            "type": "StoredFeatureSelectionReport",
            "report_type": self.report_type,
            "method": self.method,
            "target_column": self.target_column,
            "feature_count": len(self.feature_columns),
            "finding_count": len(self.findings),
            "feature_columns": self.feature_columns[:max_items],
            "findings": [
                safe_json_value(
                    finding,
                    max_items=max_items,
                    max_chars=max_chars,
                )
                for finding in self.findings[:max_items]
            ],
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


def _string_list(values: Sequence[Any], *, field_name: str) -> list[str]:
    """Coerce a sequence of values into a list of strings.

    Args:
        values (Sequence[Any]): Raw values to coerce.
        field_name (str): Field name used in validation messages.

    Returns:
        list[str]: Coerced string list.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of strings, not a string.")
    return [str(value) for value in values]


def _require_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise an error when required columns are missing from a dataframe.

    Args:
        dataframe (pd.DataFrame): Dataframe to validate.
        columns (Sequence[str]): Required column names.
    """
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}.")


class FeatureSelectionCollection(ToolCollection):
    """Diagnostic, report-oriented feature selection helpers for tabular data."""

    name = "feature_selection"
    description = (
        "Compute descriptive summaries, target-aware rankings, redundancy checks, "
        "subset evaluation metrics, and LightGBM importance diagnostics for "
        "stored dataframe handles."
    )

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: ObjectStore,
    ) -> None:
        """Initialize feature selection helpers.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (ObjectStore): Shared handle store.
        """
        self._os_access = os_access
        self._object_store = object_store

    def _resolve_host_path(self, path: str) -> Path:
        """Resolve a virtual or relative path into the host workspace.

        Args:
            path (str): Relative or `/workspace` path.

        Returns:
            Path: Resolved host path.
        """
        return self._os_access.to_host_path(PurePosixPath(path))

    def _get_dataframe(self, dataframe_handle: str) -> pd.DataFrame:
        """Fetch a dataframe from the shared object store.

        Args:
            dataframe_handle (str): Dataframe handle.

        Returns:
            pd.DataFrame: Stored dataframe.
        """
        return self._object_store.get(dataframe_handle, expected_type=pd.DataFrame)

    def _get_report(self, report_handle: str) -> StoredFeatureSelectionReport:
        """Fetch a stored report from the shared object store.

        Args:
            report_handle (str): Report handle.

        Returns:
            StoredFeatureSelectionReport: Stored report artifact.
        """
        return self._object_store.get(
            report_handle,
            expected_type=StoredFeatureSelectionReport,
        )

    def _split_features_and_target(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        feature_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Return a feature frame and optional target series.

        Args:
            dataframe_handle (str): Input dataframe handle.
            target_column (str | None): Optional target column to exclude.
            feature_columns (list[str] | None): Optional explicit feature list.

        Returns:
            tuple[pd.DataFrame, pd.Series | None]: Feature frame and target series.
        """
        dataframe = self._get_dataframe(dataframe_handle)
        target_series = None
        if target_column is not None:
            if target_column not in dataframe.columns:
                raise ValueError(
                    f"Target column {target_column!r} was not found in the dataframe."
                )
            target_series = dataframe[target_column].copy()
            dataframe = dataframe.drop(columns=[target_column])

        if feature_columns is not None:
            _require_columns(dataframe, feature_columns)
            dataframe = dataframe[feature_columns].copy()
        else:
            dataframe = dataframe.copy()
        return dataframe, target_series

    @tool
    def summarize_feature_candidates(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        feature_columns: list[str] | None = None,
        near_constant_threshold: float = 0.95,
    ) -> str:
        """Summarize candidate features without using target-aware statistics.

        Args:
            dataframe_handle (str): Handle pointing to the candidate dataframe.
            target_column (str | None): Optional target column to exclude.
            feature_columns (list[str] | None): Optional explicit features to summarize.
            near_constant_threshold (float): Threshold used to flag near-constant
                columns based on the dominant non-null value frequency.

        Returns:
            str: Handle for the stored summary report.

        Examples:
            summary_handle = summarize_feature_candidates(
                df_handle,
                target_column="target",
            )
        """
        feature_frame, _ = self._split_features_and_target(
            dataframe_handle,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        findings: list[dict[str, Any]] = []
        warnings: list[str] = []

        for column in feature_frame.columns:
            series = feature_frame[column]
            non_null = series.dropna()
            dominant_rate = (
                float(non_null.value_counts(normalize=True, dropna=False).iloc[0])
                if not non_null.empty
                else None
            )
            findings.append(
                {
                    "feature": str(column),
                    "dtype": str(series.dtype),
                    "missing_count": int(series.isna().sum()),
                    "missing_rate": float(series.isna().mean()),
                    "unique_count": int(non_null.nunique(dropna=True)),
                    "dominant_rate": dominant_rate,
                    "zero_variance": bool(non_null.nunique(dropna=True) <= 1),
                    "near_constant": bool(
                        dominant_rate is not None
                        and dominant_rate >= near_constant_threshold
                    ),
                    "non_null_count": int(non_null.shape[0]),
                }
            )

        if target_column is None:
            warnings.append(
                "This report is descriptive only and does not use the target."
            )

        report = StoredFeatureSelectionReport(
            report_type="summary",
            method="descriptive_summary",
            feature_columns=[str(column) for column in feature_frame.columns],
            target_column=target_column,
            findings=findings,
            warnings=warnings,
            metadata={
                "near_constant_threshold": float(near_constant_threshold),
            },
        )
        return self._object_store.put(report, prefix="fs")

    @tool
    def compute_feature_target_metrics(
        self,
        dataframe_handle: str,
        target_column: str,
        *,
        feature_columns: list[str] | None = None,
        method: str = "mutual_info",
        random_state: int = 0,
    ) -> str:
        """Rank features against a target with a univariate scoring method.

        Args:
            dataframe_handle (str): Handle pointing to the candidate dataframe.
            target_column (str): Target column used by the ranking method.
            feature_columns (list[str] | None): Optional explicit feature list.
            method (str): One of `mutual_info`, `f_score`, or `chi2`.
            random_state (int): Random seed for stochastic ranking methods.

        Returns:
            str: Handle for the stored ranking report.

        Examples:
            ranking_handle = compute_feature_target_metrics(
                df_handle,
                "target",
                method="mutual_info",
            )
        """
        if method not in _SUPPORTED_TARGET_METHODS:
            supported = ", ".join(sorted(_SUPPORTED_TARGET_METHODS))
            raise ValueError(
                f"Unsupported target metric method {method!r}. Supported values: {supported}."
            )

        feature_frame, target_series = self._split_features_and_target(
            dataframe_handle,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        if target_series is None:
            raise ValueError("A target column is required for target-aware metrics.")

        findings, warnings = rank_feature_target_metrics(
            feature_frame,
            target_series,
            method=method,
            random_state=random_state,
        )
        report = StoredFeatureSelectionReport(
            report_type="target_metrics",
            method=method,
            feature_columns=[str(column) for column in feature_frame.columns],
            target_column=target_column,
            findings=findings,
            warnings=warnings,
            metadata={"random_state": int(random_state)},
        )
        return self._object_store.put(report, prefix="fs")

    @tool
    def compute_feature_redundancy_metrics(
        self,
        dataframe_handle: str,
        *,
        feature_columns: list[str] | None = None,
        method: str = "correlation",
        threshold: float = 0.9,
        max_pairs: int = 50,
    ) -> str:
        """Summarize redundancy among candidate features.

        Args:
            dataframe_handle (str): Handle pointing to the candidate dataframe.
            feature_columns (list[str] | None): Optional explicit feature list.
            method (str): Currently only `correlation` is supported.
            threshold (float): Absolute-correlation threshold used to flag pairs.
            max_pairs (int): Maximum number of flagged pairs to retain.

        Returns:
            str: Handle for the stored redundancy report.

        Examples:
            redundancy_handle = compute_feature_redundancy_metrics(
                df_handle,
                method="correlation",
            )
        """
        if method not in _SUPPORTED_REDUNDANCY_METHODS:
            supported = ", ".join(sorted(_SUPPORTED_REDUNDANCY_METHODS))
            raise ValueError(
                f"Unsupported redundancy method {method!r}. Supported values: {supported}."
            )

        feature_frame, _ = self._split_features_and_target(
            dataframe_handle,
            feature_columns=feature_columns,
        )
        numeric_frame = feature_frame.select_dtypes(include=["number", "bool"]).copy()
        findings: list[dict[str, Any]] = []
        warnings: list[str] = []

        if numeric_frame.shape[1] < 2:
            warnings.append(
                "Correlation redundancy requires at least two numeric features."
            )
        else:
            correlation = numeric_frame.corr().abs()
            upper_triangle = correlation.where(
                np.triu(np.ones(correlation.shape, dtype=bool), k=1)
            )
            pairs = (
                upper_triangle.stack()
                .sort_values(ascending=False)
                .reset_index()
                .rename(
                    columns={
                        "level_0": "feature_a",
                        "level_1": "feature_b",
                        0: "abs_correlation",
                    }
                )
            )
            flagged = pairs[pairs["abs_correlation"] >= threshold].head(max_pairs)
            findings.extend(flagged.to_dict(orient="records"))
            if flagged.empty:
                warnings.append(
                    f"No numeric feature pairs exceeded the correlation threshold of {threshold}."
                )

        duplicates = [
            {
                "feature_a": str(left),
                "feature_b": str(right),
                "duplicate": True,
            }
            for index, left in enumerate(feature_frame.columns)
            for right in feature_frame.columns[index + 1 :]
            if feature_frame[left].equals(feature_frame[right])
        ]
        if duplicates:
            findings.extend(duplicates[:max_pairs])

        report = StoredFeatureSelectionReport(
            report_type="redundancy",
            method=method,
            feature_columns=[str(column) for column in feature_frame.columns],
            findings=findings,
            warnings=warnings,
            metadata={
                "threshold": float(threshold),
                "numeric_feature_count": int(numeric_frame.shape[1]),
            },
        )
        return self._object_store.put(report, prefix="fs")

    @tool
    def evaluate_feature_subset(
        self,
        dataframe_handle: str,
        target_column: str,
        feature_columns: list[str],
        *,
        validation_handle: str | None = None,
        cv_folds: int = 5,
        random_state: int = 0,
        scorer_handle: str | None = None,
        splitter_handle: str | None = None,
        group_column: str | None = None,
    ) -> str:
        """Evaluate a user-provided feature subset with LightGBM metrics.

        Args:
            dataframe_handle (str): Training dataframe handle.
            target_column (str): Target column used for model evaluation.
            feature_columns (list[str]): Feature subset to evaluate.
            validation_handle (str | None): Optional validation dataframe handle.
            cv_folds (int): Number of folds used when validation data is absent.
            random_state (int): Random seed for reproducibility.
            scorer_handle (str | None): Optional stored scorer handle used to
                compute an additional objective-ready metric.
            splitter_handle (str | None): Optional stored splitter handle used when
                validation data is absent.
            group_column (str | None): Optional grouping column required by
                ``GroupKFold``-style splitters.

        Returns:
            str: Handle for the stored evaluation report.

        Examples:
            evaluation_handle = evaluate_feature_subset(
                train_handle,
                "target",
                ["age", "income"],
                validation_handle=valid_handle,
            )
        """
        feature_frame, target_series = self._split_features_and_target(
            dataframe_handle,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        if target_series is None:
            raise ValueError("A target column is required for subset evaluation.")

        source_dataframe = self._get_dataframe(dataframe_handle)
        from .metrics_collection import StoredMetricScorer
        from .splitting import StoredSplitter

        scorer = (
            self._object_store.get(scorer_handle, expected_type=StoredMetricScorer)
            if scorer_handle is not None
            else None
        )
        splitter = (
            self._object_store.get(splitter_handle, expected_type=StoredSplitter)
            if splitter_handle is not None
            else None
        )
        group_values = None
        if group_column is not None:
            if group_column not in source_dataframe.columns:
                raise ValueError(
                    f"Group column {group_column!r} was not found in the dataframe."
                )
            group_values = source_dataframe.loc[
                feature_frame.index, group_column
            ].copy()

        validation_features = None
        validation_target = None
        if validation_handle is not None:
            validation_features, validation_target = self._split_features_and_target(
                validation_handle,
                target_column=target_column,
                feature_columns=feature_columns,
            )

        metrics, warnings = evaluate_subset_metrics(
            feature_frame,
            target_series,
            validation_features=validation_features,
            validation_target=validation_target,
            cv_folds=cv_folds,
            random_state=random_state,
            scorer=scorer,
            splitter=splitter,
            groups=group_values,
        )
        report = StoredFeatureSelectionReport(
            report_type="subset_evaluation",
            method="lightgbm_subset_evaluation",
            feature_columns=list(feature_columns),
            target_column=target_column,
            metrics=metrics,
            warnings=warnings,
            metadata={
                "validation_supplied": validation_handle is not None,
                "cv_folds": int(cv_folds),
                "random_state": int(random_state),
                "scorer_handle": scorer_handle,
                "splitter_handle": splitter_handle,
                "group_column": group_column,
            },
        )
        return self._object_store.put(report, prefix="fs")

    @tool
    def rank_feature_importance_with_lightgbm(
        self,
        dataframe_handle: str,
        target_column: str,
        *,
        feature_columns: list[str] | None = None,
        validation_handle: str | None = None,
        method: str = "lightgbm",
        random_state: int = 0,
    ) -> str:
        """Rank candidate features using a small LightGBM model.

        Args:
            dataframe_handle (str): Training dataframe handle.
            target_column (str): Target column used for model fitting.
            feature_columns (list[str] | None): Optional explicit feature list.
            validation_handle (str | None): Optional validation dataframe handle.
            method (str): Currently only `lightgbm` is supported.
            random_state (int): Random seed for reproducibility.

        Returns:
            str: Handle for the stored importance report.

        Examples:
            importance_handle = rank_feature_importance_with_lightgbm(
                train_handle,
                "target",
                validation_handle=valid_handle,
            )
        """
        if method not in _SUPPORTED_IMPORTANCE_METHODS:
            supported = ", ".join(sorted(_SUPPORTED_IMPORTANCE_METHODS))
            raise ValueError(
                f"Unsupported importance method {method!r}. Supported values: {supported}."
            )

        feature_frame, target_series = self._split_features_and_target(
            dataframe_handle,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        if target_series is None:
            raise ValueError("A target column is required for model-based importances.")

        validation_features = None
        validation_target = None
        if validation_handle is not None:
            validation_features, validation_target = self._split_features_and_target(
                validation_handle,
                target_column=target_column,
                feature_columns=feature_columns,
            )

        findings, metrics, warnings = rank_lightgbm_importance(
            feature_frame,
            target_series,
            validation_features=validation_features,
            validation_target=validation_target,
            random_state=random_state,
        )
        report = StoredFeatureSelectionReport(
            report_type="importance",
            method=method,
            feature_columns=[str(column) for column in feature_frame.columns],
            target_column=target_column,
            findings=findings,
            metrics=metrics,
            warnings=warnings,
            metadata={
                "validation_supplied": validation_handle is not None,
                "random_state": int(random_state),
            },
        )
        return self._object_store.put(report, prefix="fs")

    @tool
    def inspect_feature_selection_report(self, report_handle: str) -> dict[str, Any]:
        """Return a JSON-friendly summary of a feature-selection report.

        Args:
            report_handle (str): Handle pointing to a stored report.

        Returns:
            dict[str, Any]: Report summary.

        Examples:
            print(inspect_feature_selection_report(report_handle))
            # Returns:
            # {
            #     "report_type": "target_metrics",
            #     "method": "mutual_info",
            #     "feature_columns": ["age", "income", "premium"]
            # }
        """
        return self._get_report(report_handle).to_json_summary()

    @tool
    def list_feature_selection_findings(
        self, report_handle: str
    ) -> list[dict[str, Any]]:
        """Return the full finding rows stored in a feature-selection report.

        Args:
            report_handle (str): Handle pointing to a stored report.

        Returns:
            list[dict[str, Any]]: Full report findings.

        Examples:
            print(list_feature_selection_findings(report_handle))
        """
        return list(self._get_report(report_handle).findings)

    @tool
    def save_feature_selection_report(self, report_handle: str, path: str) -> str:
        """Persist a stored report to the workspace with joblib.

        Args:
            report_handle (str): Stored report handle.
            path (str): Relative or `/workspace` destination path.

        Returns:
            str: Virtual path to the saved report.

        Examples:
            save_feature_selection_report(
                report_handle,
                "/workspace/output/fs_report.joblib",
            )
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._get_report(report_handle), host_path)
        self._os_access.record_host_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool
    def load_feature_selection_report(self, path: str) -> str:
        """Load a previously saved report from the workspace.

        Args:
            path (str): Relative or `/workspace` path to a saved report.

        Returns:
            str: Handle for the loaded report.

        Examples:
            report_handle = load_feature_selection_report(
                "/workspace/output/fs_report.joblib"
            )
        """
        report = joblib.load(self._resolve_host_path(path))
        if not isinstance(report, StoredFeatureSelectionReport):
            raise TypeError("Loaded artifact is not a StoredFeatureSelectionReport.")
        return self._object_store.put(report, prefix="fs")


__all__ = [
    "FeatureSelectionCollection",
    "StoredFeatureSelectionReport",
]
