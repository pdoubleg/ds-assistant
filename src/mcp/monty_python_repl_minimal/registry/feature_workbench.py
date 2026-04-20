"""Feature-selection helpers for the minimal registry package."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from ..base import WorkspaceToolCollection, SafeObjectStore
from ..core.registry import tool
from ..filesystem import HostWorkspaceOSAccess
from .base import (
    FeatureScreenConfig,
    SplitConfig,
    StoredFeatureSelectionReport,
    TrainConfig,
)
from .utils import (
    _is_numeric_dtype,
    make_train_valid_split,
    train_lightgbm_once,
    univariate_categorical_score,
    univariate_numeric_score,
)


class FeatureSelectionCollection(WorkspaceToolCollection):
    """Deterministic feature-selection helpers for modeling workflows."""

    name = "feature_selection"
    description = (
        "Screen features, remove highly correlated columns, rank feature subsets, "
        "and apply saved selection reports."
    )

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: SafeObjectStore,
    ) -> None:
        """Initialize the feature-selection collection.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (SafeObjectStore): Shared in-memory object store.
        """

        super().__init__(os_access, object_store)

    def _resolve_candidate_columns(
        self,
        dataframe: pd.DataFrame,
        *,
        target_column: str | None,
        id_columns: list[str] | None,
        feature_columns: list[str] | None,
    ) -> list[str]:
        """Resolve candidate feature columns in stable dataframe order."""

        if feature_columns is not None:
            return [column for column in feature_columns if column in dataframe.columns]

        excluded = set(id_columns or [])
        if target_column is not None:
            excluded.add(target_column)
        return [
            str(column) for column in dataframe.columns if str(column) not in excluded
        ]

    def _categorical_columns(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> list[str]:
        """Return the non-numeric feature subset."""

        return [
            column for column in columns if not _is_numeric_dtype(dataframe[column])
        ]

    def _build_selected_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        target_column: str | None,
        id_columns: list[str] | None,
        selected_features: list[str],
    ) -> pd.DataFrame:
        """Build a reduced dataframe from selected features and preserved columns."""

        selected_columns: list[str] = []
        if target_column is not None and target_column in dataframe.columns:
            selected_columns.append(target_column)
        for column in list(id_columns or []):
            if column in dataframe.columns and column not in selected_columns:
                selected_columns.append(column)
        for column in selected_features:
            if column in dataframe.columns and column not in selected_columns:
                selected_columns.append(column)
        return dataframe[selected_columns].copy()

    @tool
    def screen_features(
        self,
        dataframe_handle: str,
        target_column: str,
        *,
        id_columns: list[str] | None = None,
        feature_columns: list[str] | None = None,
        max_missing_frac: float = 0.98,
        near_constant_thresh: float = 0.995,
        min_non_null: int = 100,
        top_k_univariate: int = 200,
    ) -> dict[str, Any]:
        """Screen candidate features with descriptive and univariate filters.

        This is the fastest selection pass. It removes obviously weak columns and
        ranks the remaining feature set using privacy-safe univariate signal.

        Args:
            dataframe_handle (str): Source dataframe handle.
            target_column (str): Binary target column used for screening.
            id_columns (list[str] | None): Optional identifier columns excluded from
                the candidate set.
            feature_columns (list[str] | None): Optional explicit feature subset.
            max_missing_frac (float): Maximum allowed missing-value fraction.
            near_constant_thresh (float): Maximum allowed dominant-value rate.
            min_non_null (int): Minimum required non-null count.
            top_k_univariate (int): Maximum number of kept features after ranking.

        Returns:
            dict[str, Any]: Selection report handle, reduced dataframe handle, and a
            compact workflow summary.

        Examples:
            ```python
            screen = screen_features(
                df_handle,
                "target",
                id_columns=["customer_id"],
                top_k_univariate=50,
            )
            # Returns
            # {
            #     "report_handle": "report_abc123",
            #     "dataframe_handle": "df_abc123",
            #     "summary": "Screened 10 candidate features and kept 5.",
            #     "warnings": [],
            #     "selected_columns": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            #     "categorical_columns": ["feature_6", "feature_7"],
            #     "selected_feature_count": 5,
            # }
            screen_report = inspect_feature_report(screen["report_handle"])
            # Returns
            # {
            #     "type": "FeatureSelectionReport",
            #     "target_column": "target",
            #     "selected_columns": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            #     "categorical_columns": ["feature_6", "feature_7"],
            #     "findings": [...],
            #     "metrics": {
            #         "candidate_feature_count": 10,
            #         "selected_feature_count": 5,
            #         "dropped_feature_count": 5,
            #     },
            #     "warnings": [],
            #     "metadata": {
            #         "id_columns": ["customer_id"],
            #         "top_k_univariate": 50,
            #     }
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column {target_column!r} was not found.")

        config = FeatureScreenConfig(
            max_missing_frac=max_missing_frac,
            near_constant_thresh=near_constant_thresh,
            min_non_null=min_non_null,
            top_k_univariate=top_k_univariate,
        )
        candidate_columns = self._resolve_candidate_columns(
            dataframe,
            target_column=target_column,
            id_columns=id_columns,
            feature_columns=feature_columns,
        )
        target = dataframe[target_column].astype(int)
        findings: list[dict[str, Any]] = []
        warnings: list[str] = []

        for column in candidate_columns:
            series = dataframe[column]
            non_null_count = int(series.notna().sum())
            missing_rate = float(series.isna().mean())
            unique_count = int(series.nunique(dropna=True))
            dominant_rate = (
                float(series.value_counts(dropna=False, normalize=True).iloc[0])
                if not series.empty
                else 0.0
            )

            keep = True
            reasons: list[str] = []
            if non_null_count < config.min_non_null:
                keep = False
                reasons.append("too_few_non_null")
            if missing_rate > config.max_missing_frac:
                keep = False
                reasons.append("too_missing")
            if unique_count <= 1:
                keep = False
                reasons.append("zero_variance")
            if dominant_rate >= config.near_constant_thresh:
                keep = False
                reasons.append("near_constant")

            score = 0.0
            if keep:
                if _is_numeric_dtype(series):
                    score = univariate_numeric_score(series, target)
                else:
                    score = univariate_categorical_score(series, target)

            findings.append(
                {
                    "feature": column,
                    "dtype": str(series.dtype),
                    "missing_rate": missing_rate,
                    "unique_count": unique_count,
                    "dominant_rate": dominant_rate,
                    "score": float(score),
                    "kept": keep,
                    "reasons": reasons,
                }
            )

        kept_rows = [row for row in findings if row["kept"]]
        kept_rows.sort(key=lambda row: row["score"], reverse=True)
        selected_features = [
            str(row["feature"]) for row in kept_rows[: config.top_k_univariate]
        ]
        categorical_columns = self._categorical_columns(dataframe, selected_features)
        selected_dataframe = self._build_selected_dataframe(
            dataframe,
            target_column=target_column,
            id_columns=id_columns,
            selected_features=selected_features,
        )
        report = StoredFeatureSelectionReport(
            report_type="screen_features",
            target_column=target_column,
            selected_columns=selected_features,
            categorical_columns=categorical_columns,
            findings=findings,
            metrics={
                "candidate_feature_count": len(candidate_columns),
                "selected_feature_count": len(selected_features),
                "dropped_feature_count": len(candidate_columns)
                - len(selected_features),
            },
            warnings=warnings,
            metadata={
                "id_columns": list(id_columns or []),
                "top_k_univariate": top_k_univariate,
            },
        )
        report_handle = self._object_store.put(report, prefix="fs")
        dataframe_handle_out = self._object_store.put(selected_dataframe, prefix="df")
        summary = (
            f"Screened {len(candidate_columns)} candidate features and kept "
            f"{len(selected_features)}."
        )
        return {
            "report_handle": report_handle,
            "dataframe_handle": dataframe_handle_out,
            "summary": summary,
            "warnings": warnings,
            "selected_columns": selected_features,
            "categorical_columns": categorical_columns,
            "selected_feature_count": len(selected_features),
        }

    @tool
    def analyze_feature_correlation(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        id_columns: list[str] | None = None,
        feature_columns: list[str] | None = None,
        threshold: float = 0.95,
        max_numeric_features: int = 300,
    ) -> dict[str, Any]:
        """Find highly correlated numeric features and suggest deterministic drops.

        Use this after the first screening pass when you want to reduce redundant
        numeric signal before model-based ranking or HPO.

        Args:
            dataframe_handle (str): Source dataframe handle.
            target_column (str | None): Optional target column excluded from the
                correlation candidate set.
            id_columns (list[str] | None): Optional identifier columns excluded from
                the candidate set.
            feature_columns (list[str] | None): Optional explicit feature subset.
            threshold (float): Absolute correlation threshold used to flag pairs.
            max_numeric_features (int): Maximum numeric columns considered in one pass.

        Returns:
            dict[str, Any]: Selection report handle, reduced dataframe handle, and a
            compact summary of proposed drops.

        Examples:
            ```python
            screen = screen_features(
                df_handle,
                "target",
                id_columns=["customer_id"],
                top_k_univariate=80,
            )
            decorrelated = analyze_feature_correlation(
                screen["dataframe_handle"],
                target_column="target",
                threshold=0.9,
            )
            # Returns
            # {
            #     "report_handle": "report_abc123",
            #     "dataframe_handle": "df_abc123",
            #     "summary": "Flagged 5 highly correlated pairs and proposed 2 drops.",
            #     "warnings": [],
            #     "selected_columns": ["feature_1", "feature_2", "feature_3"],
            #     "dropped_columns": ["feature_4", "feature_5"],
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        candidate_columns = self._resolve_candidate_columns(
            dataframe,
            target_column=target_column,
            id_columns=id_columns,
            feature_columns=feature_columns,
        )
        numeric_columns = [
            column
            for column in candidate_columns
            if _is_numeric_dtype(dataframe[column])
        ]
        warnings: list[str] = []
        if len(numeric_columns) > max_numeric_features:
            warnings.append(
                "Too many numeric features for one correlation pass; used the first "
                f"{max_numeric_features} numeric columns in dataframe order."
            )
            numeric_columns = numeric_columns[:max_numeric_features]

        if len(numeric_columns) < 2:
            raise ValueError("At least two numeric feature columns are required.")

        numeric_frame = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce")
        corr = numeric_frame.corr(method="pearson").abs().fillna(0.0)

        pair_rows: list[dict[str, Any]] = []
        drop_columns: list[str] = []
        dropped_set: set[str] = set()
        for left_index, left_column in enumerate(numeric_columns):
            for right_column in numeric_columns[left_index + 1 :]:
                corr_value = float(corr.loc[left_column, right_column])
                if corr_value < threshold:
                    continue
                pair_rows.append(
                    {
                        "left_feature": left_column,
                        "right_feature": right_column,
                        "abs_correlation": corr_value,
                    }
                )
                if right_column not in dropped_set:
                    drop_columns.append(right_column)
                    dropped_set.add(right_column)

        selected_features = [
            column for column in candidate_columns if column not in dropped_set
        ]
        selected_dataframe = self._build_selected_dataframe(
            dataframe,
            target_column=target_column,
            id_columns=id_columns,
            selected_features=selected_features,
        )
        report = StoredFeatureSelectionReport(
            report_type="decorrelation",
            target_column=target_column or "",
            selected_columns=selected_features,
            categorical_columns=self._categorical_columns(dataframe, selected_features),
            findings=pair_rows,
            metrics={
                "candidate_feature_count": len(candidate_columns),
                "numeric_feature_count": len(numeric_columns),
                "flagged_pair_count": len(pair_rows),
                "dropped_feature_count": len(drop_columns),
            },
            warnings=warnings,
            metadata={
                "threshold": threshold,
                "id_columns": list(id_columns or []),
            },
        )
        report_handle = self._object_store.put(report, prefix="fs")
        dataframe_handle_out = self._object_store.put(selected_dataframe, prefix="df")
        summary = (
            f"Flagged {len(pair_rows)} highly correlated pairs and proposed "
            f"{len(drop_columns)} drops."
        )
        return {
            "report_handle": report_handle,
            "dataframe_handle": dataframe_handle_out,
            "summary": summary,
            "warnings": warnings,
            "selected_columns": selected_features,
            "dropped_columns": drop_columns,
        }

    @tool
    def rank_features_by_lightgbm(
        self,
        dataframe_handle: str,
        target_column: str,
        *,
        feature_columns: list[str] | None = None,
        validation_handle: str | None = None,
        keep_top_k: int = 100,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Rank candidate features using a lightweight LightGBM model.

        This is a stronger ranking pass than `screen_features(...)` because it uses
        model-based feature importance instead of purely univariate signal.

        Args:
            dataframe_handle (str): Source dataframe handle.
            target_column (str): Target column used for modeling.
            feature_columns (list[str] | None): Optional explicit feature subset.
            validation_handle (str | None): Optional held-out validation dataframe.
            keep_top_k (int): Maximum number of top-ranked features to keep.
            random_seed (int): Random seed for splitting and LightGBM training.

        Returns:
            dict[str, Any]: Feature-report handle and a compact ranking summary.

        Examples:
            ```python
            ranked = rank_features_by_lightgbm(
                df_handle,
                "target",
                keep_top_k=75,
            )
            # Returns
            # {
            #     "report_handle": "report_abc123",
            #     "summary": "Ranked 100 features and kept the top 75 by LightGBM gain.",
            #     "selected_columns": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            #     "categorical_columns": ["feature_6", "feature_7"],
            #     "selected_feature_count": 75,
            # }
            ranked_report = inspect_feature_report(ranked["report_handle"])
            # Returns
            # {
            #     "type": "FeatureSelectionReport",
            #     "target_column": "target",
            #     "selected_columns": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            #     "categorical_columns": ["feature_6", "feature_7"],
            #     "findings": [...],
            #     "metrics": {
            #         "candidate_feature_count": 100,
            #         "selected_feature_count": 75,
            #         "valid_ppv_at_5": 0.75,
            #         "valid_recall_at_5": 0.75,
            #         "valid_lift_at_5": 1.5,
            #         "base_rate": 0.5,
            #     },
            #     "warnings": [],
            #     "metadata": {
            #         "keep_top_k": 75,
            #     }
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column {target_column!r} was not found.")
        if feature_columns is None:
            feature_columns = [
                str(column) for column in dataframe.columns if column != target_column
            ]

        model_frame = dataframe[[target_column] + feature_columns].copy()
        if validation_handle is None:
            train_df, valid_df = make_train_valid_split(
                model_frame,
                config=SplitConfig(random_seed=random_seed),
            )
        else:
            train_df = model_frame
            valid_df = self._get_dataframe(validation_handle)[
                [target_column] + feature_columns
            ].copy()

        categorical_columns = self._categorical_columns(model_frame, feature_columns)
        result = train_lightgbm_once(
            train_df,
            valid_df,
            label_col=target_column,
            categorical_cols=categorical_columns,
            train_config=TrainConfig(seed=random_seed),
            top_p=0.05,
        )
        top_rows = [
            {"feature": feature, "gain": gain}
            for feature, gain in sorted(
                result["feature_importance_gain"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:keep_top_k]
        ]
        report = StoredFeatureSelectionReport(
            report_type="lightgbm_importance",
            target_column=target_column,
            selected_columns=[row["feature"] for row in top_rows],
            categorical_columns=result["categorical_columns"],
            findings=top_rows,
            metrics={
                "candidate_feature_count": len(feature_columns),
                "selected_feature_count": len(top_rows),
                "valid_ppv_at_5": result["valid_ppv_at_5"],
                "valid_recall_at_5": result["valid_recall_at_5"],
                "valid_lift_at_5": result["valid_lift_at_5"],
                "base_rate": result["base_rate"],
            },
        )
        report_handle = self._object_store.put(report, prefix="fs")
        summary = (
            f"Ranked {len(feature_columns)} features and kept the top {len(top_rows)} "
            "by LightGBM gain."
        )
        return {
            "report_handle": report_handle,
            "summary": summary,
            "selected_columns": report.selected_columns,
            "categorical_columns": report.categorical_columns,
            "selected_feature_count": len(top_rows),
        }

    @tool
    def rank_feature_subsets(
        self,
        dataframe_handle: str,
        target_column: str,
        feature_subsets: list[list[str]],
        *,
        validation_handle: str | None = None,
        keep_top_k_per_subset: int = 25,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Rank multiple feature subsets with repeated LightGBM passes.

        Use this after `plan_feature_subsets(...)` when the dataframe is too wide
        for one ranking pass and you want one deterministic report over all subsets.

        Args:
            dataframe_handle (str): Source dataframe handle.
            target_column (str): Binary target column used for ranking.
            feature_subsets (list[list[str]]): Ordered feature subsets to evaluate.
            validation_handle (str | None): Optional held-out validation dataframe.
            keep_top_k_per_subset (int): Maximum retained features per subset.
            random_seed (int): Random seed for splitting and LightGBM training.

        Returns:
            dict[str, Any]: Batch-ranking report handle, reduced dataframe handle, and
            a compact summary.

        Examples:
            ```python
            subset_plan = plan_feature_subsets(
                df_handle,
                target_column="target",
                id_columns=["customer_id"],
                batch_size=25,
            )
            batched_rank = rank_feature_subsets(
                df_handle,
                "target",
                subset_plan["feature_subsets"],
            )
            # Returns
            # {
            #     "report_handle": "report_abc123",
            #     "dataframe_handle": "df_abc123",
            #     "summary": "Ranked 2 feature subsets and kept 5 combined features.",
            #     "warnings": [],
            #     "selected_columns": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            #     "selected_feature_count": 5,
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column {target_column!r} was not found.")

        subset_rows: list[dict[str, Any]] = []
        selected_union: list[str] = []
        warnings: list[str] = []

        for subset_index, raw_subset in enumerate(feature_subsets, start=1):
            feature_columns = [
                column
                for column in raw_subset
                if column in dataframe.columns and column != target_column
            ]
            if not feature_columns:
                warnings.append(
                    f"Skipped empty feature subset at index {subset_index}."
                )
                continue

            model_frame = dataframe[[target_column] + feature_columns].copy()
            if validation_handle is None:
                train_df, valid_df = make_train_valid_split(
                    model_frame,
                    config=SplitConfig(random_seed=random_seed),
                )
            else:
                train_df = model_frame
                valid_df = self._get_dataframe(validation_handle)[
                    [target_column] + feature_columns
                ].copy()

            categorical_columns = self._categorical_columns(
                model_frame, feature_columns
            )
            result = train_lightgbm_once(
                train_df,
                valid_df,
                label_col=target_column,
                categorical_cols=categorical_columns,
                train_config=TrainConfig(seed=random_seed),
                top_p=0.05,
            )
            ranked_columns = [
                feature
                for feature, _ in sorted(
                    result["feature_importance_gain"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:keep_top_k_per_subset]
            ]
            for column in ranked_columns:
                if column not in selected_union:
                    selected_union.append(column)
            subset_rows.append(
                {
                    "subset_index": subset_index,
                    "input_feature_count": len(feature_columns),
                    "selected_columns": ranked_columns,
                    "valid_ppv_at_5": result["valid_ppv_at_5"],
                    "valid_recall_at_5": result["valid_recall_at_5"],
                    "valid_lift_at_5": result["valid_lift_at_5"],
                }
            )

        reduced_dataframe = self._build_selected_dataframe(
            dataframe,
            target_column=target_column,
            id_columns=None,
            selected_features=selected_union,
        )
        report = StoredFeatureSelectionReport(
            report_type="batched_lightgbm_importance",
            target_column=target_column,
            selected_columns=selected_union,
            categorical_columns=self._categorical_columns(dataframe, selected_union),
            findings=subset_rows,
            metrics={
                "subset_count": len(feature_subsets),
                "selected_feature_count": len(selected_union),
            },
            warnings=warnings,
            metadata={"keep_top_k_per_subset": keep_top_k_per_subset},
        )
        report_handle = self._object_store.put(report, prefix="fs")
        dataframe_handle_out = self._object_store.put(reduced_dataframe, prefix="df")
        summary = (
            f"Ranked {len(feature_subsets)} feature subsets and kept "
            f"{len(selected_union)} combined features."
        )
        return {
            "report_handle": report_handle,
            "dataframe_handle": dataframe_handle_out,
            "summary": summary,
            "warnings": warnings,
            "selected_columns": selected_union,
            "selected_feature_count": len(selected_union),
        }

    @tool
    def apply_feature_report(
        self,
        dataframe_handle: str,
        report_handle: str,
        *,
        id_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply a saved feature-selection report to a dataframe handle.

        Args:
            dataframe_handle (str): Source dataframe handle.
            report_handle (str): Stored feature-selection report handle.
            id_columns (list[str] | None): Optional identifier columns preserved in
                the reduced dataframe.

        Returns:
            dict[str, Any]: Reduced dataframe handle and a compact summary.

        Examples:
            ```python
            screen = screen_features(
                df_handle,
                "target",
                id_columns=["customer_id"],
                top_k_univariate=40,
            )
            reduced = apply_feature_report(
                df_handle,
                screen["report_handle"],
                id_columns=["customer_id"],
            )
            # Returns
            # {
            #     "dataframe_handle": "df_abc123",
            #     "report_handle": "report_abc123",
            #     "summary": "Applied feature report with 40 selected features.",
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        report = self._object_store.get(
            report_handle,
            expected_type=StoredFeatureSelectionReport,
        )
        reduced_dataframe = self._build_selected_dataframe(
            dataframe,
            target_column=report.target_column or None,
            id_columns=id_columns,
            selected_features=report.selected_columns,
        )
        reduced_handle = self._object_store.put(reduced_dataframe, prefix="df")
        summary = (
            f"Applied feature report with {len(report.selected_columns)} selected "
            "features."
        )
        return {
            "dataframe_handle": reduced_handle,
            "report_handle": report_handle,
            "summary": summary,
        }

    @tool
    def inspect_feature_report(self, report_handle: str) -> dict[str, Any]:
        """Return a safe summary for a stored feature-selection report.

        Args:
            report_handle (str): Stored feature-report handle.

        Returns:
            dict[str, Any]: Compact feature-report summary.

        Examples:
            ```python
            report_summary = inspect_feature_report(report_handle)
            # Returns
            # {
            #     "type": "FeatureSelectionReport",
            #     "target_column": "target",
            #     "selected_columns": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            #     "categorical_columns": ["feature_6", "feature_7"],
            #     "findings": [...],
            #     "metrics": {
            #         "candidate_feature_count": 10,
            #         "selected_feature_count": 5,
            #         "dropped_feature_count": 5,
            #     },
            #     "warnings": [],
            #     "metadata": {
            #         "id_columns": ["customer_id"],
            #         "top_k_univariate": 50,
            #     }
            # }
            ```
        """

        report = self._object_store.get(
            report_handle,
            expected_type=StoredFeatureSelectionReport,
        )
        return report.to_json_summary()

    @tool
    def save_feature_report(self, report_handle: str, path: str) -> str:
        """Persist a feature report artifact to `/workspace`.

        Args:
            report_handle (str): Stored feature-report handle.
            path (str): Output artifact path under `/workspace`.

        Returns:
            str: Virtual workspace path to the saved artifact.

        Examples:
            ```python
            saved_path = save_feature_report(
                report_handle,
                "/workspace/reports/selection.joblib",
            )
            # Returns
            # "/workspace/reports/selection.joblib"
            ```
        """

        report = self._object_store.get(
            report_handle,
            expected_type=StoredFeatureSelectionReport,
        )
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(report, host_path)
        self._record_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))


__all__ = ["FeatureSelectionCollection"]
