"""Feature workbench tools for the minimal registry package."""

from __future__ import annotations

from typing import Any

import joblib

from ..base import WorkspaceToolCollection, SafeObjectStore
from ..core.registry import tool
from ..execution import FreeformDataframeExecutor, FreeformDataframeTransformer
from ..filesystem import HostWorkspaceOSAccess
from ..privacy import summarize_dataframe
from .base import (
    FeatureScreenConfig,
    SplitConfig,
    StoredFeatureSelectionReport,
    StoredFreeformTransformerArtifact,
    TrainConfig,
)
from .utils import (
    _is_numeric_dtype,
    make_train_valid_split,
    train_lightgbm_once,
    univariate_categorical_score,
    univariate_numeric_score,
)


class FeatureWorkbenchCollection(WorkspaceToolCollection):
    """Safe freeform and feature-screening helpers."""

    name = "feature_workbench"
    description = (
        "Run privacy-safe freeform dataframe transforms, fit reusable freeform "
        "transformers, and screen features for LightGBM modeling."
    )

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: SafeObjectStore,
    ) -> None:
        """Initialize the collection and its dedicated freeform executor.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (SafeObjectStore): Shared in-memory object store.
        """

        super().__init__(os_access, object_store)
        self._executor = FreeformDataframeExecutor(
            workspace_root=self._os_access.host_workspace_root
        )

    @tool
    def run_dataframe_code(self, dataframe_handle: str, code: str) -> dict[str, Any]:
        """Run privacy-safe freeform Python against a stored dataframe.

        The submitted code operates on a `df` variable and returns a transformed
        dataframe while suppressing raw row output for privacy. Prefer this only
        when predefined helpers do not already cover the transformation.

        Args:
            dataframe_handle (str): Source dataframe handle.
            code (str): Freeform Python source operating on `df`.

        Returns:
            dict[str, Any]: New dataframe handle plus safe execution metadata.

        Examples:
            run_dataframe_code(
                df_handle,
                "df['ratio'] = df['balance'] / df['income']\\nresult = df",
            )
            # Returns:
            # {
            #     "dataframe_handle": "df_123",
            #     "rows": 10000,
            #     "column_count": 9,
            #     "columns": ["balance", "income", "ratio", "target"],
            #     "columns_added": ["ratio"],
            #     "columns_removed": [],
            #     "stdout": {
            #         "suppressed": False,
            #         "line_count": 0,
            #         "character_count": 0,
            #         "message": "No stdout was captured."
            #     }
            # }
        """

        source_dataframe = self._get_dataframe(dataframe_handle)
        result = self._executor.execute(code, source_dataframe)
        handle = self._object_store.put(result.dataframe, prefix="df")
        return {
            "dataframe_handle": handle,
            "rows": result.rows,
            "column_count": len(result.columns),
            "columns": result.columns,
            "columns_added": result.columns_added,
            "columns_removed": result.columns_removed,
            "stdout": result.stdout_summary,
        }

    @tool
    def fit_freeform_transformer(
        self,
        dataframe_handle: str,
        code: str,
        *,
        target_column: str | None = None,
        params: dict[str, Any] | None = None,
        preserve_index: bool = True,
        strict_schema: bool = True,
    ) -> str:
        """Fit a reusable freeform transformer artifact from Python code.

        This lets you capture a repeatable dataframe transformation once and reuse
        it later across train, validation, or scoring dataframes.

        Args:
            dataframe_handle (str): Source dataframe handle used for fitting.
            code (str): Freeform Python source defining the transformation.
            target_column (str | None): Optional target column excluded during fit.
            params (dict[str, Any] | None): Optional parameter dictionary exposed to
                the transformer code.
            preserve_index (bool): Whether transformed dataframes keep their index.
            strict_schema (bool): Whether transformed inputs must match the fit schema.

        Returns:
            str: Handle for the fitted freeform transformer artifact.

        Examples:
            transformer_handle = fit_freeform_transformer(
                df_handle,
                "df['balance_log'] = np.log1p(df['balance'])\\nresult = df",
                target_column="target",
                params={"clip_min": 0.0, "clip_max": 100000.0},
            )
            # Returns:
            # "freeform_123"
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        if target_column is not None:
            if target_column not in dataframe.columns:
                raise ValueError(f"Target column {target_column!r} was not found.")
            feature_frame = dataframe.drop(columns=[target_column]).copy()
        else:
            feature_frame = dataframe
        estimator = FreeformDataframeTransformer(
            code=code,
            workspace_root=str(self._os_access.host_workspace_root),
            params=dict(params or {}),
            preserve_index=preserve_index,
            strict_schema=strict_schema,
        )
        estimator.fit(feature_frame)
        artifact = StoredFreeformTransformerArtifact(
            estimator=estimator,
            input_columns=list(estimator.input_columns_),
            output_columns=list(estimator.output_columns_),
            columns_added=list(estimator.columns_added_),
            columns_removed=list(estimator.columns_removed_),
            params=dict(params or {}),
            target_column=target_column,
        )
        return self._object_store.put(artifact, prefix="freeform")

    @tool
    def transform_with_freeform_transformer(
        self,
        dataframe_handle: str,
        transformer_handle: str,
        *,
        include_target: bool = False,
    ) -> dict[str, Any]:
        """Transform a dataframe with a fitted freeform transformer.

        Args:
            dataframe_handle (str): Input dataframe handle to transform.
            transformer_handle (str): Stored freeform transformer handle.
            include_target (bool): Whether to append the saved target column back to
                the transformed dataframe when available.

        Returns:
            dict[str, Any]: Transformed dataframe handle plus a safe summary.

        Examples:
            transformed = transform_with_freeform_transformer(df_handle, transformer_handle)
            # Returns:
            # {
            #     "dataframe_handle": "df_456",
            #     "summary": {
            #         "row_count": 10000,
            #         "column_count": 9,
            #         "columns": ["balance", "income", "balance_log", "target"]
            #     }
            # }
        """

        artifact = self._object_store.get(
            transformer_handle,
            expected_type=StoredFreeformTransformerArtifact,
        )
        dataframe = self._get_dataframe(dataframe_handle).copy()
        target_series = None
        if (
            artifact.target_column is not None
            and artifact.target_column in dataframe.columns
        ):
            target_series = dataframe[artifact.target_column].copy()
            dataframe = dataframe.drop(columns=[artifact.target_column]).copy()
        transformed = artifact.estimator.transform(dataframe)
        if include_target and target_series is not None:
            transformed[artifact.target_column] = target_series.values
        handle = self._object_store.put(transformed, prefix="df")
        return {
            "dataframe_handle": handle,
            "summary": summarize_dataframe(transformed),
        }

    @tool
    def inspect_freeform_transformer(self, transformer_handle: str) -> dict[str, Any]:
        """Return a safe summary of a reusable freeform transformer.

        Args:
            transformer_handle (str): Stored freeform transformer handle.

        Returns:
            dict[str, Any]: Compact transformer summary for handle inspection.

        Examples:
            summary = inspect_freeform_transformer(transformer_handle)
            # Returns:
            # {
            #     "type": "StoredFreeformTransformerArtifact",
            #     "input_columns": ["balance", "income"],
            #     "output_columns": ["balance", "income", "balance_log"],
            #     "columns_added": ["balance_log"],
            #     "params": {"clip_min": 0.0, "clip_max": 100000.0}
            # }
        """

        artifact = self._object_store.get(
            transformer_handle,
            expected_type=StoredFreeformTransformerArtifact,
        )
        return artifact.to_json_summary()

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

        This tool quickly drops obviously weak features and ranks the remaining
        candidates by simple univariate signal before modeling.

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
            dict[str, Any]: Report handle, reduced dataframe handle, and selected
            feature metadata.

        Examples:
            screen_features(
                df_handle,
                "target",
                id_columns=["customer_id"],
                top_k_univariate=50,
            )
            # Returns:
            # {
            #     "report_handle": "fs_123",
            #     "dataframe_handle": "df_789",
            #     "selected_columns": ["balance", "segment_rate", "utilization"],
            #     "categorical_columns": ["segment_rate"]
            # }
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column {target_column!r} was not found.")
        id_columns = list(id_columns or [])
        config = FeatureScreenConfig(
            max_missing_frac=max_missing_frac,
            near_constant_thresh=near_constant_thresh,
            min_non_null=min_non_null,
            top_k_univariate=top_k_univariate,
        )
        if feature_columns is None:
            candidate_columns = [
                str(column)
                for column in dataframe.columns
                if column not in [target_column] + id_columns
            ]
        else:
            candidate_columns = list(feature_columns)
        y = dataframe[target_column].astype(int)
        findings: list[dict[str, Any]] = []
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []

        for column in candidate_columns:
            if column not in dataframe.columns:
                continue
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
                    numeric_columns.append(column)
                    score = univariate_numeric_score(series, y)
                else:
                    categorical_columns.append(column)
                    score = univariate_categorical_score(series, y)

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
        selected_columns = (
            [target_column]
            + [column for column in id_columns if column in dataframe.columns]
            + selected_features
        )
        selected_dataframe = dataframe[selected_columns].copy()
        report = StoredFeatureSelectionReport(
            report_type="screen_features",
            target_column=target_column,
            selected_columns=selected_features,
            categorical_columns=[
                column for column in categorical_columns if column in selected_features
            ],
            findings=kept_rows,
            metrics={
                "selected_feature_count": len(selected_features),
                "numeric_feature_count": len(numeric_columns),
            },
            metadata={
                "id_columns": id_columns,
                "top_k_univariate": top_k_univariate,
            },
        )
        report_handle = self._object_store.put(report, prefix="fs")
        dataframe_result_handle = self._object_store.put(
            selected_dataframe, prefix="df"
        )
        return {
            "report_handle": report_handle,
            "dataframe_handle": dataframe_result_handle,
            "selected_columns": selected_features,
            "categorical_columns": report.categorical_columns,
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
        top_p: float = 0.05,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Rank candidate features using a lightweight LightGBM model.

        This is a stronger ranking pass than `screen_features` because it uses
        model-based feature importance instead of purely univariate signal.

        Args:
            dataframe_handle (str): Source dataframe handle.
            target_column (str): Target column used for modeling.
            feature_columns (list[str] | None): Optional explicit feature subset.
            validation_handle (str | None): Optional held-out validation dataframe.
            keep_top_k (int): Maximum number of top-ranked features to keep.
            top_p (float): Fraction retained for PPV-style evaluation.
            random_seed (int): Random seed for splitting and LightGBM training.

        Returns:
            dict[str, Any]: Feature-report handle plus selected feature metadata.

        Examples:
            rank_features_by_lightgbm(
                df_handle,
                "target",
                keep_top_k=75,
            )
            # Returns:
            # {
            #     "report_handle": "fs_456",
            #     "selected_columns": ["score_signal", "balance", "segment"],
            #     "categorical_columns": ["segment"]
            # }
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
        categorical_columns = [
            column
            for column in feature_columns
            if not _is_numeric_dtype(model_frame[column])
        ]
        result = train_lightgbm_once(
            train_df,
            valid_df,
            label_col=target_column,
            categorical_cols=categorical_columns,
            train_config=TrainConfig(seed=random_seed),
            top_p=top_p,
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
                "valid_ppv_at_5": result["valid_ppv_at_5"],
                "valid_recall_at_5": result["valid_recall_at_5"],
                "valid_lift_at_5": result["valid_lift_at_5"],
                "base_rate": result["base_rate"],
            },
        )
        handle = self._object_store.put(report, prefix="fs")
        return {
            "report_handle": handle,
            "selected_columns": report.selected_columns,
            "categorical_columns": report.categorical_columns,
        }

    @tool
    def inspect_feature_report(self, report_handle: str) -> dict[str, Any]:
        """Return a safe summary for a stored feature report.

        Args:
            report_handle (str): Stored feature-report handle.

        Returns:
            dict[str, Any]: Compact feature-report summary.

        Examples:
            report = inspect_feature_report(report_handle)
            # Returns:
            # {
            #     "type": "StoredFeatureSelectionReport",
            #     "report_type": "screen_features",
            #     "selected_columns": ["balance", "utilization"],
            #     "metrics": {"selected_feature_count": 2}
            # }
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
            saved_path = save_feature_report(report_handle, "/workspace/reports/screen.joblib")
            # Returns:
            # "/workspace/reports/screen.joblib"
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


__all__ = ["FeatureWorkbenchCollection"]
