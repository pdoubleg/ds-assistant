"""Freeform dataframe execution tools for the Monty Python REPL."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..core.registry import WorkspaceToolCollection, tool
from ..execution import FreeformDataframeExecutor
from ..support.sklearn_pipeline import (
    StoredSklearnStageArtifact,
    append_target_column,
    ensure_dataframe,
)


def _flatten_nested_params(value: Any, *, prefix: str) -> dict[str, Any]:
    """Flatten nested mappings into sklearn-style param names."""
    rows: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, nested_value in value.items():
            child_prefix = f"{prefix}__{key}"
            rows[child_prefix] = nested_value
            rows.update(_flatten_nested_params(nested_value, prefix=child_prefix))
    return rows


def _set_nested_param_value(
    container: dict[str, Any], path_parts: list[str], value: Any
) -> None:
    """Write a nested param value into a copied mapping."""
    current_part = path_parts[0]
    if len(path_parts) == 1:
        container[current_part] = value
        return
    nested = container.get(current_part)
    if not isinstance(nested, dict):
        nested = {}
        container[current_part] = nested
    _set_nested_param_value(nested, path_parts[1:], value)


@dataclass(slots=True)
class StoredFreeformTransformer(StoredSklearnStageArtifact):
    """Persisted reusable freeform dataframe transformer artifact.

    Args:
        estimator (BaseEstimator): Fitted reusable freeform transformer.
        spec (dict[str, Any]): Normalized creation config.
        input_columns (list[str]): Input columns seen during fitting.
        output_columns (list[str]): Output columns produced by the code.
        target_column (str | None): Optional target excluded during fitting.
        preserve_index (bool): Whether transformed outputs preserve the index.
        code (str): Validated freeform code executed against `df`.
        intent (str): User-facing intent label such as `feature_engineering`.
        strict_schema (bool): Whether transform-time columns must match fit-time columns.
        columns_added (list[str]): Columns introduced during fit.
        columns_removed (list[str]): Columns removed during fit.
        fit_stdout (str): Captured stdout emitted during fit-time validation.
        args (dict[str, Any]): Explicit transformer arguments exposed to the code.
    """

    code: str = ""
    intent: str = "feature_engineering"
    strict_schema: bool = True
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    fit_stdout: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection."""
        summary = self._base_summary(
            artifact_type="StoredFreeformTransformer",
            max_items=max_items,
            max_chars=max_chars,
        )
        summary.update(
            {
                "intent": self.intent,
                "strict_schema": self.strict_schema,
                "columns_added": self.columns_added[:max_items],
                "columns_removed": self.columns_removed[:max_items],
                "fit_stdout": self.fit_stdout[:max_chars],
                "args": self.args,
                "code": self.code[:max_chars],
            }
        )
        return summary


class FreeformDataframeTransformer(BaseEstimator, TransformerMixin):
    """Reusable sklearn-compatible wrapper around validated freeform dataframe code.

    Args:
        code (str): Freeform code that reads and writes `df`.
        workspace_root (str | None): Host workspace root used for `/workspace` paths.
        intent (str): User-facing intent label.
        params (dict[str, Any] | None): Explicit arguments exposed to the code as
            `params`.
        preserve_index (bool): Whether transformed outputs preserve the source index.
        strict_schema (bool): Whether transform-time columns must match fit-time columns.
    """

    def __init__(
        self,
        *,
        code: str,
        workspace_root: str | None = None,
        intent: str = "feature_engineering",
        params: dict[str, Any] | None = None,
        preserve_index: bool = True,
        strict_schema: bool = True,
    ) -> None:
        self.code = code
        self.workspace_root = workspace_root
        self.intent = intent
        self.params = {} if params is None else params
        self.preserve_index = preserve_index
        self.strict_schema = strict_schema

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return sklearn-style params with flattened explicit args."""
        params = super().get_params(deep=False)
        if not deep:
            return params
        params.update(_flatten_nested_params(self.params, prefix="params"))
        return params

    def set_params(self, **params: Any) -> FreeformDataframeTransformer:
        """Update top-level or nested explicit args."""
        nested_updates = {key: value for key, value in params.items() if "__" in key}
        direct_updates = {
            key: value for key, value in params.items() if "__" not in key
        }
        if direct_updates:
            super().set_params(**direct_updates)
        if not nested_updates:
            return self

        params_copy = deepcopy(self.params)
        for key, value in nested_updates.items():
            path_parts = key.split("__")
            if path_parts[0] != "params":
                raise ValueError(f"Unsupported freeform param {key!r}.")
            _set_nested_param_value(params_copy, path_parts[1:], value)
        self.params = params_copy
        return self

    def _executor(self) -> FreeformDataframeExecutor:
        """Create the dedicated executor for this transformer."""
        root = Path(self.workspace_root) if self.workspace_root else None
        return FreeformDataframeExecutor(workspace_root=root)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> FreeformDataframeTransformer:
        """Validate and execute the code once to lock in the output schema."""
        del y
        frame = ensure_dataframe(X, field_name="X")
        result = self._executor().execute(
            self.code,
            frame,
            extra_scope={
                "params": deepcopy(self.params),
                "transformer_args": deepcopy(self.params),
            },
        )
        self.input_columns_ = [str(column) for column in frame.columns]
        self.output_columns_ = list(result.columns)
        self.columns_added_ = list(result.columns_added)
        self.columns_removed_ = list(result.columns_removed)
        self.fit_stdout_ = result.stdout
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replay the validated freeform code against a fresh dataframe."""
        check_is_fitted(
            self,
            attributes=(
                "input_columns_",
                "output_columns_",
                "columns_added_",
                "columns_removed_",
            ),
        )
        frame = ensure_dataframe(X, field_name="X")
        missing_columns = [
            column for column in self.input_columns_ if column not in frame.columns
        ]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                f"Input dataframe is missing columns required by the freeform transformer: {missing_text}."
            )
        result = self._executor().execute(
            self.code,
            frame,
            extra_scope={
                "params": deepcopy(self.params),
                "transformer_args": deepcopy(self.params),
            },
        )
        if self.strict_schema and list(result.columns) != list(self.output_columns_):
            raise ValueError(
                "Freeform transformer output columns changed between fit and transform. "
                "Use deterministic code or disable strict_schema explicitly."
            )
        transformed = result.dataframe.copy()
        if self.strict_schema:
            transformed = transformed.loc[:, self.output_columns_]
        if self.preserve_index:
            return transformed
        return transformed.reset_index(drop=True)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        """Return the fit-time output column names."""
        del input_features
        check_is_fitted(self, attributes=("output_columns_",))
        return np.asarray(self.output_columns_, dtype=object)


class FreeformCodeCollection(WorkspaceToolCollection):
    """Freeform dataframe execution helpers with a focused DS runtime."""

    name = "freeform"
    description = (
        "Execute validated freeform Python against a stored dataframe handle using "
        "a dedicated executor, which exposes a mutable `df` plus a broader data "
        "science runtime including pandas, numpy, scipy, sklearn, LightGBM, "
        "Optuna, joblib, and simple `/workspace` path resolution. Inside freeform "
        "code, convert `/workspace/...` paths with `workspace_path(...)` or "
        "`resolve_workspace_path(...)` before passing them to pandas, joblib, or "
        "other host-side file APIs. The collection also supports fitting reusable "
        "freeform transformers that can participate in sklearn-style pipelines."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the collection and its dedicated dataframe executor.

        Args:
            *args: Positional arguments forwarded to the workspace helper base.
            **kwargs: Keyword arguments forwarded to the workspace helper base.
        """
        super().__init__(*args, **kwargs)
        self._executor = FreeformDataframeExecutor(
            workspace_root=self._os_access.host_workspace_root
        )

    def _resolve_host_path(self, path: str) -> Path:
        """Resolve a virtual or relative path into the host workspace."""
        return self._os_access.to_host_path(path)

    def _feature_frame(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        require_target_column: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Split a stored dataframe into feature and optional target frames."""
        dataframe = self._get_dataframe(dataframe_handle)
        if target_column is None:
            return dataframe.copy(), None
        if target_column not in dataframe.columns:
            if require_target_column:
                raise ValueError(
                    f"Target column {target_column!r} was not found in the dataframe."
                )
            return dataframe.copy(), None
        return dataframe.drop(columns=[target_column]).copy(), dataframe[
            target_column
        ].copy()

    def _get_freeform_transformer(
        self,
        freeform_transformer_handle: str,
    ) -> StoredFreeformTransformer:
        """Fetch a stored reusable freeform transformer artifact."""
        return self._object_store.get(
            freeform_transformer_handle,
            expected_type=StoredFreeformTransformer,
        )

    @tool
    def run_dataframe_code(self, dataframe_handle: str, code: str) -> dict[str, Any]:
        """Run freeform Python against a stored dataframe with DS libraries like Optuna and workspace path helpers.

        Args:
            dataframe_handle (str): Handle pointing to the source dataframe.
            code (str): Python source that reads or mutates ``df`` and leaves the
                final dataframe assigned back to ``df``. The runtime also exposes
                broader DS libraries such as Optuna and LightGBM plus
                ``workspace_path(...)``, ``resolve_workspace_path(...)``, and
                ``workspace_dir`` for files under `/workspace`. When you need to
                pass a workspace file into pandas, joblib, or another host-side
                library, convert the virtual path first instead of using a raw
                ``'/workspace/...'`` string. For the most reliable nested-string
                behavior, prefer storing the freeform source in a named multiline
                variable before calling this helper. Avoid escape-heavy inline
                snippets like ``print(f'\\nColumn: {name}')`` because outer parsing
                may consume ``\n`` before the inner freeform code runs. Prefer
                separate ``print()`` calls, or use ``\\\\n`` when the inner code
                truly needs a literal backslash escape.

        Returns:
            dict[str, Any]: Summary payload with the new dataframe handle, result
            shape, the columns added or removed by the submitted code, and any
            captured standard output emitted while the freeform code ran.

        Examples:
            freeform_code = \"\"\"
            study = optuna.create_study(direction='maximize')
            df['loss_ratio'] = df['loss'] / df['premium']
            print('')
            print('Created loss_ratio')
            \"\"\"
            result = run_dataframe_code(df_handle, freeform_code)
            print(result["dataframe_handle"])
            # Returns:
            # {
            #     "dataframe_handle": "df_2",
            #     "rows": 1000,
            #     "column_count": 18,
            #     "columns_added": ["loss_ratio"],
            #     "columns_removed": [],
            #     "stdout": "Created loss_ratio\\n"
            # }

            feature_code = \"\"\"
            df['loss_ratio'] = np.where(df['premium'] > 0, df['loss'] / df['premium'], np.nan)
            df['log_premium'] = np.log1p(df['premium'])
            \"\"\"
            feature_result = run_dataframe_code(df_handle, feature_code)
            print(feature_result["columns_added"])

            pipeline_code = \"\"\"
            reference = pd.read_csv(workspace_path('/workspace/reference.csv'))
            df['reference_rows'] = len(reference)
            \"\"\"
            pipeline_result = run_dataframe_code(df_handle, pipeline_code)
            print(pipeline_result["stdout"])
        """
        source_dataframe = self._get_dataframe(dataframe_handle)
        execution_result = self._executor.execute(code, source_dataframe)

        # Store the result under a new handle so earlier handles remain stable.
        result_handle = self._object_store.put(execution_result.dataframe, prefix="df")
        return {
            "dataframe_handle": result_handle,
            "rows": execution_result.rows,
            "column_count": len(execution_result.columns),
            "columns": execution_result.columns,
            "columns_added": execution_result.columns_added,
            "columns_removed": execution_result.columns_removed,
            "stdout": execution_result.stdout,
        }

    @tool
    def fit_freeform_transformer(
        self,
        dataframe_handle: str,
        code: str,
        *,
        target_column: str | None = None,
        intent: str = "feature_engineering",
        args: dict[str, Any] | None = None,
        preserve_index: bool = True,
        strict_schema: bool = True,
    ) -> str:
        """Fit a reusable freeform transformer from a dataframe and code snippet.

        Args:
            dataframe_handle (str): Handle pointing to the fit dataframe.
            code (str): Freeform Python that reads and writes `df`. Explicit
                transformer args are exposed as `params`. Prefer assigning this
                source to a named multiline variable before calling the helper,
                especially when the code contains strings, diagnostics, or
                escaped characters.
            target_column (str | None): Optional target column to exclude from fit.
            intent (str): User-facing intent label for inspection and UX guidance.
            args (dict[str, Any] | None): Explicit arguments exposed to the code as
                `params`.
            preserve_index (bool): Whether transformed outputs preserve their index.
            strict_schema (bool): Whether transform-time columns must match fit-time
                output columns exactly.

        Returns:
            str: Handle for the fitted freeform transformer artifact.

        Examples:
            transformer_code = \"\"\"
            df['loss_ratio'] = np.where(
                df['premium'] > params['ratio_floor'],
                df['loss'] / df['premium'],
                np.nan,
            )
            \"\"\"
            freeform_handle = fit_freeform_transformer(
                df_handle,
                transformer_code,
                args={"ratio_floor": 0.0},
                target_column="target",
            )
        """
        feature_frame, _ = self._feature_frame(
            dataframe_handle,
            target_column=target_column,
        )
        estimator = FreeformDataframeTransformer(
            code=code,
            workspace_root=str(self._os_access.host_workspace_root),
            intent=intent,
            params=dict(args or {}),
            preserve_index=preserve_index,
            strict_schema=strict_schema,
        )
        estimator.fit(feature_frame)
        artifact = StoredFreeformTransformer(
            estimator=estimator,
            spec={
                "code": code,
                "intent": intent,
                "args": dict(args or {}),
                "strict_schema": strict_schema,
                "output": {"preserve_index": preserve_index},
            },
            input_columns=list(estimator.input_columns_),
            output_columns=list(estimator.output_columns_),
            target_column=target_column,
            preserve_index=preserve_index,
            code=code,
            intent=intent,
            strict_schema=strict_schema,
            columns_added=list(estimator.columns_added_),
            columns_removed=list(estimator.columns_removed_),
            fit_stdout=str(estimator.fit_stdout_),
            args=dict(args or {}),
        )
        return self._object_store.put(artifact, prefix="freeform")

    @tool
    def transform_with_freeform_transformer(
        self,
        dataframe_handle: str,
        freeform_transformer_handle: str,
        *,
        include_target: bool = False,
    ) -> str:
        """Transform a stored dataframe with a reusable freeform transformer.

        Args:
            dataframe_handle (str): Input dataframe handle.
            freeform_transformer_handle (str): Handle pointing to the fitted
                freeform transformer artifact.
            include_target (bool): Whether to append the stored target column back.

        Returns:
            str: Handle for the transformed dataframe.

        Examples:
            engineered_handle = transform_with_freeform_transformer(
                df_handle,
                freeform_handle,
                include_target=True,
            )
        """
        artifact = self._get_freeform_transformer(freeform_transformer_handle)
        feature_frame, target_series = self._feature_frame(
            dataframe_handle,
            target_column=artifact.target_column,
            require_target_column=include_target,
        )
        transformed_frame = artifact.estimator.transform(feature_frame)
        if include_target:
            transformed_frame = append_target_column(
                transformed_frame,
                target_column=artifact.target_column,
                target_series=target_series,
                preserve_index=artifact.preserve_index,
            )
        return self._object_store.put(transformed_frame, prefix="df")

    @tool
    def fit_transform_with_freeform_transformer(
        self,
        dataframe_handle: str,
        code: str,
        *,
        target_column: str | None = None,
        intent: str = "feature_engineering",
        args: dict[str, Any] | None = None,
        preserve_index: bool = True,
        strict_schema: bool = True,
        include_target: bool = False,
    ) -> dict[str, Any]:
        """Fit a reusable freeform transformer and transform the same dataframe.

        Args:
            dataframe_handle (str): Input dataframe handle.
            code (str): Freeform Python that reads and writes `df`. Explicit
                transformer args are exposed as `params`.
            target_column (str | None): Optional target column to exclude from fit.
            intent (str): User-facing intent label for inspection and UX guidance.
            args (dict[str, Any] | None): Explicit arguments exposed to the code as
                `params`.
            preserve_index (bool): Whether transformed outputs preserve their index.
            strict_schema (bool): Whether transform-time columns must match fit-time
                output columns exactly.
            include_target (bool): Whether to append the target column back.

        Returns:
            dict[str, Any]: Handles for the fitted transformer and transformed
                dataframe. The payload includes both the specific
                `freeform_transformer_handle` key and a generic
                `transformer_handle` alias so callers can reuse a common
                fit-transform pattern across transformer types.

        Examples:
            result = fit_transform_with_freeform_transformer(
                df_handle,
                "df['loss_ratio'] = df['loss'] / params['denominator']",
                args={"denominator": 1.0},
                target_column="target",
                include_target=True,
            )
            # Returns:
            # {
            #     "freeform_transformer_handle": "freeform_1",
            #     "transformer_handle": "freeform_1",
            #     "dataframe_handle": "df_2",
            #     "transformer_type": "freeform"
            # }
        """
        transformer_handle = self.fit_freeform_transformer(
            dataframe_handle,
            code,
            target_column=target_column,
            intent=intent,
            args=args,
            preserve_index=preserve_index,
            strict_schema=strict_schema,
        )
        transformed_handle = self.transform_with_freeform_transformer(
            dataframe_handle,
            transformer_handle,
            include_target=include_target,
        )
        return {
            "freeform_transformer_handle": transformer_handle,
            "transformer_handle": transformer_handle,
            "dataframe_handle": transformed_handle,
            "transformer_type": "freeform",
        }

    @tool
    def inspect_freeform_transformer(
        self,
        freeform_transformer_handle: str,
    ) -> dict[str, Any]:
        """Return a JSON-friendly summary of a reusable freeform transformer.

        Args:
            freeform_transformer_handle (str): Handle pointing to a fitted reusable
                freeform transformer artifact.

        Returns:
            dict[str, Any]: Summary of the stored freeform transformer.

        Examples:
            print(inspect_freeform_transformer(freeform_handle))
            # Returns:
            # {
            #     "type": "StoredFreeformTransformer",
            #     "intent": "feature_engineering",
            #     "args": {"ratio_floor": 0.0},
            #     "output_columns": ["loss_ratio"]
            # }
        """
        return self._get_freeform_transformer(
            freeform_transformer_handle
        ).to_json_summary()

    @tool
    def list_freeform_transformer_features(
        self,
        freeform_transformer_handle: str,
    ) -> list[str]:
        """List output features produced by a reusable freeform transformer.

        Args:
            freeform_transformer_handle (str): Handle pointing to a fitted reusable
                freeform transformer artifact.

        Returns:
            list[str]: Output feature names in transform order.

        Examples:
            print(list_freeform_transformer_features(freeform_handle))
        """
        artifact = self._get_freeform_transformer(freeform_transformer_handle)
        return list(artifact.output_columns)

    @tool
    def save_freeform_transformer(
        self,
        freeform_transformer_handle: str,
        path: str,
    ) -> str:
        """Persist a reusable freeform transformer artifact to the workspace.

        Args:
            freeform_transformer_handle (str): Handle pointing to a fitted reusable
                freeform transformer artifact.
            path (str): Relative or `/workspace` destination path.

        Returns:
            str: Virtual path to the saved artifact.

        Examples:
            save_freeform_transformer(
                freeform_handle,
                "/workspace/output/freeform_transformer.joblib",
            )
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = self._get_freeform_transformer(freeform_transformer_handle)
        joblib.dump(artifact, host_path)
        self._os_access.record_host_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool
    def load_freeform_transformer(self, path: str) -> str:
        """Load a reusable freeform transformer artifact from the workspace.

        Args:
            path (str): Relative or `/workspace` path to a saved artifact.

        Returns:
            str: Handle for the loaded freeform transformer artifact.

        Examples:
            freeform_handle = load_freeform_transformer(
                "/workspace/output/freeform_transformer.joblib"
            )
        """
        artifact = joblib.load(self._resolve_host_path(path))
        if not isinstance(artifact, StoredFreeformTransformer):
            raise TypeError("Loaded artifact is not a StoredFreeformTransformer.")
        return self._object_store.put(artifact, prefix="freeform")


__all__ = [
    "FreeformCodeCollection",
    "FreeformDataframeTransformer",
    "StoredFreeformTransformer",
]
