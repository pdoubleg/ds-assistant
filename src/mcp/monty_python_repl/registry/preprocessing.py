"""Sklearn-backed preprocessing helpers for the Monty Python REPL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ..filesystem import HostWorkspaceOSAccess
from ..core.registry import ObjectStore, ToolCollection, tool
from ..core.registry import safe_json_value
from ..support.sklearn_pipeline import StoredSklearnStageArtifact

_SUPPORTED_STEP_KINDS = {
    "passthrough",
    "simple_imputer",
    "one_hot_encoder",
    "ordinal_encoder",
    "standard_scaler",
    "min_max_scaler",
    "robust_scaler",
    "power_transformer",
    "quantile_transformer",
    "variance_threshold",
    "log1p_transformer",
}
_SUPPORTED_SELECTORS = {"all", "numeric", "categorical"}


@dataclass(slots=True)
class ResolvedPreprocessingGroup:
    """Resolved preprocessing branch metadata.

    Args:
        name (str): Stable group name used inside the sklearn transformer.
        columns (list[str]): Concrete input columns assigned to the branch.
        steps (list[dict[str, Any]]): Normalized step definitions for the branch.
    """

    name: str
    columns: list[str]
    steps: list[dict[str, Any]]


@dataclass(slots=True)
class StoredPreprocessor(StoredSklearnStageArtifact):
    """Host-side sklearn preprocessor stored behind a Monty-safe handle.

    Args:
        estimator (ColumnTransformer): Fitted sklearn column transformer.
        spec (dict[str, Any]): Normalized preprocessing spec used to build it.
        input_columns (list[str]): Feature columns seen during fitting.
        output_columns (list[str]): Stable output column names after transform.
        groups (list[ResolvedPreprocessingGroup]): Resolved branch metadata.
        target_column (str | None): Optional target excluded during fitting.
        preserve_index (bool): Whether transformed dataframes keep source index.
    """

    groups: list[ResolvedPreprocessingGroup] = field(default_factory=list)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection.

        Args:
            max_items (int): Maximum branch or feature previews to include.
            max_chars (int): Maximum string length for nested summaries.

        Returns:
            dict[str, Any]: Summary payload suitable for ``inspect_handle``.
        """
        summary = self._base_summary(
            artifact_type="StoredPreprocessor",
            max_items=max_items,
            max_chars=max_chars,
        )
        summary["groups"] = [
            {
                "name": group.name,
                "column_count": len(group.columns),
                "columns": group.columns[:max_items],
                "steps": [
                    safe_json_value(
                        step,
                        max_items=max_items,
                        max_chars=max_chars,
                    )
                    for step in group.steps[:max_items]
                ],
            }
            for group in self.groups[:max_items]
        ]
        return summary


def _string_list(values: Sequence[Any], *, field_name: str) -> list[str]:
    """Coerce a sequence of values into a list of strings.

    Args:
        values (Sequence[Any]): Raw values to coerce.
        field_name (str): Human-readable field name for validation errors.

    Returns:
        list[str]: Coerced string list.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of column names, not a string.")
    return [str(value) for value in values]


def _normalize_step_spec(step: Any) -> dict[str, Any]:
    """Normalize a single step spec into a dict with a supported ``kind`` key.

    Args:
        step (Any): Raw step payload.

    Returns:
        dict[str, Any]: Normalized step configuration.

    Raises:
        ValueError: If the step is malformed or unsupported.
    """
    if isinstance(step, str):
        normalized = {"kind": step}
    elif isinstance(step, Mapping):
        normalized = {str(key): value for key, value in step.items()}
    else:
        raise ValueError("Each preprocessing step must be a string or mapping.")

    kind = str(normalized.get("kind", "")).strip()
    if not kind:
        raise ValueError("Each preprocessing step must declare a non-empty 'kind'.")
    if kind not in _SUPPORTED_STEP_KINDS:
        supported = ", ".join(sorted(_SUPPORTED_STEP_KINDS))
        raise ValueError(
            f"Unsupported preprocessing step kind {kind!r}. Supported kinds: {supported}."
        )

    normalized["kind"] = kind
    return normalized


def _normalize_selector_spec(selector: Any) -> dict[str, Any]:
    """Normalize a column selector into a consistent mapping shape.

    Args:
        selector (Any): Raw selector payload.

    Returns:
        dict[str, Any]: Normalized selector mapping.
    """
    if isinstance(selector, Mapping):
        normalized = {str(key): value for key, value in selector.items()}
        if "columns" in normalized:
            normalized["columns"] = _string_list(
                normalized["columns"],
                field_name="columns",
            )
        if "exclude" in normalized:
            normalized["exclude"] = _string_list(
                normalized["exclude"],
                field_name="exclude",
            )
        if "selector" in normalized:
            selector_name = str(normalized["selector"])
            if selector_name not in _SUPPORTED_SELECTORS:
                supported = ", ".join(sorted(_SUPPORTED_SELECTORS))
                raise ValueError(
                    f"Unsupported selector {selector_name!r}. Supported selectors: {supported}."
                )
            normalized["selector"] = selector_name
        if "columns" not in normalized and "selector" not in normalized:
            raise ValueError(
                "Selector mappings must include either 'columns' or 'selector'."
            )
        return normalized

    if isinstance(selector, str):
        if selector in _SUPPORTED_SELECTORS:
            return {"selector": selector}
        return {"columns": [selector]}

    if isinstance(selector, Sequence):
        return {"columns": _string_list(selector, field_name="columns")}

    raise ValueError("Column selectors must be a string, list of strings, or mapping.")


def _normalize_group_spec(
    group_name: str | None,
    group_spec: Any,
    *,
    default_selector: Any | None = None,
) -> dict[str, Any]:
    """Normalize one top-level group entry.

    Args:
        group_name (str | None): Default group name when supplied by the parent key.
        group_spec (Any): Raw group payload.
        default_selector (Any | None): Fallback selector when columns are omitted.

    Returns:
        dict[str, Any]: Normalized group mapping.
    """
    if isinstance(group_spec, Sequence) and not isinstance(group_spec, (str, bytes)):
        raw_group = {"steps": list(group_spec)}
    elif isinstance(group_spec, Mapping):
        raw_group = {str(key): value for key, value in group_spec.items()}
    else:
        raise ValueError(
            "Each preprocessing group must be a mapping or a list of steps."
        )

    normalized_name = str(raw_group.get("name", group_name or "")).strip()
    if not normalized_name:
        raise ValueError("Each preprocessing group must declare a non-empty name.")

    raw_steps = raw_group.get("steps", [])
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise ValueError("Group 'steps' must be a list of preprocessing step mappings.")

    selector = raw_group.get("columns", default_selector)
    if selector is None:
        raise ValueError(f"Group {normalized_name!r} must define a column selector.")

    return {
        "name": normalized_name,
        "columns": _normalize_selector_spec(selector),
        "steps": [_normalize_step_spec(step) for step in raw_steps],
    }


def normalize_preprocessing_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a declarative preprocessing spec.

    Args:
        spec (Mapping[str, Any]): User-provided preprocessing spec.

    Returns:
        dict[str, Any]: Normalized preprocessing spec with stable defaults.
    """
    raw_spec = {str(key): value for key, value in spec.items()}
    unknown_keys = set(raw_spec) - {
        "numeric",
        "categorical",
        "groups",
        "remainder",
        "output",
    }
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown preprocessing spec keys: {names}.")

    groups: list[dict[str, Any]] = []
    if "numeric" in raw_spec and raw_spec["numeric"] is not None:
        groups.append(
            _normalize_group_spec(
                "numeric",
                raw_spec["numeric"],
                default_selector={"selector": "numeric"},
            )
        )
    if "categorical" in raw_spec and raw_spec["categorical"] is not None:
        groups.append(
            _normalize_group_spec(
                "categorical",
                raw_spec["categorical"],
                default_selector={"selector": "categorical"},
            )
        )

    raw_groups = raw_spec.get("groups", [])
    if raw_groups:
        if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
            raise ValueError(
                "The top-level 'groups' entry must be a list of group mappings."
            )
        for raw_group in raw_groups:
            groups.append(_normalize_group_spec(None, raw_group))

    if not groups:
        raise ValueError("The preprocessing spec must define at least one group.")

    remainder = str(raw_spec.get("remainder", "drop"))
    if remainder not in {"drop", "passthrough"}:
        raise ValueError(
            "The top-level 'remainder' must be either 'drop' or 'passthrough'."
        )

    raw_output = raw_spec.get("output", {})
    if raw_output and not isinstance(raw_output, Mapping):
        raise ValueError(
            "The top-level 'output' entry must be a mapping when provided."
        )
    output = {
        "dense": bool(raw_output.get("dense", True)),
        "preserve_index": bool(raw_output.get("preserve_index", True)),
    }

    return {"groups": groups, "remainder": remainder, "output": output}


def _resolve_selector_columns(
    dataframe: pd.DataFrame,
    selector: Mapping[str, Any],
) -> list[str]:
    """Resolve a normalized selector mapping into concrete dataframe columns.

    Args:
        dataframe (pd.DataFrame): Feature dataframe used for fitting or transform.
        selector (Mapping[str, Any]): Normalized selector mapping.

    Returns:
        list[str]: Resolved column names.
    """
    if "columns" in selector:
        columns = [str(column) for column in selector["columns"]]
    else:
        selector_name = str(selector["selector"])
        if selector_name == "all":
            columns = [str(column) for column in dataframe.columns]
        elif selector_name == "numeric":
            columns = [
                str(column)
                for column in dataframe.select_dtypes(
                    include=["number", "bool"]
                ).columns
            ]
        elif selector_name == "categorical":
            numeric_columns = set(
                dataframe.select_dtypes(include=["number", "bool"]).columns
            )
            columns = [
                str(column)
                for column in dataframe.columns
                if column not in numeric_columns
            ]
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown selector: {selector_name}")

    excluded = {str(column) for column in selector.get("exclude", [])}
    return [column for column in columns if column not in excluded]


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


def _build_step(step_spec: Mapping[str, Any]) -> str | BaseEstimator:
    """Instantiate a sklearn-compatible preprocessing step from a step spec.

    Args:
        step_spec (Mapping[str, Any]): Normalized step definition.

    Returns:
        str | BaseEstimator: Sklearn estimator or the literal ``'passthrough'``.
    """
    kind = str(step_spec["kind"])
    params = {key: value for key, value in step_spec.items() if key != "kind"}

    if kind == "passthrough":
        return "passthrough"
    if kind == "simple_imputer":
        return SimpleImputer(
            strategy=str(params.get("strategy", "mean")),
            fill_value=params.get("fill_value"),
            add_indicator=bool(params.get("add_indicator", False)),
            keep_empty_features=bool(params.get("keep_empty_features", False)),
        )
    if kind == "one_hot_encoder":
        return OneHotEncoder(
            handle_unknown=str(params.get("handle_unknown", "ignore")),
            sparse_output=bool(params.get("sparse_output", False)),
            min_frequency=params.get("min_frequency"),
            max_categories=params.get("max_categories"),
        )
    if kind == "ordinal_encoder":
        encoder_params: dict[str, Any] = {
            "handle_unknown": str(params.get("handle_unknown", "use_encoded_value")),
            "dtype": params.get("dtype", float),
        }
        if encoder_params["handle_unknown"] == "use_encoded_value":
            encoder_params["unknown_value"] = params.get("unknown_value", -1)
        if "encoded_missing_value" in params:
            encoder_params["encoded_missing_value"] = params["encoded_missing_value"]
        return OrdinalEncoder(**encoder_params)
    if kind == "standard_scaler":
        return StandardScaler(
            with_mean=bool(params.get("with_mean", True)),
            with_std=bool(params.get("with_std", True)),
        )
    if kind == "min_max_scaler":
        feature_range = params.get("feature_range", [0, 1])
        if not isinstance(feature_range, Sequence) or isinstance(
            feature_range, (str, bytes)
        ):
            raise ValueError("'feature_range' must be a two-item sequence.")
        return MinMaxScaler(
            feature_range=(float(feature_range[0]), float(feature_range[1])),
            clip=bool(params.get("clip", False)),
        )
    if kind == "robust_scaler":
        quantile_range = params.get("quantile_range", [25.0, 75.0])
        if not isinstance(quantile_range, Sequence) or isinstance(
            quantile_range,
            (str, bytes),
        ):
            raise ValueError("'quantile_range' must be a two-item sequence.")
        return RobustScaler(
            with_centering=bool(params.get("with_centering", True)),
            with_scaling=bool(params.get("with_scaling", True)),
            quantile_range=(
                float(quantile_range[0]),
                float(quantile_range[1]),
            ),
        )
    if kind == "power_transformer":
        return PowerTransformer(
            method=str(params.get("method", "yeo-johnson")),
            standardize=bool(params.get("standardize", True)),
        )
    if kind == "quantile_transformer":
        return QuantileTransformer(
            n_quantiles=int(params.get("n_quantiles", 1000)),
            output_distribution=str(params.get("output_distribution", "uniform")),
            subsample=int(params.get("subsample", 10_000)),
            random_state=params.get("random_state", 0),
        )
    if kind == "variance_threshold":
        return VarianceThreshold(threshold=float(params.get("threshold", 0.0)))
    if kind == "log1p_transformer":
        return FunctionTransformer(
            np.log1p,
            validate=False,
            feature_names_out="one-to-one",
        )

    raise ValueError(f"Unsupported preprocessing step kind {kind!r}.")


def _build_group_transformer(
    step_specs: Sequence[Mapping[str, Any]],
) -> str | BaseEstimator:
    """Build one sklearn transformer branch from normalized step specs.

    Args:
        step_specs (Sequence[Mapping[str, Any]]): Ordered step configurations.

    Returns:
        str | BaseEstimator: Transformer for a single ColumnTransformer branch.
    """
    if not step_specs:
        return "passthrough"

    built_steps = []
    for index, step_spec in enumerate(step_specs, start=1):
        built_step = _build_step(step_spec)
        if built_step == "passthrough":
            if len(step_specs) > 1:
                raise ValueError(
                    "'passthrough' can only be used as the sole step in a group."
                )
            return "passthrough"
        built_steps.append((f"{step_spec['kind']}_{index}", built_step))

    if len(built_steps) == 1:
        return built_steps[0][1]
    return Pipeline(steps=built_steps)


def _as_dataframe(
    transformed: Any,
    *,
    index: pd.Index | None,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Convert a transformer output into a dataframe with stable columns.

    Args:
        transformed (Any): Raw transformer output.
        index (pd.Index | None): Optional dataframe index to preserve.
        columns (Sequence[str]): Output column names.

    Returns:
        pd.DataFrame: Materialized dataframe result.
    """
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    if isinstance(transformed, pd.DataFrame):
        transformed_frame = transformed.copy()
        transformed_frame.columns = list(columns)
        if index is not None:
            transformed_frame.index = index
        return transformed_frame

    array = np.asarray(transformed)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return pd.DataFrame(array, columns=list(columns), index=index)


class PreprocessingCollection(ToolCollection):
    """Declarative sklearn-backed preprocessing helpers for tabular data."""

    name = "preprocessing"
    description = (
        "Build, fit, inspect, persist, and apply sklearn preprocessing pipelines "
        "to stored dataframe handles."
    )

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: ObjectStore,
    ) -> None:
        """Initialize preprocessing helpers.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (ObjectStore): Shared object store for handles.
        """
        self._os_access = os_access
        self._object_store = object_store

    def _resolve_host_path(self, path: str) -> Path:
        """Resolve a virtual or relative path into the host workspace.

        Args:
            path (str): Relative or `/workspace`-scoped path.

        Returns:
            Path: Resolved host path inside the configured workspace.
        """
        return self._os_access.to_host_path(PurePosixPath(path))

    def _get_dataframe(self, dataframe_handle: str) -> pd.DataFrame:
        """Fetch a dataframe from the shared object store.

        Args:
            dataframe_handle (str): Dataframe handle stored earlier in the session.

        Returns:
            pd.DataFrame: Stored dataframe.
        """
        return self._object_store.get(dataframe_handle, expected_type=pd.DataFrame)

    def _get_preprocessor(self, preprocessor_handle: str) -> StoredPreprocessor:
        """Fetch a stored preprocessor from the shared object store.

        Args:
            preprocessor_handle (str): Preprocessor handle stored earlier.

        Returns:
            StoredPreprocessor: Stored preprocessing artifact.
        """
        return self._object_store.get(
            preprocessor_handle,
            expected_type=StoredPreprocessor,
        )

    def _build_estimator(
        self,
        dataframe: pd.DataFrame,
        spec: Mapping[str, Any],
    ) -> tuple[ColumnTransformer, list[ResolvedPreprocessingGroup]]:
        """Build an unfitted sklearn preprocessor and resolved branch metadata.

        Args:
            dataframe (pd.DataFrame): Feature dataframe to inspect for selectors.
            spec (Mapping[str, Any]): Normalized preprocessing spec.

        Returns:
            tuple[ColumnTransformer, list[ResolvedPreprocessingGroup]]:
                Built estimator and resolved group metadata.
        """
        assigned_columns: set[str] = set()
        resolved_groups: list[ResolvedPreprocessingGroup] = []
        transformers: list[tuple[str, str | BaseEstimator, list[str]]] = []

        for group in spec["groups"]:
            columns = _resolve_selector_columns(dataframe, group["columns"])
            if not columns:
                continue

            _require_columns(dataframe, columns)
            duplicate_columns = assigned_columns.intersection(columns)
            if duplicate_columns:
                duplicate_text = ", ".join(sorted(duplicate_columns))
                raise ValueError(
                    f"Columns cannot be assigned to multiple preprocessing groups: {duplicate_text}."
                )

            assigned_columns.update(columns)
            transformers.append(
                (
                    str(group["name"]),
                    _build_group_transformer(group["steps"]),
                    columns,
                )
            )
            resolved_groups.append(
                ResolvedPreprocessingGroup(
                    name=str(group["name"]),
                    columns=columns,
                    steps=[dict(step) for step in group["steps"]],
                )
            )

        if not transformers and spec["remainder"] == "drop":
            raise ValueError(
                "The preprocessing spec did not resolve any usable columns for fitting."
            )

        return (
            ColumnTransformer(
                transformers=transformers,
                remainder=str(spec["remainder"]),
                sparse_threshold=0.0 if spec["output"]["dense"] else 0.3,
                verbose_feature_names_out=True,
            ),
            resolved_groups,
        )

    def _feature_frame(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        require_target_column: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Split a stored dataframe into feature and optional target frames.

        Args:
            dataframe_handle (str): Handle pointing to a stored dataframe.
            target_column (str | None): Optional target column to exclude.
            require_target_column (bool): Whether the target column must exist in
                the input dataframe.

        Returns:
            tuple[pd.DataFrame, pd.Series | None]: Features and optional target.
        """
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

    def _transform_with_artifact(
        self,
        dataframe_handle: str,
        artifact: StoredPreprocessor,
        *,
        include_target: bool = False,
    ) -> pd.DataFrame:
        """Transform a stored dataframe with a fitted preprocessor artifact.

        Args:
            dataframe_handle (str): Input dataframe handle.
            artifact (StoredPreprocessor): Fitted preprocessing artifact.
            include_target (bool): Whether to append the target column back.

        Returns:
            pd.DataFrame: Transformed dataframe with stable columns.
        """
        feature_frame, target_series = self._feature_frame(
            dataframe_handle,
            target_column=artifact.target_column,
            require_target_column=include_target,
        )
        _require_columns(feature_frame, artifact.input_columns)
        transformed = artifact.estimator.transform(
            feature_frame[artifact.input_columns]
        )
        transformed_frame = _as_dataframe(
            transformed,
            index=feature_frame.index if artifact.preserve_index else None,
            columns=artifact.output_columns,
        )

        if include_target and artifact.target_column and target_series is not None:
            transformed_frame[artifact.target_column] = target_series.values
        return transformed_frame

    @tool
    def build_preprocessing_spec(
        self,
        *,
        numeric_steps: list[dict[str, Any]] | None = None,
        categorical_steps: list[dict[str, Any]] | None = None,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        groups: list[dict[str, Any]] | None = None,
        remainder: str = "drop",
        dense_output: bool = True,
        preserve_index: bool = True,
    ) -> dict[str, Any]:
        """Build and validate a normalized preprocessing spec.

        Args:
            numeric_steps (list[dict[str, Any]] | None): Ordered numeric branch steps.
            categorical_steps (list[dict[str, Any]] | None): Ordered categorical
                branch steps.
            numeric_columns (list[str] | None): Optional explicit numeric columns.
            categorical_columns (list[str] | None): Optional explicit categorical
                columns.
            groups (list[dict[str, Any]] | None): Optional additional named group
                definitions.
            remainder (str): ColumnTransformer remainder strategy.
            dense_output (bool): Whether the fitted transformer should prefer dense
                intermediate outputs.
            preserve_index (bool): Whether transformed dataframes keep their index.

        Returns:
            dict[str, Any]: Normalized preprocessing spec ready for fitting.

        Examples:
            spec = build_preprocessing_spec(
                numeric_steps=[{"kind": "simple_imputer", "strategy": "median"}],
                categorical_steps=[
                    {"kind": "simple_imputer", "strategy": "most_frequent"},
                    {"kind": "one_hot_encoder", "handle_unknown": "ignore"},
                ],
            )
            # Returns:
            # {
            #     "numeric": {
            #         "steps": [{"kind": "simple_imputer", "strategy": "median"}]
            #     },
            #     "categorical": {
            #         "steps": [
            #             {"kind": "simple_imputer", "strategy": "most_frequent"},
            #             {"kind": "one_hot_encoder", "handle_unknown": "ignore"}
            #         ]
            #     },
            #     "remainder": "drop",
            #     "output": {"dense": True, "preserve_index": True}
            # }
        """
        raw_spec: dict[str, Any] = {
            "remainder": remainder,
            "output": {
                "dense": dense_output,
                "preserve_index": preserve_index,
            },
        }

        if numeric_steps is not None:
            raw_spec["numeric"] = {"steps": numeric_steps}
            if numeric_columns is not None:
                raw_spec["numeric"]["columns"] = numeric_columns
        if categorical_steps is not None:
            raw_spec["categorical"] = {"steps": categorical_steps}
            if categorical_columns is not None:
                raw_spec["categorical"]["columns"] = categorical_columns
        if groups is not None:
            raw_spec["groups"] = groups

        return normalize_preprocessing_spec(raw_spec)

    @tool
    def fit_preprocessor(
        self,
        dataframe_handle: str,
        spec: dict[str, Any],
        *,
        target_column: str | None = None,
    ) -> str:
        """Fit a sklearn preprocessor from a declarative preprocessing spec.

        Args:
            dataframe_handle (str): Handle pointing to the training dataframe.
            spec (dict[str, Any]): Declarative preprocessing spec.
            target_column (str | None): Optional target column to exclude from fit.

        Returns:
            str: Handle for the fitted preprocessing artifact.

        Examples:
            prep_handle = fit_preprocessor(df_handle, spec, target_column="target")
        """
        normalized_spec = normalize_preprocessing_spec(spec)
        feature_frame, _ = self._feature_frame(
            dataframe_handle,
            target_column=target_column,
        )
        estimator, resolved_groups = self._build_estimator(
            feature_frame, normalized_spec
        )
        estimator.fit(feature_frame)

        if hasattr(estimator, "get_feature_names_out"):
            output_columns = [str(name) for name in estimator.get_feature_names_out()]
        else:  # pragma: no cover - defensive fallback
            output_columns = [
                f"feature_{index}" for index in range(feature_frame.shape[1])
            ]

        artifact = StoredPreprocessor(
            estimator=estimator,
            spec=normalized_spec,
            input_columns=[str(column) for column in feature_frame.columns],
            output_columns=output_columns,
            groups=resolved_groups,
            target_column=target_column,
            preserve_index=bool(normalized_spec["output"]["preserve_index"]),
        )
        return self._object_store.put(artifact, prefix="prep")

    @tool
    def transform_dataframe(
        self,
        dataframe_handle: str,
        preprocessor_handle: str,
        *,
        include_target: bool = False,
    ) -> str:
        """Transform a stored dataframe with a fitted preprocessing artifact.

        Args:
            dataframe_handle (str): Handle pointing to the dataframe to transform.
            preprocessor_handle (str): Handle pointing to a fitted preprocessor.
            include_target (bool): Whether to append the stored target column back.

        Returns:
            str: Handle for the transformed dataframe.

        Examples:
            encoded_handle = transform_dataframe(
                df_handle,
                prep_handle,
                include_target=True,
            )
        """
        artifact = self._get_preprocessor(preprocessor_handle)
        transformed_frame = self._transform_with_artifact(
            dataframe_handle,
            artifact,
            include_target=include_target,
        )
        return self._object_store.put(transformed_frame, prefix="df")

    @tool
    def fit_transform_dataframe(
        self,
        dataframe_handle: str,
        spec: dict[str, Any],
        *,
        target_column: str | None = None,
        include_target: bool = False,
    ) -> dict[str, str]:
        """Fit a preprocessor and immediately transform the same dataframe.

        Args:
            dataframe_handle (str): Handle pointing to the training dataframe.
            spec (dict[str, Any]): Declarative preprocessing spec.
            target_column (str | None): Optional target column to exclude from fit.
            include_target (bool): Whether to append the target column back to the
                transformed dataframe.

        Returns:
            dict[str, str]: Handles for the fitted preprocessor and transformed data.

        Examples:
            result = fit_transform_dataframe(
                df_handle,
                spec,
                target_column="target",
            )
            # Returns:
            # {
            #     "preprocessor_handle": "prep_1",
            #     "dataframe_handle": "df_2"
            # }
        """
        preprocessor_handle = self.fit_preprocessor(
            dataframe_handle,
            spec,
            target_column=target_column,
        )
        dataframe_result_handle = self.transform_dataframe(
            dataframe_handle,
            preprocessor_handle,
            include_target=include_target,
        )
        return {
            "preprocessor_handle": preprocessor_handle,
            "dataframe_handle": dataframe_result_handle,
        }

    @tool
    def inspect_preprocessor(self, preprocessor_handle: str) -> dict[str, Any]:
        """Return a JSON-friendly summary of a fitted preprocessing artifact.

        Args:
            preprocessor_handle (str): Handle pointing to a fitted preprocessor.

        Returns:
            dict[str, Any]: Summary of the stored preprocessor.

        Examples:
            print(inspect_preprocessor(prep_handle))
            # Returns:
            # {
            #     "type": "StoredPreprocessor",
            #     "target_column": "target",
            #     "output_columns": ["num__premium", "cat__segment_A"]
            # }
        """
        artifact = self._get_preprocessor(preprocessor_handle)
        return artifact.to_json_summary()

    @tool
    def list_preprocessor_features(self, preprocessor_handle: str) -> list[str]:
        """List output feature names generated by a fitted preprocessor.

        Args:
            preprocessor_handle (str): Handle pointing to a fitted preprocessor.

        Returns:
            list[str]: Output feature names in transform order.

        Examples:
            print(list_preprocessor_features(prep_handle))
        """
        return list(self._get_preprocessor(preprocessor_handle).output_columns)

    @tool
    def save_preprocessor(self, preprocessor_handle: str, path: str) -> str:
        """Persist a fitted preprocessing artifact to the workspace with joblib.

        Args:
            preprocessor_handle (str): Handle pointing to a fitted preprocessor.
            path (str): Relative or `/workspace` destination path.

        Returns:
            str: Virtual path to the saved artifact.

        Examples:
            save_preprocessor(prep_handle, "/workspace/output/preprocessor.joblib")
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = self._get_preprocessor(preprocessor_handle)
        joblib.dump(artifact, host_path)
        self._os_access.record_host_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool
    def load_preprocessor(self, path: str) -> str:
        """Load a previously saved preprocessing artifact from the workspace.

        Args:
            path (str): Relative or `/workspace` path to a saved joblib artifact.

        Returns:
            str: Handle for the loaded preprocessing artifact.

        Examples:
            prep_handle = load_preprocessor("/workspace/output/preprocessor.joblib")
        """
        artifact = joblib.load(self._resolve_host_path(path))
        if not isinstance(artifact, StoredPreprocessor):
            raise TypeError("Loaded artifact is not a StoredPreprocessor.")
        return self._object_store.put(artifact, prefix="prep")


__all__ = [
    "PreprocessingCollection",
    "ResolvedPreprocessingGroup",
    "StoredPreprocessor",
    "normalize_preprocessing_spec",
]
