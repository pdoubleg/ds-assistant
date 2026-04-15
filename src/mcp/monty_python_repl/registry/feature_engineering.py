"""Deterministic feature engineering helpers for the Monty Python REPL."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..filesystem import HostWorkspaceOSAccess
from ..core.registry import ObjectStore, ToolCollection, tool
from ..core.registry import coerce_group_keys, safe_json_value
from ..support.sklearn_pipeline import (
    StoredSklearnStageArtifact,
    append_target_column,
    ensure_dataframe,
)

_SUPPORTED_FEATURE_KINDS = {
    "ratio",
    "difference",
    "sum",
    "product",
    "absolute",
    "clip",
    "log1p",
    "square",
    "sqrt",
    "datetime_part",
    "string_length",
    "category_frequency",
    "groupby_aggregate",
}
_SUPPORTED_DATETIME_PARTS = {
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "week",
    "quarter",
    "is_month_start",
    "is_month_end",
}
_SUPPORTED_GROUPBY_AGGREGATIONS = {
    "count",
    "nunique",
    "mean",
    "sum",
    "median",
    "min",
    "max",
    "std",
}
_SUPPORTED_CONFLICT_POLICIES = {"error", "overwrite"}
_SUPPORTED_UNKNOWN_GROUP_STRATEGIES = {"null", "constant", "global"}


@dataclass(slots=True)
class ResolvedFeatureDefinition:
    """Resolved feature metadata stored inside a fitted FE artifact.

    Args:
        name (str): Stable user-facing feature name or prefix.
        kind (str): Deterministic feature kind.
        source_columns (list[str]): Source columns required at transform time.
        output_columns (list[str]): Output column names produced by the feature.
        params (dict[str, Any]): Normalized feature parameters.
        lookup_frame (pd.DataFrame | None): Optional lookup table used for
            frequency or aggregation-based features.
        fallback_values (dict[str, Any]): Per-output fallback values for unseen
            categories or groups at transform time.
    """

    name: str
    kind: str
    source_columns: list[str]
    output_columns: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    lookup_frame: pd.DataFrame | None = None
    fallback_values: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StoredFeatureEngineer(StoredSklearnStageArtifact):
    """Host-side deterministic FE artifact stored behind a Monty-safe handle.

    Args:
        spec (dict[str, Any]): Normalized feature-engineering spec used at fit.
        definitions (list[ResolvedFeatureDefinition]): Ordered fitted features.
        input_columns (list[str]): Input feature columns seen during fit.
        engineered_columns (list[str]): New or overwritten feature columns.
        output_columns (list[str]): Final output column order after transform.
        target_column (str | None): Optional target excluded during fit.
        preserve_index (bool): Whether transformed dataframes keep their index.
        conflict_policy (str): Name-collision behavior for engineered columns.
        drop_source_columns (list[str]): Columns removed after feature creation.
    """

    definitions: list[ResolvedFeatureDefinition] = field(default_factory=list)
    engineered_columns: list[str] = field(default_factory=list)
    conflict_policy: str = "error"
    drop_source_columns: list[str] = field(default_factory=list)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection.

        Args:
            max_items (int): Maximum number of preview items to include.
            max_chars (int): Maximum nested string length to retain.

        Returns:
            dict[str, Any]: Summary payload suitable for ``inspect_handle``.
        """
        summary = self._base_summary(
            artifact_type="StoredFeatureEngineer",
            max_items=max_items,
            max_chars=max_chars,
        )
        summary.update(
            {
                "engineered_column_count": len(self.engineered_columns),
                "engineered_columns": self.engineered_columns[:max_items],
                "conflict_policy": self.conflict_policy,
                "drop_source_columns": self.drop_source_columns[:max_items],
                "features": [
                    {
                        "name": definition.name,
                        "kind": definition.kind,
                        "source_columns": definition.source_columns[:max_items],
                        "output_columns": definition.output_columns[:max_items],
                        "params": safe_json_value(
                            definition.params,
                            max_items=max_items,
                            max_chars=max_chars,
                        ),
                        "has_lookup": definition.lookup_frame is not None,
                    }
                    for definition in self.definitions[:max_items]
                ],
            }
        )
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
        raise ValueError(f"{field_name} must be a list of strings, not a string.")
    return [str(value) for value in values]


def _require_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise an error when required columns are missing from a dataframe.

    Args:
        dataframe (pd.DataFrame): Dataframe to validate.
        columns (Sequence[str]): Required columns.
    """
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}.")


def _numeric_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric view of a dataframe column.

    Args:
        dataframe (pd.DataFrame): Source dataframe.
        column (str): Column name.

    Returns:
        pd.Series: Numeric series with invalid values coerced to ``NaN``.
    """
    return pd.to_numeric(dataframe[column], errors="coerce")


def _normalize_datetime_parts(feature: Mapping[str, Any]) -> list[str]:
    """Normalize datetime parts into a validated list.

    Args:
        feature (Mapping[str, Any]): Raw datetime feature mapping.

    Returns:
        list[str]: Requested datetime parts.
    """
    raw_parts = feature.get("parts")
    if raw_parts is None:
        raw_part = feature.get("part")
        if raw_part is None:
            raise ValueError("Datetime features must define 'part' or 'parts'.")
        parts = [str(raw_part)]
    else:
        parts = _string_list(raw_parts, field_name="parts")

    invalid_parts = sorted(set(parts) - _SUPPORTED_DATETIME_PARTS)
    if invalid_parts:
        supported = ", ".join(sorted(_SUPPORTED_DATETIME_PARTS))
        invalid_text = ", ".join(invalid_parts)
        raise ValueError(
            f"Unsupported datetime parts: {invalid_text}. Supported parts: {supported}."
        )
    return parts


def _normalize_groupby_aggregations(feature: Mapping[str, Any]) -> list[str]:
    """Normalize groupby aggregations into a validated list.

    Args:
        feature (Mapping[str, Any]): Raw groupby feature mapping.

    Returns:
        list[str]: Requested aggregation names.
    """
    raw_aggs = feature.get("aggregations")
    if raw_aggs is None:
        raw_agg = feature.get("aggregation")
        if raw_agg is None:
            raise ValueError(
                "Groupby aggregate features must define 'aggregation' or 'aggregations'."
            )
        aggregations = [str(raw_agg)]
    else:
        aggregations = _string_list(raw_aggs, field_name="aggregations")

    invalid = sorted(set(aggregations) - _SUPPORTED_GROUPBY_AGGREGATIONS)
    if invalid:
        supported = ", ".join(sorted(_SUPPORTED_GROUPBY_AGGREGATIONS))
        invalid_text = ", ".join(invalid)
        raise ValueError(
            f"Unsupported groupby aggregations: {invalid_text}. Supported values: {supported}."
        )
    return aggregations


def _normalize_feature_spec(feature: Any) -> dict[str, Any]:
    """Validate and normalize one feature definition.

    Args:
        feature (Any): Raw feature payload.

    Returns:
        dict[str, Any]: Normalized feature definition.
    """
    if not isinstance(feature, Mapping):
        raise ValueError("Each engineered feature must be provided as a mapping.")

    normalized = {str(key): value for key, value in feature.items()}
    name = str(normalized.get("name", "")).strip()
    kind = str(normalized.get("kind", "")).strip()
    if not name:
        raise ValueError("Each engineered feature must declare a non-empty 'name'.")
    if kind not in _SUPPORTED_FEATURE_KINDS:
        supported = ", ".join(sorted(_SUPPORTED_FEATURE_KINDS))
        raise ValueError(
            f"Unsupported engineered feature kind {kind!r}. Supported kinds: {supported}."
        )

    base: dict[str, Any] = {"name": name, "kind": kind}
    if kind in {"ratio", "difference", "sum", "product"}:
        columns = _string_list(normalized.get("columns", []), field_name="columns")
        if len(columns) != 2:
            raise ValueError(
                f"Feature {name!r} must define exactly two source columns for {kind!r}."
            )
        base["columns"] = columns
        return base

    if kind in {
        "absolute",
        "clip",
        "log1p",
        "square",
        "sqrt",
        "string_length",
        "category_frequency",
    }:
        column = normalized.get("column")
        if column is None:
            raise ValueError(f"Feature {name!r} must define a 'column' for {kind!r}.")
        base["column"] = str(column)
        if kind == "clip":
            base["lower"] = normalized.get("lower")
            base["upper"] = normalized.get("upper")
        if kind == "category_frequency":
            base["normalize"] = bool(normalized.get("normalize", True))
            base["fill_value"] = normalized.get("fill_value", 0.0)
            base["dropna"] = bool(normalized.get("dropna", False))
        return base

    if kind == "datetime_part":
        column = normalized.get("column")
        if column is None:
            raise ValueError(
                f"Feature {name!r} must define a 'column' for datetime parts."
            )
        base["column"] = str(column)
        base["parts"] = _normalize_datetime_parts(normalized)
        return base

    if kind == "groupby_aggregate":
        keys = normalized.get("keys")
        if keys is None:
            keys = normalized.get("groupby")
        if keys is None:
            raise ValueError(f"Feature {name!r} must define grouping keys.")
        base["keys"] = _string_list(coerce_group_keys(keys), field_name="keys")
        base["aggregations"] = _normalize_groupby_aggregations(normalized)
        source_column = normalized.get("source_column")
        if source_column is not None:
            base["source_column"] = str(source_column)
        elif any(aggregation != "count" for aggregation in base["aggregations"]):
            raise ValueError(
                f"Feature {name!r} requires 'source_column' for non-count aggregations."
            )
        unknown_group_strategy = str(normalized.get("unknown_group_strategy", "null"))
        if unknown_group_strategy not in _SUPPORTED_UNKNOWN_GROUP_STRATEGIES:
            supported = ", ".join(sorted(_SUPPORTED_UNKNOWN_GROUP_STRATEGIES))
            raise ValueError(
                f"Unsupported unknown_group_strategy {unknown_group_strategy!r}. "
                f"Supported values: {supported}."
            )
        base["unknown_group_strategy"] = unknown_group_strategy
        if "fill_value" in normalized:
            base["fill_value"] = normalized["fill_value"]
        return base

    raise ValueError(f"Unsupported engineered feature kind {kind!r}.")


def normalize_feature_engineering_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a feature-engineering spec.

    Args:
        spec (Mapping[str, Any]): Raw FE spec.

    Returns:
        dict[str, Any]: Normalized FE spec.
    """
    raw_spec = {str(key): value for key, value in spec.items()}
    unknown_keys = set(raw_spec) - {
        "features",
        "output",
        "conflict_policy",
        "drop_source_columns",
    }
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown feature-engineering spec keys: {names}.")

    raw_features = raw_spec.get("features", [])
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        raise ValueError(
            "The top-level 'features' entry must be a list of feature mappings."
        )
    features = [_normalize_feature_spec(feature) for feature in raw_features]
    if not features:
        raise ValueError(
            "The feature-engineering spec must define at least one feature."
        )

    raw_output = raw_spec.get("output", {})
    if raw_output and not isinstance(raw_output, Mapping):
        raise ValueError(
            "The top-level 'output' entry must be a mapping when provided."
        )

    conflict_policy = str(raw_spec.get("conflict_policy", "error"))
    if conflict_policy not in _SUPPORTED_CONFLICT_POLICIES:
        supported = ", ".join(sorted(_SUPPORTED_CONFLICT_POLICIES))
        raise ValueError(
            f"Unsupported conflict_policy {conflict_policy!r}. Supported values: {supported}."
        )

    drop_source_columns = raw_spec.get("drop_source_columns", [])
    if drop_source_columns:
        drop_source_columns = _string_list(
            drop_source_columns,
            field_name="drop_source_columns",
        )
    else:
        drop_source_columns = []

    return {
        "features": features,
        "output": {
            "preserve_index": bool(raw_output.get("preserve_index", True)),
        },
        "conflict_policy": conflict_policy,
        "drop_source_columns": drop_source_columns,
    }


def _build_output_columns(
    name: str, *, count: int, suffixes: Sequence[str] | None = None
) -> list[str]:
    """Build stable output columns for a resolved feature.

    Args:
        name (str): User-provided feature name.
        count (int): Number of output columns produced.
        suffixes (Sequence[str] | None): Optional suffixes appended to the name.

    Returns:
        list[str]: Output column names.
    """
    if count == 1 and not suffixes:
        return [name]
    if suffixes is None:
        suffixes = [str(index) for index in range(1, count + 1)]
    return [f"{name}__{suffix}" for suffix in suffixes]


def _global_aggregate_value(
    dataframe: pd.DataFrame,
    *,
    source_column: str | None,
    aggregation: str,
) -> Any:
    """Compute a global fallback aggregate for unseen groups.

    Args:
        dataframe (pd.DataFrame): Fit dataframe.
        source_column (str | None): Optional aggregated source column.
        aggregation (str): Aggregation name.

    Returns:
        Any: Global fallback value.
    """
    if aggregation == "count":
        return float(len(dataframe))

    if source_column is None:
        raise ValueError("source_column is required for the requested aggregation.")

    series = dataframe[source_column]
    if aggregation == "nunique":
        return float(series.nunique(dropna=True))
    grouped = getattr(series, aggregation)
    return safe_json_value(grouped())


def _resolve_feature_definition(
    dataframe: pd.DataFrame,
    feature_spec: Mapping[str, Any],
) -> ResolvedFeatureDefinition:
    """Fit any required lookup metadata for one feature definition.

    Args:
        dataframe (pd.DataFrame): Current fit dataframe.
        feature_spec (Mapping[str, Any]): Normalized feature configuration.

    Returns:
        ResolvedFeatureDefinition: Resolved feature metadata.
    """
    name = str(feature_spec["name"])
    kind = str(feature_spec["kind"])

    if kind in {"ratio", "difference", "sum", "product"}:
        columns = [str(column) for column in feature_spec["columns"]]
        _require_columns(dataframe, columns)
        return ResolvedFeatureDefinition(
            name=name,
            kind=kind,
            source_columns=columns,
            output_columns=[name],
        )

    if kind in {"absolute", "clip", "log1p", "square", "sqrt", "string_length"}:
        column = str(feature_spec["column"])
        _require_columns(dataframe, [column])
        params = {
            key: value
            for key, value in feature_spec.items()
            if key not in {"name", "kind", "column"}
        }
        return ResolvedFeatureDefinition(
            name=name,
            kind=kind,
            source_columns=[column],
            output_columns=[name],
            params=params,
        )

    if kind == "datetime_part":
        column = str(feature_spec["column"])
        _require_columns(dataframe, [column])
        parts = [str(part) for part in feature_spec["parts"]]
        output_columns = _build_output_columns(name, count=len(parts), suffixes=parts)
        return ResolvedFeatureDefinition(
            name=name,
            kind=kind,
            source_columns=[column],
            output_columns=output_columns,
            params={"parts": parts},
        )

    if kind == "category_frequency":
        column = str(feature_spec["column"])
        _require_columns(dataframe, [column])
        series = dataframe[column].copy()
        if series.dtype.kind in {"O", "U", "S"}:
            series = series.astype("string")
        normalize = bool(feature_spec["normalize"])
        dropna = bool(feature_spec["dropna"])
        frequency = (
            series.value_counts(normalize=normalize, dropna=dropna)
            .rename_axis(column)
            .reset_index(name=name)
        )
        return ResolvedFeatureDefinition(
            name=name,
            kind=kind,
            source_columns=[column],
            output_columns=[name],
            params={
                "normalize": normalize,
                "dropna": dropna,
            },
            lookup_frame=frequency,
            fallback_values={name: feature_spec["fill_value"]},
        )

    if kind == "groupby_aggregate":
        keys = [str(key) for key in feature_spec["keys"]]
        _require_columns(dataframe, keys)
        aggregations = [
            str(aggregation) for aggregation in feature_spec["aggregations"]
        ]
        source_column = feature_spec.get("source_column")
        if source_column is not None:
            _require_columns(dataframe, [str(source_column)])
        output_columns = _build_output_columns(
            name,
            count=len(aggregations),
            suffixes=None if len(aggregations) == 1 else aggregations,
        )

        if source_column is None:
            lookup_frame = (
                dataframe.groupby(keys, dropna=False)
                .size()
                .reset_index(name=output_columns[0])
            )
        else:
            grouped = (
                dataframe.groupby(keys, dropna=False)[str(source_column)]
                .agg(aggregations)
                .reset_index()
            )
            if len(aggregations) == 1:
                lookup_frame = grouped.rename(
                    columns={aggregations[0]: output_columns[0]}
                )
            else:
                rename_map = {
                    aggregation: output_column
                    for aggregation, output_column in zip(aggregations, output_columns)
                }
                lookup_frame = grouped.rename(columns=rename_map)

        fallback_values: dict[str, Any] = {}
        unknown_group_strategy = str(feature_spec["unknown_group_strategy"])
        if unknown_group_strategy == "constant":
            fill_value = feature_spec.get("fill_value")
            fallback_values = {
                output_column: fill_value for output_column in output_columns
            }
        elif unknown_group_strategy == "global":
            fallback_values = {
                output_column: _global_aggregate_value(
                    dataframe,
                    source_column=str(source_column)
                    if source_column is not None
                    else None,
                    aggregation=aggregation,
                )
                for aggregation, output_column in zip(aggregations, output_columns)
            }

        return ResolvedFeatureDefinition(
            name=name,
            kind=kind,
            source_columns=keys
            + ([str(source_column)] if source_column is not None else []),
            output_columns=output_columns,
            params={
                "keys": keys,
                "aggregations": aggregations,
                "source_column": str(source_column)
                if source_column is not None
                else None,
                "unknown_group_strategy": unknown_group_strategy,
            },
            lookup_frame=lookup_frame,
            fallback_values=fallback_values,
        )

    raise ValueError(f"Unsupported engineered feature kind {kind!r}.")


def _assign_engineered_columns(
    dataframe: pd.DataFrame,
    engineered: pd.DataFrame,
    *,
    conflict_policy: str,
) -> pd.DataFrame:
    """Assign engineered columns onto a working dataframe copy.

    Args:
        dataframe (pd.DataFrame): Existing dataframe state.
        engineered (pd.DataFrame): New engineered columns.
        conflict_policy (str): Collision policy for existing column names.

    Returns:
        pd.DataFrame: Updated dataframe with engineered columns applied.
    """
    updated = dataframe.copy()
    collisions = [column for column in engineered.columns if column in updated.columns]
    if collisions and conflict_policy == "error":
        collision_text = ", ".join(collisions)
        raise ValueError(
            f"Engineered columns would overwrite existing columns: {collision_text}."
        )

    for column in engineered.columns:
        updated[column] = engineered[column]
    return updated


def _apply_feature_definition(
    dataframe: pd.DataFrame,
    definition: ResolvedFeatureDefinition,
) -> pd.DataFrame:
    """Apply one fitted feature definition to a dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe for the current transform step.
        definition (ResolvedFeatureDefinition): Fitted feature definition.

    Returns:
        pd.DataFrame: Dataframe containing only the engineered output columns.
    """
    _require_columns(dataframe, definition.source_columns)

    if definition.kind == "ratio":
        numerator = _numeric_series(dataframe, definition.source_columns[0])
        denominator = _numeric_series(dataframe, definition.source_columns[1]).replace(
            0, np.nan
        )
        return pd.DataFrame(
            {definition.output_columns[0]: numerator / denominator},
            index=dataframe.index,
        )

    if definition.kind == "difference":
        left = _numeric_series(dataframe, definition.source_columns[0])
        right = _numeric_series(dataframe, definition.source_columns[1])
        return pd.DataFrame(
            {definition.output_columns[0]: left - right}, index=dataframe.index
        )

    if definition.kind == "sum":
        left = _numeric_series(dataframe, definition.source_columns[0])
        right = _numeric_series(dataframe, definition.source_columns[1])
        return pd.DataFrame(
            {definition.output_columns[0]: left + right}, index=dataframe.index
        )

    if definition.kind == "product":
        left = _numeric_series(dataframe, definition.source_columns[0])
        right = _numeric_series(dataframe, definition.source_columns[1])
        return pd.DataFrame(
            {definition.output_columns[0]: left * right}, index=dataframe.index
        )

    if definition.kind == "absolute":
        source = _numeric_series(dataframe, definition.source_columns[0])
        return pd.DataFrame(
            {definition.output_columns[0]: source.abs()}, index=dataframe.index
        )

    if definition.kind == "clip":
        source = _numeric_series(dataframe, definition.source_columns[0])
        return pd.DataFrame(
            {
                definition.output_columns[0]: source.clip(
                    lower=definition.params.get("lower"),
                    upper=definition.params.get("upper"),
                )
            },
            index=dataframe.index,
        )

    if definition.kind == "log1p":
        source = _numeric_series(dataframe, definition.source_columns[0])
        source = source.where(source > -1)
        return pd.DataFrame(
            {definition.output_columns[0]: np.log1p(source)}, index=dataframe.index
        )

    if definition.kind == "square":
        source = _numeric_series(dataframe, definition.source_columns[0])
        return pd.DataFrame(
            {definition.output_columns[0]: source.pow(2)}, index=dataframe.index
        )

    if definition.kind == "sqrt":
        source = _numeric_series(dataframe, definition.source_columns[0]).where(
            _numeric_series(dataframe, definition.source_columns[0]) >= 0
        )
        return pd.DataFrame(
            {definition.output_columns[0]: np.sqrt(source)}, index=dataframe.index
        )

    if definition.kind == "datetime_part":
        series = pd.to_datetime(
            dataframe[definition.source_columns[0]], errors="coerce"
        )
        feature_frame = pd.DataFrame(index=dataframe.index)
        for output_column, part in zip(
            definition.output_columns,
            definition.params["parts"],
        ):
            if part == "year":
                feature_frame[output_column] = series.dt.year
            elif part == "month":
                feature_frame[output_column] = series.dt.month
            elif part == "day":
                feature_frame[output_column] = series.dt.day
            elif part == "dayofweek":
                feature_frame[output_column] = series.dt.dayofweek
            elif part == "dayofyear":
                feature_frame[output_column] = series.dt.dayofyear
            elif part == "week":
                feature_frame[output_column] = series.dt.isocalendar().week.astype(
                    "float64"
                )
            elif part == "quarter":
                feature_frame[output_column] = series.dt.quarter
            elif part == "is_month_start":
                feature_frame[output_column] = series.dt.is_month_start.astype(
                    "float64"
                )
            elif part == "is_month_end":
                feature_frame[output_column] = series.dt.is_month_end.astype("float64")
        return feature_frame

    if definition.kind == "string_length":
        series = dataframe[definition.source_columns[0]]
        values = series.astype("string").str.len()
        values = values.where(series.notna(), np.nan)
        return pd.DataFrame(
            {definition.output_columns[0]: values}, index=dataframe.index
        )

    if definition.kind == "category_frequency":
        if definition.lookup_frame is None:
            raise ValueError("Frequency feature is missing fitted lookup data.")
        source_column = definition.source_columns[0]
        lookup_column = definition.output_columns[0]
        merged = dataframe[[source_column]].merge(
            definition.lookup_frame,
            how="left",
            on=source_column,
            sort=False,
        )
        values = merged[lookup_column]
        if definition.fallback_values:
            values = values.fillna(definition.fallback_values[lookup_column])
        return pd.DataFrame({lookup_column: values.to_numpy()}, index=dataframe.index)

    if definition.kind == "groupby_aggregate":
        if definition.lookup_frame is None:
            raise ValueError("Groupby aggregate feature is missing fitted lookup data.")
        keys = [str(key) for key in definition.params["keys"]]
        merged = dataframe[keys].merge(
            definition.lookup_frame,
            how="left",
            on=keys,
            sort=False,
        )
        output = merged[definition.output_columns].copy()
        for output_column, fallback in definition.fallback_values.items():
            output[output_column] = output[output_column].fillna(fallback)
        output.index = dataframe.index
        return output

    raise ValueError(f"Unsupported engineered feature kind {definition.kind!r}.")


def _flatten_nested_params(value: Any, *, prefix: str) -> dict[str, Any]:
    """Flatten nested lists and mappings into sklearn-style param names.

    Args:
        value (Any): Nested value to flatten.
        prefix (str): Prefix for generated param names.

    Returns:
        dict[str, Any]: Flattened parameter mapping.
    """
    rows: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            child_prefix = f"{prefix}__{key}"
            rows[child_prefix] = nested_value
            rows.update(_flatten_nested_params(nested_value, prefix=child_prefix))
        return rows
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            child_prefix = f"{prefix}__{index}"
            rows[child_prefix] = nested_value
            rows.update(_flatten_nested_params(nested_value, prefix=child_prefix))
    return rows


def _set_nested_param_value(container: Any, path_parts: list[str], value: Any) -> None:
    """Write a nested parameter value into a copied feature spec structure.

    Args:
        container (Any): Mutable mapping or list to update.
        path_parts (list[str]): Remaining nested param path parts.
        value (Any): Value to store.
    """
    current_part = path_parts[0]
    is_last = len(path_parts) == 1
    if isinstance(container, list):
        index = int(current_part)
        if index >= len(container):
            raise ValueError(f"Feature parameter index {index} is out of range.")
        if is_last:
            container[index] = value
            return
        _set_nested_param_value(container[index], path_parts[1:], value)
        return

    if is_last:
        container[current_part] = value
        return
    if current_part not in container:
        raise ValueError(
            f"Unknown feature-engineering param path segment {current_part!r}."
        )
    _set_nested_param_value(container[current_part], path_parts[1:], value)


class DeterministicFeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible dataframe transformer for deterministic FE specs.

    Args:
        features (list[dict[str, Any]]): Ordered engineered feature definitions.
        conflict_policy (str): Name-collision behavior for engineered columns.
        drop_source_columns (list[str] | None): Optional columns to drop after FE.
        preserve_index (bool): Whether to preserve the dataframe index on output.
    """

    def __init__(
        self,
        *,
        features: list[dict[str, Any]],
        conflict_policy: str = "error",
        drop_source_columns: list[str] | None = None,
        preserve_index: bool = True,
    ) -> None:
        self.features = features
        self.conflict_policy = conflict_policy
        self.drop_source_columns = (
            [] if drop_source_columns is None else drop_source_columns
        )
        self.preserve_index = preserve_index

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return sklearn-style parameters with flattened feature aliases.

        Args:
            deep (bool): Whether to include nested flattened feature entries.

        Returns:
            dict[str, Any]: Parameter mapping.
        """
        params = super().get_params(deep=False)
        if not deep:
            return params
        params.update(_flatten_nested_params(self.features, prefix="features"))
        params["output__preserve_index"] = self.preserve_index
        return params

    def set_params(self, **params: Any) -> DeterministicFeatureEngineeringTransformer:
        """Update top-level or flattened feature parameters.

        Args:
            **params: Parameter updates using sklearn-style names.

        Returns:
            DeterministicFeatureEngineeringTransformer: Updated transformer.
        """
        nested_updates = {key: value for key, value in params.items() if "__" in key}
        direct_updates = {
            key: value for key, value in params.items() if "__" not in key
        }
        if direct_updates:
            super().set_params(**direct_updates)
        if not nested_updates:
            return self

        features_copy = deepcopy(self.features)
        for key, value in nested_updates.items():
            if key == "output__preserve_index":
                self.preserve_index = bool(value)
                continue
            path_parts = key.split("__")
            if path_parts[0] != "features":
                raise ValueError(f"Unsupported feature-engineering param {key!r}.")
            _set_nested_param_value(features_copy, path_parts[1:], value)
        self.features = features_copy
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> DeterministicFeatureEngineeringTransformer:
        """Fit the deterministic feature definitions against a dataframe.

        Args:
            X (pd.DataFrame): Input feature dataframe.
            y (pd.Series | np.ndarray | None): Ignored sklearn compatibility hook.

        Returns:
            DeterministicFeatureEngineeringTransformer: Fitted transformer.
        """
        del y
        frame = ensure_dataframe(X, field_name="X")
        self.spec_ = normalize_feature_engineering_spec(
            {
                "features": deepcopy(self.features),
                "conflict_policy": self.conflict_policy,
                "drop_source_columns": deepcopy(self.drop_source_columns),
                "output": {"preserve_index": self.preserve_index},
            }
        )
        working_frame = frame.copy()
        definitions: list[ResolvedFeatureDefinition] = []
        engineered_columns: list[str] = []

        # Fit sequentially so later features may depend on earlier engineered columns.
        for feature_spec in self.spec_["features"]:
            definition = _resolve_feature_definition(working_frame, feature_spec)
            engineered = _apply_feature_definition(working_frame, definition)
            working_frame = _assign_engineered_columns(
                working_frame,
                engineered,
                conflict_policy=self.spec_["conflict_policy"],
            )
            definitions.append(definition)
            for column in definition.output_columns:
                if column not in engineered_columns:
                    engineered_columns.append(column)

        if self.spec_["drop_source_columns"]:
            _require_columns(working_frame, self.spec_["drop_source_columns"])
            working_frame = working_frame.drop(
                columns=self.spec_["drop_source_columns"]
            )

        self.definitions_ = definitions
        self.input_columns_ = [str(column) for column in frame.columns]
        self.engineered_columns_ = engineered_columns
        self.output_columns_ = [str(column) for column in working_frame.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted feature definitions to a dataframe.

        Args:
            X (pd.DataFrame): Input feature dataframe.

        Returns:
            pd.DataFrame: Engineered dataframe.
        """
        check_is_fitted(
            self,
            attributes=(
                "definitions_",
                "input_columns_",
                "engineered_columns_",
                "output_columns_",
            ),
        )
        frame = ensure_dataframe(X, field_name="X")
        _require_columns(frame, self.input_columns_)
        working_frame = frame.copy()
        for definition in self.definitions_:
            engineered = _apply_feature_definition(working_frame, definition)
            working_frame = _assign_engineered_columns(
                working_frame,
                engineered,
                conflict_policy=str(self.spec_["conflict_policy"]),
            )

        if self.spec_["drop_source_columns"]:
            _require_columns(working_frame, self.spec_["drop_source_columns"])
            working_frame = working_frame.drop(
                columns=self.spec_["drop_source_columns"]
            )

        transformed_frame = working_frame.loc[:, self.output_columns_].copy()
        if bool(self.spec_["output"]["preserve_index"]):
            return transformed_frame
        return transformed_frame.reset_index(drop=True)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return stable output feature names.

        Args:
            input_features (Sequence[str] | None): Ignored sklearn compatibility hook.

        Returns:
            np.ndarray: Output feature names.
        """
        del input_features
        check_is_fitted(self, attributes=("output_columns_",))
        return np.asarray(self.output_columns_, dtype=object)


class FeatureEngineeringCollection(ToolCollection):
    """Deterministic, host-side engineered feature helpers for tabular data."""

    name = "feature_engineering"
    description = (
        "Build, fit, inspect, persist, and apply deterministic feature "
        "engineering artifacts to stored dataframe handles."
    )

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: ObjectStore,
    ) -> None:
        """Initialize feature engineering helpers.

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
            Path: Resolved host workspace path.
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

    def _get_feature_engineer(
        self, feature_engineer_handle: str
    ) -> StoredFeatureEngineer:
        """Fetch a stored feature-engineering artifact.

        Args:
            feature_engineer_handle (str): FE handle.

        Returns:
            StoredFeatureEngineer: Stored FE artifact.
        """
        return self._object_store.get(
            feature_engineer_handle,
            expected_type=StoredFeatureEngineer,
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

    def _apply_artifact(
        self,
        dataframe_handle: str,
        artifact: StoredFeatureEngineer,
        *,
        include_target: bool = False,
    ) -> pd.DataFrame:
        """Apply a fitted feature engineer to a stored dataframe.

        Args:
            dataframe_handle (str): Input dataframe handle.
            artifact (StoredFeatureEngineer): Fitted FE artifact.
            include_target (bool): Whether to append the target column back.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        feature_frame, target_series = self._feature_frame(
            dataframe_handle,
            target_column=artifact.target_column,
            require_target_column=include_target,
        )
        transformed_frame = artifact.estimator.transform(
            feature_frame[artifact.input_columns]
        )
        if include_target:
            return append_target_column(
                transformed_frame,
                target_column=artifact.target_column,
                target_series=target_series,
                preserve_index=artifact.preserve_index,
            )
        return transformed_frame

    @tool
    def build_feature_engineering_spec(
        self,
        *,
        features: list[dict[str, Any]],
        conflict_policy: str = "error",
        drop_source_columns: list[str] | None = None,
        preserve_index: bool = True,
    ) -> dict[str, Any]:
        """Build and validate a normalized feature-engineering spec.

        Args:
            features (list[dict[str, Any]]): Ordered engineered feature specs.
            conflict_policy (str): Name-collision behavior for output columns.
            drop_source_columns (list[str] | None): Optional columns to remove after
                engineering is complete.
            preserve_index (bool): Whether transformed dataframes keep their index.

        Returns:
            dict[str, Any]: Normalized FE spec ready for fitting.

        Examples:
            spec = build_feature_engineering_spec(
                features=[
                    {
                        "name": "premium_ratio",
                        "kind": "ratio",
                        "columns": ["premium", "income"],
                    },
                    {
                        "name": "segment_mean",
                        "kind": "groupby_aggregate",
                        "keys": ["segment"],
                        "source_column": "premium",
                        "aggregation": "mean",
                        "unknown_group_strategy": "global",
                    },
                ]
            )
            # Returns:
            # {
            #     "features": [
            #         {"name": "premium_ratio", "kind": "ratio", "columns": ["premium", "income"]},
            #         {"name": "segment_mean", "kind": "groupby_aggregate", "keys": ["segment"]}
            #     ],
            #     "conflict_policy": "error",
            #     "drop_source_columns": [],
            #     "output": {"preserve_index": True}
            # }
        """
        return normalize_feature_engineering_spec(
            {
                "features": features,
                "conflict_policy": conflict_policy,
                "drop_source_columns": drop_source_columns or [],
                "output": {"preserve_index": preserve_index},
            }
        )

    @tool
    def fit_feature_engineer(
        self,
        dataframe_handle: str,
        spec: dict[str, Any],
        *,
        target_column: str | None = None,
    ) -> str:
        """Fit a deterministic feature engineer from a declarative spec.

        Args:
            dataframe_handle (str): Handle pointing to the fit dataframe.
            spec (dict[str, Any]): Declarative FE spec.
            target_column (str | None): Optional target column to exclude from fit.

        Returns:
            str: Handle for the fitted FE artifact.

        Examples:
            fe_handle = fit_feature_engineer(df_handle, spec, target_column="target")
        """
        normalized_spec = normalize_feature_engineering_spec(spec)
        feature_frame, _ = self._feature_frame(
            dataframe_handle,
            target_column=target_column,
        )
        estimator = DeterministicFeatureEngineeringTransformer(
            features=deepcopy(normalized_spec["features"]),
            conflict_policy=str(normalized_spec["conflict_policy"]),
            drop_source_columns=list(normalized_spec["drop_source_columns"]),
            preserve_index=bool(normalized_spec["output"]["preserve_index"]),
        )
        estimator.fit(feature_frame)

        artifact = StoredFeatureEngineer(
            estimator=estimator,
            spec=normalized_spec,
            definitions=list(estimator.definitions_),
            input_columns=list(estimator.input_columns_),
            engineered_columns=list(estimator.engineered_columns_),
            output_columns=list(estimator.output_columns_),
            target_column=target_column,
            preserve_index=bool(normalized_spec["output"]["preserve_index"]),
            conflict_policy=str(normalized_spec["conflict_policy"]),
            drop_source_columns=list(normalized_spec["drop_source_columns"]),
        )
        return self._object_store.put(artifact, prefix="fe")

    @tool
    def transform_with_feature_engineer(
        self,
        dataframe_handle: str,
        feature_engineer_handle: str,
        *,
        include_target: bool = False,
    ) -> str:
        """Transform a dataframe with a fitted feature engineer.

        Args:
            dataframe_handle (str): Input dataframe handle.
            feature_engineer_handle (str): Handle pointing to a fitted FE artifact.
            include_target (bool): Whether to append the stored target column back.

        Returns:
            str: Handle for the transformed dataframe.

        Examples:
            engineered_handle = transform_with_feature_engineer(
                df_handle,
                fe_handle,
                include_target=True,
            )
        """
        artifact = self._get_feature_engineer(feature_engineer_handle)
        transformed_frame = self._apply_artifact(
            dataframe_handle,
            artifact,
            include_target=include_target,
        )
        return self._object_store.put(transformed_frame, prefix="df")

    @tool
    def fit_transform_with_feature_engineer(
        self,
        dataframe_handle: str,
        spec: dict[str, Any],
        *,
        target_column: str | None = None,
        include_target: bool = False,
    ) -> dict[str, str]:
        """Fit a feature engineer and immediately transform the same dataframe.

        Args:
            dataframe_handle (str): Input dataframe handle.
            spec (dict[str, Any]): Declarative FE spec.
            target_column (str | None): Optional target column to exclude from fit.
            include_target (bool): Whether to append the target column back.

        Returns:
            dict[str, str]: Handles for the fitted artifact and transformed frame.

        Examples:
            result = fit_transform_with_feature_engineer(
                df_handle,
                spec,
                target_column="target",
                include_target=True,
            )
            # Returns:
            # {
            #     "feature_engineer_handle": "fe_1",
            #     "dataframe_handle": "df_2"
            # }
        """
        feature_engineer_handle = self.fit_feature_engineer(
            dataframe_handle,
            spec,
            target_column=target_column,
        )
        dataframe_result_handle = self.transform_with_feature_engineer(
            dataframe_handle,
            feature_engineer_handle,
            include_target=include_target,
        )
        return {
            "feature_engineer_handle": feature_engineer_handle,
            "dataframe_handle": dataframe_result_handle,
        }

    @tool
    def inspect_feature_engineer(self, feature_engineer_handle: str) -> dict[str, Any]:
        """Return a JSON-friendly summary of a fitted FE artifact.

        Args:
            feature_engineer_handle (str): FE handle.

        Returns:
            dict[str, Any]: Summary of the stored FE artifact.

        Examples:
            print(inspect_feature_engineer(fe_handle))
            # Returns:
            # {
            #     "type": "StoredFeatureEngineer",
            #     "engineered_columns": ["premium_ratio", "segment_mean"],
            #     "target_column": "target"
            # }
        """
        return self._get_feature_engineer(feature_engineer_handle).to_json_summary()

    @tool
    def list_engineered_features(self, feature_engineer_handle: str) -> list[str]:
        """List the engineered columns produced by a fitted FE artifact.

        Args:
            feature_engineer_handle (str): FE handle.

        Returns:
            list[str]: Engineered output column names.

        Examples:
            print(list_engineered_features(fe_handle))
        """
        return list(
            self._get_feature_engineer(feature_engineer_handle).engineered_columns
        )

    @tool
    def save_feature_engineer(self, feature_engineer_handle: str, path: str) -> str:
        """Persist a fitted FE artifact to the workspace with joblib.

        Args:
            feature_engineer_handle (str): FE handle.
            path (str): Relative or `/workspace` destination path.

        Returns:
            str: Virtual path to the saved artifact.

        Examples:
            save_feature_engineer(fe_handle, "/workspace/output/feature_engineer.joblib")
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = self._get_feature_engineer(feature_engineer_handle)
        joblib.dump(artifact, host_path)
        self._os_access.record_host_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool
    def load_feature_engineer(self, path: str) -> str:
        """Load a previously saved FE artifact from the workspace.

        Args:
            path (str): Relative or `/workspace` path to a saved artifact.

        Returns:
            str: Handle for the loaded FE artifact.

        Examples:
            fe_handle = load_feature_engineer("/workspace/output/feature_engineer.joblib")
        """
        artifact = joblib.load(self._resolve_host_path(path))
        if not isinstance(artifact, StoredFeatureEngineer):
            raise TypeError("Loaded artifact is not a StoredFeatureEngineer.")
        return self._object_store.put(artifact, prefix="fe")


__all__ = [
    "DeterministicFeatureEngineeringTransformer",
    "FeatureEngineeringCollection",
    "ResolvedFeatureDefinition",
    "StoredFeatureEngineer",
    "normalize_feature_engineering_spec",
]
