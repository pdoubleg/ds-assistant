"""Helper utilities for Monty pipeline HPO."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ..registry.feature_engineering import (
    DeterministicFeatureEngineeringTransformer,
    FeatureEngineeringCollection,
    StoredFeatureEngineer,
    normalize_feature_engineering_spec,
)
from ..registry.freeform import FreeformDataframeTransformer, StoredFreeformTransformer
from ..registry.feature_selection import StoredFeatureSelectionReport
from .metrics import (
    build_lightgbm_estimator,
    evaluate_feature_subset as evaluate_subset_metrics,
    infer_task_type,
    prepare_model_frames,
    prepare_targets,
)
from ..registry.preprocessing import (
    PreprocessingCollection,
    StoredPreprocessor,
    normalize_preprocessing_spec,
)
from ..core.registry import safe_json_value
from .sklearn_pipeline import DataFrameOutputColumnTransformer

_SUPPORTED_SUGGESTION_KINDS = {"categorical", "int", "float", "loguniform"}
_FIXED_PIPELINE_ORDER = ("freeform", "feature_engineering", "preprocessing")
_EXAMPLE_PIPELINE_CONFIG = {
    "data": {
        "train_handle": "df_train",
        "validation_handle": "df_valid",
        "target_column": "target",
    },
    "feature_engineering": {
        "spec": {
            "features": [
                {
                    "name": "loss_ratio",
                    "kind": "ratio",
                    "columns": ["premium", "income"],
                }
            ],
            "conflict_policy": "error",
            "drop_source_columns": [],
            "output": {"preserve_index": True},
        }
    },
    "freeform": {
        "code": "df['log_income'] = np.log1p(df['income'])",
        "intent": "feature_engineering",
        "args": {"income_shift": 0.0},
        "strict_schema": True,
        "output": {"preserve_index": True},
    },
    "preprocessing": {
        "spec": {
            "groups": [
                {
                    "name": "numeric",
                    "columns": {"selector": "numeric"},
                    "steps": [{"kind": "simple_imputer", "strategy": "median"}],
                },
                {
                    "name": "categorical",
                    "columns": {"selector": "categorical"},
                    "steps": [
                        {"kind": "simple_imputer", "strategy": "most_frequent"},
                        {
                            "kind": "one_hot_encoder",
                            "handle_unknown": "ignore",
                            "sparse_output": False,
                        },
                    ],
                },
            ],
            "remainder": "drop",
            "output": {"dense": True, "preserve_index": True},
        }
    },
    "model": {
        "base_params": {
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
        }
    },
    "evaluation": {
        "mode": "validation",
        "metric": "f1",
        "cv_folds": 5,
        "random_state": 0,
    },
}
_EXAMPLE_SEARCH_SPACE = [
    {
        "path": "preprocessing.spec.groups.0.steps.0.strategy",
        "kind": "categorical",
        "choices": ["mean", "median"],
    },
    {
        "path": "model.base_params.num_leaves",
        "kind": "int",
        "low": 16,
        "high": 64,
        "step": 8,
    },
    {
        "path": "model.base_params.learning_rate",
        "kind": "float",
        "low": 0.01,
        "high": 0.2,
        "log": True,
    },
]
_COMMON_LIGHTGBM_PARAMS: list[dict[str, Any]] = [
    {
        "path": "model.base_params.learning_rate",
        "description": "Shrinkage rate applied to each boosting step.",
        "suggestion_kind": "float",
    },
    {
        "path": "model.base_params.num_leaves",
        "description": "Maximum leaf count per tree.",
        "suggestion_kind": "int",
    },
    {
        "path": "model.base_params.min_child_samples",
        "description": "Minimum rows required in a child leaf.",
        "suggestion_kind": "int",
    },
    {
        "path": "model.base_params.feature_fraction",
        "description": "Fraction of features considered at each iteration.",
        "suggestion_kind": "float",
    },
    {
        "path": "model.base_params.bagging_fraction",
        "description": "Fraction of rows sampled for bagging.",
        "suggestion_kind": "float",
    },
    {
        "path": "model.base_params.lambda_l1",
        "description": "L1 regularization strength.",
        "suggestion_kind": "float",
    },
    {
        "path": "model.base_params.lambda_l2",
        "description": "L2 regularization strength.",
        "suggestion_kind": "float",
    },
]


@dataclass(slots=True)
class PipelineRunArtifacts:
    """Resolved pipeline artifacts and feature frames for one candidate config.

    Args:
        train_features (pd.DataFrame): Final training feature frame.
        train_target (pd.Series): Final training target series.
        validation_features (pd.DataFrame | None): Optional validation feature frame.
        validation_target (pd.Series | None): Optional validation target series.
        selected_features (list[str]): Final selected features passed to the model.
        preprocessor (StoredPreprocessor | None): Fitted preprocessing artifact, if any.
        feature_engineer (StoredFeatureEngineer | None): Fitted FE artifact, if any.
        freeform_transformer (StoredFreeformTransformer | None): Fitted reusable
            freeform transformer, if any.
        sklearn_pipeline (Pipeline | None): Fitted sklearn pipeline assembled from
            the candidate config for inspection or reuse.
        warnings (list[str]): Pipeline-stage warnings accumulated during resolution.
    """

    train_features: pd.DataFrame
    train_target: pd.Series
    validation_features: pd.DataFrame | None
    validation_target: pd.Series | None
    selected_features: list[str]
    preprocessor: StoredPreprocessor | None = None
    feature_engineer: StoredFeatureEngineer | None = None
    freeform_transformer: StoredFreeformTransformer | None = None
    sklearn_pipeline: Pipeline | None = None
    warnings: list[str] | None = None


def _string_list(values: list[Any], *, field_name: str) -> list[str]:
    """Coerce a list of values into strings.

    Args:
        values (list[Any]): Raw values to convert.
        field_name (str): Validation field name.

    Returns:
        list[str]: Coerced strings.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of strings, not a string.")
    return [str(value) for value in values]


def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Validate that a value is a mapping and coerce keys to strings.

    Args:
        value (Any): Value to validate.
        field_name (str): Field name for error messages.

    Returns:
        dict[str, Any]: Coerced mapping.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def normalize_pipeline_config(pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a pipeline HPO configuration.

    Args:
        pipeline_config (dict[str, Any]): Raw pipeline configuration.

    Returns:
        dict[str, Any]: Normalized pipeline configuration.
    """
    raw_config = _require_mapping(pipeline_config, field_name="pipeline_config")
    unknown_keys = set(raw_config) - {
        "data",
        "feature_engineering",
        "freeform",
        "preprocessing",
        "model",
        "evaluation",
    }
    if unknown_keys:
        unknown_text = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown pipeline config keys: {unknown_text}.")

    data = _require_mapping(raw_config.get("data"), field_name="data")
    train_handle = data.get("train_handle") or data.get("dataframe_handle")
    target_column = data.get("target_column")
    if not train_handle or not target_column:
        raise ValueError(
            "The pipeline config data section requires train_handle and target_column."
        )

    model = _require_mapping(
        raw_config.get("model", {"base_params": {}}), field_name="model"
    )
    base_params = model.get("base_params", {})
    if not isinstance(base_params, dict):
        raise ValueError("model.base_params must be a mapping.")

    evaluation = _require_mapping(
        raw_config.get("evaluation", {}), field_name="evaluation"
    )
    mode = str(
        evaluation.get(
            "mode",
            "validation" if data.get("validation_handle") else "cross_validation",
        )
    )
    if mode not in {"validation", "cross_validation"}:
        raise ValueError(
            "evaluation.mode must be either 'validation' or 'cross_validation'."
        )

    normalized = {
        "data": {
            "train_handle": str(train_handle),
            "validation_handle": (
                str(data["validation_handle"])
                if data.get("validation_handle") is not None
                else None
            ),
            "target_column": str(target_column),
        },
        "model": {
            "base_params": {str(key): value for key, value in base_params.items()},
        },
        "evaluation": {
            "mode": mode,
            "metric": str(evaluation.get("metric", "auto")),
            "cv_folds": int(evaluation.get("cv_folds", 5)),
            "random_state": int(evaluation.get("random_state", 0)),
            "scorer_handle": (
                str(evaluation["scorer_handle"])
                if evaluation.get("scorer_handle") is not None
                else None
            ),
            "splitter_handle": (
                str(evaluation["splitter_handle"])
                if evaluation.get("splitter_handle") is not None
                else None
            ),
            "group_column": (
                str(evaluation["group_column"])
                if evaluation.get("group_column") is not None
                else None
            ),
        },
    }

    for stage_name in _FIXED_PIPELINE_ORDER:
        section = raw_config.get(stage_name)
        if section is None:
            continue
        section_mapping = _require_mapping(section, field_name=stage_name)
        if stage_name == "freeform":
            if "handle" in section_mapping and "code" in section_mapping:
                raise ValueError(
                    "freeform must use either a handle or inline code, not both."
                )
            if "handle" not in section_mapping and "code" not in section_mapping:
                raise ValueError("freeform requires either a handle or code.")
            if "handle" in section_mapping:
                normalized[stage_name] = {"handle": str(section_mapping["handle"])}
            else:
                normalized[stage_name] = {
                    "code": str(section_mapping["code"]),
                    "intent": str(section_mapping.get("intent", "feature_engineering")),
                    "args": _require_mapping(
                        section_mapping.get("args", {}),
                        field_name="freeform.args",
                    ),
                    "strict_schema": bool(section_mapping.get("strict_schema", True)),
                    "output": {
                        "preserve_index": bool(
                            _require_mapping(
                                section_mapping.get("output", {"preserve_index": True}),
                                field_name="freeform.output",
                            ).get("preserve_index", True)
                        )
                    },
                }
            continue
        if "handle" in section_mapping and "spec" in section_mapping:
            raise ValueError(
                f"{stage_name} must use either a handle or a spec, not both."
            )
        if "handle" not in section_mapping and "spec" not in section_mapping:
            raise ValueError(f"{stage_name} requires either a handle or a spec.")
        normalized[stage_name] = {
            str(key): value for key, value in section_mapping.items()
        }

    return normalized


def normalize_search_space(
    search_space: list[dict[str, Any]],
    *,
    sklearn_param_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Validate and normalize a structured search-space definition.

    Args:
        search_space (list[dict[str, Any]]): Raw parameter specifications.
        sklearn_param_aliases (dict[str, str] | None): Optional mapping from
            sklearn pipeline param names to config paths.

    Returns:
        list[dict[str, Any]]: Normalized search-space entries.
    """
    if not isinstance(search_space, list):
        raise ValueError("search_space must be a list of parameter mappings.")

    normalized: list[dict[str, Any]] = []
    for entry in search_space:
        item = _require_mapping(entry, field_name="search_space entry")
        sklearn_param = str(item.get("sklearn_param", "")).strip()
        path = str(item.get("path", "")).strip()
        if not path and sklearn_param:
            if (
                sklearn_param_aliases is None
                or sklearn_param not in sklearn_param_aliases
            ):
                raise ValueError(
                    f"Search-space entry {sklearn_param!r} could not be resolved to a config path."
                )
            path = sklearn_param_aliases[sklearn_param]
        kind = str(item.get("kind", "")).strip()
        if not path or not kind:
            raise ValueError(
                "Each search-space entry requires non-empty path and kind values."
            )
        if kind not in _SUPPORTED_SUGGESTION_KINDS:
            supported = ", ".join(sorted(_SUPPORTED_SUGGESTION_KINDS))
            raise ValueError(
                f"Unsupported search-space kind {kind!r}. Supported values: {supported}."
            )

        normalized_item = {"path": path, "kind": kind}
        if sklearn_param:
            normalized_item["sklearn_param"] = sklearn_param
        if kind == "categorical":
            if "choices" not in item:
                raise ValueError(f"Search-space entry {path!r} requires 'choices'.")
            normalized_item["choices"] = list(item["choices"])
        else:
            if "low" not in item or "high" not in item:
                raise ValueError(
                    f"Search-space entry {path!r} requires low/high bounds."
                )
            normalized_item["low"] = item["low"]
            normalized_item["high"] = item["high"]
            if "step" in item:
                normalized_item["step"] = item["step"]
            if "log" in item:
                normalized_item["log"] = bool(item["log"])
            if kind == "loguniform":
                normalized_item["kind"] = "float"
                normalized_item["log"] = True

        if "condition" in item:
            condition = _require_mapping(item["condition"], field_name="condition")
            condition_path = str(condition.get("path", "")).strip()
            if not condition_path:
                raise ValueError(
                    f"Search-space condition for {path!r} requires a condition path."
                )
            normalized_item["condition"] = {
                "path": condition_path,
                "equals": condition.get("equals"),
                "in": list(condition["in"]) if "in" in condition else None,
            }

        normalized.append(normalized_item)
    return normalized


def build_hpo_config_bundle(
    pipeline_config: dict[str, Any],
    search_space: list[dict[str, Any]] | None = None,
    *,
    object_store: Any | None = None,
    os_access: Any | None = None,
) -> dict[str, Any]:
    """Build a normalized HPO config bundle.

    Args:
        pipeline_config (dict[str, Any]): Raw pipeline configuration.
        search_space (list[dict[str, Any]] | None): Optional search-space config.
        object_store (Any | None): Optional object store used for sklearn-param
            alias discovery.
        os_access (Any | None): Optional workspace adapter used for pipeline building.

    Returns:
        dict[str, Any]: Normalized bundle containing pipeline and search-space data.
    """
    normalized_pipeline = normalize_pipeline_config(pipeline_config)
    sklearn_aliases: dict[str, str] | None = None
    if object_store is not None and os_access is not None:
        try:
            sklearn_aliases = build_sklearn_pipeline(
                normalized_pipeline,
                object_store=object_store,
                os_access=os_access,
            ).param_aliases
        except Exception:
            sklearn_aliases = None
    return {
        "pipeline_config": normalized_pipeline,
        "search_space": normalize_search_space(
            search_space or [],
            sklearn_param_aliases=sklearn_aliases,
        ),
        "sklearn_param_aliases": sklearn_aliases or {},
    }


def build_hpo_schema_reference() -> dict[str, Any]:
    """Return a JSON-friendly reference for the public HPO schema.

    Returns:
        dict[str, Any]: Pipeline-config, stage, and search-space guidance.
    """
    return {
        "required_pipeline_sections": {
            "data": ["train_handle", "target_column"],
            "model": ["base_params"],
            "evaluation": [
                "mode",
                "metric",
                "cv_folds",
                "random_state",
                "scorer_handle",
                "splitter_handle",
                "group_column",
            ],
        },
        "optional_pipeline_sections": [
            "feature_engineering",
            "freeform",
            "preprocessing",
        ],
        "stage_shapes": {
            "feature_engineering": [
                {"shape": {"handle": "fe_1"}},
                {
                    "shape": {
                        "spec": _EXAMPLE_PIPELINE_CONFIG["feature_engineering"]["spec"]
                    }
                },
            ],
            "freeform": [
                {"shape": {"handle": "freeform_1"}},
                {"shape": _EXAMPLE_PIPELINE_CONFIG["freeform"]},
            ],
            "preprocessing": [
                {"shape": {"handle": "prep_1"}},
                {"shape": {"spec": _EXAMPLE_PIPELINE_CONFIG["preprocessing"]["spec"]}},
            ],
        },
        "search_space_schema": {
            "required_keys": ["kind"],
            "one_of": [
                {"shape": {"path": "model.base_params.num_leaves"}},
                {"shape": {"sklearn_param": "model__num_leaves"}},
            ],
            "supported_kinds": sorted(_SUPPORTED_SUGGESTION_KINDS),
            "categorical_shape": {
                "path": "model.base_params.num_leaves",
                "kind": "categorical",
                "choices": [16, 32, 64],
            },
            "numeric_shape": {
                "path": "model.base_params.learning_rate",
                "kind": "float",
                "low": 0.01,
                "high": 0.2,
                "log": True,
            },
            "conditional_shape": {
                "path": "model.base_params.bagging_fraction",
                "kind": "float",
                "low": 0.5,
                "high": 1.0,
                "condition": {
                    "path": "model.base_params.boosting_type",
                    "equals": "gbdt",
                },
            },
        },
        "path_guidance": [
            "Search-space paths must target the normalized pipeline config shape, not shorthand builder inputs.",
            "Always use inspect_pipeline_tunable_params(...) or build_hpo_config(...) output to discover the exact path names before authoring or refining a search space.",
            "Pipeline execution order is fixed: freeform -> feature_engineering -> preprocessing when those sections are present.",
            "inspect_pipeline_tunable_params(...) also surfaces sklearn pipeline params and config-path aliases when the live pipeline can be materialized.",
            "Search-space entries may use either `path` or `sklearn_param`. sklearn_param values are resolved back to config paths before sampling.",
            "Freeform transformer arguments should live under freeform.args so the agent can tune values without mutating the raw code string.",
            "For preprocessing specs built with build_preprocessing_spec(...), tunable paths usually live under preprocessing.spec.groups.<index>.steps.<index>.<field>.",
            "Optional evaluation.scorer_handle and evaluation.splitter_handle values should reference stored Monty metric/splitting handles.",
        ],
        "example_pipeline_config": safe_json_value(_EXAMPLE_PIPELINE_CONFIG),
        "example_search_space": safe_json_value(_EXAMPLE_SEARCH_SPACE),
    }


def build_hpo_inspection_return_schema() -> dict[str, Any]:
    """Return a JSON-friendly schema for HPO inspection payloads.

    Returns:
        dict[str, Any]: Top-level output fields, row-level schema, and usage notes.
    """
    return {
        "payload_type": "PipelineInspectionResult",
        "top_level_fields": {
            "pipeline_params": {
                "container": "list",
                "item_type": "PipelineParamRow",
                "description": (
                    "Ordered list of flattened pipeline parameter rows. Each row "
                    "describes one normalized dotted path and its current value."
                ),
            },
            "pipeline_params_by_path": {
                "container": "dict[str, PipelineParamRow]",
                "description": (
                    "Convenience wrapper keyed by normalized dotted path for "
                    "callers that need direct lookup instead of iterating the "
                    "`pipeline_params` list."
                ),
            },
            "sklearn_pipeline_params": {
                "container": "list",
                "item_type": "SklearnPipelineParamRow",
                "description": (
                    "Sklearn-native `Pipeline.get_params(deep=True)` rows with "
                    "resolved config-path aliases when available."
                ),
            },
            "sklearn_param_aliases": {
                "container": "dict[str, str]",
                "description": (
                    "Mapping from sklearn param names back to canonical config paths."
                ),
            },
            "search_space": {
                "container": "list",
                "item_type": "NormalizedSearchSpaceEntry",
                "description": "Normalized search-space entries after validation.",
            },
            "recommended_model_params": {
                "container": "list",
                "item_type": "dict[str, Any]",
                "description": (
                    "Common LightGBM parameter suggestions that are often useful "
                    "to tune."
                ),
            },
            "schema_reference": {
                "container": "dict[str, Any]",
                "description": (
                    "Input schema guidance for `pipeline_config` and "
                    "`search_space` authoring."
                ),
            },
        },
        "pipeline_param_row": {
            "required_fields": {
                "path": {
                    "type": "str",
                    "description": (
                        "Normalized dotted path inside the materialized pipeline "
                        "config."
                    ),
                },
                "stage": {
                    "type": "str",
                    "description": "Top-level pipeline stage for the path.",
                },
                "current_value": {
                    "type": "Any",
                    "description": "Current normalized value at the path.",
                },
                "sklearn_param": {
                    "type": "str | None",
                    "description": "Primary sklearn pipeline alias when available.",
                },
                "is_tunable": {
                    "type": "bool",
                    "description": (
                        "Whether the path is targeted by a normalized "
                        "search-space entry."
                    ),
                },
            },
            "optional_fields": {
                "suggestion_kind": {
                    "type": "str",
                    "description": (
                        "Present when the path is tunable. Mirrors the Optuna "
                        "suggestion kind for the matching search-space entry."
                    ),
                },
                "choices": {
                    "type": "list[Any]",
                    "description": (
                        "Categorical choices for a categorical search-space entry."
                    ),
                },
                "low": {
                    "type": "int | float",
                    "description": "Lower numeric bound for int/float suggestions.",
                },
                "high": {
                    "type": "int | float",
                    "description": "Upper numeric bound for int/float suggestions.",
                },
                "log": {
                    "type": "bool",
                    "description": (
                        "Whether the numeric parameter should be sampled on a "
                        "log scale."
                    ),
                },
                "sklearn_params": {
                    "type": "list[str]",
                    "description": "All sklearn pipeline aliases for this config path.",
                },
            },
        },
        "notes": [
            "`pipeline_params` is intentionally a list of row objects, not a dict.",
            "Use `pipeline_params_by_path` when the caller needs direct keyed lookup.",
            "Use `sklearn_pipeline_params` when you want sklearn-native param names.",
            (
                "Treat the normalized dotted `path` field as the canonical key "
                "when authoring or refining `search_space`."
            ),
        ],
    }


def _get_nested_value(config: Any, path: str) -> Any:
    """Resolve a dotted path against nested dictionaries and lists.

    Args:
        config (Any): Nested configuration object.
        path (str): Dotted path expression.

    Returns:
        Any: Resolved value.
    """
    current = config
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(f"Cannot descend into path {path!r} at {part!r}.")
    return current


def _set_nested_value(config: Any, path: str, value: Any) -> None:
    """Set a dotted-path value inside nested dictionaries and lists.

    Args:
        config (Any): Nested configuration object.
        path (str): Dotted path expression.
        value (Any): Value to write.
    """
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:  # pragma: no cover - defensive branch
            raise KeyError(f"Cannot descend into path {path!r} at {part!r}.")

    leaf = parts[-1]
    if isinstance(current, dict):
        current[leaf] = value
    elif isinstance(current, list):
        current[int(leaf)] = value
    else:  # pragma: no cover - defensive branch
        raise KeyError(f"Cannot set path {path!r}.")


def _flatten_config(
    value: Any,
    *,
    prefix: str = "",
    stage: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten nested pipeline config values into path rows.

    Args:
        value (Any): Value to flatten.
        prefix (str): Current dotted path prefix.
        stage (str | None): Current pipeline stage name.

    Returns:
        list[dict[str, Any]]: Flattened parameter rows.
    """
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            child_stage = stage or child_prefix.split(".")[0]
            rows.extend(_flatten_config(item, prefix=child_prefix, stage=child_stage))
        return rows
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            rows.extend(_flatten_config(item, prefix=child_prefix, stage=stage))
        return rows

    rows.append(
        {
            "path": prefix,
            "stage": stage or prefix.split(".")[0],
            "current_value": safe_json_value(value),
        }
    )
    return rows


def inspect_pipeline_params(
    pipeline_config: dict[str, Any],
    search_space: list[dict[str, Any]] | None = None,
    *,
    object_store: Any | None = None,
    os_access: Any | None = None,
) -> dict[str, Any]:
    """Render flattened pipeline params and tunable search-space metadata.

    Args:
        pipeline_config (dict[str, Any]): Raw or normalized pipeline config.
        search_space (list[dict[str, Any]] | None): Optional search-space config.
        object_store (Any | None): Optional object store used to build a live
            sklearn pipeline view.
        os_access (Any | None): Optional workspace adapter used to build a live
            sklearn pipeline view.

    Returns:
        dict[str, Any]: Inspection payload with:
            - `pipeline_params`: ordered `list[dict[str, Any]]` rows, where each
              row describes one normalized dotted path.
            - `pipeline_params_by_path`: `dict[str, dict[str, Any]]` keyed by the
              normalized dotted path for direct lookup.
            - `return_schema`: explicit output-schema guidance for callers that
              want to inspect the payload shape before consuming it.
            - `search_space`, `recommended_model_params`, and `schema_reference`
              for normalized HPO guidance.
    """
    normalized_bundle = build_hpo_config_bundle(
        pipeline_config,
        search_space,
        object_store=object_store,
        os_access=os_access,
    )
    normalized_pipeline = normalized_bundle["pipeline_config"]
    normalized_search_space = normalized_bundle["search_space"]
    search_space_by_path = {item["path"]: item for item in normalized_search_space}
    search_space_by_sklearn_param = {
        item["sklearn_param"]: item
        for item in normalized_search_space
        if item.get("sklearn_param") is not None
    }
    flattened = _flatten_config(normalized_pipeline)
    alias_rows_by_path: dict[str, list[str]] = {}
    sklearn_pipeline_params: list[dict[str, Any]] = []

    if object_store is not None and os_access is not None:
        try:
            built_pipeline = build_sklearn_pipeline(
                normalized_pipeline,
                object_store=object_store,
                os_access=os_access,
            )
            for sklearn_param, value in built_pipeline.pipeline.get_params(
                deep=True
            ).items():
                config_path = built_pipeline.param_aliases.get(sklearn_param)
                if config_path is not None:
                    alias_rows_by_path.setdefault(config_path, []).append(sklearn_param)
                row = {
                    "sklearn_param": sklearn_param,
                    "config_path": config_path,
                    "stage": sklearn_param.split("__")[0],
                    "current_value": safe_json_value(value),
                }
                search_space_item = search_space_by_sklearn_param.get(sklearn_param)
                if search_space_item is None and config_path is not None:
                    search_space_item = search_space_by_path.get(config_path)
                row["is_tunable"] = search_space_item is not None
                if search_space_item is not None:
                    row["suggestion_kind"] = search_space_item["kind"]
                    if "choices" in search_space_item:
                        row["choices"] = safe_json_value(search_space_item["choices"])
                    if "low" in search_space_item:
                        row["low"] = safe_json_value(search_space_item["low"])
                    if "high" in search_space_item:
                        row["high"] = safe_json_value(search_space_item["high"])
                    if "log" in search_space_item:
                        row["log"] = bool(search_space_item["log"])
                sklearn_pipeline_params.append(row)
        except Exception:
            sklearn_pipeline_params = []

    for row in flattened:
        search_space_item = search_space_by_path.get(row["path"])
        row["is_tunable"] = search_space_item is not None
        if row["path"] in alias_rows_by_path:
            row["sklearn_params"] = alias_rows_by_path[row["path"]]
            row["sklearn_param"] = alias_rows_by_path[row["path"]][0]
        if search_space_item is not None:
            row["suggestion_kind"] = search_space_item["kind"]
            if "choices" in search_space_item:
                row["choices"] = safe_json_value(search_space_item["choices"])
            if "low" in search_space_item:
                row["low"] = safe_json_value(search_space_item["low"])
            if "high" in search_space_item:
                row["high"] = safe_json_value(search_space_item["high"])
            if "log" in search_space_item:
                row["log"] = bool(search_space_item["log"])

    # Expose both the table-shaped rows and a path-keyed wrapper so callers do
    # not have to infer the intended container type from examples alone.
    pipeline_params_by_path = {row["path"]: dict(row) for row in flattened}

    return {
        "pipeline_params": flattened,
        "pipeline_params_by_path": pipeline_params_by_path,
        "sklearn_pipeline_params": sklearn_pipeline_params,
        "sklearn_param_aliases": normalized_bundle.get("sklearn_param_aliases", {}),
        "search_space": normalized_search_space,
        "recommended_model_params": _COMMON_LIGHTGBM_PARAMS,
        "schema_reference": build_hpo_schema_reference(),
        "return_schema": build_hpo_inspection_return_schema(),
    }


def _condition_matches(
    candidate_config: dict[str, Any],
    param_spec: dict[str, Any],
) -> bool:
    """Return ``True`` when a conditional search-space entry is active.

    Args:
        candidate_config (dict[str, Any]): Current partially materialized config.
        param_spec (dict[str, Any]): Search-space entry.

    Returns:
        bool: Whether the parameter should be sampled.
    """
    condition = param_spec.get("condition")
    if condition is None:
        return True
    try:
        actual_value = _get_nested_value(candidate_config, condition["path"])
    except KeyError:
        return False
    if condition.get("in") is not None:
        return actual_value in condition["in"]
    return actual_value == condition.get("equals")


def apply_search_space_to_config(
    trial: Trial,
    pipeline_config: dict[str, Any],
    search_space: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a structured search space to a pipeline config for one Optuna trial.

    Args:
        trial (Trial): Active Optuna trial.
        pipeline_config (dict[str, Any]): Normalized pipeline config.
        search_space (list[dict[str, Any]]): Normalized search-space entries.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Candidate config and sampled params.
    """
    candidate_config = deepcopy(pipeline_config)
    sampled_params: dict[str, Any] = {}
    for param_spec in search_space:
        if not _condition_matches(candidate_config, param_spec):
            continue

        param_name = str(param_spec.get("sklearn_param", param_spec["path"]))
        if param_spec["kind"] == "categorical":
            sampled_value = trial.suggest_categorical(param_name, param_spec["choices"])
        elif param_spec["kind"] == "int":
            sampled_value = trial.suggest_int(
                param_name,
                int(param_spec["low"]),
                int(param_spec["high"]),
                step=int(param_spec.get("step", 1)),
                log=bool(param_spec.get("log", False)),
            )
        else:
            sampled_value = trial.suggest_float(
                param_name,
                float(param_spec["low"]),
                float(param_spec["high"]),
                step=(
                    float(param_spec["step"])
                    if "step" in param_spec and param_spec.get("log", False) is False
                    else None
                ),
                log=bool(param_spec.get("log", False)),
            )

        _set_nested_value(candidate_config, str(param_spec["path"]), sampled_value)
        sampled_params[param_name] = sampled_value

    return candidate_config, sampled_params


def _subset_dataframe(
    dataframe: pd.DataFrame,
    selected_features: list[str],
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Return a dataframe restricted to selected features and optional target.

    Args:
        dataframe (pd.DataFrame): Source dataframe.
        selected_features (list[str]): Selected feature names.
        target_column (str | None): Optional target column to append.

    Returns:
        pd.DataFrame: Subset dataframe.
    """
    missing_columns = [
        column for column in selected_features if column not in dataframe.columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"Selected features are missing from the current dataframe: {missing_text}."
        )

    subset = dataframe[selected_features].copy()
    if target_column is not None and target_column in dataframe.columns:
        subset[target_column] = dataframe[target_column].values
    return subset


def resolve_selected_features(
    feature_columns: list[str],
    selection_config: dict[str, Any] | None,
    *,
    object_store: Any,
) -> tuple[list[str], list[str]]:
    """Resolve the final feature subset from explicit or report-informed config.

    Args:
        feature_columns (list[str]): Available feature columns at the current stage.
        selection_config (dict[str, Any] | None): Optional selection config.
        object_store (Any): Shared object store used to fetch report artifacts.

    Returns:
        tuple[list[str], list[str]]: Selected features and warnings.
    """
    if selection_config is None:
        return list(feature_columns), []

    warnings: list[str] = []
    if "feature_columns" in selection_config:
        selected_features = _string_list(
            selection_config["feature_columns"],
            field_name="feature_columns",
        )
        return selected_features, warnings

    if "report_handle" not in selection_config:
        raise ValueError(
            "feature_selection must provide either feature_columns or report_handle."
        )

    report = object_store.get(
        str(selection_config["report_handle"]),
        expected_type=StoredFeatureSelectionReport,
    )
    metric_field = str(selection_config.get("metric_field", "score"))
    candidate_rows = []
    for finding in report.findings:
        feature_name = finding.get("feature")
        if feature_name is None:
            continue
        if feature_name not in feature_columns:
            continue
        candidate_rows.append(
            {
                "feature": str(feature_name),
                "metric_value": finding.get(metric_field, finding.get("importance")),
            }
        )

    if not candidate_rows:
        raise ValueError(
            "The supplied feature-selection report does not resolve any usable features for the current pipeline stage."
        )

    candidate_rows.sort(
        key=lambda row: row["metric_value"]
        if row["metric_value"] is not None
        else float("-inf"),
        reverse=True,
    )

    min_score = selection_config.get("min_score")
    if min_score is not None:
        candidate_rows = [
            row
            for row in candidate_rows
            if row["metric_value"] is not None and row["metric_value"] >= min_score
        ]
    max_features = selection_config.get("max_features")
    if max_features is not None:
        candidate_rows = candidate_rows[: int(max_features)]

    selected_features = [row["feature"] for row in candidate_rows]
    include_features = [
        feature
        for feature in _string_list(
            selection_config.get("include_features", []),
            field_name="include_features",
        )
        if feature in feature_columns
    ]
    exclude_features = set(
        _string_list(
            selection_config.get("exclude_features", []),
            field_name="exclude_features",
        )
    )

    for feature in include_features:
        if feature not in selected_features:
            selected_features.append(feature)
    selected_features = [
        feature for feature in selected_features if feature not in exclude_features
    ]

    if report.report_type not in {"target_metrics", "importance"}:
        warnings.append(
            f"Report {report.report_type!r} is not primarily target-aware. Use caution when deriving a final feature subset from it."
        )
    if not selected_features:
        raise ValueError(
            "The feature-selection config resolved to an empty feature subset."
        )
    return selected_features, warnings


def _split_features_target(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and target.

    Args:
        dataframe (pd.DataFrame): Source dataframe.
        target_column (str): Target column name.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature frame and target series.
    """
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found in the dataframe."
        )
    return dataframe.drop(columns=[target_column]).copy(), dataframe[
        target_column
    ].copy()


@dataclass(slots=True)
class BuiltSklearnPipeline:
    """Fitted sklearn pipeline plus config-path aliases for inspection.

    Args:
        pipeline (Pipeline): Fitted sklearn pipeline assembled from the config.
        current_features (pd.DataFrame): Final transformed feature frame before model fit.
        param_aliases (dict[str, str]): Mapping from sklearn param names to config paths.
    """

    pipeline: Pipeline
    current_features: pd.DataFrame
    param_aliases: dict[str, str]


def _attach_preprocessor_metadata(
    wrapper: DataFrameOutputColumnTransformer,
    artifact: StoredPreprocessor,
) -> DataFrameOutputColumnTransformer:
    """Attach fitted metadata from a stored preprocessor onto a wrapper."""
    wrapper.input_columns_ = list(artifact.input_columns)
    wrapper.output_columns_ = list(artifact.output_columns)
    return wrapper


def _register_feature_engineering_aliases(
    *,
    stage_name: str,
    section: dict[str, Any],
    param_aliases: dict[str, str],
) -> None:
    """Register config-path aliases for FE sklearn params."""
    for feature_index, feature_spec in enumerate(section["spec"]["features"]):
        prefix = f"{stage_name}__features__{feature_index}"
        param_aliases[prefix] = f"{stage_name}.spec.features.{feature_index}"
        for key in feature_spec:
            param_aliases[f"{prefix}__{key}"] = (
                f"{stage_name}.spec.features.{feature_index}.{key}"
            )
    param_aliases[f"{stage_name}__conflict_policy"] = (
        f"{stage_name}.spec.conflict_policy"
    )
    param_aliases[f"{stage_name}__drop_source_columns"] = (
        f"{stage_name}.spec.drop_source_columns"
    )
    param_aliases[f"{stage_name}__output__preserve_index"] = (
        f"{stage_name}.spec.output.preserve_index"
    )


def _register_freeform_aliases(
    *,
    stage_name: str,
    section: dict[str, Any],
    param_aliases: dict[str, str],
) -> None:
    """Register config-path aliases for reusable freeform sklearn params."""
    param_aliases[f"{stage_name}__code"] = f"{stage_name}.code"
    param_aliases[f"{stage_name}__intent"] = f"{stage_name}.intent"
    param_aliases[f"{stage_name}__params"] = f"{stage_name}.args"
    for key in section.get("args", {}):
        param_aliases[f"{stage_name}__params__{key}"] = f"{stage_name}.args.{key}"
    param_aliases[f"{stage_name}__strict_schema"] = f"{stage_name}.strict_schema"
    param_aliases[f"{stage_name}__preserve_index"] = (
        f"{stage_name}.output.preserve_index"
    )


def _register_preprocessing_aliases(
    *,
    stage_name: str,
    section: dict[str, Any],
    param_aliases: dict[str, str],
) -> None:
    """Register config-path aliases for preprocessing sklearn params."""
    spec = section["spec"]
    param_aliases[f"{stage_name}__preserve_index"] = (
        f"{stage_name}.spec.output.preserve_index"
    )
    for group_index, group in enumerate(spec["groups"]):
        group_name = str(group["name"])
        group_prefix = f"{stage_name}__transformer__{group_name}"
        param_aliases[group_prefix] = f"{stage_name}.spec.groups.{group_index}"
        for step_index, step in enumerate(group["steps"], start=1):
            step_name = f"{step['kind']}_{step_index}"
            step_prefix = f"{group_prefix}__{step_name}"
            param_aliases[step_prefix] = (
                f"{stage_name}.spec.groups.{group_index}.steps.{step_index - 1}"
            )
            for key in step:
                if key == "kind":
                    continue
                param_aliases[f"{step_prefix}__{key}"] = (
                    f"{stage_name}.spec.groups.{group_index}.steps.{step_index - 1}.{key}"
                )


def _register_model_aliases(
    param_aliases: dict[str, str], model_params: dict[str, Any]
) -> None:
    """Register config-path aliases for LightGBM model params."""
    for key in model_params:
        param_aliases[f"model__{key}"] = f"model.base_params.{key}"


def build_sklearn_pipeline(
    pipeline_config: dict[str, Any],
    *,
    object_store: Any,
    os_access: Any,
) -> BuiltSklearnPipeline:
    """Build and fit a sklearn pipeline view of the normalized candidate config.

    Args:
        pipeline_config (dict[str, Any]): Normalized candidate pipeline config.
        object_store (Any): Shared object store.
        os_access (Any): Workspace OS adapter.

    Returns:
        BuiltSklearnPipeline: Fitted pipeline plus param alias metadata.
    """
    preprocessing_collection = PreprocessingCollection(os_access, object_store)

    target_column = str(pipeline_config["data"]["target_column"])
    train_frame = object_store.get(
        str(pipeline_config["data"]["train_handle"]),
        expected_type=pd.DataFrame,
    )
    current_features, train_target = _split_features_target(
        train_frame,
        target_column=target_column,
    )
    pipeline_steps: list[tuple[str, BaseEstimator]] = []
    param_aliases: dict[str, str] = {}

    for stage_name in _FIXED_PIPELINE_ORDER:
        section = pipeline_config.get(stage_name)
        if section is None:
            continue

        if stage_name == "feature_engineering":
            if "handle" in section:
                artifact = object_store.get(
                    str(section["handle"]),
                    expected_type=StoredFeatureEngineer,
                )
                estimator = artifact.estimator
            else:
                normalized_spec = normalize_feature_engineering_spec(section["spec"])
                estimator = DeterministicFeatureEngineeringTransformer(
                    features=deepcopy(normalized_spec["features"]),
                    conflict_policy=str(normalized_spec["conflict_policy"]),
                    drop_source_columns=list(normalized_spec["drop_source_columns"]),
                    preserve_index=bool(normalized_spec["output"]["preserve_index"]),
                )
                estimator.fit(current_features)
                section = {"spec": normalized_spec}
            pipeline_steps.append((stage_name, estimator))
            current_features = estimator.transform(current_features)
            if "spec" in section:
                _register_feature_engineering_aliases(
                    stage_name=stage_name,
                    section=section,
                    param_aliases=param_aliases,
                )
            continue

        if stage_name == "freeform":
            if "handle" in section:
                artifact = object_store.get(
                    str(section["handle"]),
                    expected_type=StoredFreeformTransformer,
                )
                estimator = artifact.estimator
            else:
                estimator = FreeformDataframeTransformer(
                    code=str(section["code"]),
                    workspace_root=str(os_access.host_workspace_root),
                    intent=str(section.get("intent", "feature_engineering")),
                    params=dict(section.get("args", {})),
                    preserve_index=bool(section["output"]["preserve_index"]),
                    strict_schema=bool(section.get("strict_schema", True)),
                )
                estimator.fit(current_features)
            pipeline_steps.append((stage_name, estimator))
            current_features = estimator.transform(current_features)
            _register_freeform_aliases(
                stage_name=stage_name,
                section=section,
                param_aliases=param_aliases,
            )
            continue

        if stage_name == "preprocessing":
            if "handle" in section:
                artifact = object_store.get(
                    str(section["handle"]),
                    expected_type=StoredPreprocessor,
                )
                estimator = _attach_preprocessor_metadata(
                    DataFrameOutputColumnTransformer(
                        artifact.estimator,
                        preserve_index=artifact.preserve_index,
                        output_columns=artifact.output_columns,
                    ),
                    artifact,
                )
            else:
                normalized_spec = normalize_preprocessing_spec(section["spec"])
                base_transformer, _ = preprocessing_collection._build_estimator(
                    current_features,
                    normalized_spec,
                )
                estimator = DataFrameOutputColumnTransformer(
                    base_transformer,
                    preserve_index=bool(normalized_spec["output"]["preserve_index"]),
                )
                estimator.fit(current_features)
                section = {"spec": normalized_spec}
            pipeline_steps.append((stage_name, estimator))
            current_features = estimator.transform(current_features)
            if "spec" in section:
                _register_preprocessing_aliases(
                    stage_name=stage_name,
                    section=section,
                    param_aliases=param_aliases,
                )
            continue

    task_type = infer_task_type(train_target)
    class_count = (
        int(pd.Series(train_target).dropna().nunique())
        if task_type == "classification"
        else None
    )
    model_estimator = build_lightgbm_estimator(
        task_type=task_type,
        class_count=class_count,
        random_state=int(pipeline_config["evaluation"]["random_state"]),
    )
    model_estimator.set_params(**dict(pipeline_config["model"]["base_params"]))
    _register_model_aliases(
        param_aliases, dict(pipeline_config["model"]["base_params"])
    )
    pipeline_steps.append(("model", model_estimator))

    return BuiltSklearnPipeline(
        pipeline=Pipeline(steps=pipeline_steps),
        current_features=current_features,
        param_aliases=param_aliases,
    )


def materialize_pipeline_data(
    pipeline_config: dict[str, Any],
    *,
    object_store: Any,
    os_access: Any,
) -> PipelineRunArtifacts:
    """Fit and transform the configured pipeline stages for one candidate config.

    Args:
        pipeline_config (dict[str, Any]): Normalized candidate pipeline config.
        object_store (Any): Shared object store.
        os_access (Any): Workspace OS adapter.

    Returns:
        PipelineRunArtifacts: Final feature frames, targets, and fitted stage artifacts.
    """
    feature_engineering_collection = FeatureEngineeringCollection(
        os_access, object_store
    )
    from ..registry.freeform import FreeformCodeCollection

    freeform_collection = FreeformCodeCollection(os_access, object_store)
    preprocessing_collection = PreprocessingCollection(os_access, object_store)

    target_column = str(pipeline_config["data"]["target_column"])
    train_handle = str(pipeline_config["data"]["train_handle"])
    validation_handle = (
        str(pipeline_config["data"]["validation_handle"])
        if pipeline_config["data"]["validation_handle"] is not None
        else None
    )
    original_validation_target = None
    if validation_handle is not None:
        original_validation_frame = object_store.get(
            validation_handle,
            expected_type=pd.DataFrame,
        )
        _, original_validation_target = _split_features_target(
            original_validation_frame,
            target_column=target_column,
        )
    current_train_handle = train_handle
    current_validation_handle = validation_handle
    fitted_feature_engineer: StoredFeatureEngineer | None = None
    fitted_freeform_transformer: StoredFreeformTransformer | None = None
    fitted_preprocessor: StoredPreprocessor | None = None
    warnings: list[str] = []

    for stage_name in _FIXED_PIPELINE_ORDER:
        section = pipeline_config.get(stage_name)
        if section is None:
            continue

        if stage_name == "feature_engineering":
            if "handle" in section:
                feature_engineer_handle = str(section["handle"])
            else:
                feature_engineer_handle = (
                    feature_engineering_collection.fit_feature_engineer(
                        current_train_handle,
                        section["spec"],
                        target_column=target_column,
                    )
                )
            current_train_handle = (
                feature_engineering_collection.transform_with_feature_engineer(
                    current_train_handle,
                    feature_engineer_handle,
                    include_target=True,
                )
            )
            if current_validation_handle is not None:
                current_validation_handle = (
                    feature_engineering_collection.transform_with_feature_engineer(
                        current_validation_handle,
                        feature_engineer_handle,
                        include_target=False,
                    )
                )
            fitted_feature_engineer = object_store.get(
                feature_engineer_handle,
                expected_type=StoredFeatureEngineer,
            )
            continue

        if stage_name == "freeform":
            if "handle" in section:
                freeform_transformer_handle = str(section["handle"])
            else:
                freeform_transformer_handle = (
                    freeform_collection.fit_freeform_transformer(
                        current_train_handle,
                        str(section["code"]),
                        target_column=target_column,
                        intent=str(section.get("intent", "feature_engineering")),
                        args=dict(section.get("args", {})),
                        preserve_index=bool(section["output"]["preserve_index"]),
                        strict_schema=bool(section.get("strict_schema", True)),
                    )
                )
            current_train_handle = (
                freeform_collection.transform_with_freeform_transformer(
                    current_train_handle,
                    freeform_transformer_handle,
                    include_target=True,
                )
            )
            if current_validation_handle is not None:
                current_validation_handle = (
                    freeform_collection.transform_with_freeform_transformer(
                        current_validation_handle,
                        freeform_transformer_handle,
                        include_target=False,
                    )
                )
            fitted_freeform_transformer = object_store.get(
                freeform_transformer_handle,
                expected_type=StoredFreeformTransformer,
            )
            continue

        if stage_name == "preprocessing":
            if "handle" in section:
                preprocessor_handle = str(section["handle"])
            else:
                preprocessor_handle = preprocessing_collection.fit_preprocessor(
                    current_train_handle,
                    section["spec"],
                    target_column=target_column,
                )
            current_train_handle = preprocessing_collection.transform_dataframe(
                current_train_handle,
                preprocessor_handle,
                include_target=True,
            )
            if current_validation_handle is not None:
                current_validation_handle = (
                    preprocessing_collection.transform_dataframe(
                        current_validation_handle,
                        preprocessor_handle,
                    )
                )
            fitted_preprocessor = object_store.get(
                preprocessor_handle,
                expected_type=StoredPreprocessor,
            )
            continue

    train_frame = object_store.get(current_train_handle, expected_type=pd.DataFrame)
    validation_frame = (
        object_store.get(current_validation_handle, expected_type=pd.DataFrame)
        if current_validation_handle is not None
        else None
    )
    train_features, train_target = _split_features_target(
        train_frame,
        target_column=target_column,
    )
    if validation_frame is not None:
        validation_features = validation_frame.copy()
        validation_target = original_validation_target.copy()
    else:
        validation_features, validation_target = None, None

    sklearn_pipeline = build_sklearn_pipeline(
        pipeline_config,
        object_store=object_store,
        os_access=os_access,
    ).pipeline

    return PipelineRunArtifacts(
        train_features=train_features,
        train_target=train_target,
        validation_features=validation_features,
        validation_target=validation_target,
        selected_features=[str(column) for column in train_features.columns],
        preprocessor=fitted_preprocessor,
        feature_engineer=fitted_feature_engineer,
        freeform_transformer=fitted_freeform_transformer,
        sklearn_pipeline=sklearn_pipeline,
        warnings=warnings,
    )


def choose_objective_metric(
    task_type: str,
    requested_metric: str,
    metrics: dict[str, float],
) -> tuple[str, float]:
    """Choose the objective metric value from a metrics dictionary.

    Args:
        task_type (str): Either `classification` or `regression`.
        requested_metric (str): Requested metric name or `auto`.
        metrics (dict[str, float]): Available metric values.

    Returns:
        tuple[str, float]: Chosen metric name and value.
    """
    if requested_metric != "auto":
        if requested_metric not in metrics:
            available = ", ".join(sorted(metrics))
            raise ValueError(
                f"Requested metric {requested_metric!r} is unavailable. Available metrics: {available}."
            )
        return requested_metric, float(metrics[requested_metric])

    if task_type == "classification":
        for candidate in ("roc_auc", "f1", "accuracy"):
            if candidate in metrics:
                return candidate, float(metrics[candidate])
    for candidate in ("r2", "rmse", "mae"):
        if candidate in metrics:
            metric_value = float(metrics[candidate])
            if candidate in {"rmse", "mae"}:
                return candidate, -metric_value
            return candidate, metric_value
    first_metric = next(iter(metrics))
    return first_metric, float(metrics[first_metric])


def fit_final_model(
    *,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    model_params: dict[str, Any],
    random_state: int,
) -> tuple[Any, list[str]]:
    """Fit a final LightGBM estimator on prepared features.

    Args:
        train_features (pd.DataFrame): Training features after all pipeline stages.
        train_target (pd.Series): Training target values.
        model_params (dict[str, Any]): LightGBM parameter overrides.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[Any, list[str]]: Fitted estimator and final model feature columns.
    """
    task_type = infer_task_type(train_target)
    class_count = (
        int(pd.Series(train_target).dropna().nunique())
        if task_type == "classification"
        else None
    )
    train_ready, _ = prepare_model_frames(train_features)
    train_target_ready, _ = prepare_targets(train_target, task_type=task_type)
    estimator = build_lightgbm_estimator(
        task_type=task_type,
        class_count=class_count,
        random_state=random_state,
    )
    estimator.set_params(**model_params)
    estimator.fit(train_ready, train_target_ready)
    return estimator, [str(column) for column in train_ready.columns]


def evaluate_pipeline_candidate(
    pipeline_config: dict[str, Any],
    *,
    object_store: Any,
    os_access: Any,
) -> dict[str, Any]:
    """Evaluate one fully materialized pipeline candidate.

    Args:
        pipeline_config (dict[str, Any]): Normalized candidate pipeline config.
        object_store (Any): Shared object store.
        os_access (Any): Workspace OS adapter.

    Returns:
        dict[str, Any]: Evaluation payload including objective value and summaries.
    """
    artifacts = materialize_pipeline_data(
        pipeline_config,
        object_store=object_store,
        os_access=os_access,
    )
    evaluation_config = pipeline_config["evaluation"]
    from ..registry.metrics_collection import StoredMetricScorer
    from ..registry.splitting import StoredSplitter

    scorer = (
        object_store.get(
            str(evaluation_config["scorer_handle"]),
            expected_type=StoredMetricScorer,
        )
        if evaluation_config.get("scorer_handle") is not None
        else None
    )
    splitter = (
        object_store.get(
            str(evaluation_config["splitter_handle"]),
            expected_type=StoredSplitter,
        )
        if evaluation_config.get("splitter_handle") is not None
        else None
    )
    group_values = None
    if evaluation_config.get("group_column") is not None:
        train_frame = object_store.get(
            str(pipeline_config["data"]["train_handle"]),
            expected_type=pd.DataFrame,
        )
        group_column = str(evaluation_config["group_column"])
        if group_column not in train_frame.columns:
            raise ValueError(
                f"Group column {group_column!r} was not found in the training dataframe."
            )
        group_values = train_frame.loc[artifacts.train_features.index, group_column]
    evaluation_summary, evaluation_warnings = evaluate_subset_metrics(
        artifacts.train_features,
        artifacts.train_target,
        validation_features=artifacts.validation_features,
        validation_target=artifacts.validation_target,
        cv_folds=int(evaluation_config["cv_folds"]),
        random_state=int(evaluation_config["random_state"]),
        model_params=dict(pipeline_config["model"]["base_params"]),
        scorer=scorer,
        splitter=splitter,
        groups=group_values,
    )
    warnings = list(artifacts.warnings or []) + list(evaluation_warnings)
    requested_metric = str(evaluation_config["metric"])
    if requested_metric == "auto" and scorer is not None:
        requested_metric = str(scorer.metric_name)
    if evaluation_config["mode"] == "validation":
        metric_name, objective_value = choose_objective_metric(
            infer_task_type(artifacts.train_target),
            requested_metric,
            evaluation_summary["metrics"],
        )
    else:
        metric_name, objective_value = choose_objective_metric(
            infer_task_type(artifacts.train_target),
            requested_metric,
            evaluation_summary["summary"]["mean_metrics"],
        )
        warnings.append(
            "Cross-validation mode fits preprocessing, freeform feature transforms, and deterministic feature engineering on the provided training dataframe before scoring folds. Treat results as adaptive-search diagnostics, not a leak-free final estimate."
        )

    return {
        "objective_metric": metric_name,
        "objective_value": float(objective_value),
        "evaluation_summary": evaluation_summary,
        "selected_features": artifacts.selected_features,
        "feature_count": len(artifacts.selected_features),
        "warnings": warnings,
        "artifacts": artifacts,
    }


def build_trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Render one Optuna trial into a JSON-friendly record.

    Args:
        trial (optuna.trial.FrozenTrial): Frozen Optuna trial object.

    Returns:
        dict[str, Any]: Trial summary record.
    """
    duration_seconds = None
    if trial.datetime_start is not None and trial.datetime_complete is not None:
        duration_seconds = float(
            (trial.datetime_complete - trial.datetime_start).total_seconds()
        )

    return {
        "trial_number": int(trial.number),
        "status": trial.state.name.lower(),
        "objective_value": float(trial.value) if trial.value is not None else None,
        "sampled_params": safe_json_value(trial.params),
        "objective_metric": trial.user_attrs.get("objective_metric"),
        "evaluation_summary": safe_json_value(
            trial.user_attrs.get("evaluation_summary")
        ),
        "selected_features": safe_json_value(trial.user_attrs.get("selected_features")),
        "feature_count": trial.user_attrs.get("feature_count"),
        "warnings": safe_json_value(trial.user_attrs.get("warnings")),
        "failure_reason": trial.user_attrs.get("failure_reason")
        or trial.system_attrs.get("fail_reason"),
        "duration_seconds": duration_seconds,
    }


def summarize_study_trials(
    trials: list[dict[str, Any]],
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Render a compact study summary from stored trial records.

    Args:
        trials (list[dict[str, Any]]): Completed trial records.
        top_n (int): Maximum number of top trials to retain.

    Returns:
        dict[str, Any]: Summary payload with top trials and score progression.
    """
    completed_trials = [
        trial
        for trial in trials
        if trial.get("objective_value") is not None
        and trial.get("status") == "complete"
    ]
    failed_trials = [trial for trial in trials if trial.get("status") == "fail"]
    top_trials = sorted(
        completed_trials,
        key=lambda trial: float(trial["objective_value"]),
        reverse=True,
    )[:top_n]

    return {
        "completed_trial_count": len(completed_trials),
        "failed_trial_count": len(failed_trials),
        "top_trials": top_trials,
        "recent_failures": [
            {
                "trial_number": int(trial["trial_number"]),
                "failure_reason": trial.get("failure_reason"),
            }
            for trial in failed_trials[-top_n:]
        ],
        "best_score_progression": [
            {
                "trial_number": int(trial["trial_number"]),
                "objective_value": float(trial["objective_value"]),
            }
            for trial in top_trials
        ],
    }
