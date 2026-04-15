"""Registry package for the Monty Python REPL."""

from __future__ import annotations

from ..filesystem import HostWorkspaceOSAccess
from ..core.registry import (
    FunctionRegistry,
    ObjectStore,
    ParsedToolDocstring,
    RegisteredCollection,
    RegisteredFunction,
    ToolArgument,
    ToolCollection,
    ToolDocstringValidationError,
    ToolMetadata,
    ToolSpec,
    tool,
    validate_tool_docstring,
)
from .eda import DataframeEDACollection, PlotlyCollection
from .freeform import FreeformCodeCollection, StoredFreeformTransformer
from .feature_engineering import (
    FeatureEngineeringCollection,
    ResolvedFeatureDefinition,
    StoredFeatureEngineer,
    normalize_feature_engineering_spec,
)
from .feature_selection import (
    FeatureSelectionCollection,
    StoredFeatureSelectionReport,
)
from .hpo import (
    HpoCollection,
    StoredHpoStudy,
    StoredTunedPipeline,
)
from .metrics_collection import (
    MetricsCollection,
    StoredMetricScorer,
    materialize_metric_scorer,
)
from ..support.hpo_utils import (
    build_hpo_config_bundle,
    inspect_pipeline_params,
    normalize_pipeline_config,
    normalize_search_space,
)
from ..support.metrics import (
    build_lightgbm_estimator,
    compute_prediction_metrics,
    evaluate_feature_subset,
    infer_task_type,
    prepare_model_frames,
    prepare_targets,
    rank_feature_target_metrics,
    rank_lightgbm_importance,
    summarize_cv_metrics,
)
from .preprocessing import (
    PreprocessingCollection,
    ResolvedPreprocessingGroup,
    StoredPreprocessor,
    normalize_preprocessing_spec,
)
from .splitting import (
    SplittingCollection,
    StoredDataSplit,
    StoredSplitter,
    materialize_splitter,
)
from .workspace_io import (
    DataIOCollection,
    HandleInspectionCollection,
    WorkspaceFileCollection,
)
from ..core.registry import coerce_group_keys, flatten_columns, safe_json_value


def build_default_registry(
    os_access: HostWorkspaceOSAccess,
    object_store: ObjectStore,
) -> FunctionRegistry:
    """Create the default sandbox helper registry.

    Args:
        os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
        object_store (ObjectStore): Shared store for dataframes and figures.

    Returns:
        FunctionRegistry: Default Monty registry populated with built-in collections.
    """
    registry = FunctionRegistry()
    registry.register_collection(WorkspaceFileCollection(os_access, object_store))
    registry.register_collection(DataIOCollection(os_access, object_store))
    registry.register_collection(DataframeEDACollection(os_access, object_store))
    registry.register_collection(HandleInspectionCollection(os_access, object_store))
    registry.register_collection(PlotlyCollection(os_access, object_store))
    registry.register_collection(PreprocessingCollection(os_access, object_store))
    registry.register_collection(FeatureEngineeringCollection(os_access, object_store))
    registry.register_collection(FeatureSelectionCollection(os_access, object_store))
    registry.register_collection(MetricsCollection(os_access, object_store))
    registry.register_collection(SplittingCollection(os_access, object_store))
    registry.register_collection(HpoCollection(os_access, object_store))
    registry.register_collection(FreeformCodeCollection(os_access, object_store))
    return registry


__all__ = [
    "DataIOCollection",
    "DataframeEDACollection",
    "FeatureEngineeringCollection",
    "FeatureSelectionCollection",
    "FreeformCodeCollection",
    "FunctionRegistry",
    "HandleInspectionCollection",
    "HpoCollection",
    "MetricsCollection",
    "ObjectStore",
    "ParsedToolDocstring",
    "PlotlyCollection",
    "PreprocessingCollection",
    "RegisteredCollection",
    "RegisteredFunction",
    "ResolvedFeatureDefinition",
    "ResolvedPreprocessingGroup",
    "SplittingCollection",
    "StoredDataSplit",
    "StoredFeatureEngineer",
    "StoredFeatureSelectionReport",
    "StoredFreeformTransformer",
    "StoredHpoStudy",
    "StoredMetricScorer",
    "StoredPreprocessor",
    "StoredSplitter",
    "StoredTunedPipeline",
    "ToolArgument",
    "ToolDocstringValidationError",
    "ToolCollection",
    "ToolMetadata",
    "ToolSpec",
    "WorkspaceFileCollection",
    "build_hpo_config_bundle",
    "build_lightgbm_estimator",
    "build_default_registry",
    "compute_prediction_metrics",
    "coerce_group_keys",
    "evaluate_feature_subset",
    "flatten_columns",
    "infer_task_type",
    "inspect_pipeline_params",
    "materialize_metric_scorer",
    "materialize_splitter",
    "normalize_feature_engineering_spec",
    "normalize_pipeline_config",
    "normalize_preprocessing_spec",
    "normalize_search_space",
    "prepare_model_frames",
    "prepare_targets",
    "rank_feature_target_metrics",
    "rank_lightgbm_importance",
    "safe_json_value",
    "summarize_cv_metrics",
    "tool",
    "validate_tool_docstring",
]
