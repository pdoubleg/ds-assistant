"""Modeling-focused safe registry for the minimal Monty REPL."""

from __future__ import annotations

from ..base import SafeObjectStore
from ..core.registry import FunctionRegistry
from ..filesystem import HostWorkspaceOSAccess
from .base import (
    StoredDataframeReport,
    StoredFeatureEngineeringPipeline,
    StoredFeatureSelectionReport,
    StoredLightGBMModelArtifact,
    StoredLightGBMStudy,
)
from .data_access import DataAccessCollection, WorkspaceFileCollection
from .data_views import DataViewCollection
from .feature_engineering import FeatureEngineeringCollection
from .feature_workbench import FeatureSelectionCollection
from .handles import HandleInspectionCollection
from .modeling import ModelingCollection
from .utils import ppv_at_top_p
from .visualizations import VisualizationCollection


def build_default_registry(
    os_access: HostWorkspaceOSAccess,
    object_store: SafeObjectStore,
) -> FunctionRegistry:
    """Build the default minimal registry.

    Args:
        os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
        object_store (SafeObjectStore): Shared in-memory object store.

    Returns:
        FunctionRegistry: Registry populated with all minimal collections.
    """

    registry = FunctionRegistry()
    registry.register_collection(DataAccessCollection(os_access, object_store))
    registry.register_collection(WorkspaceFileCollection(os_access, object_store))
    registry.register_collection(HandleInspectionCollection(os_access, object_store))
    registry.register_collection(DataViewCollection(os_access, object_store))
    registry.register_collection(VisualizationCollection(os_access, object_store))
    registry.register_collection(FeatureSelectionCollection(os_access, object_store))
    registry.register_collection(FeatureEngineeringCollection(os_access, object_store))
    registry.register_collection(ModelingCollection(os_access, object_store))
    return registry


__all__ = [
    "StoredDataframeReport",
    "StoredFeatureEngineeringPipeline",
    "StoredFeatureSelectionReport",
    "StoredLightGBMModelArtifact",
    "StoredLightGBMStudy",
    "build_default_registry",
    "ppv_at_top_p",
]
