"""Registry package for the Monty Python REPL."""

from __future__ import annotations

from ..filesystem import HostWorkspaceOSAccess
from .base import (
    FunctionRegistry,
    ObjectStore,
    RegisteredCollection,
    RegisteredFunction,
    ToolCollection,
    ToolMetadata,
    tool,
)
from .eda import (
    DataIOCollection,
    DataframeEDACollection,
    HandleInspectionCollection,
    PlotlyCollection,
)
from .parsing import ParsedToolDocstring, ToolArgument, ToolSpec
from .utils import (
    coerce_group_keys,
    flatten_columns,
    safe_json_value,
)


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
    registry.register_collection(DataIOCollection(os_access, object_store))
    registry.register_collection(DataframeEDACollection(os_access, object_store))
    registry.register_collection(HandleInspectionCollection(os_access, object_store))
    registry.register_collection(PlotlyCollection(os_access, object_store))
    return registry


__all__ = [
    "DataIOCollection",
    "DataframeEDACollection",
    "FunctionRegistry",
    "HandleInspectionCollection",
    "ObjectStore",
    "ParsedToolDocstring",
    "PlotlyCollection",
    "RegisteredCollection",
    "RegisteredFunction",
    "ToolArgument",
    "ToolCollection",
    "ToolMetadata",
    "ToolSpec",
    "build_default_registry",
    "coerce_group_keys",
    "flatten_columns",
    "safe_json_value",
    "tool",
]
