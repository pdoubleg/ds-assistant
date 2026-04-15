"""Shared workspace-aware helpers for registry collection modules."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd
import plotly.graph_objects as go

from ...filesystem import HostWorkspaceOSAccess
from .base import ObjectStore, ToolCollection


class WorkspaceToolCollection(ToolCollection):
    """Shared workspace-aware helpers for built-in data tool collections."""

    def __init__(
        self, os_access: HostWorkspaceOSAccess, object_store: ObjectStore
    ) -> None:
        """Initialize the helper collection.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (ObjectStore): Shared handle store for dataframes and figures.
        """
        self._os_access = os_access
        self._object_store = object_store

    def _resolve_host_path(self, path: str) -> Path:
        """Resolve a virtual or relative path to the host workspace."""
        return self._os_access.to_host_path(PurePosixPath(path))

    def _get_dataframe(self, dataframe_handle: str) -> pd.DataFrame:
        """Fetch a dataframe from the object store."""
        return self._object_store.get(dataframe_handle, expected_type=pd.DataFrame)

    def _get_figure(self, figure_handle: str) -> go.Figure:
        """Fetch a Plotly figure from the object store."""
        return self._object_store.get(figure_handle, expected_type=go.Figure)

    def _record_artifact(self, host_path: Path) -> None:
        """Record a host-side artifact for execution result reporting."""
        self._os_access.record_host_artifact(host_path)


__all__ = ["WorkspaceToolCollection"]
