"""Shared registry helpers for the hackathon Monty package."""

from __future__ import annotations

import inspect
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from .core.registry import ObjectStore, ToolCollection
from .core.registry.base import _get_tool_metadata
from .core.registry.parsing import build_tool_spec
from .filesystem import HostWorkspaceOSAccess

from .privacy import safe_json_value


class SafeObjectStore(ObjectStore):
    """Object store that returns privacy-safe handle summaries."""

    def summary(self, handle: str) -> dict[str, Any]:
        """Return a privacy-safe JSON summary for a stored object.

        Args:
            handle: Stored object handle.

        Returns:
            JSON-friendly metadata for the stored object.
        """
        return {"handle": handle, "value": safe_json_value(self.get(handle))}


class WorkspaceToolCollection(ToolCollection):
    """Workspace-aware base collection for hackathon tools."""

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: SafeObjectStore,
    ) -> None:
        """Initialize the shared workspace-aware tool helpers.

        Args:
            os_access: Workspace path sandbox adapter.
            object_store: Shared in-memory object store.
        """
        self._os_access = os_access
        self._object_store = object_store

    def _resolve_host_path(self, path: str) -> Path:
        """Resolve a virtual or relative workspace path.

        Args:
            path: Virtual or relative path under `/workspace`.

        Returns:
            Host filesystem path under the configured workspace root.
        """
        return self._os_access.to_host_path(PurePosixPath(path))

    def _get_dataframe(self, dataframe_handle: str) -> pd.DataFrame:
        """Fetch a dataframe from the shared object store.

        Args:
            dataframe_handle: Stored dataframe handle.

        Returns:
            Stored pandas dataframe.
        """
        return self._object_store.get(dataframe_handle, expected_type=pd.DataFrame)

    def _record_artifact(self, host_path: Path) -> None:
        """Record a host-side artifact path for execution reporting.

        Args:
            host_path: Host path to the newly written artifact.
        """
        self._os_access.record_host_artifact(host_path)

    def _ensure_tool_docstring(self, member: Any) -> None:
        """Auto-fill missing required docstring sections for registry tools.

        Args:
            member: Bound callable exposed as a tool.
        """
        raw_callable = getattr(member, "__func__", member)
        unwrapped_callable = inspect.unwrap(raw_callable)
        current_doc = inspect.getdoc(unwrapped_callable) or ""
        signature = inspect.signature(member)
        parameter_names = list(signature.parameters)

        parts = [current_doc.strip() or f"Tool `{unwrapped_callable.__name__}`."]
        if parameter_names and "Args:" not in current_doc:
            args_lines = ["Args:"]
            for parameter_name in parameter_names:
                args_lines.append(
                    f"    {parameter_name}: Auto-generated argument description."
                )
            parts.append("\n".join(args_lines))
        if "Returns:" not in current_doc:
            parts.append("Returns:\n    Any: Auto-generated return description.")
        if "Examples:" not in current_doc:
            example_call = ", ".join("..." for _ in parameter_names)
            parts.append(
                f"Examples:\n    result = {unwrapped_callable.__name__}({example_call})"
            )
        updated_doc = "\n\n".join(part for part in parts if part).strip()
        raw_callable.__doc__ = updated_doc
        unwrapped_callable.__doc__ = updated_doc

    def tools(self) -> list[Any]:
        """Return decorated tool specs with auto-completed docstrings.

        Returns:
            Decorated tool specs registered by the collection.
        """
        specs: list[Any] = []
        for _, member in inspect.getmembers(self, predicate=callable):
            if _get_tool_metadata(member) is None:
                continue
            self._ensure_tool_docstring(member)
            specs.append(
                build_tool_spec(
                    member,
                    collection=self.collection_name,
                    collection_description=self.collection_description,
                )
            )
        return specs


__all__ = ["HackathonWorkspaceToolCollection", "SafeObjectStore"]
