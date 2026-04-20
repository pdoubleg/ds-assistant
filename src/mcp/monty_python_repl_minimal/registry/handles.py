"""Handle inspection tools for the minimal registry package."""

from __future__ import annotations

from typing import Any

from ..base import WorkspaceToolCollection
from ..core.registry import tool


class HandleInspectionCollection(WorkspaceToolCollection):
    """Inspect active in-memory handles without exposing row values."""

    name = "handles"
    description = "Inspect dataframe, report, study, and model handles stored by Monty."

    @tool
    def list_object_handles(self) -> list[str]:
        """List active handles currently stored in memory.

        Use this before inspection when you want to see which artifacts or
        dataframes are available in the current REPL session.

        Returns:
            list[str]: Active object handles in insertion order.

        Examples:
            ```python
            handles = list_object_handles()
            # Returns
            # ["df_abc123", "report_abc123", "study_abc123", "model_abc123"]
            ```
        """

        return self._object_store.list_handles()

    @tool
    def inspect_handle(self, handle: str) -> dict[str, Any]:
        """Return a privacy-safe summary for any stored handle.

        This works for dataframes, feature reports, fitted models, and other
        registered artifacts stored in the shared object store.

        Args:
            handle (str): Stored handle to inspect.

        Returns:
            dict[str, Any]: Safe object summary for the requested handle.

        Examples:
            ```python
            dataset = load_csv("/workspace/train.csv", nrows=2000)
            details = inspect_handle(dataset["dataframe_handle"])
            # Returns
            # {
            #     "handle": "df_abc123",
            #     "value": {
            #         "type": "DataFrame",
            #         "shape": [2000, 10],
            #         "columns": ["customer_id", "balance", "target"],
            #     }
            # }
            ```
        """

        return self._object_store.summary(handle)


__all__ = ["HandleInspectionCollection"]
