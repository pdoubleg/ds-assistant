"""Cross-validation and dataframe splitting helpers for the Monty registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from ..filesystem import HostWorkspaceOSAccess
from ..core.registry import ObjectStore, ToolCollection, tool
from ..core.registry import safe_json_value

_SUPPORTED_SPLITTER_KINDS = {
    "kfold",
    "stratified_kfold",
    "repeated_kfold",
    "repeated_stratified_kfold",
    "group_kfold",
    "time_series_split",
}


@dataclass(slots=True)
class StoredSplitter:
    """Persisted sklearn-style splitter configuration.

    Args:
        splitter_kind (str): Canonical splitter identifier.
        params (dict[str, Any]): Normalized constructor kwargs.
        requires_target (bool): Whether ``split(...)`` expects target labels.
        requires_groups (bool): Whether ``split(...)`` expects group values.
    """

    splitter_kind: str
    params: dict[str, Any] = field(default_factory=dict)
    requires_target: bool = False
    requires_groups: bool = False

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection."""
        return {
            "type": "StoredSplitter",
            "splitter_kind": self.splitter_kind,
            "requires_target": self.requires_target,
            "requires_groups": self.requires_groups,
            "params": safe_json_value(
                self.params,
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


@dataclass(slots=True)
class StoredDataSplit:
    """Persisted dataframe split artifact.

    Args:
        split_kind (str): High-level split family such as ``holdout``.
        source_handle (str): Source dataframe handle.
        train_handle (str): Handle for the training dataframe.
        validation_handle (str | None): Optional validation dataframe handle.
        test_handle (str | None): Optional test dataframe handle.
        target_column (str | None): Optional target column used for stratification.
        row_counts (dict[str, int]): Row counts for each split.
        params (dict[str, Any]): Normalized split settings.
        warnings (list[str]): User-facing caveats for the split.
    """

    split_kind: str
    source_handle: str
    train_handle: str
    validation_handle: str | None = None
    test_handle: str | None = None
    target_column: str | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection."""
        return {
            "type": "StoredDataSplit",
            "split_kind": self.split_kind,
            "source_handle": self.source_handle,
            "train_handle": self.train_handle,
            "validation_handle": self.validation_handle,
            "test_handle": self.test_handle,
            "target_column": self.target_column,
            "row_counts": self.row_counts,
            "params": safe_json_value(
                self.params,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "warnings": self.warnings[:max_items],
        }


def materialize_splitter(splitter: StoredSplitter) -> Any:
    """Build a sklearn splitter instance from a stored splitter artifact.

    Args:
        splitter (StoredSplitter): Stored splitter configuration.

    Returns:
        Any: sklearn model-selection splitter instance.
    """
    if splitter.splitter_kind not in _SUPPORTED_SPLITTER_KINDS:
        supported = ", ".join(sorted(_SUPPORTED_SPLITTER_KINDS))
        raise ValueError(
            f"Unsupported splitter kind {splitter.splitter_kind!r}. Supported values: {supported}."
        )

    params = dict(splitter.params)
    if splitter.splitter_kind == "kfold":
        return KFold(**params)
    if splitter.splitter_kind == "stratified_kfold":
        return StratifiedKFold(**params)
    if splitter.splitter_kind == "repeated_kfold":
        return RepeatedKFold(**params)
    if splitter.splitter_kind == "repeated_stratified_kfold":
        return RepeatedStratifiedKFold(**params)
    if splitter.splitter_kind == "group_kfold":
        return GroupKFold(**params)
    return TimeSeriesSplit(**params)


class SplittingCollection(ToolCollection):
    """Reusable splitter builders and dataframe split helpers."""

    name = "splitting"
    description = (
        "Create reusable cross-validation splitter handles and materialize "
        "holdout-style dataframe splits for downstream modeling workflows."
    )

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: ObjectStore,
    ) -> None:
        """Initialize splitting helpers.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (ObjectStore): Shared handle store.
        """
        self._os_access = os_access
        self._object_store = object_store

    def _get_dataframe(self, dataframe_handle: str) -> pd.DataFrame:
        """Fetch a dataframe from the shared object store."""
        return self._object_store.get(dataframe_handle, expected_type=pd.DataFrame)

    def _get_splitter(self, splitter_handle: str) -> StoredSplitter:
        """Fetch a stored splitter from the object store."""
        return self._object_store.get(splitter_handle, expected_type=StoredSplitter)

    def _get_data_split(self, split_handle: str) -> StoredDataSplit:
        """Fetch a stored split artifact from the object store."""
        return self._object_store.get(split_handle, expected_type=StoredDataSplit)

    def _put_splitter(
        self,
        splitter_kind: str,
        *,
        params: dict[str, Any],
        requires_target: bool = False,
        requires_groups: bool = False,
    ) -> str:
        """Persist a normalized splitter artifact and return its handle."""
        artifact = StoredSplitter(
            splitter_kind=splitter_kind,
            params=params,
            requires_target=requires_target,
            requires_groups=requires_groups,
        )
        return self._object_store.put(artifact, prefix="splitter")

    @tool
    def create_kfold_splitter(
        self,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 0,
    ) -> str:
        """Create a reusable ``KFold`` splitter handle.

        Args:
            n_splits (int): Number of folds to generate.
            shuffle (bool): Whether to shuffle rows before splitting.
            random_state (int): Random seed used when shuffling is enabled.

        Returns:
            str: Handle for the stored splitter artifact.

        Examples:
            splitter_handle = create_kfold_splitter(
                n_splits=5,
                shuffle=True,
                random_state=0,
            )
        """
        return self._put_splitter(
            "kfold",
            params={
                "n_splits": int(n_splits),
                "shuffle": bool(shuffle),
                "random_state": int(random_state) if shuffle else None,
            },
        )

    @tool
    def create_stratified_kfold_splitter(
        self,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 0,
    ) -> str:
        """Create a reusable ``StratifiedKFold`` splitter handle.

        Args:
            n_splits (int): Number of folds to generate.
            shuffle (bool): Whether to shuffle rows before splitting.
            random_state (int): Random seed used when shuffling is enabled.

        Returns:
            str: Handle for the stored splitter artifact.

        Examples:
            splitter_handle = create_stratified_kfold_splitter(
                n_splits=5,
                shuffle=True,
                random_state=0,
            )
        """
        return self._put_splitter(
            "stratified_kfold",
            params={
                "n_splits": int(n_splits),
                "shuffle": bool(shuffle),
                "random_state": int(random_state) if shuffle else None,
            },
            requires_target=True,
        )

    @tool
    def create_repeated_kfold_splitter(
        self,
        *,
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int = 0,
    ) -> str:
        """Create a reusable ``RepeatedKFold`` splitter handle.

        Args:
            n_splits (int): Number of folds in each repeat.
            n_repeats (int): Number of repeated fold rounds.
            random_state (int): Random seed used by the repeated splitter.

        Returns:
            str: Handle for the stored splitter artifact.

        Examples:
            splitter_handle = create_repeated_kfold_splitter(
                n_splits=5,
                n_repeats=3,
                random_state=0,
            )
        """
        return self._put_splitter(
            "repeated_kfold",
            params={
                "n_splits": int(n_splits),
                "n_repeats": int(n_repeats),
                "random_state": int(random_state),
            },
        )

    @tool
    def create_repeated_stratified_kfold_splitter(
        self,
        *,
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int = 0,
    ) -> str:
        """Create a reusable ``RepeatedStratifiedKFold`` splitter handle.

        Args:
            n_splits (int): Number of folds in each repeat.
            n_repeats (int): Number of repeated fold rounds.
            random_state (int): Random seed used by the repeated splitter.

        Returns:
            str: Handle for the stored splitter artifact.

        Examples:
            splitter_handle = create_repeated_stratified_kfold_splitter(
                n_splits=5,
                n_repeats=3,
                random_state=0,
            )
        """
        return self._put_splitter(
            "repeated_stratified_kfold",
            params={
                "n_splits": int(n_splits),
                "n_repeats": int(n_repeats),
                "random_state": int(random_state),
            },
            requires_target=True,
        )

    @tool
    def create_group_kfold_splitter(
        self,
        *,
        n_splits: int = 5,
    ) -> str:
        """Create a reusable ``GroupKFold`` splitter handle.

        Args:
            n_splits (int): Number of folds to generate.

        Returns:
            str: Handle for the stored splitter artifact.

        Examples:
            splitter_handle = create_group_kfold_splitter(n_splits=5)
        """
        return self._put_splitter(
            "group_kfold",
            params={"n_splits": int(n_splits)},
            requires_groups=True,
        )

    @tool
    def create_time_series_splitter(
        self,
        *,
        n_splits: int = 5,
        test_size: int | None = None,
        gap: int = 0,
    ) -> str:
        """Create a reusable ``TimeSeriesSplit`` splitter handle.

        Args:
            n_splits (int): Number of sequential splits to generate.
            test_size (int | None): Optional fixed test-window size.
            gap (int): Number of rows to skip between train and test windows.

        Returns:
            str: Handle for the stored splitter artifact.

        Examples:
            splitter_handle = create_time_series_splitter(n_splits=5, test_size=24)
        """
        params: dict[str, Any] = {
            "n_splits": int(n_splits),
            "gap": int(gap),
        }
        if test_size is not None:
            params["test_size"] = int(test_size)
        return self._put_splitter("time_series_split", params=params)

    @tool
    def inspect_splitter(self, splitter_handle: str) -> dict[str, Any]:
        """Return a JSON-friendly summary of a stored splitter handle.

        Args:
            splitter_handle (str): Handle pointing to a stored splitter artifact.

        Returns:
            dict[str, Any]: JSON-friendly splitter summary.

        Examples:
            print(inspect_splitter(splitter_handle))
            # Returns:
            # {
            #     "splitter_kind": "kfold",
            #     "params": {"n_splits": 5, "shuffle": True, "random_state": 0},
            #     "requires_target": False,
            #     "requires_groups": False
            # }
        """
        return self._get_splitter(splitter_handle).to_json_summary()

    @tool
    def make_holdout_split(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        test_size: float = 0.2,
        random_state: int = 0,
        stratify: bool = False,
    ) -> str:
        """Split one dataframe into training and test subsets.

        Args:
            dataframe_handle (str): Input dataframe handle.
            target_column (str | None): Optional target column used for stratification.
            test_size (float): Fraction of rows assigned to the test split.
            random_state (int): Random seed for reproducibility.
            stratify (bool): Whether to stratify using the target column.

        Returns:
            str: Handle for the stored holdout split artifact.

        Examples:
            split_handle = make_holdout_split(
                df_handle,
                target_column="target",
                test_size=0.2,
                stratify=True,
            )
        """
        dataframe = self._get_dataframe(dataframe_handle)
        stratify_values = None
        if stratify:
            if target_column is None:
                raise ValueError("target_column is required when stratify=True.")
            if target_column not in dataframe.columns:
                raise ValueError(
                    f"Target column {target_column!r} was not found in the dataframe."
                )
            stratify_values = dataframe[target_column]

        train_frame, test_frame = train_test_split(
            dataframe,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=stratify_values,
        )
        train_handle = self._object_store.put(train_frame.copy(), prefix="df")
        test_handle = self._object_store.put(test_frame.copy(), prefix="df")
        artifact = StoredDataSplit(
            split_kind="holdout",
            source_handle=dataframe_handle,
            train_handle=train_handle,
            test_handle=test_handle,
            target_column=target_column,
            row_counts={
                "train": int(len(train_frame)),
                "test": int(len(test_frame)),
            },
            params={
                "test_size": float(test_size),
                "random_state": int(random_state),
                "stratify": bool(stratify),
            },
        )
        return self._object_store.put(artifact, prefix="split")

    @tool
    def train_validation_test_split(
        self,
        dataframe_handle: str,
        *,
        target_column: str | None = None,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 0,
        stratify: bool = False,
    ) -> str:
        """Split one dataframe into train, validation, and test subsets.

        Args:
            dataframe_handle (str): Input dataframe handle.
            target_column (str | None): Optional target column used for stratification.
            validation_size (float): Fraction of rows assigned to validation.
            test_size (float): Fraction of rows assigned to test.
            random_state (int): Random seed for reproducibility.
            stratify (bool): Whether to stratify using the target column.

        Returns:
            str: Handle for the stored split artifact.

        Examples:
            split_handle = train_validation_test_split(
                df_handle,
                target_column="target",
                validation_size=0.2,
                test_size=0.2,
                stratify=True,
            )
        """
        if validation_size <= 0 or test_size <= 0:
            raise ValueError("validation_size and test_size must both be positive.")
        if validation_size + test_size >= 1:
            raise ValueError("validation_size + test_size must be less than 1.")

        dataframe = self._get_dataframe(dataframe_handle)
        stratify_values = None
        if stratify:
            if target_column is None:
                raise ValueError("target_column is required when stratify=True.")
            if target_column not in dataframe.columns:
                raise ValueError(
                    f"Target column {target_column!r} was not found in the dataframe."
                )
            stratify_values = dataframe[target_column]

        train_frame, remainder_frame = train_test_split(
            dataframe,
            test_size=float(validation_size + test_size),
            random_state=int(random_state),
            stratify=stratify_values,
        )

        remainder_stratify = None
        if stratify_values is not None:
            remainder_stratify = dataframe.loc[remainder_frame.index, target_column]
        relative_test_size = float(test_size / (validation_size + test_size))
        validation_frame, test_frame = train_test_split(
            remainder_frame,
            test_size=relative_test_size,
            random_state=int(random_state),
            stratify=remainder_stratify,
        )

        train_handle = self._object_store.put(train_frame.copy(), prefix="df")
        validation_handle = self._object_store.put(validation_frame.copy(), prefix="df")
        test_handle = self._object_store.put(test_frame.copy(), prefix="df")
        artifact = StoredDataSplit(
            split_kind="train_validation_test",
            source_handle=dataframe_handle,
            train_handle=train_handle,
            validation_handle=validation_handle,
            test_handle=test_handle,
            target_column=target_column,
            row_counts={
                "train": int(len(train_frame)),
                "validation": int(len(validation_frame)),
                "test": int(len(test_frame)),
            },
            params={
                "validation_size": float(validation_size),
                "test_size": float(test_size),
                "random_state": int(random_state),
                "stratify": bool(stratify),
            },
        )
        return self._object_store.put(artifact, prefix="split")

    @tool
    def inspect_data_split(self, split_handle: str) -> dict[str, Any]:
        """Return a JSON-friendly summary of a stored data split.

        Args:
            split_handle (str): Handle pointing to a stored split artifact.

        Returns:
            dict[str, Any]: JSON-friendly split summary.

        Examples:
            print(inspect_data_split(split_handle))
            # Returns:
            # {
            #     "split_kind": "holdout",
            #     "source_handle": "df_1",
            #     "train_handle": "df_2",
            #     "test_handle": "df_3",
            #     "row_counts": {"train": 800, "test": 200}
            # }
        """
        return self._get_data_split(split_handle).to_json_summary()


__all__ = [
    "SplittingCollection",
    "StoredDataSplit",
    "StoredSplitter",
    "materialize_splitter",
]
