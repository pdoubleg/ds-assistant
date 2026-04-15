"""Shared sklearn/dataframe pipeline helpers for Monty artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..core.registry import safe_json_value


def ensure_dataframe(dataframe: Any, *, field_name: str = "dataframe") -> pd.DataFrame:
    """Return a defensive dataframe copy.

    Args:
        dataframe (Any): Candidate dataframe-like value.
        field_name (str): Human-readable field name for validation errors.

    Returns:
        pd.DataFrame: Defensive dataframe copy.

    Raises:
        TypeError: If the provided value is not a dataframe.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{field_name} must be a pandas.DataFrame.")
    return dataframe.copy()


def materialize_transform_output(
    transformed: Any,
    *,
    index: pd.Index | None,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Convert an estimator output into a stable dataframe.

    Args:
        transformed (Any): Raw estimator output.
        index (pd.Index | None): Optional index to preserve.
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


def append_target_column(
    dataframe: pd.DataFrame,
    *,
    target_column: str | None,
    target_series: pd.Series | None,
    preserve_index: bool,
) -> pd.DataFrame:
    """Append a target column back onto a transformed dataframe.

    Args:
        dataframe (pd.DataFrame): Transformed dataframe.
        target_column (str | None): Optional target column name.
        target_series (pd.Series | None): Optional target series to append.
        preserve_index (bool): Whether the transformed dataframe kept the source index.

    Returns:
        pd.DataFrame: Dataframe with the target appended when requested.
    """
    if target_column is None or target_series is None:
        return dataframe

    updated = dataframe.copy()
    values = target_series if preserve_index else target_series.reset_index(drop=True)
    updated[target_column] = values.values
    return updated


class DataFrameOutputColumnTransformer(BaseEstimator, TransformerMixin):
    """Wrap a `ColumnTransformer` so pipeline stages keep dataframe semantics.

    Args:
        transformer (ColumnTransformer): Unfitted or fitted sklearn column transformer.
        preserve_index (bool): Whether transformed outputs keep their source index.
        output_columns (Sequence[str] | None): Optional explicit output columns.
    """

    def __init__(
        self,
        transformer: ColumnTransformer,
        *,
        preserve_index: bool = True,
        output_columns: Sequence[str] | None = None,
    ) -> None:
        self.transformer = transformer
        self.preserve_index = preserve_index
        self.output_columns = None if output_columns is None else list(output_columns)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> DataFrameOutputColumnTransformer:
        """Fit the wrapped column transformer.

        Args:
            X (pd.DataFrame): Input feature dataframe.
            y (pd.Series | np.ndarray | None): Optional target values.

        Returns:
            DataFrameOutputColumnTransformer: Fitted transformer.
        """
        frame = ensure_dataframe(X, field_name="X")
        self.input_columns_ = [str(column) for column in frame.columns]
        self.transformer.fit(frame, y)
        if self.output_columns is not None:
            self.output_columns_ = list(self.output_columns)
        elif hasattr(self.transformer, "get_feature_names_out"):
            self.output_columns_ = [
                str(name) for name in self.transformer.get_feature_names_out()
            ]
        else:  # pragma: no cover - defensive fallback
            transformed = self.transformer.transform(frame)
            width = (
                transformed.shape[1]
                if hasattr(transformed, "shape")
                else frame.shape[1]
            )
            self.output_columns_ = [f"feature_{index}" for index in range(width)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe and return a dataframe.

        Args:
            X (pd.DataFrame): Input feature dataframe.

        Returns:
            pd.DataFrame: Dataframe-wrapped transformed output.
        """
        check_is_fitted(self, attributes=("input_columns_", "output_columns_"))
        frame = ensure_dataframe(X, field_name="X")
        missing_columns = [
            column for column in self.input_columns_ if column not in frame.columns
        ]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                f"Input dataframe is missing columns required by the preprocessor: {missing_text}."
            )
        transformed = self.transformer.transform(frame[self.input_columns_])
        return materialize_transform_output(
            transformed,
            index=frame.index if self.preserve_index else None,
            columns=self.output_columns_,
        )

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return stable output feature names.

        Args:
            input_features (Sequence[str] | None): Ignored sklearn compatibility hook.

        Returns:
            np.ndarray: Output feature names as an object array.
        """
        del input_features
        check_is_fitted(self, attributes=("output_columns_",))
        return np.asarray(self.output_columns_, dtype=object)


class ColumnSubsetTransformer(BaseEstimator, TransformerMixin):
    """Select a stable subset of dataframe columns inside a sklearn pipeline.

    Args:
        selected_columns (Sequence[str]): Ordered output columns to keep.
        preserve_index (bool): Whether to preserve the dataframe index.
    """

    def __init__(
        self,
        selected_columns: Sequence[str],
        *,
        preserve_index: bool = True,
    ) -> None:
        self.selected_columns = list(selected_columns)
        self.preserve_index = preserve_index

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> ColumnSubsetTransformer:
        """Validate the selected columns against the input dataframe.

        Args:
            X (pd.DataFrame): Input feature dataframe.
            y (pd.Series | np.ndarray | None): Optional target values.

        Returns:
            ColumnSubsetTransformer: Fitted selector.
        """
        del y
        frame = ensure_dataframe(X, field_name="X")
        self.input_columns_ = [str(column) for column in frame.columns]
        self.output_columns_ = list(self.selected_columns)
        missing_columns = [
            column for column in self.output_columns_ if column not in frame.columns
        ]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                f"Selected features are missing from the dataframe: {missing_text}."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the configured subset of columns.

        Args:
            X (pd.DataFrame): Input feature dataframe.

        Returns:
            pd.DataFrame: Selected dataframe view copied into a new dataframe.
        """
        check_is_fitted(self, attributes=("output_columns_",))
        frame = ensure_dataframe(X, field_name="X")
        missing_columns = [
            column for column in self.output_columns_ if column not in frame.columns
        ]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                f"Selected features are missing from the dataframe: {missing_text}."
            )
        selected = frame.loc[:, self.output_columns_].copy()
        if self.preserve_index:
            return selected
        return selected.reset_index(drop=True)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return the selected column names.

        Args:
            input_features (Sequence[str] | None): Ignored sklearn compatibility hook.

        Returns:
            np.ndarray: Selected feature names as an object array.
        """
        del input_features
        check_is_fitted(self, attributes=("output_columns_",))
        return np.asarray(self.output_columns_, dtype=object)


@dataclass(slots=True)
class StoredSklearnStageArtifact:
    """Common persisted shape for sklearn-backed Monty stage artifacts.

    Args:
        estimator (BaseEstimator): Fitted sklearn-compatible estimator.
        spec (dict[str, Any]): Normalized stage specification used at fit time.
        input_columns (list[str]): Input feature columns seen during fitting.
        output_columns (list[str]): Stable output columns after transform.
        target_column (str | None): Optional target excluded during fitting.
        preserve_index (bool): Whether transformed outputs keep the source index.
    """

    estimator: BaseEstimator
    spec: dict[str, Any]
    input_columns: list[str]
    output_columns: list[str]
    target_column: str | None = None
    preserve_index: bool = True

    def is_fitted(self) -> bool:
        """Return `True` when the wrapped estimator appears fitted."""
        try:
            check_is_fitted(self.estimator)
        except (NotFittedError, TypeError, AttributeError):
            return False
        return True

    def _base_summary(
        self,
        *,
        artifact_type: str,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render the common JSON summary fields.

        Args:
            artifact_type (str): User-facing artifact type label.
            max_items (int): Maximum preview items to include.
            max_chars (int): Maximum nested string length.

        Returns:
            dict[str, Any]: Shared JSON summary fields.
        """
        return {
            "type": artifact_type,
            "estimator_class": type(self.estimator).__name__,
            "is_fitted": self.is_fitted(),
            "target_column": self.target_column,
            "input_column_count": len(self.input_columns),
            "output_column_count": len(self.output_columns),
            "input_columns": self.input_columns[:max_items],
            "output_columns": self.output_columns[:max_items],
            "spec": safe_json_value(
                self.spec,
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


__all__ = [
    "ColumnSubsetTransformer",
    "DataFrameOutputColumnTransformer",
    "StoredSklearnStageArtifact",
    "append_target_column",
    "ensure_dataframe",
    "materialize_transform_output",
]
