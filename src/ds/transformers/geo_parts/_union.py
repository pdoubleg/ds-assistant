"""Feature-union helper for geo transformers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

from ._utils import _as_dataframe, _feature_names_or_raise


class GeoFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Simple pandas-friendly feature union.

    sklearn's FeatureUnion works well for numpy arrays, but for tabular modeling
    it is often convenient to preserve DataFrame columns and indexes.
    """

    def __init__(self, transformers: Sequence[tuple[str, TransformerMixin]]):
        self.transformers = transformers

    def _validate_transformer_names(self) -> None:
        names = [name for name, _ in self.transformers]
        if len(names) != len(set(names)):
            raise ValueError("Transformer names must be unique.")

    def _concat_frames(
        self, frames: list[pd.DataFrame], index: pd.Index
    ) -> pd.DataFrame:
        if not frames:
            result = pd.DataFrame(index=index)
        else:
            result = pd.concat(frames, axis=1)

        if result.columns.duplicated().any():
            duplicates = result.columns[result.columns.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate output columns: {duplicates}")

        self.feature_names_out_ = list(result.columns)
        return result

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "GeoFeatureUnion":
        X = _as_dataframe(X)
        self._validate_transformer_names()
        self.fitted_transformers_: list[tuple[str, TransformerMixin]] = []
        frames = []

        for name, transformer in self.transformers:
            fitted = clone(transformer)
            fitted.fit(X, y)
            Xt = _as_dataframe(fitted.transform(X))
            frames.append(Xt)
            self.fitted_transformers_.append((name, fitted))

        self._concat_frames(frames, X.index)
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        X = _as_dataframe(X)
        self._validate_transformer_names()
        self.fitted_transformers_ = []
        frames = []

        for name, transformer in self.transformers:
            fitted = clone(transformer)
            Xt = fitted.fit_transform(X, y)
            Xt = _as_dataframe(Xt)
            frames.append(Xt)
            self.fitted_transformers_.append((name, fitted))

        return self._concat_frames(frames, X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)

        if not hasattr(self, "fitted_transformers_"):
            raise RuntimeError("GeoFeatureUnion has not been fit.")

        frames = []
        for name, transformer in self.fitted_transformers_:
            Xt = transformer.transform(X)
            Xt = _as_dataframe(Xt)
            frames.append(Xt)

        return self._concat_frames(frames, X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)
