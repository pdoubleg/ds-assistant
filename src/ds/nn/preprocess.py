from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@dataclass
class CategoryVocab:
    column: str
    category_to_index: dict[Any, int]
    missing_index: int = 0
    unknown_index: int = 1

    @property
    def cardinality(self) -> int:
        return max(self.category_to_index.values(), default=1) + 1


class TabularTorchPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-style preprocessor for PyTorch tabular models.

    Numeric output:
        x_num: float32 array of shape (n_rows, n_numeric_features)

    Categorical output:
        x_cat: int64 array of shape (n_rows, n_categorical_features)

    Reserved categorical indices:
        0 = missing
        1 = unknown / unseen
        2+ = observed categories
    """

    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        numeric_scaling: str = "standard",
        log1p_columns: list[str] | None = None,
        min_category_frequency: int = 1,
        max_categories: int | None = None,
        missing_token: str = "__MISSING__",
        unknown_token: str = "__UNKNOWN__",
        fill_numeric_missing: bool = True,
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.numeric_scaling = numeric_scaling
        self.log1p_columns = log1p_columns
        self.min_category_frequency = min_category_frequency
        self.max_categories = max_categories
        self.missing_token = missing_token
        self.unknown_token = unknown_token
        self.fill_numeric_missing = fill_numeric_missing

    def fit(self, X: pd.DataFrame, y: Any = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabularTorchPreprocessor expects a pandas DataFrame.")

        if self.numeric_columns is None:
            self.numeric_columns_ = X.select_dtypes(include=["number"]).columns.tolist()
        else:
            self.numeric_columns_ = list(self.numeric_columns)

        if self.categorical_columns is None:
            self.categorical_columns_ = X.select_dtypes(
                include=["object", "category", "string", "bool"]
            ).columns.tolist()
        else:
            self.categorical_columns_ = list(self.categorical_columns)

        self.log1p_columns_ = list(self.log1p_columns or [])

        missing_numeric_cols = set(self.log1p_columns_) - set(self.numeric_columns_)
        if missing_numeric_cols:
            raise ValueError(
                f"log1p_columns must be numeric columns. Invalid columns: {missing_numeric_cols}"
            )

        self._fit_numeric(X)
        self._fit_categorical(X)

        return self

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        check_is_fitted(
            self,
            [
                "numeric_columns_",
                "categorical_columns_",
                "numeric_feature_names_",
                "categorical_feature_names_",
                "category_cardinalities_",
            ],
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabularTorchPreprocessor expects a pandas DataFrame.")

        x_num = self._transform_numeric(X)
        x_cat = self._transform_categorical(X)

        return {
            "x_num": x_num,
            "x_cat": x_cat,
        }

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> dict[str, np.ndarray]:
        return self.fit(X, y).transform(X)

    def _fit_numeric(self, X: pd.DataFrame) -> None:
        self.numeric_feature_names_ = list(self.numeric_columns_)

        if not self.numeric_columns_:
            self.numeric_fill_values_ = np.array([], dtype=np.float32)
            self.numeric_center_ = np.array([], dtype=np.float32)
            self.numeric_scale_ = np.array([], dtype=np.float32)
            return

        Xn = X[self.numeric_columns_].astype(float).copy()

        for col in self.log1p_columns_:
            values = Xn[col].to_numpy(dtype=np.float64)
            values = np.where(values < 0, np.nan, values)
            Xn[col] = np.log1p(values)

        self.numeric_fill_values_ = Xn.median().to_numpy(dtype=np.float32)

        if self.fill_numeric_missing:
            Xn = Xn.fillna(
                pd.Series(self.numeric_fill_values_, index=self.numeric_columns_)
            )

        if self.numeric_scaling == "standard":
            center = Xn.mean().to_numpy(dtype=np.float32)
            scale = Xn.std(ddof=0).replace(0, 1).to_numpy(dtype=np.float32)

        elif self.numeric_scaling == "robust":
            center = Xn.median().to_numpy(dtype=np.float32)
            q75 = Xn.quantile(0.75)
            q25 = Xn.quantile(0.25)
            scale = (q75 - q25).replace(0, 1).to_numpy(dtype=np.float32)

        elif self.numeric_scaling in {"none", None}:
            center = np.zeros(len(self.numeric_columns_), dtype=np.float32)
            scale = np.ones(len(self.numeric_columns_), dtype=np.float32)

        else:
            raise ValueError("numeric_scaling must be 'standard', 'robust', or 'none'.")

        scale = np.where(scale == 0, 1.0, scale)

        self.numeric_center_ = center.astype(np.float32)
        self.numeric_scale_ = scale.astype(np.float32)

    def _transform_numeric(self, X: pd.DataFrame) -> np.ndarray:
        if not self.numeric_columns_:
            return np.empty((len(X), 0), dtype=np.float32)

        Xn = X[self.numeric_columns_].astype(float).copy()

        for col in self.log1p_columns_:
            values = Xn[col].to_numpy(dtype=np.float64)
            values = np.where(values < 0, np.nan, values)
            Xn[col] = np.log1p(values)

        arr = Xn.to_numpy(dtype=np.float32)

        if self.fill_numeric_missing:
            arr = np.where(np.isnan(arr), self.numeric_fill_values_, arr)

        arr = (arr - self.numeric_center_) / self.numeric_scale_

        return arr.astype(np.float32)

    def _fit_categorical(self, X: pd.DataFrame) -> None:
        self.categorical_feature_names_ = list(self.categorical_columns_)
        self.category_vocabs_: dict[str, CategoryVocab] = {}

        for col in self.categorical_columns_:
            vc = X[col].value_counts(dropna=True)

            if self.min_category_frequency > 1:
                vc = vc[vc >= self.min_category_frequency]

            if self.max_categories is not None:
                vc = vc.head(self.max_categories)

            category_to_index = {
                self.missing_token: 0,
                self.unknown_token: 1,
            }

            for idx, category in enumerate(vc.index, start=2):
                category_to_index[category] = idx

            self.category_vocabs_[col] = CategoryVocab(
                column=col,
                category_to_index=category_to_index,
                missing_index=0,
                unknown_index=1,
            )

        self.category_cardinalities_ = [
            self.category_vocabs_[col].cardinality for col in self.categorical_columns_
        ]

    def _transform_categorical(self, X: pd.DataFrame) -> np.ndarray:
        if not self.categorical_columns_:
            return np.empty((len(X), 0), dtype=np.int64)

        encoded_columns = []

        for col in self.categorical_columns_:
            vocab = self.category_vocabs_[col]
            mapping = vocab.category_to_index

            s = X[col]
            # Cast away pandas Categorical dtype so reserved integer ids can be
            # inserted even when they were not present in the observed categories.
            encoded = s.astype("object").map(mapping)
            encoded = encoded.where(~s.isna(), vocab.missing_index)
            encoded = encoded.fillna(vocab.unknown_index)

            encoded_columns.append(encoded.to_numpy(dtype=np.int64))

        return np.column_stack(encoded_columns).astype(np.int64)

    def get_numeric_feature_names(self) -> list[str]:
        check_is_fitted(self, ["numeric_feature_names_"])
        return list(self.numeric_feature_names_)

    def get_categorical_feature_names(self) -> list[str]:
        check_is_fitted(self, ["categorical_feature_names_"])
        return list(self.categorical_feature_names_)

    def get_category_cardinalities(self) -> list[int]:
        check_is_fitted(self, ["category_cardinalities_"])
        return list(self.category_cardinalities_)


__all__ = [
    "TabularTorchPreprocessor",
]
