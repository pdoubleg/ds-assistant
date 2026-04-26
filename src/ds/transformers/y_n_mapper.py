"""Custom transformer for converting yes/no flags to numeric values."""

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class YesNoMapper(BaseEstimator, TransformerMixin):
    """Convert selected Yes/No string columns to 1/0 numeric columns."""

    def __init__(
        self,
        *,
        columns: list[str],
        yes_value: int = 1,
        no_value: int = 0,
        unknown_value: int = -1,
    ) -> None:
        self.columns = columns
        self.yes_value = yes_value
        self.no_value = no_value
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> "YesNoMapper":
        """Record the fitted feature schema.

        Args:
            X: Training dataframe.
            y: Unused target passed by sklearn.

        Returns:
            The fitted transformer instance.
        """

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map configured yes/no columns to numeric values.

        Args:
            X: Input dataframe.

        Returns:
            Dataframe with mapped flag columns.
        """

        X_out = X.copy()

        for column in self.columns:
            normalized = X_out[column].astype("string").str.lower().str.strip()
            X_out[column] = np.select(
                [
                    normalized.eq("yes"),
                    normalized.eq("no"),
                ],
                [
                    self.yes_value,
                    self.no_value,
                ],
                default=self.unknown_value,
            )

        return X_out

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        """Return unchanged output feature names for sklearn `set_output`.

        Args:
            input_features: Optional explicit input feature names.

        Returns:
            Output feature names after this transformer runs.
        """

        check_is_fitted(self, attributes=["feature_names_in_"])
        if input_features is None:
            features = self.feature_names_in_
        else:
            features = input_features
        return np.asarray(list(features), dtype=object)


__all__ = ["YesNoMapper"]
