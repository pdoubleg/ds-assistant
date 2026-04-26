"""Custom transformer for parsing car specification strings."""

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CarSpecParser(BaseEstimator, TransformerMixin):
    """Parse vehicle spec strings into numeric features.

    Examples:
        max_torque = "250Nm@2750rpm"
        max_power = "113.45bhp@4000rpm"

    New columns:
        max_torque_nm
        max_torque_rpm
        max_power_bhp
        max_power_rpm
    """

    def __init__(
        self,
        *,
        torque_column: str = "max_torque",
        power_column: str = "max_power",
        drop_original: bool = True,
    ) -> None:
        self.torque_column = torque_column
        self.power_column = power_column
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CarSpecParser":
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
        """Expand torque and power strings into numeric columns.

        Args:
            X: Input dataframe containing the spec columns.

        Returns:
            Transformed dataframe with parsed numeric features.
        """

        X_out = X.copy()

        torque_values = X_out[self.torque_column].apply(self._parse_torque)
        X_out["max_torque_nm"] = torque_values.apply(lambda value: value[0])
        X_out["max_torque_rpm"] = torque_values.apply(lambda value: value[1])

        power_values = X_out[self.power_column].apply(self._parse_power)
        X_out["max_power_bhp"] = power_values.apply(lambda value: value[0])
        X_out["max_power_rpm"] = power_values.apply(lambda value: value[1])

        if self.drop_original:
            X_out = X_out.drop(columns=[self.torque_column, self.power_column])

        return X_out

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        """Return the output feature names used by sklearn `set_output`.

        Args:
            input_features: Optional explicit input feature names.

        Returns:
            Output feature names after this transformer runs.
        """

        check_is_fitted(self, attributes=["feature_names_in_"])
        if input_features is None:
            features = self.feature_names_in_.tolist()
        else:
            features = list(input_features)
        output_features = list(features)
        parsed_feature_names = [
            "max_torque_nm",
            "max_torque_rpm",
            "max_power_bhp",
            "max_power_rpm",
        ]

        if self.drop_original:
            output_features = [
                feature
                for feature in output_features
                if feature not in {self.torque_column, self.power_column}
            ]

        for feature in parsed_feature_names:
            if feature not in output_features:
                output_features.append(feature)
        return np.asarray(output_features, dtype=object)

    @staticmethod
    def _parse_torque(value: Any) -> tuple[float, float]:
        if pd.isna(value):
            return np.nan, np.nan

        match = re.search(
            r"(?P<torque>[0-9]+(?:\.[0-9]+)?)\s*Nm\s*@\s*(?P<rpm>[0-9]+(?:\.[0-9]+)?)\s*rpm",
            str(value),
            flags=re.IGNORECASE,
        )

        if not match:
            return np.nan, np.nan

        return float(match.group("torque")), float(match.group("rpm"))

    @staticmethod
    def _parse_power(value: Any) -> tuple[float, float]:
        if pd.isna(value):
            return np.nan, np.nan

        match = re.search(
            r"(?P<power>[0-9]+(?:\.[0-9]+)?)\s*bhp\s*@\s*(?P<rpm>[0-9]+(?:\.[0-9]+)?)\s*rpm",
            str(value),
            flags=re.IGNORECASE,
        )

        if not match:
            return np.nan, np.nan

        return float(match.group("power")), float(match.group("rpm"))


__all__ = ["CarSpecParser"]
