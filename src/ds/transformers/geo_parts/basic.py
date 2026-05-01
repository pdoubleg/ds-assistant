"""Target-free basic geographic feature transformers."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .constants import MISSING_GEO_VALUE
from ._utils import (
    _as_dataframe,
    _feature_names_or_raise,
    _make_grid_cell,
    _require_columns,
    _try_make_h3_cells,
    _valid_lat_lon_mask,
    _validate_lat_lon_values,
    _validate_missing_policy,
)


class LatLonMissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """Create numeric missing/valid coordinate indicators.

    Example:
        ``LatLonMissingIndicatorTransformer().fit_transform(df)`` returns
        ``geo_lat_missing``, ``geo_lon_missing``, ``geo_lat_lon_missing``, and
        ``geo_lat_lon_valid``.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "LatLonMissingIndicatorTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        self.feature_names_out_ = [
            f"{self.prefix}_lat_missing",
            f"{self.prefix}_lon_missing",
            f"{self.prefix}_lat_lon_missing",
            f"{self.prefix}_lat_lon_valid",
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        lat = pd.to_numeric(X[self.lat_col], errors="coerce")
        lon = pd.to_numeric(X[self.lon_col], errors="coerce")
        valid = _valid_lat_lon_mask(X, self.lat_col, self.lon_col)

        out = pd.DataFrame(index=X.index)
        out[f"{self.prefix}_lat_missing"] = (lat.isna() | ~lat.between(-90, 90)).astype(
            int
        )
        out[f"{self.prefix}_lon_missing"] = (
            lon.isna() | ~lon.between(-180, 180)
        ).astype(int)
        out[f"{self.prefix}_lat_lon_missing"] = (~valid).astype(int)
        out[f"{self.prefix}_lat_lon_valid"] = valid.astype(int)
        self.feature_names_out_ = list(out.columns)
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)


class LatLonBasicTransformer(BaseEstimator, TransformerMixin):
    """
    Basic raw and transformed latitude/longitude features.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        include_raw: bool = True,
        include_trig: bool = True,
        include_interaction: bool = True,
        missing_policy: Literal["error", "impute", "sentinel"] = "sentinel",
        impute_lat: float = 0.0,
        impute_lon: float = 0.0,
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.include_raw = include_raw
        self.include_trig = include_trig
        self.include_interaction = include_interaction
        self.missing_policy = missing_policy
        self.impute_lat = float(impute_lat)
        self.impute_lon = float(impute_lon)
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "LatLonBasicTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_missing_policy(self.missing_policy)
        if self.missing_policy == "error":
            _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)
        self.feature_names_out_ = self._output_columns()
        return self

    def _output_columns(self) -> list[str]:
        cols: list[str] = []
        if self.include_raw:
            cols.extend([f"{self.prefix}_lat", f"{self.prefix}_lon"])
        if self.include_trig:
            cols.extend(
                [
                    f"{self.prefix}_sin_lat",
                    f"{self.prefix}_cos_lat",
                    f"{self.prefix}_sin_lon",
                    f"{self.prefix}_cos_lon",
                ]
            )
        if self.include_interaction:
            cols.extend(
                [
                    f"{self.prefix}_lat_lon_product",
                    f"{self.prefix}_lat_abs",
                    f"{self.prefix}_lon_abs",
                ]
            )
        return cols

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_missing_policy(self.missing_policy)
        valid = _valid_lat_lon_mask(X, self.lat_col, self.lon_col)
        if self.missing_policy == "error" and not valid.all():
            raise ValueError(
                "Latitude and longitude columns contain missing or invalid values."
            )

        lat = pd.to_numeric(X[self.lat_col], errors="coerce")
        lon = pd.to_numeric(X[self.lon_col], errors="coerce")
        if self.missing_policy == "impute":
            lat = lat.where(valid, self.impute_lat)
            lon = lon.where(valid, self.impute_lon)
        else:
            lat = lat.where(valid, np.nan)
            lon = lon.where(valid, np.nan)

        out = pd.DataFrame(index=X.index)

        if self.include_raw:
            out[f"{self.prefix}_lat"] = lat
            out[f"{self.prefix}_lon"] = lon

        if self.include_trig:
            out[f"{self.prefix}_sin_lat"] = np.sin(np.deg2rad(lat))
            out[f"{self.prefix}_cos_lat"] = np.cos(np.deg2rad(lat))
            out[f"{self.prefix}_sin_lon"] = np.sin(np.deg2rad(lon))
            out[f"{self.prefix}_cos_lon"] = np.cos(np.deg2rad(lon))

        if self.include_interaction:
            out[f"{self.prefix}_lat_lon_product"] = lat * lon
            out[f"{self.prefix}_lat_abs"] = lat.abs()
            out[f"{self.prefix}_lon_abs"] = lon.abs()

        self.feature_names_out_ = list(out.columns)
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)


class RoundedLatLonTransformer(BaseEstimator, TransformerMixin):
    """Rounded latitude/longitude features for coarse spatial splits.

    Args:
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        decimals: Decimal precisions to round coordinates to.
        include_numeric: Whether to emit rounded numeric latitude/longitude.
        include_categorical_cell: Whether to emit rounded coordinate cells.
        missing_policy: How to handle missing or invalid coordinates. ``"error"``
            raises, ``"impute"`` rounds imputed coordinates, and ``"sentinel"``
            emits numeric NaN values plus sentinel categorical cells.
        missing_value: Sentinel value for missing categorical rounded cells.
        impute_lat: Latitude value used when ``missing_policy="impute"``.
        impute_lon: Longitude value used when ``missing_policy="impute"``.
        prefix: Prefix for generated column names.

    Example:
        ``RoundedLatLonTransformer(missing_policy="sentinel").fit_transform(X)``
        returns stable rounded-coordinate columns while preserving invalid
        coordinates as explicit missing geography.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        decimals: Sequence[int] = (0, 1, 2),
        include_numeric: bool = True,
        include_categorical_cell: bool = True,
        missing_policy: Literal["error", "impute", "sentinel"] = "sentinel",
        missing_value: str = MISSING_GEO_VALUE,
        impute_lat: float = 0.0,
        impute_lon: float = 0.0,
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.decimals = tuple(decimals)
        self.include_numeric = include_numeric
        self.include_categorical_cell = include_categorical_cell
        self.missing_policy = missing_policy
        self.missing_value = missing_value
        self.impute_lat = float(impute_lat)
        self.impute_lon = float(impute_lon)
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "RoundedLatLonTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_missing_policy(self.missing_policy)
        if self.missing_policy == "error":
            _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)
        self.feature_names_out_ = self._output_columns()
        return self

    def _output_columns(self) -> list[str]:
        cols: list[str] = []
        for decimal in self.decimals:
            if self.include_numeric:
                cols.extend(
                    [
                        f"{self.prefix}_lat_round_{decimal}",
                        f"{self.prefix}_lon_round_{decimal}",
                    ]
                )
            if self.include_categorical_cell:
                cols.append(f"{self.prefix}_cell_round_{decimal}")
        return cols

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_missing_policy(self.missing_policy)
        valid = _valid_lat_lon_mask(X, self.lat_col, self.lon_col)
        if self.missing_policy == "error" and not valid.all():
            raise ValueError(
                "Latitude and longitude columns contain missing or invalid values."
            )

        out = pd.DataFrame(index=X.index)
        lat = pd.to_numeric(X[self.lat_col], errors="coerce")
        lon = pd.to_numeric(X[self.lon_col], errors="coerce")
        if self.missing_policy == "impute":
            lat = lat.where(valid, self.impute_lat)
            lon = lon.where(valid, self.impute_lon)
        else:
            lat = lat.where(valid, np.nan)
            lon = lon.where(valid, np.nan)

        for decimal in self.decimals:
            rounded_lat = lat.round(decimal)
            rounded_lon = lon.round(decimal)

            if self.include_numeric:
                out[f"{self.prefix}_lat_round_{decimal}"] = rounded_lat
                out[f"{self.prefix}_lon_round_{decimal}"] = rounded_lon

            if self.include_categorical_cell:
                cell = (
                    "round"
                    + str(decimal)
                    + "_"
                    + rounded_lat.astype("string")
                    + "_"
                    + rounded_lon.astype("string")
                )
                out[f"{self.prefix}_cell_round_{decimal}"] = (
                    cell.where(valid, self.missing_value)
                    .fillna(self.missing_value)
                    .astype("category")
                )

        self.feature_names_out_ = list(out.columns)
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)


class GeoCellTransformer(BaseEstimator, TransformerMixin):
    """
    Adds H3 or explicit fallback grid-cell categorical columns.

    Missing or invalid coordinates are emitted as ``missing_value``.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        resolutions: Sequence[int] = (4, 5, 6, 7),
        use_h3: bool = True,
        require_h3: bool = True,
        prefix: str = "geo",
        missing_value: str = MISSING_GEO_VALUE,
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.resolutions = tuple(resolutions)
        self.use_h3 = use_h3
        self.require_h3 = require_h3
        self.prefix = prefix
        self.missing_value = missing_value

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "GeoCellTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        if self.use_h3 and self.require_h3:
            try:
                import h3  # noqa: F401
            except ImportError:
                raise ImportError(
                    "h3 is required for GeoCellTransformer when require_h3=True."
                ) from None
        self.feature_names_out_ = [
            f"{self.prefix}_{'h3' if self.use_h3 else 'grid'}_r{resolution}"
            for resolution in self.resolutions
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        valid = _valid_lat_lon_mask(X, self.lat_col, self.lon_col)

        out = pd.DataFrame(index=X.index)
        lat = pd.to_numeric(X[self.lat_col], errors="coerce")
        lon = pd.to_numeric(X[self.lon_col], errors="coerce")
        valid_lat = lat.where(valid)
        valid_lon = lon.where(valid)

        for resolution in self.resolutions:
            if self.use_h3:
                cell = _try_make_h3_cells(
                    valid_lat, valid_lon, resolution, require_h3=self.require_h3
                )
                col = f"{self.prefix}_h3_r{resolution}"
            else:
                cell = _make_grid_cell(valid_lat, valid_lon, resolution)
                col = f"{self.prefix}_grid_r{resolution}"

            out[col] = (
                cell.astype("string").fillna(self.missing_value).astype("category")
            )

        self.feature_names_out_ = list(out.columns)
        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)
