"""Geographic feature transformers for direct-mail modeling.

Target-aware transformers in this module must be fit only on training data.

For honest validation:
- Preferred: place target-aware transformers inside the modeling pipeline
  evaluated by cross-validation.
- Acceptable: use a train/validation split where ``fit_transform`` is called
  on train and ``transform`` is called on validation.
- Avoid: precomputing target encodings on the full dataset before doing model
  cross-validation.

Out-of-fold encodings prevent row self-label leakage inside the fitted sample,
but they can still leak outer validation folds when generated before model CV.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import BallTree


EARTH_RADIUS_MILES = 3958.7613
MISSING_GEO_VALUE = "__MISSING__"


DEFAULT_MAJOR_CITIES = pd.DataFrame(
    [
        {
            "city": "New York",
            "state": "NY",
            "lat": 40.7128,
            "lon": -74.0060,
            "population": 8804190,
        },
        {
            "city": "Los Angeles",
            "state": "CA",
            "lat": 34.0522,
            "lon": -118.2437,
            "population": 3898747,
        },
        {
            "city": "Chicago",
            "state": "IL",
            "lat": 41.8781,
            "lon": -87.6298,
            "population": 2746388,
        },
        {
            "city": "Houston",
            "state": "TX",
            "lat": 29.7604,
            "lon": -95.3698,
            "population": 2304580,
        },
        {
            "city": "Phoenix",
            "state": "AZ",
            "lat": 33.4484,
            "lon": -112.0740,
            "population": 1608139,
        },
        {
            "city": "Philadelphia",
            "state": "PA",
            "lat": 39.9526,
            "lon": -75.1652,
            "population": 1603797,
        },
        {
            "city": "San Antonio",
            "state": "TX",
            "lat": 29.4241,
            "lon": -98.4936,
            "population": 1434625,
        },
        {
            "city": "San Diego",
            "state": "CA",
            "lat": 32.7157,
            "lon": -117.1611,
            "population": 1386932,
        },
        {
            "city": "Dallas",
            "state": "TX",
            "lat": 32.7767,
            "lon": -96.7970,
            "population": 1304379,
        },
        {
            "city": "San Jose",
            "state": "CA",
            "lat": 37.3382,
            "lon": -121.8863,
            "population": 1013240,
        },
        {
            "city": "Austin",
            "state": "TX",
            "lat": 30.2672,
            "lon": -97.7431,
            "population": 961855,
        },
        {
            "city": "Jacksonville",
            "state": "FL",
            "lat": 30.3322,
            "lon": -81.6557,
            "population": 949611,
        },
        {
            "city": "Fort Worth",
            "state": "TX",
            "lat": 32.7555,
            "lon": -97.3308,
            "population": 918915,
        },
        {
            "city": "Columbus",
            "state": "OH",
            "lat": 39.9612,
            "lon": -82.9988,
            "population": 905748,
        },
        {
            "city": "Charlotte",
            "state": "NC",
            "lat": 35.2271,
            "lon": -80.8431,
            "population": 874579,
        },
        {
            "city": "San Francisco",
            "state": "CA",
            "lat": 37.7749,
            "lon": -122.4194,
            "population": 873965,
        },
        {
            "city": "Indianapolis",
            "state": "IN",
            "lat": 39.7684,
            "lon": -86.1581,
            "population": 887642,
        },
        {
            "city": "Seattle",
            "state": "WA",
            "lat": 47.6062,
            "lon": -122.3321,
            "population": 737015,
        },
        {
            "city": "Denver",
            "state": "CO",
            "lat": 39.7392,
            "lon": -104.9903,
            "population": 715522,
        },
        {
            "city": "Washington",
            "state": "DC",
            "lat": 38.9072,
            "lon": -77.0369,
            "population": 689545,
        },
        {
            "city": "Boston",
            "state": "MA",
            "lat": 42.3601,
            "lon": -71.0589,
            "population": 675647,
        },
        {
            "city": "El Paso",
            "state": "TX",
            "lat": 31.7619,
            "lon": -106.4850,
            "population": 678815,
        },
        {
            "city": "Nashville",
            "state": "TN",
            "lat": 36.1627,
            "lon": -86.7816,
            "population": 689447,
        },
        {
            "city": "Detroit",
            "state": "MI",
            "lat": 42.3314,
            "lon": -83.0458,
            "population": 639111,
        },
        {
            "city": "Oklahoma City",
            "state": "OK",
            "lat": 35.4676,
            "lon": -97.5164,
            "population": 681054,
        },
        {
            "city": "Portland",
            "state": "OR",
            "lat": 45.5152,
            "lon": -122.6784,
            "population": 652503,
        },
        {
            "city": "Las Vegas",
            "state": "NV",
            "lat": 36.1699,
            "lon": -115.1398,
            "population": 641903,
        },
        {
            "city": "Memphis",
            "state": "TN",
            "lat": 35.1495,
            "lon": -90.0490,
            "population": 633104,
        },
        {
            "city": "Louisville",
            "state": "KY",
            "lat": 38.2527,
            "lon": -85.7585,
            "population": 633045,
        },
        {
            "city": "Baltimore",
            "state": "MD",
            "lat": 39.2904,
            "lon": -76.6122,
            "population": 585708,
        },
        {
            "city": "Milwaukee",
            "state": "WI",
            "lat": 43.0389,
            "lon": -87.9065,
            "population": 577222,
        },
        {
            "city": "Albuquerque",
            "state": "NM",
            "lat": 35.0844,
            "lon": -106.6504,
            "population": 564559,
        },
        {
            "city": "Tucson",
            "state": "AZ",
            "lat": 32.2226,
            "lon": -110.9747,
            "population": 542629,
        },
        {
            "city": "Fresno",
            "state": "CA",
            "lat": 36.7378,
            "lon": -119.7871,
            "population": 542107,
        },
        {
            "city": "Sacramento",
            "state": "CA",
            "lat": 38.5816,
            "lon": -121.4944,
            "population": 524943,
        },
        {
            "city": "Kansas City",
            "state": "MO",
            "lat": 39.0997,
            "lon": -94.5786,
            "population": 508090,
        },
        {
            "city": "Mesa",
            "state": "AZ",
            "lat": 33.4152,
            "lon": -111.8315,
            "population": 504258,
        },
        {
            "city": "Atlanta",
            "state": "GA",
            "lat": 33.7490,
            "lon": -84.3880,
            "population": 498715,
        },
        {
            "city": "Omaha",
            "state": "NE",
            "lat": 41.2565,
            "lon": -95.9345,
            "population": 486051,
        },
        {
            "city": "Colorado Springs",
            "state": "CO",
            "lat": 38.8339,
            "lon": -104.8214,
            "population": 478961,
        },
        {
            "city": "Raleigh",
            "state": "NC",
            "lat": 35.7796,
            "lon": -78.6382,
            "population": 467665,
        },
        {
            "city": "Miami",
            "state": "FL",
            "lat": 25.7617,
            "lon": -80.1918,
            "population": 442241,
        },
        {
            "city": "Virginia Beach",
            "state": "VA",
            "lat": 36.8529,
            "lon": -75.9780,
            "population": 459470,
        },
        {
            "city": "Oakland",
            "state": "CA",
            "lat": 37.8044,
            "lon": -122.2712,
            "population": 440646,
        },
        {
            "city": "Minneapolis",
            "state": "MN",
            "lat": 44.9778,
            "lon": -93.2650,
            "population": 429954,
        },
        {
            "city": "Tulsa",
            "state": "OK",
            "lat": 36.1540,
            "lon": -95.9928,
            "population": 413066,
        },
        {
            "city": "Arlington",
            "state": "TX",
            "lat": 32.7357,
            "lon": -97.1081,
            "population": 394266,
        },
        {
            "city": "Tampa",
            "state": "FL",
            "lat": 27.9506,
            "lon": -82.4572,
            "population": 384959,
        },
        {
            "city": "New Orleans",
            "state": "LA",
            "lat": 29.9511,
            "lon": -90.0715,
            "population": 383997,
        },
        {
            "city": "Wichita",
            "state": "KS",
            "lat": 37.6872,
            "lon": -97.3301,
            "population": 397532,
        },
        {
            "city": "Cleveland",
            "state": "OH",
            "lat": 41.4993,
            "lon": -81.6944,
            "population": 372624,
        },
        {
            "city": "Bakersfield",
            "state": "CA",
            "lat": 35.3733,
            "lon": -119.0187,
            "population": 403455,
        },
        {
            "city": "Aurora",
            "state": "CO",
            "lat": 39.7294,
            "lon": -104.8319,
            "population": 386261,
        },
        {
            "city": "Anaheim",
            "state": "CA",
            "lat": 33.8366,
            "lon": -117.9143,
            "population": 346824,
        },
        {
            "city": "Honolulu",
            "state": "HI",
            "lat": 21.3069,
            "lon": -157.8583,
            "population": 350964,
        },
        {
            "city": "Santa Ana",
            "state": "CA",
            "lat": 33.7455,
            "lon": -117.8677,
            "population": 310227,
        },
        {
            "city": "Riverside",
            "state": "CA",
            "lat": 33.9806,
            "lon": -117.3755,
            "population": 314998,
        },
        {
            "city": "Corpus Christi",
            "state": "TX",
            "lat": 27.8006,
            "lon": -97.3964,
            "population": 317863,
        },
        {
            "city": "Lexington",
            "state": "KY",
            "lat": 38.0406,
            "lon": -84.5037,
            "population": 322570,
        },
        {
            "city": "Henderson",
            "state": "NV",
            "lat": 36.0395,
            "lon": -114.9817,
            "population": 317610,
        },
        {
            "city": "Stockton",
            "state": "CA",
            "lat": 37.9577,
            "lon": -121.2908,
            "population": 320804,
        },
        {
            "city": "Saint Paul",
            "state": "MN",
            "lat": 44.9537,
            "lon": -93.0900,
            "population": 311527,
        },
        {
            "city": "Cincinnati",
            "state": "OH",
            "lat": 39.1031,
            "lon": -84.5120,
            "population": 309317,
        },
        {
            "city": "St. Louis",
            "state": "MO",
            "lat": 38.6270,
            "lon": -90.1994,
            "population": 301578,
        },
        {
            "city": "Pittsburgh",
            "state": "PA",
            "lat": 40.4406,
            "lon": -79.9959,
            "population": 302971,
        },
    ]
)


def _require_columns(X: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in X.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("This transformer expects a pandas DataFrame.")


def _validate_lat_lon_values(
    X: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    allow_missing: bool = True,
) -> None:
    lat = X[lat_col]
    lon = X[lon_col]

    if not allow_missing:
        if lat.isna().any() or lon.isna().any():
            raise ValueError("Latitude and longitude columns contain missing values.")

    lat_non_null = lat.dropna()
    lon_non_null = lon.dropna()

    if not lat_non_null.between(-90, 90).all():
        raise ValueError(f"{lat_col} contains values outside [-90, 90].")

    if not lon_non_null.between(-180, 180).all():
        raise ValueError(f"{lon_col} contains values outside [-180, 180].")


def _valid_lat_lon_mask(
    X: pd.DataFrame,
    lat_col: str,
    lon_col: str,
) -> pd.Series:
    """Return rows with non-missing latitude/longitude inside valid ranges.

    Args:
        X: Input frame containing coordinate columns.
        lat_col: Latitude column name.
        lon_col: Longitude column name.

    Returns:
        Boolean series indexed like ``X``. ``True`` means both coordinates are
        present and inside latitude/longitude bounds.
    """
    _require_columns(X, [lat_col, lon_col])
    lat = pd.to_numeric(X[lat_col], errors="coerce")
    lon = pd.to_numeric(X[lon_col], errors="coerce")
    return lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)


def _validate_missing_policy(missing_policy: str) -> None:
    """Validate a coordinate missingness policy value."""
    if missing_policy not in {"error", "impute", "sentinel"}:
        raise ValueError(
            "missing_policy must be one of: 'error', 'impute', 'sentinel'."
        )


def _feature_names_or_raise(estimator: Any) -> np.ndarray:
    """Return fitted feature names with a consistent sklearn-style error."""
    if not hasattr(estimator, "feature_names_out_"):
        raise RuntimeError(f"{estimator.__class__.__name__} has not been fit.")
    return np.array(estimator.feature_names_out_, dtype=object)


def _validate_binary_target(
    y: pd.Series | np.ndarray,
    index: pd.Index,
    estimator_name: str,
) -> pd.Series:
    """Validate a binary 0/1 target and align it to an input index."""
    y_series = pd.Series(y, index=index).astype(float)
    if y_series.isna().any():
        raise ValueError("Target contains missing values.")
    if not y_series.isin([0.0, 1.0]).all():
        raise ValueError(f"{estimator_name} expects a binary target with values 0/1.")
    return y_series


def _as_key_series(
    series: pd.Series, missing_value: str = MISSING_GEO_VALUE
) -> pd.Series:
    """Cast a categorical key series while preserving missing as an explicit key."""
    return series.astype("string").fillna(missing_value)


def _lat_lon_to_radians(df: pd.DataFrame, lat_col: str, lon_col: str) -> np.ndarray:
    coords = df[[lat_col, lon_col]].astype(float).to_numpy()
    return np.deg2rad(coords)


def _haversine_distance_miles(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_MILES * c


def _make_grid_cell(
    lat: pd.Series,
    lon: pd.Series,
    resolution: int,
) -> pd.Series:
    """
    Pure-python fallback for H3-like cells.

    This is not a true equal-area hex grid. It is a simple rounded lat/lon grid
    with resolution-specific precision. It is useful when you want deterministic
    geographic cells without installing h3.

    Approximate behavior:
    - resolution 3: very coarse
    - resolution 4: coarse
    - resolution 5: medium
    - resolution 6: fine
    - resolution 7: very fine
    """
    precision_by_resolution = {
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 4,
        10: 4,
    }
    precision = precision_by_resolution.get(resolution, max(0, min(5, resolution - 3)))

    lat_rounded = lat.round(precision)
    lon_rounded = lon.round(precision)

    return (
        "grid_r"
        + str(resolution)
        + "_"
        + lat_rounded.astype("string")
        + "_"
        + lon_rounded.astype("string")
    )


def _try_make_h3_cells(
    lat: pd.Series,
    lon: pd.Series,
    resolution: int,
    require_h3: bool = True,
) -> pd.Series:
    try:
        import h3  # type: ignore
    except ImportError:
        if require_h3:
            raise ImportError(
                "h3 is required for GeoCellTransformer when require_h3=True."
            ) from None
        return _make_grid_cell(lat=lat, lon=lon, resolution=resolution)

    def to_cell(row: tuple[float, float]) -> str | pd.NA:
        row_lat, row_lon = row
        if pd.isna(row_lat) or pd.isna(row_lon):
            return pd.NA

        if hasattr(h3, "latlng_to_cell"):
            return h3.latlng_to_cell(float(row_lat), float(row_lon), resolution)

        if hasattr(h3, "geo_to_h3"):
            return h3.geo_to_h3(float(row_lat), float(row_lon), resolution)

        raise RuntimeError(
            "Installed h3 package does not expose a recognized cell function."
        )

    values = [to_cell((la, lo)) for la, lo in zip(lat, lon)]
    return pd.Series(values, index=lat.index, dtype="string")


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
    """
    Rounded latitude/longitude features for coarse spatial splits.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        decimals: Sequence[int] = (0, 1, 2),
        include_numeric: bool = True,
        include_categorical_cell: bool = True,
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.decimals = tuple(decimals)
        self.include_numeric = include_numeric
        self.include_categorical_cell = include_categorical_cell
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "RoundedLatLonTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col)
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
        _validate_lat_lon_values(X, self.lat_col, self.lon_col)

        out = pd.DataFrame(index=X.index)
        lat = X[self.lat_col].astype(float)
        lon = X[self.lon_col].astype(float)

        for decimal in self.decimals:
            rounded_lat = lat.round(decimal)
            rounded_lon = lon.round(decimal)

            if self.include_numeric:
                out[f"{self.prefix}_lat_round_{decimal}"] = rounded_lat
                out[f"{self.prefix}_lon_round_{decimal}"] = rounded_lon

            if self.include_categorical_cell:
                out[f"{self.prefix}_cell_round_{decimal}"] = (
                    "round"
                    + str(decimal)
                    + "_"
                    + rounded_lat.astype("string")
                    + "_"
                    + rounded_lon.astype("string")
                ).astype("category")

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


class NearestCityDistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Distance to nearest city, optionally by population threshold.

    Example output:
    - geo_nearest_city_distance_miles
    - geo_nearest_city_population
    - geo_nearest_city_name
    - geo_nearest_city_state
    - geo_nearest_city_pop_gt_100000_distance_miles
    - geo_nearest_city_pop_gt_250000_distance_miles
    - geo_nearest_city_pop_gt_1000000_distance_miles
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        city_df: pd.DataFrame | None = None,
        city_lat_col: str = "lat",
        city_lon_col: str = "lon",
        city_name_col: str = "city",
        city_state_col: str = "state",
        city_population_col: str = "population",
        population_thresholds: Sequence[int] = (100_000, 250_000, 1_000_000),
        include_nearest_city_name: bool = True,
        include_nearest_city_population: bool = True,
        missing_policy: Literal["error", "impute", "sentinel"] = "sentinel",
        impute_lat: float = 0.0,
        impute_lon: float = 0.0,
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.city_df = city_df
        self.city_lat_col = city_lat_col
        self.city_lon_col = city_lon_col
        self.city_name_col = city_name_col
        self.city_state_col = city_state_col
        self.city_population_col = city_population_col
        self.population_thresholds = tuple(population_thresholds)
        self.include_nearest_city_name = include_nearest_city_name
        self.include_nearest_city_population = include_nearest_city_population
        self.missing_policy = missing_policy
        self.impute_lat = float(impute_lat)
        self.impute_lon = float(impute_lon)
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "NearestCityDistanceTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_missing_policy(self.missing_policy)
        if self.missing_policy == "error":
            _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)

        cities = (
            DEFAULT_MAJOR_CITIES.copy() if self.city_df is None else self.city_df.copy()
        )
        required = [
            self.city_lat_col,
            self.city_lon_col,
            self.city_name_col,
            self.city_state_col,
            self.city_population_col,
        ]
        _require_columns(cities, required)
        _validate_lat_lon_values(
            cities, self.city_lat_col, self.city_lon_col, allow_missing=False
        )

        cities = cities.dropna(
            subset=[self.city_lat_col, self.city_lon_col]
        ).reset_index(drop=True)
        if cities.empty:
            raise ValueError("city_df has no valid city rows.")

        self.city_df_ = cities
        self.city_coords_rad_ = np.deg2rad(
            cities[[self.city_lat_col, self.city_lon_col]].astype(float).to_numpy()
        )
        self.city_tree_ = BallTree(self.city_coords_rad_, metric="haversine")

        self.threshold_trees_: dict[int, tuple[pd.DataFrame, BallTree]] = {}
        for threshold in self.population_thresholds:
            subset = cities[
                cities[self.city_population_col].astype(float) >= threshold
            ].reset_index(drop=True)
            if subset.empty:
                warnings.warn(
                    f"No cities found for population threshold >= {threshold}."
                )
                continue

            subset_coords_rad = np.deg2rad(
                subset[[self.city_lat_col, self.city_lon_col]].astype(float).to_numpy()
            )
            self.threshold_trees_[threshold] = (
                subset,
                BallTree(subset_coords_rad, metric="haversine"),
            )

        self.feature_names_out_ = self._output_columns()
        return self

    def _output_columns(self) -> list[str]:
        cols = [f"{self.prefix}_nearest_city_distance_miles"]
        if self.include_nearest_city_population:
            cols.append(f"{self.prefix}_nearest_city_population")
        if self.include_nearest_city_name:
            cols.extend(
                [
                    f"{self.prefix}_nearest_city_name",
                    f"{self.prefix}_nearest_city_state",
                ]
            )
        for threshold in self.population_thresholds:
            cols.append(f"{self.prefix}_nearest_city_pop_gt_{threshold}_distance_miles")
            if self.include_nearest_city_name:
                cols.extend(
                    [
                        f"{self.prefix}_nearest_city_pop_gt_{threshold}_name",
                        f"{self.prefix}_nearest_city_pop_gt_{threshold}_state",
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

        out = pd.DataFrame(index=X.index)
        out[f"{self.prefix}_nearest_city_distance_miles"] = np.nan
        if self.include_nearest_city_population:
            out[f"{self.prefix}_nearest_city_population"] = np.nan
        if self.include_nearest_city_name:
            out[f"{self.prefix}_nearest_city_name"] = MISSING_GEO_VALUE
            out[f"{self.prefix}_nearest_city_state"] = MISSING_GEO_VALUE
        for threshold in self.population_thresholds:
            out[f"{self.prefix}_nearest_city_pop_gt_{threshold}_distance_miles"] = (
                np.nan
            )
            if self.include_nearest_city_name:
                out[f"{self.prefix}_nearest_city_pop_gt_{threshold}_name"] = (
                    MISSING_GEO_VALUE
                )
                out[f"{self.prefix}_nearest_city_pop_gt_{threshold}_state"] = (
                    MISSING_GEO_VALUE
                )

        query = X[[self.lat_col, self.lon_col]].copy()
        if self.missing_policy == "impute":
            query[self.lat_col] = pd.to_numeric(
                query[self.lat_col], errors="coerce"
            ).where(valid, self.impute_lat)
            query[self.lon_col] = pd.to_numeric(
                query[self.lon_col], errors="coerce"
            ).where(valid, self.impute_lon)
            query_index = X.index
        else:
            query = query.loc[valid]
            query_index = query.index

        if query.empty:
            for col in out.columns:
                if out[col].dtype == object:
                    out[col] = out[col].astype("category")
            self.feature_names_out_ = list(out.columns)
            return out

        coords_rad = _lat_lon_to_radians(query, self.lat_col, self.lon_col)

        dist_rad, ind = self.city_tree_.query(coords_rad, k=1)
        dist_miles = dist_rad[:, 0] * EARTH_RADIUS_MILES
        nearest_idx = ind[:, 0]
        nearest_cities = self.city_df_.iloc[nearest_idx].reset_index(drop=True)

        out.loc[query_index, f"{self.prefix}_nearest_city_distance_miles"] = dist_miles

        if self.include_nearest_city_population:
            out.loc[query_index, f"{self.prefix}_nearest_city_population"] = (
                nearest_cities[self.city_population_col].astype(float).to_numpy()
            )

        if self.include_nearest_city_name:
            out.loc[query_index, f"{self.prefix}_nearest_city_name"] = (
                nearest_cities[self.city_name_col].astype(str).to_numpy()
            )
            out.loc[query_index, f"{self.prefix}_nearest_city_state"] = (
                nearest_cities[self.city_state_col].astype(str).to_numpy()
            )

        for threshold, (subset, tree) in self.threshold_trees_.items():
            threshold_dist_rad, threshold_ind = tree.query(coords_rad, k=1)
            threshold_dist_miles = threshold_dist_rad[:, 0] * EARTH_RADIUS_MILES
            out.loc[
                query_index,
                f"{self.prefix}_nearest_city_pop_gt_{threshold}_distance_miles",
            ] = threshold_dist_miles

            if self.include_nearest_city_name:
                threshold_nearest = subset.iloc[threshold_ind[:, 0]].reset_index(
                    drop=True
                )
                out.loc[
                    query_index, f"{self.prefix}_nearest_city_pop_gt_{threshold}_name"
                ] = threshold_nearest[self.city_name_col].astype(str).to_numpy()
                out.loc[
                    query_index, f"{self.prefix}_nearest_city_pop_gt_{threshold}_state"
                ] = threshold_nearest[self.city_state_col].astype(str).to_numpy()

        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype("category")

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


class CustomerDensityTransformer(BaseEstimator, TransformerMixin):
    """
    Counts training/reference customers within radius bands.

    This is target-free and safe for Phase 1.

    By default, transform(X_train) will include the row itself in counts when
    called on the same records used in fit. You can subtract self matches with
    exclude_self=True. This is usually useful when X passed to transform is the
    same training data.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        radii_miles: Sequence[float] = (10.0, 25.0, 50.0),
        exclude_self: bool = True,
        add_log_density: bool = True,
        missing_policy: Literal["error", "impute", "sentinel"] = "sentinel",
        impute_lat: float = 0.0,
        impute_lon: float = 0.0,
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.radii_miles = tuple(float(r) for r in radii_miles)
        self.exclude_self = exclude_self
        self.add_log_density = add_log_density
        self.missing_policy = missing_policy
        self.impute_lat = float(impute_lat)
        self.impute_lon = float(impute_lon)
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "CustomerDensityTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_missing_policy(self.missing_policy)
        valid = _valid_lat_lon_mask(X, self.lat_col, self.lon_col)
        if self.missing_policy == "error" and not valid.all():
            raise ValueError(
                "Latitude and longitude columns contain missing or invalid values."
            )

        self.fit_index_ = X.index.copy()
        if self.missing_policy == "impute":
            ref = X[[self.lat_col, self.lon_col]].copy()
            ref[self.lat_col] = pd.to_numeric(ref[self.lat_col], errors="coerce").where(
                valid, self.impute_lat
            )
            ref[self.lon_col] = pd.to_numeric(ref[self.lon_col], errors="coerce").where(
                valid, self.impute_lon
            )
        else:
            ref = X.loc[valid, [self.lat_col, self.lon_col]]
        self.reference_X_ = ref
        self.coords_rad_ = (
            _lat_lon_to_radians(ref, self.lat_col, self.lon_col)
            if len(ref)
            else np.empty((0, 2), dtype=float)
        )
        self.tree_ = (
            BallTree(self.coords_rad_, metric="haversine") if len(ref) else None
        )
        self.feature_names_out_ = self._output_columns()
        return self

    def _output_columns(self) -> list[str]:
        cols: list[str] = []
        for radius in self.radii_miles:
            count_col = f"{self.prefix}_customer_count_within_{int(radius)}mi"
            density_col = (
                f"{self.prefix}_customer_density_per_sqmi_within_{int(radius)}mi"
            )
            cols.extend([count_col, density_col])
            if self.add_log_density:
                cols.extend([f"{count_col}_log1p", f"{density_col}_log1p"])
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

        same_index = X.index.equals(self.fit_index_)

        out = pd.DataFrame(index=X.index)
        for col in self._output_columns():
            out[col] = 0.0

        if self.tree_ is None:
            self.feature_names_out_ = list(out.columns)
            return out

        query = X[[self.lat_col, self.lon_col]].copy()
        if self.missing_policy == "impute":
            query[self.lat_col] = pd.to_numeric(
                query[self.lat_col], errors="coerce"
            ).where(valid, self.impute_lat)
            query[self.lon_col] = pd.to_numeric(
                query[self.lon_col], errors="coerce"
            ).where(valid, self.impute_lon)
            query_index = X.index
        else:
            query = query.loc[valid]
            query_index = query.index

        if query.empty:
            self.feature_names_out_ = list(out.columns)
            return out

        coords_rad = _lat_lon_to_radians(query, self.lat_col, self.lon_col)

        for radius in self.radii_miles:
            radius_rad = radius / EARTH_RADIUS_MILES
            counts = self.tree_.query_radius(
                coords_rad, r=radius_rad, count_only=True
            ).astype(float)

            if self.exclude_self and same_index:
                counts = np.maximum(counts - 1.0, 0.0)

            count_col = f"{self.prefix}_customer_count_within_{int(radius)}mi"
            out.loc[query_index, count_col] = counts

            area = math.pi * radius**2
            density_col = (
                f"{self.prefix}_customer_density_per_sqmi_within_{int(radius)}mi"
            )
            out.loc[query_index, density_col] = counts / area

            if self.add_log_density:
                out.loc[query_index, f"{count_col}_log1p"] = np.log1p(counts)
                out.loc[query_index, f"{density_col}_log1p"] = np.log1p(counts / area)

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


class GeoCellCountTransformer(BaseEstimator, TransformerMixin):
    """
    Count observations in each geo cell. Useful as a simple density proxy.

    This transformer expects precomputed cell columns so H3 generation happens
    once upstream.
    """

    def __init__(
        self,
        cell_cols: Sequence[str],
        unseen_count_value: float = 0.0,
        add_log_count: bool = True,
        add_frequency: bool = True,
        prefix: str = "geo",
    ):
        self.cell_cols = tuple(cell_cols)
        self.unseen_count_value = float(unseen_count_value)
        self.add_log_count = add_log_count
        self.add_frequency = add_frequency
        self.prefix = prefix

    def _make_cells(self, X: pd.DataFrame) -> pd.DataFrame:
        _require_columns(X, list(self.cell_cols))
        return X[list(self.cell_cols)].apply(_as_key_series)

    def _output_columns(self) -> list[str]:
        cols: list[str] = []
        for col in self.cell_cols:
            cols.append(f"{col}_train_count")
            if self.add_log_count:
                cols.append(f"{col}_train_count_log1p")
            if self.add_frequency:
                cols.append(f"{col}_train_frequency")
        return cols

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "GeoCellCountTransformer":
        X = _as_dataframe(X)
        cells = self._make_cells(X)
        self.cell_cols_ = list(cells.columns)
        self.global_count_ = float(len(X))

        self.count_maps_: dict[str, pd.Series] = {}
        for col in self.cell_cols_:
            self.count_maps_[col] = cells[col].value_counts(dropna=False)

        self.feature_names_out_ = self._output_columns()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        cells = self._make_cells(X)
        out = pd.DataFrame(index=X.index)

        for col in self.cell_cols_:
            counts = (
                cells[col]
                .map(self.count_maps_[col])
                .astype(float)
                .fillna(self.unseen_count_value)
            )
            out[f"{col}_train_count"] = counts
            if self.add_log_count:
                out[f"{col}_train_count_log1p"] = np.log1p(counts)
            if self.add_frequency:
                frequency = counts / self.global_count_
                out[f"{col}_train_frequency"] = frequency.where(counts > 0, 0.0)

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


class OOFGeoTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed out-of-fold target encoding for geo cell or area columns.

    This is the main Phase 2 transformer.

    During fit_transform:
    - creates OOF encodings for training rows
    - also stores full-data encodings for future transform calls

    During transform:
    - applies mappings learned from the full training data
    - unseen categories receive the global target rate

    Smoothing formula:
        encoded = (sum_y + alpha * global_mean) / (count + alpha)
    """

    def __init__(
        self,
        cols: Sequence[str] | None = None,
        lat_col: str = "lat",
        lon_col: str = "lon",
        create_geo_cells: bool = False,
        resolutions: Sequence[int] = (5, 6, 7),
        use_h3: bool = True,
        require_h3: bool = True,
        alpha: float = 50.0,
        min_samples_leaf: int = 1,
        handle_unknown: Literal["global_mean", "nan"] = "global_mean",
        n_splits: int = 5,
        random_state: int = 42,
        stratified: bool = True,
        add_count_features: bool = True,
        add_count_log1p_features: bool = True,
        add_reliability_features: bool = True,
        add_logit_features: bool = True,
        output_suffix: str = "oof",
        prefix: str = "te",
    ):
        self.cols = None if cols is None else tuple(cols)
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.create_geo_cells = create_geo_cells
        self.resolutions = tuple(resolutions)
        self.use_h3 = use_h3
        self.require_h3 = require_h3
        self.alpha = float(alpha)
        self.min_samples_leaf = int(min_samples_leaf)
        self.handle_unknown = handle_unknown
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.stratified = stratified
        self.add_count_features = add_count_features
        self.add_count_log1p_features = add_count_log1p_features
        self.add_reliability_features = add_reliability_features
        self.add_logit_features = add_logit_features
        self.output_suffix = output_suffix
        self.prefix = prefix

    def _build_encoding_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.cols is not None:
            _require_columns(X, list(self.cols))
            parts.append(X[list(self.cols)].apply(_as_key_series))

        if self.create_geo_cells:
            celler = GeoCellTransformer(
                lat_col=self.lat_col,
                lon_col=self.lon_col,
                resolutions=self.resolutions,
                use_h3=self.use_h3,
                require_h3=self.require_h3,
                prefix="geo",
            )
            parts.append(celler.fit_transform(X).apply(_as_key_series))

        if not parts:
            raise ValueError("No columns supplied and create_geo_cells=False.")

        return pd.concat(parts, axis=1)

    def _unknown_fill_value(self) -> float:
        if self.handle_unknown == "global_mean":
            return self.global_mean_
        if self.handle_unknown == "nan":
            return np.nan
        raise ValueError("handle_unknown must be one of: 'global_mean', 'nan'.")

    def _output_columns(self) -> list[str]:
        cols: list[str] = []
        for col in self.key_cols_:
            cols.append(f"{self.prefix}_{col}_rate_{self.output_suffix}")
            if self.add_count_features:
                cols.append(f"{self.prefix}_{col}_count_{self.output_suffix}")
            if self.add_count_log1p_features:
                cols.append(f"{self.prefix}_{col}_count_log1p_{self.output_suffix}")
            if self.add_reliability_features:
                cols.append(f"{self.prefix}_{col}_reliability_{self.output_suffix}")
            if self.add_logit_features:
                cols.append(f"{self.prefix}_{col}_logit_{self.output_suffix}")
        return cols

    def _fit_maps_for_frame(
        self,
        keys: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        maps: dict[str, pd.DataFrame] = {}

        for col in keys.columns:
            stats = (
                pd.DataFrame(
                    {"key": _as_key_series(keys[col]), "y": y.astype(float).to_numpy()}
                )
                .groupby("key", dropna=False)["y"]
                .agg(["sum", "count"])
            )
            stats["encoded"] = (stats["sum"] + self.alpha * self.global_mean_) / (
                stats["count"] + self.alpha
            )
            stats.loc[stats["count"] < self.min_samples_leaf, "encoded"] = (
                self.global_mean_
            )
            stats["reliability"] = stats["count"] / (stats["count"] + self.alpha)
            stats["logit"] = np.log(
                np.clip(stats["encoded"], 1e-6, 1 - 1e-6)
                / np.clip(1 - stats["encoded"], 1e-6, 1)
            )
            maps[col] = stats

        return maps

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "OOFGeoTargetEncoder":
        X = _as_dataframe(X)
        if y is None:
            raise ValueError("OOFGeoTargetEncoder requires y during fit.")

        y_series = _validate_binary_target(y, X.index, "OOFGeoTargetEncoder")

        keys = self._build_encoding_frame(X)
        self.key_cols_ = list(keys.columns)
        self.global_mean_ = float(y_series.mean())
        self.full_maps_ = self._fit_maps_for_frame(keys, y_series)
        self.feature_names_out_ = self._output_columns()
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        X = _as_dataframe(X)
        if y is None:
            raise ValueError("OOFGeoTargetEncoder requires y during fit_transform.")

        y_series = _validate_binary_target(y, X.index, "OOFGeoTargetEncoder")

        keys = self._build_encoding_frame(X)
        self.key_cols_ = list(keys.columns)
        self.global_mean_ = float(y_series.mean())

        out = pd.DataFrame(index=X.index)
        self.feature_names_out_ = self._output_columns()

        for col in self.feature_names_out_:
            out[col] = np.nan

        if self.stratified:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X, y_series)
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X)

        for train_pos, valid_pos in split_iter:
            train_index = X.index[train_pos]
            valid_index = X.index[valid_pos]

            train_keys = keys.loc[train_index]
            valid_keys = keys.loc[valid_index]
            train_y = y_series.loc[train_index]

            fold_maps = self._fit_maps_for_frame(train_keys, train_y)

            for col in self.key_cols_:
                stats = fold_maps[col]
                encoded = (
                    valid_keys[col]
                    .map(stats["encoded"])
                    .astype(float)
                    .fillna(self._unknown_fill_value())
                )
                out.loc[
                    valid_index, f"{self.prefix}_{col}_rate_{self.output_suffix}"
                ] = encoded.to_numpy()

                if self.add_count_features:
                    counts = (
                        valid_keys[col].map(stats["count"]).astype(float).fillna(0.0)
                    )
                    out.loc[
                        valid_index, f"{self.prefix}_{col}_count_{self.output_suffix}"
                    ] = counts.to_numpy()
                if self.add_count_log1p_features:
                    counts = (
                        valid_keys[col].map(stats["count"]).astype(float).fillna(0.0)
                    )
                    out.loc[
                        valid_index,
                        f"{self.prefix}_{col}_count_log1p_{self.output_suffix}",
                    ] = np.log1p(counts).to_numpy()
                if self.add_reliability_features:
                    reliability = (
                        valid_keys[col]
                        .map(stats["reliability"])
                        .astype(float)
                        .fillna(0.0)
                    )
                    out.loc[
                        valid_index,
                        f"{self.prefix}_{col}_reliability_{self.output_suffix}",
                    ] = reliability.to_numpy()

                if self.add_logit_features:
                    logits = (
                        valid_keys[col]
                        .map(stats["logit"])
                        .astype(float)
                        .fillna(
                            math.log(
                                np.clip(self.global_mean_, 1e-6, 1 - 1e-6)
                                / np.clip(1 - self.global_mean_, 1e-6, 1)
                            )
                        )
                    )
                    out.loc[
                        valid_index, f"{self.prefix}_{col}_logit_{self.output_suffix}"
                    ] = logits.to_numpy()

        self.full_maps_ = self._fit_maps_for_frame(keys, y_series)

        for col in out.columns:
            out[col] = out[col].astype(float)

        self.feature_names_out_ = list(out.columns)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        if not hasattr(self, "full_maps_"):
            raise RuntimeError("OOFGeoTargetEncoder has not been fit.")

        keys = self._build_encoding_frame(X)
        out = pd.DataFrame(index=X.index)

        global_logit = math.log(
            np.clip(self.global_mean_, 1e-6, 1 - 1e-6)
            / np.clip(1 - self.global_mean_, 1e-6, 1)
        )

        for col in self.key_cols_:
            stats = self.full_maps_[col]
            encoded = (
                keys[col]
                .map(stats["encoded"])
                .astype(float)
                .fillna(self._unknown_fill_value())
            )
            out[f"{self.prefix}_{col}_rate_{self.output_suffix}"] = encoded

            if self.add_count_features:
                counts = keys[col].map(stats["count"]).astype(float).fillna(0.0)
                out[f"{self.prefix}_{col}_count_{self.output_suffix}"] = counts
            if self.add_count_log1p_features:
                counts = keys[col].map(stats["count"]).astype(float).fillna(0.0)
                out[f"{self.prefix}_{col}_count_log1p_{self.output_suffix}"] = np.log1p(
                    counts
                )
            if self.add_reliability_features:
                out[f"{self.prefix}_{col}_reliability_{self.output_suffix}"] = (
                    keys[col].map(stats["reliability"]).astype(float).fillna(0.0)
                )

            if self.add_logit_features:
                logits = (
                    keys[col].map(stats["logit"]).astype(float).fillna(global_logit)
                )
                out[f"{self.prefix}_{col}_logit_{self.output_suffix}"] = logits

        self.feature_names_out_ = list(out.columns)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)


class LocalVsParentTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target-aware difference between a local geographic rate and a broader parent rate.

    Example:
    - local_col = h3_res_7
    - parent_col = h3_res_5
    - output = smoothed local bind rate - smoothed parent bind rate

    This can be useful because it tells the model whether the local pocket
    over- or under-performs its broader area.
    """

    def __init__(
        self,
        local_col: str,
        parent_col: str,
        alpha_local: float = 50.0,
        alpha_parent: float = 100.0,
        n_splits: int = 5,
        random_state: int = 42,
        stratified: bool = True,
        add_component_rates: bool = True,
        add_count_features: bool = True,
        add_reliability_features: bool = True,
        add_ratio_feature: bool = True,
        prefix: str = "te",
    ):
        self.local_col = local_col
        self.parent_col = parent_col
        self.alpha_local = float(alpha_local)
        self.alpha_parent = float(alpha_parent)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.stratified = stratified
        self.add_component_rates = add_component_rates
        self.add_count_features = add_count_features
        self.add_reliability_features = add_reliability_features
        self.add_ratio_feature = add_ratio_feature
        self.prefix = prefix

    def _fit_single_map(
        self,
        keys: pd.Series,
        y: pd.Series,
        alpha: float,
        global_mean: float,
    ) -> pd.DataFrame:
        stats = (
            pd.DataFrame({"key": _as_key_series(keys), "y": y.astype(float).to_numpy()})
            .groupby("key", dropna=False)["y"]
            .agg(["sum", "count"])
        )
        stats["rate"] = (stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)
        stats["reliability"] = stats["count"] / (stats["count"] + alpha)
        return stats

    def _output_columns(self) -> list[str]:
        base = f"{self.prefix}_{self.local_col}_vs_{self.parent_col}"
        cols: list[str] = []
        if self.add_component_rates:
            cols.extend([f"{base}_local_rate_oof", f"{base}_parent_rate_oof"])
        if self.add_count_features:
            cols.extend(
                [
                    f"{base}_local_count_oof",
                    f"{base}_parent_count_oof",
                    f"{base}_local_count_log1p_oof",
                    f"{base}_parent_count_log1p_oof",
                ]
            )
        if self.add_reliability_features:
            cols.extend(
                [f"{base}_local_reliability_oof", f"{base}_parent_reliability_oof"]
            )
        cols.append(f"{base}_local_minus_parent_rate_oof")
        if self.add_ratio_feature:
            cols.append(f"{base}_local_div_parent_rate_oof")
        return cols

    def _transform_from_maps(
        self,
        X: pd.DataFrame,
        local_map: pd.DataFrame,
        parent_map: pd.DataFrame,
    ) -> pd.DataFrame:
        base = f"{self.prefix}_{self.local_col}_vs_{self.parent_col}"
        local_keys = _as_key_series(X[self.local_col])
        parent_keys = _as_key_series(X[self.parent_col])

        local_rate = (
            local_keys.map(local_map["rate"]).astype(float).fillna(self.global_mean_)
        )
        parent_rate = (
            parent_keys.map(parent_map["rate"]).astype(float).fillna(self.global_mean_)
        )
        local_count = local_keys.map(local_map["count"]).astype(float).fillna(0.0)
        parent_count = parent_keys.map(parent_map["count"]).astype(float).fillna(0.0)
        local_reliability = (
            local_keys.map(local_map["reliability"]).astype(float).fillna(0.0)
        )
        parent_reliability = (
            parent_keys.map(parent_map["reliability"]).astype(float).fillna(0.0)
        )

        out = pd.DataFrame(index=X.index)
        if self.add_component_rates:
            out[f"{base}_local_rate_oof"] = local_rate
            out[f"{base}_parent_rate_oof"] = parent_rate
        if self.add_count_features:
            out[f"{base}_local_count_oof"] = local_count
            out[f"{base}_parent_count_oof"] = parent_count
            out[f"{base}_local_count_log1p_oof"] = np.log1p(local_count)
            out[f"{base}_parent_count_log1p_oof"] = np.log1p(parent_count)
        if self.add_reliability_features:
            out[f"{base}_local_reliability_oof"] = local_reliability
            out[f"{base}_parent_reliability_oof"] = parent_reliability
        out[f"{base}_local_minus_parent_rate_oof"] = local_rate - parent_rate
        if self.add_ratio_feature:
            out[f"{base}_local_div_parent_rate_oof"] = local_rate / np.maximum(
                parent_rate, 1e-6
            )
        return out.astype(float)

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "LocalVsParentTargetEncoder":
        X = _as_dataframe(X)
        _require_columns(X, [self.local_col, self.parent_col])

        if y is None:
            raise ValueError("LocalVsParentTargetEncoder requires y during fit.")

        y_series = _validate_binary_target(y, X.index, "LocalVsParentTargetEncoder")
        self.global_mean_ = float(y_series.mean())
        self.local_map_ = self._fit_single_map(
            X[self.local_col], y_series, self.alpha_local, self.global_mean_
        )
        self.parent_map_ = self._fit_single_map(
            X[self.parent_col], y_series, self.alpha_parent, self.global_mean_
        )
        self.feature_names_out_ = self._output_columns()
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.local_col, self.parent_col])

        if y is None:
            raise ValueError(
                "LocalVsParentTargetEncoder requires y during fit_transform."
            )

        y_series = _validate_binary_target(y, X.index, "LocalVsParentTargetEncoder")
        self.global_mean_ = float(y_series.mean())

        out = pd.DataFrame(index=X.index)
        self.feature_names_out_ = self._output_columns()
        for col in self.feature_names_out_:
            out[col] = np.nan

        if self.stratified:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X, y_series)
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X)

        for train_pos, valid_pos in split_iter:
            train_index = X.index[train_pos]
            valid_index = X.index[valid_pos]

            train_y = y_series.loc[train_index]

            local_map = self._fit_single_map(
                X.loc[train_index, self.local_col],
                train_y,
                self.alpha_local,
                self.global_mean_,
            )
            parent_map = self._fit_single_map(
                X.loc[train_index, self.parent_col],
                train_y,
                self.alpha_parent,
                self.global_mean_,
            )

            fold_out = self._transform_from_maps(
                X.loc[valid_index], local_map=local_map, parent_map=parent_map
            )
            out.loc[valid_index, fold_out.columns] = fold_out

        self.local_map_ = self._fit_single_map(
            X[self.local_col], y_series, self.alpha_local, self.global_mean_
        )
        self.parent_map_ = self._fit_single_map(
            X[self.parent_col], y_series, self.alpha_parent, self.global_mean_
        )

        return out.astype(float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.local_col, self.parent_col])

        out = self._transform_from_maps(
            X, local_map=self.local_map_, parent_map=self.parent_map_
        )
        self.feature_names_out_ = list(out.columns)
        return out.astype(float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)


class OOFCellNeighborhoodTargetRateTransformer(BaseEstimator, TransformerMixin):
    """Out-of-fold target-rate features using aggregated nearby geo cells.

    The transformer groups training rows by ``cell_col`` and builds a BallTree
    over cell centroids, which is much cheaper than row-level radius queries for
    large direct-mail files.

    Example:
        ``OOFCellNeighborhoodTargetRateTransformer(cell_col="geo_h3_r7")`` adds
        smoothed nearby bind-rate, count, exposure, log-exposure, and
        reliability features for each configured radius.
    """

    def __init__(
        self,
        cell_col: str = "geo_h3_r7",
        lat_col: str = "lat",
        lon_col: str = "lon",
        radii_miles: Sequence[float] = (25.0, 50.0),
        alpha: float = 50.0,
        n_splits: int = 5,
        random_state: int = 42,
        stratified: bool = True,
        prefix: str = "geo",
    ):
        self.cell_col = cell_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.radii_miles = tuple(float(r) for r in radii_miles)
        self.alpha = float(alpha)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.stratified = stratified
        self.prefix = prefix

    def _output_columns(self) -> list[str]:
        cols: list[str] = []
        for radius in self.radii_miles:
            radius_label = int(radius)
            cols.extend(
                [
                    f"{self.prefix}_cell_neighborhood_bind_rate_{radius_label}mi_oof",
                    f"{self.prefix}_cell_neighborhood_bind_count_{radius_label}mi_oof",
                    f"{self.prefix}_cell_neighborhood_exposure_count_{radius_label}mi_oof",
                    f"{self.prefix}_cell_neighborhood_exposure_count_{radius_label}mi_log1p_oof",
                    f"{self.prefix}_cell_neighborhood_reliability_{radius_label}mi_oof",
                ]
            )
        return cols

    def _build_cell_index(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        valid = _valid_lat_lon_mask(X, self.lat_col, self.lon_col)
        frame = pd.DataFrame(
            {
                "cell": _as_key_series(X.loc[valid, self.cell_col]),
                "lat": pd.to_numeric(X.loc[valid, self.lat_col], errors="coerce"),
                "lon": pd.to_numeric(X.loc[valid, self.lon_col], errors="coerce"),
                "y": y.loc[valid].astype(float),
            }
        )
        if frame.empty:
            return {
                "cell_stats": pd.DataFrame(),
                "tree": None,
                "binds": np.array([], dtype=float),
                "exposures": np.array([], dtype=float),
            }

        cell_stats = (
            frame.groupby("cell", dropna=False)
            .agg(
                cell_bind_count=("y", "sum"),
                cell_exposure_count=("y", "size"),
                centroid_lat=("lat", "mean"),
                centroid_lon=("lon", "mean"),
            )
            .reset_index()
        )
        centroid_rad = np.deg2rad(
            cell_stats[["centroid_lat", "centroid_lon"]].astype(float).to_numpy()
        )
        return {
            "cell_stats": cell_stats,
            "tree": BallTree(centroid_rad, metric="haversine"),
            "binds": cell_stats["cell_bind_count"].astype(float).to_numpy(),
            "exposures": cell_stats["cell_exposure_count"].astype(float).to_numpy(),
        }

    def _compute_from_cell_index(
        self, X: pd.DataFrame, cell_index: dict[str, Any]
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self._output_columns():
            out[col] = 0.0

        tree = cell_index["tree"]
        valid = _valid_lat_lon_mask(X, self.lat_col, self.lon_col)
        if tree is None or not valid.any():
            for radius in self.radii_miles:
                radius_label = int(radius)
                out[
                    f"{self.prefix}_cell_neighborhood_bind_rate_{radius_label}mi_oof"
                ] = self.global_mean_
            return out

        query = X.loc[valid, [self.lat_col, self.lon_col]]
        query_coords = _lat_lon_to_radians(query, self.lat_col, self.lon_col)
        binds_by_cell = cell_index["binds"]
        exposures_by_cell = cell_index["exposures"]

        for radius in self.radii_miles:
            radius_label = int(radius)
            radius_rad = radius / EARTH_RADIUS_MILES
            neighbor_indices = tree.query_radius(query_coords, r=radius_rad)

            bind_counts = np.zeros(len(query), dtype=float)
            exposure_counts = np.zeros(len(query), dtype=float)
            for i, neighbors in enumerate(neighbor_indices):
                if len(neighbors):
                    bind_counts[i] = float(binds_by_cell[neighbors].sum())
                    exposure_counts[i] = float(exposures_by_cell[neighbors].sum())

            rates = (bind_counts + self.alpha * self.global_mean_) / (
                exposure_counts + self.alpha
            )
            reliability = exposure_counts / (exposure_counts + self.alpha)

            out.loc[
                query.index,
                f"{self.prefix}_cell_neighborhood_bind_rate_{radius_label}mi_oof",
            ] = rates
            out.loc[
                query.index,
                f"{self.prefix}_cell_neighborhood_bind_count_{radius_label}mi_oof",
            ] = bind_counts
            out.loc[
                query.index,
                f"{self.prefix}_cell_neighborhood_exposure_count_{radius_label}mi_oof",
            ] = exposure_counts
            out.loc[
                query.index,
                f"{self.prefix}_cell_neighborhood_exposure_count_{radius_label}mi_log1p_oof",
            ] = np.log1p(exposure_counts)
            out.loc[
                query.index,
                f"{self.prefix}_cell_neighborhood_reliability_{radius_label}mi_oof",
            ] = reliability

            invalid_index = X.index[~valid]
            out.loc[
                invalid_index,
                f"{self.prefix}_cell_neighborhood_bind_rate_{radius_label}mi_oof",
            ] = self.global_mean_

        return out.astype(float)

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "OOFCellNeighborhoodTargetRateTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.cell_col, self.lat_col, self.lon_col])
        if y is None:
            raise ValueError(
                "OOFCellNeighborhoodTargetRateTransformer requires y during fit."
            )
        y_series = _validate_binary_target(
            y, X.index, "OOFCellNeighborhoodTargetRateTransformer"
        )
        self.global_mean_ = float(y_series.mean())
        self.cell_index_ = self._build_cell_index(X, y_series)
        self.feature_names_out_ = self._output_columns()
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.cell_col, self.lat_col, self.lon_col])
        if y is None:
            raise ValueError(
                "OOFCellNeighborhoodTargetRateTransformer requires y during fit_transform."
            )
        y_series = _validate_binary_target(
            y, X.index, "OOFCellNeighborhoodTargetRateTransformer"
        )
        self.global_mean_ = float(y_series.mean())
        self.feature_names_out_ = self._output_columns()
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_out_:
            out[col] = np.nan

        if self.stratified:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X, y_series)
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X)

        for train_pos, valid_pos in split_iter:
            train_index = X.index[train_pos]
            valid_index = X.index[valid_pos]
            cell_index = self._build_cell_index(
                X.loc[train_index], y_series.loc[train_index]
            )
            fold_out = self._compute_from_cell_index(X.loc[valid_index], cell_index)
            out.loc[valid_index, fold_out.columns] = fold_out

        self.cell_index_ = self._build_cell_index(X, y_series)
        return out.astype(float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.cell_col, self.lat_col, self.lon_col])
        if not hasattr(self, "cell_index_"):
            raise RuntimeError(
                "OOFCellNeighborhoodTargetRateTransformer has not been fit."
            )
        out = self._compute_from_cell_index(X, self.cell_index_)
        self.feature_names_out_ = list(out.columns)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)


class OOFNearbyTargetRateTransformer(BaseEstimator, TransformerMixin):
    """
    Out-of-fold radius-based local target-rate features.

    For each row, computes the smoothed target rate among nearby training rows.

    This is more expensive than cell target encoding, but avoids hard geographic
    boundaries.

    Features for each radius:
    - nearby_bind_rate_{radius}mi_oof
    - nearby_bind_count_{radius}mi_oof
    - nearby_exposure_count_{radius}mi_oof

    Smoothing:
        rate = (nearby_binds + alpha * global_mean) / (nearby_count + alpha)
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        radii_miles: Sequence[float] = (25.0, 50.0),
        alpha: float = 50.0,
        n_splits: int = 5,
        random_state: int = 42,
        stratified: bool = True,
        max_neighbors: int | None = None,
        prefix: str = "geo",
    ):
        warnings.warn(
            "OOFNearbyTargetRateTransformer is row-level and may be expensive on large datasets. "
            "Prefer OOFCellNeighborhoodTargetRateTransformer for large training data.",
            UserWarning,
            stacklevel=2,
        )
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.radii_miles = tuple(float(r) for r in radii_miles)
        self.alpha = float(alpha)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.stratified = stratified
        self.max_neighbors = max_neighbors
        self.prefix = prefix

    def _output_columns(self) -> list[str]:
        cols: list[str] = []
        for radius in self.radii_miles:
            radius_label = int(radius)
            cols.extend(
                [
                    f"{self.prefix}_nearby_bind_rate_{radius_label}mi_oof",
                    f"{self.prefix}_nearby_bind_count_{radius_label}mi_oof",
                    f"{self.prefix}_nearby_exposure_count_{radius_label}mi_oof",
                    f"{self.prefix}_nearby_exposure_count_{radius_label}mi_log1p_oof",
                ]
            )
        return cols

    def _compute_features_from_tree(
        self,
        query_X: pd.DataFrame,
        reference_X: pd.DataFrame,
        reference_y: pd.Series,
        tree: BallTree,
    ) -> pd.DataFrame:
        query_coords = _lat_lon_to_radians(query_X, self.lat_col, self.lon_col)
        out = pd.DataFrame(index=query_X.index)

        for radius in self.radii_miles:
            radius_rad = radius / EARTH_RADIUS_MILES
            neighbor_indices = tree.query_radius(query_coords, r=radius_rad)

            rates = np.empty(len(query_X), dtype=float)
            bind_counts = np.empty(len(query_X), dtype=float)
            exposure_counts = np.empty(len(query_X), dtype=float)

            ref_y_values = reference_y.astype(float).to_numpy()

            for i, neighbors in enumerate(neighbor_indices):
                if (
                    self.max_neighbors is not None
                    and len(neighbors) > self.max_neighbors
                ):
                    neighbors = neighbors[: self.max_neighbors]

                nearby_count = float(len(neighbors))
                nearby_binds = (
                    float(ref_y_values[neighbors].sum()) if len(neighbors) else 0.0
                )

                rate = (nearby_binds + self.alpha * self.global_mean_) / (
                    nearby_count + self.alpha
                )

                rates[i] = rate
                bind_counts[i] = nearby_binds
                exposure_counts[i] = nearby_count

            radius_label = int(radius)
            out[f"{self.prefix}_nearby_bind_rate_{radius_label}mi_oof"] = rates
            out[f"{self.prefix}_nearby_bind_count_{radius_label}mi_oof"] = bind_counts
            out[f"{self.prefix}_nearby_exposure_count_{radius_label}mi_oof"] = (
                exposure_counts
            )
            out[f"{self.prefix}_nearby_exposure_count_{radius_label}mi_log1p_oof"] = (
                np.log1p(exposure_counts)
            )

        return out

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "OOFNearbyTargetRateTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)

        if y is None:
            raise ValueError("OOFNearbyTargetRateTransformer requires y during fit.")

        y_series = _validate_binary_target(y, X.index, "OOFNearbyTargetRateTransformer")
        self.global_mean_ = float(y_series.mean())
        self.reference_X_ = X[[self.lat_col, self.lon_col]].copy()
        self.reference_y_ = y_series.copy()
        self.reference_tree_ = BallTree(
            _lat_lon_to_radians(self.reference_X_, self.lat_col, self.lon_col),
            metric="haversine",
        )
        self.feature_names_out_ = self._output_columns()
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)

        if y is None:
            raise ValueError(
                "OOFNearbyTargetRateTransformer requires y during fit_transform."
            )

        y_series = _validate_binary_target(y, X.index, "OOFNearbyTargetRateTransformer")
        self.global_mean_ = float(y_series.mean())

        if self.stratified:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X, y_series)
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X)

        out = pd.DataFrame(index=X.index)

        for train_pos, valid_pos in split_iter:
            train_index = X.index[train_pos]
            valid_index = X.index[valid_pos]

            reference_X = X.loc[train_index, [self.lat_col, self.lon_col]]
            reference_y = y_series.loc[train_index]

            tree = BallTree(
                _lat_lon_to_radians(reference_X, self.lat_col, self.lon_col),
                metric="haversine",
            )

            fold_out = self._compute_features_from_tree(
                query_X=X.loc[valid_index],
                reference_X=reference_X,
                reference_y=reference_y,
                tree=tree,
            )
            out.loc[valid_index, fold_out.columns] = fold_out

        self.reference_X_ = X[[self.lat_col, self.lon_col]].copy()
        self.reference_y_ = y_series.copy()
        self.reference_tree_ = BallTree(
            _lat_lon_to_radians(self.reference_X_, self.lat_col, self.lon_col),
            metric="haversine",
        )

        out = out.astype(float)
        self.feature_names_out_ = list(out.columns)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)

        if not hasattr(self, "reference_tree_"):
            raise RuntimeError("OOFNearbyTargetRateTransformer has not been fit.")

        out = self._compute_features_from_tree(
            query_X=X,
            reference_X=self.reference_X_,
            reference_y=self.reference_y_,
            tree=self.reference_tree_,
        ).astype(float)
        self.feature_names_out_ = list(out.columns)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names after fitting."""
        return _feature_names_or_raise(self)


def make_geo_key_features(
    lat_col: str = "lat",
    lon_col: str = "lon",
    use_h3: bool = True,
    require_h3: bool = True,
    resolutions: Sequence[int] = (4, 5, 6, 7),
) -> GeoFeatureUnion:
    """Build reusable target-free geography keys.

    Example:
        ``make_geo_key_features().fit_transform(X_train)`` computes missing
        coordinate indicators and H3 cell columns once for reuse downstream.
    """
    return GeoFeatureUnion(
        transformers=[
            (
                "lat_lon_missing",
                LatLonMissingIndicatorTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    prefix="geo",
                ),
            ),
            (
                "geo_cells",
                GeoCellTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    resolutions=resolutions,
                    use_h3=use_h3,
                    require_h3=require_h3,
                    prefix="geo",
                ),
            ),
        ]
    )


def make_phase1_geo_features(
    lat_col: str = "lat",
    lon_col: str = "lon",
    h3_cols: Sequence[str] = ("geo_h3_r4", "geo_h3_r5", "geo_h3_r6", "geo_h3_r7"),
    city_df: pd.DataFrame | None = None,
    include_rounded: bool = False,
    include_row_density: bool = False,
) -> GeoFeatureUnion:
    """Build target-free geographic features from precomputed geography keys."""
    transformers: list[tuple[str, TransformerMixin]] = [
        (
            "basic_lat_lon",
            LatLonBasicTransformer(
                lat_col=lat_col,
                lon_col=lon_col,
                include_raw=True,
                include_trig=True,
                include_interaction=True,
                missing_policy="sentinel",
                prefix="geo",
            ),
        ),
        (
            "nearest_city",
            NearestCityDistanceTransformer(
                lat_col=lat_col,
                lon_col=lon_col,
                city_df=city_df,
                population_thresholds=(100_000, 250_000, 1_000_000),
                include_nearest_city_name=True,
                include_nearest_city_population=True,
                missing_policy="sentinel",
                prefix="geo",
            ),
        ),
        (
            "cell_counts",
            GeoCellCountTransformer(
                cell_cols=h3_cols,
                unseen_count_value=0.0,
                add_log_count=True,
                add_frequency=True,
                prefix="geo",
            ),
        ),
    ]

    if include_rounded:
        transformers.append(
            (
                "rounded_lat_lon",
                RoundedLatLonTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    decimals=(0, 1, 2),
                    include_numeric=True,
                    include_categorical_cell=True,
                    prefix="geo",
                ),
            )
        )

    if include_row_density:
        transformers.append(
            (
                "customer_density",
                CustomerDensityTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    radii_miles=(10.0, 25.0, 50.0),
                    exclude_self=True,
                    add_log_density=True,
                    missing_policy="sentinel",
                    prefix="geo",
                ),
            )
        )

    return GeoFeatureUnion(transformers=transformers)


def make_phase2_geo_features(
    lat_col: str = "lat",
    lon_col: str = "lon",
    area_cols: Sequence[str] | None = None,
    h3_cols: Sequence[str] = ("geo_h3_r5", "geo_h3_r6", "geo_h3_r7"),
    alpha: float = 100.0,
    n_splits: int = 5,
    random_state: int = 42,
    include_local_parent: bool = True,
    include_cell_neighborhood: bool = False,
) -> GeoFeatureUnion:
    """Build leakage-aware target features from explicit area and H3 columns."""
    area_cols = tuple(area_cols or ())
    te_cols = [*area_cols, *h3_cols]
    transformers: list[tuple[str, TransformerMixin]] = [
        (
            "geo_target_encoding",
            OOFGeoTargetEncoder(
                cols=te_cols,
                lat_col=lat_col,
                lon_col=lon_col,
                create_geo_cells=False,
                alpha=alpha,
                n_splits=n_splits,
                random_state=random_state,
                stratified=True,
                add_count_features=True,
                add_count_log1p_features=True,
                add_reliability_features=True,
                add_logit_features=True,
                output_suffix="oof",
                prefix="te",
            ),
        ),
    ]

    if include_local_parent and {"geo_h3_r5", "geo_h3_r7"}.issubset(set(h3_cols)):
        transformers.append(
            (
                "local_vs_parent",
                LocalVsParentTargetEncoder(
                    local_col="geo_h3_r7",
                    parent_col="geo_h3_r5",
                    alpha_local=alpha / 2.0,
                    alpha_parent=alpha,
                    n_splits=n_splits,
                    random_state=random_state,
                    stratified=True,
                    add_component_rates=True,
                    add_count_features=True,
                    add_reliability_features=True,
                    add_ratio_feature=True,
                    prefix="te",
                ),
            )
        )

    if include_cell_neighborhood:
        transformers.append(
            (
                "cell_neighborhood_target_rate",
                OOFCellNeighborhoodTargetRateTransformer(
                    cell_col="geo_h3_r7",
                    lat_col=lat_col,
                    lon_col=lon_col,
                    radii_miles=(25.0, 50.0),
                    alpha=alpha,
                    n_splits=n_splits,
                    random_state=random_state,
                    stratified=True,
                    prefix="geo",
                ),
            ),
        )

    return GeoFeatureUnion(transformers=transformers)


__all__ = [
    "GeoFeatureUnion",
    "LatLonMissingIndicatorTransformer",
    "LatLonBasicTransformer",
    "GeoCellTransformer",
    "GeoCellCountTransformer",
    "NearestCityDistanceTransformer",
    "RoundedLatLonTransformer",
    "CustomerDensityTransformer",
    "OOFGeoTargetEncoder",
    "LocalVsParentTargetEncoder",
    "OOFCellNeighborhoodTargetRateTransformer",
    "OOFNearbyTargetRateTransformer",
    "make_geo_key_features",
    "make_phase1_geo_features",
    "make_phase2_geo_features",
]
