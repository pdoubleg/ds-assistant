from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import BallTree


EARTH_RADIUS_MILES = 3958.7613


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
) -> pd.Series:
    try:
        import h3  # type: ignore
    except ImportError:
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
        self.transformers = list(transformers)

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "GeoFeatureUnion":
        X = _as_dataframe(X)
        self.fitted_transformers_: list[tuple[str, TransformerMixin]] = []

        for name, transformer in self.transformers:
            fitted = clone(transformer)
            fitted.fit(X, y)
            self.fitted_transformers_.append((name, fitted))

        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        X = _as_dataframe(X)
        self.fitted_transformers_ = []
        frames = []

        for name, transformer in self.transformers:
            fitted = clone(transformer)
            Xt = fitted.fit_transform(X, y)
            Xt = _as_dataframe(Xt)
            frames.append(Xt)
            self.fitted_transformers_.append((name, fitted))

        if not frames:
            return pd.DataFrame(index=X.index)

        return pd.concat(frames, axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)

        if not hasattr(self, "fitted_transformers_"):
            raise RuntimeError("GeoFeatureUnion has not been fit.")

        frames = []
        for name, transformer in self.fitted_transformers_:
            Xt = transformer.transform(X)
            Xt = _as_dataframe(Xt)
            frames.append(Xt)

        if not frames:
            return pd.DataFrame(index=X.index)

        return pd.concat(frames, axis=1)


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
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.include_raw = include_raw
        self.include_trig = include_trig
        self.include_interaction = include_interaction
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "LatLonBasicTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col)

        lat = X[self.lat_col].astype(float)
        lon = X[self.lon_col].astype(float)

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

        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


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
        return self

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

        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class GeoCellTransformer(BaseEstimator, TransformerMixin):
    """
    Adds H3 or fallback grid-cell categorical columns.

    If `use_h3=True` and h3 is installed, true H3 cells are used.
    Otherwise a rounded lat/lon grid fallback is used.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        resolutions: Sequence[int] = (5, 6, 7),
        use_h3: bool = True,
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.resolutions = tuple(resolutions)
        self.use_h3 = use_h3
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "GeoCellTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col)

        out = pd.DataFrame(index=X.index)
        lat = X[self.lat_col].astype(float)
        lon = X[self.lon_col].astype(float)

        for resolution in self.resolutions:
            if self.use_h3:
                cell = _try_make_h3_cells(lat, lon, resolution)
                col = f"{self.prefix}_h3_r{resolution}"
            else:
                cell = _make_grid_cell(lat, lon, resolution)
                col = f"{self.prefix}_grid_r{resolution}"

            out[col] = cell.astype("category")

        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


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
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "NearestCityDistanceTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col)

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

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col)

        coords_rad = _lat_lon_to_radians(X, self.lat_col, self.lon_col)
        out = pd.DataFrame(index=X.index)

        dist_rad, ind = self.city_tree_.query(coords_rad, k=1)
        dist_miles = dist_rad[:, 0] * EARTH_RADIUS_MILES
        nearest_idx = ind[:, 0]
        nearest_cities = self.city_df_.iloc[nearest_idx].reset_index(drop=True)

        out[f"{self.prefix}_nearest_city_distance_miles"] = dist_miles

        if self.include_nearest_city_population:
            out[f"{self.prefix}_nearest_city_population"] = (
                nearest_cities[self.city_population_col].astype(float).to_numpy()
            )

        if self.include_nearest_city_name:
            out[f"{self.prefix}_nearest_city_name"] = (
                nearest_cities[self.city_name_col].astype(str).to_numpy()
            )
            out[f"{self.prefix}_nearest_city_state"] = (
                nearest_cities[self.city_state_col].astype(str).to_numpy()
            )

        for threshold, (subset, tree) in self.threshold_trees_.items():
            threshold_dist_rad, threshold_ind = tree.query(coords_rad, k=1)
            threshold_dist_miles = threshold_dist_rad[:, 0] * EARTH_RADIUS_MILES
            out[f"{self.prefix}_nearest_city_pop_gt_{threshold}_distance_miles"] = (
                threshold_dist_miles
            )

            if self.include_nearest_city_name:
                threshold_nearest = subset.iloc[threshold_ind[:, 0]].reset_index(
                    drop=True
                )
                out[f"{self.prefix}_nearest_city_pop_gt_{threshold}_name"] = (
                    threshold_nearest[self.city_name_col].astype(str).to_numpy()
                )
                out[f"{self.prefix}_nearest_city_pop_gt_{threshold}_state"] = (
                    threshold_nearest[self.city_state_col].astype(str).to_numpy()
                )

        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype("category")

        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


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
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.radii_miles = tuple(float(r) for r in radii_miles)
        self.exclude_self = exclude_self
        self.add_log_density = add_log_density
        self.prefix = prefix

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "CustomerDensityTransformer":
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)

        self.fit_index_ = X.index.copy()
        self.coords_rad_ = _lat_lon_to_radians(X, self.lat_col, self.lon_col)
        self.tree_ = BallTree(self.coords_rad_, metric="haversine")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)

        coords_rad = _lat_lon_to_radians(X, self.lat_col, self.lon_col)
        same_index = X.index.equals(self.fit_index_)

        out = pd.DataFrame(index=X.index)

        for radius in self.radii_miles:
            radius_rad = radius / EARTH_RADIUS_MILES
            counts = self.tree_.query_radius(
                coords_rad, r=radius_rad, count_only=True
            ).astype(float)

            if self.exclude_self and same_index:
                counts = np.maximum(counts - 1.0, 0.0)

            count_col = f"{self.prefix}_customer_count_within_{int(radius)}mi"
            out[count_col] = counts

            area = math.pi * radius**2
            density_col = (
                f"{self.prefix}_customer_density_per_sqmi_within_{int(radius)}mi"
            )
            out[density_col] = counts / area

            if self.add_log_density:
                out[f"{count_col}_log1p"] = np.log1p(counts)
                out[f"{density_col}_log1p"] = np.log1p(counts / area)

        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class GeoCellCountTransformer(BaseEstimator, TransformerMixin):
    """
    Count observations in each geo cell. Useful as a simple density proxy.

    You can pass pre-existing cell columns or have this transformer create
    fallback grid/H3-like cells from lat/lon.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        cell_cols: Sequence[str] | None = None,
        resolutions: Sequence[int] = (5, 6, 7),
        use_h3: bool = True,
        smoothing_count: float = 0.0,
        prefix: str = "geo",
    ):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.cell_cols = None if cell_cols is None else tuple(cell_cols)
        self.resolutions = tuple(resolutions)
        self.use_h3 = use_h3
        self.smoothing_count = smoothing_count
        self.prefix = prefix

    def _make_cells(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.cell_cols is not None:
            _require_columns(X, list(self.cell_cols))
            return X[list(self.cell_cols)].astype("string")

        celler = GeoCellTransformer(
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            resolutions=self.resolutions,
            use_h3=self.use_h3,
            prefix=self.prefix,
        )
        return celler.fit_transform(X).astype("string")

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
                .fillna(self.smoothing_count)
            )
            out[f"{col}_train_count"] = counts
            out[f"{col}_train_count_log1p"] = np.log1p(counts)

        return out

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


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
        create_geo_cells: bool = True,
        resolutions: Sequence[int] = (5, 6, 7),
        use_h3: bool = True,
        alpha: float = 50.0,
        n_splits: int = 5,
        random_state: int = 42,
        stratified: bool = True,
        add_count_features: bool = True,
        add_logit_features: bool = True,
        prefix: str = "te",
    ):
        self.cols = None if cols is None else tuple(cols)
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.create_geo_cells = create_geo_cells
        self.resolutions = tuple(resolutions)
        self.use_h3 = use_h3
        self.alpha = float(alpha)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.stratified = stratified
        self.add_count_features = add_count_features
        self.add_logit_features = add_logit_features
        self.prefix = prefix

    def _build_encoding_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.cols is not None:
            _require_columns(X, list(self.cols))
            parts.append(X[list(self.cols)].astype("string"))

        if self.create_geo_cells:
            celler = GeoCellTransformer(
                lat_col=self.lat_col,
                lon_col=self.lon_col,
                resolutions=self.resolutions,
                use_h3=self.use_h3,
                prefix="geo",
            )
            parts.append(celler.fit_transform(X).astype("string"))

        if not parts:
            raise ValueError("No columns supplied and create_geo_cells=False.")

        return pd.concat(parts, axis=1)

    def _fit_maps_for_frame(
        self,
        keys: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        maps: dict[str, pd.DataFrame] = {}

        for col in keys.columns:
            stats = (
                pd.DataFrame(
                    {"key": keys[col].astype("string"), "y": y.astype(float).to_numpy()}
                )
                .groupby("key", dropna=False)["y"]
                .agg(["sum", "count"])
            )
            stats["encoded"] = (stats["sum"] + self.alpha * self.global_mean_) / (
                stats["count"] + self.alpha
            )
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

        y_series = pd.Series(y, index=X.index).astype(float)
        if not y_series.isin([0.0, 1.0]).all():
            raise ValueError(
                "OOFGeoTargetEncoder expects a binary target with values 0/1."
            )

        keys = self._build_encoding_frame(X)
        self.key_cols_ = list(keys.columns)
        self.global_mean_ = float(y_series.mean())
        self.full_maps_ = self._fit_maps_for_frame(keys, y_series)
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

        y_series = pd.Series(y, index=X.index).astype(float)
        if not y_series.isin([0.0, 1.0]).all():
            raise ValueError(
                "OOFGeoTargetEncoder expects a binary target with values 0/1."
            )

        keys = self._build_encoding_frame(X)
        self.key_cols_ = list(keys.columns)
        self.global_mean_ = float(y_series.mean())

        out = pd.DataFrame(index=X.index)

        for col in self.key_cols_:
            out[f"{self.prefix}_{col}_rate_oof"] = np.nan
            if self.add_count_features:
                out[f"{self.prefix}_{col}_count_oof"] = np.nan
            if self.add_logit_features:
                out[f"{self.prefix}_{col}_logit_oof"] = np.nan

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
                    .fillna(self.global_mean_)
                )
                out.loc[valid_index, f"{self.prefix}_{col}_rate_oof"] = (
                    encoded.to_numpy()
                )

                if self.add_count_features:
                    counts = (
                        valid_keys[col].map(stats["count"]).astype(float).fillna(0.0)
                    )
                    out.loc[valid_index, f"{self.prefix}_{col}_count_oof"] = (
                        counts.to_numpy()
                    )

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
                    out.loc[valid_index, f"{self.prefix}_{col}_logit_oof"] = (
                        logits.to_numpy()
                    )

        self.full_maps_ = self._fit_maps_for_frame(keys, y_series)

        for col in out.columns:
            out[col] = out[col].astype(float)

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
                keys[col].map(stats["encoded"]).astype(float).fillna(self.global_mean_)
            )
            out[f"{self.prefix}_{col}_rate_oof"] = encoded

            if self.add_count_features:
                counts = keys[col].map(stats["count"]).astype(float).fillna(0.0)
                out[f"{self.prefix}_{col}_count_oof"] = counts

            if self.add_logit_features:
                logits = (
                    keys[col].map(stats["logit"]).astype(float).fillna(global_logit)
                )
                out[f"{self.prefix}_{col}_logit_oof"] = logits

        return out


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
        prefix: str = "te",
    ):
        self.local_col = local_col
        self.parent_col = parent_col
        self.alpha_local = float(alpha_local)
        self.alpha_parent = float(alpha_parent)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.stratified = stratified
        self.prefix = prefix

    def _fit_single_map(
        self,
        keys: pd.Series,
        y: pd.Series,
        alpha: float,
        global_mean: float,
    ) -> pd.DataFrame:
        stats = (
            pd.DataFrame(
                {"key": keys.astype("string"), "y": y.astype(float).to_numpy()}
            )
            .groupby("key", dropna=False)["y"]
            .agg(["sum", "count"])
        )
        stats["rate"] = (stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)
        return stats

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> "LocalVsParentTargetEncoder":
        X = _as_dataframe(X)
        _require_columns(X, [self.local_col, self.parent_col])

        if y is None:
            raise ValueError("LocalVsParentTargetEncoder requires y during fit.")

        y_series = pd.Series(y, index=X.index).astype(float)
        self.global_mean_ = float(y_series.mean())
        self.local_map_ = self._fit_single_map(
            X[self.local_col], y_series, self.alpha_local, self.global_mean_
        )
        self.parent_map_ = self._fit_single_map(
            X[self.parent_col], y_series, self.alpha_parent, self.global_mean_
        )
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

        y_series = pd.Series(y, index=X.index).astype(float)
        self.global_mean_ = float(y_series.mean())

        out_col = f"{self.prefix}_{self.local_col}_minus_{self.parent_col}_rate_oof"
        out = pd.DataFrame(index=X.index)
        out[out_col] = np.nan

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

            local_rate = (
                X.loc[valid_index, self.local_col]
                .astype("string")
                .map(local_map["rate"])
                .astype(float)
                .fillna(self.global_mean_)
            )
            parent_rate = (
                X.loc[valid_index, self.parent_col]
                .astype("string")
                .map(parent_map["rate"])
                .astype(float)
                .fillna(self.global_mean_)
            )

            out.loc[valid_index, out_col] = (local_rate - parent_rate).to_numpy()

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

        local_rate = (
            X[self.local_col]
            .astype("string")
            .map(self.local_map_["rate"])
            .astype(float)
            .fillna(self.global_mean_)
        )
        parent_rate = (
            X[self.parent_col]
            .astype("string")
            .map(self.parent_map_["rate"])
            .astype(float)
            .fillna(self.global_mean_)
        )

        out = pd.DataFrame(index=X.index)
        out[f"{self.prefix}_{self.local_col}_minus_{self.parent_col}_rate_oof"] = (
            local_rate - parent_rate
        )
        return out.astype(float)


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
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.radii_miles = tuple(float(r) for r in radii_miles)
        self.alpha = float(alpha)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.stratified = stratified
        self.max_neighbors = max_neighbors
        self.prefix = prefix

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

        y_series = pd.Series(y, index=X.index).astype(float)
        self.global_mean_ = float(y_series.mean())
        self.reference_X_ = X[[self.lat_col, self.lon_col]].copy()
        self.reference_y_ = y_series.copy()
        self.reference_tree_ = BallTree(
            _lat_lon_to_radians(self.reference_X_, self.lat_col, self.lon_col),
            metric="haversine",
        )
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

        y_series = pd.Series(y, index=X.index).astype(float)
        self.global_mean_ = float(y_series.mean())

        out_parts = []

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

        return out.astype(float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_dataframe(X)
        _require_columns(X, [self.lat_col, self.lon_col])
        _validate_lat_lon_values(X, self.lat_col, self.lon_col, allow_missing=False)

        if not hasattr(self, "reference_tree_"):
            raise RuntimeError("OOFNearbyTargetRateTransformer has not been fit.")

        return self._compute_features_from_tree(
            query_X=X,
            reference_X=self.reference_X_,
            reference_y=self.reference_y_,
            tree=self.reference_tree_,
        ).astype(float)


def make_phase1_geo_features(
    lat_col: str = "lat",
    lon_col: str = "lon",
    city_df: pd.DataFrame | None = None,
    use_h3: bool = True,
) -> GeoFeatureUnion:
    """
    Recommended Phase 1 feature bundle.
    """
    return GeoFeatureUnion(
        transformers=[
            (
                "basic_lat_lon",
                LatLonBasicTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    include_raw=True,
                    include_trig=True,
                    include_interaction=True,
                    prefix="geo",
                ),
            ),
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
            ),
            (
                "geo_cells",
                GeoCellTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    resolutions=(5, 6, 7),
                    use_h3=use_h3,
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
                    prefix="geo",
                ),
            ),
            (
                "customer_density",
                CustomerDensityTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    radii_miles=(10.0, 25.0, 50.0),
                    exclude_self=True,
                    add_log_density=True,
                    prefix="geo",
                ),
            ),
            (
                "cell_counts",
                GeoCellCountTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    cell_cols=None,
                    resolutions=(5, 6, 7),
                    use_h3=use_h3,
                    smoothing_count=0.0,
                    prefix="geo",
                ),
            ),
        ]
    )


def make_phase2_geo_features(
    lat_col: str = "lat",
    lon_col: str = "lon",
    area_cols: Sequence[str] | None = None,
    use_h3: bool = True,
    alpha: float = 50.0,
    n_splits: int = 5,
    random_state: int = 42,
) -> GeoFeatureUnion:
    """
    Recommended Phase 2 target-aware feature bundle.

    area_cols can include columns such as:
    - state
    - county
    - zip3
    - rating_territory
    - agency_territory
    """
    return GeoFeatureUnion(
        transformers=[
            (
                "geo_target_encoding",
                OOFGeoTargetEncoder(
                    cols=area_cols,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    create_geo_cells=True,
                    resolutions=(5, 6, 7),
                    use_h3=use_h3,
                    alpha=alpha,
                    n_splits=n_splits,
                    random_state=random_state,
                    stratified=True,
                    add_count_features=True,
                    add_logit_features=True,
                    prefix="te",
                ),
            ),
            (
                "nearby_target_rate",
                OOFNearbyTargetRateTransformer(
                    lat_col=lat_col,
                    lon_col=lon_col,
                    radii_miles=(25.0, 50.0),
                    alpha=alpha,
                    n_splits=n_splits,
                    random_state=random_state,
                    stratified=True,
                    max_neighbors=None,
                    prefix="geo",
                ),
            ),
        ]
    )


__all__ = [
    "make_phase1_geo_features",
    "make_phase2_geo_features",
]
