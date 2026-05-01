"""Spatial distance and density transformers."""

from __future__ import annotations

import warnings
import math
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import BallTree

from .constants import DEFAULT_MAJOR_CITIES, EARTH_RADIUS_MILES, MISSING_GEO_VALUE
from ._utils import (
    _as_dataframe,
    _feature_names_or_raise,
    _lat_lon_to_radians,
    _require_columns,
    _valid_lat_lon_mask,
    _validate_lat_lon_values,
    _validate_missing_policy,
)


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

        for threshold in self.population_thresholds:
            tree_info = self.threshold_trees_.get(threshold)
            if tree_info is None:
                continue

            subset, tree = tree_info
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
