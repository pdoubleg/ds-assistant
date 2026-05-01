"""Target-aware geographic encoding transformers."""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import BallTree

from .basic import GeoCellTransformer
from .constants import EARTH_RADIUS_MILES
from ._utils import (
    _as_dataframe,
    _as_key_series,
    _feature_names_or_raise,
    _lat_lon_to_radians,
    _require_columns,
    _valid_lat_lon_mask,
    _validate_binary_target,
    _validate_lat_lon_values,
)


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
    ):
        self.cell_cols = tuple(cell_cols)
        self.unseen_count_value = float(unseen_count_value)
        self.add_log_count = add_log_count
        self.add_frequency = add_frequency

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

    ``min_samples_leaf`` shrinks the encoded rate to ``global_mean`` for
    low-count categories, but count and reliability features still report the
    observed fold count.
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

    def _unknown_logit_fill_value(self) -> float:
        if self.handle_unknown == "global_mean":
            clipped_global_mean = np.clip(self.global_mean_, 1e-6, 1 - 1e-6)
            clipped_global_complement = np.clip(1 - self.global_mean_, 1e-6, 1)
            return math.log(clipped_global_mean / clipped_global_complement)
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
                        .fillna(self._unknown_logit_fill_value())
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
                    keys[col]
                    .map(stats["logit"])
                    .astype(float)
                    .fillna(self._unknown_logit_fill_value())
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
        output_suffix: str = "oof",
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
        self.output_suffix = output_suffix
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
        suffix = self.output_suffix
        cols: list[str] = []
        if self.add_component_rates:
            cols.extend([f"{base}_local_rate_{suffix}", f"{base}_parent_rate_{suffix}"])
        if self.add_count_features:
            cols.extend(
                [
                    f"{base}_local_count_{suffix}",
                    f"{base}_parent_count_{suffix}",
                    f"{base}_local_count_log1p_{suffix}",
                    f"{base}_parent_count_log1p_{suffix}",
                ]
            )
        if self.add_reliability_features:
            cols.extend(
                [
                    f"{base}_local_reliability_{suffix}",
                    f"{base}_parent_reliability_{suffix}",
                ]
            )
        cols.append(f"{base}_local_minus_parent_rate_{suffix}")
        if self.add_ratio_feature:
            cols.append(f"{base}_local_div_parent_rate_{suffix}")
        return cols

    def _transform_from_maps(
        self,
        X: pd.DataFrame,
        local_map: pd.DataFrame,
        parent_map: pd.DataFrame,
    ) -> pd.DataFrame:
        base = f"{self.prefix}_{self.local_col}_vs_{self.parent_col}"
        suffix = self.output_suffix
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
            out[f"{base}_local_rate_{suffix}"] = local_rate
            out[f"{base}_parent_rate_{suffix}"] = parent_rate
        if self.add_count_features:
            out[f"{base}_local_count_{suffix}"] = local_count
            out[f"{base}_parent_count_{suffix}"] = parent_count
            out[f"{base}_local_count_log1p_{suffix}"] = np.log1p(local_count)
            out[f"{base}_parent_count_log1p_{suffix}"] = np.log1p(parent_count)
        if self.add_reliability_features:
            out[f"{base}_local_reliability_{suffix}"] = local_reliability
            out[f"{base}_parent_reliability_{suffix}"] = parent_reliability
        out[f"{base}_local_minus_parent_rate_{suffix}"] = local_rate - parent_rate
        if self.add_ratio_feature:
            out[f"{base}_local_div_parent_rate_{suffix}"] = local_rate / np.maximum(
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

        query_cells = (
            X.loc[valid, [self.cell_col, self.lat_col, self.lon_col]]
            .assign(cell=lambda d: _as_key_series(d[self.cell_col]))
            .groupby("cell", dropna=False)
            .agg(
                query_lat=(self.lat_col, "mean"),
                query_lon=(self.lon_col, "mean"),
            )
            .reset_index()
        )
        query_coords = np.deg2rad(
            query_cells[["query_lat", "query_lon"]].astype(float).to_numpy()
        )
        query_cell_keys = _as_key_series(X.loc[valid, self.cell_col])
        binds_by_cell = cell_index["binds"]
        exposures_by_cell = cell_index["exposures"]

        for radius in self.radii_miles:
            radius_label = int(radius)
            radius_rad = radius / EARTH_RADIUS_MILES
            neighbor_indices = tree.query_radius(query_coords, r=radius_rad)

            bind_counts = np.zeros(len(query_cells), dtype=float)
            exposure_counts = np.zeros(len(query_cells), dtype=float)
            for i, neighbors in enumerate(neighbor_indices):
                if len(neighbors):
                    bind_counts[i] = float(binds_by_cell[neighbors].sum())
                    exposure_counts[i] = float(exposures_by_cell[neighbors].sum())

            rates = (bind_counts + self.alpha * self.global_mean_) / (
                exposure_counts + self.alpha
            )
            reliability = exposure_counts / (exposure_counts + self.alpha)

            cell_features = pd.DataFrame(
                {
                    "cell": query_cells["cell"],
                    "rate": rates,
                    "bind_count": bind_counts,
                    "exposure_count": exposure_counts,
                    "exposure_count_log1p": np.log1p(exposure_counts),
                    "reliability": reliability,
                }
            ).set_index("cell")

            out.loc[
                query_cell_keys.index,
                f"{self.prefix}_cell_neighborhood_bind_rate_{radius_label}mi_oof",
            ] = query_cell_keys.map(cell_features["rate"]).astype(float).to_numpy()
            out.loc[
                query_cell_keys.index,
                f"{self.prefix}_cell_neighborhood_bind_count_{radius_label}mi_oof",
            ] = (
                query_cell_keys.map(cell_features["bind_count"])
                .astype(float)
                .to_numpy()
            )
            out.loc[
                query_cell_keys.index,
                f"{self.prefix}_cell_neighborhood_exposure_count_{radius_label}mi_oof",
            ] = (
                query_cell_keys.map(cell_features["exposure_count"])
                .astype(float)
                .to_numpy()
            )
            out.loc[
                query_cell_keys.index,
                f"{self.prefix}_cell_neighborhood_exposure_count_{radius_label}mi_log1p_oof",
            ] = (
                query_cell_keys.map(cell_features["exposure_count_log1p"])
                .astype(float)
                .to_numpy()
            )
            out.loc[
                query_cell_keys.index,
                f"{self.prefix}_cell_neighborhood_reliability_{radius_label}mi_oof",
            ] = (
                query_cell_keys.map(cell_features["reliability"])
                .astype(float)
                .to_numpy()
            )

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
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.radii_miles = tuple(float(r) for r in radii_miles)
        self.alpha = float(alpha)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.stratified = stratified
        self.max_neighbors = max_neighbors
        self.prefix = prefix

    def _warn_expensive(self) -> None:
        """Warn when the expensive row-level target-rate transformer is fit."""
        warnings.warn(
            "OOFNearbyTargetRateTransformer is row-level and may be expensive on large datasets. "
            "Prefer OOFCellNeighborhoodTargetRateTransformer for large training data.",
            UserWarning,
            stacklevel=2,
        )

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
        self._warn_expensive()
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
        self._warn_expensive()
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
