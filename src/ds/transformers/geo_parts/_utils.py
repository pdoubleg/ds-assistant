"""Shared utility helpers for geo transformers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .constants import EARTH_RADIUS_MILES, MISSING_GEO_VALUE


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
