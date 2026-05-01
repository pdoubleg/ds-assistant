"""Factory functions for recommended geo feature sets."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.base import TransformerMixin

from ._union import GeoFeatureUnion
from .basic import (
    GeoCellTransformer,
    LatLonBasicTransformer,
    LatLonMissingIndicatorTransformer,
    RoundedLatLonTransformer,
)
from .spatial import CustomerDensityTransformer, NearestCityDistanceTransformer
from .target_encoding import (
    GeoCellCountTransformer,
    LocalVsParentTargetEncoder,
    OOFCellNeighborhoodTargetRateTransformer,
    OOFGeoTargetEncoder,
)


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
                include_nearest_city_name=False,
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
                    missing_policy="sentinel",
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
    neighborhood_cell_col: str = "geo_h3_r7",
    local_col: str = "geo_h3_r7",
    parent_col: str = "geo_h3_r5",
) -> GeoFeatureUnion:
    """Build leakage-aware target features from explicit area and H3 columns."""
    area_cols = tuple(area_cols or ())
    h3_cols = tuple(h3_cols)
    h3_col_set = set(h3_cols)

    if include_cell_neighborhood and neighborhood_cell_col not in h3_col_set:
        raise ValueError(
            f"include_cell_neighborhood=True requires {neighborhood_cell_col!r} "
            "to be present in h3_cols."
        )
    if include_local_parent and {local_col, parent_col}.difference(h3_col_set):
        missing = sorted({local_col, parent_col}.difference(h3_col_set))
        raise ValueError(
            "include_local_parent=True requires local_col and parent_col to be "
            f"present in h3_cols. Missing: {missing}."
        )

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

    if include_local_parent:
        transformers.append(
            (
                "local_vs_parent",
                LocalVsParentTargetEncoder(
                    local_col=local_col,
                    parent_col=parent_col,
                    alpha_local=alpha / 2.0,
                    alpha_parent=alpha,
                    n_splits=n_splits,
                    random_state=random_state,
                    stratified=True,
                    add_component_rates=True,
                    add_count_features=True,
                    add_reliability_features=True,
                    add_ratio_feature=True,
                    output_suffix="oof",
                    prefix="te",
                ),
            )
        )

    if include_cell_neighborhood:
        transformers.append(
            (
                "cell_neighborhood_target_rate",
                OOFCellNeighborhoodTargetRateTransformer(
                    cell_col=neighborhood_cell_col,
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
