# Geo Transformer Parts

This package contains the implementation modules behind the public
`src.ds.transformers.geo` API. Import transformers and factories from
`src.ds.transformers.geo` unless you are editing internals.

The geo transformers are organized for large-scale direct-mail modeling, where
coordinates may be incomplete, H3 cells are reused across feature phases, and
target-aware features must avoid row self-label leakage.

## Module Map

- `constants.py`: Earth radius, missing geo sentinel, and built-in city data.
- `_utils.py`: dataframe validation, coordinate checks, key coercion, H3/grid helpers, and shared feature-name helpers.
- `_union.py`: `GeoFeatureUnion`, a lightweight sklearn-style feature union for dataframe outputs.
- `basic.py`: target-free coordinate, missingness, rounded-coordinate, and H3/grid cell transformers.
- `spatial.py`: nearest-city distance features and target-free customer density features.
- `target_encoding.py`: geo cell counts, OOF target encoders, local-vs-parent encoders, and neighborhood target-rate transformers.
- `factories.py`: recommended feature bundles for key, Phase 1, and Phase 2 geo features.

## Recommended Usage

Build reusable geography keys first, then feed those keys into target-free and
target-aware feature phases.

```python
import pandas as pd

from src.ds.transformers.geo import (
    make_geo_key_features,
    make_phase1_geo_features,
    make_phase2_geo_features,
)

X_train = pd.DataFrame(
    {
        "lat": [40.7128, 34.0522, None],
        "lon": [-74.0060, -118.2437, -80.0],
        "state": ["NY", "CA", "PA"],
    }
)
y_train = pd.Series([1, 0, 0])

geo_keys = make_geo_key_features(require_h3=False).fit_transform(X_train)
X_train_geo = pd.concat([X_train, geo_keys], axis=1)

phase1 = make_phase1_geo_features().fit_transform(X_train_geo)
phase2 = make_phase2_geo_features(
    area_cols=("state",),
    h3_cols=("geo_h3_r5", "geo_h3_r6", "geo_h3_r7"),
).fit_transform(X_train_geo, y_train)
```

Expected column families:

| Step | Example columns |
| --- | --- |
| `geo_keys` | `geo_lat_lon_valid`, `geo_lat_lon_missing`, `geo_grid_r4`, `geo_grid_r5`, `geo_grid_r6`, `geo_grid_r7` |
| `phase1` | `geo_lat`, `geo_lon`, `geo_sin_lat`, `geo_nearest_city_distance_miles`, `geo_nearest_city_population`, `geo_h3_r7_train_count` |
| `phase2` | `te_state_rate_oof`, `te_geo_h3_r7_rate_oof`, `te_geo_h3_r7_count_oof`, `te_geo_h3_r7_reliability_oof`, `te_geo_h3_r7_logit_oof` |

For validation or test data, fit transformers on training data only and call
`transform()` on holdout data. For cross-validation, keep target-aware
transformers inside the modeling pipeline so out-of-fold encodings do not leak
outer validation folds.

## Key Points

- Prefer `OOFCellNeighborhoodTargetRateTransformer` over `OOFNearbyTargetRateTransformer` for large training files. The cell-neighborhood transformer queries spatial neighbors once per unique query cell instead of once per row.
- `make_phase1_geo_features()` avoids nearest-city name/state categorical columns by default. Use `NearestCityDistanceTransformer(include_nearest_city_name=True)` directly if those columns are intentionally needed.
- `make_phase2_geo_features()` validates requested H3 columns before adding local-vs-parent or cell-neighborhood features. Pass `local_col`, `parent_col`, and `neighborhood_cell_col` when using non-default cell names.
- `NearestCityDistanceTransformer` always emits stable columns for all configured population thresholds. Empty thresholds produce `NaN` distances and missing categorical values.
- `RoundedLatLonTransformer(missing_policy="sentinel")` emits numeric `NaN` rounded coordinates and sentinel categorical cells for invalid coordinates.
- `OOFGeoTargetEncoder(handle_unknown="nan")` emits `NaN` rate/logit features for unseen categories while count, log-count, and reliability features remain `0.0`.
- `min_samples_leaf` in `OOFGeoTargetEncoder` shrinks low-count encoded rates to the global mean, but count and reliability features still report the observed fold counts.

## Direct Transformer Examples

Use nearest-city distances with stable threshold columns:

```python
from src.ds.transformers.geo import NearestCityDistanceTransformer

nearest = NearestCityDistanceTransformer(
    population_thresholds=(100_000, 250_000, 1_000_000),
    include_nearest_city_name=False,
    missing_policy="sentinel",
)
nearest_features = nearest.fit_transform(X_train)
```

Expected output columns:

| Feature type | Example columns |
| --- | --- |
| Base nearest city | `geo_nearest_city_distance_miles`, `geo_nearest_city_population` |
| Population thresholds | `geo_nearest_city_pop_gt_100000_distance_miles`, `geo_nearest_city_pop_gt_250000_distance_miles`, `geo_nearest_city_pop_gt_1000000_distance_miles` |
| Optional categories | `geo_nearest_city_name`, `geo_nearest_city_state`, `geo_nearest_city_pop_gt_100000_name` |

Use custom local-vs-parent suffixes when distinguishing OOF and full-map
encodings:

```python
from src.ds.transformers.geo import LocalVsParentTargetEncoder

encoder = LocalVsParentTargetEncoder(
    local_col="geo_h3_r7",
    parent_col="geo_h3_r5",
    output_suffix="full",
)
local_parent_features = encoder.fit_transform(X_train_geo, y_train)
```

Expected output columns with `output_suffix="full"`:

| Feature type | Example columns |
| --- | --- |
| Component rates | `te_geo_h3_r7_vs_geo_h3_r5_local_rate_full`, `te_geo_h3_r7_vs_geo_h3_r5_parent_rate_full` |
| Counts | `te_geo_h3_r7_vs_geo_h3_r5_local_count_full`, `te_geo_h3_r7_vs_geo_h3_r5_parent_count_log1p_full` |
| Reliability | `te_geo_h3_r7_vs_geo_h3_r5_local_reliability_full`, `te_geo_h3_r7_vs_geo_h3_r5_parent_reliability_full` |
| Comparisons | `te_geo_h3_r7_vs_geo_h3_r5_local_minus_parent_rate_full`, `te_geo_h3_r7_vs_geo_h3_r5_local_div_parent_rate_full` |

Add scalable cell-neighborhood target-rate features only when the requested
cell column exists:

```python
from src.ds.transformers.geo import OOFCellNeighborhoodTargetRateTransformer

neighborhood = OOFCellNeighborhoodTargetRateTransformer(
    cell_col="geo_h3_r7",
    radii_miles=(25.0, 50.0),
    alpha=100.0,
)
neighborhood_features = neighborhood.fit_transform(X_train_geo, y_train)
```

Expected output columns:

| Radius | Example columns |
| --- | --- |
| `25mi` | `geo_cell_neighborhood_bind_rate_25mi_oof`, `geo_cell_neighborhood_bind_count_25mi_oof`, `geo_cell_neighborhood_exposure_count_25mi_oof`, `geo_cell_neighborhood_reliability_25mi_oof` |
| `50mi` | `geo_cell_neighborhood_bind_rate_50mi_oof`, `geo_cell_neighborhood_bind_count_50mi_oof`, `geo_cell_neighborhood_exposure_count_50mi_log1p_oof`, `geo_cell_neighborhood_reliability_50mi_oof` |

## Development Notes

- Keep public imports flowing through `src.ds.transformers.geo`.
- Add new implementation code to the most specific `geo_parts` module rather than growing the facade.
- Preserve sklearn constructor behavior: store constructor arguments on `self`, avoid work in `__init__`, and keep schemas stable between `fit_transform()` and `transform()`.
- Target-aware transformers should require `y` during fitting and should document any leakage assumptions clearly.
