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

from .geo_parts.constants import (
    EARTH_RADIUS_MILES,
    MISSING_GEO_VALUE,
    DEFAULT_MAJOR_CITIES,
)
from .geo_parts._union import GeoFeatureUnion
from .geo_parts.basic import (
    GeoCellTransformer,
    LatLonBasicTransformer,
    LatLonMissingIndicatorTransformer,
    RoundedLatLonTransformer,
)
from .geo_parts.spatial import (
    CustomerDensityTransformer,
    NearestCityDistanceTransformer,
)
from .geo_parts.target_encoding import (
    GeoCellCountTransformer,
    LocalVsParentTargetEncoder,
    OOFCellNeighborhoodTargetRateTransformer,
    OOFGeoTargetEncoder,
    OOFNearbyTargetRateTransformer,
)
from .geo_parts.factories import (
    make_geo_key_features,
    make_phase1_geo_features,
    make_phase2_geo_features,
)

__all__ = [
    "EARTH_RADIUS_MILES",
    "MISSING_GEO_VALUE",
    "DEFAULT_MAJOR_CITIES",
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
