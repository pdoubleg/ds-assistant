from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ds.transformers.geo import (
    GeoCellCountTransformer,
    GeoCellTransformer,
    GeoFeatureUnion,
    LatLonMissingIndicatorTransformer,
    LocalVsParentTargetEncoder,
    NearestCityDistanceTransformer,
    OOFGeoTargetEncoder,
)


def test_missing_coordinate_handling() -> None:
    df = pd.DataFrame(
        {
            "lat": [40.0, np.nan, 91.0],
            "lon": [-75.0, -80.0, -181.0],
            "y": [0, 1, 0],
        }
    )

    indicators = LatLonMissingIndicatorTransformer().fit_transform(df)
    assert indicators["geo_lat_lon_valid"].tolist() == [1, 0, 0]
    assert indicators["geo_lat_lon_missing"].tolist() == [0, 1, 1]

    cells = GeoCellTransformer(
        resolutions=(4,), use_h3=False, require_h3=False
    ).fit_transform(df)
    assert cells["geo_grid_r4"].astype(str).tolist()[1:] == [
        "__MISSING__",
        "__MISSING__",
    ]

    nearest = NearestCityDistanceTransformer(
        population_thresholds=(), missing_policy="sentinel"
    ).fit_transform(df)
    assert nearest.loc[1, "geo_nearest_city_name"] == "__MISSING__"
    assert np.isnan(nearest.loc[2, "geo_nearest_city_distance_miles"])


def test_oof_geo_target_encoder_no_leakage_shape_and_unknowns() -> None:
    X = pd.DataFrame({"area": [f"a{i % 4}" for i in range(20)]})
    y = pd.Series([0, 1] * 10)

    encoder = OOFGeoTargetEncoder(cols=["area"], n_splits=5, alpha=2.0)
    train_out = encoder.fit_transform(X, y)
    test_out = encoder.transform(pd.DataFrame({"area": ["new_area"]}))

    assert not train_out.isna().any().any()
    assert list(train_out.columns) == list(encoder.transform(X).columns)
    assert test_out.loc[0, "te_area_rate_oof"] == pytest.approx(float(y.mean()))
    assert "te_area_count_oof" in train_out.columns
    assert "te_area_count_log1p_oof" in train_out.columns
    assert "te_area_reliability_oof" in train_out.columns


def test_local_vs_parent_outputs() -> None:
    X = pd.DataFrame(
        {
            "local": [f"l{i % 5}" for i in range(30)],
            "parent": [f"p{i % 2}" for i in range(30)],
        }
    )
    y = pd.Series([0, 1] * 15)

    out = LocalVsParentTargetEncoder(
        local_col="local",
        parent_col="parent",
        n_splits=3,
    ).fit_transform(X, y)

    expected_fragments = [
        "local_rate",
        "parent_rate",
        "local_minus_parent_rate",
        "local_count_log1p",
        "parent_count_log1p",
    ]
    for fragment in expected_fragments:
        assert any(fragment in col for col in out.columns)


def test_geo_cell_count_transformer_counts_frequency_and_log() -> None:
    train = pd.DataFrame({"cell": ["a", "a", "b", "__MISSING__"]})
    test = pd.DataFrame({"cell": ["a", "c", "__MISSING__"]})

    transformer = GeoCellCountTransformer(cell_cols=["cell"])
    transformer.fit(train)
    out = transformer.transform(test)

    assert out["cell_train_count"].tolist() == [2.0, 0.0, 1.0]
    assert out["cell_train_frequency"].tolist() == [0.5, 0.0, 0.25]
    assert out["cell_train_count_log1p"].tolist() == pytest.approx(
        np.log1p([2.0, 0.0, 1.0])
    )


class _ConstantFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y=None):
        self.feature_names_out_ = [self.column]
        return self

    def transform(self, X):
        return pd.DataFrame({self.column: 1}, index=X.index)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)


def test_geo_feature_union_validation_and_feature_names() -> None:
    X = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(ValueError, match="Transformer names must be unique"):
        GeoFeatureUnion(
            [
                ("dup", _ConstantFrameTransformer("a")),
                ("dup", _ConstantFrameTransformer("b")),
            ]
        ).fit_transform(X)

    with pytest.raises(ValueError, match="Duplicate output columns"):
        GeoFeatureUnion(
            [
                ("a", _ConstantFrameTransformer("same")),
                ("b", _ConstantFrameTransformer("same")),
            ]
        ).fit_transform(X)

    union = GeoFeatureUnion(
        [
            ("a", _ConstantFrameTransformer("one")),
            ("b", _ConstantFrameTransformer("two")),
        ]
    )
    out = union.fit_transform(X)

    assert list(out.columns) == ["one", "two"]
    assert union.get_feature_names_out().tolist() == ["one", "two"]
