"""Focused tests for LLM-driven encoding in ``CAAFETransformer``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb
import xgboost as xgb
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.fe_v2 import (
    CAAFETransformer,
    FeatureEncodingCode,
    FeatureEngineeringDependencies,
    get_feature_generation_prompt,
    get_system_prompt,
)
from src.run_llm_code import run_llm_code, run_llm_encoder_code

FEATURE_CODE = 'df["num_x2"] = df["num"] * 2'
ENCODING_CODE = """
numeric_columns = list(df.select_dtypes(include=["number", "bool"]).columns)
categorical_columns = [col for col in df.columns if col not in numeric_columns]
transformers = []
if numeric_columns:
    transformers.append(("num", "passthrough", numeric_columns))
if categorical_columns:
    transformers.append(
        (
            "cat",
            sklearn.preprocessing.OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_columns,
        )
    )
encoder = sklearn.compose.ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=0.0,
)
"""

ENCODING_CODE_WITH_IMPORTS = """
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric_columns = list(df.select_dtypes(include=[np.number, "bool"]).columns)
categorical_columns = [col for col in df.columns if col not in numeric_columns]
transformers = []
if numeric_columns:
    transformers.append(("num", "passthrough", numeric_columns))
if categorical_columns:
    transformers.append(
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_columns,
        )
    )
encoder = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=0.0,
)
"""


def make_dataset() -> pd.DataFrame:
    """Build a small mixed-type dataset for encoding tests."""
    return pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "city": ["a", "b", "a", "c", "b", "d"],
            "flag": [True, False, True, False, True, False],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


def make_transformer() -> CAAFETransformer:
    """Build a lightweight transformer for deterministic local tests."""
    return CAAFETransformer(
        target_name="target",
        base_classifier=LogisticRegression(max_iter=200),
        iterations=1,
        n_splits=2,
        n_repeats=1,
        random_state=7,
    )


def test_xgboost_defaults_remain_backwards_compatible() -> None:
    """The transformer should still default to XGBoost when unspecified."""
    transformer = CAAFETransformer(
        target_name="target",
        iterations=1,
        n_splits=2,
        n_repeats=1,
        random_state=7,
    )

    assert transformer.estimator_type == "xgboost"
    assert isinstance(transformer.base_classifier, xgb.XGBClassifier)
    assert transformer.estimator_name == "XGBoost"


def test_lightgbm_estimator_type_uses_lightgbm_default_classifier() -> None:
    """Explicit LightGBM requests should create a LightGBM evaluator."""
    transformer = CAAFETransformer(
        target_name="target",
        estimator_type="lightgbm",
        iterations=1,
        n_splits=2,
        n_repeats=1,
        random_state=7,
    )

    assert transformer.estimator_type == "lightgbm"
    assert isinstance(transformer.base_classifier, lgb.LGBMClassifier)
    assert transformer.estimator_name == "LightGBM"


def test_estimator_family_is_inferred_from_supplied_classifier() -> None:
    """Known booster implementations should auto-select the matching family."""
    transformer = CAAFETransformer(
        target_name="target",
        base_classifier=lgb.LGBMClassifier(random_state=7, verbosity=-1),
        iterations=1,
        n_splits=2,
        n_repeats=1,
        random_state=7,
    )

    assert transformer.estimator_type == "lightgbm"
    assert transformer.estimator_name == "LightGBM"


def test_model_specific_prompts_include_lightgbm_guidance() -> None:
    """Prompt builders should inject LightGBM-specific encoding instructions."""
    system_prompt = get_system_prompt("lightgbm")
    feature_prompt = get_feature_generation_prompt(
        dataset_description="Claims dataset with mixed numeric and categorical fields.",
        target_name="target",
        dataset_summary="A small binary classification dataset summary.",
        max_features=3,
        current_features="num, city, flag",
        estimator_type="lightgbm",
    )

    assert "LightGBM" in system_prompt
    assert "compact" in system_prompt
    assert "LightGBM" in feature_prompt
    assert "frequency" in feature_prompt


def test_feature_encoding_code_requires_encoder_assignment() -> None:
    """Encoding code must assign a transformer to ``encoder``."""
    with pytest.raises(ValidationError):
        FeatureEncodingCode(
            name="missing_encoder",
            reasoning="This should fail validation.",
            code='sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")',
        )


def test_run_llm_encoder_code_returns_numeric_features() -> None:
    """The helper should fit and transform a mixed-type dataframe."""
    dataset = make_dataset()
    train = dataset.iloc[:4].drop(columns=["target"])
    valid = dataset.iloc[4:].drop(columns=["target"])

    train_x, valid_x, encoder = run_llm_encoder_code(
        ENCODING_CODE,
        df_train=train,
        df_test=valid,
    )

    assert train_x.shape[0] == len(train)
    assert valid_x.shape[0] == len(valid)
    assert hasattr(encoder, "transform")
    assert np.asarray(train_x).dtype.kind in {"f", "i"}


def test_run_llm_encoder_code_allows_typical_modeling_imports() -> None:
    """Typical sklearn import patterns and ``np.number`` should be accepted."""
    dataset = make_dataset()
    train = dataset.iloc[:4].drop(columns=["target"])
    valid = dataset.iloc[4:].drop(columns=["target"])

    train_x, valid_x, encoder = run_llm_encoder_code(
        ENCODING_CODE_WITH_IMPORTS,
        df_train=train,
        df_test=valid,
    )

    assert train_x.shape[0] == len(train)
    assert valid_x.shape[0] == len(valid)
    assert hasattr(encoder, "transform")


def test_evaluate_features_uses_candidate_encoding_code() -> None:
    """Fold evaluation should succeed with separate feature and encoding code."""
    dataset = make_dataset()
    transformer = make_transformer()
    transformer.deps = FeatureEngineeringDependencies(
        original_dataset=dataset,
        dataset=dataset,
        target_name="target",
        dataset_description="mixed dataset",
        current_features=["num", "city", "flag"],
        agent_notepad=[],
    )

    old_results, new_results = transformer.evaluate_features(
        full_code="",
        code=FEATURE_CODE,
        full_encoding_code="",
        encoding_code=ENCODING_CODE,
    )

    assert old_results["accuracy"]
    assert new_results["accuracy"]
    assert len(old_results["accuracy"]) == len(new_results["accuracy"])


def test_transform_applies_feature_and_encoding_pipeline() -> None:
    """Transform should execute feature code and the fitted encoding pipeline."""
    dataset = make_dataset()
    transformer = make_transformer()
    transformer.deps = FeatureEngineeringDependencies(
        original_dataset=dataset,
        dataset=run_llm_code(FEATURE_CODE, dataset),
        target_name="target",
        dataset_description="mixed dataset",
        current_features=["num", "city", "flag"],
        agent_notepad=[],
    )
    transformer.full_code = FEATURE_CODE
    transformer.full_encoding_code = ENCODING_CODE
    transformer._refresh_fitted_encoder()
    transformer._is_fitted = True

    transformed = transformer.transform(dataset.drop(columns=["target"]))

    assert transformed.shape[0] == len(dataset)
