"""
Utility functions for the AutoML Streamlit Demo App

This module provides self-contained utility functions for data processing,
analysis, and visualization without external dependencies on the main codebase.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


def get_dummy_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic insurance claims dataset for demonstration purposes.
    
    Creates a realistic dataset with numerical and categorical features,
    missing values, and a binary target variable for high-severity claims.
    
    Args:
        n_samples (int): Number of samples to generate. Defaults to 10000.
        
    Returns:
        pd.DataFrame: Synthetic insurance claims dataset with features and target.
        
    Example:
        >>> df = get_dummy_data(1000)
        >>> print(df.shape)
        (1000, 24)
        >>> print(df['claim'].value_counts())
        0    850
        1    150
    """
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    # Generate synthetic insurance claims data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.85, 0.15],  # Imbalanced classes (typical for insurance claims)
        random_state=42,
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [
        "age",
        "income",
        "credit_score",
        "driving_experience",
        "vehicle_age",
        "vehicle_value",
        "annual_mileage",
        "location_risk",
        "previous_claims",
        "policy_duration",
        "deductible_amount",
        "coverage_type",
        "marital_status",
        "education_level",
        "occupation_risk",
    ] + [f"feature_{i}" for i in range(15, 20)]

    df = pd.DataFrame(X, columns=feature_names)
    df["claim"] = y

    # Add categorical columns
    np.random.seed(42)  # For reproducibility
    df["region"] = np.random.choice(["North", "South", "East", "West"], size=len(df))
    df["policy_type"] = np.random.choice(["Basic", "Standard", "Premium"], size=len(df))
    df["vehicle_type"] = np.random.choice(["Sedan", "SUV", "Truck", "Coupe"], size=len(df))

    # Add missing values to selected columns (realistic patterns)
    # Set 5% of income values to NaN
    df.loc[
        np.random.choice(df.index, size=int(0.05 * len(df)), replace=False), "income"
    ] = np.nan

    # Set 3% of credit_score values to NaN
    df.loc[
        np.random.choice(df.index, size=int(0.03 * len(df)), replace=False),
        "credit_score",
    ] = np.nan

    # Set 7% of vehicle_value values to NaN
    df.loc[
        np.random.choice(df.index, size=int(0.07 * len(df)), replace=False),
        "vehicle_value",
    ] = np.nan

    # Set 2% of policy_type values to NaN
    df.loc[
        np.random.choice(df.index, size=int(0.02 * len(df)), replace=False),
        "policy_type",
    ] = np.nan

    return df


def get_dataset_summary_with_importance(
    df_train: pd.DataFrame,
    target_name: str,
    df_test: Optional[pd.DataFrame] = None,
    model: Optional[BaseEstimator] = None,
    n_samples: int = 10,
    cv_folds: int = 5,
    include_importance: bool = True,
) -> str:
    """
    Generate a comprehensive dataset summary with feature importance analysis.
    
    Creates a detailed summary suitable for LLM-based feature engineering,
    including dataset characteristics, feature types, and importance scores.
    
    Args:
        df_train (pd.DataFrame): Training dataset with features and target
        target_name (str): Name of the target column
        df_test (Optional[pd.DataFrame]): Test dataset (optional)
        model (Optional[BaseEstimator]): Model for importance calculation (optional)
        n_samples (int): Number of sample rows to include. Defaults to 10.
        cv_folds (int): Number of CV folds for importance calculation. Defaults to 5.
        include_importance (bool): Whether to calculate feature importance. Defaults to True.
        
    Returns:
        str: Comprehensive dataset summary formatted for LLM consumption
        
    Example:
        >>> df = get_dummy_data(1000)
        >>> summary = get_dataset_summary_with_importance(df, 'claim')
        >>> print(summary[:200])
        Dataset Summary Report
        ======================
        
        Dataset Information:
        - Training set shape: 1000 rows × 24 columns
        ...
    """
    
    lines = [
        "Dataset Summary Report",
        "=" * 50,
        ""
    ]
    
    # Basic dataset information
    lines.extend([
        "Dataset Information:",
        f"- Training set shape: {df_train.shape[0]:,} rows × {df_train.shape[1]} columns",
    ])
    
    if df_test is not None:
        lines.append(f"- Test set shape: {df_test.shape[0]:,} rows × {df_test.shape[1]} columns")
    
    lines.extend([
        f"- Target variable: {target_name}",
        f"- Feature columns: {df_train.shape[1] - 1}",
        ""
    ])
    
    # Feature columns (excluding target)
    feature_cols = [col for col in df_train.columns if col != target_name]
    lines.extend([
        "Feature Columns:",
        ", ".join(feature_cols),
        ""
    ])
    
    # Data types
    lines.extend([
        "Data Types:",
    ])
    for col, dtype in df_train.dtypes.items():
        lines.append(f"- {col}: {dtype}")
    lines.append("")
    
    # Missing values
    missing_summary = df_train.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if len(missing_cols) > 0:
        lines.extend([
            "Missing Values:",
        ])
        for col, count in missing_cols.items():
            pct = (count / len(df_train) * 100)
            lines.append(f"- {col}: {count} ({pct:.1f}%)")
    else:
        lines.append("Missing Values: None")
    lines.append("")
    
    # Target distribution
    target_counts = df_train[target_name].value_counts()
    lines.extend([
        "Target Distribution:",
    ])
    for value, count in target_counts.items():
        pct = (count / len(df_train) * 100)
        lines.append(f"- {value}: {count} ({pct:.1f}%)")
    lines.append("")
    
    # Feature importance if requested and possible
    if include_importance and model is not None:
        try:
            def prepare_modeling_data(df):
                """Prepare data for modeling by handling categorical variables."""
                df_processed = df.copy()
                
                # Convert categorical columns to numeric
                for col in df_processed.select_dtypes(include=['object', 'category']).columns:
                    if col != target_name:
                        df_processed[col] = pd.factorize(df_processed[col])[0]
                
                # Fill missing values
                df_processed = df_processed.fillna(-999)
                
                return df_processed
            
            # Prepare data
            df_processed = prepare_modeling_data(df_train)
            X = df_processed.drop(columns=[target_name])
            y = df_processed[target_name]
            
            # Calculate feature importance using cross-validation
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                model.fit(X, y)
                importance_scores = model.feature_importances_
                
                feature_importance = list(zip(X.columns, importance_scores))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                lines.extend([
                    "Feature Importance (based on model):",
                    f"- CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})",
                    "- Top features:"
                ])
                
                for feature, importance in feature_importance[:10]:
                    lines.append(f"  • {feature}: {importance:.3f}")
            
            else:
                # Use mutual information for other models
                mi_scores = mutual_info_classif(X, y, random_state=42)
                feature_importance = list(zip(X.columns, mi_scores))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                lines.extend([
                    "Feature Importance (mutual information):",
                    "- Top features:"
                ])
                
                for feature, importance in feature_importance[:10]:
                    lines.append(f"  • {feature}: {importance:.3f}")
                    
        except Exception as e:
            lines.extend([
                "Feature Importance: Could not calculate",
                f"- Error: {str(e)}"
            ])
        
        lines.append("")
    
    # Sample data
    lines.extend([
        f"Sample Data (first {n_samples} rows):",
        df_train.head(n_samples).to_string(),
        ""
    ])
    
    # Basic statistics for numeric columns
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        lines.extend([
            "Summary Statistics (numeric columns):",
            df_train[numeric_cols].describe().to_string(),
            ""
        ])
    
    return "\n".join(lines)


def to_code_markdown(text: str) -> str:
    """
    Convert plain text to markdown-formatted code block.
    
    Args:
        text (str): Plain text to format
        
    Returns:
        str: Text wrapped in markdown code fences
        
    Example:
        >>> code = "df['new_feature'] = df['old_feature'] * 2"
        >>> markdown = to_code_markdown(code)
        >>> print(markdown)
        ```python
        df['new_feature'] = df['old_feature'] * 2
        ```
    """
    return f"```python\n{text}\n```"


def get_X_y(df_train: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features (X) and target (y).
    
    Args:
        df_train (pd.DataFrame): Dataset containing features and target
        target_name (str): Name of the target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target as separate objects
        
    Example:
        >>> df = get_dummy_data(100)
        >>> X, y = get_X_y(df, 'claim')
        >>> print(X.shape, y.shape)
        (100, 23) (100,)
    """
    X = df_train.drop(columns=[target_name])
    y = df_train[target_name]
    return X, y


def make_df_numeric(
    df: pd.DataFrame,
    encoder: Optional[OrdinalEncoder] = None,
    categorical_cols: Optional[List[str]] = None,
    target_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert categorical columns in a DataFrame to numeric values.
    
    This function can work in two modes:
    1. With a pre-fitted encoder: Uses the encoder to transform the data
    2. Without an encoder: Creates a new encoder and fits it on the data
    
    This function automatically handles:
    - Pandas categorical columns
    - Object/string columns
    - Missing values
    - Unknown categories
    
    Args:
        df (pd.DataFrame): Input DataFrame to process
        encoder (Optional[OrdinalEncoder]): Pre-fitted encoder to use for transformation
        categorical_cols (Optional[List[str]]): Specific columns to encode
        target_column (Optional[str]): Target column name to exclude from encoding
        
    Returns:
        pd.DataFrame: Processed DataFrame with categorical columns converted to numeric
        
    Example:
        >>> df = pd.DataFrame({'cat': ['A', 'B', 'C'], 'num': [1, 2, 3]})
        >>> df_processed = make_df_numeric(df)
        >>> print(df_processed['cat'].values)
        [0 1 2]
    """
    # Create a copy to avoid modifying original dataframe
    df_processed = df.copy()
    
    # Identify categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
    
    if not categorical_cols:
        return df_processed
    
    # Initialize encoder if not provided
    if encoder is None:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=np.int32
        )
    
    # Process categorical columns
    for col in categorical_cols:
        # Handle pandas categorical columns
        if pd.api.types.is_categorical_dtype(df_processed[col]):
            # Add 'missing' category if not present
            if 'missing' not in df_processed[col].cat.categories:
                df_processed[col] = df_processed[col].cat.add_categories('missing')
            # Fill missing values
            df_processed[col] = df_processed[col].fillna('missing')
        
        # Handle object/string columns
        else:
            # Convert to string and fill missing values
            df_processed[col] = df_processed[col].astype(str).fillna('missing')
    
    # Fit encoder if not pre-fitted
    if not hasattr(encoder, 'categories_'):
        encoder.fit(df_processed[categorical_cols])
    
    # Transform the data
    df_processed[categorical_cols] = encoder.transform(df_processed[categorical_cols])
    
    return df_processed


def make_dataset_numeric(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    target_column: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    return_encoder: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]]:
    """
    Convert categorical columns in train/test datasets to numeric values.
    
    This function automatically handles:
    - Pandas categorical columns
    - Object/string columns
    - Missing values
    - Unknown categories in test set
    
    Args:
        df_train (pd.DataFrame): Training dataset
        df_test (Optional[pd.DataFrame]): Test dataset (optional)
        target_column (Optional[str]): Target column name to exclude from encoding
        categorical_cols (Optional[List[str]]): Specific columns to encode
        return_encoder (bool): Whether to return the fitted encoder
        
    Returns:
        Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]]:
            Processed training and test datasets, optionally with encoder
            
    Example:
        >>> train = pd.DataFrame({'cat': ['A', 'B'], 'num': [1, 2], 'target': [0, 1]})
        >>> test = pd.DataFrame({'cat': ['A', 'C'], 'num': [3, 4], 'target': [1, 0]})
        >>> train_proc, test_proc = make_dataset_numeric(train, test, 'target')
        >>> print(train_proc['cat'].values, test_proc['cat'].values)
        [0 1] [0 2]
    """
    # Create copies to avoid modifying original dataframes
    df_train_processed = df_train.copy()
    df_test_processed = df_test.copy() if df_test is not None else None
    
    # Identify categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = df_train_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
    
    if not categorical_cols:
        if return_encoder:
            return df_train_processed, df_test_processed, None
        return df_train_processed, df_test_processed
    
    # Initialize encoder
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        dtype=np.int32
    )
    
    # Process categorical columns
    for col in categorical_cols:
        # Handle pandas categorical columns
        if pd.api.types.is_categorical_dtype(df_train_processed[col]):
            # Add 'missing' category if not present
            if 'missing' not in df_train_processed[col].cat.categories:
                df_train_processed[col] = df_train_processed[col].cat.add_categories('missing')
            # Fill missing values
            df_train_processed[col] = df_train_processed[col].fillna('missing')
            
            if df_test_processed is not None:
                # Ensure test set has same categories as train set
                test_categories = set(df_test_processed[col].cat.categories)
                train_categories = set(df_train_processed[col].cat.categories)
                missing_categories = train_categories - test_categories
                
                if missing_categories:
                    df_test_processed[col] = df_test_processed[col].cat.add_categories(list(missing_categories))
                df_test_processed[col] = df_test_processed[col].fillna('missing')
        
        # Handle object/string columns
        else:
            # Convert to string and fill missing values
            df_train_processed[col] = df_train_processed[col].astype(str).fillna('missing')
            if df_test_processed is not None:
                df_test_processed[col] = df_test_processed[col].astype(str).fillna('missing')
    
    # Fit encoder on training data
    encoder.fit(df_train_processed[categorical_cols])
    
    # Transform both datasets
    df_train_processed[categorical_cols] = encoder.transform(df_train_processed[categorical_cols])
    if df_test_processed is not None:
        df_test_processed[categorical_cols] = encoder.transform(df_test_processed[categorical_cols])
    
    if return_encoder:
        return df_train_processed, df_test_processed, encoder
    return df_train_processed, df_test_processed


def auc_metric(target: Union[np.ndarray, pd.Series], pred: np.ndarray, multi_class: str = "ovo", numpy: bool = False) -> float:
    """
    Calculate Area Under the ROC Curve (AUC) for binary or multiclass classification.
    
    Args:
        target: True binary labels or multiclass labels
        pred: Predicted probabilities (2D array for binary, 2D+ for multiclass)
        multi_class: Strategy for multiclass AUC calculation ('ovo' or 'ovr')
        numpy: Whether to return as numpy scalar (unused, kept for compatibility)
        
    Returns:
        float: AUC score
        
    Example:
        >>> target = [0, 1, 1, 0]
        >>> pred = [[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.8, 0.2]]
        >>> score = auc_metric(target, pred)
        >>> print(f"{score:.3f}")
        1.000
    """
    pred = np.array(pred)
    
    # Handle binary classification
    if pred.ndim == 2 and pred.shape[1] == 2:
        pred_proba = pred[:, 1]  # Use positive class probabilities
    elif pred.ndim == 1:
        pred_proba = pred
    else:
        # Multiclass case
        return roc_auc_score(target, pred, multi_class=multi_class)
    
    return roc_auc_score(target, pred_proba)


def accuracy_metric(target: Union[np.ndarray, pd.Series], pred: np.ndarray) -> float:
    """
    Calculate accuracy from predicted probabilities.
    
    Args:
        target: True binary labels
        pred: Predicted probabilities (2D array for binary classification)
        
    Returns:
        float: Accuracy score
        
    Example:
        >>> target = [0, 1, 1, 0]
        >>> pred = [[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.8, 0.2]]
        >>> score = accuracy_metric(target, pred)
        >>> print(f"{score:.3f}")
        1.000
    """
    pred = np.array(pred)
    
    # Convert probabilities to predicted classes
    if pred.ndim == 2 and pred.shape[1] == 2:
        pred_classes = np.argmax(pred, axis=1)
    elif pred.ndim == 1:
        pred_classes = (pred > 0.5).astype(int)
    else:
        pred_classes = np.argmax(pred, axis=1)
    
    return accuracy_score(target, pred_classes)


def metric_ppv(
    y_true: Union[list, pd.Series], 
    y_pred: Union[list, pd.Series], 
    top_p: float
) -> float:
    """
    Computes PPV (Positive Predictive Value) at the top p% predicted probability scores.

    This metric calculates precision among the samples with the highest predicted probabilities,
    which is useful for scenarios where you care about precision in your most confident predictions.

    Args:
        y_true (Union[list, pd.Series]): Ground truth binary labels (0 or 1).
        y_pred (Union[list, pd.Series]): Predicted probabilities (not hard labels).
        top_p (float): Fraction (0 < top_p <= 1) of samples to include in the top predictions.

    Returns:
        float: Precision/PPV in the top_p most confident predictions.

    Raises:
        ValueError: If top_p is not between 0 and 1, or if y_true and y_pred have different lengths.

    Example:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0.1, 0.9, 0.8, 0.2, 0.7]
        >>> metric_ppv(y_true, y_pred, top_p=0.4)  # Top 40% (2 samples)
        1.0  # Both top predictions were correct positives
    """
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be between 0 and 1.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")

    top_num = max(1, math.ceil(len(y_true) * top_p))

    ranked = pd.DataFrame({
        "label": pd.Series(y_true).values,
        "predicted_prob": pd.Series(y_pred).values,
    })

    top_ranked = ranked.sort_values("predicted_prob", ascending=False).head(top_num)
    ppv = top_ranked["label"].value_counts(normalize=True).get(1, 0.0)

    return ppv