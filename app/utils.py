"""
Utility functions for the AutoML Streamlit Demo App

This module provides self-contained utility functions for data processing,
analysis, and visualization without external dependencies on the main codebase.
"""

import logging
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


def exploratory_data_analysis(
    df: pd.DataFrame, target: str, n_sample: int = 10
) -> Dict[str, Any]:
    """
    Perform comprehensive exploratory data analysis on a dataset.
    
    Analyzes dataset characteristics, feature types, missing values,
    correlations, and target distribution to provide insights for
    automated feature engineering and model selection.
    
    Args:
        df (pd.DataFrame): Input dataset including features and target
        target (str): Name of the target column
        n_sample (int): Number of sample rows to include. Defaults to 10.
        
    Returns:
        Dict[str, Any]: Comprehensive analysis results including:
            - basic_info: Dataset shape, memory usage, dtypes
            - missing_values: Missing value statistics
            - feature_types: Categorization of features (numeric, binary, categorical)
            - target_analysis: Target distribution and class balance
            - correlations: Feature correlations with target
            - sample_data: Sample rows from the dataset
            - summary_stats: Descriptive statistics
            
    Example:
        >>> df = get_dummy_data(1000)
        >>> eda = exploratory_data_analysis(df, 'claim')
        >>> print(eda['basic_info']['shape'])
        (1000, 24)
    """
    
    results = {}
    
    # Basic dataset information
    results['basic_info'] = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'dtypes': df.dtypes.to_dict(),
        'column_names': df.columns.tolist()
    }
    
    # Missing values analysis
    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / len(df) * 100).round(2)
    results['missing_values'] = {
        'missing_counts': missing_stats.to_dict(),
        'missing_percentages': missing_pct.to_dict(),
        'columns_with_missing': missing_stats[missing_stats > 0].index.tolist()
    }
    
    # Feature type detection
    feature_types = _get_variable_types(df.drop(columns=[target]))
    results['feature_types'] = feature_types
    
    # Target analysis
    target_counts = df[target].value_counts()
    target_dist = (target_counts / len(df) * 100).round(2)
    results['target_analysis'] = {
        'class_counts': target_counts.to_dict(),
        'class_distribution_pct': target_dist.to_dict(),
        'is_balanced': all(dist >= 30 for dist in target_dist.values),
        'unique_values': df[target].nunique(),
        'target_type': 'binary' if df[target].nunique() == 2 else 'multiclass'
    }
    
    # Correlation analysis (numeric features only)
    numeric_cols = feature_types.get('numeric', [])
    if numeric_cols and target in df.select_dtypes(include=[np.number]).columns:
        correlations = df[numeric_cols + [target]].corr()[target].drop(target)
        results['correlations'] = {
            'target_correlations': correlations.to_dict(),
            'high_correlation_features': correlations[abs(correlations) > 0.5].index.tolist(),
            'low_correlation_features': correlations[abs(correlations) < 0.1].index.tolist()
        }
    else:
        results['correlations'] = {'target_correlations': {}, 'note': 'Limited correlation analysis (non-numeric target or no numeric features)'}
    
    # Sample data
    results['sample_data'] = df.head(n_sample).to_dict('records')
    
    # Summary statistics
    results['summary_stats'] = df.describe().to_dict()
    
    # Feature importance (quick mutual information)
    if len(feature_types.get('numeric', [])) > 0:
        try:
            X_numeric = df[feature_types['numeric']].fillna(-999)
            if df[target].dtype in ['object', 'category']:
                y_encoded = pd.factorize(df[target])[0]
            else:
                y_encoded = df[target]
            
            mi_scores = mutual_info_classif(X_numeric, y_encoded, random_state=42)
            mi_dict = dict(zip(feature_types['numeric'], mi_scores))
            results['feature_importance'] = {
                'mutual_info_scores': mi_dict,
                'top_features': sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        except Exception as e:
            results['feature_importance'] = {'error': str(e)}
    
    return results


def _get_variable_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect and categorize variable types in a DataFrame.
    
    Categorizes features into:
    - numeric: Continuous numerical variables
    - binary: Variables with exactly 2 unique values
    - categorical: Non-numeric variables with multiple categories
    - high_cardinality: Categorical variables with many unique values
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping variable types to column names
        
    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'is_student': [1, 0, 1],
        ...     'city': ['NYC', 'LA', 'Chicago']
        ... })
        >>> types = _get_variable_types(df)
        >>> print(types)
        {'numeric': ['age'], 'binary': ['is_student'], 'categorical': ['city']}
    """
    
    variable_types = {
        'numeric': [],
        'binary': [],
        'categorical': [],
        'high_cardinality': []
    }
    
    for col in df.columns:
        unique_count = df[col].nunique()
        
        # Binary variables (exactly 2 unique values)
        if unique_count == 2:
            variable_types['binary'].append(col)
        # Numeric variables with more than 2 unique values
        elif pd.api.types.is_numeric_dtype(df[col]) and unique_count > 2:
            # Check if it's truly continuous (not just integer IDs)
            if unique_count > 0.1 * len(df) or df[col].dtype == 'float':
                variable_types['numeric'].append(col)
            else:
                variable_types['categorical'].append(col)
        # Categorical variables
        else:
            # High cardinality threshold
            if unique_count > 50:
                variable_types['high_cardinality'].append(col)
            else:
                variable_types['categorical'].append(col)
    
    return variable_types


def format_eda_for_llm(
    eda: Dict[str, Any],
    include_sample_data: bool = True,
    include_detailed_info: bool = True,
) -> str:
    """
    Format exploratory data analysis results for LLM consumption.
    
    Converts the structured EDA results into a comprehensive text summary
    that provides context for LLM-based feature engineering and modeling decisions.
    
    Args:
        eda (Dict[str, Any]): Results from exploratory_data_analysis()
        include_sample_data (bool): Whether to include sample data rows. Defaults to True.
        include_detailed_info (bool): Whether to include detailed statistics. Defaults to True.
        
    Returns:
        str: Formatted text summary suitable for LLM prompts
        
    Example:
        >>> df = get_dummy_data(1000)
        >>> eda = exploratory_data_analysis(df, 'claim')
        >>> summary = format_eda_for_llm(eda)
        >>> print(summary[:200])
        Dataset Analysis Summary:
        ========================
        
        Basic Information:
        - Dataset Shape: 1000 rows × 24 columns
        - Memory Usage: 0.18 MB
        ...
    """
    
    lines = [
        "Dataset Analysis Summary:",
        "=" * 50,
        ""
    ]
    
    # Basic Information
    basic_info = eda.get('basic_info', {})
    lines.extend([
        "Basic Information:",
        f"- Dataset Shape: {basic_info.get('shape', 'N/A')[0]:,} rows × {basic_info.get('shape', 'N/A')[1]} columns",
        f"- Memory Usage: {basic_info.get('memory_usage_mb', 0):.2f} MB",
        f"- Total Columns: {len(basic_info.get('column_names', []))}",
        ""
    ])
    
    # Missing Values
    missing_info = eda.get('missing_values', {})
    columns_with_missing = missing_info.get('columns_with_missing', [])
    if columns_with_missing:
        lines.extend([
            "Missing Values:",
            f"- Columns with missing values: {len(columns_with_missing)}",
        ])
        for col in columns_with_missing[:10]:  # Limit to first 10
            pct = missing_info.get('missing_percentages', {}).get(col, 0)
            lines.append(f"  • {col}: {pct}%")
        if len(columns_with_missing) > 10:
            lines.append(f"  • ... and {len(columns_with_missing) - 10} more columns")
    else:
        lines.append("Missing Values: None detected")
    lines.append("")
    
    # Feature Types
    feature_types = eda.get('feature_types', {})
    lines.extend([
        "Feature Types:",
        f"- Numeric features: {len(feature_types.get('numeric', []))} ({', '.join(feature_types.get('numeric', [])[:5])}{'...' if len(feature_types.get('numeric', [])) > 5 else ''})",
        f"- Binary features: {len(feature_types.get('binary', []))} ({', '.join(feature_types.get('binary', [])[:5])}{'...' if len(feature_types.get('binary', [])) > 5 else ''})",
        f"- Categorical features: {len(feature_types.get('categorical', []))} ({', '.join(feature_types.get('categorical', [])[:5])}{'...' if len(feature_types.get('categorical', [])) > 5 else ''})",
        f"- High cardinality features: {len(feature_types.get('high_cardinality', []))} ({', '.join(feature_types.get('high_cardinality', [])[:3])}{'...' if len(feature_types.get('high_cardinality', [])) > 3 else ''})",
        ""
    ])
    
    # Target Analysis
    target_analysis = eda.get('target_analysis', {})
    lines.extend([
        "Target Variable Analysis:",
        f"- Type: {target_analysis.get('target_type', 'unknown')}",
        f"- Unique values: {target_analysis.get('unique_values', 'N/A')}",
        f"- Class balance: {'Balanced' if target_analysis.get('is_balanced', False) else 'Imbalanced'}",
    ])
    
    class_dist = target_analysis.get('class_distribution_pct', {})
    if class_dist:
        lines.append("- Class distribution:")
        for class_val, pct in class_dist.items():
            lines.append(f"  • Class {class_val}: {pct}%")
    lines.append("")
    
    # Correlations
    correlations = eda.get('correlations', {})
    if 'target_correlations' in correlations and correlations['target_correlations']:
        lines.extend([
            "Feature-Target Correlations:",
            f"- High correlation features (|r| > 0.5): {len(correlations.get('high_correlation_features', []))}",
            f"- Low correlation features (|r| < 0.1): {len(correlations.get('low_correlation_features', []))}",
        ])
        
        # Show top correlations
        target_corrs = correlations['target_correlations']
        sorted_corrs = sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        if sorted_corrs:
            lines.append("- Top correlations:")
            for feature, corr in sorted_corrs[:5]:
                lines.append(f"  • {feature}: {corr:.3f}")
        lines.append("")
    
    # Feature Importance
    if 'feature_importance' in eda and 'top_features' in eda['feature_importance']:
        lines.extend([
            "Feature Importance (Mutual Information):",
        ])
        for feature, score in eda['feature_importance']['top_features'][:5]:
            lines.append(f"- {feature}: {score:.3f}")
        lines.append("")
    
    # Sample Data
    if include_sample_data and 'sample_data' in eda:
        lines.extend([
            "Sample Data (first few rows):",
            str(pd.DataFrame(eda['sample_data']).to_string()),
            ""
        ])
    
    # Detailed Statistics
    if include_detailed_info and 'summary_stats' in eda:
        lines.extend([
            "Summary Statistics:",
            str(pd.DataFrame(eda['summary_stats']).to_string()),
            ""
        ])
    
    return "\n".join(lines)


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