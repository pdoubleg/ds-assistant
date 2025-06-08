from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from datamodels import AutoMLDependencies



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

