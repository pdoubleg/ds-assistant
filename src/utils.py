import io
import json
import logging
import math
import re
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


def get_dummy_data(n_samples: int = 10000) -> pd.DataFrame:
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
        weights=[0.9, 0.15],
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
    df["region"] = np.random.choice(["North", "South", "East", "West"], size=len(df))
    df["policy_type"] = np.random.choice(["Basic", "Standard", "Premium"], size=len(df))
    df["vehicle_type"] = np.random.choice(["Sedan", "SUV", "Truck"], size=len(df))

    # Add missing values to selected columns
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


def extract_python_code_blocks(markdown: str) -> list[str]:
    """
    Extract all Python code blocks from a markdown string.

    Parameters:
    - markdown (str): Markdown-formatted text with fenced code blocks.

    Returns:
    - List of strings, each containing code from one fenced block.
    """
    pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(pattern, markdown, re.DOTALL)
    return [match.strip() for match in matches]


def metrics_display(
    y_test: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_pred_proba: list | np.ndarray,
) -> None:
    """Display various classification metrics and confusion matrix visualization.

    Args:
        y_test: Ground truth (correct) target values
        y_pred: Estimated targets as returned by a classifier
        y_pred_proba: Probability estimates for samples

    Returns:
        None. Prints metrics and displays confusion matrix plot.

    Example:
        >>> y_test = [0, 1, 1, 0]
        >>> y_pred = [0, 1, 0, 0]
        >>> y_pred_proba = [0.2, 0.7, 0.4, 0.3]
        >>> metrics_display(y_test, y_pred, y_pred_proba)
        ROC_AUC score: 0.750
        f1 score: 0.667
        Accuracy: 75.00%
        ...
    """

    # Obtain confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Output classification metrics
    tn, fp, fn, tp = cm.ravel()

    print(f"ROC_AUC score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print(f"f1 score: {f1_score(y_test, y_pred):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Detection rate: {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"False alarm rate: {fp / (tn + fp) * 100}%")
    print(f"MCC: {matthews_corrcoef(y_test, y_pred):.2f}")

    # Display confusion matrix
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, values_format='.5g', colorbar=False)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


def data_report(
    df: pd.DataFrame, n_sample: int = 10, include_stats: bool = True
) -> str:
    """Generate a detailed report of dataset characteristics including class distribution and feature analysis.

    Automatically identifies feature types:
    - Numerical: Features with numeric dtype and more than 2 unique values
    - Binary: Features with exactly 2 unique values (regardless of dtype)
    - Nominal: Non-numeric features with more than 2 unique values

    Args:
        df: Input DataFrame containing features and target variable (target assumed to be last column)
        n_sample: Number of sample rows to display in the report. Defaults to 10.
        include_stats: Whether to include descriptive statistics and detailed info. Defaults to True.

    Returns:
        str: Formatted text report containing dataset statistics and characteristics

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35, 40],
        ...     'is_student': [1, 0, 0, 1],
        ...     'city': ['NYC', 'LA', 'NYC', 'Chicago'],
        ...     'target': [0, 1, 1, 0]
        ... })
        >>> report = data_report(df)
        >>> print(report)
        Data Characteristics Report:
        ...
    """

    # Last column is the label
    target = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # General dataset info
    num_instances = len(df)
    num_features = features.shape[1]

    # Automatically identify feature types
    num_feats = []  # Numerical features
    bin_feats = []  # Binary features
    nom_feats = []  # Nominal/categorical features

    for col in features.columns:
        unique_values = features[col].nunique()

        # Binary features: exactly 2 unique values
        if unique_values == 2:
            bin_feats.append(col)
        # Numerical features: numeric dtype and more than 2 unique values
        elif pd.api.types.is_numeric_dtype(features[col]) and unique_values > 2:
            num_feats.append(col)
        # Nominal features: non-numeric or numeric with limited unique values
        else:
            nom_feats.append(col)

    # Label class analysis
    class_counts = target.value_counts()
    class_distribution = class_counts / num_instances
    if any(class_distribution < 0.3) or any(class_distribution > 0.7):
        class_imbalance = True
    else:
        class_imbalance = False

    # Missing value analysis
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n  ".join(
        [f"  {col}: {val:.2f}%" for col, val in missing_stats.items() if val > 0]
    )
    if not missing_summary:
        missing_summary = "  No missing values detected"

    # Column data types
    column_types = "\n  ".join(
        [f"  {col}: {dtype}" for col, dtype in df.dtypes.items()]
    )

    # Unique value counts
    unique_counts = df.nunique()
    unique_counts_summary = "\n  ".join(
        [f"  {col}: {count}" for col, count in unique_counts.items()]
    )

    # Memory usage info
    memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # Convert to MB

    # Create a text report
    report = f"""Data Characteristics Report:

- General information:
  - Number of Instances: {num_instances}
  - Number of Features: {num_features}
  - Memory Usage: {memory_usage:.2f} MB
  - Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns

- Data Types:
{column_types}

- Missing Value Analysis:
{missing_summary}

- Unique Value Counts:
{unique_counts_summary}

- Class distribution analysis:
  - Class Distribution: {class_distribution.to_string()}
  {"- Warning: Class imbalance detected." if class_imbalance else "- Class distribution appears balanced."}

- Feature analysis:
  - Feature names: {features.columns.to_list()}
  - Number of numerical features: {len(num_feats)}
  - Numerical feature names: {num_feats}
  - Number of binary features: {len(bin_feats)}
  - Binary feature names: {bin_feats}
  - Number of nominal features: {len(nom_feats)}
  - Nominal feature names: {nom_feats}

- Sample Data (first {n_sample} rows):
{df.head(n_sample).to_string()}"""

    # Add detailed statistics if requested
    if include_stats:
        # Capture df.info() output
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()

        report += f"""

- Descriptive Statistics:
{df.describe().to_string()}

- Detailed DataFrame Info:
{info_text}"""

    return report


def simplify_logs(log_data: dict) -> list[dict]:
    """
    Simplifies log data by extracting essential information from each log entry.

    Args:
        log_data (dict): A dictionary containing a 'log' key, which is a list of JSON-formatted log strings.

    Returns:
        list[dict]: A list of dictionaries, each containing 'record_id', 'validation_loss', and 'config' for each log entry
                    that does not contain the key 'curr_best_record_id'.

    Example:
        >>> log_data = {
        ...     'log': [
        ...         '{"record_id": 1, "validation_loss": 0.2, "config": {"lr": 0.01}}',
        ...         '{"record_id": 2, "validation_loss": 0.15, "config": {"lr": 0.02}, "curr_best_record_id": 2}'
        ...     ]
        ... }
        >>> simplify_logs(log_data)
        [{'record_id': 1, 'validation_loss': 0.2, 'config': {'lr': 0.01}}]
    """
    simplified_logs: list[dict] = []

    # Go through each log entry and extract essential information
    for log in log_data["log"]:
        # Only process logs that do not contain 'curr_best_record_id'
        if "curr_best_record_id" not in log:
            log_json = json.loads(log)
            simplified_entry = {
                "record_id": log_json["record_id"],
                "validation_loss": log_json["validation_loss"],
                "config": log_json["config"],
            }
            simplified_logs.append(simplified_entry)

    return simplified_logs


def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    n_sample: int = 20,
    skip_stats: bool = False,
) -> List[str]:
    """
    Generate a summary for one or more DataFrames. Accepts a single DataFrame, a list of DataFrames,
    or a dictionary mapping names to DataFrames.

    Parameters
    ----------
    dataframes : pandas.DataFrame or list of pandas.DataFrame or dict of (str -> pandas.DataFrame)
        - Single DataFrame: produce a single summary (returned within a one-element list).
        - List of DataFrames: produce a summary for each DataFrame, using index-based names.
        - Dictionary of DataFrames: produce a summary for each DataFrame, using dictionary keys as names.
    n_sample : int, default 30
        Number of rows to display in the "Data (first 20 rows)" section.
    skip_stats : bool, default False
        If True, skip the descriptive statistics and DataFrame info sections.

    Example:
    --------
    ``` python
    import pandas as pd
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    dataframes = {
        "iris": data.frame,
        "iris_target": data.target,
    }
    summaries = get_dataframe_summary(dataframes)
    print(summaries[0])
    ```

    Returns
    -------
    list of str
        A list of summaries, one for each provided DataFrame. Each summary includes:
        - Shape of the DataFrame (rows, columns)
        - Column data types
        - Missing value percentage
        - Unique value counts
        - First 30 rows
        - Descriptive statistics
        - DataFrame info output
    """

    summaries = []

    # --- Dictionary Case ---
    if isinstance(dataframes, dict):
        for dataset_name, df in dataframes.items():
            summaries.append(
                _summarize_dataframe(df, dataset_name, n_sample, skip_stats)
            )

    # --- Single DataFrame Case ---
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(
            _summarize_dataframe(dataframes, "Single_Dataset", n_sample, skip_stats)
        )

    # --- List of DataFrames Case ---
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(
                _summarize_dataframe(df, dataset_name, n_sample, skip_stats)
            )

    else:
        raise TypeError(
            "Input must be a single DataFrame, a list of DataFrames, or a dictionary of DataFrames."
        )

    return summaries


def _summarize_dataframe(
    df: pd.DataFrame, dataset_name: str, n_sample=20, skip_stats=False
) -> str:
    """Generate a summary string for a single DataFrame."""
    # 1. Convert dictionary-type cells to strings
    #    This prevents unhashable dict errors during df.nunique().
    df = df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, dict) else x))

    # 2. Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    # 3. Calculate missing value stats
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join(
        [f"{col}: {val:.2f}%" for col, val in missing_stats.items()]
    )

    # 4. Get column data types
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])

    # 5. Get unique value counts
    unique_counts = df.nunique()  # Will no longer fail on unhashable dict
    unique_counts_summary = "\n".join(
        [f"{col}: {count}" for col, count in unique_counts.items()]
    )

    # 6. Generate the summary text
    if not skip_stats:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}

        Unique Value Counts:
        {unique_counts_summary}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}

        Data Description:
        {df.describe().to_string()}

        Data Info:
        {info_text}
        """
    else:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}
        """

    return summary_text.strip()


def _detect_binary_variables(df: pd.DataFrame) -> List[str]:
    """
    Detect binary variables in a DataFrame, regardless of their data type.

    Binary variables are detected as columns with exactly 2 unique non-null values.
    Common patterns include: 0/1, True/False, Yes/No, Male/Female, etc.

    Args:
        df: Input DataFrame

    Returns:
        List of column names that are binary variables

    Example:
        >>> df = pd.DataFrame({
        ...     'binary_numeric': [0, 1, 0, 1],
        ...     'binary_bool': [True, False, True, False],
        ...     'binary_string': ['Yes', 'No', 'Yes', 'No'],
        ...     'numeric': [1.5, 2.3, 4.1, 5.7],
        ...     'categorical': ['A', 'B', 'C', 'A']
        ... })
        >>> _detect_binary_variables(df)
        ['binary_numeric', 'binary_bool', 'binary_string']
    """
    binary_cols = []
    for col in df.columns:
        # Check if column has exactly 2 unique non-null values
        unique_count = df[col].nunique()
        if unique_count == 2:
            binary_cols.append(col)
    return binary_cols


def _get_variable_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify DataFrame columns into numeric, binary, and categorical variables.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with keys 'numeric', 'binary', 'categorical' and lists of column names

    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'is_student': [0, 1, 0],
        ...     'city': ['NYC', 'LA', 'Chicago']
        ... })
        >>> _get_variable_types(df)
        {'numeric': ['age'], 'binary': ['is_student'], 'categorical': ['city']}
    """
    # First detect binary variables
    binary_cols = _detect_binary_variables(df)

    # Get potential numeric columns (excluding binary ones)
    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in binary_cols
    ]

    # Get categorical columns (non-numeric and non-binary)
    categorical_cols = [
        col for col in df.columns if col not in numeric_cols and col not in binary_cols
    ]

    return {
        "numeric": numeric_cols,
        "binary": binary_cols,
        "categorical": categorical_cols,
    }


def exploratory_data_analysis(
    df: pd.DataFrame, target: str, n_sample: int = 10
) -> Dict[str, Any]:
    """
    Performs extended exploratory data analysis on a dataset, including:
      - Basic shape/memory/dtypes/missing
      - Numeric, binary, and categorical stats
      - Correlation analysis
      - **New**:
        * target distribution or summary
        * feature–target relationships (corr or mutual info)
        * outlier counts (>3σ) & percentages for numeric only
        * range & IQR for numeric features
        * categorical cardinality & ratios
        * binary feature analysis
        * feature-to-sample ratio
      - **Enhanced**:
        * sample data capture
        * comprehensive unique value analysis
        * detailed DataFrame info
        * memory usage breakdown

    Args:
        df: Input DataFrame to analyze
        target: Name of the target column
        n_sample: Number of sample rows to capture for analysis. Defaults to 10.

    Returns:
        Dict containing comprehensive EDA results

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> results = exploratory_data_analysis(df, 'target_column')
        >>> summary = format_eda_for_llm(results)
        >>> print(summary)
    """
    logger.info("Performing exploratory data analysis on dataset")
    if df is None or df.empty:
        raise ValueError("No data found or empty dataset")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    num_rows, num_cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # dtypes
    dtypes_count = df.dtypes.value_counts().to_dict()
    dtypes_by_column = {col: str(dt) for col, dt in df.dtypes.items()}

    # Capture DataFrame info
    buffer = io.StringIO()
    df.info(buf=buffer)
    dataframe_info = buffer.getvalue()

    # Comprehensive unique value analysis for all columns
    unique_analysis = {}
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / num_rows
        unique_analysis[col] = {
            "unique_count": int(unique_count),
            "unique_ratio": round(unique_ratio, 4),
            "is_unique": unique_count == num_rows,
            "is_constant": unique_count == 1,
            "is_binary": unique_count == 2,
        }

    # Sample data capture
    sample_data = {
        "first_rows": df.head(n_sample).to_dict("records"),
        "column_sample_values": {},
    }

    # For each column, capture a few sample values to understand the data better
    for col in df.columns:
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            sample_values = non_null_values.head(5).tolist()
            sample_data["column_sample_values"][col] = sample_values
        else:
            sample_data["column_sample_values"][col] = ["All NaN"]

    # missing
    missing_values = df.isnull().sum().to_dict()
    missing_pct = {
        col: round(100 * cnt / num_rows, 2)
        for col, cnt in missing_values.items()
        if cnt > 0
    }

    # Enhanced variable type classification
    var_types = _get_variable_types(df)
    numeric_cols = var_types["numeric"]
    binary_cols = var_types["binary"]
    categorical_cols = var_types["categorical"]

    # numeric stats
    numeric_stats: Dict[str, Any] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        skew = {
            c: float(df[c].skew()) for c in numeric_cols if not df[c].isnull().all()
        }
        kurt = {
            c: float(df[c].kurtosis()) for c in numeric_cols if not df[c].isnull().all()
        }
        numeric_stats = {"summary": desc, "skewness": skew, "kurtosis": kurt}

    # Binary variable analysis
    binary_stats: Dict[str, Any] = {}
    if binary_cols:
        binary_analysis = {}
        for c in binary_cols:
            vc = df[c].value_counts()
            values = vc.index.tolist()
            counts = vc.values.tolist()
            proportions = (vc / num_rows * 100).round(2)

            binary_analysis[c] = {
                "values": values,
                "counts": counts,
                "proportions": proportions.tolist(),
                "balance_ratio": min(proportions) / max(proportions)
                if max(proportions) > 0
                else 0,
                "is_balanced": min(proportions) >= 30.0,  # At least 30% for each class
            }
        binary_stats = binary_analysis

    # categorical stats
    categorical_stats: Dict[str, Any] = {}
    if categorical_cols:
        cat = {}
        for c in categorical_cols:
            vc = df[c].value_counts().to_dict()
            cat[c] = {
                "unique_count": int(df[c].nunique()),
                "top_values": dict(list(vc.items())[:10]),
            }
        categorical_stats = cat

    # correlation matrix & high correlations (numeric only)
    correlation_analysis: Dict[str, Any] = {}
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        high = []
        for i, f1 in enumerate(corr.columns):
            for f2 in corr.columns[i + 1 :]:
                val = corr.at[f1, f2]
                if abs(val) > 0.7:
                    high.append(
                        {"feature1": f1, "feature2": f2, "correlation": float(val)}
                    )
        # to dict, converting NaN→None
        m = corr.to_dict()
        for a in m:
            for b in m[a]:
                if pd.isna(m[a][b]):
                    m[a][b] = None
        correlation_analysis = {"high_correlations": high, "matrix": m}

    # data_quality_issues (updated for better binary handling)
    issues: list = []
    # high missing
    hm = [c for c, pct in missing_pct.items() if pct > 20]
    if hm:
        issues.append(
            {
                "issue_type": "high_missing_values",
                "description": f"Columns with >20% missing: {', '.join(hm)}",
            }
        )
    # high skew (numeric only)
    hs = [c for c, s in numeric_stats.get("skewness", {}).items() if abs(s) > 3]
    if hs:
        issues.append(
            {
                "issue_type": "high_skewness",
                "description": f"Highly skewed numeric features: {', '.join(hs)}",
            }
        )
    # high corr (numeric only)
    if correlation_analysis.get("high_correlations"):
        pairs = [
            f"{p['feature1']}/{p['feature2']}"
            for p in correlation_analysis["high_correlations"]
        ]
        issues.append(
            {
                "issue_type": "high_correlation",
                "description": f"Highly correlated numeric pairs: {', '.join(pairs)}",
            }
        )
    # imbalanced binary features
    imbalanced_binary = [
        c for c, stats in binary_stats.items() if not stats["is_balanced"]
    ]
    if imbalanced_binary:
        issues.append(
            {
                "issue_type": "imbalanced_binary_features",
                "description": f"Imbalanced binary features (<30% minority class): {', '.join(imbalanced_binary)}",
            }
        )

    # ==== ENHANCED TARGET METRICS ====

    # Determine target type more accurately
    target_is_binary = target in binary_cols
    target_is_numeric = target in numeric_cols
    target_is_categorical = target in categorical_cols

    if target_is_binary:
        vc = df[target].value_counts()
        proportions = (vc / num_rows * 100).round(2)
        target_stats = {
            "type": "binary",
            "values": vc.index.tolist(),
            "counts": vc.values.tolist(),
            "proportions": proportions.tolist(),
            "balance_ratio": min(proportions) / max(proportions)
            if max(proportions) > 0
            else 0,
            "is_balanced": min(proportions) >= 30.0,
        }
    elif target_is_numeric:
        tdesc = df[target].describe().to_dict()
        t_iqr = float(df[target].quantile(0.75) - df[target].quantile(0.25))
        target_stats = {"type": "numeric", "summary": tdesc, "IQR": t_iqr}
    else:
        vc = df[target].value_counts()
        target_stats = {
            "type": "categorical",
            "counts": vc.to_dict(),
            "percentages": (vc / num_rows * 100).round(2).to_dict(),
        }

    # 2. Enhanced Feature–target relationship
    if target_is_binary or target_is_categorical:
        # For classification targets: use mutual info for all feature types
        feature_target_scores = {}

        # Encode target for mutual info calculation
        if target_is_binary:
            y = pd.Categorical(df[target]).codes
        else:
            y = df[target].astype("category").cat.codes

        # Calculate mutual info for numeric features
        if numeric_cols:
            clean_numeric = df[numeric_cols].fillna(0)
            mi_numeric = mutual_info_classif(clean_numeric, y, discrete_features=False)
            feature_target_scores.update(dict(zip(numeric_cols, mi_numeric.tolist())))

        # Calculate mutual info for binary features (excluding target)
        binary_features = [c for c in binary_cols if c != target]
        if binary_features:
            # For binary features, we can use them as discrete
            clean_binary = df[binary_features].fillna(-1)
            # Convert to numeric codes for mutual info
            binary_encoded = clean_binary.apply(lambda x: pd.Categorical(x).codes)
            mi_binary = mutual_info_classif(binary_encoded, y, discrete_features=True)
            feature_target_scores.update(dict(zip(binary_features, mi_binary.tolist())))

        feature_target = {
            "method": "mutual_info_classification",
            "scores": feature_target_scores,
        }

    else:
        # For regression targets: use correlation for numeric, mutual info for others
        feature_target_scores = {}

        # Pearson correlation for numeric features
        if numeric_cols:
            corr_with_target = (
                df[numeric_cols + [target]].corr()[target].drop(target).abs().to_dict()
            )
            feature_target_scores.update(corr_with_target)

        # For binary features with numeric target, use point-biserial correlation
        if binary_cols:
            for col in binary_cols:
                if not df[col].isnull().all() and not df[target].isnull().all():
                    # Convert binary to numeric (0/1) for correlation
                    binary_numeric = pd.Categorical(df[col]).codes
                    corr_val = abs(np.corrcoef(binary_numeric, df[target])[0, 1])
                    if not np.isnan(corr_val):
                        feature_target_scores[col] = corr_val

        feature_target = {
            "method": "correlation_regression",
            "scores": feature_target_scores,
        }

    # 3. Outlier counts (>3σ) - NUMERIC ONLY
    outliers = {}
    for c in numeric_cols:
        mean, std = df[c].mean(), df[c].std()
        if std > 0:  # Avoid division by zero
            cnt = int(((df[c] - mean).abs() > 3 * std).sum())
            outliers[c] = {"count": cnt, "percentage": round(100 * cnt / num_rows, 2)}

    # 4. Range & IQR - NUMERIC ONLY
    range_iqr = {}
    for c in numeric_cols:
        vals = df[c]
        range_iqr[c] = {
            "range": float(vals.max() - vals.min()),
            "IQR": float(vals.quantile(0.75) - vals.quantile(0.25)),
        }

    # 5. Categorical cardinality
    cardinality = {}
    for c in categorical_cols:
        uc = df[c].nunique()
        cardinality[c] = {"unique": int(uc), "ratio": round(uc / num_rows, 6)}

    # 6. Feature-to-sample ratio
    feature_sample_ratio = round(num_cols / num_rows, 4)

    # ==== Compile everything ====
    eda_results: Dict[str, Any] = {
        "dataset_info": {
            "rows": num_rows,
            "columns": num_cols,
            "memory_usage_mb": round(memory_usage, 2),
            "column_names": df.columns.tolist(),
        },
        "variable_types": {
            "numeric": numeric_cols,
            "binary": binary_cols,
            "categorical": categorical_cols,
            "summary": {
                "numeric_count": len(numeric_cols),
                "binary_count": len(binary_cols),
                "categorical_count": len(categorical_cols),
            },
        },
        "data_types": {
            "summary": {str(k): int(v) for k, v in dtypes_count.items()},
            "by_column": dtypes_by_column,
        },
        "missing_values": {
            "columns_with_missing": {
                k: int(v) for k, v in missing_values.items() if v > 0
            },
            "missing_percentage": missing_pct,
        },
        "numeric_analysis": numeric_stats,
        "binary_analysis": binary_stats,
        "categorical_analysis": categorical_stats,
        "correlation_analysis": correlation_analysis,
        "target_column": target,
        "data_quality_issues": issues,
        # new consolidated metrics
        "data_quality_results": {
            "target_stats": target_stats,
            "feature_target_relationship": feature_target,
            "outlier_stats": outliers,
            "range_iqr": range_iqr,
            "categorical_cardinality": cardinality,
            "feature_sample_ratio": feature_sample_ratio,
        },
        "dataframe_info": dataframe_info,
        "unique_analysis": unique_analysis,
        "sample_data": sample_data,
    }

    logger.info("EDA complete")
    return eda_results


def format_eda_for_llm(
    eda: Dict[str, Any],
    include_sample_data: bool = True,
    include_detailed_info: bool = True,
) -> str:
    """
    Build a concise plain-text summary of extended EDA results,
    including the new metrics under data_quality_results and enhanced details.

    Args:
        eda: EDA results dictionary from exploratory_data_analysis()
        include_sample_data: Whether to include sample data in the output. Defaults to True.
        include_detailed_info: Whether to include detailed DataFrame info. Defaults to True.

    Returns:
        Formatted string summary of the EDA results

    Example:
        >>> results = exploratory_data_analysis(df, 'target')
        >>> summary = format_eda_for_llm(results)
        >>> print(summary)
    """
    lines = []
    info = eda["dataset_info"]
    lines.append(
        f"Dataset: {info['rows']}×{info['columns']} ({info['memory_usage_mb']} MB)"
    )
    lines.append("Columns: " + ", ".join(info["column_names"]))
    lines.append(f"Target: {eda['target_column']}")
    lines.append("")

    # Enhanced variable type classification
    if "variable_types" in eda:
        var_types = eda["variable_types"]
        summary = var_types["summary"]
        lines.append("Variable classification:")
        lines.append(
            f" • Numeric features: {summary['numeric_count']} ({', '.join(var_types['numeric']) if var_types['numeric'] else 'None'})"
        )
        lines.append(
            f" • Binary features: {summary['binary_count']} ({', '.join(var_types['binary']) if var_types['binary'] else 'None'})"
        )
        lines.append(
            f" • Categorical features: {summary['categorical_count']} ({', '.join(var_types['categorical']) if var_types['categorical'] else 'None'})"
        )
        lines.append("")

    # Data types analysis
    dt_summary = eda["data_types"]["summary"]
    lines.append("Data types summary:")
    for dtype, count in dt_summary.items():
        lines.append(f" • {dtype}: {count} columns")
    lines.append("")

    # Enhanced unique value analysis
    if "unique_analysis" in eda:
        unique_stats = eda["unique_analysis"]

        # Get numeric columns to exclude from high cardinality check
        numeric_cols = eda.get("variable_types", {}).get("numeric", [])

        # Find potential issues
        unique_cols = [col for col, stats in unique_stats.items() if stats["is_unique"]]
        constant_cols = [
            col for col, stats in unique_stats.items() if stats["is_constant"]
        ]
        high_cardinality = [
            col
            for col, stats in unique_stats.items()
            if stats["unique_ratio"] > 0.8
            and not stats["is_unique"]
            and not stats["is_binary"]
            and col not in numeric_cols
        ]

        if unique_cols:
            lines.append(f"Unique identifier columns: {', '.join(unique_cols)}")
        if constant_cols:
            lines.append(f"Constant value columns: {', '.join(constant_cols)}")
        if high_cardinality:
            lines.append(
                f"High cardinality columns (>80% unique): {', '.join(high_cardinality)}"
            )
        if unique_cols or constant_cols or high_cardinality:
            lines.append("")

    # Missing values analysis
    mv = eda["missing_values"]["columns_with_missing"]
    if mv:
        lines.append("Missing values:")
        for c, cnt in mv.items():
            pct = eda["missing_values"]["missing_percentage"][c]
            lines.append(f" • {c}: {cnt} missing ({pct} %)")
    else:
        lines.append("No missing values.")
    lines.append("")

    # Sample data display
    if include_sample_data and "sample_data" in eda:
        sample_data = eda["sample_data"]
        lines.append("Sample data (first few rows):")

        # Show first 3 rows in a readable format
        first_rows = sample_data["first_rows"][:3]
        for i, row in enumerate(first_rows, 1):
            lines.append(f"Row {i}:")
            for col, val in row.items():
                # Truncate long values
                str_val = str(val)
                if len(str_val) > 50:
                    str_val = str_val[:47] + "..."
                lines.append(f"  {col}: {str_val}")
        lines.append("")

        # Show sample values for each column
        lines.append("Sample values by column:")
        for col, values in sample_data["column_sample_values"].items():
            if values != ["All NaN"]:
                # Truncate and format sample values
                formatted_values = []
                for val in values:
                    str_val = str(val)
                    if len(str_val) > 30:
                        str_val = str_val[:27] + "..."
                    formatted_values.append(str_val)
                lines.append(f" • {col}: {', '.join(formatted_values)}")
            else:
                lines.append(f" • {col}: All NaN")
        lines.append("")

    # Enhanced Target column analysis
    dqr = eda["data_quality_results"]
    ts = dqr["target_stats"]
    if ts["type"] == "binary":
        values_info = "; ".join(
            f"{val}: {count} ({prop}%)"
            for val, count, prop in zip(ts["values"], ts["counts"], ts["proportions"])
        )
        balance_status = (
            "balanced"
            if ts["is_balanced"]
            else f"imbalanced (ratio: {ts['balance_ratio']:.3f})"
        )
        lines.append(f"Target (binary) distribution: {values_info} - {balance_status}")
    elif ts["type"] == "numeric":
        s = ts["summary"]
        lines.append(
            f"Target (numeric) summary: mean={s['mean']:.3f}, std={s['std']:.3f}, "
            f"min={s['min']:.3f}, max={s['max']:.3f}, IQR={ts['IQR']:.3f}"
        )
    else:
        pct = ts["percentages"]
        dist = "; ".join(f"{k}={v} %" for k, v in pct.items())
        lines.append("Target (categorical) distribution: " + dist)
    lines.append("")

    # Binary features analysis
    if "binary_analysis" in eda and eda["binary_analysis"]:
        lines.append("Binary features analysis:")
        for feature, stats in eda["binary_analysis"].items():
            if feature != eda["target_column"]:  # Don't repeat target info
                values_info = "; ".join(
                    f"{val}: {count} ({prop}%)"
                    for val, count, prop in zip(
                        stats["values"], stats["counts"], stats["proportions"]
                    )
                )
                balance_status = (
                    "balanced"
                    if stats["is_balanced"]
                    else f"imbalanced (ratio: {stats['balance_ratio']:.3f})"
                )
                lines.append(f" • {feature}: {values_info} - {balance_status}")
        lines.append("")

    # Feature–target relationship analysis
    ftr = dqr["feature_target_relationship"]

    # Handle nested scores structure - flatten if needed
    scores = ftr["scores"]
    if scores:
        # Check if scores is nested (contains dictionaries as values)
        if isinstance(next(iter(scores.values())), dict):
            # Flatten nested structure: extract all feature-score pairs
            flat_scores = {}
            for target_name, feature_dict in scores.items():
                if isinstance(feature_dict, dict):
                    flat_scores.update(feature_dict)
                else:
                    # If it's not a dict, treat it as a direct score
                    flat_scores[target_name] = feature_dict
            scores = flat_scores

        # Sort and get top features
        top_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        method_name = ftr["method"].replace("_", " ").title()
        lines.append(
            f"Top 5 feature–target ({method_name}): "
            + ", ".join(f"{f}={score:.3f}" for f, score in top_feats)
        )
    else:
        lines.append("No feature-target relationships calculated.")
    lines.append("")

    # Outlier analysis (show any with >1% prevalence) - NUMERIC ONLY
    ol = {c: v for c, v in dqr["outlier_stats"].items() if v["percentage"] > 1}
    if ol:
        lines.append("Outliers in numeric features (>3σ, >1% of samples):")
        for c, v in ol.items():
            lines.append(f" • {c}: {v['count']} ({v['percentage']}%)")
    else:
        lines.append("No major outlier issues in numeric features (>1%).")
    lines.append("")

    # Categorical cardinality analysis
    card = dqr["categorical_cardinality"]
    if card:
        lines.append("Categorical cardinality (unique, ratio):")
        for c, stats in card.items():
            lines.append(f" • {c}: {stats['unique']} ({stats['ratio'] * 100:.2f}%)")
    else:
        lines.append("No categorical features found.")
    lines.append("")

    # Feature/sample ratio
    lines.append(f"Feature/sample ratio: {dqr['feature_sample_ratio']}")
    lines.append("")

    # Data quality issues
    issues = eda["data_quality_issues"]
    if issues:
        lines.append("Data quality issues:")
        for i in issues:
            lines.append(f" • [{i['issue_type']}] {i['description']}")
    else:
        lines.append("No data quality issues flagged.")
    lines.append("")

    # Detailed DataFrame info (optional)
    if include_detailed_info and "dataframe_info" in eda:
        lines.append("Detailed DataFrame Information:")
        lines.append(eda["dataframe_info"])

    return "\n".join(lines)


def to_code_markdown(text: str) -> str:
    """
    Converts markdown h1 headers to code comments and wraps the block in python code fencing.
    """
    lines = text.splitlines()
    formatted_lines = []
    for line in lines:
        stripped = line.lstrip()
        # Convert markdown header to code comment, but avoid double commenting
        if stripped.startswith("# ") and not stripped.startswith("##"):
            # Make sure only single # at the beginning, not code comment
            # Convert to code comment if not already
            if not line.startswith("#"):
                line = "# " + line.lstrip("#").strip()
        formatted_lines.append(line)
    # Remove leading/trailing blank lines for neatness
    while formatted_lines and not formatted_lines[0].strip():
        formatted_lines.pop(0)
    while formatted_lines and not formatted_lines[-1].strip():
        formatted_lines.pop()
    code_block = "\n".join(formatted_lines)
    return f"```python\n{code_block}\n```"


def double_newlines_outside_code(text: str) -> str:
    """
    Replace all '\n' with '\n\n' except inside code fencing (triple backticks).
    """
    # This regex splits on code blocks: everything between triple backticks (supports ```python too)
    pattern = r"(```.*?```)"
    segments = re.split(pattern, text, flags=re.DOTALL)
    output = []
    for seg in segments:
        if seg.startswith("```") and seg.endswith("```"):
            # Code block: keep as is
            output.append(seg)
        else:
            # Outside code block: double the newlines
            seg = seg.replace("\n", "\n\n")
            output.append(seg)
    return "".join(output)


def smart_display_markdown(
    text: str, display_method: Optional[Literal["print", "display"]] = None
):
    """
    Display markdown in a Jupyter notebook if available; otherwise, print as plain text.

    Args:
        text: The markdown text to display
        display_method: Optional method to force either "print" or "display". If None,
            automatically determines best method based on environment.
    """

    def in_jupyter_notebook() -> bool:
        try:
            from IPython import get_ipython

            shell = get_ipython()
            if shell is None:
                return False
            # Jupyter notebook or qtconsole
            if shell.__class__.__name__ == "ZMQInteractiveShell":
                return True
            # Terminal running IPython
            elif shell.__class__.__name__ == "TerminalInteractiveShell":
                return False
            else:
                return False
        except ImportError:
            return False

    # Use specified display method if provided
    if display_method == "print":
        print(text)
        return
    elif display_method == "display":
        from IPython.display import Markdown, display

        display(Markdown(double_newlines_outside_code(text)))
        return

    # Auto-detect method if not specified
    if in_jupyter_notebook():
        from IPython.display import Markdown, display

        display(Markdown(double_newlines_outside_code(text)))
    else:
        print(text)


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
    Generate an dataset summary that includes leave-one-out feature importance analysis.

    This function combines dataset summarization with feature importance calculation using
    a leave-one-out approach. It evaluates how much each feature contributes to model
    performance by measuring the performance drop when that feature is removed.

    Args:
        df_train: Training DataFrame containing features and target
        target_name: Name of the target column
        df_test: Test DataFrame (optional). If None, uses cross-validation
        model: Sklearn-compatible estimator. If None, uses RandomForestClassifier/Regressor
        n_samples: Number of sample values to show per feature. Defaults to 10
        cv_folds: Number of cross-validation folds. Defaults to 5
        include_importance: Whether to calculate feature importance. Defaults to True

    Returns:
        Enhanced string summary of the dataset including feature importances

    Raises:
        ValueError: If target column is not found in the dataset

    """

    # Validate inputs
    if target_name not in df_train.columns:
        raise ValueError(f"Target column '{target_name}' not found in training data")

    # Start building the summary
    summary_parts = []

    # Basic dataset information
    summary_parts.append(
        f"Dataset shape: {df_train.shape[0]} rows, {df_train.shape[1]} columns"
    )
    summary_parts.append(f"Target column: {target_name}")

    # Feature importances dictionary (will be populated if include_importance=True)
    feature_importances = None

    # Calculate leave-one-out feature importance if requested
    if include_importance:
        try:
            # Prepare features list
            features = [col for col in df_train.columns if col != target_name]

            if len(features) == 0:
                summary_parts.append(
                    "\nNo features available for importance calculation."
                )
            else:
                # Determine task type
                y_train = df_train[target_name]
                unique_values = y_train.nunique()
                task_type = "classification" if unique_values <= 10 else "regression"

                # Set default model if none provided
                if model is None:
                    if task_type == "classification":
                        model = RandomForestClassifier(n_estimators=10, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=10, random_state=42)

                # Prepare data for modeling
                def prepare_modeling_data(df):
                    """Simple data preparation for modeling."""
                    df_processed = df.copy()

                    # Handle categorical variables with label encoding
                    label_encoders = {}
                    for col in df_processed.columns:
                        if col != target_name and df_processed[col].dtype == "object":
                            le = LabelEncoder()
                            # Handle missing values
                            df_processed[col] = df_processed[col].fillna("missing")
                            df_processed[col] = le.fit_transform(
                                df_processed[col].astype(str)
                            )
                            label_encoders[col] = le

                    # Fill numeric missing values with median
                    numeric_cols = df_processed.select_dtypes(
                        include=[np.number]
                    ).columns
                    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                        df_processed[numeric_cols].median()
                    )

                    return df_processed, label_encoders

                # Prepare training data
                df_train_processed, encoders = prepare_modeling_data(df_train)
                X_train = df_train_processed.drop(columns=[target_name])
                y_train = df_train_processed[target_name]

                # Prepare test data if provided
                X_test, y_test = None, None
                if df_test is not None:
                    df_test_processed = df_test.copy()

                    # Apply same preprocessing to test data
                    for col, le in encoders.items():
                        if col in df_test_processed.columns:
                            df_test_processed[col] = df_test_processed[col].fillna(
                                "missing"
                            )
                            # Handle unseen categories
                            unknown_mask = ~df_test_processed[col].astype(str).isin(
                                le.classes_
                            )
                            df_test_processed.loc[unknown_mask, col] = "missing"
                            df_test_processed[col] = le.transform(
                                df_test_processed[col].astype(str)
                            )

                    # Fill numeric missing values
                    numeric_cols = df_test_processed.select_dtypes(
                        include=[np.number]
                    ).columns
                    df_test_processed[numeric_cols] = df_test_processed[
                        numeric_cols
                    ].fillna(df_train_processed[numeric_cols].median())

                    X_test = df_test_processed.drop(columns=[target_name])
                    y_test = df_test_processed[target_name]

                # Calculate baseline performance
                if df_test is not None:
                    # Use train/test split
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if task_type == "classification":
                        baseline_score = accuracy_score(y_test, y_pred)
                    else:
                        baseline_score = r2_score(y_test, y_pred)
                else:
                    # Use cross-validation
                    if task_type == "classification":
                        scores = cross_val_score(
                            model, X_train, y_train, cv=cv_folds, scoring="accuracy"
                        )
                    else:
                        scores = cross_val_score(
                            model, X_train, y_train, cv=cv_folds, scoring="r2"
                        )
                    baseline_score = scores.mean()

                # Calculate importance for each feature
                importances = {}

                for feat in features:
                    # Create datasets without this feature
                    X_train_reduced = X_train.drop(columns=[feat])

                    if df_test is not None:
                        X_test_reduced = X_test.drop(columns=[feat])
                        # Evaluate with train/test split
                        model.fit(X_train_reduced, y_train)
                        y_pred = model.predict(X_test_reduced)
                        if task_type == "classification":
                            reduced_score = accuracy_score(y_test, y_pred)
                        else:
                            reduced_score = r2_score(y_test, y_pred)
                    else:
                        # Use cross-validation
                        if task_type == "classification":
                            scores = cross_val_score(
                                model,
                                X_train_reduced,
                                y_train,
                                cv=cv_folds,
                                scoring="accuracy",
                            )
                        else:
                            scores = cross_val_score(
                                model,
                                X_train_reduced,
                                y_train,
                                cv=cv_folds,
                                scoring="r2",
                            )
                        reduced_score = scores.mean()

                    # Importance = drop in performance when feature is removed
                    importance = baseline_score - reduced_score
                    importances[feat] = round(importance, 6)

                # Sort by importance (descending)
                feature_importances = dict(
                    sorted(importances.items(), key=lambda x: x[1], reverse=True)
                )

                # summary_parts.append(f"\nLeave-one-out Feature Importance (baseline {metric}: {baseline_score:.4f}):")
                # for feat, imp in feature_importances.items():
                #     summary_parts.append(f"  {feat}: {imp:.6f}")

        except Exception as e:
            summary_parts.append(
                f"\nWarning: Could not calculate feature importances: {e}"
            )

    # Column information with importance if available
    summary_parts.append("\nColumns:")
    for col in df_train.columns:
        dtype = df_train[col].dtype
        nan_freq = df_train[col].isna().mean() * 100

        # Get sample values
        samples = df_train[col].dropna().head(n_samples).tolist()
        if str(dtype) == "float64":
            samples = [round(s, 2) for s in samples]

        # Add feature importance if available
        importance_info = ""
        if feature_importances and col in feature_importances:
            importance_info = f", Importance: {feature_importances[col]:.6f}"

        summary_parts.append(
            f"- {col} ({dtype}): NaN-freq [{nan_freq:.1f}%], Samples: {samples[:5]}{importance_info}"
        )

    # Target distribution
    if target_name in df_train.columns:
        target_dist = df_train[target_name].value_counts(normalize=True).to_dict()
        summary_parts.append(f"\nTarget distribution: {target_dist}")

    return "\n".join(summary_parts)


def get_X_y(df_train, target_name):
    y = torch.tensor(df_train[target_name].astype(int).to_numpy())
    x = torch.tensor(df_train.drop(target_name, axis=1).to_numpy())

    return x, y


def make_df_numeric(
    df: pd.DataFrame,
    encoder: Optional[OrdinalEncoder] = None,
    categorical_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Converts the categorical columns in the given dataframe to integer values
    using either provided mappings or a sklearn OrdinalEncoder.

    Parameters:
    df: DataFrame to convert.
    encoder: Fitted sklearn OrdinalEncoder.
    categorical_cols: List of categorical columns to encode (required if using encoder).

    Returns:
    Converted DataFrame.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df_out = df.copy()

    if encoder is not None:
        if categorical_cols is None:
            raise ValueError("categorical_cols must be provided if using encoder.")

        # Make sure all values are strings, as OrdinalEncoder expects
        df_out[categorical_cols] = df_out[categorical_cols].astype(str)
        # Transform and insert back
        df_out[categorical_cols] = encoder.transform(df_out[categorical_cols])
        df_out[categorical_cols] = df_out[categorical_cols].astype(float)
        return df_out


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


def auc_metric(target, pred, multi_class="ovo", numpy=False):
    lib = np if numpy else torch
    try:
        if not numpy:
            target = torch.tensor(target) if not torch.is_tensor(target) else target
            pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
        if len(lib.unique(target)) > 2:
            if not numpy:
                return torch.tensor(
                    roc_auc_score(target, pred, multi_class=multi_class)
                )
            return roc_auc_score(target, pred, multi_class=multi_class)
        else:
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred))
            return roc_auc_score(target, pred)
    except ValueError as e:
        print(e)
        return np.nan if numpy else torch.tensor(np.nan)


def accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(accuracy_score(target, pred[:, 1] > 0.5))


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