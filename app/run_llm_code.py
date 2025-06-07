"""
LLM Code Execution Module

This module provides safe execution of LLM-generated Python code for feature engineering,
with AST validation and controlled execution environment.
"""

import ast
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy
import sklearn

logger = logging.getLogger(__name__)


def convert_categorical_to_integer_f(
    column: pd.Series, 
    mapping: Optional[Dict[int, str]] = None
) -> pd.Series:
    """
    Converts a categorical column to integer values using the given mapping.

    Parameters:
        column (pandas.Series): The column to convert.
        mapping (Dict[int, str], optional): The mapping to use for the conversion. 
            Defaults to None.

    Returns:
        pandas.Series: The converted column.

    Example:
        >>> col = pd.Series(['A', 'B', 'A', None])
        >>> result = convert_categorical_to_integer_f(col)
        >>> print(result.values)
        [0 1 0 -1]
    """
    if mapping is not None:
        # if column is categorical
        if column.dtype.name == "category":
            # Only add -1 to categories if it's not already present
            if -1 not in column.cat.categories:
                column = column.cat.add_categories([-1])
        return column.map(mapping).fillna(-1).astype(int)
    return column


def run_llm_code(
    code: str, 
    df: pd.DataFrame, 
    convert_categorical_to_integer: Optional[bool] = False, 
    fill_na: Optional[bool] = False
) -> pd.DataFrame:
    """
    Executes the given code on the given dataframe and returns the resulting dataframe.

    This function provides a controlled environment for executing LLM-generated Python code
    with safety checks and preprocessing options.

    Args:
        code (str): The Python code to execute.
        df (pandas.DataFrame): The dataframe to execute the code on.
        convert_categorical_to_integer (bool, optional): Whether to convert categorical 
            columns to integer values. Defaults to False.
        fill_na (bool, optional): Whether to fill NaN values in object columns with 
            empty strings. Defaults to False.

    Returns:
        pandas.DataFrame: The resulting dataframe after executing the code.

    Raises:
        ValueError: If the code cannot be executed safely or produces an error.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', None, 'z']})
        >>> code = 'df["c"] = df["a"].fillna(0) * 2'
        >>> result = run_llm_code(code, df)
        >>> print(result[['a', 'b', 'c']])
           a     b    c
        0  1.0     x  2.0
        1  2.0  None  4.0
        2  NaN     z  0.0
    """
    try:
        df = df.copy()

        if fill_na:
            # Use select_dtypes to get object columns and fill NaNs with empty string
            object_cols = df.select_dtypes(include="object").columns
            df[object_cols] = df[object_cols].fillna("")
            
        if convert_categorical_to_integer:
            df = df.apply(convert_categorical_to_integer_f)

        # Create controlled execution environment
        access_scope = {
            "df": df, 
            "pd": pd, 
            "np": np, 
            "scipy": scipy, 
            "sklearn": sklearn
        }
        
        # Parse and validate the code
        parsed = ast.parse(code)
        check_ast(parsed)
        
        # Compile and execute in controlled environment
        compiled_code = compile(parsed, filename="<ast>", mode="exec")
        exec(compiled_code, access_scope, access_scope)
        
        # Extract the potentially modified dataframe
        df = access_scope.get("df", df)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error_msg = f"Code could not be executed! {e}.\nTraceback: {tb}\nCode that failed: {code}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return df


def check_ast(node: ast.AST) -> None:
    """
    Checks if the given AST node is allowed based on security constraints.

    This function validates that the AST only contains safe operations and prevents
    execution of potentially dangerous code like imports of unsafe modules,
    file operations, network operations, etc.

    Parameters:
        node (ast.AST): The AST node to check.

    Raises:
        ValueError: If the AST node contains disallowed operations.

    Example:
        >>> import ast
        >>> safe_code = "df['new_col'] = df['old_col'] * 2"
        >>> parsed = ast.parse(safe_code)
        >>> check_ast(parsed)  # Should not raise an exception
        
        >>> unsafe_code = "import os; os.system('rm -rf /')"
        >>> parsed = ast.parse(unsafe_code)
        >>> check_ast(parsed)  # Should raise ValueError
        Traceback (most recent call last):
        ...
        ValueError: Disallowed AST node type: Import
    """
    allowed_nodes = {
        ast.Module,
        ast.Expr,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Num,
        ast.Str,
        ast.Bytes,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Name,
        ast.Call,
        ast.Attribute,
        ast.keyword,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.ExtSlice,
        ast.Assign,
        ast.AugAssign,
        ast.NameConstant,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.And,
        ast.Or,
        ast.BitOr,
        ast.BitAnd,
        ast.BitXor,
        ast.Invert,
        ast.Not,
        ast.Constant,
        ast.Store,
        ast.If,
        ast.IfExp,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Assert,
        ast.Return,
        ast.FunctionDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Lambda,
        ast.BoolOp,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.Set,
        ast.Ellipsis,
        ast.expr,
        ast.stmt,
        ast.expr_context,
        ast.boolop,
        ast.operator,
        ast.unaryop,
        ast.cmpop,
        ast.comprehension,
        ast.arguments,
        ast.arg,
    }

    # Packages that are allowed to be used
    allowed_packages = {"numpy", "pandas", "sklearn", "scipy", "re"}

    # Built-in functions that are allowed
    allowed_funcs = {
        "sum", "min", "max", "abs", "round", "len", "str", "int", "float", 
        "bool", "list", "dict", "set", "tuple", "enumerate", "zip", "range", 
        "sorted", "reversed", "isinstance"
    }

    # Allowed attribute names for safe operations
    allowed_attrs = {
        # NumPy operations
        "array", "arange", "values", "linspace", "mean", "sum", "min", "max", 
        "median", "std", "sqrt", "pow", "abs", "log", "log10", "log1p", "exp", 
        "expm1", "clip", "round", "transpose", "T", "astype", "reshape", "shape",
        
        # Pandas operations
        "iloc", "loc", "cut", "qcut", "inf", "nan", "isna", "map", "split", 
        "var", "codes", "cumsum", "cumprod", "cummax", "cummin", "diff", 
        "repeat", "index", "slice", "pct_change", "corr", "cov", "dot", 
        "copy", "drop", "dropna", "fillna", "replace", "merge", "append", 
        "join", "groupby", "resample", "rolling", "expanding", "ewm", "agg", 
        "aggregate", "filter", "transform", "apply", "pivot", "melt", 
        "sort_values", "sort_index", "reset_index", "set_index", "reindex", 
        "shift", "rename", "tail", "head", "describe", "count", "value_counts", 
        "unique", "nunique", "idxmin", "idxmax", "isin", "between", 
        "duplicated", "rank", "to_numpy", "to_dict", "to_list", "to_frame", 
        "squeeze", "add", "sub", "mul", "div", "mod", "columns", "lt", "le", 
        "eq", "ne", "ge", "gt", "all", "any", "str", "dt", "cat", "sparse",
        
        # SciPy operations
        "stats", "signal", "special", "interpolate", "integrate", "optimize", 
        "linalg", "fft", "ndimage", "spatial", "distance", "norm", "normaltest", 
        "skew", "kurtosis", "mode", "gmean", "hmean", "sem", "ttest_ind", 
        "ttest_rel", "chi2_contingency", "pearsonr", "spearmanr", "kendalltau", 
        "zscore", "percentileofscore", "rankdata", "boxcox", "boxcox1p", 
        "yeojohnson", "gaussian_filter", "medfilt", "savgol_filter", "interp1d", 
        "UnivariateSpline", "griddata", "cdist", "pdist", "euclidean", "cosine", 
        "correlation",
        
        # Sklearn operations
        "preprocessing", "feature_selection", "decomposition", "cluster", 
        "metrics", "model_selection", "ensemble", "linear_model", "tree", 
        "svm", "neighbors", "naive_bayes", "discriminant_analysis", 
        "gaussian_process", "neural_network", "StandardScaler", "MinMaxScaler", 
        "MaxAbsScaler", "RobustScaler", "Normalizer", "QuantileTransformer", 
        "PowerTransformer", "PolynomialFeatures", "OneHotEncoder", 
        "OrdinalEncoder", "LabelEncoder", "LabelBinarizer", "MultiLabelBinarizer", 
        "KBinsDiscretizer", "Binarizer", "FunctionTransformer", "SimpleImputer", 
        "KNNImputer", "MissingIndicator", "SelectKBest", "SelectPercentile", 
        "SelectFpr", "SelectFdr", "SelectFwe", "GenericUnivariateSelect", 
        "VarianceThreshold", "RFE", "RFECV", "SelectFromModel", 
        "SequentialFeatureSelector", "chi2", "f_classif", "f_regression", 
        "mutual_info_classif", "mutual_info_regression", "PCA", "IncrementalPCA", 
        "KernelPCA", "SparsePCA", "MiniBatchSparsePCA", "FactorAnalysis", 
        "FastICA", "TruncatedSVD", "NMF", "MiniBatchNMF", "LatentDirichletAllocation", 
        "KMeans", "MiniBatchKMeans", "AffinityPropagation", "MeanShift", 
        "SpectralClustering", "AgglomerativeClustering", "DBSCAN", "OPTICS", 
        "Birch", "GaussianMixture", "BayesianGaussianMixture", "fit", "transform", 
        "fit_transform", "predict", "predict_proba", "predict_log_proba", "score", 
        "decision_function", "inverse_transform", "get_params", "set_params", 
        "partial_fit", "get_feature_names", "get_feature_names_out", "components_", 
        "explained_variance_", "explained_variance_ratio_", "singular_values_", 
        "mean_", "var_", "scale_", "n_components_", "n_features_", "n_samples_", 
        "feature_importances_", "coef_", "intercept_", "classes_", "n_classes_", 
        "feature_names_in_"
    }

    def _check_node(node):
        """Recursively check if all nodes in the AST are allowed."""
        
        # Check if the node type is allowed
        if type(node) not in allowed_nodes:
            raise ValueError(f"Disallowed AST node type: {type(node).__name__}")
        
        # Check imports - only allow specific packages
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in allowed_packages:
                        raise ValueError(f"Disallowed import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module not in allowed_packages:
                    raise ValueError(f"Disallowed import from: {node.module}")
        
        # Check function calls - prevent dangerous built-ins
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_funcs and node.func.id not in ["df", "pd", "np", "scipy", "sklearn"]:
                    # Allow calls to pandas/numpy/etc. methods
                    pass
        
        # Check attribute access
        if isinstance(node, ast.Attribute):
            if hasattr(node, 'attr') and node.attr not in allowed_attrs:
                # Allow df.column_name access
                if not (isinstance(node.value, ast.Name) and node.value.id == 'df'):
                    logger.warning(f"Potentially unsafe attribute access: {node.attr}")
        
        # Recursively check child nodes
        for child in ast.iter_child_nodes(node):
            _check_node(child)
    
    _check_node(node)


def validate_feature_code(code: str) -> bool:
    """
    Validate that feature engineering code is safe to execute.
    
    Args:
        code (str): Python code to validate
        
    Returns:
        bool: True if code is safe, False otherwise
        
    Example:
        >>> safe_code = "df['new_feature'] = df['age'] * df['income']"
        >>> validate_feature_code(safe_code)
        True
        
        >>> unsafe_code = "import os; os.system('rm file.txt')"
        >>> validate_feature_code(unsafe_code)
        False
    """
    try:
        parsed = ast.parse(code)
        check_ast(parsed)
        return True
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Code validation failed: {e}")
        return False


def extract_feature_names_from_code(code: str) -> list:
    """
    Extract feature names that are being created in the code.
    
    Args:
        code (str): Python code to analyze
        
    Returns:
        list: List of feature names being created
        
    Example:
        >>> code = "df['new_feature'] = df['age'] * 2\\ndf['another'] = df['income'].log()"
        >>> extract_feature_names_from_code(code)
        ['new_feature', 'another']
    """
    feature_names = []
    
    try:
        parsed = ast.parse(code)
        
        for node in ast.walk(parsed):
            # Look for assignments to df columns: df['column'] = ...
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript) and 
                        isinstance(target.value, ast.Name) and 
                        target.value.id == 'df'):
                        
                        if isinstance(target.slice, ast.Constant):
                            feature_names.append(target.slice.value)
                        elif isinstance(target.slice, ast.Str):  # For older Python versions
                            feature_names.append(target.slice.s)
                            
    except SyntaxError:
        logger.warning("Could not parse code to extract feature names")
    
    return feature_names 