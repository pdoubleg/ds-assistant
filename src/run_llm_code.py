import copy
import numpy as np
from typing import Dict, Optional
import pandas as pd
import ast
import scipy
import sklearn


def convert_categorical_to_integer_f(column: pd.Series, mapping: Optional[Dict[int, str]] = None) -> pd.Series:
    """
    Converts a categorical column to integer values using the given mapping.

    Parameters:
    column (pandas.Series): The column to convert.
    mapping (Dict[int, str], optional): The mapping to use for the conversion. Defaults to None.

    Returns:
    pandas.Series: The converted column.
    """
    if mapping is not None:
        # if column is categorical
        if column.dtype.name == "category":
            # Only add -1 to categories if it's not already present
            if -1 not in column.cat.categories:
                column = column.cat.add_categories([-1])
        return column.map(mapping).fillna(-1).astype(int)
    return column


def run_llm_code(code: str, df: pd.DataFrame, convert_categorical_to_integer: Optional[bool] = False, fill_na: Optional[bool] = False) -> pd.DataFrame:
    """
    Executes the given code on the given dataframe and returns the resulting dataframe.

    Args:
        code (str): The code to execute.
        df (pandas.DataFrame): The dataframe to execute the code on.
        convert_categorical_to_integer (bool, optional): Whether to convert categorical columns to integer values. Defaults to False.
        fill_na (bool, optional): Whether to fill NaN values in object columns with empty strings. Defaults to False.

    Returns:
        pandas.DataFrame: The resulting dataframe after executing the code.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', None, 'z']})
        >>> code = 'df["a"] = df["a"].fillna(0)'
        >>> run_llm_code(code, df)
           a  b
        0  1.0  x
        1  2.0  
        2  0.0  z
    """
    try:
        df = df.copy()

        if fill_na:
            # Use select_dtypes to get object columns and fill NaNs with empty string
            object_cols = df.select_dtypes(include="object").columns
            df[object_cols] = df[object_cols].fillna("")
        if convert_categorical_to_integer:
            df = df.apply(convert_categorical_to_integer_f)

        access_scope = {"df": df, "pd": pd, "np": np, "scipy": scipy, "sklearn": sklearn}
        parsed = ast.parse(code)
        check_ast(parsed)
        # Use the same namespace for both global and local scope so variables are accessible
        exec(compile(parsed, filename="<ast>", mode="exec"), access_scope, access_scope)
        # Extract the potentially modified dataframe from the execution scope
        df = access_scope.get("df", df)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise ValueError(f"Code could not be executed! {e}. \nTraceback: {tb}\nCode that failed: {code}")

    return df


def check_ast(node: ast.AST) -> None:
    """
    Checks if the given AST node is allowed.

    Parameters:
    node (ast.AST): The AST node to check.

    Raises:
    ValueError: If the AST node is not allowed.
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
        # These nodes represent loop structures. If you allow arbitrary loops, a user could potentially create an infinite loop that consumes system resources and slows down or crashes your system.
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
        # These nodes represent the yield keyword, which is used in generator functions. If you allow arbitrary generator functions, a user might be able to create a generator that produces an infinite sequence, potentially consuming system resources and slowing down or crashing your system.
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
        ast.Import,
        ast.ImportFrom,
        ast.alias,
    }

    allowed_packages = {"numpy", "pandas", "sklearn", "scipy", "re"}

    allowed_funcs = {
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "enumerate": enumerate,
        "zip": zip,
        "range": range,
        "sorted": sorted,
        "reversed": reversed,
        "isinstance": isinstance,
        # Add other functions you want to allow here.
    }

    allowed_attrs = {
        # NP
        "array",
        "arange",
        "values",
        "linspace",
        # PD
        "mean",
        "sum",
        "contains",
        "where",
        "min",
        "max",
        "median",
        "std",
        "sqrt",
        "pow",
        "iloc",
        "cut",
        "qcut",
        "inf",
        "nan",
        "isna",
        "map",
        "reshape",
        "shape",
        "split",
        "var",
        "codes",
        "abs",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "diff",
        "repeat",
        "index",
        "log",
        "log10",
        "log1p",
        "slice",
        "exp",
        "expm1",
        "pow",
        "pct_change",
        "corr",
        "cov",
        "round",
        "clip",
        "dot",
        "transpose",
        "T",
        "astype",
        "copy",
        "drop",
        "dropna",
        "fillna",
        "replace",
        "merge",
        "append",
        "join",
        "groupby",
        "resample",
        "rolling",
        "expanding",
        "ewm",
        "agg",
        "aggregate",
        "filter",
        "transform",
        "apply",
        "pivot",
        "melt",
        "sort_values",
        "sort_index",
        "reset_index",
        "set_index",
        "reindex",
        "shift",
        # "extract",
        "rename",
        "tail",
        "head",
        "describe",
        "count",
        "value_counts",
        "unique",
        "nunique",
        "idxmin",
        "idxmax",
        "isin",
        "between",
        "duplicated",
        "rank",
        "to_numpy",
        "to_dict",
        "to_list",
        "to_frame",
        "squeeze",
        "add",
        "sub",
        "mul",
        "div",
        "mod",
        "columns",
        "loc",
        "lt",
        "le",
        "eq",
        "ne",
        "ge",
        "gt",
        "all",
        "any",
        "clip",
        "conj",
        "conjugate",
        "round",
        "trace",
        "cumprod",
        "cumsum",
        "prod",
        "dot",
        "flatten",
        "ravel",
        "T",
        "transpose",
        "swapaxes",
        "clip",
        "item",
        "tolist",
        "argmax",
        "argmin",
        "argsort",
        "max",
        "mean",
        "min",
        "nonzero",
        "ptp",
        "sort",
        "std",
        "var",
        "str",
        "dt",
        "cat",
        "sparse",
        "plot",
        # SCIPY attributes
        "stats",
        "signal",
        "special",
        "interpolate",
        "integrate",
        "optimize",
        "linalg",
        "fft",
        "ndimage",
        "spatial",
        "distance",
        "norm",
        "normaltest",
        "skew",
        "kurtosis",
        "mode",
        "gmean",
        "hmean",
        "sem",
        "ttest_ind",
        "ttest_rel",
        "chi2_contingency",
        "pearsonr",
        "spearmanr",
        "kendalltau",
        "zscore",
        "percentileofscore",
        "rankdata",
        "boxcox",
        "boxcox1p",
        "yeojohnson",
        "gaussian_filter",
        "medfilt",
        "savgol_filter",
        "interp1d",
        "UnivariateSpline",
        "griddata",
        "cdist",
        "pdist",
        "euclidean",
        "cosine",
        "correlation",
        # SKLEARN attributes
        "preprocessing",
        "feature_selection",
        "decomposition",
        "cluster",
        "metrics",
        "model_selection",
        "ensemble",
        "linear_model",
        "tree",
        "svm",
        "neighbors",
        "naive_bayes",
        "discriminant_analysis",
        "gaussian_process",
        "neural_network",
        "StandardScaler",
        "MinMaxScaler",
        "MaxAbsScaler",
        "RobustScaler",
        "Normalizer",
        "QuantileTransformer",
        "PowerTransformer",
        "PolynomialFeatures",
        "OneHotEncoder",
        "OrdinalEncoder",
        "LabelEncoder",
        "LabelBinarizer",
        "MultiLabelBinarizer",
        "KBinsDiscretizer",
        "Binarizer",
        "FunctionTransformer",
        "SimpleImputer",
        "KNNImputer",
        "MissingIndicator",
        "SelectKBest",
        "SelectPercentile",
        "SelectFpr",
        "SelectFdr",
        "SelectFwe",
        "GenericUnivariateSelect",
        "VarianceThreshold",
        "RFE",
        "RFECV",
        "SelectFromModel",
        "SequentialFeatureSelector",
        "chi2",
        "f_classif",
        "f_regression",
        "mutual_info_classif",
        "mutual_info_regression",
        "PCA",
        "IncrementalPCA",
        "KernelPCA",
        "SparsePCA",
        "MiniBatchSparsePCA",
        "FactorAnalysis",
        "FastICA",
        "TruncatedSVD",
        "NMF",
        "MiniBatchNMF",
        "LatentDirichletAllocation",
        "KMeans",
        "MiniBatchKMeans",
        "AffinityPropagation",
        "MeanShift",
        "SpectralClustering",
        "AgglomerativeClustering",
        "DBSCAN",
        "OPTICS",
        "Birch",
        "GaussianMixture",
        "BayesianGaussianMixture",
        "fit",
        "transform",
        "fit_transform",
        "predict",
        "predict_proba",
        "predict_log_proba",
        "score",
        "decision_function",
        "inverse_transform",
        "get_params",
        "set_params",
        "partial_fit",
        "get_feature_names",
        "get_feature_names_out",
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
        "mean_",
        "var_",
        "scale_",
        "n_components_",
        "n_features_",
        "n_samples_",
        "feature_importances_",
        "coef_",
        "intercept_",
        "classes_",
        "n_classes_",
        "feature_names_in_",
        "n_features_in_",
        "data_min_",
        "data_max_",
        "data_range_",
        "min_",
        "max_",
        "center_",
        "scale_",
        "mean_",
        "var_",
        "n_samples_seen_",
        "isnull",
        "notnull",
        "isna",
        "dtype",
        # Add other DataFrame methods you want to allow here.
    }

    if type(node) not in allowed_nodes:
        raise ValueError(f"Disallowed code: {ast.unparse(node)} is {type(node)}")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in allowed_funcs:
            raise ValueError(f"Disallowed function: {node.func.id}")

    if isinstance(node, ast.Attribute) and node.attr not in allowed_attrs:
        raise ValueError(f"Disallowed attribute: {node.attr}")

    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name not in allowed_packages:
                raise ValueError(f"Disallowed package import: {alias.name}")

    for child in ast.iter_child_nodes(node):
        check_ast(child)