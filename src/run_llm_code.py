from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import ast
import scipy
import sklearn
from scipy import sparse
from sklearn import compose, feature_extraction, impute, pipeline, preprocessing


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


def get_default_tabular_encoder(df: pd.DataFrame) -> Any:
    """Build a safe default encoder for tabular model evaluation.

    Args:
        df: Feature dataframe without the target column.

    Returns:
        A sklearn-compatible transformer.

    Raises:
        ValueError: If the dataframe does not contain any feature columns.
    """
    if df.empty:
        raise ValueError("Cannot build a default encoder for an empty feature dataframe.")

    numeric_columns = list(df.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [col for col in df.columns if col not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", "passthrough", numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                preprocessing.OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("Unable to infer any columns for default tabular encoding.")

    return compose.ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def _build_execution_scope(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    encoder: Any = None,
) -> Dict[str, Any]:
    """Build the execution namespace for LLM-authored code.

    Args:
        df: Primary dataframe reference exposed to the LLM.
        df_train: Training dataframe for encoding code.
        df_test: Optional validation dataframe for encoding code.
        encoder: Optional pre-existing encoder instance.

    Returns:
        Execution scope passed to ``exec``.
    """
    scope: Dict[str, Any] = {
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "encoder": encoder,
        "pd": pd,
        "np": np,
        "scipy": scipy,
        "sklearn": sklearn,
        "compose": compose,
        "feature_extraction": feature_extraction,
        "impute": impute,
        "pipeline": pipeline,
        "preprocessing": preprocessing,
    }
    return {key: value for key, value in scope.items() if value is not None}


def _coerce_feature_matrix(
    matrix: Any,
    expected_rows: int,
    dataset_name: str,
) -> Any:
    """Validate and normalize an encoded feature matrix.

    Args:
        matrix: Matrix returned by a fitted sklearn transformer.
        expected_rows: Expected number of rows after transform.
        dataset_name: Human-readable dataset label for error messages.

    Returns:
        A numeric dataframe, numpy array, or scipy sparse matrix.

    Raises:
        ValueError: If the transformed output is empty, row-misaligned, or non-numeric.
    """
    if matrix is None:
        raise ValueError(f"The {dataset_name} encoder output is None.")

    if sparse.issparse(matrix):
        if matrix.shape[0] != expected_rows:
            raise ValueError(
                f"The {dataset_name} encoder output row count {matrix.shape[0]} "
                f"does not match the input row count {expected_rows}."
            )
        return matrix.astype(np.float32)

    if isinstance(matrix, pd.DataFrame):
        if len(matrix) != expected_rows:
            raise ValueError(
                f"The {dataset_name} encoder output row count {len(matrix)} "
                f"does not match the input row count {expected_rows}."
            )
        numeric_df = matrix.replace([np.inf, -np.inf], np.nan)
        try:
            return numeric_df.astype(np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"The {dataset_name} encoder output must be numeric. {exc}"
            ) from exc

    array = np.asarray(matrix)
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.shape[0] != expected_rows:
        raise ValueError(
            f"The {dataset_name} encoder output row count {array.shape[0]} "
            f"does not match the input row count {expected_rows}."
        )

    try:
        array = array.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"The {dataset_name} encoder output must be numeric. {exc}"
        ) from exc

    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def run_llm_encoder_code(
    code: str,
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None,
    encoder: Any = None,
    fit_encoder: bool = True,
) -> tuple[Any, Optional[Any], Any]:
    """Execute LLM-authored sklearn encoding code and transform data.

    The encoding contract requires the code to assign a sklearn-compatible
    transformer to ``encoder``. The helper fits the encoder on ``df_train`` and
    transforms both train and optional test datasets.

    Args:
        code: Python code that assigns a transformer to ``encoder``.
        df_train: Training dataframe without the target column.
        df_test: Optional validation/inference dataframe without the target.
        encoder: Optional pre-built or pre-fitted encoder.
        fit_encoder: Whether to fit the encoder on ``df_train`` before transform.

    Returns:
        Tuple of ``(encoded_train, encoded_test, encoder)``.

    Raises:
        ValueError: If the code cannot be executed, does not assign ``encoder``,
            or produces invalid transformed outputs.
    """
    if code.strip():
        parsed = ast.parse(code)
        check_ast(parsed)
        scope = _build_execution_scope(
            df=df_train.copy(),
            df_train=df_train.copy(),
            df_test=df_test.copy() if df_test is not None else None,
            encoder=encoder,
        )
        try:
            # Share the same global/local scope so assigned variables are reusable.
            exec(compile(parsed, filename="<ast>", mode="exec"), scope, scope)
        except Exception as exc:
            raise ValueError(
                f"Encoding code could not be executed! {exc}. Code that failed: {code}"
            ) from exc
        encoder = scope.get("encoder", encoder)
    elif encoder is None:
        encoder = get_default_tabular_encoder(df_train)

    if encoder is None:
        raise ValueError(
            "Encoding code must assign a sklearn-compatible transformer to `encoder`."
        )
    if not hasattr(encoder, "fit") or not hasattr(encoder, "transform"):
        raise ValueError(
            "The `encoder` object must define both `fit` and `transform` methods."
        )

    try:
        if fit_encoder:
            encoder.fit(df_train)
        encoded_train = encoder.transform(df_train)
        encoded_test = encoder.transform(df_test) if df_test is not None else None
    except Exception as exc:
        raise ValueError(f"Encoding pipeline execution failed! {exc}") from exc

    encoded_train = _coerce_feature_matrix(
        encoded_train,
        expected_rows=len(df_train),
        dataset_name="train",
    )
    encoded_test = (
        _coerce_feature_matrix(
            encoded_test,
            expected_rows=len(df_test),
            dataset_name="test",
        )
        if df_test is not None
        else None
    )

    return encoded_train, encoded_test, encoder


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
        ast.Match,
        ast.match_case,
        ast.MatchValue,
        ast.MatchSingleton,
        ast.MatchSequence,
        ast.MatchMapping,
        ast.MatchClass,
        ast.MatchStar,
        ast.MatchAs,
        ast.MatchOr,
        # These nodes represent loop structures. If you allow arbitrary loops, a user could potentially create an infinite loop that consumes system resources and slows down or crashes your system.
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Assert,
        ast.Try,
        ast.ExceptHandler,
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

    allowed_package_roots = {
        "collections",
        "datetime",
        "functools",
        "itertools",
        "math",
        "numpy",
        "pandas",
        "re",
        "scipy",
        "sklearn",
        "statistics",
        "typing",
    }

    dangerous_call_names = {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "quit",
        "setattr",
        "vars",
    }

    dangerous_attrs = {
        "__class__",
        "__dict__",
        "__globals__",
        "__subclasses__",
        "check_call",
        "check_output",
        "chmod",
        "chown",
        "dump",
        "dumps",
        "exec_module",
        "fromfile",
        "kill",
        "listdir",
        "load",
        "loads",
        "makedirs",
        "mkdir",
        "popen",
        "read_bytes",
        "read_pickle",
        "read_sql",
        "remove",
        "removedirs",
        "rename",
        "rmdir",
        "run",
        "runcall",
        "rmtree",
        "save",
        "savetxt",
        "sleep",
        "symlink",
        "system",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_file",
        "to_hdf",
        "to_json",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "tofile",
        "touch",
        "unlink",
        "walk",
        "write_bytes",
        "write_text",
    }

    if type(node) not in allowed_nodes:
        raise ValueError(f"Disallowed code: {ast.unparse(node)} is {type(node)}")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in dangerous_call_names:
            raise ValueError(f"Disallowed function: {node.func.id}")

    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__") or node.attr in dangerous_attrs:
            raise ValueError(f"Disallowed attribute: {node.attr}")

    if isinstance(node, ast.Import):
        for alias in node.names:
            root_name = alias.name.split(".")[0]
            if root_name not in allowed_package_roots:
                raise ValueError(f"Disallowed package import: {alias.name}")

    if isinstance(node, ast.ImportFrom):
        module_name = node.module or ""
        root_name = module_name.split(".")[0]
        if root_name not in allowed_package_roots:
            raise ValueError(f"Disallowed package import: {module_name}")

    for child in ast.iter_child_nodes(node):
        check_ast(child)