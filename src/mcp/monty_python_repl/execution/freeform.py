"""Freeform dataframe execution helpers for the Monty Python REPL."""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier, LGBMRegressor
from optuna.samplers import TPESampler
from sklearn import (
    base,
    compose,
    feature_extraction,
    feature_selection,
    impute,
    metrics,
    model_selection,
    pipeline,
    preprocessing,
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "joblib",
        "json",
        "lightgbm",
        "pathlib",
        "collections",
        "datetime",
        "functools",
        "itertools",
        "math",
        "numpy",
        "optuna",
        "pandas",
        "re",
        "scipy",
        "sklearn",
        "statistics",
        "typing",
    }
)
_DANGEROUS_CALL_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "input",
        "quit",
    }
)
_DANGEROUS_ATTRS = frozenset(
    {
        "__class__",
        "__dict__",
        "__globals__",
        "__subclasses__",
        "check_call",
        "check_output",
        "chmod",
        "chown",
        "exec_module",
        "fromfile",
        "kill",
        "popen",
        "removedirs",
        "rmdir",
        "runcall",
        "rmtree",
        "sleep",
        "unlink",
        "walk",
        "write_bytes",
    }
)
_FORBIDDEN_NODE_TYPES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
)


@dataclass(slots=True)
class FreeformExecutionResult:
    """Structured result returned by the freeform dataframe executor.

    Attributes:
        dataframe: Executed dataframe snapshot ready for persistence.
        rows: Row count for the resulting dataframe.
        columns: Resulting dataframe column names.
        columns_added: Columns introduced by the submitted code.
        columns_removed: Columns removed by the submitted code.
        stdout: All standard output captured while executing the submitted code.
    """

    dataframe: pd.DataFrame
    rows: int
    columns: list[str]
    columns_added: list[str]
    columns_removed: list[str]
    stdout: str


class FreeformCodeError(ValueError):
    """Raised when freeform dataframe code fails validation or execution."""

    def __init__(self, stage: str, message: str) -> None:
        """Initialize the error with a machine-friendly stage prefix.

        Args:
            stage: High-level execution stage such as ``validation_error``.
            message: Human-readable error details for the LLM or caller.
        """
        self.stage = stage
        super().__init__(f"{stage}: {message}")


class FreeformDataframeExecutor:
    """Execute validated freeform Python against a pandas dataframe.

    The executor keeps the contract intentionally small: a caller supplies a
    dataframe plus Python source, the code operates on ``df``, and the final
    dataframe must still be assigned back to ``df``.

    Example:
        >>> executor = FreeformDataframeExecutor()
        >>> frame = pd.DataFrame({"premium": [10, 15], "loss": [1, 2]})
        >>> freeform_code = '''
        ... df["margin"] = df["premium"] - df["loss"]
        ... print("")
        ... print("Created margin")
        ... '''
        >>> result = executor.execute(
        ...     freeform_code,
        ...     frame,
        ... )
        >>> result.columns_added
        ['margin']
    """

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize the executor with an optional host workspace root.

        Args:
            workspace_root: Concrete host directory that backs `/workspace`.
                When omitted, relative paths still resolve from the current
                process working directory.
        """
        self._workspace_root = workspace_root.resolve() if workspace_root else None

    def execute(
        self,
        code: str,
        dataframe: pd.DataFrame,
        *,
        extra_scope: dict[str, Any] | None = None,
    ) -> FreeformExecutionResult:
        """Run freeform Python against a dataframe copy and return the result.

        Args:
            code: Python source that reads or mutates ``df``.
            dataframe: Input dataframe to expose as ``df``.
            extra_scope: Optional additional symbols to expose to the execution
                scope, such as explicit transformer args. When this code is
                itself nested inside another Python string, prefer passing a
                named multiline variable and avoid escape-heavy snippets like
                ``print(f'\\n...')`` unless the literal backslash is meant to
                survive outer parsing.

        Returns:
            FreeformExecutionResult: Result dataframe and summary metadata.

        Raises:
            FreeformCodeError: If the code cannot be parsed, validated, or run.
        """
        tree = self._parse(code)
        self._validate_tree(tree)

        original_columns = [str(column) for column in dataframe.columns]
        working_dataframe = dataframe.copy()
        execution_scope = self._build_scope(working_dataframe, extra_scope=extra_scope)
        stdout_buffer = io.StringIO()

        try:
            # Capture every print emitted inside the freeform exec block so the
            # caller can inspect the full execution log afterward.
            with contextlib.redirect_stdout(stdout_buffer):
                exec(
                    compile(tree, filename="<monty_freeform>", mode="exec"),
                    execution_scope,
                    execution_scope,
                )
        except Exception as exc:  # pragma: no cover - exercised in tests
            raise FreeformCodeError(
                "runtime_error",
                self._format_runtime_error(exc, stdout=stdout_buffer.getvalue()),
            ) from exc

        if "df" not in execution_scope:
            raise FreeformCodeError(
                "postcondition_error",
                "Code removed `df` from scope. Assign the final dataframe back to `df`.",
            )

        result_dataframe = execution_scope["df"]
        if not isinstance(result_dataframe, pd.DataFrame):
            raise FreeformCodeError(
                "postcondition_error",
                "`df` must remain a pandas.DataFrame after execution.",
            )

        result_dataframe = result_dataframe.copy()
        result_columns = [str(column) for column in result_dataframe.columns]
        return FreeformExecutionResult(
            dataframe=result_dataframe,
            rows=int(result_dataframe.shape[0]),
            columns=result_columns,
            columns_added=[
                column for column in result_columns if column not in original_columns
            ],
            columns_removed=[
                column for column in original_columns if column not in result_columns
            ],
            stdout=stdout_buffer.getvalue(),
        )

    def _parse(self, code: str) -> ast.Module:
        """Parse user code and raise a stage-aware syntax error when invalid.

        Args:
            code: User-supplied Python source.

        Returns:
            ast.Module: Parsed AST for the source.

        Raises:
            FreeformCodeError: If the code is not valid Python syntax.
        """
        try:
            return ast.parse(code)
        except SyntaxError as exc:
            line_suffix = f" on line {exc.lineno}" if exc.lineno else ""
            raise FreeformCodeError(
                "syntax_error",
                f"{exc.msg}{line_suffix}.",
            ) from exc

    def _validate_tree(self, tree: ast.AST) -> None:
        """Validate that the AST stays within the dataframe tool contract.

        Args:
            tree: Parsed user code.

        Raises:
            FreeformCodeError: If disallowed syntax, imports, or calls appear.
        """
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN_NODE_TYPES):
                raise FreeformCodeError(
                    "validation_error",
                    f"Unsupported syntax `{type(node).__name__}`.",
                )

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _DANGEROUS_CALL_NAMES:
                    raise FreeformCodeError(
                        "validation_error",
                        f"Disallowed function call `{node.func.id}`.",
                    )

            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") or node.attr in _DANGEROUS_ATTRS:
                    raise FreeformCodeError(
                        "validation_error",
                        f"Disallowed attribute access `{node.attr}`.",
                    )

            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._validate_import_root(alias.name)

            if isinstance(node, ast.ImportFrom):
                if node.level != 0:
                    raise FreeformCodeError(
                        "validation_error",
                        "Relative imports are not allowed.",
                    )
                self._validate_import_root(node.module or "")

    def _validate_import_root(self, module_name: str) -> None:
        """Validate an import root against the approved package allowlist.

        Args:
            module_name: Imported module path, such as ``numpy.random``.

        Raises:
            FreeformCodeError: If the root package is outside the allowlist.
        """
        root_name = module_name.split(".")[0]
        if root_name not in _ALLOWED_IMPORT_ROOTS:
            raise FreeformCodeError(
                "validation_error",
                f"Disallowed package import `{module_name}`.",
            )

    def _resolve_workspace_path(self, path: str | Path) -> Path:
        """Translate a virtual or relative workspace path into a host path.

        Args:
            path: Relative path, host path, or virtual `/workspace/...` path.

        Returns:
            Path: Host-resolved path suitable for pandas/joblib IO.
        """
        candidate = Path(path)
        if self._workspace_root is None:
            return candidate

        raw_path = str(path)
        if raw_path.startswith("/workspace"):
            relative = PurePosixPath(raw_path).relative_to("/workspace")
            return (self._workspace_root / Path(relative.as_posix())).resolve(
                strict=False
            )
        if candidate.is_absolute():
            return candidate
        return (self._workspace_root / candidate).resolve(strict=False)

    def _workspace_path(self, path: str | Path) -> Path:
        """Return a host path for a virtual `/workspace` location.

        Args:
            path: Relative path, host path, or virtual `/workspace/...` path.

        Returns:
            Path: Host-resolved path suitable for pandas, joblib, and similar IO.

        Example:
            >>> executor = FreeformDataframeExecutor(workspace_root=Path("/tmp/workspace"))
            >>> executor._workspace_path("/workspace/data.csv")
            PosixPath('/tmp/workspace/data.csv')
        """
        return self._resolve_workspace_path(path)

    def _build_scope(
        self,
        dataframe: pd.DataFrame,
        *,
        extra_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the execution scope exposed to freeform code.

        Args:
            dataframe: Working dataframe copy exposed as ``df``.
            extra_scope: Optional additional values injected into the scope.

        Returns:
            dict[str, Any]: Shared globals/locals namespace for ``exec``.
        """
        workspace_dir = self._workspace_root or Path.cwd()
        scope = {
            "__builtins__": self._build_safe_builtins(),
            "__name__": "__monty_freeform__",
            "df": dataframe,
            "base": base,
            "ColumnTransformer": ColumnTransformer,
            "pd": pd,
            "np": np,
            "json": json,
            "joblib": joblib,
            "lgb": lgb,
            "lightgbm": lgb,
            "LGBMClassifier": LGBMClassifier,
            "LGBMRegressor": LGBMRegressor,
            "math": math,
            "optuna": optuna,
            "Path": Path,
            "TPESampler": TPESampler,
            "scipy": scipy,
            "sklearn": sklearn,
            "compose": compose,
            "feature_extraction": feature_extraction,
            "feature_selection": feature_selection,
            "impute": impute,
            "metrics": metrics,
            "model_selection": model_selection,
            "pipeline": pipeline,
            "preprocessing": preprocessing,
            "FunctionTransformer": FunctionTransformer,
            "OrdinalEncoder": OrdinalEncoder,
            "Pipeline": Pipeline,
            "SelectFromModel": SelectFromModel,
            "SimpleImputer": SimpleImputer,
            "StratifiedKFold": StratifiedKFold,
            "train_test_split": train_test_split,
            "workspace_dir": workspace_dir,
            "WORKSPACE_DIR": workspace_dir,
            "workspace_path": self._workspace_path,
            "resolve_workspace_path": self._resolve_workspace_path,
        }
        if extra_scope:
            scope.update(extra_scope)
        return scope

    def _build_safe_builtins(self) -> dict[str, Any]:
        """Return the limited builtin namespace used during execution.

        Returns:
            dict[str, Any]: Approved builtin functions, types, and exceptions.
        """
        # Keep the builtin surface intentionally small while still supporting
        # normal dataframe expressions, helper functions, and basic debugging.
        return {
            "__import__": self._safe_import,
            "Exception": Exception,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "RuntimeError": RuntimeError,
            "TypeError": TypeError,
            "ValueError": ValueError,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "callable": callable,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "getattr": self._safe_getattr,
            "hasattr": self._safe_hasattr,
            "int": int,
            "iter": iter,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "slice": slice,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
        }

    def _validate_builtin_attribute_name(self, attr_name: str) -> None:
        """Reject unsafe attribute names used by builtin helper wrappers.

        Args:
            attr_name: Attribute name supplied to ``getattr`` or ``hasattr``.

        Raises:
            AttributeError: If the attribute name targets blocked introspection.
            TypeError: If the supplied attribute name is not a string.
        """
        if not isinstance(attr_name, str):
            raise TypeError("attribute name must be string")
        if attr_name.startswith("__") or attr_name in _DANGEROUS_ATTRS:
            raise AttributeError(f"Disallowed attribute access `{attr_name}`.")

    def _safe_getattr(
        self,
        obj: Any,
        attr_name: str,
        default: Any = ...,
    ) -> Any:
        """Return a safe ``getattr`` result for approved attribute names.

        Args:
            obj: Object to inspect.
            attr_name: Requested attribute name.
            default: Optional default value returned when the attribute is absent.

        Returns:
            Any: The requested attribute value or the supplied default.
        """
        self._validate_builtin_attribute_name(attr_name)
        if default is ...:
            return getattr(obj, attr_name)
        return getattr(obj, attr_name, default)

    def _safe_hasattr(self, obj: Any, attr_name: str) -> bool:
        """Return ``True`` when an approved attribute exists on an object.

        Args:
            obj: Object to inspect.
            attr_name: Requested attribute name.

        Returns:
            bool: Whether the approved attribute exists on ``obj``.
        """
        self._validate_builtin_attribute_name(attr_name)
        return hasattr(obj, attr_name)

    def _safe_import(
        self,
        name: str,
        globals_dict: dict[str, Any] | None = None,
        locals_dict: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Guard Python imports so only approved package roots are available.

        Args:
            name: Module requested by the import statement.
            globals_dict: Standard ``__import__`` global namespace parameter.
            locals_dict: Standard ``__import__`` local namespace parameter.
            fromlist: Imported names requested by ``from x import y``.
            level: Relative import level.

        Returns:
            Any: Imported module object.

        Raises:
            ImportError: If a disallowed import is requested.
        """
        del globals_dict, locals_dict
        if level != 0:
            raise ImportError("Relative imports are not allowed.")

        root_name = name.split(".")[0]
        if root_name not in _ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"Disallowed package import `{name}`.")

        return builtins.__import__(name, globals(), locals(), fromlist, level)

    def _format_runtime_error(self, exc: Exception, stdout: str = "") -> str:
        """Build a compact runtime error message for LLM retry loops.

        Args:
            exc: The original runtime exception.
            stdout: Captured standard output emitted before the failure.

        Returns:
            str: Condensed error string with a small traceback tail.
        """
        traceback_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        traceback_tail = "".join(traceback_lines[-4:]).strip()
        message = f"{type(exc).__name__}: {exc}"
        if stdout:
            message = f"{message}\nCaptured stdout:\n{stdout.rstrip()}"
        if traceback_tail:
            return f"{message}\nTraceback:\n{traceback_tail}"
        return message


__all__ = [
    "FreeformCodeError",
    "FreeformDataframeExecutor",
    "FreeformExecutionResult",
]
