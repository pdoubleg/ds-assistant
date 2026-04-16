"""Freeform dataframe execution helpers for the hackathon Monty REPL."""

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
import sklearn
from lightgbm import LGBMClassifier, LGBMRegressor
from optuna.samplers import TPESampler
from sklearn import base, compose, feature_selection, impute, metrics, model_selection
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from ..privacy import sanitize_exception, summarize_stdout

_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "collections",
        "datetime",
        "functools",
        "itertools",
        "joblib",
        "json",
        "lightgbm",
        "math",
        "numpy",
        "optuna",
        "pandas",
        "pathlib",
        "re",
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
        "open",
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
    """Structured result returned by the freeform executor.

    Attributes:
        dataframe: Final dataframe snapshot.
        rows: Resulting row count.
        columns: Result column names.
        columns_added: Columns introduced by the code.
        columns_removed: Columns removed by the code.
        stdout_summary: Privacy-safe stdout summary.
    """

    dataframe: pd.DataFrame
    rows: int
    columns: list[str]
    columns_added: list[str]
    columns_removed: list[str]
    stdout_summary: dict[str, Any]


class FreeformCodeError(ValueError):
    """Raised when freeform dataframe code fails validation or execution."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        error_type: str | None = None,
        line_number: int | None = None,
    ) -> None:
        """Initialize the sanitized freeform error.

        Args:
            stage: High-level failure stage.
            message: Privacy-safe error message.
            error_type: Optional exception type name.
            line_number: Optional line number.
        """
        self.stage = stage
        self.error_type = error_type
        self.line_number = line_number
        super().__init__(message)


class FreeformDataframeExecutor:
    """Execute validated freeform Python against a pandas dataframe.

    Example:
        >>> executor = FreeformDataframeExecutor()
        >>> frame = pd.DataFrame({"premium": [10, 15], "loss": [1, 2]})
        >>> result = executor.execute(
        ...     "df['margin'] = df['premium'] - df['loss']",
        ...     frame,
        ... )
        >>> result.columns_added
        ['margin']
    """

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize the executor.

        Args:
            workspace_root: Optional host directory backing `/workspace`.
        """
        self._workspace_root = workspace_root.resolve() if workspace_root else None

    def execute(
        self,
        code: str,
        dataframe: pd.DataFrame,
        *,
        extra_scope: dict[str, Any] | None = None,
    ) -> FreeformExecutionResult:
        """Run freeform Python against a dataframe copy.

        Args:
            code: Python source operating on ``df``.
            dataframe: Input dataframe exposed as ``df``.
            extra_scope: Optional additional symbols available in the runtime.

        Returns:
            Structured execution result with privacy-safe stdout metadata.

        Raises:
            FreeformCodeError: If parsing, validation, or runtime execution fails.
        """
        tree = self._parse(code)
        self._validate_tree(tree)

        original_columns = [str(column) for column in dataframe.columns]
        working_dataframe = dataframe.copy()
        execution_scope = self._build_scope(
            working_dataframe,
            extra_scope=extra_scope,
        )
        stdout_buffer = io.StringIO()

        try:
            # Capture stdout so the REPL can report that output existed without
            # exposing row values or printed examples back to the model.
            with contextlib.redirect_stdout(stdout_buffer):
                exec(
                    compile(tree, filename="<monty_freeform>", mode="exec"),
                    execution_scope,
                    execution_scope,
                )
        except Exception as exc:  # pragma: no cover - covered indirectly in tests
            sanitized = sanitize_exception(
                exc,
                traceback_text=traceback.format_exc(),
            )
            raise FreeformCodeError(
                "runtime_error",
                sanitized["message"],
                error_type=sanitized.get("error_type"),
                line_number=sanitized.get("line_number"),
            ) from exc

        if "df" not in execution_scope:
            raise FreeformCodeError(
                "postcondition_error",
                "Freeform code must leave the final dataframe assigned to `df`.",
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
            stdout_summary=summarize_stdout(stdout_buffer.getvalue()),
        )

    def _parse(self, code: str) -> ast.Module:
        """Parse user code into an AST.

        Args:
            code: User-supplied Python source.

        Returns:
            Parsed module AST.

        Raises:
            FreeformCodeError: If the code is syntactically invalid.
        """
        try:
            return ast.parse(code)
        except SyntaxError as exc:
            raise FreeformCodeError(
                "syntax_error",
                "Freeform code contains invalid Python syntax.",
                error_type="SyntaxError",
                line_number=exc.lineno,
            ) from exc

    def _validate_tree(self, tree: ast.AST) -> None:
        """Validate the AST against the allowed freeform contract.

        Args:
            tree: Parsed AST.

        Raises:
            FreeformCodeError: If disallowed syntax or imports are detected.
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
        """Validate an import root against the allowlist.

        Args:
            module_name: Imported module path.

        Raises:
            FreeformCodeError: If the import root is not allowed.
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
            Host-resolved path.
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
            path: Relative or virtual workspace path.

        Returns:
            Host path.
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
            extra_scope: Optional injected symbols.

        Returns:
            Execution namespace used for ``exec``.
        """
        workspace_dir = self._workspace_root or Path.cwd()
        scope = {
            "__builtins__": self._build_safe_builtins(),
            "__name__": "__monty_freeform__",
            "df": dataframe,
            "base": base,
            "ColumnTransformer": ColumnTransformer,
            "FunctionTransformer": FunctionTransformer,
            "LGBMClassifier": LGBMClassifier,
            "LGBMRegressor": LGBMRegressor,
            "OrdinalEncoder": OrdinalEncoder,
            "Path": Path,
            "Pipeline": Pipeline,
            "SimpleImputer": SimpleImputer,
            "TPESampler": TPESampler,
            "compose": compose,
            "feature_selection": feature_selection,
            "impute": impute,
            "joblib": joblib,
            "json": json,
            "lgb": lgb,
            "lightgbm": lgb,
            "math": math,
            "metrics": metrics,
            "model_selection": model_selection,
            "np": np,
            "optuna": optuna,
            "pd": pd,
            "sklearn": sklearn,
            "workspace_dir": workspace_dir,
            "workspace_path": self._workspace_path,
            "resolve_workspace_path": self._resolve_workspace_path,
        }
        if extra_scope:
            scope.update(extra_scope)
        return scope

    def _build_safe_builtins(self) -> dict[str, Any]:
        """Return the limited builtin namespace exposed to freeform code.

        Returns:
            Approved builtin functions, types, and exceptions.
        """
        return {
            "__import__": self._safe_import,
            "Exception": Exception,
            "KeyError": KeyError,
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
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
        }

    def _validate_builtin_attribute_name(self, attr_name: str) -> None:
        """Reject unsafe attribute names.

        Args:
            attr_name: Attribute name used by builtin wrappers.

        Raises:
            AttributeError: If the attribute is blocked.
        """
        if not isinstance(attr_name, str):
            raise TypeError("attribute name must be a string")
        if attr_name.startswith("__") or attr_name in _DANGEROUS_ATTRS:
            raise AttributeError(f"Disallowed attribute access `{attr_name}`.")

    def _safe_getattr(
        self,
        obj: Any,
        attr_name: str,
        default: Any = ...,
    ) -> Any:
        """Return a safe ``getattr`` result for approved attribute names."""
        self._validate_builtin_attribute_name(attr_name)
        if default is ...:
            return getattr(obj, attr_name)
        return getattr(obj, attr_name, default)

    def _safe_hasattr(self, obj: Any, attr_name: str) -> bool:
        """Return ``True`` when an approved attribute exists on an object."""
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
        """Guard Python imports so only approved package roots are available."""
        del globals_dict, locals_dict
        if level != 0:
            raise ImportError("Relative imports are not allowed.")
        root_name = name.split(".")[0]
        if root_name not in _ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"Disallowed package import `{name}`.")
        return builtins.__import__(name, globals(), locals(), fromlist, level)


class FreeformDataframeTransformer(base.BaseEstimator, base.TransformerMixin):
    """Reusable sklearn-style transformer backed by freeform code."""

    def __init__(
        self,
        code: str,
        *,
        workspace_root: str | None = None,
        params: dict[str, Any] | None = None,
        preserve_index: bool = True,
        strict_schema: bool = True,
    ) -> None:
        """Initialize the transformer.

        Args:
            code: Freeform source code operating on ``df``.
            workspace_root: Host workspace directory string.
            params: Explicit parameters exposed to the code as ``params``.
            preserve_index: Whether to preserve the input index.
            strict_schema: Whether transform-time schema must match fit-time schema.
        """
        self.code = code
        self.workspace_root = workspace_root
        self.params = dict(params or {})
        self.preserve_index = preserve_index
        self.strict_schema = strict_schema

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> FreeformDataframeTransformer:
        """Fit the transformer by executing the code once on the input frame.

        Args:
            X: Training features.
            y: Optional ignored target.

        Returns:
            Fitted transformer.
        """
        del y
        executor = FreeformDataframeExecutor(
            workspace_root=Path(self.workspace_root)
            if self.workspace_root is not None
            else None
        )
        result = executor.execute(
            self.code,
            X.copy(),
            extra_scope={"params": dict(self.params)},
        )
        self.input_columns_ = [str(column) for column in X.columns]
        self.output_columns_ = list(result.columns)
        self.columns_added_ = list(result.columns_added)
        self.columns_removed_ = list(result.columns_removed)
        self.fit_stdout_summary_ = dict(result.stdout_summary)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe using the fitted freeform code.

        Args:
            X: Input dataframe.

        Returns:
            Transformed dataframe.
        """
        executor = FreeformDataframeExecutor(
            workspace_root=Path(self.workspace_root)
            if self.workspace_root is not None
            else None
        )
        result = executor.execute(
            self.code,
            X.copy(),
            extra_scope={"params": dict(self.params)},
        )
        transformed = result.dataframe.copy()
        if self.strict_schema and list(transformed.columns) != list(
            self.output_columns_
        ):
            raise ValueError(
                "Transformed columns do not match the schema learned during fit."
            )
        if self.preserve_index:
            transformed.index = X.index
        return transformed


__all__ = [
    "FreeformCodeError",
    "FreeformDataframeExecutor",
    "FreeformDataframeTransformer",
    "FreeformExecutionResult",
]
