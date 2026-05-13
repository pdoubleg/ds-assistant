"""Focused Monty interpreter for REPL-style execution."""

from __future__ import annotations

import ast
import asyncio
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import pydantic_monty

from src.rlm.types import CodeExecutionError

from .core.registry.base import RegisteredFunction


class _TopLevelNameCollector:
    """Collect global-scope assignment and deletion targets from module code."""

    def __init__(self) -> None:
        """Initialize empty name collections."""
        self.assigned_names: set[str] = set()
        self.deleted_names: set[str] = set()

    def collect(
        self,
        code: str,
        *,
        safe_call_names: set[str],
    ) -> tuple[list[str], list[str]]:
        """Parse code and return top-level assigned and deleted names."""
        self.assigned_names = set()
        self.deleted_names = set()
        module = ast.parse(code)
        for statement in module.body:
            self._visit_statement(statement, safe_call_names=safe_call_names)
        return sorted(self.assigned_names), sorted(self.deleted_names)

    def _visit_statement(
        self, statement: ast.stmt, *, safe_call_names: set[str]
    ) -> None:
        """Visit a statement that executes in module scope."""
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                self._record_target(target, assigned=True)
            return
        if isinstance(statement, ast.AnnAssign):
            if statement.value:
                self._record_target(statement.target, assigned=True)
            return
        if isinstance(statement, ast.AugAssign):
            self._record_target(statement.target, assigned=True)
            return
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            self._visit_block(statement.body, safe_call_names=safe_call_names)
            self._visit_block(statement.orelse, safe_call_names=safe_call_names)
            return
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            self._visit_block(statement.body, safe_call_names=safe_call_names)
            return
        if isinstance(statement, ast.If):
            self._visit_block(statement.body, safe_call_names=safe_call_names)
            self._visit_block(statement.orelse, safe_call_names=safe_call_names)
            return
        if isinstance(statement, ast.While):
            self._visit_block(statement.body, safe_call_names=safe_call_names)
            self._visit_block(statement.orelse, safe_call_names=safe_call_names)
            return
        if isinstance(statement, ast.Try):
            self._visit_block(statement.body, safe_call_names=safe_call_names)
            self._visit_block(statement.orelse, safe_call_names=safe_call_names)
            self._visit_block(statement.finalbody, safe_call_names=safe_call_names)
            for handler in statement.handlers:
                self._visit_block(handler.body, safe_call_names=safe_call_names)
            return
        if isinstance(statement, ast.Match):
            for case in statement.cases:
                if case.pattern is not None:
                    self._record_pattern(case.pattern)
                self._visit_block(case.body, safe_call_names=safe_call_names)
            return
        if isinstance(statement, ast.Delete):
            for target in statement.targets:
                self._record_target(target, assigned=False)
            return
        if isinstance(
            statement,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return

    def _visit_block(
        self, statements: list[ast.stmt], *, safe_call_names: set[str]
    ) -> None:
        """Visit a sequence of module-scope statements."""
        for statement in statements:
            self._visit_statement(statement, safe_call_names=safe_call_names)

    def _record_pattern(self, pattern: ast.pattern) -> None:
        """Record names bound by a ``match`` pattern."""
        for node in ast.walk(pattern):
            if isinstance(node, ast.MatchAs) and node.name:
                self.assigned_names.add(node.name)
            elif isinstance(node, ast.MatchStar) and node.name:
                self.assigned_names.add(node.name)

    def _record_target(self, target: ast.expr, *, assigned: bool) -> None:
        """Record names from an assignment or deletion target."""
        destination = self.assigned_names if assigned else self.deleted_names
        if isinstance(target, ast.Name):
            destination.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._record_target(element, assigned=assigned)

    def _is_persistable_expression(
        self,
        expression: ast.expr,
        safe_call_names: set[str],
    ) -> bool:
        """Return ``True`` when an expression is likely safe to persist."""
        if isinstance(expression, (ast.Constant, ast.Name)):
            return True
        if isinstance(expression, ast.JoinedStr):
            return all(
                not isinstance(value, ast.FormattedValue)
                or self._is_persistable_expression(value.value, safe_call_names)
                for value in expression.values
            )
        if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
            return all(
                self._is_persistable_expression(element, safe_call_names)
                for element in expression.elts
            )
        if isinstance(expression, ast.Dict):
            return all(
                (key is None or self._is_persistable_expression(key, safe_call_names))
                and self._is_persistable_expression(value, safe_call_names)
                for key, value in zip(expression.keys, expression.values)
            )
        if isinstance(expression, ast.UnaryOp):
            return self._is_persistable_expression(expression.operand, safe_call_names)
        if isinstance(expression, ast.BinOp):
            return self._is_persistable_expression(
                expression.left, safe_call_names
            ) and self._is_persistable_expression(expression.right, safe_call_names)
        if isinstance(expression, ast.BoolOp):
            return all(
                self._is_persistable_expression(value, safe_call_names)
                for value in expression.values
            )
        if isinstance(expression, ast.Compare):
            return self._is_persistable_expression(
                expression.left, safe_call_names
            ) and all(
                self._is_persistable_expression(comparator, safe_call_names)
                for comparator in expression.comparators
            )
        if isinstance(expression, ast.IfExp):
            return all(
                self._is_persistable_expression(item, safe_call_names)
                for item in (expression.test, expression.body, expression.orelse)
            )
        if isinstance(expression, ast.Subscript):
            return self._is_persistable_expression(expression.value, safe_call_names)
        if isinstance(expression, ast.Call):
            return (
                isinstance(expression.func, ast.Name)
                and expression.func.id in safe_call_names
            )
        return False


@dataclass(slots=True)
class InterpreterRunResult:
    """Structured result returned by the Monty REPL interpreter."""

    stdout: str
    persisted_names: list[str]
    persistence_failures: list[dict[str, str]]


class MontyReplInterpreter:
    """Monty interpreter backed by a persistent ``MontyRepl`` snapshot."""

    _persist_tool_name = "monty_repl_persist"
    _delete_tool_name = "monty_repl_delete"
    _persist_error_tool_name = "monty_repl_persist_error"
    _legacy_persist_tool_name = "__monty_repl_persist__"
    _legacy_delete_tool_name = "__monty_repl_delete__"
    _legacy_persist_error_tool_name = "__monty_repl_persist_error__"

    def __init__(
        self,
        *,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        tool_entries: Mapping[str, RegisteredFunction] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        limits: pydantic_monty.ResourceLimits | None = None,
        os_access: pydantic_monty.AbstractOS | None = None,
    ) -> None:
        """Initialize the interpreter."""
        self._tool_entries: dict[str, RegisteredFunction] = (
            dict(tool_entries) if tool_entries else {}
        )
        self._tools: dict[str, Callable[..., Any]] = (
            {name: entry.func for name, entry in self._tool_entries.items()}
            if self._tool_entries
            else dict(tools) if tools else {}
        )
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self._limits = limits
        self._os_access = os_access
        self._state: dict[str, Any] = {}
        self._state_names: set[str] = set()
        self._name_collector = _TopLevelNameCollector()
        self._has_executed = False
        self._repl = self._create_repl()

    @staticmethod
    def _matches_monty_exception(
        exc: BaseException,
        exception_name: str,
    ) -> bool:
        """Return whether *exc* matches a Monty exception by export or name.

        Args:
            exc: Raised exception to inspect.
            exception_name: Expected Monty exception class name.

        Returns:
            True when the exception matches the requested Monty exception type.
        """
        exported_type = getattr(pydantic_monty, exception_name, None)
        if isinstance(exported_type, type) and issubclass(exported_type, BaseException):
            return isinstance(exc, exported_type)
        return type(exc).__name__ == exception_name

    def _raise_known_monty_error(
        self,
        exc: BaseException,
        traceback: TracebackType | None = None,
    ) -> None:
        """Translate Monty-specific exceptions into local interpreter errors.

        Args:
            exc: Raised exception from the Monty runtime.
            traceback: Optional traceback to preserve when re-raising.

        Raises:
            SyntaxError: When Monty reports a syntax error.
            CodeExecutionError: When Monty reports a typing or runtime error.
        """
        if self._matches_monty_exception(exc, "MontySyntaxError"):
            error = SyntaxError(str(exc))
        elif self._matches_monty_exception(exc, "MontyTypingError"):
            error = CodeExecutionError(str(exc))
        elif self._matches_monty_exception(exc, "MontyRuntimeError"):
            error = CodeExecutionError(str(exc))
        else:
            raise exc.with_traceback(traceback)
        raise error.with_traceback(traceback) from exc

    @property
    def state(self) -> dict[str, Any]:
        """Return a shallow copy of host-visible REPL variable values."""
        return dict(self._state)

    def _create_repl(self) -> pydantic_monty.MontyRepl:
        """Create a fresh Monty REPL with registry-derived type stubs.

        Returns:
            pydantic_monty.MontyRepl: A clean persistent interpreter instance.
        """
        return pydantic_monty.MontyRepl(
            limits=self._limits,
            type_check=self._type_check,
            type_check_stubs=self._build_type_check_stubs(),
        )

    def _build_type_check_stubs(self) -> str | None:
        """Build Monty type-check stubs for injected registry tools."""
        stub_lines: list[str] = []
        stub_lines.append("from typing import Any")
        for tool_name, func in self._tools.items():
            stub = self._render_tool_stub(tool_name, func)
            if stub:
                stub_lines.append(stub)
        stub_lines.extend(
            (
                f"def {self._persist_tool_name}(name: str, value: Any) -> None: ...",
                f"def {self._delete_tool_name}(name: str) -> None: ...",
                f"def {self._persist_error_tool_name}(name: str, error: str) -> None: ...",
            )
        )

        if self._type_check_stubs:
            if stub_lines:
                stub_lines.append("")
            stub_lines.append(self._type_check_stubs)

        return "\n".join(stub_lines) if stub_lines else None

    def _render_tool_stub(self, name: str, func: Callable[..., Any]) -> str | None:
        """Render one callable stub for Monty's type checker.

        Args:
            name: Exported sandbox helper name.
            func: Host callable exposed under ``name``.

        Returns:
            A Python stub line, or ``None`` when the name cannot be rendered as a
            Python function definition.
        """
        if not name.isidentifier():
            return None

        prefix = "async def" if inspect.iscoroutinefunction(func) else "def"
        entry = self._tool_entries.get(name)
        if entry is not None:
            return f"{prefix} {entry.render_signature(multiline=False)}: ..."
        return f"{prefix} {name}(*args: Any, **kwargs: Any) -> Any: ..."

    def _wrap_code(
        self,
        code: str,
        assigned_names: list[str],
        deleted_names: list[str],
    ) -> str:
        """Append hidden host-state capture calls after user code.

        Monty's own REPL keeps executable state such as imports and function
        definitions. These hidden calls maintain the host-side ``state`` mapping
        used by tests and diagnostic helpers without replaying variables into a
        fresh interpreter.
        """
        wrapped_lines = [code.rstrip(), ""]
        for name in assigned_names:
            wrapped_lines.extend(
                [
                    "try:",
                    f"    {self._persist_tool_name}({name!r}, {name})",
                    "except Exception as exc:",
                    f"    {self._persist_error_tool_name}({name!r}, str(exc).strip() or exc.__class__.__name__)",
                    "",
                ]
            )

        for name in deleted_names:
            wrapped_lines.append(f"{self._delete_tool_name}({name!r})")

        return "\n".join(wrapped_lines).strip() + "\n"

    def _start_monty(
        self,
        code: str,
        print_callback: Callable[[str, str], None],
        *,
        tools: Mapping[str, Callable[..., Any]] | None = None,
    ) -> (
        pydantic_monty.FunctionSnapshot
        | pydantic_monty.NameLookupSnapshot
        | pydantic_monty.FutureSnapshot
        | pydantic_monty.MontyComplete
    ):
        """Start Monty execution across nearby API versions."""
        external_tools = dict(tools) if tools is not None else self._tools
        start_kwargs: dict[str, Any] = {
            "inputs": external_tools or None,
            "print_callback": print_callback,
            "skip_type_check": self._type_check and self._has_executed,
        }
        if self._os_access is not None:
            start_kwargs["os"] = self._os_access

        try:
            return self._repl.feed_start(code, **start_kwargs)
        except TypeError as exc:
            if self._os_access is None or "os" not in str(exc):
                raise
            start_kwargs.pop("os", None)
            try:
                return self._repl.feed_start(code, **start_kwargs)
            except Exception as retry_exc:
                self._raise_known_monty_error(retry_exc, retry_exc.__traceback__)
                raise
        except Exception as exc:
            self._raise_known_monty_error(exc, exc.__traceback__)
            raise

    def _resume_function_snapshot(
        self,
        snapshot: pydantic_monty.FunctionSnapshot,
        result: pydantic_monty.ExternalResult,
    ) -> (
        pydantic_monty.FunctionSnapshot
        | pydantic_monty.NameLookupSnapshot
        | pydantic_monty.FutureSnapshot
        | pydantic_monty.MontyComplete
    ):
        """Resume a function snapshot across old and new Monty APIs."""
        resume_kwargs: dict[str, Any] = {}
        if self._os_access is not None:
            resume_kwargs["os"] = self._os_access

        try:
            return snapshot.resume(result, **resume_kwargs)
        except TypeError:
            if not result or len(result) != 1:
                raise
            key, value = next(iter(result.items()))
            if key not in {"return_value", "exception", "future"}:
                raise
            try:
                return snapshot.resume(**{key: value})
            except Exception as retry_exc:
                self._raise_known_monty_error(retry_exc, retry_exc.__traceback__)
                raise
        except Exception as exc:
            self._raise_known_monty_error(exc, exc.__traceback__)
            raise

    def _resume_future_snapshot(
        self,
        snapshot: pydantic_monty.FutureSnapshot,
        results: dict[int, pydantic_monty.ExternalResult],
    ) -> (
        pydantic_monty.FunctionSnapshot
        | pydantic_monty.NameLookupSnapshot
        | pydantic_monty.FutureSnapshot
        | pydantic_monty.MontyComplete
    ):
        """Resume a future snapshot across old and new Monty APIs."""
        resume_kwargs: dict[str, Any] = {}
        if self._os_access is not None:
            resume_kwargs["os"] = self._os_access

        try:
            return snapshot.resume(results, **resume_kwargs)
        except Exception as exc:
            self._raise_known_monty_error(exc, exc.__traceback__)
            raise

    async def execute(self, code: str) -> InterpreterRunResult:
        """Execute code in the persistent Monty REPL.

        Args:
            code: Python source to run in the sandbox.

        Returns:
            InterpreterRunResult: Captured stdout and top-level names assigned by
            the completed snippet.
        """
        assigned_names, deleted_names = self._name_collector.collect(
            code,
            safe_call_names=set(self._tools),
        )
        captured_state: dict[str, Any] = {}
        deleted_state_names: set[str] = set()
        persistence_failures: list[dict[str, str]] = []

        def _persist_variable(name: str, value: Any) -> None:
            captured_state[name] = value

        def _delete_variable(name: str) -> None:
            deleted_state_names.add(name)

        def _record_persist_failure(name: str, error: str) -> None:
            persistence_failures.append({"name": name, "error": error})

        _persist_variable.__name__ = self._persist_tool_name
        _delete_variable.__name__ = self._delete_tool_name
        _record_persist_failure.__name__ = self._persist_error_tool_name

        external_tools = dict(self._tools)
        external_tools[self._persist_tool_name] = _persist_variable
        external_tools[self._delete_tool_name] = _delete_variable
        external_tools[self._persist_error_tool_name] = _record_persist_failure
        external_tools[self._legacy_persist_tool_name] = _persist_variable
        external_tools[self._legacy_delete_tool_name] = _delete_variable
        external_tools[self._legacy_persist_error_tool_name] = _record_persist_failure
        wrapped_code = self._wrap_code(code, assigned_names, deleted_names)

        stdout_parts: list[str] = []

        def _capture_print(_stream: str, text: str) -> None:
            stdout_parts.append(text)

        progress = self._start_monty(wrapped_code, _capture_print, tools=external_tools)

        pending_tasks: dict[int, asyncio.Task[pydantic_monty.ExternalResult]] = {}

        async def _resolve_async_tool(result: Any) -> pydantic_monty.ExternalResult:
            try:
                return {"return_value": await result}
            except Exception as exc:  # pragma: no cover
                return {"exception": exc}

        try:
            while not isinstance(progress, pydantic_monty.MontyComplete):
                if isinstance(progress, pydantic_monty.NameLookupSnapshot):
                    resolved_name = external_tools.get(progress.variable_name)
                    if resolved_name is not None:
                        progress = progress.resume(value=resolved_name)
                    else:
                        progress = progress.resume()
                    continue

                if isinstance(progress, pydantic_monty.FunctionSnapshot):
                    if progress.is_os_function:
                        if self._os_access is None:
                            raise CodeExecutionError(
                                f"OS function {progress.function_name} called "
                                "but no OS access handler is configured"
                            )
                        try:
                            result = self._os_access(
                                progress.function_name,
                                progress.args,
                                progress.kwargs,
                            )
                        except Exception as exc:
                            raise CodeExecutionError(
                                f"OS function {progress.function_name} failed: {exc}"
                            ) from exc
                        progress = self._resume_function_snapshot(
                            progress,
                            {"return_value": result},
                        )
                        continue

                    func = external_tools.get(progress.function_name)
                    if func is None:
                        progress = self._resume_function_snapshot(
                            progress,
                            {
                                "exception": NameError(
                                    f"Unknown function: {progress.function_name}"
                                )
                            },
                        )
                        continue

                    try:
                        result = func(*progress.args, **progress.kwargs)
                    except Exception as exc:
                        progress = self._resume_function_snapshot(
                            progress,
                            {"exception": exc},
                        )
                        continue

                    if inspect.iscoroutine(result):
                        pending_tasks[progress.call_id] = asyncio.create_task(
                            _resolve_async_tool(result)
                        )
                        progress = self._resume_function_snapshot(
                            progress,
                            {"future": ...},
                        )
                        continue

                    progress = self._resume_function_snapshot(
                        progress,
                        {"return_value": result},
                    )
                    continue

                if isinstance(progress, pydantic_monty.FutureSnapshot):
                    results: dict[int, pydantic_monty.ExternalResult] = {}
                    gather_ids = [
                        call_id
                        for call_id in progress.pending_call_ids
                        if call_id in pending_tasks
                    ]
                    missing_ids = [
                        call_id
                        for call_id in progress.pending_call_ids
                        if call_id not in pending_tasks
                    ]
                    for call_id in missing_ids:
                        results[call_id] = {
                            "exception": RuntimeError(
                                f"No pending async tool result for call id {call_id}."
                            )
                        }

                    if gather_ids:
                        settled = await asyncio.gather(
                            *(pending_tasks[call_id] for call_id in gather_ids),
                            return_exceptions=True,
                        )
                        for call_id in gather_ids:
                            pending_tasks.pop(call_id, None)
                        for call_id, outcome in zip(gather_ids, settled):
                            if isinstance(outcome, Exception):
                                results[call_id] = {"exception": outcome}
                            elif isinstance(outcome, BaseException):  # pragma: no cover
                                raise outcome
                            else:
                                results[call_id] = outcome

                    progress = self._resume_future_snapshot(progress, results)
                    continue

                raise CodeExecutionError(
                    f"Unexpected Monty progress type: {type(progress).__name__}"
                )
        except CodeExecutionError:
            raise
        except Exception as exc:
            self._raise_known_monty_error(exc, exc.__traceback__)
            raise
        finally:
            for task in pending_tasks.values():
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks.values(), return_exceptions=True)

        self._state_names.update(assigned_names)
        for deleted_name in deleted_names:
            self._state_names.discard(deleted_name)
        self._state.update(captured_state)
        for deleted_name in deleted_state_names:
            self._state.pop(deleted_name, None)
        self._has_executed = True

        return InterpreterRunResult(
            stdout="".join(stdout_parts),
            persisted_names=sorted(captured_state),
            persistence_failures=persistence_failures,
        )
