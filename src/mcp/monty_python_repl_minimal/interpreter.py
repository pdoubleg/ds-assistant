"""Focused Monty interpreter for standalone REPL-style execution."""

from __future__ import annotations

import ast
import asyncio
import inspect
import keyword
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import pydantic_monty

from .exceptions import CodeExecutionError
from .privacy import safe_json_value


class _TopLevelNameCollector:
    """Collect global-scope assignment and deletion targets from module code."""

    def __init__(self) -> None:
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

    def _visit_block(
        self, statements: list[ast.stmt], *, safe_call_names: set[str]
    ) -> None:
        for statement in statements:
            self._visit_statement(statement, safe_call_names=safe_call_names)

    def _record_pattern(self, pattern: ast.pattern) -> None:
        """Record names bound by a `match` pattern."""
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


@dataclass(slots=True)
class InterpreterRunResult:
    """Structured result returned by the Monty REPL interpreter."""

    stdout: str
    persisted_names: list[str]
    persisted_value_summaries: dict[str, Any]
    last_expression_summary: Any | None
    persistence_failures: list[dict[str, str]]


class MontyReplInterpreter:
    """Monty interpreter with automatic variable persistence."""

    _persist_tool_name = "__monty_repl_persist__"
    _delete_tool_name = "__monty_repl_delete__"
    _persist_error_tool_name = "__monty_repl_persist_error__"
    _capture_last_expression_tool_name = "__monty_repl_capture_last_expression__"
    _last_expression_name = "__monty_repl_last_expression__"

    def __init__(
        self,
        *,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        limits: pydantic_monty.ResourceLimits | None = None,
        os_access: pydantic_monty.AbstractOS | None = None,
    ) -> None:
        """Initialize the interpreter."""
        self._tools: dict[str, Callable[..., Any]] = dict(tools) if tools else {}
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self._limits = limits
        self._os_access = os_access
        self._state: dict[str, Any] = {}
        self._name_collector = _TopLevelNameCollector()

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
        """Return a shallow copy of the persisted interpreter state."""
        return dict(self._state)

    def _build_type_check_stubs(
        self,
        tool_names: list[str],
        awaited_tool_names: set[str],
    ) -> str | None:
        """Build Monty type-check stubs for injected tools."""
        stub_lines: list[str] = []
        if tool_names:
            stub_lines.append("from typing import Any")
            for tool_name in tool_names:
                if tool_name.isidentifier() and not keyword.iskeyword(tool_name):
                    prefix = "async def" if tool_name in awaited_tool_names else "def"
                    stub_lines.append(
                        f"{prefix} {tool_name}(*args: Any, **kwargs: Any) -> Any: ..."
                    )

        if self._type_check_stubs:
            if stub_lines:
                stub_lines.append("")
            stub_lines.append(self._type_check_stubs)

        return "\n".join(stub_lines) if stub_lines else None

    def _find_awaited_tool_names(self, code: str, tool_names: set[str]) -> set[str]:
        """Return tool names used in `await tool(...)` expressions."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()

        awaited_tool_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Await):
                awaited_value = node.value
                if isinstance(awaited_value, ast.Call) and isinstance(
                    awaited_value.func, ast.Name
                ):
                    tool_name = awaited_value.func.id
                    if tool_name in tool_names:
                        awaited_tool_names.add(tool_name)
        return awaited_tool_names

    def _wrap_code(
        self,
        code: str,
        assigned_names: list[str],
        deleted_names: list[str],
        *,
        capture_last_expression: bool,
    ) -> str:
        """Append hidden persistence calls after user code."""
        wrapped_lines = [code.rstrip(), ""]
        if capture_last_expression:
            wrapped_lines.extend(
                [
                    f"{self._capture_last_expression_tool_name}({self._last_expression_name})",
                    "",
                ]
            )
        for name in assigned_names:
            wrapped_lines.extend(
                [
                    "try:",
                    f"    {self._persist_tool_name}({name!r}, {name})",
                    "except Exception as exc:",
                    "    error_message = str(exc).strip() or exc.__class__.__name__",
                    f"    {self._persist_error_tool_name}({name!r}, error_message)",
                    "",
                ]
            )

        for name in deleted_names:
            wrapped_lines.append(f"{self._delete_tool_name}({name!r})")

        return "\n".join(wrapped_lines).strip() + "\n"

    def _rewrite_final_expression(self, code: str) -> tuple[str, bool]:
        """Rewrite the final top-level expression into a hidden assignment.

        Args:
            code: User-submitted source code.

        Returns:
            Tuple of rewritten code and whether a final expression was captured.
        """
        module = ast.parse(code)
        if not module.body:
            return code, False

        last_statement = module.body[-1]
        if not isinstance(last_statement, ast.Expr):
            return code, False

        module.body[-1] = ast.Assign(
            targets=[ast.Name(id=self._last_expression_name, ctx=ast.Store())],
            value=last_statement.value,
        )
        ast.fix_missing_locations(module)
        return ast.unparse(module), True

    def _start_monty(
        self,
        monty: pydantic_monty.Monty,
        merged_vars: dict[str, Any],
        print_callback: Callable[[str, str], None],
    ) -> (
        pydantic_monty.FunctionSnapshot
        | pydantic_monty.NameLookupSnapshot
        | pydantic_monty.FutureSnapshot
        | pydantic_monty.MontyComplete
    ):
        """Start Monty execution across nearby API versions."""
        start_kwargs: dict[str, Any] = {
            "inputs": merged_vars or None,
            "limits": self._limits,
            "print_callback": print_callback,
        }
        if self._os_access is not None:
            start_kwargs["os"] = self._os_access

        try:
            return monty.start(**start_kwargs)
        except TypeError as exc:
            if self._os_access is None or "os" not in str(exc):
                raise
            start_kwargs.pop("os", None)
            try:
                return monty.start(**start_kwargs)
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

    async def execute(self, code: str) -> InterpreterRunResult:
        """Execute code and persist supported top-level variables."""
        assigned_names, deleted_names = self._name_collector.collect(
            code,
            safe_call_names=set(self._tools),
        )
        rewritten_code, capture_last_expression = self._rewrite_final_expression(code)
        wrapped_code = self._wrap_code(
            rewritten_code,
            assigned_names,
            deleted_names,
            capture_last_expression=capture_last_expression,
        )

        captured_state: dict[str, Any] = {}
        deleted_state_names: set[str] = set()
        last_expression_seen = False
        last_expression_value: Any = None
        persistence_failures: list[dict[str, str]] = []
        all_tools = dict(self._tools)

        def _persist_variable(name: str, value: Any) -> None:
            captured_state[name] = value

        def _delete_variable(name: str) -> None:
            deleted_state_names.add(name)

        def _record_persist_failure(name: str, error: str) -> None:
            persistence_failures.append({"name": name, "error": error})

        def _capture_last_expression(value: Any) -> None:
            nonlocal last_expression_seen, last_expression_value
            last_expression_seen = True
            last_expression_value = value

        all_tools[self._persist_tool_name] = _persist_variable
        all_tools[self._delete_tool_name] = _delete_variable
        all_tools[self._persist_error_tool_name] = _record_persist_failure
        all_tools[self._capture_last_expression_tool_name] = _capture_last_expression

        merged_vars = dict(self._state)
        awaited_tool_names = self._find_awaited_tool_names(wrapped_code, set(all_tools))
        type_check_stubs = self._build_type_check_stubs(
            list(all_tools),
            awaited_tool_names,
        )

        try:
            monty = pydantic_monty.Monty(
                wrapped_code,
                inputs=list(merged_vars) if merged_vars else [],
                type_check=self._type_check,
                type_check_stubs=type_check_stubs,
            )
        except Exception as exc:
            self._raise_known_monty_error(exc, exc.__traceback__)
            raise

        stdout_parts: list[str] = []

        def _capture_print(_stream: str, text: str) -> None:
            stdout_parts.append(text)

        progress = self._start_monty(monty, merged_vars, _capture_print)

        pending_tasks: dict[
            int, asyncio.Task[tuple[int, pydantic_monty.ExternalResult]]
        ] = {}

        async def _resolve_async_tool(
            call_id: int,
            result: Any,
        ) -> tuple[int, pydantic_monty.ExternalResult]:
            try:
                return call_id, {"return_value": await result}
            except Exception as exc:  # pragma: no cover
                return call_id, {"exception": exc}

        try:
            while not isinstance(progress, pydantic_monty.MontyComplete):
                if isinstance(progress, pydantic_monty.NameLookupSnapshot):
                    resolved_name = all_tools.get(progress.variable_name)
                    if resolved_name is not None:
                        progress = progress.resume(value=resolved_name)
                    else:
                        progress = progress.resume()
                    continue

                if isinstance(progress, pydantic_monty.FunctionSnapshot):
                    if progress.is_os_function:
                        if self._os_access is None:
                            raise CodeExecutionError(
                                f"OS function {progress.function_name} called but no OS access handler is configured"
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

                    func = all_tools.get(progress.function_name)
                    if func is None:
                        raise CodeExecutionError(
                            f"Unknown function: {progress.function_name}"
                        )
                    try:
                        result = func(*progress.args, **progress.kwargs)
                        if inspect.iscoroutine(result):
                            if progress.function_name in awaited_tool_names:
                                pending_tasks[progress.call_id] = asyncio.create_task(
                                    _resolve_async_tool(progress.call_id, result)
                                )
                                progress = self._resume_function_snapshot(
                                    progress,
                                    {"future": ...},
                                )
                                continue
                            result = await result
                    except Exception as exc:
                        raise CodeExecutionError(
                            f"Tool {progress.function_name} failed: {exc}"
                        ) from exc

                    progress = self._resume_function_snapshot(
                        progress,
                        {"return_value": result},
                    )
                    continue

                if isinstance(progress, pydantic_monty.FutureSnapshot):
                    current_tasks = [
                        pending_tasks[call_id]
                        for call_id in progress.pending_call_ids
                        if call_id in pending_tasks
                    ]
                    done, _ = await asyncio.wait(
                        current_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    results: dict[int, pydantic_monty.ExternalResult] = {}
                    for task in done:
                        call_id, ext_result = task.result()
                        results[call_id] = ext_result
                        pending_tasks.pop(call_id, None)
                    progress = progress.resume(results)
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

        self._state.update(captured_state)
        for deleted_name in deleted_state_names:
            self._state.pop(deleted_name, None)

        persisted_value_summaries = {
            name: safe_json_value(captured_state[name])
            for name in sorted(captured_state)
        }
        last_expression_summary = (
            safe_json_value(last_expression_value) if last_expression_seen else None
        )

        return InterpreterRunResult(
            stdout="".join(stdout_parts),
            persisted_names=sorted(captured_state),
            persisted_value_summaries=persisted_value_summaries,
            last_expression_summary=last_expression_summary,
            persistence_failures=persistence_failures,
        )


__all__ = ["InterpreterRunResult", "MontyReplInterpreter"]
