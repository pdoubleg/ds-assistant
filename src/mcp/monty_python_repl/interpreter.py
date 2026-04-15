"""Focused Monty interpreter for REPL-style execution."""

from __future__ import annotations

import ast
import asyncio
import inspect
import keyword
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pydantic_monty

from src.rlm.types import CodeExecutionError


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
    """Monty interpreter with automatic variable persistence."""

    _persist_tool_name = "__monty_repl_persist__"
    _delete_tool_name = "__monty_repl_delete__"
    _persist_error_tool_name = "__monty_repl_persist_error__"

    def __init__(
        self,
        *,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        limits: pydantic_monty.ResourceLimits | None = None,
        os_access: pydantic_monty.OSAccess | None = None,
    ) -> None:
        """Initialize the interpreter."""
        self._tools: dict[str, Callable[..., Any]] = dict(tools) if tools else {}
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self._limits = limits
        self._os_access = os_access
        self._state: dict[str, Any] = {}
        self._name_collector = _TopLevelNameCollector()

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
        """Return tool names used in ``await tool(...)`` expressions."""
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
    ) -> str:
        """Append hidden persistence calls after user code."""
        wrapped_lines = [code.rstrip(), ""]
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

    async def execute(self, code: str) -> InterpreterRunResult:
        """Execute code and persist supported top-level variables."""
        assigned_names, deleted_names = self._name_collector.collect(
            code,
            safe_call_names=set(self._tools),
        )
        wrapped_code = self._wrap_code(code, assigned_names, deleted_names)

        captured_state: dict[str, Any] = {}
        deleted_state_names: set[str] = set()
        persistence_failures: list[dict[str, str]] = []
        all_tools = dict(self._tools)

        def _persist_variable(name: str, value: Any) -> None:
            captured_state[name] = value

        def _delete_variable(name: str) -> None:
            deleted_state_names.add(name)

        def _record_persist_failure(name: str, error: str) -> None:
            persistence_failures.append({"name": name, "error": error})

        all_tools[self._persist_tool_name] = _persist_variable
        all_tools[self._delete_tool_name] = _delete_variable
        all_tools[self._persist_error_tool_name] = _record_persist_failure

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
        except pydantic_monty.MontySyntaxError as exc:
            raise SyntaxError(str(exc)) from exc
        except pydantic_monty.MontyTypingError as exc:
            raise CodeExecutionError(str(exc)) from exc

        stdout_parts: list[str] = []

        def _capture_print(_stream: str, text: str) -> None:
            stdout_parts.append(text)

        try:
            progress = monty.start(
                inputs=merged_vars or None,
                limits=self._limits,
                print_callback=_capture_print,
            )
        except pydantic_monty.MontyRuntimeError as exc:
            raise CodeExecutionError(str(exc)) from exc

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
                        progress = progress.resume(return_value=result)
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
                                progress = progress.resume(future=...)
                                continue
                            result = await result
                    except Exception as exc:
                        raise CodeExecutionError(
                            f"Tool {progress.function_name} failed: {exc}"
                        ) from exc

                    progress = progress.resume(return_value=result)
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
        finally:
            for task in pending_tasks.values():
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks.values(), return_exceptions=True)

        self._state.update(captured_state)
        for deleted_name in deleted_state_names:
            self._state.pop(deleted_name, None)

        return InterpreterRunResult(
            stdout="".join(stdout_parts),
            persisted_names=sorted(captured_state),
            persistence_failures=persistence_failures,
        )
