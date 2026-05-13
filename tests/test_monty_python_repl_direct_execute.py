"""Tests for direct execution payloads from the Monty-backed REPL."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.mcp.monty_python_repl import MontyPythonREPL


def run_execute(repl: MontyPythonREPL, code: str) -> dict[str, object]:
    """Execute sandbox code inside a synchronous pytest test.

    Args:
        repl: Active Monty REPL service.
        code: Python source to execute.

    Returns:
        dict[str, object]: Full execution payload returned by the REPL.
    """
    return asyncio.run(repl.execute(code))


def test_execute_returns_stdout_and_artifacts_directly(tmp_path: Path) -> None:
    """The execute call should return the complete result without a results drain."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    payload = run_execute(
        repl,
        "\n".join(
            [
                "write_workspace_text('/workspace/notes.txt', 'hello')",
                "print('saved notes')",
            ]
        ),
    )

    assert payload["status"] == "success"
    assert payload["stdout"] == "saved notes\n"
    assert payload["artifacts"] == ["/workspace/notes.txt"]
    assert payload["error"] is None
    assert "pending_result_count" not in payload
    assert "session_id" not in payload


def test_execute_keeps_persistent_repl_state_without_session_ids(tmp_path: Path) -> None:
    """Each REPL instance should persist state naturally across execute calls."""
    repl = MontyPythonREPL(workspace_root=tmp_path, type_check=True)

    first = run_execute(repl, "answer = 41")
    second = run_execute(repl, "print(answer + 1)")

    assert first["status"] == "success"
    assert first["persisted_variables"] == ["answer"]
    assert second["status"] == "success"
    assert second["stdout"] == "42\n"
    assert repl.interpreter.state["answer"] == 41
