"""Focused tests for the minimal Monty-backed Python REPL."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_repl_module = importlib.import_module("src.mcp.monty_python_repl_minimal.repl")
MinimalMontyPythonREPL = _repl_module.MinimalMontyPythonREPL


def _run_execute(repl: MinimalMontyPythonREPL, code: str) -> dict[str, Any]:
    """Execute async REPL code inside a synchronous pytest test.

    Args:
        repl: Active minimal Monty REPL.
        code: Python code to execute.

    Returns:
        Execute payload returned by the REPL.
    """
    return asyncio.run(repl.execute(code))


def _make_training_frame() -> pd.DataFrame:
    """Create a compact binary-classification dataframe for screening tests.

    Returns:
        Synthetic training dataframe with numeric and categorical features.
    """
    rows: list[dict[str, Any]] = []
    for index in range(120):
        bucket = index % 6
        rows.append(
            {
                "customer_id": index,
                "segment": f"segment_{bucket % 3}",
                "balance": 50 + index * 3,
                "score_signal": (index % 10) / 10.0,
                "utilization": (index % 15) / 15.0,
                "target": 1 if bucket in (0, 1) else 0,
            }
        )
    return pd.DataFrame(rows)


def test_help_highlights_restricted_runtime_and_workspace_helpers(tmp_path: Path) -> None:
    """Help output should explain sandbox limits and preferred file helpers.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    overview = repl.help()
    combined_overview_text = " ".join(
        overview["notes"] + overview["workflow"] + overview["limitations"]
    )
    assert "restricted Python runtime" in combined_overview_text
    assert "write_workspace_text(...)" in combined_overview_text
    assert "Path.write_text(...)" in combined_overview_text

    workspace_help = repl.help("workspace")
    workspace_notes = " ".join(workspace_help["notes"])
    assert "open(...)" in workspace_notes
    assert "Path.write_text(...)" in workspace_notes

    write_help = repl.help("write_workspace_text")
    usage_guidance = " ".join(write_help["function"]["usage_guidance"])
    detailed_description = write_help["function"]["detailed_description"]
    note_text = " ".join(write_help["notes"])
    assert "restricted" in usage_guidance
    assert "open(...)" in usage_guidance
    assert "Path.write_text(...)" in usage_guidance
    assert "scripts" in detailed_description
    assert "Path.write_text(...)" in note_text


def test_execute_surfaces_persisted_helper_summaries(tmp_path: Path) -> None:
    """Assigned helper results should be visible without leaking raw values.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    dataframe = pd.DataFrame(
        {
            "city": ["secret_city", "other_city", "hidden_city"],
            "amount": [100.0, 125.0, 150.0],
            "target": [1, 0, 1],
        }
    )
    dataframe.to_csv(tmp_path / "input.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "payload = load_csv('/workspace/input.csv')",
                "details = inspect_handle(payload['dataframe_handle'])",
            ]
        ),
    )

    assert execution["status"] == "success"
    assert execution["persisted_variables"] == ["details", "payload"]
    assert execution["last_expression_summary"] is None
    assert execution["persisted_value_summaries"]["payload"]["dataframe_handle"] == "df_1"
    assert execution["persisted_value_summaries"]["details"]["handle"] == "df_1"
    rendered_execution = str(execution)
    assert "secret_city" not in rendered_execution
    assert "other_city" not in rendered_execution
    assert "hidden_city" not in rendered_execution

    buffered = repl.results()
    assert buffered["executions"][0]["persisted_value_summaries"] == execution[
        "persisted_value_summaries"
    ]
    rendered_buffered = str(buffered)
    assert "secret_city" not in rendered_buffered
    assert "other_city" not in rendered_buffered
    assert "hidden_city" not in rendered_buffered


def test_execute_surfaces_final_helper_expression(tmp_path: Path) -> None:
    """A bare final helper call should be visible in execute and results.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    training_frame = _make_training_frame()
    training_frame.to_csv(tmp_path / "training.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    seed = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/training.csv')",
                "df_handle = dataset['dataframe_handle']",
            ]
        ),
    )
    assert seed["status"] == "success"

    execution = _run_execute(
        repl,
        "screen_features(df_handle, 'target', id_columns=['customer_id'], top_k_univariate=2)",
    )

    assert execution["status"] == "success"
    assert execution["persisted_variables"] == []
    assert execution["persisted_value_summaries"] == {}
    assert execution["last_expression_summary"]["report_handle"] == "fs_1"
    assert execution["last_expression_summary"]["dataframe_handle"] == "df_2"
    assert len(execution["last_expression_summary"]["selected_columns"]) == 2

    buffered = repl.results()
    assert len(buffered["executions"]) == 2
    assert buffered["executions"][1]["last_expression_summary"] == execution[
        "last_expression_summary"
    ]


def test_freeform_expression_summary_stays_privacy_safe(tmp_path: Path) -> None:
    """Freeform helpers should surface summaries without exposing raw rows.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    dataframe = pd.DataFrame(
        {
            "city": ["secret_city", "other_city", "hidden_city"],
            "amount": [100.0, 125.0, 150.0],
            "target": [1, 0, 1],
        }
    )
    dataframe.to_csv(tmp_path / "input.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/input.csv')",
                "df_handle = dataset['dataframe_handle']",
            ]
        ),
    )
    execution = _run_execute(
        repl,
        'run_dataframe_code(df_handle, "df[\'ratio\'] = df[\'amount\'] / (df[\'amount\'] + 1)\\nprint(df[\'city\'].iloc[0])\\nresult = df")',
    )

    assert execution["status"] == "success"
    assert execution["last_expression_summary"]["dataframe_handle"] == "df_2"
    assert execution["last_expression_summary"]["columns_added"] == ["ratio"]
    assert execution["last_expression_summary"]["stdout"]["suppressed"] is True
    rendered_execution = str(execution)
    assert "secret_city" not in rendered_execution
    assert "other_city" not in rendered_execution
    assert "hidden_city" not in rendered_execution

    buffered = repl.results()
    rendered_buffered = str(buffered)
    assert "secret_city" not in rendered_buffered
    assert "other_city" not in rendered_buffered
    assert "hidden_city" not in rendered_buffered
