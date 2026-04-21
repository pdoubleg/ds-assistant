"""Focused tests for the minimal Monty-backed Python REPL."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_exceptions_module = importlib.import_module(
    "src.mcp.monty_python_repl_minimal.exceptions"
)
CodeExecutionError = _exceptions_module.CodeExecutionError
_interpreter_module = importlib.import_module(
    "src.mcp.monty_python_repl_minimal.interpreter"
)
MontyReplInterpreter = _interpreter_module.MontyReplInterpreter
_privacy_module = importlib.import_module("src.mcp.monty_python_repl_minimal.privacy")
MONTY_HINTS = _privacy_module.MONTY_HINTS
sanitize_exception = _privacy_module.sanitize_exception
_repl_module = importlib.import_module("src.mcp.monty_python_repl_minimal.repl")
MinimalMontyPythonREPL = _repl_module.MinimalMontyPythonREPL
_parsing_module = importlib.import_module(
    "src.mcp.monty_python_repl_minimal.core.registry.parsing"
)
parse_tool_docstring = _parsing_module.parse_tool_docstring


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


def test_interpreter_uses_monty_v0016_start_signature() -> None:
    """The interpreter should call `Monty.start()` with the current API.

    Returns:
        None
    """

    observed_kwargs: dict[str, Any] = {}
    os_access = object()
    interpreter = MontyReplInterpreter(os_access=os_access)

    class FakeMonty:
        """Capture `Monty.start()` kwargs for assertions."""

        def start(self, **kwargs: Any) -> str:
            """Record the provided start kwargs.

            Args:
                **kwargs: Start kwargs captured from the interpreter.

            Returns:
                Fixed sentinel result.
            """

            observed_kwargs.update(kwargs)
            return "started"

    result = interpreter._start_monty(  # pyright: ignore[reportPrivateUsage]
        FakeMonty(),
        {"answer": 42},
        lambda _stream, _text: None,
    )

    assert result == "started"
    assert observed_kwargs["inputs"] == {"answer": 42}
    assert observed_kwargs["limits"] is None
    assert callable(observed_kwargs["print_callback"])
    assert observed_kwargs["os"] is os_access


def test_interpreter_resume_helpers_pass_os_access_to_all_snapshots() -> None:
    """All snapshot resume helpers should thread `os_access` through resumes.

    Returns:
        None
    """

    os_access = object()
    interpreter = MontyReplInterpreter(os_access=os_access)
    call_log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class FakeFunctionSnapshot:
        """Capture function snapshot resumes."""

        def resume(self, *args: Any, **kwargs: Any) -> str:
            """Record the resume payload.

            Args:
                *args: Positional resume args.
                **kwargs: Keyword resume args.

            Returns:
                Fixed sentinel result.
            """

            call_log.append(("function", args, kwargs))
            return "function"

    class FakeNameLookupSnapshot:
        """Capture name lookup snapshot resumes."""

        def resume(self, *args: Any, **kwargs: Any) -> str:
            """Record the resume payload.

            Args:
                *args: Positional resume args.
                **kwargs: Keyword resume args.

            Returns:
                Fixed sentinel result.
            """

            call_log.append(("name_lookup", args, kwargs))
            return "name_lookup"

    class FakeFutureSnapshot:
        """Capture future snapshot resumes."""

        def resume(self, *args: Any, **kwargs: Any) -> str:
            """Record the resume payload.

            Args:
                *args: Positional resume args.
                **kwargs: Keyword resume args.

            Returns:
                Fixed sentinel result.
            """

            call_log.append(("future", args, kwargs))
            return "future"

    function_result = interpreter._resume_function_snapshot(  # pyright: ignore[reportPrivateUsage]
        FakeFunctionSnapshot(),
        {"return_value": 7},
    )
    name_lookup_value_result = interpreter._resume_name_lookup_snapshot(  # pyright: ignore[reportPrivateUsage]
        FakeNameLookupSnapshot(),
        value="resolved",
        has_value=True,
    )
    name_lookup_missing_result = interpreter._resume_name_lookup_snapshot(  # pyright: ignore[reportPrivateUsage]
        FakeNameLookupSnapshot(),
    )
    future_result = interpreter._resume_future_snapshot(  # pyright: ignore[reportPrivateUsage]
        FakeFutureSnapshot(),
        {1: {"return_value": "done"}},
    )

    assert function_result == "function"
    assert name_lookup_value_result == "name_lookup"
    assert name_lookup_missing_result == "name_lookup"
    assert future_result == "future"
    assert call_log == [
        ("function", ({"return_value": 7},), {"os": os_access}),
        ("name_lookup", (), {"value": "resolved", "os": os_access}),
        ("name_lookup", (), {"os": os_access}),
        ("future", ({1: {"return_value": "done"}},), {"os": os_access}),
    ]


def test_interpreter_translates_syntax_type_and_runtime_errors() -> None:
    """Monty syntax, typing, and runtime failures should map locally.

    Returns:
        None
    """

    syntax_interpreter = MontyReplInterpreter()
    with pytest.raises(SyntaxError):
        asyncio.run(syntax_interpreter.execute("if True print('broken')"))

    type_check_interpreter = MontyReplInterpreter(type_check=True)
    with pytest.raises(CodeExecutionError):
        asyncio.run(type_check_interpreter.execute("value: int = 'not an int'"))

    runtime_interpreter = MontyReplInterpreter()
    with pytest.raises(CodeExecutionError):
        asyncio.run(runtime_interpreter.execute("1 / 0"))


def test_sanitize_exception_uses_static_safe_hints() -> None:
    """Sanitized errors should expose deterministic exception hints.

    Returns:
        None
    """

    permission_payload = sanitize_exception(PermissionError())
    assert permission_payload["hint"] == MONTY_HINTS["PermissionError"]
    assert MONTY_HINTS["PermissionError"] in permission_payload["message"]

    wrapped_import_error = CodeExecutionError(
        "ModuleNotFoundError: No module named 'pandas'"
    )
    wrapped_import_error.__cause__ = ModuleNotFoundError("pandas")
    wrapped_import_payload = sanitize_exception(wrapped_import_error)
    assert wrapped_import_payload["hint"] == MONTY_HINTS["ImportError"]
    assert MONTY_HINTS["ImportError"] in wrapped_import_payload["message"]

    file_payload = sanitize_exception(FileNotFoundError())
    assert file_payload["hint"] == MONTY_HINTS["FileNotFoundError"]
    assert MONTY_HINTS["FileNotFoundError"] in file_payload["message"]


def test_execute_surfaces_static_safe_hints_for_wrapped_errors(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """REPL execute payloads should include static safe hints for errors.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
        monkeypatch: Pytest fixture for temporary module patching.
    """

    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    async def _raise_wrapped_import_error(_code: str) -> Any:
        """Raise a wrapped import-style failure.

        Args:
            _code: Ignored execute payload.

        Returns:
            Never returns.

        Raises:
            CodeExecutionError: Always raised for the test.
        """

        error = CodeExecutionError("Suppressed sandbox details")
        error.__cause__ = ImportError("blocked")
        raise error

    monkeypatch.setattr(repl.interpreter, "execute", _raise_wrapped_import_error)

    execution = _run_execute(repl, "import pandas")

    assert execution["status"] == "error"
    assert execution["error"]["error_type"] == "CodeExecutionError"
    assert execution["error"]["hint"] == MONTY_HINTS["ImportError"]
    assert MONTY_HINTS["ImportError"] in execution["error"]["message"]


def test_help_highlights_restricted_runtime_and_workspace_helpers(tmp_path: Path) -> None:
    """Help output should explain sandbox limits and preferred file helpers.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    overview = repl.help()
    assert "Monty Minimal Help" in overview
    assert "restricted Python runtime" in overview
    assert "write_workspace_text(...)" in overview
    assert "Path.write_text(...)" in overview
    assert "Arguments ending in `_handle` expect stored handle strings" in overview
    assert "compact status payload" in overview
    assert "Use results() as the detailed output channel" in overview

    workspace_help = repl.help("workspace")
    assert "Collection: workspace" in workspace_help
    assert "Available Tools:" in workspace_help
    assert "write_workspace_text(" in workspace_help
    assert "open(...)" in workspace_help
    assert "Path.write_text(...)" in workspace_help

    write_help = repl.help("write_workspace_text")
    assert "Tool: write_workspace_text" in write_help
    assert "Signature:" in write_help
    assert "Arguments:" in write_help
    assert "Usage example:" in write_help
    assert "scripts" in write_help
    assert "Guidance:" not in write_help

    missing_help = repl.help("not_a_real_tool")
    assert "No collection or function named 'not_a_real_tool' is registered." in missing_help
    assert "Collections:" in missing_help
    assert "Tools:" in missing_help


def test_schema_view_summaries_split_overview_and_column_details(
    tmp_path: Path,
) -> None:
    """Schema tools should keep overview payloads lean and details targeted.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    dataframe = pd.DataFrame(
        {
            "city": ["secret_city", "other_city", "hidden_city"],
            "segment": ["A", "B", "A"],
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
                "dataset = load_csv('/workspace/input.csv')",
                "df_handle = dataset['dataframe_handle']",
                "overview = summarize_dataframe(df_handle)",
                "details = summarize_dataframe_columns(df_handle, ['amount', 'target'])",
            ]
        ),
    )

    assert execution["status"] == "success"
    buffered = repl.results()
    overview = buffered["executions"][0]["persisted_value_summaries"]["overview"]
    details = buffered["executions"][0]["persisted_value_summaries"]["details"]

    assert overview["type"] == "DataFrame"
    assert overview["shape"] == [3, 4]
    assert overview["column_count"] == 4
    assert overview["column_type_counts"] == {
        "numeric": 2,
        "datetime": 0,
        "categorical": 2,
        "other": 0,
    }
    assert overview["missingness"]["total_missing_cells"] == 0
    assert "column_summaries" not in overview
    assert "summarize_dataframe_columns" in overview["usage_hint"]

    assert details["type"] == "DataFrameColumnDetails"
    assert details["requested_columns"] == ["amount", "target"]
    assert len(details["column_summaries"]) == 2
    assert details["column_summaries"][0]["column"] == "amount"
    assert "numeric_summary" in details["column_summaries"][0]
    assert details["column_summaries"][1]["column"] == "target"
    assert "numeric_summary" in details["column_summaries"][1]


def test_load_csv_returns_lightweight_dataframe_overview(tmp_path: Path) -> None:
    """Data loading should default to an overview instead of full column stats.

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

    execution = _run_execute(repl, "load_csv('/workspace/input.csv')")

    assert execution["status"] == "success"
    buffered = repl.results()
    payload = buffered["executions"][0]["last_expression_summary"]
    assert payload["dataframe_handle"] == "df_1"
    assert payload["summary"]["type"] == "DataFrame"
    assert payload["summary"]["shape"] == [3, 3]
    assert payload["summary"]["column_count"] == 3
    assert "column_summaries" not in payload["summary"]
    assert "summarize_dataframe_columns" in payload["summary"]["usage_hint"]


def test_duplicate_final_expression_summary_is_omitted_from_execute_and_results(
    tmp_path: Path,
) -> None:
    """Duplicate final summaries should not be surfaced twice.

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
                "dataset = load_csv('/workspace/input.csv')",
                "df_handle = dataset['dataframe_handle']",
                "summarize_dataframe(df_handle)",
            ]
        ),
    )

    assert execution["status"] == "success"
    assert execution["summary"] == "Execution succeeded. Call results() for buffered details."

    buffered = repl.results()
    assert (
        buffered["executions"][0]["persisted_value_summaries"]["dataset"]["summary"][
            "column_count"
        ]
        == 3
    )
    assert buffered["executions"][0]["last_expression_summary"] is None


def test_score_model_dataframe_returns_compact_scoring_summary(tmp_path: Path) -> None:
    """Scoring should return a scored handle without leaking raw row values.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    dataframe = _make_training_frame().assign(
        city=[
            "secret_city_a" if index % 2 == 0 else "secret_city_b"
            for index in range(120)
        ]
    )
    dataframe.to_csv(tmp_path / "train.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/train.csv')",
                "df_handle = dataset['dataframe_handle']",
                "baseline = train_lightgbm_baseline(",
                "    df_handle,",
                "    'target',",
                "    id_columns=['customer_id'],",
                "    num_threads=1,",
                ")",
                "scored = score_model_dataframe(",
                "    baseline['model_handle'],",
                "    df_handle,",
                ")",
            ]
        ),
    )

    assert execution["status"] == "success"
    assert execution["summary"] == "Execution succeeded. Call results() for buffered details."

    buffered = repl.results()
    scored = buffered["executions"][0]["persisted_value_summaries"]["scored"]
    assert scored["score_column"] == "pred_score"
    assert scored["row_count"] == 120
    assert scored["dataframe_handle"] == "df_2"
    assert scored["score_min"] is not None
    assert scored["score_max"] is not None

    rendered_buffered = str(buffered)
    assert "secret_city_a" not in rendered_buffered
    assert "secret_city_b" not in rendered_buffered


def test_summarize_top_p_predictions_returns_expected_metrics(tmp_path: Path) -> None:
    """Top-p summary should match deterministic scored rows.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    dataframe = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 0],
            "pred_score": [0.90, 0.80, 0.20, 0.10, 0.05],
            "city": [
                "secret_city_a",
                "secret_city_b",
                "secret_city_c",
                "secret_city_d",
                "secret_city_e",
            ],
        }
    )
    dataframe.to_csv(tmp_path / "scored.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/scored.csv')",
                "df_handle = dataset['dataframe_handle']",
                "top_summary = summarize_top_p_predictions(",
                "    df_handle,",
                "    'target',",
                "    'pred_score',",
                "    top_p=0.4,",
                ")",
            ]
        ),
    )

    assert execution["status"] == "success"
    buffered = repl.results()
    top_summary = buffered["executions"][0]["persisted_value_summaries"]["top_summary"]
    assert top_summary["top_p"] == 0.4
    assert top_summary["row_count"] == 5
    assert top_summary["top_p_row_count"] == 2
    assert top_summary["score_threshold"] == 0.8
    assert top_summary["true_positive_count"] == 1
    assert top_summary["false_positive_count"] == 1
    assert top_summary["ppv_at_p"] == 0.5
    assert top_summary["recall_at_p"] == 0.5
    assert top_summary["lift_at_p"] == 1.25
    assert top_summary["base_rate"] == 0.4


def test_analyze_top_p_false_positives_returns_privacy_safe_report(
    tmp_path: Path,
) -> None:
    """False-positive analysis should return only aggregate report details.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    dataframe = pd.DataFrame(
        {
            "customer_id": list(range(10)),
            "target": [1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "pred_score": [0.99, 0.95, 0.93, 0.91, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05],
            "risk_signal": [10.0, 12.0, 100.0, 110.0, 9.0, 95.0, 8.0, 90.0, 85.0, 7.0],
            "device_group": [
                "trusted_a",
                "trusted_b",
                "secret_fp_a",
                "secret_fp_b",
                "trusted_a",
                "secret_fp_a",
                "trusted_b",
                "secret_fp_b",
                "secret_fp_c",
                "trusted_a",
            ],
        }
    )
    dataframe.to_csv(tmp_path / "fp_scored.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/fp_scored.csv')",
                "df_handle = dataset['dataframe_handle']",
                "analysis = analyze_top_p_false_positives(",
                "    df_handle,",
                "    'target',",
                "    'pred_score',",
                "    top_p=0.4,",
                "    id_columns=['customer_id'],",
                ")",
                "report = inspect_handle(analysis['report_handle'])",
            ]
        ),
    )

    assert execution["status"] == "success"
    buffered = repl.results()
    analysis = buffered["executions"][0]["persisted_value_summaries"]["analysis"]
    report = buffered["executions"][0]["persisted_value_summaries"]["report"]

    assert analysis["top_p"] == 0.4
    assert analysis["analyzed_column_count"] == 2
    assert report["handle"] == analysis["report_handle"]
    assert report["value"]["type"] == "StoredDataframeReport"
    assert report["value"]["report_type"] == "top_p_false_positive_analysis"
    assert report["value"]["details"]["top_p_summary"]["false_positive_count"] == 2
    assert report["value"]["details"]["top_p_summary"]["true_positive_count"] == 2
    assert report["value"]["details"]["numeric_findings"][0]["column"] == "risk_signal"
    assert report["value"]["details"]["categorical_findings"][0]["column"] == "device_group"

    rendered_buffered = str(buffered)
    assert "secret_fp_a" not in rendered_buffered
    assert "secret_fp_b" not in rendered_buffered
    assert "secret_fp_c" not in rendered_buffered


def test_plot_prediction_vs_actual_slices_returns_safe_panel_metadata(
    tmp_path: Path,
) -> None:
    """Prediction-vs-actual slice plots should avoid returning raw feature values.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """
    dataframe = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 1, 0, 1, 0],
            "pred_score": [0.95, 0.85, 0.70, 0.62, 0.55, 0.40, 0.22, 0.10],
            "risk_signal": [10.0, 20.0, 35.0, 40.0, 55.0, 60.0, 75.0, 90.0],
            "segment": [
                "secret_segment_a",
                "secret_segment_b",
                "secret_segment_a",
                "secret_segment_c",
                "secret_segment_b",
                "secret_segment_d",
                "secret_segment_a",
                "secret_segment_e",
            ],
        }
    )
    dataframe.to_csv(tmp_path / "scored.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/scored.csv')",
                "df_handle = dataset['dataframe_handle']",
                "chart = plot_prediction_vs_actual_slices(",
                "    df_handle,",
                "    'target',",
                "    'pred_score',",
                "    '/workspace/output/pred_vs_actual.png',",
                "    feature_columns=['risk_signal', 'segment'],",
                "    bins=4,",
                "    top_n_categories=3,",
                ")",
            ]
        ),
    )

    assert execution["status"] == "success"
    buffered = repl.results()
    chart = buffered["executions"][0]["persisted_value_summaries"]["chart"]

    assert chart["plot_type"] == "prediction_vs_actual_slices"
    assert chart["path"] == "/workspace/output/pred_vs_actual.png"
    assert chart["panel_count"] == 3
    assert chart["feature_count"] == 2
    assert chart["panels"][0]["analysis_type"] == "global"
    assert chart["panels"][1]["feature_column"] == "risk_signal"
    assert chart["panels"][1]["analysis_type"] == "numeric"
    assert chart["panels"][2]["feature_column"] == "segment"
    assert chart["panels"][2]["analysis_type"] == "categorical"
    assert (tmp_path / "output" / "pred_vs_actual.png").exists()

    rendered_buffered = str(buffered)
    assert "secret_segment_a" not in rendered_buffered
    assert "secret_segment_b" not in rendered_buffered
    assert "secret_segment_c" not in rendered_buffered


def test_parse_tool_docstring_uses_griffe_for_multiline_google_sections() -> None:
    """Google-style parsing should preserve structured section content.

    Returns:
        None
    """

    def summarize_handle(handle: str, limit: int = 5) -> dict[str, Any]:
        """Summarize a stored handle.

        Include enough context for help payload rendering.

        Args:
            handle (str): Stored handle to inspect.
            limit (int): Maximum number of rows to include
                in the preview payload.

        Returns:
            dict[str, Any]: Privacy-safe summary for the handle.

        Examples:
            result = summarize_handle("df_1", limit=3)
            print(result["handle"])
        """
        return {"handle": handle, "limit": limit}

    parsed = parse_tool_docstring(summarize_handle)

    assert parsed.summary == "Summarize a stored handle."
    assert parsed.details == (
        "Summarize a stored handle.\n\n"
        "Include enough context for help payload rendering."
    )
    assert parsed.parameter_descriptions["handle"] == "Stored handle to inspect."
    assert (
        parsed.parameter_descriptions["limit"]
        == "Maximum number of rows to include in the preview payload."
    )
    assert parsed.returns_description == "dict[str, Any]: Privacy-safe summary for the handle."
    assert parsed.example == 'result = summarize_handle("df_1", limit=3)\nprint(result["handle"])'


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
    assert execution["summary"] == "Execution succeeded. Call results() for buffered details."
    rendered_execution = str(execution)
    assert "secret_city" not in rendered_execution
    assert "other_city" not in rendered_execution
    assert "hidden_city" not in rendered_execution

    buffered = repl.results()
    assert buffered["executions"][0]["persisted_variables"] == ["details", "payload"]
    assert buffered["executions"][0]["persisted_value_summaries"]["payload"][
        "dataframe_handle"
    ] == "df_1"
    assert buffered["executions"][0]["persisted_value_summaries"]["details"][
        "handle"
    ] == "df_1"
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

    buffered = repl.results()
    assert len(buffered["executions"]) == 2
    assert buffered["executions"][1]["persisted_variables"] == []
    assert buffered["executions"][1]["persisted_value_summaries"] == {}
    assert buffered["executions"][1]["last_expression_summary"]["report_handle"] == "fs_1"
    assert buffered["executions"][1]["last_expression_summary"]["dataframe_handle"] == "df_2"
    assert len(buffered["executions"][1]["last_expression_summary"]["selected_columns"]) == 2


def test_feature_pipeline_expression_summary_stays_privacy_safe(tmp_path: Path) -> None:
    """Feature-pipeline helpers should surface summaries without exposing raw rows.

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
                "pipeline = fit_feature_pipeline(",
                "    df_handle,",
                "    [",
                "        {",
                "            'kind': 'ratio_features',",
                "            'definitions': [",
                "                {",
                "                    'name': 'ratio',",
                "                    'numerator': 'amount',",
                "                    'denominator': 'amount',",
                "                    'fill_value': 0.0,",
                "                }",
                "            ],",
                "        }",
                "    ],",
                "    target_column='target',",
                ")",
            ]
        ),
    )
    execution = _run_execute(
        repl,
        "transform_with_feature_pipeline(df_handle, pipeline['pipeline_handle'], include_target=True)",
    )

    assert execution["status"] == "success"
    rendered_execution = str(execution)
    assert "secret_city" not in rendered_execution
    assert "other_city" not in rendered_execution
    assert "hidden_city" not in rendered_execution

    buffered = repl.results()
    assert buffered["executions"][1]["last_expression_summary"]["dataframe_handle"] == "df_2"
    assert buffered["executions"][1]["last_expression_summary"]["pipeline_handle"] == "pipeline_1"
    assert buffered["executions"][1]["last_expression_summary"]["summary"].startswith(
        "Applied feature pipeline"
    )
    rendered_buffered = str(buffered)
    assert "secret_city" not in rendered_buffered
    assert "other_city" not in rendered_buffered
    assert "hidden_city" not in rendered_buffered


def test_wide_table_eda_helpers_plan_feature_subsets(tmp_path: Path) -> None:
    """Wide-table EDA helpers should recommend deterministic feature batching.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """

    wide_frame = pd.DataFrame(
        {
            **{f"feature_{index}": [float(index), float(index + 1)] for index in range(12)},
            "target": [0, 1],
            "customer_id": [101, 102],
        }
    )
    wide_frame.to_csv(tmp_path / "wide.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/wide.csv')",
                "df_handle = dataset['dataframe_handle']",
                "triage = triage_dataframe(",
                "    df_handle,",
                "    target_column='target',",
                "    id_columns=['customer_id'],",
                "    max_columns_before_batching=5,",
                "    recommended_batch_size=4,",
                ")",
                "plan_feature_subsets(",
                "    df_handle,",
                "    target_column='target',",
                "    id_columns=['customer_id'],",
                "    batch_size=4,",
                ")",
            ]
        ),
    )

    assert execution["status"] == "success"
    triage = repl.interpreter.state["triage"]
    assert triage["wide_table"] is True
    assert triage["subset_count"] == 3
    buffered = repl.results()
    assert buffered["executions"][0]["last_expression_summary"]["report_handle"] == "report_2"
    assert buffered["executions"][0]["last_expression_summary"]["subset_count"] == 3
    assert len(buffered["executions"][0]["last_expression_summary"]["feature_subsets"]) == 3


def test_feature_selection_and_correlation_helpers_return_standardized_handles(
    tmp_path: Path,
) -> None:
    """Selection helpers should return handle-first standardized payloads.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """

    dataframe = _make_training_frame().assign(
        balance_copy=lambda frame: frame["balance"],
        score_signal_copy=lambda frame: frame["score_signal"],
    )
    dataframe.to_csv(tmp_path / "train.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/train.csv')",
                "df_handle = dataset['dataframe_handle']",
                "screen = screen_features(",
                "    df_handle,",
                "    'target',",
                "    id_columns=['customer_id'],",
                "    top_k_univariate=4,",
                ")",
                "analyze_feature_correlation(",
                "    screen['dataframe_handle'],",
                "    target_column='target',",
                "    threshold=0.99,",
                ")",
            ]
        ),
    )

    assert execution["status"] == "success"
    buffered = repl.results()
    assert buffered["executions"][0]["last_expression_summary"]["report_handle"] == "fs_2"
    assert buffered["executions"][0]["last_expression_summary"]["dataframe_handle"] == "df_3"
    assert "dropped_columns" in buffered["executions"][0]["last_expression_summary"]
    assert buffered["executions"][0]["last_expression_summary"]["summary"].startswith("Flagged")


def test_modeling_surface_exposes_tunable_params_and_hpo_handles(tmp_path: Path) -> None:
    """Modeling helpers should expose tunables and study/model handles.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
    """

    dataframe = _make_training_frame()
    dataframe.to_csv(tmp_path / "train.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/train.csv')",
                "df_handle = dataset['dataframe_handle']",
                "tunables = list_lightgbm_tunable_params()",
                "study = tune_lightgbm(",
                "    df_handle,",
                "    'target',",
                "    id_columns=['customer_id'],",
                "    n_trials=2,",
                "    num_threads=1,",
                ")",
                "fit_best_lightgbm(study['study_handle'])",
            ]
        ),
    )

    assert execution["status"] == "success"
    tunables = repl.interpreter.state["tunables"]
    assert tunables["objective_metric"] == "ppv_at_5"
    assert tunables["native_categorical_handling"] is True
    assert any(item["name"] == "learning_rate" for item in tunables["params"])
    buffered = repl.results()
    assert buffered["executions"][0]["last_expression_summary"]["model_handle"] == "model_1"
    assert buffered["executions"][0]["last_expression_summary"]["summary"].startswith("Fit")


def test_tune_lightgbm_handles_linear_tree_trials(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Tuning should support trials that toggle LightGBM linear trees.

    Args:
        tmp_path: Pytest-managed temporary workspace root.
        monkeypatch: Pytest fixture for temporary module patching.
    """

    dataframe = _make_training_frame()
    dataframe.to_csv(tmp_path / "train.csv", index=False)
    repl = MinimalMontyPythonREPL(workspace_root=tmp_path)
    modeling_module = importlib.import_module(
        "src.mcp.monty_python_repl_minimal.registry.modeling"
    )

    def _suggest_linear_tree_params(
        trial: Any,
        *,
        num_threads: int,
        seed: int,
    ) -> dict[str, Any]:
        """Return deterministic params that alternate linear-tree usage.

        Args:
            trial: Active Optuna trial.
            num_threads: Requested LightGBM worker count.
            seed: Trial seed forwarded to LightGBM.

        Returns:
            dict[str, Any]: LightGBM parameter set for the current trial.
        """

        return {
            "objective": "binary",
            "metric": "None",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": seed,
            "num_threads": num_threads,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "min_child_weight": 1e-3,
            "feature_fraction": 0.9,
            "feature_fraction_bynode": 1.0,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "min_gain_to_split": 0.0,
            "max_depth": -1,
            "extra_trees": False,
            "linear_tree": trial.number % 2 == 0,
            "force_row_wise": True,
            "cat_smooth": 10.0,
            "cat_l2": 10.0,
            "max_cat_to_onehot": 8,
            "min_data_per_group": 20,
            "max_cat_threshold": 32,
        }

    monkeypatch.setattr(
        modeling_module,
        "suggest_lgbm_params",
        _suggest_linear_tree_params,
    )

    execution = _run_execute(
        repl,
        "\n".join(
            [
                "dataset = load_csv('/workspace/train.csv')",
                "df_handle = dataset['dataframe_handle']",
                "study = tune_lightgbm(",
                "    df_handle,",
                "    'target',",
                "    id_columns=['customer_id'],",
                "    n_trials=2,",
                "    num_threads=1,",
                ")",
                "fit_best_lightgbm(study['study_handle'])",
            ]
        ),
    )

    assert execution["status"] == "success"
    buffered = repl.results()
    assert buffered["executions"][0]["last_expression_summary"]["model_handle"] == "model_1"
