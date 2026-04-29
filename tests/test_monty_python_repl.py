"""Focused tests for the Monty-backed MCP Python REPL."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.mcp.monty_python_repl import FunctionRegistry, MontyPythonREPL
from src.mcp.monty_python_repl.core.registry.utils import safe_json_value
from src.mcp.monty_python_repl.registry import (
    StoredFeatureEngineer,
    StoredFeatureSelectionReport,
    StoredFreeformTransformer,
    StoredHpoStudy,
    StoredDataSplit,
    StoredMetricScorer,
    StoredPreprocessor,
    StoredSplitter,
    StoredTunedPipeline,
    ToolCollection,
    ToolDocstringValidationError,
    tool,
)
from src.mcp.monty_python_repl.support import metrics as metrics_support


def run_execute(repl: MontyPythonREPL, code: str) -> dict[str, object]:
    """Execute sandbox code inside a synchronous pytest test."""
    return asyncio.run(repl.execute(code))


def test_help_lists_default_collections_and_repl_notes(tmp_path: Path) -> None:
    """The default help text should advertise collections and usage notes."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    payload = repl.help()

    assert isinstance(payload, str)
    assert "Monty Sandbox Overview" in payload
    assert "Purpose:" in payload
    assert "[workspace] (5 tools)" in payload
    assert "[data_io] (4 tools)" in payload
    assert "[dataframe] (9 tools)" in payload
    assert "[feature_engineering] (8 tools)" in payload
    assert "[feature_selection] (9 tools)" in payload
    assert "[freeform] (8 tools)" in payload
    assert "[handles] (2 tools)" in payload
    assert "[hpo] (16 tools)" in payload
    assert "[metrics] (7 tools)" in payload
    assert "[preprocessing] (8 tools)" in payload
    assert "[splitting] (10 tools)" in payload
    assert "[visualizations] (9 tools)" in payload
    assert (
        "Tools: list_workspace_files, read_workspace_json, read_workspace_text"
        in payload
    )
    assert "Tools: load_csv, load_excel, save_csv, save_excel" in payload
    assert "Tools: dataframe_columns, dataframe_describe, dataframe_dtypes" in payload
    assert "Tools: create_metric_scorer, create_ppv_scorer" in payload
    assert "results() returns and clears accumulated outputs" in payload
    assert "Call help() to explore collections." in payload
    assert 'Call help("<collection>") to see available tools.' in payload
    assert 'Call help("<tool>") before writing execute(...) code.' in payload
    assert "Supported native imports:\ndataclasses, datetime, json, math, re" in payload
    assert "No class definitions inside execute(...)" in payload
    assert "Keep all files inside /workspace" in payload


def test_safe_json_value_preserves_long_code_strings_and_mapping_contents() -> None:
    """Code-bearing config strings should remain intact in JSON-safe renders."""
    long_code = "\n".join(
        [f"df['feature_{index}'] = df['base'] + {index}" for index in range(40)]
    )

    rendered = safe_json_value(
        {
            "freeform": {
                "code": long_code,
                "args": {"alpha": 0.1, "beta": 2.0},
            }
        },
        max_chars=80,
    )

    assert rendered["freeform"]["code"] == long_code
    assert rendered["freeform"]["args"] == {"alpha": 0.1, "beta": 2.0}


def test_safe_json_value_preserves_all_dataframe_columns_but_truncates_preview_values() -> (
    None
):
    """Dataframe schema should stay intact even when preview cell values are large."""
    frame = pd.DataFrame(
        [
            {
                **{f"feature_{index}": index for index in range(30)},
                "very_long_text": "x" * 200,
            }
        ]
    )

    rendered = safe_json_value(frame, max_items=5, max_chars=40)

    assert rendered["shape"] == [1, 31]
    assert len(rendered["columns"]) == 31
    assert rendered["columns"][0] == "feature_0"
    assert rendered["columns"][-1] == "very_long_text"
    assert rendered["preview"][0]["very_long_text"].endswith("... [truncated]")
    assert len(rendered["preview"][0]["very_long_text"]) < 200


def test_help_can_filter_by_collection_and_surface_arguments(tmp_path: Path) -> None:
    """Collection and tool help should expose formatted docstring-derived details."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    collection_help = repl.help("data_io")
    single_tool_help = repl.help("load_csv")

    assert "Collection: data_io" in collection_help
    assert (
        "Purpose:\nLoad and save pandas dataframes as CSV and Excel files."
        in collection_help
    )
    assert "load_csv(path: str, *, nrows: int | None = None) -> str" in collection_help
    assert (
        "load_excel(path: str, *, sheet_name: str | int = 0, nrows: int | None = None) -> str"
        in collection_help
    )
    assert (
        "save_excel(dataframes: dict[str, str], path: str, *, index: bool = False) -> str"
        in collection_help
    )
    assert "Read or write" not in collection_help

    assert "Tool: load_csv" in single_tool_help
    assert "Collection: data_io" in single_tool_help
    assert (
        "Purpose: Load a CSV file from `/workspace` and return a dataframe handle."
        in single_tool_help
    )
    assert "Arguments:" in single_tool_help
    assert (
        "- nrows (int | None, optional, default=None): Optional maximum row count to load."
        in single_tool_help
    )
    assert "Returns:\n- str: Handle for the stored dataframe." in single_tool_help
    assert 'df_handle = load_csv("/workspace/input/data.csv")' in single_tool_help
    assert "Call this helper directly inside `execute(...)` code" in single_tool_help


def test_help_can_describe_workspace_file_collection(tmp_path: Path) -> None:
    """Workspace collection help should surface formatted file helper guidance."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    payload = repl.help("workspace")
    write_json_help = repl.help("write_workspace_json")
    read_json_help = repl.help("read_workspace_json")

    assert "Collection: workspace" in payload
    assert "Read and write common text files inside /workspace" in payload
    assert "Supported file extensions:" in payload
    assert 'list_workspace_files(subdir: str = ".") -> list[str]' in payload
    assert (
        "read_workspace_text(path: str, *, max_chars: int = 200000000) -> dict[str, Any]"
        in payload
    )
    assert (
        "write_workspace_json(path: str, data: Any, *, overwrite: bool = True) -> dict[str, Any]"
        in payload
    )

    assert "Tool: write_workspace_json" in write_json_help
    assert "Collection: workspace" in write_json_help
    assert '{"mode": "demo", "retries": 2' in write_json_help
    assert '"features": {"beta": True}' in write_json_help
    assert "Tool: read_workspace_json" in read_json_help
    assert '"data": {' in read_json_help


def test_help_surfaces_structured_examples_for_dict_shaped_tools(
    tmp_path: Path,
) -> None:
    """Dict-shaped tool examples should show representative input and output structure."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    aggregate_help = repl.help("groupby_aggregate")
    shape_help = repl.help("dataframe_shape")

    assert '{"loss": ["mean", "sum"], "premium": "mean"}' in aggregate_help
    assert '{"rows": 1000, "columns": 24}' in shape_help


def test_help_exposes_hpo_inspection_output_schema_guidance(tmp_path: Path) -> None:
    """HPO inspect help should spell out the list-vs-dict output contract."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    payload = repl.help("inspect_pipeline_tunable_params")

    assert "Tool: inspect_pipeline_tunable_params" in payload
    assert "Collection: hpo" in payload
    assert "pipeline_params" in payload
    assert "list[dict[str, Any]]" in payload
    assert "pipeline_params_by_path" in payload
    assert "return_schema" in payload


def test_registry_can_be_customized_with_manual_registration(tmp_path: Path) -> None:
    """A caller should still be able to provide a custom sandbox registry."""
    registry = FunctionRegistry()

    def scale_value(value: int, factor: int = 2) -> int:
        """Scale an integer by the requested factor.

        Args:
            value (int): Base integer value.
            factor (int): Multiplier to apply.

        Returns:
            int: Scaled integer result.

        Examples:
            print(scale_value(3, factor=4))
        """
        return value * factor

    registry.register(
        scale_value,
        collection="math",
        collection_description="Custom math helpers for registry injection.",
    )

    repl = MontyPythonREPL(workspace_root=tmp_path, registry=registry)
    payload = repl.help("scale_value")

    assert "Tool: scale_value" in payload
    assert "Collection: math" in payload
    assert "scale_value(" in payload
    assert "print(scale_value(3, factor=4))" in payload
    assert "- factor (int, optional, default=2): Multiplier to apply." in payload


def test_registry_can_register_decorated_collections(tmp_path: Path) -> None:
    """Decorated collections should expose docstring-derived help metadata."""

    class MathCollection(ToolCollection):
        """Custom arithmetic helpers."""

        name = "math"
        description = "Small arithmetic helpers for custom registry composition."

        @tool
        def scale_value(self, value: int, factor: int = 2) -> int:
            """Scale an integer by the requested factor.

            Args:
                value (int): Base integer value.
                factor (int): Multiplier to apply.

            Returns:
                int: Scaled integer result.

            Examples:
                print(scale_value(3, factor=4))
            """
            return value * factor

    registry = FunctionRegistry()
    registry.register_collection(MathCollection())

    repl = MontyPythonREPL(workspace_root=tmp_path, registry=registry)
    payload = repl.help("math")
    execution = run_execute(repl, "print(scale_value(3, factor=4))")
    buffered = repl.results()

    assert "Collection: math" in payload
    assert "scale_value(value: int, factor: int = 2) -> int" in payload
    assert "Scale an integer by the requested factor." in payload
    assert payload.index("scale_value(") < payload.index("Next Steps:")
    assert execution["status"] == "success"
    assert "12" in buffered["combined_output"]


def test_register_collection_rejects_duplicate_tool_names() -> None:
    """Multiple collections should not be allowed to silently override tools."""

    class FirstCollection(ToolCollection):
        """First collection used to seed the registry."""

        name = "first"

        @tool
        def shared_tool(self) -> str:
            """Return the first value.

            Returns:
                str: First string value.

            Examples:
                print(shared_tool())
            """
            return "first"

    class SecondCollection(ToolCollection):
        """Second collection that conflicts with the first."""

        name = "second"

        @tool
        def shared_tool(self) -> str:
            """Return the second value.

            Returns:
                str: Second string value.

            Examples:
                print(shared_tool())
            """
            return "second"

    registry = FunctionRegistry()
    registry.register_collection(FirstCollection())

    try:
        registry.register_collection(SecondCollection())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected duplicate tool registration to fail.")


def test_register_rejects_invalid_tool_docstrings() -> None:
    """Registered tools should fail fast when required docstring sections are missing."""
    registry = FunctionRegistry()

    def missing_args(value: int) -> int:
        """Increment a value.

        Returns:
            int: Incremented value.

        Examples:
            print(missing_args(1))
        """
        return value + 1

    def missing_returns(value: int) -> int:
        """Increment a value.

        Args:
            value (int): Base integer value.

        Examples:
            print(missing_returns(1))
        """
        return value + 1

    def missing_examples(value: int) -> int:
        """Increment a value.

        Args:
            value (int): Base integer value.

        Returns:
            int: Incremented value.
        """
        return value + 1

    invalid_tools = [
        (missing_args, "Args"),
        (missing_returns, "Returns"),
        (missing_examples, "Examples"),
    ]

    for invalid_tool, expected_message in invalid_tools:
        with pytest.raises(ToolDocstringValidationError, match=expected_message):
            registry.register(invalid_tool, collection="broken")


def test_help_preserves_alphabetical_function_order_within_collection(
    tmp_path: Path,
) -> None:
    """Collection help should keep function ordering stable and alphabetical."""
    registry = FunctionRegistry()

    def beta_tool() -> str:
        """Return the beta marker.

        Returns:
            str: Beta marker.

        Examples:
            print(beta_tool())
        """
        return "beta"

    def alpha_tool() -> str:
        """Return the alpha marker.

        Returns:
            str: Alpha marker.

        Examples:
            print(alpha_tool())
        """
        return "alpha"

    registry.register(
        beta_tool,
        collection="letters",
        collection_description="Simple letter helpers.",
    )
    registry.register(alpha_tool, collection="letters")

    repl = MontyPythonREPL(workspace_root=tmp_path, registry=registry)
    payload = repl.help("letters")

    assert payload.index("alpha_tool() -> str") < payload.index("beta_tool() -> str")


def test_help_returns_valid_names_for_unknown_lookup(tmp_path: Path) -> None:
    """Unknown help lookups should return valid collection and tool names."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    payload = repl.help("not_a_real_name")

    assert "No collection or function named 'not_a_real_name' is registered." in payload
    assert "Available collections:\n" in payload
    assert "data_io" in payload
    assert "load_csv" in payload


def test_execute_persists_assigned_state_and_results_are_drained(
    tmp_path: Path,
) -> None:
    """Top-level assignments should persist and results should be drainable."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    first = run_execute(repl, "answer = 41\nprint('saved')")
    second = run_execute(repl, "print(answer + 1)")
    buffered = repl.results()
    drained = repl.results()

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert "answer" in first["persisted_variables"]
    assert buffered["status"] == "ok"
    assert len(buffered["executions"]) == 2
    assert "saved" in buffered["combined_output"]
    assert "42" in buffered["combined_output"]
    assert drained["status"] == "empty"


def test_execute_reports_persistence_failures_without_hiding_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Unsupported values should warn when they cannot be persisted."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    def force_persistence_warning(
        _code: str,
        _assigned_names: list[str],
        _deleted_names: list[str],
    ) -> str:
        return "\n".join(
            [
                "bad_value = 123",
                "print('custom object assigned')",
                "__monty_repl_persist_error__('bad_value', 'forced failure')",
            ]
        )

    monkeypatch.setattr(repl.interpreter, "_wrap_code", force_persistence_warning)

    execution = run_execute(
        repl,
        "bad_value = 123",
    )
    buffered = repl.results()

    assert execution["status"] == "success"
    assert execution["persisted_variables"] == []
    assert execution["persistence_failures"]
    assert execution["persistence_failures"][0]["name"] == "bad_value"
    assert "custom object assigned" in buffered["combined_output"]
    assert "Persistence warnings:" in buffered["combined_output"]


def test_execute_enforces_workspace_paths_and_tracks_artifacts(tmp_path: Path) -> None:
    """Relative and /workspace paths should work while external paths fail."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    success = run_execute(
        repl,
        "\n".join(
            [
                "from pathlib import Path",
                "Path('notes.txt').write_text('hello from monty')",
                "print(Path('/workspace/notes.txt').read_text())",
            ]
        ),
    )
    failure = run_execute(
        repl,
        "\n".join(
            [
                "from pathlib import Path",
                "Path('/tmp/escape.txt').write_text('should fail')",
            ]
        ),
    )
    buffered = repl.results()

    assert success["status"] == "success"
    assert "/workspace/notes.txt" in success["artifacts"]
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello from monty"

    assert failure["status"] == "error"
    assert "outside the /workspace sandbox" in str(failure["error"])
    assert "hello from monty" in buffered["combined_output"]
    assert "outside the /workspace sandbox" in buffered["combined_output"]


def test_workspace_file_helpers_can_read_and_write_text_and_json(
    tmp_path: Path,
) -> None:
    """Workspace helpers should support safe text and JSON authoring flows."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    result = run_execute(
        repl,
        "\n".join(
            [
                "notes_result = write_workspace_text('/workspace/docs/notes.md', '# Notes\\nReady to go')",
                "notes_payload = read_workspace_text('/workspace/docs/notes.md')",
                "json_result = write_workspace_json('/workspace/config/settings.json', {'mode': 'demo', 'retries': 2})",
                "json_payload = read_workspace_json('/workspace/config/settings.json')",
                "workspace_files = list_workspace_files('.')",
                "print(notes_payload['content'])",
                "print(json_payload['data']['mode'])",
                "print(workspace_files)",
            ]
        ),
    )
    buffered = repl.results()

    assert result["status"] == "success"
    assert "/workspace/docs/notes.md" in result["artifacts"]
    assert "/workspace/config/settings.json" in result["artifacts"]
    assert (tmp_path / "docs" / "notes.md").read_text(encoding="utf-8") == (
        "# Notes\nReady to go"
    )
    assert (tmp_path / "config" / "settings.json").is_file()
    assert "Ready to go" in buffered["combined_output"]
    assert "demo" in buffered["combined_output"]
    assert "/workspace/docs/notes.md" in buffered["combined_output"]
    assert "/workspace/config/settings.json" in buffered["combined_output"]


def test_default_eda_helpers_can_generate_artifacts(tmp_path: Path) -> None:
    """The seeded registry should support EDA, Plotly HTML, and Excel export."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame(
        {
            "segment": ["a", "a", "b"],
            "premium": [10, 15, 30],
            "loss": [1, 3, 4],
        }
    )
    input_frame.to_csv(tmp_path / "input.csv", index=False)

    first = run_execute(
        repl,
        "\n".join(
            [
                "df_handle = load_csv('/workspace/input.csv')",
                "summary_handle = groupby_aggregate(df_handle, ['segment'], {'premium': 'sum', 'loss': 'mean'})",
                "fig_handle = create_bar_chart(summary_handle, 'segment', 'premium')",
                "saved_plot_paths = save_plotly_figure(fig_handle, '/workspace/output/chart.html')",
                "saved_excel_path = save_excel({'raw': df_handle, 'summary': summary_handle}, '/workspace/output/report.xlsx')",
                "print(dataframe_shape(df_handle))",
            ]
        ),
    )
    second = run_execute(
        repl,
        "\n".join(
            [
                "print(dataframe_head(summary_handle))",
                "print(saved_plot_paths)",
                "print(saved_excel_path)",
            ]
        ),
    )
    buffered = repl.results()

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert {
        "df_handle",
        "summary_handle",
        "fig_handle",
        "saved_plot_paths",
        "saved_excel_path",
    } <= set(first["persisted_variables"])
    assert (tmp_path / "output" / "chart.html").is_file()
    assert (tmp_path / "output" / "report.xlsx").is_file()
    assert len(buffered["executions"]) == 2
    assert "/workspace/output/chart.html" in buffered["combined_output"]
    assert "/workspace/output/report.xlsx" in buffered["combined_output"]
    assert "segment" in buffered["combined_output"]


def test_data_io_helpers_can_load_excel_workbooks(tmp_path: Path) -> None:
    """The dataframe IO collection should load Excel worksheets into dataframe handles."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame(
        {
            "segment": ["a", "b"],
            "premium": [10, 25],
        }
    )
    input_frame.to_excel(tmp_path / "input.xlsx", index=False)

    result = run_execute(
        repl,
        "\n".join(
            [
                "df_handle = load_excel('/workspace/input.xlsx')",
                "print(dataframe_shape(df_handle))",
                "print(dataframe_columns(df_handle))",
            ]
        ),
    )
    buffered = repl.results()

    assert result["status"] == "success"
    assert "df_handle" in result["persisted_variables"]
    assert "rows" in buffered["combined_output"]
    assert "premium" in buffered["combined_output"]


def test_help_can_describe_the_freeform_dataframe_tool(tmp_path: Path) -> None:
    """Detailed help should surface the freeform dataframe tool contract."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    payload = repl.help("run_dataframe_code")

    assert "Tool: run_dataframe_code" in payload
    assert "Collection: freeform" in payload
    assert "stored dataframe" in payload
    assert "Optuna" in payload
    assert "workspace path" in payload
    assert 'freeform_code = """' in payload
    assert "result = run_dataframe_code(df_handle, freeform_code)" in payload
    assert "- code (str, required): Python source that reads or mutates" in payload
    assert "final dataframe assigned back to ``df``" in payload
    assert "convert the virtual path first" in payload
    assert "prefer storing the freeform source in a named multiline" in payload


def test_freeform_dataframe_tool_can_create_a_new_dataframe_handle(
    tmp_path: Path,
) -> None:
    """Freeform dataframe execution should preserve the source handle and persist a new one."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame(
        {
            "premium": [10.0, 20.0],
            "loss": [1.0, 5.0],
        }
    )
    input_frame.to_csv(tmp_path / "claims.csv", index=False)
    freeform_code = "df['margin'] = df['premium'] - df['loss']"

    execution = run_execute(
        repl,
        "\n".join(
            [
                "source_handle = load_csv('/workspace/claims.csv')",
                f"result = run_dataframe_code(source_handle, {freeform_code!r})",
                "print(result)",
            ]
        ),
    )
    buffered = repl.results()

    source_handle = str(repl.interpreter.state["source_handle"])
    result_payload = repl.interpreter.state["result"]
    assert isinstance(result_payload, dict)

    source_dataframe = repl.object_store.get(source_handle, expected_type=pd.DataFrame)
    transformed_dataframe = repl.object_store.get(
        result_payload["dataframe_handle"],
        expected_type=pd.DataFrame,
    )

    assert execution["status"] == "success"
    assert result_payload["dataframe_handle"] != source_handle
    assert result_payload["rows"] == 2
    assert result_payload["column_count"] == 3
    assert result_payload["columns_added"] == ["margin"]
    assert "margin" not in source_dataframe.columns
    assert list(transformed_dataframe.columns) == ["premium", "loss", "margin"]
    assert transformed_dataframe["margin"].tolist() == [9.0, 15.0]
    assert "dataframe_handle" in buffered["combined_output"]


def test_freeform_dataframe_tool_supports_rebinding_df(tmp_path: Path) -> None:
    """The freeform tool should persist a rebound dataframe assigned back to ``df``."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame(
        {
            "premium": [10.0, 20.0],
            "loss": [1.0, 5.0],
        }
    )
    input_frame.to_csv(tmp_path / "claims.csv", index=False)
    freeform_code = (
        "df = df[['premium']].rename(columns={'premium': 'amount'})\n"
        "df['amount'] = df['amount'] * 2"
    )

    execution = run_execute(
        repl,
        "\n".join(
            [
                "source_handle = load_csv('/workspace/claims.csv')",
                f"result = run_dataframe_code(source_handle, {freeform_code!r})",
            ]
        ),
    )

    result_payload = repl.interpreter.state["result"]
    assert isinstance(result_payload, dict)
    rebound_dataframe = repl.object_store.get(
        result_payload["dataframe_handle"],
        expected_type=pd.DataFrame,
    )

    assert execution["status"] == "success"
    assert result_payload["columns"] == ["amount"]
    assert set(result_payload["columns_removed"]) == {"loss", "premium"}
    assert rebound_dataframe["amount"].tolist() == [20.0, 40.0]


def test_freeform_dataframe_tool_returns_captured_stdout(tmp_path: Path) -> None:
    """Freeform execution should return every line printed inside the exec call."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame(
        {
            "premium": [10.0, 20.0],
            "loss": [1.0, 5.0],
        }
    )
    input_frame.to_csv(tmp_path / "claims.csv", index=False)
    freeform_code = "\n".join(
        [
            "print('starting freeform')",
            'print(f"rows={len(df)}")',
            "df['margin'] = df['premium'] - df['loss']",
            "print('finished freeform')",
        ]
    )

    execution = run_execute(
        repl,
        "\n".join(
            [
                "source_handle = load_csv('/workspace/claims.csv')",
                f"result = run_dataframe_code(source_handle, {freeform_code!r})",
            ]
        ),
    )

    result_payload = repl.interpreter.state["result"]
    assert isinstance(result_payload, dict)

    assert execution["status"] == "success"
    assert result_payload["stdout"] == (
        "starting freeform\nrows=2\nfinished freeform\n"
    )
    assert result_payload["columns_added"] == ["margin"]


def test_freeform_dataframe_tool_supports_workspace_path_helper(
    tmp_path: Path,
) -> None:
    """Freeform execution should resolve workspace files through the ergonomic helper."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame(
        {
            "premium": [10.0, 20.0],
            "loss": [1.0, 5.0],
        }
    )
    input_frame.to_csv(tmp_path / "claims.csv", index=False)

    freeform_code = "\n".join(
        [
            "reference = pd.read_csv(workspace_path('/workspace/reference.csv'))",
            'print(f"reference_rows={len(reference)}")',
            "df['reference_value'] = reference.loc[0, 'value']",
        ]
    )

    execution = run_execute(
        repl,
        "\n".join(
            [
                "write_workspace_text('/workspace/reference.csv', 'value\\n7\\n')",
                "source_handle = load_csv('/workspace/claims.csv')",
                f"result = run_dataframe_code(source_handle, {freeform_code!r})",
            ]
        ),
    )

    result_payload = repl.interpreter.state["result"]
    assert isinstance(result_payload, dict)
    transformed_dataframe = repl.object_store.get(
        result_payload["dataframe_handle"],
        expected_type=pd.DataFrame,
    )

    assert execution["status"] == "success"
    assert result_payload["stdout"] == "reference_rows=1\n"
    assert result_payload["columns_added"] == ["reference_value"]
    assert transformed_dataframe["reference_value"].tolist() == [7, 7]


def test_freeform_dataframe_tool_supports_common_safe_builtins(
    tmp_path: Path,
) -> None:
    """Freeform execution should expose common safe DS builtins."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame(
        {
            "premium": [10.0, 20.0],
            "loss": [1.0, 5.0],
        }
    )
    input_frame.to_csv(tmp_path / "claims.csv", index=False)
    freeform_code = "\n".join(
        [
            "df['has_shape'] = hasattr(df, 'shape')",
            "df['premium_is_series'] = hasattr(df['premium'], 'dtype')",
            "df['shape_type'] = repr(type(getattr(df, 'shape')))",
            "df['premium_callable'] = callable(getattr(df['premium'], 'mean'))",
            "df['first_column'] = next(iter(df.columns))",
            "df['slice_stop'] = slice(0, 2).stop",
            "df['pow_value'] = pow(2, 3)",
        ]
    )

    execution = run_execute(
        repl,
        "\n".join(
            [
                "source_handle = load_csv('/workspace/claims.csv')",
                f"result = run_dataframe_code(source_handle, {freeform_code!r})",
            ]
        ),
    )

    result_payload = repl.interpreter.state["result"]
    assert isinstance(result_payload, dict)
    transformed_dataframe = repl.object_store.get(
        result_payload["dataframe_handle"],
        expected_type=pd.DataFrame,
    )

    assert execution["status"] == "success"
    assert result_payload["columns_added"] == [
        "has_shape",
        "premium_is_series",
        "shape_type",
        "premium_callable",
        "first_column",
        "slice_stop",
        "pow_value",
    ]
    assert transformed_dataframe["has_shape"].tolist() == [True, True]
    assert transformed_dataframe["premium_is_series"].tolist() == [True, True]
    assert transformed_dataframe["shape_type"].tolist() == [
        "<class 'tuple'>",
        "<class 'tuple'>",
    ]
    assert transformed_dataframe["premium_callable"].tolist() == [True, True]
    assert transformed_dataframe["first_column"].tolist() == ["premium", "premium"]
    assert transformed_dataframe["slice_stop"].tolist() == [2, 2]
    assert transformed_dataframe["pow_value"].tolist() == [8, 8]


def test_freeform_dataframe_tool_blocks_dunder_lookup_via_builtin_helpers(
    tmp_path: Path,
) -> None:
    """Builtin helper wrappers should reject blocked dunder introspection."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame({"premium": [10.0], "loss": [2.0]})
    input_frame.to_csv(tmp_path / "claims.csv", index=False)

    seed = run_execute(repl, "df_handle = load_csv('/workspace/claims.csv')")
    failure = run_execute(
        repl,
        "\n".join(
            [
                "bad_code = \"df['kind'] = repr(getattr(df, '__class__'))\"",
                "run_dataframe_code(df_handle, bad_code)",
            ]
        ),
    )
    buffered = repl.results()

    assert seed["status"] == "success"
    assert failure["status"] == "error"
    assert "runtime_error" in str(failure["error"])
    assert "Disallowed attribute access `__class__`" in str(failure["error"])
    assert "runtime_error" in buffered["combined_output"]


def test_freeform_dataframe_tool_supports_optuna_pipeline_libraries(
    tmp_path: Path,
) -> None:
    """The freeform tool should expose the broader Optuna pipeline runtime."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame(
        {
            "premium": [10.0, 20.0, 15.0, 30.0, 18.0, 22.0],
            "loss": [1.0, 5.0, 3.0, 7.0, 2.0, 4.0],
            "segment": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    input_frame.to_csv(tmp_path / "claims.csv", index=False)
    freeform_code = "\n".join(
        [
            "import json",
            "import joblib",
            "import optuna",
            "import pandas as pd",
            "from lightgbm import LGBMClassifier",
            "from optuna.samplers import TPESampler",
            "from sklearn.base import clone",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.feature_selection import SelectFromModel",
            "from sklearn.impute import SimpleImputer",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder",
            "",
            "loaded = pd.read_csv(resolve_workspace_path('/workspace/claims.csv'))",
            "X = loaded[['premium', 'loss', 'segment']]",
            "y = loaded['target']",
            "X_train, _, y_train, _ = train_test_split(",
            "    X,",
            "    y,",
            "    test_size=0.33,",
            "    random_state=0,",
            "    stratify=y,",
            ")",
            "preprocessor = ColumnTransformer(",
            "    transformers=[",
            "        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]), ['premium', 'loss']),",
            "        ('cat', Pipeline(steps=[",
            "            ('imputer', SimpleImputer(strategy='most_frequent')),",
            "            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),",
            "        ]), ['segment']),",
            "    ],",
            "    remainder='drop',",
            "    sparse_threshold=0.0,",
            ")",
            "selector = SelectFromModel(",
            "    estimator=LGBMClassifier(n_estimators=5, min_child_samples=1, random_state=0, n_jobs=1, verbosity=-1),",
            "    threshold=-np.inf,",
            "    max_features=2,",
            ")",
            "workflow = Pipeline(",
            "    steps=[",
            "        ('feature_passthrough', FunctionTransformer(lambda frame: frame, validate=False)),",
            "        ('preprocessor', preprocessor),",
            "        ('selector', selector),",
            "        ('model', LGBMClassifier(n_estimators=5, min_child_samples=1, random_state=0, n_jobs=1, verbosity=-1)),",
            "    ]",
            ")",
            "cloned_workflow = clone(workflow)",
            "transformed = cloned_workflow.named_steps['preprocessor'].fit_transform(X_train, y_train)",
            "study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=0))",
            "def objective(trial):",
            "    return float(transformed.shape[1]) + trial.suggest_float('score_proxy', 0.1, 0.9)",
            "study.optimize(objective, n_trials=1)",
            "artifact_path = resolve_workspace_path('/workspace/freeform_optuna_artifact.joblib')",
            "joblib.dump({'best_value': study.best_value}, artifact_path)",
            "artifact = joblib.load(artifact_path)",
            "payload_json = json.dumps(artifact, sort_keys=True)",
            "df = loaded.copy()",
            "df['study_best_value'] = artifact['best_value']",
            "df['artifact_payload_length'] = len(payload_json)",
        ]
    )

    execution = run_execute(
        repl,
        "\n".join(
            [
                "source_handle = load_csv('/workspace/claims.csv')",
                f"result = run_dataframe_code(source_handle, {freeform_code!r})",
            ]
        ),
    )

    result_payload = repl.interpreter.state["result"]
    transformed_dataframe = repl.object_store.get(
        result_payload["dataframe_handle"],
        expected_type=pd.DataFrame,
    )

    assert execution["status"] == "success"
    assert "study_best_value" in result_payload["columns"]
    assert "artifact_payload_length" in result_payload["columns"]
    assert transformed_dataframe["artifact_payload_length"].nunique() == 1
    assert transformed_dataframe["study_best_value"].notna().all()


def test_freeform_dataframe_tool_surfaces_validation_errors(tmp_path: Path) -> None:
    """Disallowed imports should be returned as validation failures."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame({"premium": [10.0], "loss": [2.0]})
    input_frame.to_csv(tmp_path / "claims.csv", index=False)

    seed = run_execute(repl, "df_handle = load_csv('/workspace/claims.csv')")
    failure = run_execute(
        repl,
        "\n".join(
            [
                "bad_code = \"import os\\ndf['x'] = 1\"",
                "run_dataframe_code(df_handle, bad_code)",
            ]
        ),
    )
    buffered = repl.results()

    assert seed["status"] == "success"
    assert failure["status"] == "error"
    assert "validation_error" in str(failure["error"])
    assert "Disallowed package import `os`" in str(failure["error"])
    assert "validation_error" in buffered["combined_output"]


def test_freeform_dataframe_tool_surfaces_runtime_errors(tmp_path: Path) -> None:
    """Broken dataframe logic should be returned as runtime failures."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    input_frame = pd.DataFrame({"premium": [10.0], "loss": [2.0]})
    input_frame.to_csv(tmp_path / "claims.csv", index=False)

    seed = run_execute(repl, "df_handle = load_csv('/workspace/claims.csv')")
    failure = run_execute(
        repl,
        "\n".join(
            [
                "bad_code = \"df['ratio'] = df['loss'] / df['missing_column']\"",
                "run_dataframe_code(df_handle, bad_code)",
            ]
        ),
    )
    buffered = repl.results()

    assert seed["status"] == "success"
    assert failure["status"] == "error"
    assert "runtime_error" in str(failure["error"])
    assert "missing_column" in str(failure["error"])
    assert "runtime_error" in buffered["combined_output"]


def test_freeform_transformer_helpers_can_fit_transform_and_persist_artifacts(
    tmp_path: Path,
) -> None:
    """Reusable freeform transformers should behave like stable pipeline stages."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    train_frame = pd.DataFrame(
        {
            "premium": [10.0, 20.0, 30.0],
            "loss": [1.0, 4.0, 9.0],
            "target": [0, 1, 0],
        }
    )
    score_frame = pd.DataFrame(
        {
            "premium": [12.0, 24.0],
            "loss": [2.0, 6.0],
            "target": [1, 0],
        }
    )
    train_frame.to_csv(tmp_path / "freeform_train.csv", index=False)
    score_frame.to_csv(tmp_path / "freeform_score.csv", index=False)

    first = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/freeform_train.csv')",
                "score_handle = load_csv('/workspace/freeform_score.csv')",
                "freeform_handle = fit_freeform_transformer(",
                "    train_handle,",
                "    \"df['loss_ratio'] = np.where(df['premium'] > params['ratio_floor'], df['loss'] / df['premium'], np.nan)\\n\"",
                "    \"df['loss_gap'] = (df['premium'] - df['loss']) * params['gap_scale']\",",
                "    target_column='target',",
                "    args={'ratio_floor': 0.0, 'gap_scale': 1.0},",
                ")",
                "engineered_train = transform_with_freeform_transformer(train_handle, freeform_handle, include_target=True)",
                "engineered_score = transform_with_freeform_transformer(score_handle, freeform_handle)",
                "print(inspect_freeform_transformer(freeform_handle))",
            ]
        ),
    )
    second = run_execute(
        repl,
        "\n".join(
            [
                "saved_freeform_path = save_freeform_transformer(freeform_handle, '/workspace/output/freeform_transformer.joblib')",
                "reloaded_freeform = load_freeform_transformer('/workspace/output/freeform_transformer.joblib')",
                "reloaded_score = transform_with_freeform_transformer(score_handle, reloaded_freeform)",
                "print(list_freeform_transformer_features(reloaded_freeform))",
            ]
        ),
    )

    state = repl.interpreter.state
    freeform_handle = state["freeform_handle"]
    engineered_train_handle = state["engineered_train"]
    engineered_score_handle = state["engineered_score"]
    reloaded_freeform_handle = state["reloaded_freeform"]
    reloaded_score_handle = state["reloaded_score"]

    artifact = repl.object_store.get(
        freeform_handle,
        expected_type=StoredFreeformTransformer,
    )
    reloaded_artifact = repl.object_store.get(
        reloaded_freeform_handle,
        expected_type=StoredFreeformTransformer,
    )
    engineered_train = repl.object_store.get(
        engineered_train_handle,
        expected_type=pd.DataFrame,
    )
    engineered_score = repl.object_store.get(
        engineered_score_handle,
        expected_type=pd.DataFrame,
    )
    reloaded_score = repl.object_store.get(
        reloaded_score_handle,
        expected_type=pd.DataFrame,
    )

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert artifact.is_fitted() is True
    assert reloaded_artifact.is_fitted() is True
    assert artifact.columns_added == ["loss_ratio", "loss_gap"]
    assert artifact.args == {"ratio_floor": 0.0, "gap_scale": 1.0}
    assert artifact.target_column == "target"
    assert "target" in engineered_train.columns
    assert "target" not in engineered_score.columns
    assert np.isclose(engineered_score.loc[0, "loss_ratio"], 2.0 / 12.0)
    assert engineered_score.loc[1, "loss_gap"] == 18.0
    assert engineered_score.columns.tolist() == reloaded_score.columns.tolist()
    assert (tmp_path / "output" / "freeform_transformer.joblib").is_file()


def test_combined_freeform_transformer_helper_returns_stable_alias_keys(
    tmp_path: Path,
) -> None:
    """The combined freeform helper should return both specific and generic handles."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    frame = pd.DataFrame(
        {
            "premium": [10.0, 20.0],
            "loss": [1.0, 5.0],
            "target": [0, 1],
        }
    )
    frame.to_csv(tmp_path / "freeform_combined.csv", index=False)

    execution = run_execute(
        repl,
        "\n".join(
            [
                "df_handle = load_csv('/workspace/freeform_combined.csv')",
                "result = fit_transform_with_freeform_transformer(",
                "    df_handle,",
                "    \"df['loss_ratio'] = np.where(df['premium'] > params['ratio_floor'], df['loss'] / df['premium'], np.nan)\",",
                "    target_column='target',",
                "    args={'ratio_floor': 0.0},",
                "    include_target=True,",
                ")",
            ]
        ),
    )

    result = repl.interpreter.state["result"]
    transformed = repl.object_store.get(
        result["dataframe_handle"],
        expected_type=pd.DataFrame,
    )

    assert execution["status"] == "success"
    assert result["freeform_transformer_handle"] == result["transformer_handle"]
    assert result["transformer_type"] == "freeform"
    assert "target" in transformed.columns
    assert "loss_ratio" in transformed.columns


def test_preprocessing_helpers_can_fit_transform_and_persist_artifacts(
    tmp_path: Path,
) -> None:
    """The preprocessing collection should fit, transform, inspect, and reload sklearn artifacts."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    train_frame = pd.DataFrame(
        {
            "age": [21.0, np.nan, 39.0, 44.0],
            "income": [50_000.0, 61_000.0, np.nan, 80_000.0],
            "city": ["a", "b", "a", None],
            "segment": ["retail", "retail", "enterprise", "enterprise"],
            "target": [0, 1, 0, 1],
        }
    )
    score_frame = pd.DataFrame(
        {
            "age": [35.0, np.nan],
            "income": [72_000.0, 58_000.0],
            "city": ["c", "a"],
            "segment": ["retail", None],
            "target": [1, 0],
        }
    )
    train_frame.to_csv(tmp_path / "train.csv", index=False)
    score_frame.to_csv(tmp_path / "score.csv", index=False)

    first = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/train.csv')",
                "score_handle = load_csv('/workspace/score.csv')",
                "onehot_spec = build_preprocessing_spec(",
                "    numeric_steps=[{'kind': 'simple_imputer', 'strategy': 'median'}],",
                "    categorical_steps=[",
                "        {'kind': 'simple_imputer', 'strategy': 'most_frequent'},",
                "        {'kind': 'one_hot_encoder', 'handle_unknown': 'ignore', 'sparse_output': False},",
                "    ],",
                ")",
                "onehot_prep = fit_preprocessor(train_handle, onehot_spec, target_column='target')",
                "onehot_train = transform_dataframe(train_handle, onehot_prep, include_target=True)",
                "onehot_score = transform_dataframe(score_handle, onehot_prep)",
                "ordinal_spec = build_preprocessing_spec(",
                "    numeric_steps=[{'kind': 'simple_imputer', 'strategy': 'median'}],",
                "    categorical_steps=[",
                "        {'kind': 'simple_imputer', 'strategy': 'most_frequent'},",
                "        {'kind': 'ordinal_encoder', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1},",
                "    ],",
                ")",
                "ordinal_result = fit_transform_dataframe(train_handle, ordinal_spec, target_column='target', include_target=True)",
                "ordinal_score = transform_dataframe(score_handle, ordinal_result['preprocessor_handle'])",
                "print(inspect_preprocessor(onehot_prep))",
            ]
        ),
    )
    second = run_execute(
        repl,
        "\n".join(
            [
                "saved_prep_path = save_preprocessor(onehot_prep, '/workspace/output/preprocessor.joblib')",
                "reloaded_prep = load_preprocessor('/workspace/output/preprocessor.joblib')",
                "reloaded_score = transform_dataframe(score_handle, reloaded_prep)",
                "print(inspect_handle(onehot_prep))",
                "print(dataframe_columns(reloaded_score))",
            ]
        ),
    )
    buffered = repl.results()

    state = repl.interpreter.state
    onehot_prep_handle = state["onehot_prep"]
    onehot_train_handle = state["onehot_train"]
    onehot_score_handle = state["onehot_score"]
    ordinal_result = state["ordinal_result"]
    ordinal_score_handle = state["ordinal_score"]
    reloaded_prep_handle = state["reloaded_prep"]
    reloaded_score_handle = state["reloaded_score"]

    onehot_artifact = repl.object_store.get(
        onehot_prep_handle,
        expected_type=StoredPreprocessor,
    )
    reloaded_artifact = repl.object_store.get(
        reloaded_prep_handle,
        expected_type=StoredPreprocessor,
    )
    onehot_train_encoded = repl.object_store.get(
        onehot_train_handle,
        expected_type=pd.DataFrame,
    )
    onehot_score_encoded = repl.object_store.get(
        onehot_score_handle,
        expected_type=pd.DataFrame,
    )
    ordinal_train_encoded = repl.object_store.get(
        ordinal_result["dataframe_handle"],
        expected_type=pd.DataFrame,
    )
    ordinal_score_encoded = repl.object_store.get(
        ordinal_score_handle,
        expected_type=pd.DataFrame,
    )
    reloaded_score_encoded = repl.object_store.get(
        reloaded_score_handle,
        expected_type=pd.DataFrame,
    )
    inspected_summary = repl.object_store.summary(onehot_prep_handle)

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert onehot_artifact.is_fitted() is True
    assert reloaded_artifact.is_fitted() is True
    assert onehot_artifact.target_column == "target"
    assert "type" in inspected_summary["value"]
    assert inspected_summary["value"]["type"] == "StoredPreprocessor"
    assert "one_hot_encoder" in str(inspected_summary["value"]["groups"])
    assert "target" in onehot_train_encoded.columns
    assert "target" not in onehot_score_encoded.columns
    assert (
        onehot_score_encoded.columns.tolist() == reloaded_score_encoded.columns.tolist()
    )
    assert onehot_artifact.output_columns == onehot_score_encoded.columns.tolist()
    assert ordinal_result["preprocessor_handle"].startswith("prep_")
    assert ordinal_train_encoded.columns[-1] == "target"
    assert "categorical__city" in ordinal_score_encoded.columns
    assert float(ordinal_score_encoded.loc[0, "categorical__city"]) == -1.0
    assert (tmp_path / "output" / "preprocessor.joblib").is_file()
    assert "/workspace/output/preprocessor.joblib" in buffered["combined_output"]


def test_feature_engineering_helpers_can_fit_transform_and_compose(
    tmp_path: Path,
) -> None:
    """The FE collection should support deterministic transforms, aggregates, and preprocessing composition."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    train_frame = pd.DataFrame(
        {
            "premium": [100.0, 150.0, 220.0, 80.0],
            "income": [1000.0, 1200.0, 2000.0, 500.0],
            "city": ["a", "a", "b", "c"],
            "segment": ["retail", "retail", "enterprise", "retail"],
            "signup_date": pd.to_datetime(
                ["2024-01-01", "2024-02-15", "2024-03-20", "2024-01-10"]
            ),
            "target": [0, 1, 1, 0],
        }
    )
    score_frame = pd.DataFrame(
        {
            "premium": [90.0, 300.0],
            "income": [900.0, 2500.0],
            "city": ["a", "z"],
            "segment": ["retail", "unknown"],
            "signup_date": pd.to_datetime(["2024-04-01", "2024-05-10"]),
            "target": [1, 0],
        }
    )
    train_frame.to_csv(tmp_path / "train_fe.csv", index=False)
    score_frame.to_csv(tmp_path / "score_fe.csv", index=False)

    first = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/train_fe.csv')",
                "score_handle = load_csv('/workspace/score_fe.csv')",
                "fe_spec = build_feature_engineering_spec(",
                "    features=[",
                "        {'name': 'premium_income_ratio', 'kind': 'ratio', 'columns': ['premium', 'income']},",
                "        {'name': 'premium_gap', 'kind': 'difference', 'columns': ['income', 'premium']},",
                "        {'name': 'city_frequency', 'kind': 'category_frequency', 'column': 'city', 'normalize': True, 'fill_value': 0.0},",
                "        {'name': 'signup_parts', 'kind': 'datetime_part', 'column': 'signup_date', 'parts': ['month', 'dayofweek']},",
                "        {'name': 'segment_premium_mean', 'kind': 'groupby_aggregate', 'keys': ['segment'], 'source_column': 'premium', 'aggregation': 'mean', 'unknown_group_strategy': 'global'},",
                "        {'name': 'city_segment_count', 'kind': 'groupby_aggregate', 'keys': ['city', 'segment'], 'aggregation': 'count', 'unknown_group_strategy': 'constant', 'fill_value': -1},",
                "    ]",
                ")",
                "fe_handle = fit_feature_engineer(train_handle, fe_spec, target_column='target')",
                "fe_train = transform_with_feature_engineer(train_handle, fe_handle, include_target=True)",
                "fe_score = transform_with_feature_engineer(score_handle, fe_handle)",
                "prep_after_fe = build_preprocessing_spec(",
                "    numeric_steps=[{'kind': 'simple_imputer', 'strategy': 'median'}],",
                "    categorical_steps=[",
                "        {'kind': 'simple_imputer', 'strategy': 'most_frequent'},",
                "        {'kind': 'one_hot_encoder', 'handle_unknown': 'ignore', 'sparse_output': False},",
                "    ],",
                ")",
                "prep_after_fe_handle = fit_preprocessor(fe_train, prep_after_fe, target_column='target')",
                "post_fe_prep_score = transform_dataframe(fe_score, prep_after_fe_handle)",
                "prep_first = build_preprocessing_spec(",
                "    numeric_steps=[{'kind': 'simple_imputer', 'strategy': 'median'}],",
                "    categorical_steps=[",
                "        {'kind': 'simple_imputer', 'strategy': 'most_frequent'},",
                "        {'kind': 'ordinal_encoder', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1},",
                "    ],",
                ")",
                "prep_first_handle = fit_preprocessor(train_handle, prep_first, target_column='target')",
                "prep_first_train = transform_dataframe(train_handle, prep_first_handle, include_target=True)",
                "prep_first_score = transform_dataframe(score_handle, prep_first_handle)",
                "fe_after_prep_spec = build_feature_engineering_spec(",
                "    features=[",
                "        {'name': 'scaled_gap', 'kind': 'difference', 'columns': ['numeric__income', 'numeric__premium']},",
                "        {'name': 'encoded_city_abs', 'kind': 'absolute', 'column': 'categorical__city'},",
                "    ]",
                ")",
                "fe_after_prep = fit_feature_engineer(prep_first_train, fe_after_prep_spec, target_column='target')",
                "fe_after_prep_score = transform_with_feature_engineer(prep_first_score, fe_after_prep)",
                "print(inspect_feature_engineer(fe_handle))",
            ]
        ),
    )
    second = run_execute(
        repl,
        "\n".join(
            [
                "saved_fe_path = save_feature_engineer(fe_handle, '/workspace/output/feature_engineer.joblib')",
                "reloaded_fe = load_feature_engineer('/workspace/output/feature_engineer.joblib')",
                "reloaded_score = transform_with_feature_engineer(score_handle, reloaded_fe)",
                "print(inspect_handle(fe_handle))",
                "print(list_engineered_features(reloaded_fe))",
            ]
        ),
    )
    buffered = repl.results()

    state = repl.interpreter.state
    fe_handle = state["fe_handle"]
    fe_train_handle = state["fe_train"]
    fe_score_handle = state["fe_score"]
    post_fe_prep_score_handle = state["post_fe_prep_score"]
    fe_after_prep_handle = state["fe_after_prep"]
    fe_after_prep_score_handle = state["fe_after_prep_score"]
    reloaded_fe_handle = state["reloaded_fe"]
    reloaded_score_handle = state["reloaded_score"]

    fe_artifact = repl.object_store.get(
        fe_handle,
        expected_type=StoredFeatureEngineer,
    )
    reloaded_fe_artifact = repl.object_store.get(
        reloaded_fe_handle,
        expected_type=StoredFeatureEngineer,
    )
    fe_train_frame = repl.object_store.get(
        fe_train_handle,
        expected_type=pd.DataFrame,
    )
    fe_score_frame = repl.object_store.get(
        fe_score_handle,
        expected_type=pd.DataFrame,
    )
    post_fe_prep_score_frame = repl.object_store.get(
        post_fe_prep_score_handle,
        expected_type=pd.DataFrame,
    )
    fe_after_prep_artifact = repl.object_store.get(
        fe_after_prep_handle,
        expected_type=StoredFeatureEngineer,
    )
    fe_after_prep_score_frame = repl.object_store.get(
        fe_after_prep_score_handle,
        expected_type=pd.DataFrame,
    )
    reloaded_score_frame = repl.object_store.get(
        reloaded_score_handle,
        expected_type=pd.DataFrame,
    )
    inspected_summary = repl.object_store.summary(fe_handle)

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert fe_artifact.is_fitted() is True
    assert reloaded_fe_artifact.is_fitted() is True
    assert fe_artifact.target_column == "target"
    assert inspected_summary["value"]["type"] == "StoredFeatureEngineer"
    assert "groupby_aggregate" in str(inspected_summary["value"]["features"])
    assert "target" in fe_train_frame.columns
    assert "target" not in fe_score_frame.columns
    assert "premium_income_ratio" in fe_score_frame.columns
    assert "signup_parts__month" in fe_score_frame.columns
    assert np.isclose(fe_score_frame.loc[0, "premium_income_ratio"], 0.1)
    assert np.isclose(fe_score_frame.loc[0, "city_frequency"], 0.5)
    assert np.isclose(fe_score_frame.loc[1, "segment_premium_mean"], 137.5)
    assert float(fe_score_frame.loc[1, "city_segment_count"]) == -1.0
    assert "numeric__premium_income_ratio" in post_fe_prep_score_frame.columns
    assert fe_after_prep_artifact.engineered_columns == [
        "scaled_gap",
        "encoded_city_abs",
    ]
    assert "scaled_gap" in fe_after_prep_score_frame.columns
    assert "encoded_city_abs" in fe_after_prep_score_frame.columns
    assert fe_score_frame.columns.tolist() == reloaded_score_frame.columns.tolist()
    assert (tmp_path / "output" / "feature_engineer.joblib").is_file()
    assert "/workspace/output/feature_engineer.joblib" in buffered["combined_output"]


def test_feature_selection_helpers_can_generate_reports_and_metrics(
    tmp_path: Path,
) -> None:
    """The feature selection collection should generate diagnostic reports and evaluation metrics."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    train_frame = pd.DataFrame(
        {
            "signal_num": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
            "weak_num": [1, 2, 2, 3, 2, 3],
            "dup_signal": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
            "cat_signal": ["low", "low", "high", "high", "low", "high"],
            "constant_col": [1, 1, 1, 1, 1, 1],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    valid_frame = pd.DataFrame(
        {
            "signal_num": [0.12, 0.88, 0.18, 0.92],
            "weak_num": [1, 3, 2, 3],
            "dup_signal": [0.12, 0.88, 0.18, 0.92],
            "cat_signal": ["low", "high", "low", "high"],
            "constant_col": [1, 1, 1, 1],
            "target": [0, 1, 0, 1],
        }
    )
    train_frame.to_csv(tmp_path / "train_fs.csv", index=False)
    valid_frame.to_csv(tmp_path / "valid_fs.csv", index=False)

    first = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/train_fs.csv')",
                "valid_handle = load_csv('/workspace/valid_fs.csv')",
                "summary_handle = summarize_feature_candidates(train_handle, target_column='target')",
                "target_metrics_handle = compute_feature_target_metrics(train_handle, 'target', method='mutual_info')",
                "redundancy_handle = compute_feature_redundancy_metrics(train_handle, threshold=0.95)",
                "evaluation_handle = evaluate_feature_subset(train_handle, 'target', ['signal_num', 'cat_signal'], validation_handle=valid_handle)",
                "importance_handle = rank_feature_importance_with_lightgbm(train_handle, 'target', validation_handle=valid_handle)",
                "print(inspect_feature_selection_report(summary_handle))",
            ]
        ),
    )
    second = run_execute(
        repl,
        "\n".join(
            [
                "saved_report_path = save_feature_selection_report(target_metrics_handle, '/workspace/output/fs_report.joblib')",
                "reloaded_report = load_feature_selection_report('/workspace/output/fs_report.joblib')",
                "print(inspect_handle(target_metrics_handle))",
                "print(list_feature_selection_findings(reloaded_report))",
            ]
        ),
    )
    buffered = repl.results()

    state = repl.interpreter.state
    summary_handle = state["summary_handle"]
    target_metrics_handle = state["target_metrics_handle"]
    redundancy_handle = state["redundancy_handle"]
    evaluation_handle = state["evaluation_handle"]
    importance_handle = state["importance_handle"]
    reloaded_report_handle = state["reloaded_report"]

    summary_report = repl.object_store.get(
        summary_handle,
        expected_type=StoredFeatureSelectionReport,
    )
    target_report = repl.object_store.get(
        target_metrics_handle,
        expected_type=StoredFeatureSelectionReport,
    )
    redundancy_report = repl.object_store.get(
        redundancy_handle,
        expected_type=StoredFeatureSelectionReport,
    )
    evaluation_report = repl.object_store.get(
        evaluation_handle,
        expected_type=StoredFeatureSelectionReport,
    )
    importance_report = repl.object_store.get(
        importance_handle,
        expected_type=StoredFeatureSelectionReport,
    )
    reloaded_report = repl.object_store.get(
        reloaded_report_handle,
        expected_type=StoredFeatureSelectionReport,
    )
    inspected_summary = repl.object_store.summary(target_metrics_handle)

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert summary_report.report_type == "summary"
    assert any(
        row["feature"] == "constant_col" and row["zero_variance"]
        for row in summary_report.findings
    )
    assert target_report.report_type == "target_metrics"
    assert target_report.findings[0]["feature"] in {
        "signal_num",
        "dup_signal",
        "cat_signal",
    }
    assert any("Target-aware rankings" in warning for warning in target_report.warnings)
    assert redundancy_report.report_type == "redundancy"
    assert any(
        row.get("feature_a") == "signal_num" and row.get("feature_b") == "dup_signal"
        for row in redundancy_report.findings
    )
    assert evaluation_report.report_type == "subset_evaluation"
    assert evaluation_report.metrics["mode"] == "validation"
    assert "metrics" in evaluation_report.metrics
    assert importance_report.report_type == "importance"
    assert (
        importance_report.findings[0]["feature"]
        in train_frame.drop(columns=["target"]).columns
    )
    assert reloaded_report.findings == target_report.findings
    assert inspected_summary["value"]["type"] == "StoredFeatureSelectionReport"
    assert (tmp_path / "output" / "fs_report.joblib").is_file()
    assert "/workspace/output/fs_report.joblib" in buffered["combined_output"]


def test_visualizations_collection_can_plot_feature_importance_from_supported_handles(
    tmp_path: Path,
) -> None:
    """Visualization helpers should accept both report and tuned-pipeline handles."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    report_handle = repl.object_store.put(
        StoredFeatureSelectionReport(
            report_type="importance",
            method="lightgbm",
            feature_columns=["signal_a", "signal_b"],
            target_column="target",
            findings=[
                {"feature": "signal_a", "importance": 0.8},
                {"feature": "signal_b", "importance": 0.2},
            ],
        ),
        prefix="fs",
    )

    training_features = pd.DataFrame(
        {
            "signal_a": [0, 1, 0, 1, 0, 1],
            "signal_b": [1, 0, 1, 0, 1, 0],
        }
    )
    training_target = pd.Series([0, 1, 0, 1, 0, 1])
    estimator = metrics_support.build_lightgbm_estimator(
        task_type="classification",
        class_count=2,
        random_state=0,
    )
    estimator.fit(training_features, training_target)

    tuned_handle = repl.object_store.put(
        StoredTunedPipeline(
            pipeline_config={"model": {"kind": "lightgbm"}},
            fitted_model=estimator,
            model_feature_columns=["signal_a", "signal_b"],
            selected_features=["signal_a", "signal_b"],
            evaluation_summary={"status": "ok"},
        ),
        prefix="tuned",
    )
    unsupported_handle = repl.object_store.put(
        StoredTunedPipeline(
            pipeline_config={"model": {"kind": "unsupported"}},
            fitted_model=object(),
            model_feature_columns=[],
            selected_features=[],
            evaluation_summary={"status": "unsupported"},
        ),
        prefix="tuned",
    )

    success = run_execute(
        repl,
        "\n".join(
            [
                f"report_plot = plot_feature_importance('{report_handle}', '/workspace/output/report_importance.png')",
                f"tuned_plot = plot_feature_importance('{tuned_handle}', '/workspace/output/tuned_importance.png')",
                "print(report_plot)",
                "print(tuned_plot)",
            ]
        ),
    )
    failure = run_execute(
        repl,
        f"plot_feature_importance('{unsupported_handle}', '/workspace/output/unsupported.png')",
    )
    buffered = repl.results()

    report_plot = repl.interpreter.state["report_plot"]
    tuned_plot = repl.interpreter.state["tuned_plot"]

    assert success["status"] == "success"
    assert failure["status"] == "error"
    assert report_plot["source_type"] == "feature_selection_report"
    assert report_plot["importance_kind"] == "lightgbm"
    assert report_plot["feature_count"] == 2
    assert tuned_plot["source_type"] == "tuned_pipeline"
    assert tuned_plot["feature_count"] == 2
    assert tuned_plot["model_class"].startswith("LGBM")
    assert (tmp_path / "output" / "report_importance.png").is_file()
    assert (tmp_path / "output" / "tuned_importance.png").is_file()
    assert "/workspace/output/report_importance.png" in buffered["combined_output"]
    assert "/workspace/output/tuned_importance.png" in buffered["combined_output"]
    assert "does not expose `feature_importances_`, `coef_`" in str(failure["error"])


def test_metrics_and_splitting_helpers_create_reusable_handles(
    tmp_path: Path,
) -> None:
    """Metrics and splitting collections should create reusable artifacts."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    scored_frame = pd.DataFrame(
        {
            "feature": list(range(1, 11)),
            "target": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
            "prediction": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
            "probability": [0.05, 0.9, 0.8, 0.2, 0.7, 0.1, 0.15, 0.88, 0.76, 0.12],
        }
    )
    scored_frame.to_csv(tmp_path / "metrics_split.csv", index=False)

    execution = run_execute(
        repl,
        "\n".join(
            [
                "df_handle = load_csv('/workspace/metrics_split.csv')",
                "ppv_handle = create_ppv_scorer(top_k=2)",
                "auc_handle = create_metric_scorer('roc_auc', task_type='classification', needs_proba=True)",
                "splitter_handle = create_repeated_stratified_kfold_splitter(n_splits=2, n_repeats=2, random_state=0)",
                "split_handle = train_validation_test_split(df_handle, target_column='target', validation_size=0.2, test_size=0.2, stratify=True, random_state=0)",
                "score_summary = score_with_metric_handle(ppv_handle, y_true=[0, 1, 1, 0], y_pred=[0, 1, 1, 0], y_pred_proba=[0.1, 0.95, 0.9, 0.2])",
                "df_summary = evaluate_prediction_dataframe(",
                "    df_handle,",
                "    target_column='target',",
                "    prediction_column='prediction',",
                "    probability_column='probability',",
                "    scorer_handle=auc_handle,",
                ")",
                "print(inspect_metric_scorer(ppv_handle))",
                "print(inspect_splitter(splitter_handle))",
                "print(inspect_data_split(split_handle))",
            ]
        ),
    )
    buffered = repl.results()

    state = repl.interpreter.state
    ppv_handle = state["ppv_handle"]
    auc_handle = state["auc_handle"]
    splitter_handle = state["splitter_handle"]
    split_handle = state["split_handle"]
    score_summary = state["score_summary"]
    df_summary = state["df_summary"]

    ppv_scorer = repl.object_store.get(ppv_handle, expected_type=StoredMetricScorer)
    auc_scorer = repl.object_store.get(auc_handle, expected_type=StoredMetricScorer)
    splitter = repl.object_store.get(splitter_handle, expected_type=StoredSplitter)
    data_split = repl.object_store.get(split_handle, expected_type=StoredDataSplit)

    assert execution["status"] == "success"
    assert ppv_scorer.metric_name == "ppv"
    assert ppv_scorer.top_k == 2
    assert auc_scorer.metric_name == "roc_auc"
    assert auc_scorer.needs_proba is True
    assert splitter.splitter_kind == "repeated_stratified_kfold"
    assert splitter.requires_target is True
    assert data_split.validation_handle is not None
    assert data_split.test_handle is not None
    assert sum(data_split.row_counts.values()) == len(scored_frame)
    assert score_summary["metric_name"] == "ppv"
    assert score_summary["score"] == 1.0
    assert df_summary["metrics"]["accuracy"] == 1.0
    assert df_summary["metrics"]["roc_auc"] == 1.0
    assert "StoredMetricScorer" in buffered["combined_output"]
    assert "StoredSplitter" in buffered["combined_output"]
    assert "StoredDataSplit" in buffered["combined_output"]


def test_hpo_can_consume_metric_and_splitter_handles(
    tmp_path: Path,
) -> None:
    """HPO should accept reusable scorer and splitter handles."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    train_frame = pd.DataFrame(
        {
            "premium": [100.0, 120.0, 90.0, 240.0, 300.0, 80.0, 95.0, 260.0],
            "income": [1000.0, 1200.0, 900.0, 2000.0, 2400.0, 700.0, 750.0, 2100.0],
            "segment": ["low", "low", "low", "high", "high", "low", "low", "high"],
            "state": ["a", "a", "b", "b", "b", "a", "c", "b"],
            "target": [0, 0, 0, 1, 1, 0, 0, 1],
        }
    )
    train_frame.to_csv(tmp_path / "train_metric_handles.csv", index=False)

    execution = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/train_metric_handles.csv')",
                "metric_handle = create_ppv_scorer(top_k=2)",
                "splitter_handle = create_repeated_stratified_kfold_splitter(n_splits=2, n_repeats=2, random_state=0)",
                "fs_eval = evaluate_feature_subset(",
                "    train_handle,",
                "    'target',",
                "    ['premium', 'income', 'segment', 'state'],",
                "    scorer_handle=metric_handle,",
                "    splitter_handle=splitter_handle,",
                ")",
                "pipeline_config = {",
                "    'data': {'train_handle': train_handle, 'target_column': 'target'},",
                "    'preprocessing': {",
                "        'spec': build_preprocessing_spec(",
                "            numeric_steps=[{'kind': 'simple_imputer', 'strategy': 'median'}],",
                "            categorical_steps=[",
                "                {'kind': 'simple_imputer', 'strategy': 'most_frequent'},",
                "                {'kind': 'one_hot_encoder', 'handle_unknown': 'ignore', 'sparse_output': False},",
                "            ],",
                "        )",
                "    },",
                "    'model': {'base_params': {'n_estimators': 20, 'num_leaves': 8, 'learning_rate': 0.1, 'min_child_samples': 1}},",
                "    'evaluation': {",
                "        'mode': 'cross_validation',",
                "        'metric': 'auto',",
                "        'random_state': 0,",
                "        'scorer_handle': metric_handle,",
                "        'splitter_handle': splitter_handle,",
                "    },",
                "}",
                "search_space = [",
                "    {'path': 'model.base_params.num_leaves', 'kind': 'int', 'low': 4, 'high': 12, 'step': 4},",
                "]",
                "study_handle = create_hpo_study(pipeline_config, search_space, study_name='metric_handle_hpo')",
                "iteration = run_hpo_iteration(study_handle, 1, top_n=1)",
                "print(iteration)",
            ]
        ),
    )
    repl.results()

    state = repl.interpreter.state
    fs_eval_handle = state["fs_eval"]
    study_handle = state["study_handle"]
    study_artifact = repl.object_store.get(study_handle, expected_type=StoredHpoStudy)
    evaluation_report = repl.object_store.get(
        fs_eval_handle,
        expected_type=StoredFeatureSelectionReport,
    )

    assert execution["status"] == "success"
    assert evaluation_report.metrics["mode"] == "cross_validation"
    assert "ppv" in evaluation_report.metrics["summary"]["mean_metrics"]
    assert evaluation_report.metadata["scorer_handle"] == state["metric_handle"]
    assert evaluation_report.metadata["splitter_handle"] == state["splitter_handle"]
    assert study_artifact.objective_metric == "ppv"
    assert study_artifact.best_metrics["summary"]["mean_metrics"]["ppv"] >= 0.0


def test_hpo_inspection_exposes_sklearn_params_and_accepts_sklearn_search_space(
    tmp_path: Path,
) -> None:
    """HPO inspection should expose sklearn params and resolve sklearn-param search spaces."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    train_frame = pd.DataFrame(
        {
            "premium": [100.0, 120.0, 80.0, 240.0, 260.0, 90.0],
            "income": [1000.0, 1200.0, 850.0, 2000.0, 2200.0, 900.0],
            "loss": [10.0, 12.0, 4.0, 30.0, 40.0, 6.0],
            "segment": ["low", "low", "low", "high", "high", "low"],
            "target": [0, 0, 0, 1, 1, 0],
        }
    )
    valid_frame = pd.DataFrame(
        {
            "premium": [110.0, 95.0, 250.0],
            "income": [1100.0, 920.0, 2100.0],
            "loss": [11.0, 5.0, 32.0],
            "segment": ["low", "low", "high"],
            "target": [0, 0, 1],
        }
    )
    train_frame.to_csv(tmp_path / "inspect_train.csv", index=False)
    valid_frame.to_csv(tmp_path / "inspect_valid.csv", index=False)

    execution = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/inspect_train.csv')",
                "valid_handle = load_csv('/workspace/inspect_valid.csv')",
                "pipeline_config = {",
                "    'data': {'train_handle': train_handle, 'validation_handle': valid_handle, 'target_column': 'target'},",
                "    'freeform': {",
                "        'code': \"df['loss_ratio'] = np.where(df['premium'] > params['ratio_floor'], df['loss'] / df['premium'], np.nan)\",",
                "        'args': {'ratio_floor': 0.0},",
                "        'intent': 'feature_engineering',",
                "    },",
                "    'preprocessing': {",
                "        'spec': build_preprocessing_spec(",
                "            numeric_steps=[{'kind': 'simple_imputer', 'strategy': 'median'}],",
                "            categorical_steps=[",
                "                {'kind': 'simple_imputer', 'strategy': 'most_frequent'},",
                "                {'kind': 'ordinal_encoder', 'handle_unknown': 'use_encoded_value', 'unknown_value': -1},",
                "            ],",
                "        )",
                "    },",
                "    'model': {'base_params': {'n_estimators': 10, 'num_leaves': 8, 'learning_rate': 0.1, 'min_child_samples': 1}},",
                "    'evaluation': {'mode': 'validation', 'metric': 'f1', 'random_state': 0},",
                "}",
                "search_space = [",
                "    {'sklearn_param': 'model__num_leaves', 'kind': 'int', 'low': 4, 'high': 12, 'step': 4},",
                "    {'path': 'freeform.args.ratio_floor', 'kind': 'float', 'low': 0.0, 'high': 1.0},",
                "    {'sklearn_param': 'preprocessing__transformer__numeric__simple_imputer_1__strategy', 'kind': 'categorical', 'choices': ['mean', 'median']},",
                "]",
                "inspection = inspect_pipeline_tunable_params(pipeline_config, search_space)",
                "study_handle = create_hpo_study(pipeline_config, search_space, study_name='sklearn_alias_hpo')",
                "iteration = run_hpo_iteration(study_handle, 1, top_n=1)",
            ]
        ),
    )
    repl.results()

    inspection = repl.interpreter.state["inspection"]
    study_handle = repl.interpreter.state["study_handle"]
    study_artifact = repl.object_store.get(study_handle, expected_type=StoredHpoStudy)

    assert execution["status"] == "success"
    assert inspection["search_space"][0]["path"] == "model.base_params.num_leaves"
    assert inspection["search_space"][1]["path"] == "freeform.args.ratio_floor"
    assert (
        inspection["search_space"][2]["path"]
        == "preprocessing.spec.groups.0.steps.0.strategy"
    )
    assert (
        inspection["sklearn_param_aliases"]["model__num_leaves"]
        == "model.base_params.num_leaves"
    )
    assert (
        inspection["sklearn_param_aliases"]["freeform__params__ratio_floor"]
        == "freeform.args.ratio_floor"
    )
    assert any(
        row["sklearn_param"] == "model__num_leaves"
        for row in inspection["sklearn_pipeline_params"]
    )
    assert any(
        row["sklearn_param"] == "freeform__params__ratio_floor"
        for row in inspection["sklearn_pipeline_params"]
    )
    assert study_artifact.trials[0]["status"] in {"complete", "fail"}


def test_hpo_helpers_can_run_iterative_tuning_and_save_artifacts(
    tmp_path: Path,
) -> None:
    """The HPO collection should inspect tunables, run Optuna trials, and persist tuned artifacts."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    train_frame = pd.DataFrame(
        {
            "premium": [100.0, 120.0, np.nan, 240.0, 300.0, 80.0, 90.0, 260.0],
            "income": [1000.0, 1200.0, 900.0, 2000.0, 2400.0, 700.0, 650.0, 2100.0],
            "segment": ["low", "low", "low", "high", "high", "low", "low", "high"],
            "state": ["a", "a", "b", "b", "b", "a", "c", "b"],
            "target": [0, 0, 0, 1, 1, 0, 0, 1],
        }
    )
    valid_frame = pd.DataFrame(
        {
            "premium": [110.0, np.nan, 280.0, 95.0],
            "income": [1100.0, 850.0, 2300.0, 800.0],
            "segment": ["low", "low", "high", "low"],
            "state": ["a", "c", "b", "a"],
            "target": [0, 0, 1, 0],
        }
    )
    train_frame.to_csv(tmp_path / "train_hpo.csv", index=False)
    valid_frame.to_csv(tmp_path / "valid_hpo.csv", index=False)

    first = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/train_hpo.csv')",
                "valid_handle = load_csv('/workspace/valid_hpo.csv')",
                "pipeline_config = {",
                "    'data': {'train_handle': train_handle, 'validation_handle': valid_handle, 'target_column': 'target'},",
                "    'feature_engineering': {",
                "        'spec': build_feature_engineering_spec(features=[",
                "            {'name': 'loss_ratio', 'kind': 'ratio', 'columns': ['premium', 'income']}",
                "        ])",
                "    },",
                "    'preprocessing': {",
                "        'spec': build_preprocessing_spec(",
                "            numeric_steps=[{'kind': 'simple_imputer', 'strategy': 'median'}],",
                "            categorical_steps=[",
                "                {'kind': 'simple_imputer', 'strategy': 'most_frequent'},",
                "                {'kind': 'one_hot_encoder', 'handle_unknown': 'ignore', 'sparse_output': False},",
                "            ],",
                "        )",
                "    },",
                "    'model': {'base_params': {'n_estimators': 40, 'num_leaves': 8, 'learning_rate': 0.1, 'min_child_samples': 2}},",
                "    'evaluation': {'mode': 'validation', 'metric': 'f1', 'random_state': 0},",
                "}",
                "search_space = [",
                "    {'path': 'preprocessing.spec.groups.0.steps.0.strategy', 'kind': 'categorical', 'choices': ['mean', 'median']},",
                "    {'path': 'model.base_params.num_leaves', 'kind': 'int', 'low': 4, 'high': 16, 'step': 4},",
                "    {'path': 'model.base_params.learning_rate', 'kind': 'float', 'low': 0.05, 'high': 0.2},",
                "]",
                "tunable_summary = inspect_pipeline_tunable_params(pipeline_config, search_space)",
                "study_handle = create_hpo_study(pipeline_config, search_space, study_name='tiny_lgbm_hpo')",
                "iteration_result = run_hpo_iteration(study_handle, 2, top_n=2)",
                "best_config = inspect_hpo_best_config(study_handle)",
                "study_summary = summarize_hpo_study(study_handle)",
                "saved_tuned = save_tuned_pipeline(study_handle, '/workspace/output/tuned_pipeline.joblib')",
                "reloaded_tuned = load_tuned_pipeline('/workspace/output/tuned_pipeline.joblib')",
                "saved_hpo_report = save_hpo_study_report(study_handle, '/workspace/output/hpo_report.md')",
                "saved_trials_table = save_hpo_trials_table(study_handle, '/workspace/output/hpo_trials.csv')",
                "saved_importance_plot = save_hpo_parameter_importances_plot(study_handle, '/workspace/output/hpo_importances.html')",
                "saved_tuned_report = save_tuned_pipeline_report(reloaded_tuned, '/workspace/output/tuned_pipeline_report.md')",
                "saved_pipeline_python = export_best_pipeline_python(study_handle, '/workspace/output/best_pipeline.py', tuned_handle=reloaded_tuned)",
                "print(iteration_result)",
            ]
        ),
    )
    second = repl.results()

    state = repl.interpreter.state
    study_handle = state["study_handle"]
    reloaded_tuned_handle = state["reloaded_tuned"]
    study_artifact = repl.object_store.get(study_handle, expected_type=StoredHpoStudy)
    tuned_artifact = repl.object_store.get(
        reloaded_tuned_handle,
        expected_type=StoredTunedPipeline,
    )
    study_summary = state["study_summary"]
    best_config = state["best_config"]
    tunable_summary = state["tunable_summary"]
    iteration_result = state["iteration_result"]
    saved_tuned = state["saved_tuned"]
    saved_hpo_report = state["saved_hpo_report"]
    saved_trials_table = state["saved_trials_table"]
    saved_importance_plot = state["saved_importance_plot"]
    saved_tuned_report = state["saved_tuned_report"]
    saved_pipeline_python = state["saved_pipeline_python"]

    assert first["status"] == "success"
    assert study_artifact.best_config is not None
    assert len(study_artifact.trials) >= 2
    assert iteration_result["best_trial_number"] is not None
    assert best_config["best_config"]["model"]["base_params"]["n_estimators"] == 40
    assert any(
        row["path"] == "model.base_params.num_leaves" and row["is_tunable"]
        for row in tunable_summary["pipeline_params"]
    )
    assert (
        tunable_summary["pipeline_params_by_path"]["model.base_params.num_leaves"][
            "is_tunable"
        ]
        is True
    )
    assert (
        tunable_summary["return_schema"]["payload_type"] == "PipelineInspectionResult"
    )
    assert (
        tunable_summary["return_schema"]["top_level_fields"]["pipeline_params"][
            "container"
        ]
        == "list"
    )
    assert (
        tunable_summary["return_schema"]["top_level_fields"]["pipeline_params_by_path"][
            "container"
        ]
        == "dict[str, PipelineParamRow]"
    )
    assert "schema_reference" in tunable_summary
    assert any(
        "preprocessing.spec.groups" in note
        for note in tunable_summary["schema_reference"]["path_guidance"]
    )
    assert "example_pipeline_config" in tunable_summary["schema_reference"]
    assert study_summary["trial_count"] >= 2
    assert study_summary["completed_trial_count"] >= 1
    assert study_summary["failed_trial_count"] == 0
    assert len(study_summary["top_trials"]) >= 1
    assert tuned_artifact.model_feature_columns
    assert "evaluation_summary" in tuned_artifact.to_json_summary()
    assert saved_tuned["path"] == "/workspace/output/tuned_pipeline.joblib"
    assert saved_hpo_report["path"] == "/workspace/output/hpo_report.md"
    assert saved_trials_table == "/workspace/output/hpo_trials.csv"
    assert saved_importance_plot == "/workspace/output/hpo_importances.html"
    assert saved_tuned_report["path"] == "/workspace/output/tuned_pipeline_report.md"
    assert saved_pipeline_python["path"] == "/workspace/output/best_pipeline.py"
    assert (tmp_path / "output" / "tuned_pipeline.joblib").is_file()
    assert (tmp_path / "output" / "hpo_report.md").is_file()
    assert (tmp_path / "output" / "hpo_trials.csv").is_file()
    assert (tmp_path / "output" / "hpo_importances.html").is_file()
    assert (tmp_path / "output" / "tuned_pipeline_report.md").is_file()
    assert (tmp_path / "output" / "best_pipeline.py").is_file()
    assert "BEST_PIPELINE_CONFIG" in (
        tmp_path / "output" / "best_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "/workspace/output/tuned_pipeline.joblib" in second["combined_output"]
    assert "/workspace/output/hpo_report.md" in second["combined_output"]
    assert "/workspace/output/best_pipeline.py" in second["combined_output"]


def test_evaluate_tuned_pipeline_handles_binary_predict_proba_arrays(
    tmp_path: Path,
) -> None:
    """Saved tuned pipelines should evaluate binary probability matrices correctly."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    train_frame = pd.DataFrame(
        {
            "premium": [100.0, 120.0, 90.0, 240.0, 300.0, 80.0, 95.0, 260.0],
            "income": [1000.0, 1200.0, 900.0, 2000.0, 2400.0, 700.0, 750.0, 2100.0],
            "segment": ["low", "low", "low", "high", "high", "low", "low", "high"],
            "state": ["a", "a", "b", "b", "b", "a", "c", "b"],
            "target": [0, 0, 0, 1, 1, 0, 0, 1],
        }
    )
    validation_frame = pd.DataFrame(
        {
            "premium": [105.0, 255.0, 88.0, 285.0],
            "income": [1050.0, 2250.0, 880.0, 2350.0],
            "segment": ["low", "high", "low", "high"],
            "state": ["a", "b", "c", "b"],
            "target": [0, 1, 0, 1],
        }
    )
    holdout_frame = pd.DataFrame(
        {
            "premium": [110.0, 250.0, 85.0, 275.0],
            "income": [1100.0, 2200.0, 850.0, 2300.0],
            "segment": ["low", "high", "low", "high"],
            "state": ["a", "b", "c", "b"],
            "target": [0, 1, 0, 1],
        }
    )
    train_frame.to_csv(tmp_path / "train_tuned_eval.csv", index=False)
    validation_frame.to_csv(tmp_path / "validation_tuned_eval.csv", index=False)
    holdout_frame.to_csv(tmp_path / "holdout_tuned_eval.csv", index=False)

    execution = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/train_tuned_eval.csv')",
                "validation_handle = load_csv('/workspace/validation_tuned_eval.csv')",
                "holdout_handle = load_csv('/workspace/holdout_tuned_eval.csv')",
                "auc_handle = create_metric_scorer('roc_auc', task_type='classification', needs_proba=True)",
                "pipeline_config = {",
                "    'data': {'train_handle': train_handle, 'validation_handle': validation_handle, 'target_column': 'target'},",
                "    'preprocessing': {",
                "        'spec': build_preprocessing_spec(",
                "            numeric_steps=[{'kind': 'simple_imputer', 'strategy': 'median'}],",
                "            categorical_steps=[",
                "                {'kind': 'simple_imputer', 'strategy': 'most_frequent'},",
                "                {'kind': 'one_hot_encoder', 'handle_unknown': 'ignore', 'sparse_output': False},",
                "            ],",
                "        )",
                "    },",
                "    'model': {'base_params': {'n_estimators': 20, 'num_leaves': 8, 'learning_rate': 0.1, 'min_child_samples': 1}},",
                "    'evaluation': {'mode': 'validation', 'metric': 'roc_auc', 'random_state': 0, 'scorer_handle': auc_handle},",
                "}",
                "study_handle = create_hpo_study(pipeline_config, [])",
                "iteration_result = run_hpo_iteration(study_handle, 1, top_n=1)",
                "saved_tuned = save_tuned_pipeline(study_handle, '/workspace/output/tuned_eval_pipeline.joblib')",
                "holdout_metrics = evaluate_tuned_pipeline(",
                "    saved_tuned['tuned_handle'],",
                "    holdout_handle,",
                "    target_column='target',",
                "    scorer_handle=auc_handle,",
                ")",
            ]
        ),
    )

    state = repl.interpreter.state
    holdout_metrics = state["holdout_metrics"]

    assert execution["status"] == "success"
    assert state["iteration_result"]["completed_trial_count"] >= 1
    assert holdout_metrics["row_count"] == len(holdout_frame)
    assert "roc_auc" in holdout_metrics["metrics"]
    assert "log_loss" in holdout_metrics["metrics"]
    assert holdout_metrics["scorer_result"]["metric_name"] == "roc_auc"


def test_evaluate_tuned_pipeline_replays_saved_freeform_transformer_on_raw_input(
    tmp_path: Path,
) -> None:
    """Saved tuned pipelines should reapply attached freeform transforms at scoring time."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    train_frame = pd.DataFrame(
        {
            "premium": [100.0, 120.0, 90.0, 240.0, 300.0, 80.0, 95.0, 260.0],
            "loss": [10.0, 12.0, 9.0, 50.0, 70.0, 8.0, 10.0, 55.0],
            "target": [0, 0, 0, 1, 1, 0, 0, 1],
        }
    )
    validation_frame = pd.DataFrame(
        {
            "premium": [105.0, 255.0, 88.0, 285.0],
            "loss": [11.0, 52.0, 8.5, 62.0],
            "target": [0, 1, 0, 1],
        }
    )
    holdout_frame = pd.DataFrame(
        {
            "premium": [110.0, 250.0, 85.0, 275.0],
            "loss": [10.5, 49.0, 7.5, 58.0],
            "target": [0, 1, 0, 1],
        }
    )
    train_frame.to_csv(tmp_path / "train_tuned_freeform_eval.csv", index=False)
    validation_frame.to_csv(
        tmp_path / "validation_tuned_freeform_eval.csv", index=False
    )
    holdout_frame.to_csv(tmp_path / "holdout_tuned_freeform_eval.csv", index=False)

    execution = run_execute(
        repl,
        "\n".join(
            [
                "train_handle = load_csv('/workspace/train_tuned_freeform_eval.csv')",
                "validation_handle = load_csv('/workspace/validation_tuned_freeform_eval.csv')",
                "holdout_handle = load_csv('/workspace/holdout_tuned_freeform_eval.csv')",
                "pipeline_config = {",
                "    'data': {'train_handle': train_handle, 'validation_handle': validation_handle, 'target_column': 'target'},",
                "    'freeform': {",
                '        \'code\': """',
                "df['premium_floor'] = np.where(df['premium'] > params['premium_floor'], df['premium'], params['premium_floor'])",
                "df['loss_ratio'] = df['loss'] / df['premium_floor']",
                '""",',
                "        'args': {'premium_floor': 1.0},",
                "    },",
                "    'model': {'base_params': {'n_estimators': 20, 'num_leaves': 8, 'learning_rate': 0.1, 'min_child_samples': 1}},",
                "    'evaluation': {'mode': 'validation', 'metric': 'roc_auc', 'random_state': 0},",
                "}",
                "study_handle = create_hpo_study(pipeline_config, [])",
                "iteration_result = run_hpo_iteration(study_handle, 1, top_n=1)",
                "saved_tuned = save_tuned_pipeline(study_handle, '/workspace/output/tuned_freeform_eval_pipeline.joblib')",
                "holdout_metrics = evaluate_tuned_pipeline(",
                "    saved_tuned['tuned_handle'],",
                "    holdout_handle,",
                "    target_column='target',",
                ")",
            ]
        ),
    )

    state = repl.interpreter.state
    holdout_metrics = state["holdout_metrics"]

    assert execution["status"] == "success"
    assert state["iteration_result"]["completed_trial_count"] >= 1
    assert holdout_metrics["row_count"] == len(holdout_frame)
    assert "roc_auc" in holdout_metrics["metrics"]
    assert holdout_metrics["tuned_pipeline"]["feature_count"] >= 3


def test_hpo_iteration_surfaces_failed_trials_and_reasons(tmp_path: Path) -> None:
    """Failed Optuna trials should be summarized instead of disappearing silently."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    execution = run_execute(
        repl,
        "\n".join(
            [
                "pipeline_config = {",
                "    'data': {'train_handle': 'missing_df', 'validation_handle': None, 'target_column': 'target'},",
                "    'model': {'base_params': {'n_estimators': 10, 'num_leaves': 4, 'learning_rate': 0.1, 'min_child_samples': 1}},",
                "    'evaluation': {'mode': 'cross_validation', 'metric': 'f1', 'random_state': 0, 'cv_folds': 2},",
                "}",
                "search_space = [",
                "    {'path': 'model.base_params.num_leaves', 'kind': 'int', 'low': 4, 'high': 8, 'step': 4},",
                "]",
                "study_handle = create_hpo_study(pipeline_config, search_space, study_name='failing_hpo')",
                "iteration_result = run_hpo_iteration(study_handle, 1, top_n=1)",
                "study_summary = summarize_hpo_study(study_handle)",
            ]
        ),
    )

    assert execution["status"] == "success"
    assert execution["persisted_variables"] == [
        "iteration_result",
        "pipeline_config",
        "search_space",
        "study_handle",
        "study_summary",
    ]
    assert repl.interpreter.state["iteration_result"]["failed_trial_count"] == 1
    assert repl.interpreter.state["iteration_result"]["recent_failures"]
    assert "Unknown object handle" in str(
        repl.interpreter.state["iteration_result"]["recent_failures"][0][
            "failure_reason"
        ]
    )
    assert repl.interpreter.state["study_summary"]["failed_trial_count"] == 1


def test_hpo_rejects_executable_feature_selection_and_stage_order(
    tmp_path: Path,
) -> None:
    """The fixed-order HPO config should not accept executable feature selection or stage_order."""
    repl = MontyPythonREPL(workspace_root=tmp_path)
    frame = pd.DataFrame({"premium": [1.0, 2.0], "target": [0, 1]})
    frame.to_csv(tmp_path / "invalid_hpo.csv", index=False)

    seed = run_execute(repl, "train_handle = load_csv('/workspace/invalid_hpo.csv')")
    failure = run_execute(
        repl,
        "\n".join(
            [
                "pipeline_config = {",
                "    'data': {'train_handle': train_handle, 'target_column': 'target'},",
                "    'feature_selection': {'feature_columns': ['premium']},",
                "    'stage_order': ['freeform', 'preprocessing'],",
                "    'model': {'base_params': {'n_estimators': 10}},",
                "    'evaluation': {'mode': 'cross_validation', 'metric': 'f1', 'random_state': 0},",
                "}",
                "create_hpo_study(pipeline_config, [])",
            ]
        ),
    )

    assert seed["status"] == "success"
    assert failure["status"] == "error"
    assert "Unknown pipeline config keys" in str(failure["error"])


def test_evaluate_feature_subset_applies_model_params(monkeypatch: Any) -> None:
    """Evaluation should pass sampled model params into the LightGBM estimator."""

    captured_params: list[dict[str, Any]] = []

    class DummyEstimator:
        """Minimal estimator stub that records applied params."""

        def __init__(self) -> None:
            self.params: dict[str, Any] = {}

        def set_params(self, **params: Any) -> DummyEstimator:
            self.params.update(params)
            return self

        def fit(self, X: pd.DataFrame, y: pd.Series) -> DummyEstimator:
            del X, y
            captured_params.append(dict(self.params))
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            del X
            return np.array([0, 1])

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            del X
            return np.array([[0.9, 0.1], [0.1, 0.9]])

    monkeypatch.setattr(
        metrics_support,
        "build_lightgbm_estimator",
        lambda **kwargs: DummyEstimator(),
    )

    summary, warnings = metrics_support.evaluate_feature_subset(
        pd.DataFrame({"feature": [0.1, 0.9]}),
        pd.Series([0, 1]),
        validation_features=pd.DataFrame({"feature": [0.2, 0.8]}),
        validation_target=pd.Series([0, 1]),
        model_params={"num_leaves": 99, "learning_rate": 0.25},
    )

    assert warnings
    assert summary["mode"] == "validation"
    assert captured_params == [{"num_leaves": 99, "learning_rate": 0.25}]


def test_prepare_model_frames_raises_clear_error_on_validation_schema_mismatch() -> (
    None
):
    """Validation schema drift should fail fast with a helpful column diff."""
    train_frame = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    validation_frame = pd.DataFrame(
        {
            "feature_a": [1.5, 2.5],
            "feature_b": [3.5, 4.5],
            "feature_c": [5.0, 6.0],
        }
    )

    with pytest.raises(ValueError) as exc_info:
        metrics_support.prepare_model_frames(train_frame, validation_frame)

    error_text = str(exc_info.value)
    assert "Validation feature columns do not match" in error_text
    assert "Unexpected validation columns: feature_c." in error_text
