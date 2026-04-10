"""Focused tests for the Monty-backed MCP Python REPL."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from src.mcp.monty_python_repl import FunctionRegistry, MontyPythonREPL
from src.mcp.monty_python_repl.registry import ToolCollection, tool


def run_execute(repl: MontyPythonREPL, code: str) -> dict[str, object]:
    """Execute sandbox code inside a synchronous pytest test."""
    return asyncio.run(repl.execute(code))


def test_help_lists_default_collections_and_repl_notes(tmp_path: Path) -> None:
    """The default help payload should advertise collections and usage notes."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    payload = repl.help()

    collection_names = {item["name"] for item in payload["collections"]}
    data_io_collection = next(
        item for item in payload["collections"] if item["name"] == "data_io"
    )
    dataframe_collection = next(
        item for item in payload["collections"] if item["name"] == "dataframe"
    )
    handles_collection = next(
        item for item in payload["collections"] if item["name"] == "handles"
    )
    plotly_collection = next(
        item for item in payload["collections"] if item["name"] == "plotly"
    )

    assert payload["functions"] == []
    assert {"data_io", "dataframe", "handles", "plotly"} <= collection_names
    assert "load_csv" in data_io_collection["tools"]
    assert "save_excel" in data_io_collection["tools"]
    assert "list_workspace_files" in data_io_collection["tools"]
    assert "dataframe_head" in dataframe_collection["tools"]
    assert "groupby_aggregate" in dataframe_collection["tools"]
    assert "inspect_handle" in handles_collection["tools"]
    assert "list_object_handles" in handles_collection["tools"]
    assert "save_plotly_figure" in plotly_collection["tools"]
    assert "create_bar_chart" in plotly_collection["tools"]
    assert any("persist automatically" in note for note in payload["notes"])
    assert any("results tool returns and clears" in note for note in payload["notes"])
    assert any("help(collection='<name>')" in note for note in payload["notes"])


def test_help_can_filter_by_collection_and_surface_arguments(tmp_path: Path) -> None:
    """Collection help should expose tool metadata parsed from decorated docstrings."""
    repl = MontyPythonREPL(workspace_root=tmp_path)

    payload = repl.help(collection="data_io")
    function_names = {item["name"] for item in payload["functions"]}
    load_csv_help = next(item for item in payload["functions"] if item["name"] == "load_csv")
    single_tool_payload = repl.help(name="load_csv")
    nrows_argument = next(
        argument for argument in load_csv_help["arguments"] if argument["name"] == "nrows"
    )

    assert payload["collection"] == "data_io"
    assert "load_csv" in function_names
    assert "save_excel" in function_names
    assert "list_workspace_files" in function_names
    assert load_csv_help["collection"] == "data_io"
    assert load_csv_help["description"] == "Load a CSV file from `/workspace` and return a dataframe handle."
    assert load_csv_help["usage_example"] == "df_handle = load_csv('/workspace/input/data.csv')"
    assert nrows_argument["annotation"] == "int | None"
    assert nrows_argument["default"] is None
    assert nrows_argument["required"] is False
    assert nrows_argument["description"] == "Optional maximum row count to load."
    assert single_tool_payload["functions"][0]["collection"] == "data_io"


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
        """
        return value * factor

    registry.register(
        scale_value,
        usage_example="print(scale_value(3, factor=4))",
        categories=("custom",),
        collection="math",
        collection_description="Custom math helpers for registry injection.",
    )

    repl = MontyPythonREPL(workspace_root=tmp_path, registry=registry)
    payload = repl.help("scale_value")
    factor_argument = next(
        argument for argument in payload["functions"][0]["arguments"] if argument["name"] == "factor"
    )

    assert payload["functions"][0]["signature"].startswith("scale_value(")
    assert payload["functions"][0]["categories"] == ["custom"]
    assert payload["functions"][0]["collection"] == "math"
    assert payload["functions"][0]["usage_example"] == "print(scale_value(3, factor=4))"
    assert factor_argument["description"] == "Multiplier to apply."
    assert factor_argument["default"] == 2


def test_registry_can_register_decorated_collections(tmp_path: Path) -> None:
    """Decorated collections should expose docstring-derived help metadata."""

    class MathCollection(ToolCollection):
        """Custom arithmetic helpers."""

        name = "math"
        description = "Small arithmetic helpers for custom registry composition."

        @tool(categories=("custom",), usage_example="print(scale_value(3, factor=4))")
        def scale_value(self, value: int, factor: int = 2) -> int:
            """Scale an integer by the requested factor.

            Args:
                value (int): Base integer value.
                factor (int): Multiplier to apply.

            Returns:
                int: Scaled integer result.
            """
            return value * factor

    registry = FunctionRegistry()
    registry.register_collection(MathCollection())

    repl = MontyPythonREPL(workspace_root=tmp_path, registry=registry)
    payload = repl.help(collection="math")
    function_help = payload["functions"][0]
    factor_argument = next(
        argument for argument in function_help["arguments"] if argument["name"] == "factor"
    )
    execution = run_execute(repl, "print(scale_value(3, factor=4))")
    buffered = repl.results()

    assert payload["collections"][0]["name"] == "math"
    assert function_help["signature"].startswith("scale_value(")
    assert function_help["description"] == "Scale an integer by the requested factor."
    assert function_help["categories"] == ["custom"]
    assert function_help["collection"] == "math"
    assert function_help["usage_example"] == "print(scale_value(3, factor=4))"
    assert factor_argument["annotation"] == "int"
    assert factor_argument["default"] == 2
    assert factor_argument["description"] == "Multiplier to apply."
    assert execution["status"] == "success"
    assert "12" in buffered["combined_output"]


def test_register_collection_rejects_duplicate_tool_names() -> None:
    """Multiple collections should not be allowed to silently override tools."""

    class FirstCollection(ToolCollection):
        """First collection used to seed the registry."""

        name = "first"

        @tool(name="shared_tool")
        def one(self) -> str:
            """Return the first value.

            Returns:
                str: First string value.
            """
            return "first"

    class SecondCollection(ToolCollection):
        """Second collection that conflicts with the first."""

        name = "second"

        @tool(name="shared_tool")
        def two(self) -> str:
            """Return the second value.

            Returns:
                str: Second string value.
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


def test_help_preserves_alphabetical_function_order_within_collection(
    tmp_path: Path,
) -> None:
    """Collection help should keep function ordering stable and alphabetical."""
    registry = FunctionRegistry()

    def beta_tool() -> str:
        """Return the beta marker.

        Returns:
            str: Beta marker.
        """
        return "beta"

    def alpha_tool() -> str:
        """Return the alpha marker.

        Returns:
            str: Alpha marker.
        """
        return "alpha"

    registry.register(
        beta_tool,
        collection="letters",
        collection_description="Simple letter helpers.",
    )
    registry.register(alpha_tool, collection="letters")

    repl = MontyPythonREPL(workspace_root=tmp_path, registry=registry)
    payload = repl.help(collection="letters")

    assert [item["name"] for item in payload["functions"]] == ["alpha_tool", "beta_tool"]


def test_execute_persists_assigned_state_and_results_are_drained(tmp_path: Path) -> None:
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
    assert {"df_handle", "summary_handle", "fig_handle", "saved_plot_paths", "saved_excel_path"} <= set(
        first["persisted_variables"]
    )
    assert (tmp_path / "output" / "chart.html").is_file()
    assert (tmp_path / "output" / "report.xlsx").is_file()
    assert len(buffered["executions"]) == 2
    assert "/workspace/output/chart.html" in buffered["combined_output"]
    assert "/workspace/output/report.xlsx" in buffered["combined_output"]
    assert "segment" in buffered["combined_output"]
