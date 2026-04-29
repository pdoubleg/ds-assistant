"""Tests for the hackathon Monty Python REPL package."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_repl_module = importlib.import_module("src.mcp.monty_python_repl_hackathon.repl")
_registry_module = importlib.import_module(
    "src.mcp.monty_python_repl_hackathon.registry"
)

HackathonMontyPythonREPL = _repl_module.HackathonMontyPythonREPL
StoredLightGBMModelArtifact = _registry_module.StoredLightGBMModelArtifact
StoredLightGBMStudy = _registry_module.StoredLightGBMStudy


def _run_execute(repl: HackathonMontyPythonREPL, code: str) -> dict[str, Any]:
    """Execute async REPL code inside a synchronous test.

    Args:
        repl: Active hackathon REPL.
        code: Python code to execute.

    Returns:
        Execute payload returned by the REPL.
    """
    return asyncio.run(repl.execute(code))


def _make_training_frame() -> pd.DataFrame:
    """Create a compact binary-classification dataframe for tests.

    Returns:
        Synthetic training dataframe with numeric and categorical features.
    """
    rows: list[dict[str, Any]] = []
    for index in range(120):
        bucket = index % 6
        segment = f"segment_{bucket % 3}"
        score_signal = (index % 10) / 10.0
        target = 1 if bucket in (0, 1) else 0
        rows.append(
            {
                "customer_id": index,
                "segment": segment,
                "balance": 50 + index * 3,
                "score_signal": score_signal,
                "utilization": (index % 15) / 15.0,
                "target": target,
            }
        )
    return pd.DataFrame(rows)


def test_package_has_no_repo_internal_imports() -> None:
    """The hackathon package should not depend on repo-internal modules."""
    package_root = PROJECT_ROOT / "src" / "mcp" / "monty_python_repl_hackathon"
    forbidden_snippets = [
        "from src.mcp.monty_python_repl",
        "import src.mcp.monty_python_repl",
        "from src.rlm",
        "from src.tools",
        "from src.clai",
        "from src.message_history",
    ]

    for path in package_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in text, f"Found forbidden import {snippet!r} in {path}"


def test_results_suppress_raw_stdout_and_values(tmp_path: Path) -> None:
    """The REPL should suppress raw stdout and avoid leaking row values."""
    dataframe = pd.DataFrame(
        {
            "city": ["secret_city", "other_city"],
            "amount": [100.0, 125.0],
            "target": [1, 0],
        }
    )
    dataframe.to_csv(tmp_path / "input.csv", index=False)
    repl = HackathonMontyPythonREPL(workspace_root=tmp_path)

    _run_execute(
        repl,
        "\n".join(
            [
                "payload = load_csv('/workspace/input.csv')",
                "df_handle = payload['dataframe_handle']",
                "print(inspect_handle(df_handle))",
            ]
        ),
    )
    buffered = repl.results()

    rendered = str(buffered)
    assert "secret_city" not in rendered
    assert "other_city" not in rendered
    assert buffered["executions"][0]["stdout"]["suppressed"] is True
    assert "suppressed for privacy" in buffered["combined_output"]


def test_freeform_errors_are_sanitized(tmp_path: Path) -> None:
    """Freeform runtime errors should not leak raw row values."""
    dataframe = pd.DataFrame(
        {
            "city": ["secret_city", "other_city"],
            "amount": [100.0, 125.0],
            "target": [1, 0],
        }
    )
    dataframe.to_csv(tmp_path / "input.csv", index=False)
    repl = HackathonMontyPythonREPL(workspace_root=tmp_path)

    _run_execute(
        repl,
        "\n".join(
            [
                "payload = load_csv('/workspace/input.csv')",
                "df_handle = payload['dataframe_handle']",
                "bad_code = \"print(df.iloc[0].to_dict())\\nraise ValueError(df.iloc[0]['city'])\"",
                "result = run_dataframe_code(df_handle, bad_code)",
            ]
        ),
    )
    buffered = repl.results()

    rendered = str(buffered)
    assert "secret_city" not in rendered
    assert "other_city" not in rendered
    assert buffered["executions"][0]["status"] == "error"
    assert buffered["executions"][0]["error"]["error_type"] == "CodeExecutionError"


def test_load_csv_and_screen_features_workflow(tmp_path: Path) -> None:
    """CSV loading and feature screening should produce reusable outputs."""
    dataframe = _make_training_frame()
    dataframe.to_csv(tmp_path / "train.csv", index=False)
    repl = HackathonMontyPythonREPL(workspace_root=tmp_path)

    execute_payload = _run_execute(
        repl,
        "\n".join(
            [
                "payload = load_csv('/workspace/train.csv')",
                "df_handle = payload['dataframe_handle']",
                "screen = screen_features(",
                "    df_handle,",
                "    'target',",
                "    id_columns=['customer_id'],",
                "    top_k_univariate=3,",
                ")",
            ]
        ),
    )

    assert execute_payload["status"] == "success"
    screen = repl.interpreter.state["screen"]
    assert screen["selected_columns"]
    assert len(screen["selected_columns"]) <= 3
    assert "customer_id" not in screen["selected_columns"]


def test_train_lightgbm_baseline_uses_native_categoricals(tmp_path: Path) -> None:
    """Baseline LightGBM training should preserve categorical feature handling."""
    dataframe = _make_training_frame()
    dataframe.to_csv(tmp_path / "train.csv", index=False)
    repl = HackathonMontyPythonREPL(workspace_root=tmp_path)

    _run_execute(
        repl,
        "\n".join(
            [
                "payload = load_csv('/workspace/train.csv')",
                "df_handle = payload['dataframe_handle']",
                "model_result = train_lightgbm_baseline(",
                "    df_handle,",
                "    'target',",
                "    id_columns=['customer_id'],",
                "    num_threads=1,",
                ")",
            ]
        ),
    )
    model_result = repl.interpreter.state["model_result"]
    artifact = repl.object_store.get(
        model_result["model_handle"],
        expected_type=StoredLightGBMModelArtifact,
    )

    assert "segment" in artifact.categorical_columns
    assert "valid_ppv_at_5" in artifact.evaluation_summary
    assert artifact.best_iteration >= 1


def test_tune_lightgbm_and_fit_best_model(tmp_path: Path) -> None:
    """Optuna tuning should produce a study and a fitted best-model artifact."""
    dataframe = _make_training_frame()
    dataframe.to_csv(tmp_path / "train.csv", index=False)
    repl = HackathonMontyPythonREPL(workspace_root=tmp_path)

    _run_execute(
        repl,
        "\n".join(
            [
                "payload = load_csv('/workspace/train.csv')",
                "df_handle = payload['dataframe_handle']",
                "study_result = tune_lightgbm(",
                "    df_handle,",
                "    'target',",
                "    id_columns=['customer_id'],",
                "    n_trials=2,",
                "    num_threads=1,",
                ")",
                "best_model = fit_best_lightgbm(study_result['study_handle'])",
            ]
        ),
    )

    study_result = repl.interpreter.state["study_result"]
    best_model = repl.interpreter.state["best_model"]
    study_artifact = repl.object_store.get(
        study_result["study_handle"],
        expected_type=StoredLightGBMStudy,
    )
    model_artifact = repl.object_store.get(
        best_model["model_handle"],
        expected_type=StoredLightGBMModelArtifact,
    )

    assert len(study_artifact.study.trials) >= 2
    assert model_artifact.evaluation_summary["valid_ppv_at_5"] >= 0.0


def test_plot_tools_return_artifacts_without_payloads(tmp_path: Path) -> None:
    """Plot tools should save files and return safe aggregate metadata only."""
    dataframe = _make_training_frame()
    dataframe.to_csv(tmp_path / "train.csv", index=False)
    repl = HackathonMontyPythonREPL(workspace_root=tmp_path)

    _run_execute(
        repl,
        "\n".join(
            [
                "payload = load_csv('/workspace/train.csv')",
                "df_handle = payload['dataframe_handle']",
                "plot_result = plot_missingness(df_handle, '/workspace/output/missing.png')",
            ]
        ),
    )
    plot_result = repl.interpreter.state["plot_result"]

    assert plot_result["plot_type"] == "missingness_bar"
    assert plot_result["path"] == "/workspace/output/missing.png"
    assert (tmp_path / "output" / "missing.png").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow is required for parquet tests",
)
def test_load_parquet_slice_supports_local_workspace_paths(tmp_path: Path) -> None:
    """The parquet loader should support local workspace parquet datasets."""
    dataframe = _make_training_frame()
    parquet_dir = tmp_path / "parts"
    parquet_dir.mkdir()
    dataframe.iloc[:60].to_parquet(parquet_dir / "part_a.parquet", index=False)
    dataframe.iloc[60:].to_parquet(parquet_dir / "part_b.parquet", index=False)

    repl = HackathonMontyPythonREPL(workspace_root=tmp_path)
    _run_execute(
        repl,
        "\n".join(
            [
                "slice_payload = load_parquet_slice(",
                "    '/workspace/parts',",
                "    label_col='target',",
                "    id_cols=['customer_id'],",
                "    sample_n_rows=20,",
                "    max_files=1,",
                ")",
            ]
        ),
    )
    slice_payload = repl.interpreter.state["slice_payload"]

    assert slice_payload["summary"]["shape"][0] <= 20
    assert "customer_id" in slice_payload["summary"]["columns"]
