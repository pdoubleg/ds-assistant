# `monty_python_repl_minimal`

`monty_python_repl_minimal` is a modeling-focused MCP server that exposes a safe, persistent Python REPL for tabular workflows. It is designed for exploratory analysis, feature work, and LightGBM-based modeling without exposing raw training rows, raw workspace text, or row-level chart payloads.

The server is implemented as a FastMCP tool surface in `server.py` and can be started over stdio with:

```bash
uv run -m src.mcp.monty_python_repl_minimal
```

## What The Server Exposes

This MCP server intentionally keeps its public API small. Clients interact through three top-level tools:

### `help(name: str | None = None) -> str`

Use `help()` to discover the available collections, then `help("<collection>")` or `help("<tool>")` to inspect a specific helper before executing code.

- Returns human-readable help text.
- Supports both collection names and individual tool names.
- Best first call for new clients and agents.

### `execute(code: str) -> dict[str, Any]`

Use `execute(...)` to run Python inside the persistent Monty sandbox.

- The runtime is stateful across calls.
- Registered helpers are the only way to load data, inspect schemas, build features, train models, and save artifacts.
- Standard stdout is suppressed for privacy, so `print(...)` is not the main output channel.
- The immediate return value is intentionally compact and points callers to `results()`.

### `results() -> dict[str, Any]`

Use `results()` to retrieve the buffered execution history from prior `execute(...)` calls.

- Returns detailed, privacy-safe execution records.
- Surfaces helper summaries, stored handles, created artifacts, and sanitized errors.
- Clears the pending result buffer after returning it.

## Typical Workflow

Most clients should follow this pattern:

1. Call `help()` to discover collections and tool names.
2. Call `help("<collection-or-tool>")` when a helper is unfamiliar.
3. Call `execute(...)` with short orchestration-focused code.
4. Call `results()` to inspect the buffered summaries and handles.

Example:

`help()`

`help('data_access')`

`help('load_csv')`

```python
execute("train_df = load_csv('data/train.csv')")
```

`results()`

A few important runtime rules:

- Prefer registered helpers over ad hoc sandbox logic.
- Prefer `write_workspace_text(...)`, `write_workspace_json(...)`, `read_workspace_text(...)`, and `read_workspace_json(...)` over direct file IO in executed code.
- End `execute(...)` with a compact expression when you want `results()` to expose a specific final value.
- Avoid `import pandas as pd` inside `execute(...)`; helpers handle heavier dependencies internally.

## Collections

The default registry is organized into nine collections.

### `data_access`

Load local or remote tabular data into reusable dataframe handles without returning raw row previews.

- `load_csv(...)`: Load a CSV from `/workspace` into a stored dataframe handle.
- `load_parquet_slice(...)`: Load a sampled parquet slice, including remote parquet sources such as S3-backed datasets.
- `select_columns(...)`: Build a new dataframe handle containing only the requested columns.

### `workspace`

Use these helpers for safe text and JSON file access inside `/workspace`.

- `list_workspace_files(...)`: List files under `/workspace`.
- `read_workspace_text(...)`: Read a supported text file from `/workspace`.
- `write_workspace_text(...)`: Write a supported text file to `/workspace`.
- `read_workspace_json(...)`: Read and parse a JSON file.
- `write_workspace_json(...)`: Serialize JSON data into a workspace file.

### `handles`

Inspect objects already stored in Monty's in-memory object store.

- `list_object_handles()`: List active handles currently available in memory.
- `inspect_handle(...)`: Return a privacy-safe summary for any stored handle.

### `schema_views`

Inspect dataframe shape, columns, dtypes, and aggregate column summaries without exposing raw row values.

- `dataframe_shape(...)`: Return row and column counts.
- `dataframe_columns(...)`: Return the dataframe's column names.
- `dataframe_dtypes(...)`: Return dtypes keyed by column name.
- `summarize_dataframe(...)`: Return a lightweight dataframe overview.
- `summarize_dataframe_columns(...)`: Return focused summaries for a selected set of columns.
- `summarize_target(...)`: Return aggregate statistics for a target column.

### `eda`

Support privacy-safe exploratory work on wide tables.

- `triage_dataframe(...)`: Assess dataset size and recommend an EDA workflow.
- `plan_feature_subsets(...)`: Split a wide dataframe into deterministic feature batches.
- `summarize_feature_subset(...)`: Summarize a chosen batch of feature columns.

### `visualizations`

Create aggregate plots and save them to `/workspace`.

- `plot_missingness(...)`: Save a missingness bar chart.
- `plot_numeric_histogram(...)`: Save a histogram for a numeric feature.
- `plot_target_rate_by_numeric_bin(...)`: Save a target-rate-by-bin chart.
- `plot_prediction_diagnostics(...)`: Save score-bucket diagnostics for predictions.
- `plot_prediction_vs_actual_slices(...)`: Save global and feature-sliced prediction diagnostics.
- `plot_feature_importance(...)`: Save a feature-importance chart for a fitted model.

### `feature_selection`

Screen, rank, and refine candidate feature sets before full model training.

- `screen_features(...)`: Apply descriptive and univariate screening filters.
- `analyze_feature_correlation(...)`: Detect highly correlated numeric features and suggest drops.
- `rank_features_by_lightgbm(...)`: Rank candidate features with a lightweight LightGBM pass.
- `rank_feature_subsets(...)`: Compare multiple feature subsets with repeated LightGBM runs.
- `apply_feature_report(...)`: Apply a saved feature-selection report to a dataframe.
- `inspect_feature_report(...)`: Inspect a stored feature report.
- `save_feature_report(...)`: Persist a feature report artifact to `/workspace`.

### `feature_engineering`

Fit and apply deterministic, approved feature-pipeline steps.

- `list_feature_pipeline_steps()`: List the supported pipeline step kinds.
- `fit_feature_pipeline(...)`: Fit a deterministic feature pipeline against a dataframe.
- `transform_with_feature_pipeline(...)`: Apply a fitted pipeline to another dataframe handle.
- `inspect_feature_pipeline(...)`: Inspect the fitted pipeline summary.

### `modeling`

Train, tune, inspect, score, and persist LightGBM models optimized for PPV@5-style workflows.

- `list_lightgbm_tunable_params()`: Show the parameter catalog used by Optuna tuning.
- `train_lightgbm_baseline(...)`: Train a single baseline LightGBM model.
- `score_model_dataframe(...)`: Score a dataframe with a fitted model.
- `summarize_top_p_predictions(...)`: Summarize aggregate prediction quality in the top-p slice.
- `analyze_top_p_false_positives(...)`: Inspect false-positive patterns in the top-ranked slice.
- `tune_lightgbm(...)`: Run Optuna tuning against PPV@5.
- `inspect_hpo_study(...)`: Inspect a stored tuning study.
- `fit_best_lightgbm(...)`: Fit the best model from a stored study.
- `inspect_model(...)`: Inspect a stored model summary.
- `save_model_artifact(...)`: Persist a model artifact to `/workspace`.
- `load_model_artifact(...)`: Reload a saved model artifact.

## PydanticAI `capability.py` Alternative

In addition to the FastMCP server in `server.py`, this package includes an alternate integration in `capability.py` for PydanticAI-based agents.

`MinimalMontyPythonCapability`:

- Implements `AbstractCapability[Any]`.
- Builds a `FunctionToolset` that exposes the same `help`, `execute`, and `results` tools.
- Reuses the same lazily initialized `MinimalMontyPythonREPL` backend as the MCP server.
- Provides a modeling-focused system prompt through `get_instructions()`.
- Optionally wraps the toolset with `LoggingToolset` for Rich console logging.

Use the FastMCP server when you want a standalone MCP endpoint. Use `MinimalMontyPythonCapability` when you want to attach the same minimal Monty tool surface directly to a PydanticAI agent without running the MCP server as a separate process.
