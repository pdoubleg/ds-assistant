"""Shared help content and guidance for the Monty Python REPL."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CollectionWorkflowStep:
    """Curated workflow step for a collection-level help view.

    Attributes:
        title: Human-readable step title.
        tools: Representative tools for the step.
        detail: Optional extra guidance for the step.
    """

    title: str
    tools: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CollectionHelpContent:
    """Curated content used to render collection-specific help text.

    Attributes:
        purpose: Optional override for the collection purpose line.
        when_to_use: Common situations where the collection is the right choice.
        workflow: Typical sequence of tasks for the collection.
        key_concepts: Important terms explained in plain language.
        common_patterns: Reusable working patterns to highlight.
        common_mistakes: Frequent pitfalls to call out.
        next_steps: Suggested follow-up help calls.
    """

    purpose: str | None = None
    when_to_use: tuple[str, ...] = ()
    workflow: tuple[CollectionWorkflowStep, ...] = ()
    key_concepts: dict[str, str] = field(default_factory=dict)
    common_patterns: tuple[str, ...] = ()
    common_mistakes: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()


OVERVIEW_TITLE = "Monty Sandbox Overview"
OVERVIEW_PURPOSE = (
    "Discover and use collections of tools for dataframe operations, feature "
    "engineering, modeling, evaluation, visualization, and file I/O."
)
OVERVIEW_WORKFLOW: tuple[str, ...] = (
    "Call help() to explore collections.",
    'Call help("<collection>") to see available tools.',
    'Call help("<tool>") before writing execute(...) code.',
)
OVERVIEW_KEY_NOTES: tuple[str, ...] = (
    "All paths resolve under: /workspace",
    "Tools operate on dataframe handles, not raw dataframes",
    "Use run_dataframe_code for broader pandas/sklearn/lightgbm/optuna workflows",
    "Add print() statements inside execute(...) or freeform helpers for diagnostics",
    "execute(...) returns stdout, artifacts, errors, and persisted variable names directly",
)
SUPPORTED_NATIVE_IMPORTS: tuple[str, ...] = (
    "dataclasses",
    "datetime",
    "json",
    "math",
    "re",
)
OVERVIEW_LIMITATIONS: tuple[str, ...] = (
    "No class definitions inside execute(...)",
    "Keep all files inside /workspace",
)

DIRECT_EXECUTE_GUIDANCE = (
    "Call this helper directly inside `execute(...)` code, not as a method on a "
    "dataframe or collection object."
)
HANDLE_ARGUMENT_GUIDANCE = (
    "Arguments ending in `_handle` expect a stored handle string returned by an "
    "earlier Monty step."
)
USAGE_EXAMPLE_GUIDANCE = (
    "Start from the usage example, then adapt the variable names and paths to the "
    "current REPL."
)

SANDBOX_OVERVIEW_SENTENCE = (
    "A Monty-sandboxed Python REPL. Use `execute` to run code in a persistent "
    "interpreter-like REPL, `help` to explore collections or inspect a "
    "specific collection/tool by name. Each `execute` call returns stdout, "
    "warnings, errors, artifacts, and persisted variable names directly."
)
WORKSPACE_AND_IMPORTS_SENTENCE = (
    "Keep files in `/workspace`. Native imports inside `execute(...)` are "
    "intentionally limited to a small built-in set, and class definitions are "
    "not supported."
)
FREEFORM_RUNTIME_SENTENCE = (
    "When you need broader dataframe-oriented data science library access over a "
    "stored pandas dataframe, inspect and use the `run_dataframe_code(...)` "
    "freeform helper. For reusable pipeline-safe logic, inspect the "
    "`fit_freeform_transformer(...)` helpers in the same collection."
)
FREEFORM_PATHS_SENTENCE = (
    "Inside freeform code, convert `/workspace/...` paths with "
    "`workspace_path(...)` or `resolve_workspace_path(...)` before passing them "
    "to pandas, joblib, or other host-side file APIs."
)
FREEFORM_STRING_SENTENCE = (
    "When submitting nested code strings, prefer assigning the inner code to a "
    "named multiline variable first, avoid escape-heavy inline strings like "
    "`print(f'\\n...')`, use separate `print()` calls for blank lines or "
    "diagnostics, and only use `\\\\n` when the inner code truly needs a literal "
    "backslash escape to survive outer parsing."
)
MODELING_FLOW_SENTENCE = (
    "Prefer the modeling flow of reusable freeform, then declarative feature "
    "engineering, then preprocessing when that sequence fits the task."
)


def build_shared_runtime_guidance() -> str:
    """Return the shared high-level runtime guidance paragraph.

    Returns:
        str: Consolidated runtime guidance for prompts and tool descriptions.
    """

    return " ".join(
        (
            SANDBOX_OVERVIEW_SENTENCE,
            WORKSPACE_AND_IMPORTS_SENTENCE,
            FREEFORM_RUNTIME_SENTENCE,
            FREEFORM_PATHS_SENTENCE,
            FREEFORM_STRING_SENTENCE,
            MODELING_FLOW_SENTENCE,
        )
    )


def build_help_tool_description() -> str:
    """Return the shared description for the exported help tool.

    Returns:
        str: Help tool description used by server and capability layers.
    """

    return " ".join(
        (
            "Discover registered sandbox helper functions and capability groups.",
            "Call `help()` to explore task-focused collections,",
            "`help('<collection-name>')` to inspect a collection, and",
            "`help('<tool-name>')` right before using an unfamiliar helper in "
            "`execute(...)`.",
            FREEFORM_RUNTIME_SENTENCE,
            FREEFORM_PATHS_SENTENCE,
            FREEFORM_STRING_SENTENCE,
            MODELING_FLOW_SENTENCE,
        )
    )


def build_execute_tool_description() -> str:
    """Return the shared description for the exported execute tool.

    Returns:
        str: Execute tool description used by server and capability layers.
    """

    return " ".join(
        (
            "Run Python code in a persistent interpreter-like REPL.",
            "Keep all files and outputs in `/workspace`.",
            WORKSPACE_AND_IMPORTS_SENTENCE,
            FREEFORM_RUNTIME_SENTENCE,
            "The ad hoc helper still returns a single final dataframe, so print "
            "intermediate diagnostics inside your code when needed.",
            FREEFORM_PATHS_SENTENCE,
            FREEFORM_STRING_SENTENCE,
            "Prefer registered helper functions for common dataframe loading, "
            "EDA, visualization, export, and modeling workflows when useful.",
        )
    )


COLLECTION_HELP_CONTENT: dict[str, CollectionHelpContent] = {
    "data_io": CollectionHelpContent(
        when_to_use=(
            "Loading a training or scoring dataset from CSV or Excel",
            "Saving one or more dataframe handles back to /workspace",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Load source data",
                tools=("load_csv", "load_excel"),
            ),
            CollectionWorkflowStep(
                title="Inspect or transform the resulting dataframe handle",
                tools=("dataframe_head", "dataframe_describe"),
            ),
            CollectionWorkflowStep(
                title="Save curated outputs",
                tools=("save_csv", "save_excel"),
            ),
        ),
        key_concepts={
            "dataframe_handle": "Reference to a stored dataframe produced by a prior Monty step.",
            "workspace_path": "All file inputs and outputs stay under /workspace.",
        },
        common_mistakes=(
            "Passing host file paths instead of /workspace-relative paths",
            "Expecting load_* tools to return raw pandas dataframes instead of handles",
        ),
    ),
    "dataframe": CollectionHelpContent(
        when_to_use=(
            "Checking schema, missingness, or value distributions",
            "Creating filtered or aggregated dataframe handles for downstream work",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Inspect structure and preview rows",
                tools=("dataframe_shape", "dataframe_columns", "dataframe_head"),
            ),
            CollectionWorkflowStep(
                title="Summarize quality or distributions",
                tools=(
                    "dataframe_missing_summary",
                    "dataframe_describe",
                    "value_counts",
                ),
            ),
            CollectionWorkflowStep(
                title="Create a derived dataframe handle",
                tools=("filter_dataframe", "groupby_aggregate"),
            ),
        ),
        key_concepts={
            "derived_dataframe": "A new handle created from filtering, grouping, or other tabular transforms.",
        },
        common_mistakes=(
            "Forgetting that helper outputs are new handles instead of in-place dataframe mutations",
        ),
    ),
    "feature_engineering": CollectionHelpContent(
        when_to_use=(
            "Creating deterministic derived features such as ratios, aggregations, and transformations",
            "Building reusable feature logic to apply consistently across datasets",
            "Persisting fitted feature artifacts for later reuse",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Build a feature spec",
                tools=("build_feature_engineering_spec",),
            ),
            CollectionWorkflowStep(
                title="Fit a feature engineer",
                tools=("fit_feature_engineer",),
            ),
            CollectionWorkflowStep(
                title="Apply transformations",
                tools=(
                    "transform_with_feature_engineer",
                    "fit_transform_with_feature_engineer",
                ),
                detail="Use fit_transform_with_feature_engineer(...) for a single fit + transform step.",
            ),
            CollectionWorkflowStep(
                title="Inspect or debug the artifact",
                tools=("inspect_feature_engineer", "list_engineered_features"),
            ),
            CollectionWorkflowStep(
                title="Persist reusable artifacts",
                tools=("save_feature_engineer", "load_feature_engineer"),
            ),
        ),
        key_concepts={
            "spec": "Declarative definition of engineered features, conflict policy, and output behavior.",
            "feature_engineer_handle": "Reference to a fitted feature engineering artifact stored by Monty.",
            "dataframe_handle": "Reference to a stored dataframe rather than a raw pandas object.",
        },
        common_patterns=(
            "Fit once on training data, then reuse the same feature engineer on validation or test handles.",
            "Use inspect_feature_engineer(...) and list_engineered_features(...) to debug engineered outputs before modeling.",
        ),
        common_mistakes=(
            "Passing raw dataframes instead of dataframe handles",
            "Forgetting that transform outputs are handles, not materialized pandas objects",
            "Re-fitting instead of reusing a saved or previously fitted feature engineer",
            "Skipping include_target=True when downstream steps expect the target column to remain attached",
        ),
        next_steps=(
            'Call help("build_feature_engineering_spec") to define feature specs.',
            'Call help("<tool-name>") before using an unfamiliar feature-engineering helper.',
        ),
    ),
    "feature_selection": CollectionHelpContent(
        when_to_use=(
            "Ranking predictive power, redundancy, or feature importance before model fitting",
            "Packaging feature-screening results for inspection or reuse",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Compute ranking or redundancy metrics",
                tools=(
                    "compute_feature_target_metrics",
                    "compute_feature_redundancy_metrics",
                    "rank_feature_importance_with_lightgbm",
                ),
            ),
            CollectionWorkflowStep(
                title="Summarize or evaluate candidate subsets",
                tools=("summarize_feature_candidates", "evaluate_feature_subset"),
            ),
            CollectionWorkflowStep(
                title="Inspect or persist findings",
                tools=(
                    "inspect_feature_selection_report",
                    "save_feature_selection_report",
                    "load_feature_selection_report",
                ),
            ),
        ),
        common_mistakes=(
            "Treating ranking outputs as final modeling decisions without validating a subset",
        ),
    ),
    "freeform": CollectionHelpContent(
        when_to_use=(
            "You need pandas, numpy, sklearn, LightGBM, Optuna, or joblib logic that is too custom for declarative helpers",
            "You want a reusable sklearn-compatible freeform transformer artifact",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Prototype ad hoc dataframe logic",
                tools=("run_dataframe_code",),
            ),
            CollectionWorkflowStep(
                title="Promote stable logic into a reusable transformer",
                tools=(
                    "fit_freeform_transformer",
                    "transform_with_freeform_transformer",
                    "fit_transform_with_freeform_transformer",
                ),
            ),
            CollectionWorkflowStep(
                title="Inspect or persist reusable artifacts",
                tools=(
                    "inspect_freeform_transformer",
                    "list_freeform_transformer_features",
                    "save_freeform_transformer",
                    "load_freeform_transformer",
                ),
            ),
        ),
        key_concepts={
            "df": "Mutable dataframe exposed inside freeform code.",
            "workspace_path": "Helper that converts /workspace paths into host paths for pandas, joblib, and similar libraries.",
            "freeform_transformer_handle": "Reference to a fitted reusable freeform transformer artifact.",
        },
        common_mistakes=(
            "Passing raw '/workspace/...' strings to pandas or joblib instead of converting them first",
            "Embedding deeply nested escaped strings inside freeform source instead of using a named multiline variable",
            "Expecting run_dataframe_code(...) to keep intermediate dataframes instead of returning one final dataframe handle",
        ),
    ),
    "handles": CollectionHelpContent(
        when_to_use=(
            "Listing active dataframe or figure handles in the current REPL",
            "Inspecting what a previously returned handle refers to",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="List stored handles",
                tools=("list_object_handles",),
            ),
            CollectionWorkflowStep(
                title="Inspect a specific handle",
                tools=("inspect_handle",),
            ),
        ),
        common_mistakes=(
            "Guessing handle contents instead of inspecting them before a downstream step",
        ),
    ),
    "hpo": CollectionHelpContent(
        when_to_use=(
            "Running Optuna-guided search over pipeline or estimator configurations",
            "Inspecting, exporting, or persisting tuned pipeline artifacts",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Build or inspect a tuning configuration",
                tools=("build_hpo_config", "inspect_pipeline_tunable_params"),
            ),
            CollectionWorkflowStep(
                title="Create a study and run optimization steps",
                tools=("create_hpo_study", "run_hpo_iteration"),
            ),
            CollectionWorkflowStep(
                title="Inspect outcomes and persist artifacts",
                tools=(
                    "inspect_hpo_best_config",
                    "summarize_hpo_study",
                    "save_tuned_pipeline",
                ),
            ),
        ),
        common_mistakes=(
            "Skipping inspection of the pipeline/search-space schema before starting a study",
            "Treating one best trial as final without reviewing study-level diagnostics",
        ),
    ),
    "metrics": CollectionHelpContent(
        when_to_use=(
            "Creating reusable scorer definitions",
            "Evaluating predictions, subsets, or tuned pipelines with explicit metrics",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Build a scorer configuration",
                tools=("create_metric_scorer", "create_ppv_scorer"),
            ),
            CollectionWorkflowStep(
                title="Evaluate predictions or pipelines",
                tools=("evaluate_predictions", "evaluate_tuned_pipeline"),
            ),
        ),
        common_mistakes=(
            "Comparing model runs without holding the metric definitions constant",
        ),
    ),
    "preprocessing": CollectionHelpContent(
        when_to_use=(
            "Building reusable sklearn preprocessing pipelines for numeric and categorical features",
            "Transforming train and inference datasets with the same fitted preprocessing artifact",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Build a preprocessing spec",
                tools=("build_preprocessing_spec",),
            ),
            CollectionWorkflowStep(
                title="Fit a preprocessor",
                tools=("fit_preprocessor",),
            ),
            CollectionWorkflowStep(
                title="Transform data",
                tools=("transform_dataframe", "fit_transform_dataframe"),
                detail="Use fit_transform_dataframe(...) when you want a one-step fit + transform on the same training handle.",
            ),
            CollectionWorkflowStep(
                title="Inspect or persist reusable artifacts",
                tools=(
                    "inspect_preprocessor",
                    "list_preprocessor_features",
                    "save_preprocessor",
                    "load_preprocessor",
                ),
            ),
        ),
        key_concepts={
            "spec": "Declarative preprocessing definition covering groups, steps, remainder behavior, and output settings.",
            "preprocessor_handle": "Reference to a fitted preprocessing artifact stored by Monty.",
            "include_target": "Whether to append the target column back to the transformed dataframe output.",
        },
        common_patterns=(
            "Fit on the training dataframe handle, then reuse the same preprocessor on validation and test handles.",
        ),
        common_mistakes=(
            "Passing raw dataframes instead of handles",
            "Forgetting that fit_transform_dataframe(...) returns handles instead of concrete pandas objects",
            "Assigning the same column to multiple preprocessing groups",
        ),
    ),
    "splitting": CollectionHelpContent(
        when_to_use=(
            "Creating holdout datasets or reusable CV splitter artifacts",
            "Persisting train/validation/test splits for later modeling stages",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Create a splitter or holdout split",
                tools=(
                    "create_kfold_splitter",
                    "create_stratified_kfold_splitter",
                    "make_holdout_split",
                    "train_validation_test_split",
                ),
            ),
            CollectionWorkflowStep(
                title="Inspect or persist the resulting artifact",
                tools=("inspect_splitter", "inspect_data_split", "save_data_split"),
            ),
        ),
        common_mistakes=(
            "Using an unstratified split for imbalanced classification tasks when a stratified option is available",
        ),
    ),
    "visualizations": CollectionHelpContent(
        when_to_use=(
            "Creating quick exploratory plots from stored dataframe handles",
            "Persisting Plotly or matplotlib outputs for reports",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="Create a figure from a dataframe handle",
                tools=("create_histogram", "create_scatter_plot", "create_bar_chart"),
            ),
            CollectionWorkflowStep(
                title="Inspect or save the resulting figure",
                tools=(
                    "inspect_handle",
                    "save_plotly_figure",
                    "save_matplotlib_figure",
                ),
            ),
        ),
        common_mistakes=(
            "Assuming figure helpers return inline images instead of figure handles or saved files",
        ),
    ),
    "workspace": CollectionHelpContent(
        when_to_use=(
            "Reading or writing text and JSON files inside /workspace",
            "Listing what workspace files already exist before a downstream step",
        ),
        workflow=(
            CollectionWorkflowStep(
                title="List or inspect files",
                tools=(
                    "list_workspace_files",
                    "read_workspace_text",
                    "read_workspace_json",
                ),
            ),
            CollectionWorkflowStep(
                title="Write or update workspace files",
                tools=("write_workspace_text", "write_workspace_json"),
            ),
        ),
        key_concepts={
            "workspace": "Sandboxed project directory where Monty is allowed to read and write files.",
        },
        common_mistakes=(
            "Trying to write unsupported file types with text helpers",
            "Using raw open(...) or Path.write_text(...) inside execute(...) when a workspace helper is clearer",
        ),
    ),
}


def get_collection_help_content(name: str) -> CollectionHelpContent:
    """Return curated collection help content when available.

    Args:
        name: Registered collection name.

    Returns:
        CollectionHelpContent: Curated content or an empty default entry.
    """

    return COLLECTION_HELP_CONTENT.get(name, CollectionHelpContent())
