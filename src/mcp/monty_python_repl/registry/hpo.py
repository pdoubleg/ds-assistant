"""Structured-spec HPO helpers for the Monty Python REPL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from pprint import pformat
from textwrap import indent
from typing import Any
import pandas as pd
import plotly.express as px
import joblib
import optuna

from ..filesystem import HostWorkspaceOSAccess
from ..core.registry import ObjectStore, ToolCollection, tool
from .feature_engineering import StoredFeatureEngineer
from .freeform import StoredFreeformTransformer
from ..support.hpo_utils import (
    apply_search_space_to_config,
    build_trial_record,
    build_hpo_config_bundle,
    evaluate_pipeline_candidate,
    fit_final_model,
    inspect_pipeline_params,
    materialize_pipeline_data,
    summarize_study_trials,
)
from .preprocessing import StoredPreprocessor
from ..core.registry import safe_json_value


@dataclass(slots=True)
class StoredHpoStudy:
    """Persisted HPO study state stored behind a Monty-safe handle.

    Args:
        study_name (str): Human-readable study name.
        pipeline_config (dict[str, Any]): Normalized pipeline config.
        search_space (list[dict[str, Any]]): Normalized search-space entries.
        evaluation_mode (str): Validation or cross-validation tuning mode.
        objective_metric (str): Metric optimized by completed trials.
        warnings (list[str]): High-level study warnings.
        trials (list[dict[str, Any]]): Completed or failed trial records.
        run_history (list[dict[str, Any]]): Per-iteration execution chunks.
        best_trial_number (int | None): Trial number of the current best trial.
        best_value (float | None): Best objective value seen so far.
        best_config (dict[str, Any] | None): Best fully resolved candidate config.
        best_metrics (dict[str, Any]): Best-trial evaluation summary.
        optuna_study (optuna.Study | None): In-memory Optuna study object.
    """

    study_name: str
    pipeline_config: dict[str, Any]
    search_space: list[dict[str, Any]]
    evaluation_mode: str
    objective_metric: str = "auto"
    warnings: list[str] = field(default_factory=list)
    trials: list[dict[str, Any]] = field(default_factory=list)
    run_history: list[dict[str, Any]] = field(default_factory=list)
    best_trial_number: int | None = None
    best_value: float | None = None
    best_config: dict[str, Any] | None = None
    best_metrics: dict[str, Any] = field(default_factory=dict)
    optuna_study: optuna.Study | None = field(default=None, repr=False)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection.

        Args:
            max_items (int): Maximum trial or search-space items to preview.
            max_chars (int): Maximum nested string length to retain.

        Returns:
            dict[str, Any]: Summary payload suitable for `inspect_handle`.
        """
        return {
            "type": "StoredHpoStudy",
            "study_name": self.study_name,
            "evaluation_mode": self.evaluation_mode,
            "objective_metric": self.objective_metric,
            "trial_count": len(self.trials),
            "failed_trial_count": sum(
                1 for trial in self.trials if trial.get("status") == "fail"
            ),
            "best_trial_number": self.best_trial_number,
            "best_value": self.best_value,
            "warnings": self.warnings[:max_items],
            "search_space": safe_json_value(
                self.search_space[:max_items],
                max_items=max_items,
                max_chars=max_chars,
            ),
            "best_config": safe_json_value(
                self.best_config,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "best_metrics": safe_json_value(
                self.best_metrics,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "top_trials": safe_json_value(
                summarize_study_trials(self.trials, top_n=min(max_items, 5)).get(
                    "top_trials", []
                ),
                max_items=max_items,
                max_chars=max_chars,
            ),
            "recent_failures": safe_json_value(
                [
                    {
                        "trial_number": trial.get("trial_number"),
                        "failure_reason": trial.get("failure_reason"),
                    }
                    for trial in self.trials
                    if trial.get("status") == "fail"
                ][:max_items],
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


@dataclass(slots=True)
class StoredTunedPipeline:
    """Persisted best-fit pipeline and LightGBM model artifact.

    Args:
        pipeline_config (dict[str, Any]): Fully resolved best pipeline config.
        fitted_model (Any): Trained LightGBM estimator.
        model_feature_columns (list[str]): Final model feature columns used at fit time.
        selected_features (list[str]): Selected features before model-ready coercion.
        evaluation_summary (dict[str, Any]): Stored evaluation summary for the best config.
        preprocessor (StoredPreprocessor | None): Fitted preprocessing artifact, if any.
        feature_engineer (StoredFeatureEngineer | None): Fitted FE artifact, if any.
        freeform_transformer (StoredFreeformTransformer | None): Fitted reusable
            freeform transformer, if any.
    """

    pipeline_config: dict[str, Any]
    fitted_model: Any
    model_feature_columns: list[str]
    selected_features: list[str]
    evaluation_summary: dict[str, Any]
    preprocessor: StoredPreprocessor | None = None
    feature_engineer: StoredFeatureEngineer | None = None
    freeform_transformer: StoredFreeformTransformer | None = None

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection.

        Args:
            max_items (int): Maximum feature names to preview.
            max_chars (int): Maximum nested string length to retain.

        Returns:
            dict[str, Any]: Summary payload suitable for `inspect_handle`.
        """
        return {
            "type": "StoredTunedPipeline",
            "model_class": type(self.fitted_model).__name__,
            "model_feature_columns": self.model_feature_columns[:max_items],
            "selected_features": self.selected_features[:max_items],
            "has_preprocessor": self.preprocessor is not None,
            "has_feature_engineer": self.feature_engineer is not None,
            "has_freeform_transformer": self.freeform_transformer is not None,
            "evaluation_summary": safe_json_value(
                self.evaluation_summary,
                max_items=max_items,
                max_chars=max_chars,
            ),
            "pipeline_config": safe_json_value(
                self.pipeline_config,
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


class HpoCollection(ToolCollection):
    """Structured Optuna-based HPO helpers for the full Monty pipeline."""

    name = "hpo"
    description = (
        "Inspect tunable pipeline params, create structured Optuna studies, "
        "run iterative tuning rounds, inspect best configs, and save tuned "
        "LightGBM pipeline artifacts plus reports and reproducible exports."
    )

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: ObjectStore,
    ) -> None:
        """Initialize HPO helpers.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (ObjectStore): Shared handle store.
        """
        self._os_access = os_access
        self._object_store = object_store

    def _resolve_host_path(self, path: str) -> Path:
        """Resolve a virtual or relative path into the host workspace.

        Args:
            path (str): Relative or `/workspace` path.

        Returns:
            Path: Resolved host path.
        """
        return self._os_access.to_host_path(PurePosixPath(path))

    def _get_study(self, study_handle: str) -> StoredHpoStudy:
        """Fetch a stored HPO study from the object store.

        Args:
            study_handle (str): Study handle.

        Returns:
            StoredHpoStudy: Stored study artifact.
        """
        return self._object_store.get(study_handle, expected_type=StoredHpoStudy)

    def _get_tuned_pipeline(self, tuned_handle: str) -> StoredTunedPipeline:
        """Fetch a stored tuned pipeline artifact from the object store.

        Args:
            tuned_handle (str): Tuned pipeline handle.

        Returns:
            StoredTunedPipeline: Stored tuned artifact.
        """
        return self._object_store.get(
            tuned_handle,
            expected_type=StoredTunedPipeline,
        )

    def _record_artifact(self, host_path: Path) -> None:
        """Record a host-side artifact for execution result reporting."""
        self._os_access.record_host_artifact(host_path)

    def _write_text_artifact(self, path: str, content: str) -> str:
        """Write a UTF-8 text artifact inside the workspace.

        Args:
            path (str): Relative or `/workspace` destination path.
            content (str): File content to persist.

        Returns:
            str: Virtual path to the saved artifact.
        """
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_text(content, encoding="utf-8")
        self._record_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    def _write_json_artifact(self, path: str, payload: dict[str, Any]) -> str:
        """Write a JSON artifact inside the workspace."""
        return self._write_text_artifact(
            path,
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        )

    def _render_markdown_report(
        self,
        *,
        title: str,
        summary_items: list[tuple[str, Any]],
        warnings: list[str],
        top_trials: list[dict[str, Any]] | None = None,
        parameter_importances: dict[str, float] | None = None,
    ) -> str:
        """Render a compact Markdown report for users."""
        lines = [f"# {title}", ""]
        if summary_items:
            lines.extend(["## Summary", ""])
            for label, value in summary_items:
                lines.append(f"- **{label}**: {value}")
            lines.append("")
        if warnings:
            lines.extend(["## Warnings", ""])
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")
        if top_trials:
            lines.extend(
                [
                    "## Top Trials",
                    "",
                    "| Trial | Status | Objective | Metric |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for trial in top_trials:
                lines.append(
                    "| {trial_number} | {status} | {objective_value} | {objective_metric} |".format(
                        trial_number=trial.get("trial_number"),
                        status=trial.get("status"),
                        objective_value=trial.get("objective_value"),
                        objective_metric=trial.get("objective_metric"),
                    )
                )
            lines.append("")
        if parameter_importances:
            lines.extend(["## Parameter Importances", ""])
            for name, value in parameter_importances.items():
                lines.append(f"- `{name}`: {value}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _build_pipeline_python_export(
        self,
        *,
        study: StoredHpoStudy,
        tuned_artifact: StoredTunedPipeline | None = None,
    ) -> str:
        """Generate a reproducible Python export for the best pipeline."""
        if study.best_config is None:
            raise ValueError("No completed trial is available yet for this study.")

        embedded_config = pformat(study.best_config, sort_dicts=True)
        embedded_metrics = pformat(
            safe_json_value(study.best_metrics, max_items=50, max_chars=1000),
            sort_dicts=True,
        )
        feature_columns = (
            pformat(tuned_artifact.model_feature_columns, sort_dicts=True)
            if tuned_artifact is not None
            else "[]"
        )
        selected_features = (
            pformat(tuned_artifact.selected_features, sort_dicts=True)
            if tuned_artifact is not None
            else "[]"
        )
        config_block = indent(embedded_config, "    ")
        metrics_block = indent(embedded_metrics, "    ")
        feature_columns_block = indent(feature_columns, "    ")
        selected_features_block = indent(selected_features, "    ")
        return "\n".join(
            [
                '"""Reproducible export of Monty\'s best tuned pipeline.',
                "",
                "This script captures the normalized pipeline configuration that won the",
                "tuning study and provides helper functions you can adapt into a fuller",
                "training or batch-scoring workflow.",
                '"""',
                "",
                "from __future__ import annotations",
                "",
                "import json",
                "from pathlib import Path",
                "from typing import Any",
                "",
                "BEST_PIPELINE_CONFIG: dict[str, Any] = (",
                config_block,
                ")",
                "BEST_EVALUATION_SUMMARY: dict[str, Any] = (",
                metrics_block,
                ")",
                "MODEL_FEATURE_COLUMNS: list[str] = (",
                feature_columns_block,
                ")",
                "SELECTED_FEATURES: list[str] = (",
                selected_features_block,
                ")",
                "",
                "",
                "def get_best_pipeline_config() -> dict[str, Any]:",
                '    """Return the normalized best pipeline configuration."""',
                "    return json.loads(json.dumps(BEST_PIPELINE_CONFIG))",
                "",
                "",
                "def get_best_evaluation_summary() -> dict[str, Any]:",
                '    """Return the stored evaluation summary for the best configuration."""',
                "    return json.loads(json.dumps(BEST_EVALUATION_SUMMARY))",
                "",
                "",
                "def save_pipeline_config(path: str | Path) -> Path:",
                '    """Write the embedded pipeline config to a JSON file."""',
                "    output_path = Path(path)",
                "    output_path.parent.mkdir(parents=True, exist_ok=True)",
                "    output_path.write_text(",
                '        json.dumps(BEST_PIPELINE_CONFIG, indent=2, sort_keys=True) + "\\n",',
                '        encoding="utf-8",',
                "    )",
                "    return output_path",
                "",
                "",
                'if __name__ == "__main__":',
                '    output_dir = Path("monty_pipeline_export")',
                "    output_dir.mkdir(parents=True, exist_ok=True)",
                '    config_path = save_pipeline_config(output_dir / "best_pipeline_config.json")',
                '    print("Saved best pipeline config to:", config_path)',
                '    print("Best evaluation summary:")',
                "    print(json.dumps(BEST_EVALUATION_SUMMARY, indent=2, sort_keys=True))",
                "",
            ]
        )

    @tool
    def build_hpo_config(
        self,
        pipeline_config: dict[str, Any],
        search_space: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build and validate a normalized HPO config bundle.

        This is the safest entry point when a caller is still learning the
        schema. The return payload includes both the normalized config that the
        study will use and a `schema_reference` section with canonical examples,
        stage-specific shapes, and search-space path guidance.

        Args:
            pipeline_config (dict[str, Any]): Structured pipeline config with top-level
                `data`, `model`, and `evaluation` sections plus optional
                `freeform`, `feature_engineering`, and `preprocessing`
                sections, which execute in that fixed order before model fitting.
            search_space (list[dict[str, Any]] | None): Structured search-space
                entries targeting normalized dotted paths such as
                `model.base_params.num_leaves` or
                `preprocessing.spec.groups.0.steps.0.strategy`.

        Returns:
            dict[str, Any]: Normalized bundle with pipeline config, search space,
            and schema reference guidance.

        Examples:
            bundle = build_hpo_config(pipeline_config, search_space)
            print(bundle["pipeline_config"]["preprocessing"]["spec"]["groups"][0]["steps"])
            # Returns:
            # {
            #     "pipeline_config": {"data": {...}, "model": {...}, "evaluation": {...}},
            #     "search_space": [{"path": "model.base_params.num_leaves", "kind": "int"}],
            #     "schema_reference": {"path_guidance": [...], "example_search_space": [...]}
            # }
        """
        bundle = build_hpo_config_bundle(
            pipeline_config,
            search_space,
            object_store=self._object_store,
            os_access=self._os_access,
        )
        return {
            **bundle,
            "schema_reference": inspect_pipeline_params(
                pipeline_config,
                search_space,
                object_store=self._object_store,
                os_access=self._os_access,
            )["schema_reference"],
        }

    @tool
    def inspect_pipeline_tunable_params(
        self,
        pipeline_config: dict[str, Any],
        search_space: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Render current pipeline params and tunable search-space metadata.

        Use this before creating or refining a study. It flattens the normalized
        pipeline config into exact dotted paths, marks which paths are currently
        tunable, and returns both a `schema_reference` block for input authoring
        and a `return_schema` block that makes the inspection payload shape
        explicit to callers.

        Args:
            pipeline_config (dict[str, Any]): Structured pipeline config.
            search_space (list[dict[str, Any]] | None): Optional search-space
                entries. If present, each entry must include a normalized `path`
                and a supported Optuna-style `kind`.

        Returns:
            dict[str, Any]: Inspection payload whose `pipeline_params` field is
            an ordered `list[dict[str, Any]]`, plus a path-keyed
            `pipeline_params_by_path` wrapper, normalized search-space metadata,
            recommended LightGBM params, and a `return_schema` field with the
            explicit output contract.

        Examples:
            inspection = inspect_pipeline_tunable_params(pipeline_config, search_space)
            tunable_paths = [row["path"] for row in inspection["pipeline_params"] if row["is_tunable"]]
            num_leaves_row = inspection["pipeline_params_by_path"]["model.base_params.num_leaves"]
            # Returns:
            # {
            #     "pipeline_params": [{"path": "model.base_params.num_leaves", "is_tunable": True}],
            #     "pipeline_params_by_path": {"model.base_params.num_leaves": {...}},
            #     "return_schema": {"top_level_fields": {"pipeline_params": "list[dict[str, Any]]"}},
            #     "schema_reference": {"example_search_space": [...]}
            # }
        """
        return inspect_pipeline_params(
            pipeline_config,
            search_space,
            object_store=self._object_store,
            os_access=self._os_access,
        )

    @tool
    def inspect_hpo_config(self, study_handle: str) -> dict[str, Any]:
        """Inspect the normalized config stored in a persisted HPO study.

        Args:
            study_handle (str): HPO study handle.

        Returns:
            dict[str, Any]: Stored pipeline config, search-space metadata, and
            the same schema-reference guidance exposed during pre-study
            inspection.

        Examples:
            print(inspect_hpo_config(study_handle))
            # Returns:
            # {
            #     "pipeline_params": [...],
            #     "normalized_search_space": [...],
            #     "schema_reference": {...}
            # }
        """
        study = self._get_study(study_handle)
        return inspect_pipeline_params(
            study.pipeline_config,
            study.search_space,
            object_store=self._object_store,
            os_access=self._os_access,
        )

    @tool
    def create_hpo_study(
        self,
        pipeline_config: dict[str, Any],
        search_space: list[dict[str, Any]],
        *,
        study_name: str | None = None,
    ) -> str:
        """Create a persisted Optuna study from structured pipeline inputs.

        The `pipeline_config` and `search_space` must already target the
        normalized schema. Call `inspect_pipeline_tunable_params(...)` first if
        you need exact dotted paths or example stage shapes.

        Args:
            pipeline_config (dict[str, Any]): Structured pipeline config.
            search_space (list[dict[str, Any]]): Structured search-space entries.
            study_name (str | None): Optional study name.

        Returns:
            str: Handle for the persisted HPO study.

        Examples:
            study_handle = create_hpo_study(
                pipeline_config,
                search_space,
                study_name="lgbm_search",
            )
        """
        bundle = build_hpo_config_bundle(
            pipeline_config,
            search_space,
            object_store=self._object_store,
            os_access=self._os_access,
        )
        normalized_pipeline = bundle["pipeline_config"]
        normalized_search_space = bundle["search_space"]
        random_state = int(normalized_pipeline["evaluation"]["random_state"])
        normalized_study_name = (
            study_name or f"hpo_{len(self._object_store.list_handles()) + 1}"
        )

        warnings: list[str] = []
        evaluation_mode = normalized_pipeline["evaluation"]["mode"]
        if evaluation_mode in ("cv", "cross_validation"):
            warnings.append(
                "This study is configured for cross-validation tuning. Repeated adaptive search on the same dataframe can produce optimistic results."
            )
        elif normalized_pipeline["data"]["validation_handle"] is None:
            warnings.append(
                "Validation mode was requested without a validation handle. Cross-validation will be used instead."
            )
            normalized_pipeline["evaluation"]["mode"] = "cross_validation"

        study = optuna.create_study(
            direction="maximize",
            study_name=normalized_study_name,
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        artifact = StoredHpoStudy(
            study_name=normalized_study_name,
            pipeline_config=normalized_pipeline,
            search_space=normalized_search_space,
            evaluation_mode=str(normalized_pipeline["evaluation"]["mode"]),
            objective_metric=str(normalized_pipeline["evaluation"]["metric"]),
            warnings=warnings,
            optuna_study=study,
        )
        return self._object_store.put(artifact, prefix="hpo")

    @tool
    def run_hpo_iteration(
        self,
        study_handle: str,
        n_trials: int,
        *,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Run one iterative block of Optuna trials for a persisted study.

        Args:
            study_handle (str): HPO study handle.
            n_trials (int): Number of new trials to evaluate.
            top_n (int): Maximum number of top trials to include in the summary.

        Returns:
            dict[str, Any]: Summary of newly added trials and current study bests.

        Examples:
            print(run_hpo_iteration(study_handle, n_trials=10, top_n=5))
            # Returns:
            # {
            #     "study_handle": "hpo_1",
            #     "added_trials": 10,
            #     "completed_trial_count": 8,
            #     "failed_trial_count": 2,
            #     "best_value": 0.8421,
            #     "top_trials": [...],
            #     "recent_failures": [...]
            # }
        """
        study = self._get_study(study_handle)
        if study.optuna_study is None:
            raise ValueError(
                "The HPO study is missing its in-memory Optuna study state."
            )

        previous_trial_count = len(study.optuna_study.trials)

        def objective(trial: optuna.trial.Trial) -> float:
            try:
                candidate_config, sampled_params = apply_search_space_to_config(
                    trial,
                    study.pipeline_config,
                    study.search_space,
                )
                evaluation = evaluate_pipeline_candidate(
                    candidate_config,
                    object_store=self._object_store,
                    os_access=self._os_access,
                )
                trial.set_user_attr("sampled_params", sampled_params)
                trial.set_user_attr("candidate_config", candidate_config)
                trial.set_user_attr("objective_metric", evaluation["objective_metric"])
                trial.set_user_attr(
                    "evaluation_summary", evaluation["evaluation_summary"]
                )
                trial.set_user_attr(
                    "selected_features", evaluation["selected_features"]
                )
                trial.set_user_attr("feature_count", evaluation["feature_count"])
                trial.set_user_attr("warnings", evaluation["warnings"])
                return float(evaluation["objective_value"])
            except Exception as exc:
                trial.set_user_attr(
                    "failure_reason",
                    f"{type(exc).__name__}: {exc}",
                )
                raise

        study.optuna_study.optimize(
            objective, n_trials=int(n_trials), catch=(Exception,)
        )

        for frozen_trial in study.optuna_study.trials[previous_trial_count:]:
            trial_record = build_trial_record(frozen_trial)
            trial_record["candidate_config"] = safe_json_value(
                frozen_trial.user_attrs.get("candidate_config")
            )
            trial_record["failure_reason"] = frozen_trial.user_attrs.get(
                "failure_reason"
            ) or frozen_trial.system_attrs.get("fail_reason")
            study.trials.append(trial_record)

        completed_trials = [
            trial for trial in study.optuna_study.trials if trial.value is not None
        ]
        if completed_trials:
            best_trial = study.optuna_study.best_trial
            study.best_trial_number = int(best_trial.number)
            study.best_value = (
                float(best_trial.value) if best_trial.value is not None else None
            )
            study.best_config = best_trial.user_attrs.get("candidate_config")
            study.best_metrics = best_trial.user_attrs.get("evaluation_summary", {})
            study.objective_metric = str(
                best_trial.user_attrs.get("objective_metric", study.objective_metric)
            )

        run_summary = summarize_study_trials(study.trials, top_n=top_n)
        study.run_history.append(
            {
                "added_trials": int(
                    len(study.optuna_study.trials) - previous_trial_count
                ),
                "completed_trial_count": int(run_summary["completed_trial_count"]),
                "failed_trial_count": int(run_summary["failed_trial_count"]),
                "best_value": study.best_value,
            }
        )
        return {
            "study_handle": study_handle,
            "added_trials": len(study.optuna_study.trials) - previous_trial_count,
            "completed_trial_count": run_summary["completed_trial_count"],
            "failed_trial_count": run_summary["failed_trial_count"],
            "best_trial_number": study.best_trial_number,
            "best_value": study.best_value,
            "objective_metric": study.objective_metric,
            "top_trials": run_summary["top_trials"],
            "recent_failures": run_summary["recent_failures"],
            "warnings": study.warnings,
        }

    @tool
    def list_hpo_trials(
        self, study_handle: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List stored HPO trial summaries for a persisted study.

        Args:
            study_handle (str): HPO study handle.
            limit (int): Maximum number of most recent trials to return.

        Returns:
            list[dict[str, Any]]: Trial summaries in descending trial-number order.

        Examples:
            print(list_hpo_trials(study_handle, limit=10))
        """
        study = self._get_study(study_handle)
        return list(reversed(study.trials[-limit:]))

    @tool
    def inspect_hpo_best_config(self, study_handle: str) -> dict[str, Any]:
        """Return the current best configuration for a persisted HPO study.

        Args:
            study_handle (str): HPO study handle.

        Returns:
            dict[str, Any]: Best-config payload grouped by pipeline stage.

        Examples:
            print(inspect_hpo_best_config(study_handle))
            # Returns:
            # {
            #     "study_name": "lgbm_search",
            #     "best_trial_number": 7,
            #     "best_value": 0.8421,
            #     "objective_metric": "roc_auc",
            #     "best_config": {"model": {...}, "evaluation": {...}},
            #     "best_metrics": {"roc_auc": 0.8421}
            # }
        """
        study = self._get_study(study_handle)
        if study.best_config is None:
            raise ValueError("No completed trial is available yet for this study.")

        return {
            "study_name": study.study_name,
            "best_trial_number": study.best_trial_number,
            "best_value": study.best_value,
            "objective_metric": study.objective_metric,
            "best_config": safe_json_value(study.best_config),
            "best_metrics": safe_json_value(study.best_metrics),
        }

    @tool
    def summarize_hpo_study(self, study_handle: str) -> dict[str, Any]:
        """Summarize the top trials and parameter importances for a study.

        Args:
            study_handle (str): HPO study handle.

        Returns:
            dict[str, Any]: Study summary payload.

        Examples:
            print(summarize_hpo_study(study_handle))
            # Returns:
            # {
            #     "study_name": "lgbm_search",
            #     "trial_count": 25,
            #     "best_value": 0.8421,
            #     "top_trials": [...],
            #     "parameter_importances": {"model.base_params.num_leaves": 0.34},
            #     "run_history": [...]
            # }
        """
        study = self._get_study(study_handle)
        summary = summarize_study_trials(study.trials, top_n=5)
        parameter_importances: dict[str, float] = {}
        if study.optuna_study is not None and len(study.optuna_study.trials) >= 2:
            try:
                parameter_importances = {
                    str(key): float(value)
                    for key, value in optuna.importance.get_param_importances(
                        study.optuna_study
                    ).items()
                }
            except Exception:
                parameter_importances = {}

        return {
            "study_name": study.study_name,
            "trial_count": len(study.trials),
            "objective_metric": study.objective_metric,
            "best_trial_number": study.best_trial_number,
            "best_value": study.best_value,
            "completed_trial_count": summary["completed_trial_count"],
            "failed_trial_count": summary["failed_trial_count"],
            "top_trials": summary["top_trials"],
            "recent_failures": summary["recent_failures"],
            "parameter_importances": parameter_importances,
            "run_history": safe_json_value(study.run_history),
            "warnings": study.warnings,
        }

    @tool
    def save_tuned_pipeline(self, study_handle: str, path: str) -> dict[str, Any]:
        """Fit and persist the current best tuned pipeline artifact.

        Args:
            study_handle (str): HPO study handle.
            path (str): Relative or `/workspace` destination path.

        Returns:
            dict[str, Any]: Saved path and tuned artifact handle.

        Examples:
            print(save_tuned_pipeline(study_handle, "/workspace/output/tuned_pipeline.joblib"))
            # Returns:
            # {
            #     "tuned_handle": "tuned_1",
            #     "path": "/workspace/output/tuned_pipeline.joblib"
            # }
        """
        study = self._get_study(study_handle)
        if study.best_config is None:
            raise ValueError("No completed trial is available yet for this study.")

        artifacts = materialize_pipeline_data(
            study.best_config,
            object_store=self._object_store,
            os_access=self._os_access,
        )
        evaluation_random_state = int(study.best_config["evaluation"]["random_state"])
        model_params = dict(study.best_config["model"]["base_params"])
        fitted_model, model_feature_columns = fit_final_model(
            train_features=artifacts.train_features,
            train_target=artifacts.train_target,
            model_params=model_params,
            random_state=evaluation_random_state,
        )
        tuned_artifact = StoredTunedPipeline(
            pipeline_config=study.best_config,
            fitted_model=fitted_model,
            model_feature_columns=model_feature_columns,
            selected_features=list(artifacts.selected_features),
            evaluation_summary=study.best_metrics,
            preprocessor=artifacts.preprocessor,
            feature_engineer=artifacts.feature_engineer,
            freeform_transformer=artifacts.freeform_transformer,
        )
        tuned_handle = self._object_store.put(tuned_artifact, prefix="tuned")

        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(tuned_artifact, host_path)
        self._record_artifact(host_path)
        return {
            "tuned_handle": tuned_handle,
            "path": str(self._os_access.virtualize_host_path(host_path)),
        }

    @tool
    def load_tuned_pipeline(self, path: str) -> str:
        """Load a previously saved tuned pipeline artifact from the workspace.

        Args:
            path (str): Relative or `/workspace` path to a saved tuned artifact.

        Returns:
            str: Handle for the loaded tuned pipeline artifact.

        Examples:
            tuned_handle = load_tuned_pipeline("/workspace/output/tuned_pipeline.joblib")
        """
        artifact = joblib.load(self._resolve_host_path(path))
        if not isinstance(artifact, StoredTunedPipeline):
            raise TypeError("Loaded artifact is not a StoredTunedPipeline.")
        return self._object_store.put(artifact, prefix="tuned")

    @tool
    def inspect_tuned_pipeline(self, tuned_handle: str) -> dict[str, Any]:
        """Return a JSON-friendly summary of a stored tuned pipeline artifact.

        Args:
            tuned_handle (str): Tuned pipeline handle.

        Returns:
            dict[str, Any]: Tuned pipeline summary.

        Examples:
            print(inspect_tuned_pipeline(tuned_handle))
            # Returns:
            # {
            #     "model_class": "LGBMClassifier",
            #     "model_feature_columns": ["age", "income", "premium_ratio"],
            #     "selected_features": ["age", "income", "premium_ratio"],
            #     "evaluation_summary": {"roc_auc": 0.8421}
            # }
        """
        return self._get_tuned_pipeline(tuned_handle).to_json_summary()

    @tool
    def save_hpo_study_report(self, study_handle: str, path: str) -> dict[str, Any]:
        """Save a user-facing HPO study report as JSON or Markdown.

        Args:
            study_handle (str): HPO study handle.
            path (str): Relative or `/workspace` destination ending in `.json` or `.md`.

        Returns:
            dict[str, Any]: Saved path and report format.

        Examples:
            print(save_hpo_study_report(study_handle, "/workspace/output/hpo_report.md"))
            # Returns:
            # {
            #     "path": "/workspace/output/hpo_report.md",
            #     "format": "md"
            # }
        """
        summary = self.summarize_hpo_study(study_handle)
        suffix = self._resolve_host_path(path).suffix.lower()
        if suffix == ".json":
            saved_path = self._write_json_artifact(path, summary)
        else:
            markdown = self._render_markdown_report(
                title=f"HPO Study Report: {summary['study_name']}",
                summary_items=[
                    ("Trial count", summary["trial_count"]),
                    ("Completed trial count", summary["completed_trial_count"]),
                    ("Failed trial count", summary["failed_trial_count"]),
                    ("Best trial number", summary["best_trial_number"]),
                    ("Best value", summary["best_value"]),
                    ("Objective metric", summary["objective_metric"]),
                ],
                warnings=list(summary["warnings"]),
                top_trials=list(summary["top_trials"]),
                parameter_importances=dict(summary["parameter_importances"]),
            )
            saved_path = self._write_text_artifact(path, markdown)
        return {"path": saved_path, "format": suffix.lstrip(".") or "md"}

    @tool
    def save_hpo_trials_table(self, study_handle: str, path: str) -> str:
        """Save all stored HPO trial records as a CSV table.

        Args:
            study_handle (str): HPO study handle.
            path (str): Relative or `/workspace` destination path.

        Returns:
            str: Virtual path to the saved CSV file.

        Examples:
            print(save_hpo_trials_table(study_handle, "/workspace/output/hpo_trials.csv"))
        """
        study = self._get_study(study_handle)
        host_path = self._resolve_host_path(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(study.trials).to_csv(host_path, index=False)
        self._record_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool
    def save_hpo_parameter_importances_plot(
        self, study_handle: str, html_path: str
    ) -> str:
        """Save a Plotly HTML chart of HPO parameter importances.

        Args:
            study_handle (str): HPO study handle.
            html_path (str): Relative or `/workspace` HTML destination.

        Returns:
            str: Virtual path to the saved HTML file.

        Examples:
            print(
                save_hpo_parameter_importances_plot(
                    study_handle,
                    "/workspace/output/hpo_importances.html",
                )
            )
        """
        summary = self.summarize_hpo_study(study_handle)
        importance_rows = [
            {"parameter": name, "importance": value}
            for name, value in summary["parameter_importances"].items()
        ]
        if not importance_rows:
            importance_rows = [
                {"parameter": "no_parameter_importances", "importance": 0.0}
            ]
        figure = px.bar(
            pd.DataFrame(importance_rows),
            x="parameter",
            y="importance",
            title=f"Optuna parameter importances: {summary['study_name']}",
        )
        host_path = self._resolve_host_path(html_path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(host_path, include_plotlyjs="cdn")
        self._record_artifact(host_path)
        return str(self._os_access.virtualize_host_path(host_path))

    @tool
    def save_tuned_pipeline_report(
        self, tuned_handle: str, path: str
    ) -> dict[str, Any]:
        """Save a user-facing tuned pipeline summary as JSON or Markdown.

        Args:
            tuned_handle (str): Tuned pipeline handle.
            path (str): Relative or `/workspace` destination ending in `.json` or `.md`.

        Returns:
            dict[str, Any]: Saved path and report format.

        Examples:
            print(
                save_tuned_pipeline_report(
                    tuned_handle,
                    "/workspace/output/tuned_pipeline_report.md",
                )
            )
            # Returns:
            # {
            #     "path": "/workspace/output/tuned_pipeline_report.md",
            #     "format": "md"
            # }
        """
        summary = self.inspect_tuned_pipeline(tuned_handle)
        suffix = self._resolve_host_path(path).suffix.lower()
        if suffix == ".json":
            saved_path = self._write_json_artifact(path, summary)
        else:
            markdown = self._render_markdown_report(
                title="Tuned Pipeline Report",
                summary_items=[
                    ("Model class", summary["model_class"]),
                    ("Model feature count", len(summary["model_feature_columns"])),
                    ("Selected feature count", len(summary["selected_features"])),
                    ("Has preprocessor", summary["has_preprocessor"]),
                    ("Has feature engineer", summary["has_feature_engineer"]),
                ],
                warnings=[],
            )
            markdown += "\n## Evaluation Summary\n\n```json\n"
            markdown += json.dumps(
                summary["evaluation_summary"], indent=2, sort_keys=True
            )
            markdown += "\n```\n"
            saved_path = self._write_text_artifact(path, markdown)
        return {"path": saved_path, "format": suffix.lstrip(".") or "md"}

    @tool
    def export_best_pipeline_python(
        self,
        study_handle: str,
        path: str,
        *,
        tuned_handle: str | None = None,
    ) -> dict[str, Any]:
        """Export the best tuned pipeline configuration as a reusable Python file.

        Args:
            study_handle (str): HPO study handle with a completed best trial.
            path (str): Relative or `/workspace` Python destination.
            tuned_handle (str | None): Optional tuned pipeline handle used to embed
                final feature lists from a fitted artifact.

        Returns:
            dict[str, Any]: Saved path and exported config summary.

        Examples:
            print(
                export_best_pipeline_python(
                    study_handle,
                    "/workspace/output/best_pipeline.py",
                )
            )
            # Returns:
            # {
            #     "path": "/workspace/output/best_pipeline.py",
            #     "study_name": "lgbm_search"
            # }
        """
        study = self._get_study(study_handle)
        tuned_artifact = (
            self._get_tuned_pipeline(tuned_handle) if tuned_handle is not None else None
        )
        export_source = self._build_pipeline_python_export(
            study=study,
            tuned_artifact=tuned_artifact,
        )
        # Re-parse the generated module source to ensure we emitted valid Python.
        compile(export_source, str(path), "exec")
        saved_path = self._write_text_artifact(path, export_source)
        return {
            "path": saved_path,
            "best_trial_number": study.best_trial_number,
            "objective_metric": study.objective_metric,
            "best_value": study.best_value,
        }


__all__ = [
    "HpoCollection",
    "StoredHpoStudy",
    "StoredTunedPipeline",
]
