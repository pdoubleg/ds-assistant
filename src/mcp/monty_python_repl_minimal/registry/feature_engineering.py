"""Deterministic feature-engineering helpers for the minimal registry package."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..base import WorkspaceToolCollection
from ..core.registry import tool
from ..privacy import summarize_dataframe
from .base import StoredFeatureEngineeringPipeline


class FeatureEngineeringCollection(WorkspaceToolCollection):
    """Composable feature-engineering helpers built from approved transforms."""

    name = "feature_engineering"
    description = (
        "Fit, inspect, and apply deterministic feature-engineering pipelines "
        "built from pre-defined transform steps."
    )

    @tool
    def list_feature_pipeline_steps(self) -> dict[str, Any]:
        """Describe the supported deterministic feature-pipeline step kinds.

        Call this before `fit_feature_pipeline(...)` so the agent can compose a
        valid pipeline spec from approved transforms rather than inventing ad hoc
        code inside `execute(...)`.

        Returns:
            dict[str, Any]: Step catalog and example step specifications.

        Examples:
            ```python
            step_catalog = list_feature_pipeline_steps()
            # Returns
            # {
            #     "step_kinds": [
            #         {
            #             "kind": "drop_columns",
            #             "required_keys": ["columns"],
            #             "optional_keys": [],
            #             "description": "Drop existing columns from the feature frame.",
            #         },
            #         {
            #             "kind": "keep_columns",
            #             "required_keys": ["columns"],
            #             "optional_keys": [],
            #             "description": "Keep only the requested feature columns.",
            #         },
            #         ...
            #     ]
            # }
            ```
        """

        return {
            "step_kinds": [
                {
                    "kind": "drop_columns",
                    "required_keys": ["columns"],
                    "optional_keys": [],
                    "description": "Drop existing columns from the feature frame.",
                },
                {
                    "kind": "keep_columns",
                    "required_keys": ["columns"],
                    "optional_keys": [],
                    "description": "Keep only the requested feature columns.",
                },
                {
                    "kind": "ratio_features",
                    "required_keys": ["definitions"],
                    "optional_keys": [],
                    "description": (
                        "Create ratio columns from numerator and denominator columns."
                    ),
                    "definition_schema": {
                        "name": "debt_to_income",
                        "numerator": "debt",
                        "denominator": "income",
                        "fill_value": 0.0,
                        "epsilon": 1e-6,
                    },
                },
                {
                    "kind": "log1p_features",
                    "required_keys": ["columns"],
                    "optional_keys": ["suffix", "clip_min"],
                    "description": (
                        "Create new `log1p` feature columns from numeric inputs."
                    ),
                },
                {
                    "kind": "clip_columns",
                    "required_keys": ["columns"],
                    "optional_keys": ["lower", "upper"],
                    "description": "Clip numeric columns in place.",
                },
                {
                    "kind": "fill_missing",
                    "required_keys": ["columns", "strategy"],
                    "optional_keys": ["value"],
                    "description": (
                        "Fill missing values with a fitted constant, mean, median, "
                        "or mode."
                    ),
                },
            ]
        }

    def _ensure_columns_exist(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:
        """Validate that all requested columns exist."""

        missing_columns = [
            column for column in columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}.")

    def _resolve_fill_value(
        self,
        series: pd.Series,
        *,
        strategy: str,
        explicit_value: Any | None,
    ) -> Any:
        """Resolve a deterministic fill value from a column and strategy."""

        if strategy == "constant":
            return explicit_value
        if strategy == "mean":
            return float(pd.to_numeric(series, errors="coerce").mean())
        if strategy == "median":
            return float(pd.to_numeric(series, errors="coerce").median())
        if strategy == "mode":
            modes = series.dropna().mode()
            return None if modes.empty else modes.iloc[0]
        raise ValueError("`strategy` must be one of: constant, mean, median, or mode.")

    def _fit_pipeline_steps(
        self,
        dataframe: pd.DataFrame,
        steps: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
        """Fit and apply supported pipeline steps to a feature dataframe."""

        working = dataframe.copy()
        resolved_steps: list[dict[str, Any]] = []
        warnings: list[str] = []

        for raw_step in steps:
            kind = str(raw_step.get("kind", "")).strip()
            if not kind:
                raise ValueError("Each pipeline step must include a non-empty `kind`.")

            if kind == "drop_columns":
                columns = list(raw_step.get("columns") or [])
                self._ensure_columns_exist(working, columns)
                working = working.drop(columns=columns)
                resolved_steps.append({"kind": kind, "columns": columns})
                continue

            if kind == "keep_columns":
                columns = list(raw_step.get("columns") or [])
                self._ensure_columns_exist(working, columns)
                working = working[columns].copy()
                resolved_steps.append({"kind": kind, "columns": columns})
                continue

            if kind == "ratio_features":
                definitions = list(raw_step.get("definitions") or [])
                if not definitions:
                    raise ValueError(
                        "`ratio_features` requires non-empty `definitions`."
                    )

                resolved_definitions: list[dict[str, Any]] = []
                for definition in definitions:
                    numerator = str(definition["numerator"])
                    denominator = str(definition["denominator"])
                    name = str(definition["name"])
                    fill_value = float(definition.get("fill_value", 0.0))
                    epsilon = float(definition.get("epsilon", 1e-6))
                    self._ensure_columns_exist(working, [numerator, denominator])

                    numerator_values = pd.to_numeric(
                        working[numerator],
                        errors="coerce",
                    )
                    denominator_values = pd.to_numeric(
                        working[denominator],
                        errors="coerce",
                    )
                    safe_denominator = denominator_values.where(
                        denominator_values.abs() > epsilon
                    )
                    ratio = numerator_values / safe_denominator
                    working[name] = ratio.replace([np.inf, -np.inf], np.nan).fillna(
                        fill_value
                    )
                    resolved_definitions.append(
                        {
                            "name": name,
                            "numerator": numerator,
                            "denominator": denominator,
                            "fill_value": fill_value,
                            "epsilon": epsilon,
                        }
                    )

                resolved_steps.append(
                    {"kind": kind, "definitions": resolved_definitions}
                )
                continue

            if kind == "log1p_features":
                columns = list(raw_step.get("columns") or [])
                suffix = str(raw_step.get("suffix", "_log1p"))
                clip_min = float(raw_step.get("clip_min", 0.0))
                self._ensure_columns_exist(working, columns)

                for column in columns:
                    numeric_values = pd.to_numeric(working[column], errors="coerce")
                    working[f"{column}{suffix}"] = np.log1p(
                        numeric_values.clip(lower=clip_min)
                    )

                resolved_steps.append(
                    {
                        "kind": kind,
                        "columns": columns,
                        "suffix": suffix,
                        "clip_min": clip_min,
                    }
                )
                continue

            if kind == "clip_columns":
                columns = list(raw_step.get("columns") or [])
                self._ensure_columns_exist(working, columns)
                lower = raw_step.get("lower")
                upper = raw_step.get("upper")
                if lower is None and upper is None:
                    raise ValueError(
                        "`clip_columns` requires `lower`, `upper`, or both."
                    )

                for column in columns:
                    numeric_values = pd.to_numeric(working[column], errors="coerce")
                    working[column] = numeric_values.clip(lower=lower, upper=upper)

                resolved_steps.append(
                    {
                        "kind": kind,
                        "columns": columns,
                        "lower": lower,
                        "upper": upper,
                    }
                )
                continue

            if kind == "fill_missing":
                columns = list(raw_step.get("columns") or [])
                strategy = str(raw_step.get("strategy", "")).strip()
                explicit_value = raw_step.get("value")
                self._ensure_columns_exist(working, columns)

                fill_values: dict[str, Any] = {}
                for column in columns:
                    fill_value = self._resolve_fill_value(
                        working[column],
                        strategy=strategy,
                        explicit_value=explicit_value,
                    )
                    fill_values[column] = fill_value
                    working[column] = working[column].fillna(fill_value)

                resolved_steps.append(
                    {
                        "kind": kind,
                        "columns": columns,
                        "strategy": strategy,
                        "fill_values": fill_values,
                    }
                )
                continue

            raise ValueError(f"Unsupported feature pipeline step kind: {kind!r}.")

        return working, resolved_steps, warnings

    def _apply_pipeline_steps(
        self,
        dataframe: pd.DataFrame,
        pipeline: StoredFeatureEngineeringPipeline,
    ) -> pd.DataFrame:
        """Apply a fitted deterministic feature pipeline to a new dataframe."""

        working = dataframe.copy()
        for step in pipeline.steps:
            kind = step["kind"]

            if kind == "drop_columns":
                columns = [
                    column for column in step["columns"] if column in working.columns
                ]
                working = working.drop(columns=columns)
                continue

            if kind == "keep_columns":
                self._ensure_columns_exist(working, list(step["columns"]))
                working = working[list(step["columns"])].copy()
                continue

            if kind == "ratio_features":
                for definition in step["definitions"]:
                    numerator = definition["numerator"]
                    denominator = definition["denominator"]
                    self._ensure_columns_exist(working, [numerator, denominator])
                    numerator_values = pd.to_numeric(
                        working[numerator], errors="coerce"
                    )
                    denominator_values = pd.to_numeric(
                        working[denominator],
                        errors="coerce",
                    )
                    safe_denominator = denominator_values.where(
                        denominator_values.abs() > definition["epsilon"]
                    )
                    working[definition["name"]] = (
                        (numerator_values / safe_denominator)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(definition["fill_value"])
                    )
                continue

            if kind == "log1p_features":
                for column in step["columns"]:
                    self._ensure_columns_exist(working, [column])
                    numeric_values = pd.to_numeric(working[column], errors="coerce")
                    working[f"{column}{step['suffix']}"] = np.log1p(
                        numeric_values.clip(lower=step["clip_min"])
                    )
                continue

            if kind == "clip_columns":
                for column in step["columns"]:
                    self._ensure_columns_exist(working, [column])
                    numeric_values = pd.to_numeric(working[column], errors="coerce")
                    working[column] = numeric_values.clip(
                        lower=step["lower"],
                        upper=step["upper"],
                    )
                continue

            if kind == "fill_missing":
                for column, fill_value in step["fill_values"].items():
                    self._ensure_columns_exist(working, [column])
                    working[column] = working[column].fillna(fill_value)
                continue

            raise ValueError(
                f"Unsupported fitted feature pipeline step kind: {kind!r}."
            )

        expected_columns = list(pipeline.output_columns)
        missing_columns = [
            column for column in expected_columns if column not in working.columns
        ]
        if missing_columns:
            raise ValueError(
                "Transformed dataframe is missing expected output columns: "
                + ", ".join(missing_columns)
                + "."
            )
        return working

    @tool
    def fit_feature_pipeline(
        self,
        dataframe_handle: str,
        steps: list[dict[str, Any]],
        *,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """Fit a deterministic feature-engineering pipeline from approved steps.

        Use this to build reusable feature transforms without embedding ad hoc
        code strings. The fitted pipeline can then be applied to training,
        validation, or scoring dataframes through a stored handle.

        Args:
            dataframe_handle (str): Source dataframe handle used for fitting.
            steps (list[dict[str, Any]]): Ordered list of approved pipeline steps.
            target_column (str | None): Optional target column excluded during fit.

        Returns:
            dict[str, Any]: Pipeline handle and compact fit summary.

        Examples:
            ```python
            dataset = load_csv("/workspace/train.csv")
            pipeline = fit_feature_pipeline(
                dataset["dataframe_handle"],
                [
                    {"kind": "drop_columns", "columns": ["customer_id"]},
                    {
                        "kind": "ratio_features",
                        "definitions": [
                            {
                                "name": "balance_to_limit",
                                "numerator": "balance",
                                "denominator": "credit_limit",
                                "fill_value": 0.0,
                            }
                        ],
                    },
                ],
                target_column="target",
            )
            # Returns
            # {
            #     "pipeline_handle": "pipeline_abc123",
            #     "summary": "Fitted feature pipeline with 2 step(s).",
            #     "output_column_count": 1,
            #     "warnings": [],
            # }
            ```
        """

        dataframe = self._get_dataframe(dataframe_handle).copy()
        feature_frame = dataframe.copy()
        if target_column is not None:
            if target_column not in feature_frame.columns:
                raise ValueError(f"Target column {target_column!r} was not found.")
            feature_frame = feature_frame.drop(columns=[target_column]).copy()

        transformed, resolved_steps, warnings = self._fit_pipeline_steps(
            feature_frame,
            steps,
        )
        pipeline = StoredFeatureEngineeringPipeline(
            target_column=target_column,
            steps=resolved_steps,
            input_columns=[str(column) for column in feature_frame.columns],
            output_columns=[str(column) for column in transformed.columns],
            warnings=warnings,
            metadata={"step_count": len(resolved_steps)},
        )
        pipeline_handle = self._object_store.put(pipeline, prefix="pipeline")
        summary = f"Fitted feature pipeline with {len(resolved_steps)} step(s)."
        return {
            "pipeline_handle": pipeline_handle,
            "summary": summary,
            "output_column_count": len(pipeline.output_columns),
            "warnings": warnings,
        }

    @tool
    def transform_with_feature_pipeline(
        self,
        dataframe_handle: str,
        pipeline_handle: str,
        *,
        include_target: bool = False,
    ) -> dict[str, Any]:
        """Apply a fitted deterministic feature pipeline to a stored dataframe.

        Args:
            dataframe_handle (str): Input dataframe handle to transform.
            pipeline_handle (str): Stored fitted feature-pipeline handle.
            include_target (bool): Whether to append the saved target column back to
                the transformed dataframe when available.

        Returns:
            dict[str, Any]: Transformed dataframe handle and compact summary.

        Examples:
            ```python
            dataset = load_csv("/workspace/train.csv")
            pipeline = fit_feature_pipeline(
                dataset["dataframe_handle"],
                [{"kind": "log1p_features", "columns": ["balance"]}],
                target_column="target",
            )
            transformed = transform_with_feature_pipeline(
                dataset["dataframe_handle"],
                pipeline["pipeline_handle"],
                include_target=True,
            )
            # Returns
            # {
            #     "dataframe_handle": "df_abc123",
            #     "pipeline_handle": "pipeline_abc123",
            #     "summary": "Applied feature pipeline with 1 step(s).",
            #     "warnings": [],
            #     "dataframe_summary": {
            #         "type": "DataFrame",
            #         "shape": [1000, 1],
            #         "columns": ["balance_log1p"],
            #         "column_type_counts": {
            #             "numeric": 1,
            #             "datetime": 0,
            #             "categorical": 0,
            #             "other": 0,
            #         },
            #     }
            # }
            ```
        """

        pipeline = self._object_store.get(
            pipeline_handle,
            expected_type=StoredFeatureEngineeringPipeline,
        )
        dataframe = self._get_dataframe(dataframe_handle).copy()
        target_series = None
        if (
            pipeline.target_column is not None
            and pipeline.target_column in dataframe.columns
        ):
            target_series = dataframe[pipeline.target_column].copy()
            dataframe = dataframe.drop(columns=[pipeline.target_column]).copy()

        transformed = self._apply_pipeline_steps(dataframe, pipeline)
        if include_target and target_series is not None:
            transformed[pipeline.target_column] = target_series.values

        output_handle = self._object_store.put(transformed, prefix="df")
        summary = f"Applied feature pipeline with {len(pipeline.steps)} step(s)."
        return {
            "dataframe_handle": output_handle,
            "pipeline_handle": pipeline_handle,
            "summary": summary,
            "warnings": list(pipeline.warnings),
            "dataframe_summary": summarize_dataframe(transformed),
        }

    @tool
    def inspect_feature_pipeline(self, pipeline_handle: str) -> dict[str, Any]:
        """Return a compact summary of a fitted feature-engineering pipeline.

        Args:
            pipeline_handle (str): Stored fitted pipeline handle.

        Returns:
            dict[str, Any]: Safe pipeline summary.

        Examples:
            ```python
            pipeline_summary = inspect_feature_pipeline(pipeline_handle)
            # Returns
            # {
            #     "type": "StoredFeatureEngineeringPipeline",
            #     "target_column": "target",
            #     "input_columns": ["balance", "income"],
            #     "output_columns": ["balance_log1p", "income_log1p"],
            #     "steps": [
            #         {
            #             "kind": "log1p_features",
            #             "columns": ["balance", "income"],
            #             "suffix": "_log1p",
            #             "clip_min": 0.0
            #         }
            #     ],
            #     "warnings": [],
            #     "metadata": {}
            # }
            ```
        """

        pipeline = self._object_store.get(
            pipeline_handle,
            expected_type=StoredFeatureEngineeringPipeline,
        )
        return pipeline.to_json_summary()


__all__ = ["FeatureEngineeringCollection"]
