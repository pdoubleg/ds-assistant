"""Registered metrics helpers for reusable scorer and evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

from ..filesystem import HostWorkspaceOSAccess
from ..core.registry import ObjectStore, ToolCollection, tool
from .feature_engineering import StoredFeatureEngineer
from .freeform import StoredFreeformTransformer
from ..support.metrics import (
    compute_metric_score,
    compute_prediction_metrics,
    infer_task_type,
    metric_ppv,
    prepare_model_frames,
)
from .preprocessing import (
    StoredPreprocessor,
    _as_dataframe,
    _require_columns as require_preprocessing_columns,
)
from ..core.registry import safe_json_value

if TYPE_CHECKING:
    from .hpo import StoredTunedPipeline


@dataclass(slots=True)
class StoredMetricScorer:
    """Persisted scorer configuration for reusable modeling workflows.

    Args:
        metric_name (str): Canonical metric name or alias.
        task_type (str): Either ``classification`` or ``regression``.
        needs_proba (bool): Whether the scorer should consume probabilities.
        greater_is_better (bool): Whether higher values indicate better performance.
        average (str | None): Optional averaging mode for class metrics.
        pos_label (int | str): Positive class label for binary metrics.
        top_p (float | None): Optional PPV top-fraction cutoff.
        top_k (int | None): Optional PPV top-k cutoff.
        metric_kwargs (dict[str, Any]): Extra kwargs forwarded to the metric.
    """

    metric_name: str
    task_type: str
    needs_proba: bool = False
    greater_is_better: bool = True
    average: str | None = None
    pos_label: int | str = 1
    top_p: float | None = None
    top_k: int | None = None
    metric_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_json_summary(
        self,
        *,
        max_items: int = 100,
        max_chars: int = 1000,
    ) -> dict[str, Any]:
        """Render a compact JSON-friendly summary for handle inspection."""
        return {
            "type": "StoredMetricScorer",
            "metric_name": self.metric_name,
            "task_type": self.task_type,
            "needs_proba": self.needs_proba,
            "greater_is_better": self.greater_is_better,
            "average": self.average,
            "pos_label": self.pos_label,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "metric_kwargs": safe_json_value(
                self.metric_kwargs,
                max_items=max_items,
                max_chars=max_chars,
            ),
        }


def materialize_metric_scorer(scorer: StoredMetricScorer) -> Any:
    """Build a sklearn-compatible scorer object from a stored scorer artifact.

    Args:
        scorer (StoredMetricScorer): Stored scorer definition.

    Returns:
        Any: sklearn scorer object.
    """

    # PPV needs a small wrapper because sklearn scorers do not know about
    # top-p/top-k conventions out of the box.
    if scorer.metric_name in {"ppv", "ppv_at_k", "precision_at_k"}:
        return make_scorer(
            metric_ppv,
            greater_is_better=scorer.greater_is_better,
            needs_proba=True,
            top_p=scorer.top_p,
            top_k=scorer.top_k,
            positive_label=scorer.pos_label,
        )

    def _score_predictions(
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        **kwargs: Any,
    ) -> float:
        """Delegate to the shared metric-scoring helper with bound options."""
        return compute_metric_score(
            y_true,
            y_pred,
            metric_name=scorer.metric_name,
            task_type=scorer.task_type,
            y_pred_proba=kwargs.get("y_pred_proba"),
            average=scorer.average,
            pos_label=scorer.pos_label,
            greater_is_better=scorer.greater_is_better,
            metric_kwargs={**scorer.metric_kwargs, **kwargs},
        )

    return make_scorer(
        _score_predictions,
        greater_is_better=scorer.greater_is_better,
        needs_proba=scorer.needs_proba,
    )


def _coerce_probability_input(
    y_pred_proba: Sequence[Any] | None,
) -> np.ndarray | None:
    """Normalize probability-like tool arguments into NumPy arrays.

    Args:
        y_pred_proba (Sequence[Any] | None): Probability-like payload from a tool
            argument or fitted estimator.

    Returns:
        np.ndarray | None: One- or two-dimensional probability array, if provided.
    """
    if y_pred_proba is None:
        return None
    if isinstance(y_pred_proba, pd.Series):
        return y_pred_proba.to_numpy()
    if isinstance(y_pred_proba, pd.DataFrame):
        return y_pred_proba.to_numpy()
    return np.asarray(y_pred_proba)


def _prepare_target_for_tuned_model(
    target: pd.Series,
    *,
    task_type: str,
) -> pd.Series:
    """Validate evaluation targets for stored tuned-pipeline scoring."""
    if task_type == "regression":
        return pd.to_numeric(target, errors="coerce")
    if pd.api.types.is_integer_dtype(target.dropna()):
        return pd.to_numeric(target, errors="coerce")
    raise ValueError(
        "Stored tuned pipelines currently support direct evaluation only when "
        "classification targets are already numeric/encoded."
    )


def _apply_feature_engineer_to_frame(
    feature_frame: pd.DataFrame,
    artifact: StoredFeatureEngineer,
) -> pd.DataFrame:
    """Apply a stored feature engineer to an in-memory dataframe."""
    return artifact.estimator.transform(feature_frame[artifact.input_columns])


def _apply_freeform_transformer_to_frame(
    feature_frame: pd.DataFrame,
    artifact: StoredFreeformTransformer,
) -> pd.DataFrame:
    """Apply a stored freeform transformer to an in-memory dataframe."""
    return artifact.estimator.transform(feature_frame)


def _apply_preprocessor_to_frame(
    feature_frame: pd.DataFrame,
    artifact: StoredPreprocessor,
) -> pd.DataFrame:
    """Apply a stored preprocessor to an in-memory dataframe."""
    require_preprocessing_columns(feature_frame, artifact.input_columns)
    transformed = artifact.estimator.transform(feature_frame[artifact.input_columns])
    return _as_dataframe(
        transformed,
        index=feature_frame.index if artifact.preserve_index else None,
        columns=artifact.output_columns,
    )


class MetricsCollection(ToolCollection):
    """Reusable scorer factories and evaluation helpers."""

    name = "metrics"
    description = (
        "Create reusable scorer handles, inspect metric definitions, and evaluate "
        "predictions or tuned pipeline artifacts with common sklearn-style metrics."
    )

    def __init__(
        self,
        os_access: HostWorkspaceOSAccess,
        object_store: ObjectStore,
    ) -> None:
        """Initialize metrics helpers.

        Args:
            os_access (HostWorkspaceOSAccess): Workspace path sandbox adapter.
            object_store (ObjectStore): Shared handle store.
        """
        self._os_access = os_access
        self._object_store = object_store

    def _get_dataframe(self, dataframe_handle: str) -> pd.DataFrame:
        """Fetch a dataframe from the shared object store."""
        return self._object_store.get(dataframe_handle, expected_type=pd.DataFrame)

    def _get_metric_scorer(self, scorer_handle: str) -> StoredMetricScorer:
        """Fetch a stored scorer artifact from the object store."""
        return self._object_store.get(scorer_handle, expected_type=StoredMetricScorer)

    def _get_tuned_pipeline(self, tuned_handle: str) -> StoredTunedPipeline:
        """Fetch a stored tuned-pipeline artifact from the object store."""
        from .hpo import StoredTunedPipeline

        return self._object_store.get(tuned_handle, expected_type=StoredTunedPipeline)

    def _put_metric_scorer(
        self,
        *,
        metric_name: str,
        task_type: str,
        needs_proba: bool = False,
        greater_is_better: bool = True,
        average: str | None = None,
        pos_label: int | str = 1,
        top_p: float | None = None,
        top_k: int | None = None,
        metric_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Persist a normalized scorer artifact and return its handle."""
        artifact = StoredMetricScorer(
            metric_name=str(metric_name),
            task_type=str(task_type),
            needs_proba=bool(needs_proba),
            greater_is_better=bool(greater_is_better),
            average=average,
            pos_label=pos_label,
            top_p=float(top_p) if top_p is not None else None,
            top_k=int(top_k) if top_k is not None else None,
            metric_kwargs=dict(metric_kwargs or {}),
        )
        return self._object_store.put(artifact, prefix="metric")

    def _evaluate_prediction_payload(
        self,
        *,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        task_type: str | None = None,
        y_pred_proba: Sequence[Any] | None = None,
        scorer: StoredMetricScorer | None = None,
    ) -> dict[str, Any]:
        """Build a JSON-friendly evaluation payload from raw prediction arrays."""
        resolved_task_type = task_type or infer_task_type(pd.Series(y_true))
        probabilities = _coerce_probability_input(y_pred_proba)
        metrics = compute_prediction_metrics(
            pd.Series(y_true),
            pd.Series(y_pred),
            task_type=resolved_task_type,
            y_pred_proba=probabilities,
        )
        score_payload = None
        if scorer is not None:
            scorer_score = compute_metric_score(
                y_true,
                y_pred,
                metric_name=scorer.metric_name,
                task_type=resolved_task_type,
                y_pred_proba=probabilities,
                average=scorer.average,
                pos_label=scorer.pos_label,
                top_p=scorer.top_p,
                top_k=scorer.top_k,
                greater_is_better=True,
                metric_kwargs=scorer.metric_kwargs,
            )
            metrics[str(scorer.metric_name)] = scorer_score
            score_payload = {
                "metric_name": scorer.metric_name,
                "score": scorer_score,
            }
        return {
            "task_type": resolved_task_type,
            "metrics": metrics,
            "scorer": scorer.to_json_summary() if scorer is not None else None,
            "scorer_result": score_payload,
        }

    @tool
    def create_metric_scorer(
        self,
        metric_name: str,
        *,
        task_type: str = "classification",
        needs_proba: bool = False,
        greater_is_better: bool = True,
        average: str | None = None,
        pos_label: int | str = 1,
        metric_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Create a reusable scorer handle for a common metric.

        Args:
            metric_name (str): Canonical metric name or alias.
            task_type (str): Modeling task type, usually ``classification`` or
                ``regression``.
            needs_proba (bool): Whether the scorer expects probability estimates.
            greater_is_better (bool): Whether larger metric values are better.
            average (str | None): Optional sklearn-style averaging mode.
            pos_label (int | str): Positive class label for binary metrics.
            metric_kwargs (dict[str, Any] | None): Optional metric-specific keyword
                arguments.

        Returns:
            str: Handle for the stored metric scorer artifact.

        Examples:
            metric_handle = create_metric_scorer(
                "roc_auc",
                task_type="classification",
                needs_proba=True,
                metric_kwargs={"multi_class": "ovr"},
            )
        """
        return self._put_metric_scorer(
            metric_name=metric_name,
            task_type=task_type,
            needs_proba=needs_proba,
            greater_is_better=greater_is_better,
            average=average,
            pos_label=pos_label,
            metric_kwargs=metric_kwargs,
        )

    @tool
    def create_ppv_scorer(
        self,
        *,
        top_p: float | None = None,
        top_k: int | None = None,
        pos_label: int | str = 1,
    ) -> str:
        """Create a reusable PPV scorer handle using either top-p or top-k ranking.

        Args:
            top_p (float | None): Optional top-fraction cutoff.
            top_k (int | None): Optional top-k cutoff.
            pos_label (int | str): Positive class label for binary targets.

        Returns:
            str: Handle for the stored PPV scorer artifact.

        Examples:
            metric_handle = create_ppv_scorer(top_k=50)
        """
        return self._put_metric_scorer(
            metric_name="ppv",
            task_type="classification",
            needs_proba=True,
            pos_label=pos_label,
            top_p=top_p,
            top_k=top_k,
        )

    @tool
    def inspect_metric_scorer(self, scorer_handle: str) -> dict[str, Any]:
        """Return a JSON-friendly summary of a stored metric scorer.

        Args:
            scorer_handle (str): Handle pointing to a stored metric scorer.

        Returns:
            dict[str, Any]: JSON-friendly scorer configuration summary.

        Examples:
            print(inspect_metric_scorer(metric_handle))
            # Returns:
            # {
            #     "type": "StoredMetricScorer",
            #     "metric_name": "roc_auc",
            #     "task_type": "classification",
            #     "needs_proba": True,
            #     "metric_kwargs": {"multi_class": "ovr"}
            # }
        """
        return self._get_metric_scorer(scorer_handle).to_json_summary()

    @tool
    def score_with_metric_handle(
        self,
        scorer_handle: str,
        *,
        y_true: list[Any],
        y_pred: list[Any],
        y_pred_proba: list[Any] | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Score one prediction set with a reusable scorer handle.

        Args:
            scorer_handle (str): Handle pointing to a stored metric scorer.
            y_true (list[Any]): Ground-truth labels or values.
            y_pred (list[Any]): Predicted labels or values.
            y_pred_proba (list[Any] | None): Optional probability estimates.
            task_type (str | None): Optional task-type override.

        Returns:
            dict[str, Any]: Scorer name, resolved task type, and computed score.

        Examples:
            print(
                score_with_metric_handle(
                    metric_handle,
                    y_true=[0, 1],
                    y_pred=[0, 1],
                    y_pred_proba=[0.2, 0.9],
                )
            )
            # Returns:
            # {
            #     "task_type": "classification",
            #     "metric_name": "roc_auc",
            #     "score": 1.0
            # }
        """
        scorer = self._get_metric_scorer(scorer_handle)
        payload = self._evaluate_prediction_payload(
            y_true=y_true,
            y_pred=y_pred,
            task_type=task_type or scorer.task_type,
            y_pred_proba=y_pred_proba,
            scorer=scorer,
        )
        return {
            "task_type": payload["task_type"],
            "metric_name": scorer.metric_name,
            "score": payload["scorer_result"]["score"],
        }

    @tool
    def evaluate_predictions(
        self,
        *,
        y_true: list[Any],
        y_pred: list[Any],
        y_pred_proba: list[Any] | None = None,
        task_type: str | None = None,
        scorer_handle: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate raw predictions and optionally apply a stored scorer.

        Args:
            y_true (list[Any]): Ground-truth labels or values.
            y_pred (list[Any]): Predicted labels or values.
            y_pred_proba (list[Any] | None): Optional probability estimates.
            task_type (str | None): Optional task-type override.
            scorer_handle (str | None): Optional stored scorer handle.

        Returns:
            dict[str, Any]: Metrics payload with optional scorer metadata.

        Examples:
            print(
                evaluate_predictions(
                    y_true=[0, 1],
                    y_pred=[0, 1],
                    y_pred_proba=[0.1, 0.9],
                )
            )
            # Returns:
            # {
            #     "task_type": "classification",
            #     "metrics": {"accuracy": 1.0, "roc_auc": 1.0},
            #     "scorer": None,
            #     "scorer_result": None
            # }
        """
        scorer = self._get_metric_scorer(scorer_handle) if scorer_handle else None
        return self._evaluate_prediction_payload(
            y_true=y_true,
            y_pred=y_pred,
            task_type=task_type or (scorer.task_type if scorer else None),
            y_pred_proba=y_pred_proba,
            scorer=scorer,
        )

    @tool
    def evaluate_prediction_dataframe(
        self,
        dataframe_handle: str,
        *,
        target_column: str,
        prediction_column: str,
        probability_column: str | None = None,
        task_type: str | None = None,
        scorer_handle: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate prediction columns stored in a dataframe handle.

        Args:
            dataframe_handle (str): Handle pointing to the evaluation dataframe.
            target_column (str): Column containing ground-truth values.
            prediction_column (str): Column containing predicted values.
            probability_column (str | None): Optional probability column.
            task_type (str | None): Optional task-type override.
            scorer_handle (str | None): Optional stored scorer handle.

        Returns:
            dict[str, Any]: Metrics payload with optional scorer metadata.

        Examples:
            print(
                evaluate_prediction_dataframe(
                    df_handle,
                    target_column="target",
                    prediction_column="prediction",
                    probability_column="score",
                )
            )
            # Returns:
            # {
            #     "task_type": "classification",
            #     "metrics": {"accuracy": 0.92, "roc_auc": 0.95},
            #     "scorer": None,
            #     "scorer_result": None
            # }
        """
        dataframe = self._get_dataframe(dataframe_handle)
        required_columns = [target_column, prediction_column]
        if probability_column is not None:
            required_columns.append(probability_column)
        missing_columns = [
            column for column in required_columns if column not in dataframe.columns
        ]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Missing required columns: {missing_text}.")

        scorer = self._get_metric_scorer(scorer_handle) if scorer_handle else None
        return self._evaluate_prediction_payload(
            y_true=dataframe[target_column].tolist(),
            y_pred=dataframe[prediction_column].tolist(),
            task_type=task_type or (scorer.task_type if scorer else None),
            y_pred_proba=(
                dataframe[probability_column].tolist()
                if probability_column is not None
                else None
            ),
            scorer=scorer,
        )

    @tool
    def evaluate_tuned_pipeline(
        self,
        tuned_handle: str,
        dataframe_handle: str,
        *,
        target_column: str,
        scorer_handle: str | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a stored tuned pipeline artifact on a dataframe handle.

        Args:
            tuned_handle (str): Handle pointing to a stored tuned pipeline artifact.
            dataframe_handle (str): Handle pointing to the evaluation dataframe.
            target_column (str): Name of the target column in the dataframe.
            scorer_handle (str | None): Optional stored scorer handle.
            task_type (str | None): Optional task-type override.

        Returns:
            dict[str, Any]: Metrics payload augmented with pipeline summary metadata.

        Examples:
            print(
                evaluate_tuned_pipeline(
                    tuned_handle,
                    df_handle,
                    target_column="target",
                )
            )
            # Returns:
            # {
            #     "task_type": "classification",
            #     "metrics": {"accuracy": 0.91, "roc_auc": 0.94},
            #     "row_count": 250,
            #     "tuned_pipeline": {
            #         "model_class": "LGBMClassifier",
            #         "feature_count": 18
            #     }
            # }
        """
        tuned_artifact = self._get_tuned_pipeline(tuned_handle)
        dataframe = self._get_dataframe(dataframe_handle)
        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column {target_column!r} was not found in the dataframe."
            )

        target = dataframe[target_column].copy()
        feature_frame = dataframe.drop(columns=[target_column]).copy()
        if tuned_artifact.freeform_transformer is not None:
            feature_frame = _apply_freeform_transformer_to_frame(
                feature_frame,
                tuned_artifact.freeform_transformer,
            )
        if tuned_artifact.feature_engineer is not None:
            feature_frame = _apply_feature_engineer_to_frame(
                feature_frame,
                tuned_artifact.feature_engineer,
            )
        if tuned_artifact.preprocessor is not None:
            feature_frame = _apply_preprocessor_to_frame(
                feature_frame,
                tuned_artifact.preprocessor,
            )

        required_columns = (
            tuned_artifact.model_feature_columns
            if tuned_artifact.model_feature_columns
            else tuned_artifact.selected_features
        )
        if required_columns:
            missing_columns = [
                column
                for column in required_columns
                if column not in feature_frame.columns
            ]
            if missing_columns:
                missing_text = ", ".join(missing_columns[:10])
                raise ValueError(
                    "The evaluation dataframe is missing model input columns: "
                    f"{missing_text}."
                )
            feature_frame = feature_frame[required_columns].copy()

        resolved_task_type = task_type or infer_task_type(target)
        prepared_target = _prepare_target_for_tuned_model(
            target,
            task_type=resolved_task_type,
        )
        prepared_features, _ = prepare_model_frames(feature_frame)
        predictions = tuned_artifact.fitted_model.predict(prepared_features)
        probabilities = (
            tuned_artifact.fitted_model.predict_proba(prepared_features)
            if hasattr(tuned_artifact.fitted_model, "predict_proba")
            else None
        )

        scorer = self._get_metric_scorer(scorer_handle) if scorer_handle else None
        payload = self._evaluate_prediction_payload(
            y_true=prepared_target.tolist(),
            y_pred=list(predictions),
            task_type=resolved_task_type,
            y_pred_proba=(
                probabilities.tolist() if probabilities is not None else None
            ),
            scorer=scorer,
        )
        payload["row_count"] = int(len(prepared_features))
        payload["tuned_pipeline"] = {
            "model_class": type(tuned_artifact.fitted_model).__name__,
            "feature_count": len(required_columns),
        }
        return payload


__all__ = [
    "MetricsCollection",
    "StoredMetricScorer",
    "materialize_metric_scorer",
]
