import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.tune.utils import make_serializable


TEST_LOGGER = False


class BaseTuningLogger(ABC):
    """Abstract base class for tuning experiment loggers."""

    @abstractmethod
    def start_run(self, run_name: str = None) -> None:
        """Start a new logging run/experiment."""
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current logging run."""
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        pass

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter value."""
        pass

    @abstractmethod
    def log_artifact(self, data: Any, name: str) -> None:
        """Log an artifact (file, object, etc.)."""
        pass

    @abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration data."""
        pass

    def log_agent_iteration(
        self, iteration: int, agent_type: str, result: dict
    ) -> None:
        """
        Log agent iteration results with structured data and key metrics.

        This method handles the complete logging pattern from _log_agent_iteration:
        1. Creates structured iteration data with timestamp
        2. Logs the data as an artifact with a specific path structure
        3. Logs key metrics from the result

        Args:
            iteration (int): The iteration number
            agent_type (str): Type of agent (e.g., "analysis", "initial_search_space", "refinement")
            result (dict): The result dictionary containing agent output
        """
        # Create structured iteration data (matching original format)
        iteration_data = {
            "iteration": iteration,
            "agent_type": agent_type,
            "timestamp": pd.Timestamp.now().isoformat(),
            "result": result,
        }

        # Log as artifact with specific path structure (matching original)
        artifact_name = f"agent_iterations/iteration_{iteration}_{agent_type}"
        self.log_artifact(iteration_data, artifact_name)

        # Log key metrics from result (matching original behavior)
        if "best_score" in result:
            self.log_metric(
                f"iteration_{iteration}_best_score",
                result["best_score"],
                step=iteration,
            )
        if "improvement" in result:
            self.log_metric(
                f"iteration_{iteration}_improvement",
                result["improvement"],
                step=iteration,
            )

    def save_best_params(
        self, best_params: dict, export_json: bool = True, export_yaml: bool = True
    ) -> None:
        """
        Save best model parameters in user-friendly formats.

        Args:
            best_params (dict): Best hyperparameters found during tuning
            export_json (bool): Whether to save as JSON
            export_yaml (bool): Whether to save as YAML
        """
        # Default implementation does nothing - override in specific loggers
        pass

    def save_tuning_summary(
        self, summary: dict, export_json: bool = True, export_yaml: bool = True
    ) -> None:
        """
        Save comprehensive tuning summary.

        Args:
            summary (dict): Complete tuning summary
            export_json (bool): Whether to save as JSON
            export_yaml (bool): Whether to save as YAML
        """
        # Default implementation does nothing - override in specific loggers
        pass


class MLflowLogger(BaseTuningLogger):
    """MLflow implementation of TuningLogger."""

    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name
        self._active_run = None

        try:
            import mlflow  # type: ignore

            self.mlflow = mlflow
            self.available = True
            if experiment_name:
                mlflow.set_experiment(experiment_name)
        except ImportError:
            self.available = False
            self.mlflow = None

    def start_run(self, run_name: str = None) -> None:
        if self.available:
            self._active_run = self.mlflow.start_run(run_name=run_name, nested=True)

    def end_run(self) -> None:
        if self.available and self._active_run:
            self.mlflow.end_run()
            self._active_run = None

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self.available:
            self.mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: Any) -> None:
        if self.available:
            self.mlflow.log_param(key, str(value))

    def log_artifact(self, data: Any, name: str) -> None:
        if self.available:
            import os
            import tempfile

            # Handle the artifact path structure for MLflow
            # If name contains a path, extract directory and filename
            if "/" in name:
                artifact_path_parts = name.split("/")
                filename = artifact_path_parts[-1] + ".json"
                artifact_dir = "/".join(artifact_path_parts[:-1])
            else:
                filename = name + ".json"
                artifact_dir = None

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name

            if artifact_dir:
                self.mlflow.log_artifact(temp_path, f"{artifact_dir}/{filename}")
            else:
                self.mlflow.log_artifact(temp_path, filename)
            os.unlink(temp_path)

    def log_config(self, config: Dict[str, Any]) -> None:
        if self.available:
            for key, value in config.items():
                self.log_param(key, value)

    def save_best_params(
        self, best_params: dict, export_json: bool = True, export_yaml: bool = True
    ) -> None:
        """Save best parameters as MLflow artifacts."""
        if not self.available:
            return

        serializable_params = make_serializable(best_params)

        if export_json:
            self.log_artifact(serializable_params, "best_model_params")

        if export_yaml:
            # For MLflow, we'll save YAML as an artifact too
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(
                    {"best_parameters": serializable_params},
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
                temp_path = f.name

            self.mlflow.log_artifact(temp_path, "best_model_params.yaml")
            os.unlink(temp_path)

    def save_tuning_summary(
        self, summary: dict, export_json: bool = True, export_yaml: bool = True
    ) -> None:
        """Save tuning summary as MLflow artifacts."""
        if not self.available:
            return

        serializable_summary = make_serializable(summary)

        if export_json:
            self.log_artifact(serializable_summary, "tuning_summary")

        if export_yaml:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(
                    serializable_summary,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
                temp_path = f.name

            self.mlflow.log_artifact(temp_path, "tuning_summary.yaml")
            os.unlink(temp_path)


class LocalFileLogger(BaseTuningLogger):
    """Local file system implementation of TuningLogger."""

    def __init__(self, output_dir: str = "tunning_logs"):
        self.output_directory = Path(output_dir)
        self.current_run_dir = None
        self.metrics = {}
        self.params = {}
        self.artifacts = {}

    def start_run(self, run_name: str = None) -> None:
        from datetime import datetime

        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_run_dir = self.output_directory / run_name
        self.current_run_dir.mkdir(parents=True, exist_ok=True)

        # Reset tracking
        self.metrics = {}
        self.params = {}
        self.artifacts = {}

    def end_run(self) -> None:
        if self.current_run_dir:
            # Save final summary
            summary = {
                "metrics": self.metrics,
                "params": self.params,
                "artifacts": list(self.artifacts.keys()),
            }

            with open(self.current_run_dir / "run_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if step is not None:
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append({"step": step, "value": value})
        else:
            self.metrics[key] = value

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = str(value)

    def log_artifact(self, data: Any, name: str) -> None:
        if self.current_run_dir:
            # Handle nested directory structure (e.g., "agent_iterations/iteration_1_analysis")
            if "/" in name:
                artifact_path_parts = name.split("/")
                subdir = self.current_run_dir / artifact_path_parts[0]
                subdir.mkdir(exist_ok=True)
                file_path = subdir / f"{artifact_path_parts[1]}.json"
            else:
                file_path = self.current_run_dir / f"{name}.json"

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self.artifacts[name] = str(file_path)

    def log_config(self, config: Dict[str, Any]) -> None:
        for key, value in config.items():
            self.log_param(key, value)

    def save_best_params(
        self, best_params: dict, export_json: bool = True, export_yaml: bool = True
    ) -> None:
        """Save best model parameters in user-friendly formats."""
        if not self.current_run_dir:
            return

        serializable_params = make_serializable(best_params)  # Using shared function

        if export_json:
            json_path = self.current_run_dir / "best_model_params.json"
            with open(json_path, "w") as f:
                json.dump(serializable_params, f, indent=2, ensure_ascii=False)

        if export_yaml:
            yaml_data = {
                "best_parameters": serializable_params,
                "usage_example": {
                    "description": "Load these parameters with **params syntax",
                    "python_code": [
                        "import yaml",
                        "with open('best_model_params.yaml', 'r') as f:",
                        "    data = yaml.safe_load(f)",
                        "    best_params = data['best_parameters']",
                        "",
                        "# Use with your estimator:",
                        "# model = XGBClassifier(**best_params)",
                        "# model = LGBMRegressor(**best_params)",
                        "# etc.",
                    ],
                },
            }

            yaml_path = self.current_run_dir / "best_model_params.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(
                    yaml_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )

    def save_tuning_summary(
        self, summary: dict, export_json: bool = True, export_yaml: bool = True
    ) -> None:
        """Save comprehensive tuning summary."""
        if not self.current_run_dir:
            return

        serializable_summary = make_serializable(summary)  # Using shared function

        if export_json:
            json_path = self.current_run_dir / "tuning_summary.json"
            with open(json_path, "w") as f:
                json.dump(serializable_summary, f, indent=2, ensure_ascii=False)

        if export_yaml:
            yaml_path = self.current_run_dir / "tuning_summary.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(
                    serializable_summary,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )


class NoOpLogger(BaseTuningLogger):
    """No-operation logger that does nothing."""

    def start_run(self, run_name: str = None) -> None:
        if TEST_LOGGER:
            print(f"Starting run: {run_name}")
        pass

    def end_run(self) -> None:
        if TEST_LOGGER:
            print("Ending run")
        pass

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if TEST_LOGGER:
            print(f"Logging metric: {key} = {value}")
        pass

    def log_param(self, key: str, value: Any) -> None:
        if TEST_LOGGER:
            print(f"Logging param: {key} = {value}")
        pass

    def log_artifact(self, data: Any, name: str) -> None:
        if TEST_LOGGER:
            print(f"Logging artifact: {name}")
        pass

    def log_config(self, config: Dict[str, Any]) -> None:
        if TEST_LOGGER:
            print(f"Logging config: {config}")
        pass
