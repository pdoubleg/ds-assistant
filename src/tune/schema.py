import ast
import yaml
from dataclasses import asdict, dataclass
from typing import Any, Callable, Union

import numpy as np
import optuna
import pandas as pd
import scipy
from pydantic import BaseModel, Field, field_validator
from sklearn.base import BaseEstimator


class PythonCode(BaseModel):
    """A valid python code block and its reasoning."""

    reasoning: str = Field(description="LLM explanation")
    code: str = Field(description="A Python code block")

    @field_validator("code", mode="after")
    def is_syntax_valid(cls, v: Any) -> str:
        """Validate that the code has valid Python syntax."""
        try:
            ast.parse(v, mode="exec")
            return v
        except SyntaxError as e:
            raise ValueError(f"Code cannot be compiled: {e}")

    @field_validator("code", mode="after")
    def is_executable(cls, v: Any) -> str:
        """Validate that the code is executable."""
        safe_globals = {"scipy": scipy, "optuna": optuna, "np": np}
        try:
            exec(v, safe_globals)
            return v
        except Exception as e:
            raise ValueError(f"Code is not executable: {e}")

    @property
    def code_markdown(self) -> str:
        """Return code formatted as markdown."""
        return f"```python\n{self.code}\n```"


class AnalysisAndRecommendations(BaseModel):
    """A summary analysis and recommendations for downstream HPO."""

    domain_analysis: str = Field(
        description="Domain-level insights and HPO related recommendations"
    )
    dataset_analysis: str = Field(
        description="Data-driven insights and HPO related recommendations"
    )


@dataclass
class AutoMLDependencies:
    """Dependencies for AutoML agents."""

    dataset: pd.DataFrame
    target_column: str
    estimator_type: str | None = None
    custom_estimator: BaseEstimator | None = None
    use_dataset_analysis: bool = True
    task_type: str = "auto"  # "classification", "regression", or "auto"
    verbose: int = 2  # Verbosity level for display control
    metric: Union[str, Callable, None] = None
    direction: str = "maximize"


@dataclass
class AutoTunerConfig:
    """Configuration class for AutoTuner."""

    # Data settings
    data_path: str | None = None
    target_column: str | None = None
    output_directory: str = "llm_tuning_output"

    # General settings
    random_state: int = 42
    verbose: int = (
        1  # 0: start/end only, 1: minimal, 2: current default, 3: includes LLM prompts
    )
    n_jobs: int = -1

    # ML Model settings
    task_type: str = "auto"
    metric: str = "accuracy"
    estimator_type: str | None = None
    custom_estimator: BaseEstimator | None = None

    # Cross-validation settings
    cv_folds: int = 5
    n_repeats: int = 2
    test_size: float = 0.2
    stratify: bool = True

    # Optuna settings
    n_trials: int = 100
    max_iterations: int = 5
    max_no_improve: int = 3
    top_n_configs: int = 5
    sampler: str = "TPESampler"
    pruner: str = "MedianPruner"
    direction: str = "maximize"

    # LLM settings
    model: str = "gpt-4.1-mini"
    use_dataset_analysis: bool = True

    # Task settings
    task_description: str = ""

    # Display settings
    show_progress_bar: bool = True
    max_table_rows: int = 20
    decimal_precision: int = 4

    # File logging settings
    enable_file_logging: bool = True
    output_directory: str = "tunning_logs"
    export_json: bool = True
    export_yaml: bool = True
    save_tuning_summary: bool = True

    # MLflow settings
    enable_mlflow: bool = True
    experiment_name: str = "autotuner_optimization"
    log_agent_iterations: bool = True
    log_trial_details: bool = False
    artifact_logging: bool = True

    @classmethod
    def from_yaml(cls, config_path: str) -> "AutoTunerConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested config
        flat_config = {}

        # Data settings
        if "data" in config_dict:
            data_config = config_dict["data"]
            flat_config["data_path"] = data_config.get("data_path")
            flat_config["target_column"] = data_config.get("target_column")

        # General settings
        if "general" in config_dict:
            general_config = config_dict["general"]
            flat_config["random_state"] = general_config.get("random_state", 42)
            flat_config["verbose"] = general_config.get("verbose", 1)
            flat_config["n_jobs"] = general_config.get("n_jobs", -1)

        # ML Model settings
        if "ml_model" in config_dict:
            ml_model_config = config_dict["ml_model"]
            flat_config["task_type"] = ml_model_config.get("task_type", "auto")
            flat_config["metric"] = ml_model_config.get("metric", "accuracy")
            flat_config["estimator_type"] = ml_model_config.get(
                "estimator_type", "xgboost"
            )

        # Cross-validation settings
        if "cross_validation" in config_dict:
            cv_config = config_dict["cross_validation"]
            flat_config["cv_folds"] = cv_config.get("cv_folds", 5)
            flat_config["n_repeats"] = cv_config.get("n_repeats", 2)
            flat_config["test_size"] = cv_config.get("test_size", 0.2)
            flat_config["stratify"] = cv_config.get("stratify", True)

        # Optuna settings
        if "optuna" in config_dict:
            optuna_config = config_dict["optuna"]
            flat_config["n_trials"] = optuna_config.get("n_trials", 100)
            flat_config["max_iterations"] = optuna_config.get("max_iterations", 5)
            flat_config["max_no_improve"] = optuna_config.get("max_no_improve", 3)
            flat_config["top_n_configs"] = optuna_config.get("top_n_configs", 5)
            flat_config["sampler"] = optuna_config.get("sampler", "TPESampler")
            flat_config["pruner"] = optuna_config.get("pruner", "MedianPruner")
            flat_config["direction"] = optuna_config.get("direction", "maximize")

        # LLM settings
        if "llm" in config_dict:
            llm_config = config_dict["llm"]
            flat_config["model"] = llm_config.get("model", "gpt-4.1-mini")
            flat_config["use_dataset_analysis"] = llm_config.get(
                "use_dataset_analysis", True
            )

        # Task description
        if "task" in config_dict and "description" in config_dict["task"]:
            flat_config["task_description"] = config_dict["task"]["description"]

        # Display settings
        if "display" in config_dict:
            display_config = config_dict["display"]
            flat_config["show_progress_bar"] = display_config.get(
                "show_progress_bar", True
            )
            flat_config["max_table_rows"] = display_config.get("max_table_rows", 20)
            flat_config["decimal_precision"] = display_config.get(
                "decimal_precision", 4
            )

        # File logging settings
        if "file_logging" in config_dict:
            file_logging_config = config_dict["file_logging"]
            flat_config["enable_file_logging"] = file_logging_config.get(
                "enable_file_logging", False
            )
            flat_config["output_directory"] = file_logging_config.get(
                "output_directory", "tunning_logs"
            )
            flat_config["export_json"] = file_logging_config.get("export_json", False)
            flat_config["export_yaml"] = file_logging_config.get("export_yaml", False)
            flat_config["save_tuning_summary"] = file_logging_config.get(
                "save_tuning_summary", False
            )

        # MLflow settings
        if "mlflow" in config_dict:
            mlflow_config = config_dict["mlflow"]
            flat_config["enable_mlflow"] = mlflow_config.get("enable_mlflow", True)
            flat_config["experiment_name"] = mlflow_config.get(
                "experiment_name", "autotuner_optimization"
            )
            flat_config["log_agent_iterations"] = mlflow_config.get(
                "log_agent_iterations", True
            )
            flat_config["log_trial_details"] = mlflow_config.get(
                "log_trial_details", False
            )
            flat_config["artifact_logging"] = mlflow_config.get(
                "artifact_logging", True
            )

        return cls(**flat_config)


@dataclass
class HPOProfile:
    # Basic shape
    n_samples: int
    n_features: int

    # Dtype breakdown
    num_numeric: int
    num_categorical: int
    num_boolean: int
    num_datetime: int

    # Global rates
    frac_missing_overall: float
    frac_zero_overall_numeric: (
        float | None
    )  # mean zero rate over numeric cols, None if no numeric

    # Sparsity proxy & categorical rarity
    sparsity_proxy: (
        float  # higher => sparser (combines numeric density & rare-category rate)
    )
    rare_category_rate_mean: (
        float  # average fraction of rare categories across categorical cols (<1% freq)
    )

    # Correlation snapshot (numeric only)
    corr_num_used: int  # number of numeric columns used for corr
    corr_median_abs: float | None
    corr_q90_abs: float | None
    corr_max_abs: float | None
    corr_top_pairs: list[dict[str, str | float]]  # only when mode='thorough'

    # Feature scale dispersion (numeric)
    feature_scale_cov: float

    # Cardinality (categorical)
    categorical_cardinality: dict[str, int]
    avg_categorical_cardinality: float | None
    high_cardinality_columns: list[
        str
    ]  # heuristic: cardinality > min(100, 0.1 * n_samples)

    # Missingness by column (top-N)
    top_missing_columns: list[tuple[str, float]]

    # Zero-inflation by column (numeric, top-N)
    top_zero_fraction_numeric: list[tuple[str, float]]

    # Target facts
    task: str
    target_binary_multiclass: dict[str, Any] | None
    target_regression: dict[str, Any] | None

    def render_markdown_facts(self, max_list_items: int = 10) -> str:
        """Concise, fact-only Markdown suitable for LLM input or a report."""
        lines: list[str] = []
        lines.append("# Dataset Facts Characteristics")
        lines.append("")
        lines.append("## **Shape & Types**")
        lines.append(f"- Samples: {self.n_samples:,}")
        lines.append(f"- Features: {self.n_features:,}")
        lines.append(
            f"- Numeric: {self.num_numeric} | Categorical: {self.num_categorical} | Boolean: {self.num_boolean} | Datetime: {self.num_datetime}"
        )
        lines.append("")
        lines.append("## **Missingness & Sparsity**")
        lines.append(f"- Overall missing fraction: {self.frac_missing_overall:.4f}")
        if self.frac_zero_overall_numeric is not None:
            lines.append(
                f"- Mean zero fraction (numeric columns): {self.frac_zero_overall_numeric:.4f}"
            )
        lines.append(
            f"- Sparsity proxy (0 dense → 1 sparse): {self.sparsity_proxy:.3f}"
        )
        lines.append(
            f"- Mean rare-category rate (<1% freq across categoricals): {self.rare_category_rate_mean:.4f}"
        )
        if self.top_missing_columns:
            lines.append("- Top columns by missing fraction:")
            for col, rate in self.top_missing_columns[:max_list_items]:
                lines.append(f"  - {col}: {rate:.4f}")
        if self.top_zero_fraction_numeric:
            lines.append("- Top numeric columns by zero fraction:")
            for col, rate in self.top_zero_fraction_numeric[:max_list_items]:
                lines.append(f"  - {col}: {rate:.4f}")
        lines.append("")
        lines.append("## **Correlation Snapshot (numeric)**")
        lines.append(f"- Numeric columns used for correlation: {self.corr_num_used}")
        if self.corr_median_abs is not None:
            lines.append(f"- Median |corr|: {self.corr_median_abs:.4f}")
            lines.append(f"- 90th percentile |corr|: {self.corr_q90_abs:.4f}")
            lines.append(f"- Max |corr|: {self.corr_max_abs:.4f}")
        if self.corr_top_pairs:
            lines.append("- Top correlated pairs (|corr|):")
            for row in self.corr_top_pairs[:max_list_items]:
                lines.append(
                    f"  - {row['feature_a']} ↔ {row['feature_b']}: {row['abs_corr']:.4f}"
                )
        lines.append("")
        lines.append("## **Feature Scale Dispersion (numeric)**")
        lines.append(
            f"- CoV of per-feature std (higher = more varied scales): {self.feature_scale_cov:.4f}"
        )
        lines.append("")
        lines.append("## **Categorical Cardinality**")
        lines.append(
            f"- Average cardinality: {self.avg_categorical_cardinality if self.avg_categorical_cardinality is not None else 'NA'}"
        )
        if self.categorical_cardinality:
            # show top-k highest
            top_card = sorted(
                self.categorical_cardinality.items(), key=lambda kv: kv[1], reverse=True
            )[:max_list_items]
            lines.append("- Highest-cardinality categorical columns:")
            for col, k in top_card:
                lines.append(f"  - {col}: {k}")
        if self.high_cardinality_columns:
            lines.append(
                f"- High-cardinality columns (cardinality > min(100, 0.1*n_samples)): {len(self.high_cardinality_columns)}"
            )
            for col in self.high_cardinality_columns[:max_list_items]:
                lines.append(f"  - {col}")
        lines.append("")
        lines.append("## **Target**")
        lines.append(f"- Task: {self.task}")
        if self.target_binary_multiclass is not None:
            t = self.target_binary_multiclass
            lines.append(f"- Classes: {t.get('num_classes')}")
            lines.append(
                "- Class counts: "
                + ", ".join(
                    [f"{k}={v}" for k, v in t.get("class_frequencies", {}).items()]
                )
            )
            lines.append(
                "- Class probabilities: "
                + ", ".join(
                    [f"{k}={v:.4f}" for k, v in t.get("class_probs", {}).items()]
                )
            )
            lines.append(
                f"- Minority class fraction: {t.get('minority_class_fraction'):.4f}"
            )
            if t.get("target_entropy_bits") is not None:
                lines.append(f"- Target entropy (bits): {t['target_entropy_bits']:.4f}")
        if self.target_regression is not None:
            t = self.target_regression
            lines.append(f"- Non-null count: {t['count_non_null']}")
            lines.append(
                f"- Mean={t['mean']:.6f} | Std={t['std']:.6f} | Skew={t['skew']:.6f} | Kurtosis={t['kurtosis']:.6f}"
            )
            lines.append(f"- Outlier rate (|z|>3): {t['outlier_rate_std_gt_3']:.4f}")
            lines.append(
                f"- Percentiles: p01={t['p01']:.6f}, p05={t['p05']:.6f}, p50={t['p50']:.6f}, p95={t['p95']:.6f}, p99={t['p99']:.6f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict)."""
        return asdict(self)
