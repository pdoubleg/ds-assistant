"""
This module provides a class for automated hyperparameter tuning with MLflow integration.

It supports both classification and regression tasks, and can work with most scikit-learn compatible models.

The tuner uses LLM agents to analyze the dataset, generate initial search spaces, and refine them based on
optimization results. It supports multiple iterations of tuning with automated refinement between iterations.

"""

import os
import warnings
from dataclasses import asdict
from logging import getLogger
from typing import Any, Callable, Union

from sklearn.base import BaseEstimator

try:
    import mlflow  # type: ignore

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

import numpy as np
import optuna
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)

from .agents import (
    get_analysis_and_recommendations_agent,
    get_initial_search_space_agent,
    get_refine_search_space_agent,
)
from .prompts import (
    get_analysis_and_recommendations_prompt,
    get_initial_search_space_prompt,
    get_refine_search_space_prompt,
)
from .schema import AutoMLDependencies, AutoTunerConfig
from .utils import (
    Markdown,
    extract_logs_from_study,
    get_model_pipeline,
)
from .evaluation import (
    get_comprehensive_metrics,
    get_default_metric,
    get_scorer_smart,
    get_cv,
)
from .logging import BaseTuningLogger, MLflowLogger, LocalFileLogger, NoOpLogger

# Removes warnings in the current job
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs
os.environ["PYTHONWARNINGS"] = "ignore"

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

logger = getLogger(__name__)

console = Console()


class MLflowAutoTuner:
    """An automated hyperparameter tuning class that integrates with MLflow for experiment tracking.

    This class provides automated hyperparameter optimization for machine learning models using Optuna,
    with built-in MLflow integration for experiment tracking. It supports both classification and regression tasks,
    and can work with most scikit-learn compatible models.

    The tuner uses LLM agents to analyze the dataset, generate initial search spaces, and refine them based on
    optimization results. It supports multiple iterations of tuning with automated refinement between iterations.

    Args:
        dataset (pd.DataFrame): The input dataset containing features and target.
        config_path (str | None): Path to YAML configuration file. Defaults to None.
        config (AutoTunerConfig | None): Configuration object. Defaults to None.
        metric (Union[str, Callable, None]): Metric to optimize. Can be string name or callable. Defaults to None.
        experiment_name (str | None): Name for MLflow experiment. Defaults to None.
        custom_estimator (BaseEstimator | None): Custom sklearn-compatible estimator. Defaults to None.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification()
        >>> df = pd.DataFrame(X)
        >>> df['target'] = y
        >>> tuner = MLflowAutoTuner(
        ...     dataset=df,
        ...     config_path='config.yaml',
        ...     metric='accuracy',
        ...     experiment_name='classification_experiment'
        ... )
        >>> results = tuner.tune(max_iterations=5)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        config_path: str | None = None,
        config: AutoTunerConfig | None = None,
        metric: Union[str, Callable, None] = None,
        experiment_name: str | None = None,
        custom_estimator: BaseEstimator | None = None,
    ):
        # Load base AutoTuner configuration
        if config_path and config is None:
            self.config = AutoTunerConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            self.config = AutoTunerConfig()

        if not MLFLOW_AVAILABLE and self.config.enable_mlflow:
            console.print("MLflow not available, disabling MLflow logging", style="red")
            self.config.enable_mlflow = False

        self.logger = self._create_default_logger()

        # Set data and task parameters directly
        self.dataset = dataset
        self.metric = metric or self.config.metric
        self.experiment_name = experiment_name
        self.custom_estimator = custom_estimator
        # Set up agents
        self.analysis_and_recommendations_agent = (
            get_analysis_and_recommendations_agent(self.config.model)
        )
        self.initial_search_space_agent = get_initial_search_space_agent(
            self.config.model
        )
        self.refine_search_space_agent = get_refine_search_space_agent(
            self.config.model
        )

        # Auto-detect task type if needed
        if (
            self.config.task_type == "auto"
            and self.dataset is not None
            and self.config.target_column is not None
        ):
            self.task_type = self._detect_task_type(
                self.dataset[self.config.target_column]
            )
        else:
            self.task_type = self.config.task_type

        # Initialize tracking variables
        self.best_configs: list[dict] = []
        self.last_values: list[float] = []
        self.studies: list[optuna.Study] = []
        self.baseline_metrics: dict | None = None
        self.best_test_metrics: dict | None = None
        self.agent_results: list[dict] = []

        # Initialize dependencies for LLM agents
        self.deps = AutoMLDependencies(
            dataset=self.dataset,
            target_column=self.config.target_column,
            estimator_type=self.config.estimator_type,
            use_dataset_analysis=self.config.use_dataset_analysis,
            task_type=self.task_type,
            verbose=self.config.verbose,
            metric=metric,
            custom_estimator=self.custom_estimator,
            direction=self.config.direction,
        )

    def tune(self, max_iterations: int = None, user_description: str = None) -> dict:
        """Main tuning method with clean logging integration."""

        # Start logging run
        self.logger.start_run("autotuner_optimization")

        try:
            # Log configuration
            self.logger.log_config(asdict(self.config))

            # Core tuning logic (mostly unchanged)
            results = self._run_tuning_process(max_iterations, user_description)

            # Log final results
            self._log_final_results(results)

            return results

        finally:
            # Always end the logging run
            self.logger.end_run()

    def _run_tuning_process(
        self,
        max_iterations: int | None = None,
        user_description: str | None = None,
    ) -> dict:
        """
        Main function to run the automated tuning process with MLflow logging.

        Args:
            max_iterations (int, optional): Maximum number of iterations.
            user_description (str, optional): Custom task description overriding config.

        Returns:
            dict: Tuning results including best configuration and metrics.
        """
        if self.dataset is None or self.config.target_column is None:
            raise ValueError("Dataset and target must be provided for tuning")

        if max_iterations is None:
            max_iterations = self.config.max_iterations

        if user_description is not None:
            self.config.task_description = user_description

        # Log configuration
        config_dict = asdict(self.config)
        self.logger.log_config(config_dict)
        self.logger.log_param("task_type", self.config.task_type)
        self.logger.log_param("model_name", self.config.estimator_type)

        if self.config.verbose >= 1:
            # Display the actual metric being used (including smart defaults)
            if isinstance(self.metric, str):
                metric_display = self.metric
            elif callable(self.metric):
                metric_display = "Custom Function"
            else:
                metric_display = (
                    f"{get_default_metric(self.config.task_type)} (default)"
                )

            console.print(
                Panel.fit(
                    f"[bold blue]🚀 MLflow AutoTuner - Starting Optimization[/bold blue]\n"
                    f"[cyan]Task Type:[/cyan] {self.config.task_type.title()}\n"
                    f"[cyan]Model:[/cyan] {self.config.estimator_type}\n"
                    f"[cyan]Metric:[/cyan] {metric_display} ({self.config.direction})\n"
                    f"[cyan]Max Iterations:[/cyan] {max_iterations}\n"
                    f"[cyan]Dataset Analysis:[/cyan] {self.config.use_dataset_analysis}\n"
                    f"[cyan]Verbose:[/cyan] {self.config.verbose}\n"
                    f"[cyan]LLM:[/cyan] {self.config.model}",
                    title="Configuration",
                    border_style="blue",
                )
            )

        # Evaluate baseline
        self.baseline_metrics = self.evaluate_model(final=False, run_label="baseline")

        # Run analysis and recommendations
        if self.config.verbose >= 1:
            console.print(
                "🧠 Getting LLM analysis and recommendations...", style="blue"
            )

        prompt = get_analysis_and_recommendations_prompt(
            self.config.task_description,
            self.config.estimator_type,
            self.config.task_type,
            self.metric if isinstance(self.metric, str) else None,
        )

        # Display prompt for verbosity level 3
        self._display_llm_prompt(prompt, "Analysis and Recommendations")

        # Temporarily disable dataset for LLM if configured
        if not self.config.use_dataset_analysis:
            self.deps.dataset = None

        analysis = self.analysis_and_recommendations_agent.run_sync(
            prompt, deps=self.deps, model_settings=dict(parallel_tool_calls=False)
        )

        # Reset dataset
        if not self.config.use_dataset_analysis:
            self.deps.dataset = self.dataset

        # Log analysis results
        analysis_result = {
            "domain_analysis": analysis.output.domain_analysis,
            "dataset_analysis": analysis.output.dataset_analysis,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self.logger.log_artifact(analysis_result, "analysis")

        if self.config.verbose >= 2:
            console.print(
                Panel(
                    f"[yellow]Domain Analysis:[/yellow]\n{analysis.output.domain_analysis}\n\n"
                    f"[yellow]Dataset Analysis:[/yellow]\n{analysis.output.dataset_analysis}",
                    title="LLM Analysis",
                    border_style="yellow",
                )
            )
        elif self.config.verbose >= 1:
            console.print("✓ Analysis complete")

        # Generate initial search space
        if self.config.verbose >= 1:
            console.print("🔍 Generating initial search space...", style="blue")

        analysis_text = f"Domain Analysis: {analysis.output.domain_analysis}\n\nDataset Analysis: {analysis.output.dataset_analysis}"

        estimator = self._build_estimator()

        init_prompt = get_initial_search_space_prompt(
            self.config.estimator_type,
            self.task_type,
            analysis_text,
            estimator=estimator,
        )
        # Display prompt for verbosity level 3
        self._display_llm_prompt(init_prompt, "Initial Search Space")

        # Get initial search space
        init_sc = self.initial_search_space_agent.run_sync(
            init_prompt,
            deps=self.deps,
            message_history=analysis.all_messages(),
            usage=analysis.usage(),
        )

        # Log initial search space
        initial_result = {
            "iteration": 0,
            "agent_type": "initial_search_space",
            "reasoning": init_sc.output.reasoning,
            "code": init_sc.output.code,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self.logger.log_agent_iteration(0, "initial_search_space", initial_result)
        self.agent_results.append(initial_result)

        # Display initial search space results based on verbosity
        if self.config.verbose >= 2:
            reasoning_markdown = Markdown(init_sc.output.reasoning)
            console.print(
                Panel(
                    reasoning_markdown,
                    title="Initial Search Space",
                    border_style="green",
                )
            )
            md = Markdown(init_sc.output.code_markdown, code_theme="github-dark")
            console.print(Panel(md, title="Generated Code", border_style="green"))
        elif self.config.verbose >= 1:
            console.print("✓ Initial search space generated")

        current_code = init_sc.output.code
        last_history = init_sc.all_messages()
        self.usage = init_sc.usage()
        no_improve = 0

        # Main optimization loop
        for iteration in range(max_iterations):
            if self.config.verbose >= 1:
                console.print(
                    f"\n🔄 [bold]Starting Iteration {iteration + 1}/{max_iterations}[/bold]"
                )

            # Generate search space function from the generated code
            define_fn = self.generate_search_space_from_code(current_code)

            # Run optimization using Optuna
            study = self._run_optimization(define_fn, iteration + 1)
            self.studies.append(study)

            # Extract results
            top_summary, best_val = extract_logs_from_study(
                study, top_n=self.config.top_n_configs
            )

            # Display results based on verbosity
            if self.config.verbose >= 2:
                table = self._create_results_table(top_summary, iteration + 1)
                console.print(table)

            if self.config.verbose >= 1:
                console.print(
                    f"[bold green]Best value this iteration:[/bold green] {best_val:.{self.config.decimal_precision}f}"
                )

            # Add progress summary
            if self.config.verbose >= 1:
                # For the first iteration, compare against baseline; otherwise compare against previous best
                if not self.best_configs and self.baseline_metrics:
                    # First iteration: compare against baseline
                    current_best = self.baseline_metrics["cv_mean"]
                    change = (
                        best_val - current_best
                        if self.config.direction == "maximize"
                        else current_best - best_val
                    )
                    comparison_label = "vs Baseline"
                elif self.best_configs:
                    # Subsequent iterations: compare against previous best HPO result
                    current_best = self.best_configs[0]["score"]
                    change = best_val - current_best
                    comparison_label = "vs Previous Best"
                else:
                    # Fallback (no baseline available)
                    current_best = best_val
                    change = 0.0
                    comparison_label = "First Run"

                console.print(
                    f"📊 Score: {best_val:.4f} | {comparison_label}: {current_best:.4f} | Change: {change:+.4f}"
                )

            # Update best configurations
            improved = self._update_best(best_val, study.best_params, iteration + 1)
            self.last_values.append(best_val)

            # Calculate improvement
            improvement = 0
            if self.baseline_metrics:
                if self.config.direction == "maximize":
                    improvement = best_val - self.baseline_metrics["cv_mean"]
                else:
                    improvement = self.baseline_metrics["cv_mean"] - best_val

            # Log iteration results
            iteration_result = {
                "iteration": iteration + 1,
                "agent_type": "optimization",
                "best_score": best_val,
                "best_params": study.best_params,
                "improvement": improvement,
                "improved": improved,
                "top_trials": top_summary,
            }
            self.logger.log_agent_iteration(
                iteration + 1, "optimization", iteration_result
            )
            self.agent_results.append(iteration_result)

            if self.config.verbose >= 1:
                console.print(
                    f"[bold green]Best value this iteration:[/bold green] {best_val:.{self.config.decimal_precision}f}"
                )

            if improved:
                if self.config.verbose >= 1:
                    console.print(
                        "✅ [bold green]New best configuration found![/bold green]"
                    )
                no_improve = 0
            else:
                no_improve += 1
                if self.config.verbose >= 1:
                    console.print(
                        f"❌ [bold red]No improvement ({no_improve}/{self.config.max_no_improve})[/bold red]"
                    )

            # Check early stopping
            if no_improve >= self.config.max_no_improve:
                if self.config.verbose >= 1:
                    console.print(
                        "🛑 [bold yellow]Early stopping triggered[/bold yellow]"
                    )
                break

            if iteration + 1 == max_iterations:
                if self.config.verbose >= 1:
                    console.print(
                        "🏁 [bold blue]Reached maximum iterations[/bold blue]"
                    )
                break

            # Generate refinement prompt
            all_time = "\n".join(
                f"Iteration {e['iteration']}: score={e['score']:.4f}, params={e['config']}"
                for e in self.best_configs
            )

            refine_prompt = get_refine_search_space_prompt(
                top_n=top_summary,
                last_value=best_val,
                best_value=self.best_configs[0]["score"],
                estimator_type=self.config.estimator_type,
                task_type=self.config.task_type,
                all_time_configs=all_time,
                iteration=iteration,
                max_iterations=max_iterations,
                estimator=estimator,
            )
            # Display refinement prompt for verbosity level 3
            self._display_llm_prompt(
                refine_prompt,
                f"Search Space Refinement (Iteration {iteration + 1})",
            )

            # Get refined search space
            if self.config.verbose >= 1:
                console.print("🔧 Refining search space...", style="blue")

            ref_sc = self.refine_search_space_agent.run_sync(
                refine_prompt,
                message_history=last_history,
                deps=self.deps,
                usage=self.usage,
            )

            # Log refinement results
            refinement_result = {
                "iteration": iteration + 1,
                "agent_type": "refinement",
                "reasoning": ref_sc.output.reasoning,
                "code": ref_sc.output.code,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
            self.logger.log_agent_iteration(
                iteration + 1, "refinement", refinement_result
            )
            self.agent_results.append(refinement_result)

            if self.config.verbose >= 2:
                reasoning_markdown = Markdown(ref_sc.output.reasoning)
                console.print(
                    Panel(
                        reasoning_markdown,
                        title="Search Space Refinement",
                        border_style="cyan",
                    )
                )
                md = Markdown(ref_sc.output.code_markdown, code_theme="github-dark")
                console.print(Panel(md, title="Updated Code", border_style="cyan"))
            elif self.config.verbose >= 1:
                console.print("✓ Search space refined")

            current_code = ref_sc.output.code
            last_history = ref_sc.all_messages()
            self.usage = ref_sc.usage()

        # Final summary and logging
        final_results = self._create_final_summary()
        # Add best model evaluation on test set
        if self.best_configs:
            best_metrics = self.evaluate_model(
                final=True, run_label="best", params=self.best_configs[0]["config"]
            )
            final_results["best_test_metrics"] = best_metrics
            self.best_test_metrics = best_metrics

        self.agent_results.append(final_results)
        self._log_final_results(final_results)
        self._save_results_via_logger()

        if self.config.verbose >= 1:
            self._print_final_summary(final_results)

        return final_results

    def generate_search_space_from_code(
        self, code: str
    ) -> Callable[[optuna.trial.Trial], dict]:
        """Execute LLM code and return the define_search_space function."""

        local_ns: dict[str, Any] = {"optuna": optuna, "np": np}
        exec(code, local_ns)
        return local_ns["define_search_space"]

    def evaluate_model(
        self,
        final: bool,
        run_label: str,
        params: dict[str, Any] = None,
    ) -> dict[str, float]:
        """Evaluate model performance using either baseline or final evaluation strategy.

        For baseline evaluation (final=False), performs cross-validation on training data
        and scores on test set using default parameters. For final evaluation (final=True),
        trains on full training set using best/provided parameters and evaluates on test set.

        Args:
            final: Whether to do final evaluation with best parameters (True) or baseline
                  evaluation with defaults (False)
            run_label: Label prefix for logging metrics ("baseline" or "best")
            params: Optional hyperparameters to use. If None and final=True, uses
                   best parameters from optimization

        Returns:
            Dictionary of evaluation metrics. For baseline includes CV and test metrics.
            For final includes comprehensive test set metrics.
        """
        # 1) Make or reuse a stable split so baseline and final are directly comparable
        X = self.dataset.drop(columns=[self.config.target_column])
        y = self.dataset[self.config.target_column]

        X_train, X_test, y_train, y_test = self._get_or_make_split(X, y)

        # 2) Build estimator
        if final:
            # Expect best params (or provided params) for final evaluation
            best_params = params
            if best_params is None:
                if not getattr(self, "best_configs", None):
                    if getattr(self.config, "verbose", 0) >= 1:
                        console.print(
                            "⚠️ No best configuration available for evaluation",
                            style="yellow",
                        )
                    return {}
                best_params = self.best_configs[0]["config"]

            estimator = self._build_estimator(model_hyperparameters=best_params)
        else:
            # Baseline uses defaults
            estimator = self._build_estimator(model_hyperparameters=None)

        # 3) Scorer and CV (only used for baseline)
        scorer = get_scorer_smart(self.metric, self.config.task_type)

        # 4) Train/evaluate
        if not final:
            # Baseline path: CV on training fold, then fit on all train and score test
            cv = get_cv(
                self.config.task_type,
                self.config.cv_folds,
                self.config.n_repeats,
                self.config.stratify,
                self.config.random_state,
            )
            cv_scores = cross_val_score(
                estimator,
                X_train,
                y_train,
                cv=cv,
                scoring=scorer,
                n_jobs=self.config.n_jobs,
            )
            estimator.fit(X_train, y_train)
            test_score = scorer(estimator, X_test, y_test)

            metrics = {
                "cv_mean": float(np.mean(cv_scores)),
                "cv_std": float(np.std(cv_scores)),
                "test_score": float(test_score),
            }

            # Logging
            for k, v in metrics.items():
                self.logger.log_metric(f"{run_label}_{k}", v)

            if self.config.verbose >= 1:
                console.print(
                    f"📊 Baseline - CV: {metrics['cv_mean']:.{self.config.decimal_precision}f} "
                    f"(±{metrics['cv_std']:.{self.config.decimal_precision}f}), "
                    f"Test: {metrics['test_score']:.{self.config.decimal_precision}f}",
                    style="yellow",
                )
            return metrics

        else:
            # Final path: fit once on train, compute comprehensive metrics on test
            if self.config.verbose >= 1:
                console.print("🏆 Evaluating best model on test set...", style="green")

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)

            y_pred_proba = None
            if self.config.task_type == "classification" and hasattr(
                estimator, "predict_proba"
            ):
                try:
                    y_pred_proba = estimator.predict_proba(X_test)
                except Exception as e:
                    if self.config.verbose >= 1:
                        console.print(
                            f"Error predicting probabilities: {e}", style="red"
                        )

            comp = get_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            metrics = {f"test_{k}": float(v) for k, v in comp.items()}

            # Logging
            for k, v in metrics.items():
                self.logger.log_metric(f"{run_label}_{k}", v)

            if self.config.verbose >= 1:
                console.print("🏆 Best Model Test Set Performance:")
                for k, v in comp.items():
                    console.print(f"   {k}: {v:.{self.config.decimal_precision}f}")

            return metrics

    def _create_default_logger(self) -> BaseTuningLogger:
        if self.config.enable_mlflow:
            return MLflowLogger(self.config.experiment_name)
        elif self.config.enable_file_logging:
            return LocalFileLogger(self.config.output_directory)
        else:
            return NoOpLogger()

    def _build_estimator(self, model_hyperparameters: dict[str, Any] | None = None):
        return get_model_pipeline(
            model_type=self.config.estimator_type,
            task_type=self.config.task_type,
            custom_estimator=self.custom_estimator,
            model_hyperparameters=model_hyperparameters,
        )

    def _get_or_make_split(self, X, y):
        """
        Cache/resuse indices so all evaluations share the same train/test split.
        """
        if not hasattr(self, "_cached_split_indices"):
            strat = (
                y
                if (self.config.task_type == "classification" and self.config.stratify)
                else None
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=strat,
            )
            # Cache indices (safer than caching arrays if X is large)
            self._cached_split_indices = {
                "train_idx": X_train.index,
                "test_idx": X_test.index,
            }
            return X_train, X_test, y_train, y_test
        else:
            tr = self._cached_split_indices["train_idx"]
            te = self._cached_split_indices["test_idx"]
            return X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]

    def _run_optimization(
        self, define_search_space: Callable, iteration: int
    ) -> optuna.Study:
        """Run a single optimization iteration."""

        X = self.dataset.drop(columns=[self.config.target_column])
        y = self.dataset[self.config.target_column]

        # Get scorer using smart default handling
        scorer = get_scorer_smart(self.metric, self.config.task_type)

        def objective(trial: optuna.trial.Trial) -> float:
            model_hyperparameters = define_search_space(trial)

            model = self._build_estimator(model_hyperparameters)

            # Cross-validation
            cv = get_cv(
                self.config.task_type,
                self.config.cv_folds,
                self.config.n_repeats,
                self.config.stratify,
                self.config.random_state,
            )

            scores = cross_val_score(
                model, X, y, scoring=scorer, cv=cv, n_jobs=self.config.n_jobs
            )
            return float(np.mean(scores))

        # Create study
        sampler_map = {
            "TPESampler": optuna.samplers.TPESampler(seed=self.config.random_state),
            "RandomSampler": optuna.samplers.RandomSampler(
                seed=self.config.random_state
            ),
            "CmaEsSampler": optuna.samplers.CmaEsSampler(seed=self.config.random_state),
        }

        pruner_map = {
            "MedianPruner": optuna.pruners.MedianPruner(),
            "HyperbandPruner": optuna.pruners.HyperbandPruner(),
            "null": None,
            None: None,
        }

        study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler_map.get(
                self.config.sampler,
                optuna.samplers.TPESampler(seed=self.config.random_state),
            ),
            pruner=pruner_map.get(self.config.pruner),
        )

        # Run optimization with progress bar
        if self.config.show_progress_bar and self.config.verbose >= 2:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Iteration {iteration} Optimization", total=self.config.n_trials
                )

                def callback(study, trial):
                    progress.update(task, advance=1)

                study.optimize(
                    objective,
                    n_trials=self.config.n_trials,
                    n_jobs=self.config.n_jobs,
                    callbacks=[callback],
                )
        else:
            study.optimize(
                objective, n_trials=self.config.n_trials, n_jobs=self.config.n_jobs
            )

        return study

    def _detect_task_type(self) -> str:
        """Automatically detect if the task is classification or regression."""
        if (
            self.dataset[self.config.target_column].dtype == "object"
            or len(self.dataset[self.config.target_column].unique()) <= 20
        ):
            return "classification"
        else:
            return "regression"

    def _update_best(self, value: float, params: dict, iteration: int) -> bool:
        """Update the list of best configurations."""
        previous_best_score = (
            self.best_configs[0]["score"]
            if self.best_configs
            else (
                float("-inf") if self.config.direction == "maximize" else float("inf")
            )
        )

        entry = {"score": value, "config": params.copy(), "iteration": iteration}
        self.best_configs.append(entry)

        reverse_sort = self.config.direction == "maximize"
        self.best_configs.sort(key=lambda x: x["score"], reverse=reverse_sort)
        self.best_configs = self.best_configs[: self.config.top_n_configs]

        if self.config.direction == "maximize":
            return value > previous_best_score
        else:
            return value < previous_best_score

    def _create_final_summary(self) -> dict:
        """Create comprehensive final summary including all tuning results."""
        summary = {
            "config": asdict(self.config),
            "task_type": self.config.task_type,
            "model_name": self.config.estimator_type,
            "metric": self.metric
            if isinstance(self.metric, str)
            else (
                get_default_metric(self.config.task_type)
                if self.metric is None
                else "custom"
            ),
            "best_score": self.best_configs[0]["score"] if self.best_configs else None,
            "best_config": self.best_configs[0]["config"]
            if self.best_configs
            else None,
            "top_configs": self.best_configs.copy(),
            "iterations": len(self.last_values),
            "progression": self.last_values.copy(),
            "baseline_metrics": self.baseline_metrics.copy()
            if self.baseline_metrics
            else None,
            "best_test_metrics": self.best_test_metrics.copy()
            if self.best_test_metrics
            else None,
            "agent_results": self.agent_results.copy(),
            "improvement": None,
            "api_calls": self.usage.requests,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.input_tokens + self.usage.output_tokens,
        }

        if self.baseline_metrics and self.best_configs:
            summary["improvement"] = (
                self.best_configs[0]["score"] - self.baseline_metrics["cv_mean"]
            )

        return summary

    def _log_final_results(self, results: dict) -> None:
        """Log final results using the pluggable logger."""

        if results.get("best_score"):
            self.logger.log_metric("final_best_score", results["best_score"])

        if results.get("improvement"):
            self.logger.log_metric("final_improvement", results["improvement"])

        # Log comprehensive results as artifact
        self.logger.log_artifact(results, "final_results")
        self.logger.log_metric("total_iterations", results["iterations"])
        # Log best parameters
        if results.get("best_config"):
            self.logger.log_artifact(results["best_config"], "best_parameters")

    def _create_results_table(self, top_trials: str, iteration: int) -> Table:
        """Create a rich table showing trial results for the given iteration."""
        table = Table(
            title=f"Iteration {iteration} - Top Trials",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Score", justify="right")
        table.add_column("Parameters", style="cyan")

        lines = top_trials.split("\n")
        for i, line in enumerate(lines[: self.config.max_table_rows], 1):
            if line.strip():
                parts = line.split(": ")
                if len(parts) >= 2:
                    score_part = parts[0].split("=")[-1].rstrip(")")
                    params_part = parts[1] if len(parts) > 1 else ""
                    table.add_row(str(i), score_part, params_part)

        return table

    def _display_llm_prompt(self, prompt: str, title: str) -> None:
        """Display LLM prompt if verbosity level is 3."""
        if self.config.verbose >= 3:
            console.print(
                Panel(prompt, title=f"🤖 LLM Prompt: {title}", border_style="magenta")
            )

    def _print_final_summary(self, final_results: dict):
        """Print final tuning summary."""
        if self.config.verbose == 0:
            # Minimal output for verbosity 0
            if self.best_configs:
                best = self.best_configs[0]
                console.print(
                    f"✓ AutoTuner complete. Best score: {best['score']:.{self.config.decimal_precision}f}"
                )
            else:
                console.print("✓ AutoTuner complete. No valid configurations found.")
            return

        console.print("\n" + "=" * 60)
        console.print(
            "[bold green]🎉 TUNING COMPLETE 🎉[/bold green]", justify="center"
        )
        console.print("=" * 60)

        if self.best_configs:
            best = self.best_configs[0]

            # Best configuration panel
            if self.config.verbose >= 2:
                console.print(
                    Panel(
                        f"[bold green]Score:[/bold green] {best['score']:.{self.config.decimal_precision}f}\n"
                        f"[bold green]Iteration:[/bold green] {best['iteration']}\n"
                        f"[bold green]Parameters:[/bold green]\n"
                        + "\n".join(
                            [f"  • {k}: {v}" for k, v in best["config"].items()]
                        ),
                        title="🏆 Best Configuration",
                        border_style="gold1",
                    )
                )
            elif self.config.verbose >= 1:
                console.print(
                    f"🏆 Best score: {best['score']:.{self.config.decimal_precision}f} (iteration {best['iteration']})"
                )
            if self.config.verbose >= 1:
                console.print(
                    Panel(
                        f"[green]Usage Summary:[/green]\n"
                        f"[cyan]API Calls:[/cyan] {final_results['api_calls']}\n"
                        f"[cyan]Input Tokens:[/cyan] {final_results['input_tokens']}\n"
                        f"[cyan]Output Tokens:[/cyan] {final_results['output_tokens']}\n"
                        f"[cyan]Total Tokens:[/cyan] {final_results['total_tokens']}",
                        title="Usage Summary",
                        border_style="green",
                    )
                )

            # Progress chart
            if len(self.last_values) > 1 and self.config.verbose >= 2:
                progress_text = "Progress: " + " → ".join(
                    [f"{v:.{self.config.decimal_precision}f}" for v in self.last_values]
                )
                console.print(f"[dim]{progress_text}[/dim]")

        # Improvement comparison if baseline available
        if self.baseline_metrics and self.config.verbose >= 1:
            best_score = self.best_configs[0]["score"] if self.best_configs else None
            if best_score is not None:
                # Calculate improvement based on optimization direction
                if self.config.direction == "maximize":
                    improvement = best_score - self.baseline_metrics["cv_mean"]
                else:  # minimize
                    improvement = self.baseline_metrics["cv_mean"] - best_score

                improvement_pct = (
                    improvement / abs(self.baseline_metrics["cv_mean"])
                ) * 100

                # Determine if improvement is good (always positive improvement is good now)
                is_improvement = improvement > 0
                color = "green" if is_improvement else "red"

                if self.config.verbose >= 2:
                    console.print(
                        Panel(
                            f"[blue]Baseline CV Score:[/blue] {self.baseline_metrics['cv_mean']:.{self.config.decimal_precision}f}\n"
                            f"[green]Final CV Score:[/green] {best_score:.{self.config.decimal_precision}f}\n"
                            f"[bold {color}]Improvement:[/bold {color}] "
                            f"{improvement:+.{self.config.decimal_precision}f} ({improvement_pct:+.2f}%)",
                            title="📈 Performance Improvement",
                            border_style="blue",
                        )
                    )
                else:
                    console.print(
                        f"📈 Improvement: {improvement:+.{self.config.decimal_precision}f} ({improvement_pct:+.2f}%)"
                    )

    def _save_results_via_logger(self):
        """Save results using the pluggable logger system."""

        if not self.best_configs:
            return

        # Save best parameters (simple JSON for **params usage)
        best_params = self.best_configs[0]["config"]
        self.logger.save_best_params(
            best_params,
            export_json=getattr(self.config, "export_json", True),
            export_yaml=getattr(self.config, "export_yaml", True),
        )

        # Save comprehensive tuning summary
        if self.config.save_tuning_summary:
            summary = self.get_tuning_summary()
            self.logger.save_tuning_summary(
                summary,
                export_json=getattr(self.config, "export_json", True),
                export_yaml=getattr(self.config, "export_yaml", True),
            )

        if self.config.verbose >= 1:
            console.print("✅ Results saved successfully!")

    def get_best_config(self) -> dict | None:
        """Get the best hyperparameter configuration."""
        return self.best_configs[0]["config"] if self.best_configs else None

    def get_best_configs(self, n: int | None = None) -> list[dict]:
        """Get the top N best configurations."""
        return self.best_configs if n is None else self.best_configs[:n]

    def get_tuning_summary(self) -> dict:
        """Get a comprehensive summary of the tuning process."""
        return self._create_final_summary()


def run_mlflow_tuning(
    dataset: pd.DataFrame,
    config_path: str = "src/tune/config.yml",
    metric: str | None = None,
    experiment_name: str | None = None,
    **kwargs,
) -> dict:
    """
    Convenience function to run MLflow tuning with minimal setup.

    Args:
        dataset (pd.DataFrame): Training dataset
        config_path (str): Path to configuration file
        metric (str, optional): Scoring metric
        experiment_name (str, optional): MLflow experiment name
        **kwargs: Additional tuner arguments

    Returns:
        Dict: Tuning results
    """
    tuner = MLflowAutoTuner(
        dataset=dataset,
        config_path=config_path,
        metric=metric,
        experiment_name=experiment_name,
        **kwargs,
    )

    return tuner.tune()
