import json
import os
import tempfile
import warnings
from dataclasses import asdict
from logging import getLogger
from typing import Any, Callable, Union

from sklearn.base import BaseEstimator

try:
    import mlflow

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
    RepeatedKFold,
    RepeatedStratifiedKFold,
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
from .tools import generate_search_space_from_code
from .utils import (
    Markdown,
    extract_logs_from_study,
    get_default_metric,
    get_model_pipeline,
    get_scorer_smart,
)

# Removes warnings in the current job
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs
os.environ["PYTHONWARNINGS"] = "ignore"

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

logger = getLogger(__name__)

console = Console()


class MLflowAutoTuner:
    """
    MLflow-integrated AutoTuner for hyperparameter optimization.

    Args:
        dataset (pd.DataFrame): Training dataset
        config_path (str, optional): Path to YAML configuration file
        metric (Union[str, Callable, None], optional): Scoring metric
        experiment_name (str, optional): MLflow experiment name
        custom_estimator (BaseEstimator, optional): Custom sklearn estimator
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
        """
        Initialize MLflowAutoTuner with simplified configuration.

        Args:
            dataset (pd.DataFrame): Training dataset
            config_path (str, optional): Path to YAML configuration file
            config (AutoTunerConfig, optional): AutoTuner configuration
            metric (Union[str, Callable, None], optional): Scoring metric
            experiment_name (str, optional): MLflow experiment name
            custom_estimator (BaseEstimator, optional): Custom sklearn estimator
        """
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
        self.agent_results: list[dict] = []  # Store agent iteration results

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

    def tune(
        self,
        max_iterations: int | None = None,
        user_description: str | None = None,
    ) -> dict:
        """
        Run the automated tuning process with MLflow logging.

        Args:
            max_iterations (int, optional): Maximum number of iterations.
            user_description (str, optional): Custom task description.

        Returns:
            dict: Comprehensive tuning results
        """
        if self.dataset is None or self.config.target_column is None:
            raise ValueError("Dataset and target must be provided for tuning")

        if max_iterations is None:
            max_iterations = self.config.max_iterations

        if user_description is not None:
            self.config.task_description = user_description

        # Set MLflow experiment if provided
        if (
            MLFLOW_AVAILABLE
            and self.config.enable_mlflow
            and self.config.experiment_name
        ):
            mlflow.set_experiment(self.experiment_name)

        # Use MLflow context if available, otherwise just run normally
        if MLFLOW_AVAILABLE and self.config.enable_mlflow:
            mlflow_context = mlflow.start_run(
                nested=True, run_name="autotuner_optimization"
            )
        else:
            from contextlib import nullcontext

            mlflow_context = nullcontext()

        with mlflow_context:
            # Log configuration
            config_dict = asdict(self.config)
            self._log_to_mlflow("config", config_dict)
            self._log_to_mlflow("task_type", self.config.task_type)
            self._log_to_mlflow("model_name", self.config.estimator_type)

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
            self.baseline_metrics = self._evaluate_baseline()

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
            }
            self._log_agent_iteration(0, "analysis", analysis_result)

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
            estimator = get_model_pipeline(
                model_type=self.config.estimator_type,
                task_type=self.config.task_type,
                custom_estimator=self.custom_estimator,
            )
            init_prompt = get_initial_search_space_prompt(
                self.config.estimator_type,
                self.task_type,
                analysis_text,
                estimator=estimator,
            )
            # Display prompt for verbosity level 3
            self._display_llm_prompt(init_prompt, "Initial Search Space")
            init_sc = self.initial_search_space_agent.run_sync(
                init_prompt,
                deps=self.deps,
                message_history=analysis.all_messages(),
                usage=analysis.usage(),
            )

            # Log initial search space
            initial_result = {
                "reasoning": init_sc.output.reasoning,
                "code": init_sc.output.code,
            }
            self._log_agent_iteration(0, "initial_search_space", initial_result)

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

                # Generate search space function
                define_fn = generate_search_space_from_code(current_code)

                # Run optimization
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

                # Add concise progress summary
                if self.config.verbose >= 1:
                    # For the first iteration, compare against baseline; otherwise compare against previous best
                    if not self.best_configs and self.baseline_metrics:
                        # First iteration: compare against baseline
                        current_best = self.baseline_metrics["cv_mean"]
                        change = best_val - current_best if self.config.direction == "maximize" else current_best - best_val
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
                    "best_score": best_val,
                    "best_params": study.best_params,
                    "improvement": improvement,
                    "improved": improved,
                    "top_trials": top_summary,
                }
                self._log_agent_iteration(
                    iteration + 1, "optimization", iteration_result
                )

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
                    "reasoning": ref_sc.output.reasoning,
                    "code": ref_sc.output.code,
                }
                self._log_agent_iteration(
                    iteration + 1, "refinement", refinement_result
                )

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
                best_metrics = self._evaluate_best()
                final_results["best_test_metrics"] = best_metrics
                self.best_test_metrics = best_metrics

            self._log_final_results(final_results)

            if self.config.verbose >= 1:
                self._print_final_summary(final_results)

            return final_results

    def _evaluate_baseline(self) -> dict[str, float]:
        """Evaluate baseline model performance with train/test split."""
        
        if self.config.verbose >= 1:
            console.print("📊 Evaluating baseline model performance...", style="yellow")

        X = self.dataset.drop(columns=[self.config.target_column])
        y = self.dataset[self.config.target_column]

        # Single train/test split for fair comparison with final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
            if self.config.task_type == "classification" and self.config.stratify
            else None,
        )

        # Use default parameters for baseline
        estimator = get_model_pipeline(
            model_type=self.config.estimator_type,
            task_type=self.config.task_type,
            custom_estimator=self.custom_estimator,
        )

        # Get scorer
        scorer = get_scorer_smart(self.metric, self.config.task_type)

        # CV on training set only
        if self.config.task_type == "classification" and self.config.stratify:
            cv = RepeatedStratifiedKFold(
                n_splits=self.config.cv_folds,
                n_repeats=self.config.n_repeats,
                random_state=self.config.random_state,
            )
        else:
            cv = RepeatedKFold(
                n_splits=self.config.cv_folds,
                n_repeats=self.config.n_repeats,
                random_state=self.config.random_state,
            )

        # CV scores on training data only
        cv_scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scorer,
            n_jobs=self.config.n_jobs,
        )

        # Fit on full training set and evaluate on test set
        estimator.fit(X_train, y_train)
        test_score = scorer(estimator, X_test, y_test)

        metrics = {
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "test_score": float(test_score),
        }

        # Log baseline metrics
        for metric_name, value in metrics.items():
            self._log_to_mlflow(f"baseline_{metric_name}", value)

        if self.config.verbose >= 1:
            console.print(
                f"📊 Baseline - CV: {metrics['cv_mean']:.{self.config.decimal_precision}f} (±{metrics['cv_std']:.{self.config.decimal_precision}f}), Test: {metrics['test_score']:.{self.config.decimal_precision}f}"
            )

        return metrics

    def _evaluate_best(self) -> dict[str, float]:
        """
        Evaluate the best model with comprehensive metrics on a test set.
        This provides the final, unbiased performance estimate.
        """
        if not self.best_configs:
            console.print(
                "⚠️ No best configuration available for evaluation", style="yellow"
            )
            return {}

        if self.config.verbose >= 1:
            console.print(
                "🏆 Evaluating best model on test set...", style="green"
            )

        X = self.dataset.drop(columns=[self.config.target_column])
        y = self.dataset[self.config.target_column]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
            if self.config.task_type == "classification" and self.config.stratify
            else None,
        )

        # Build best model
        best_params = self.best_configs[0]["config"]
        best_model = get_model_pipeline(
            model_hyperparameters=best_params,
            model_type=self.config.estimator_type,
            task_type=self.config.task_type,
            custom_estimator=self.custom_estimator,
        )

        # Fit and predict
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Get probabilities for classification
        y_pred_proba = None
        if self.config.task_type == "classification" and hasattr(
            best_model, "predict_proba"
        ):
            try:
                y_pred_proba = best_model.predict_proba(X_test)
            except Exception as e:
                console.print(f"Error predicting probabilities: {e}", style="red")
                pass

        # Get comprehensive metrics
        comprehensive_metrics = self._get_comprehensive_metrics(
            y_test, y_pred, y_pred_proba
        )

        # Add test set prefix
        metrics = {f"test_{k}": v for k, v in comprehensive_metrics.items()}

        # Log best model metrics to MLflow
        for metric_name, value in metrics.items():
            self._log_to_mlflow(f"best_{metric_name}", value)

        if self.config.verbose >= 1:
            console.print("🏆 Best Model Test Set Performance:")
            for metric_name, value in comprehensive_metrics.items():
                console.print(
                    f"   {metric_name}: {value:.{self.config.decimal_precision}f}"
                )

        return metrics

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

            model = get_model_pipeline(
                model_hyperparameters,
                model_type=self.config.estimator_type,
                task_type=self.config.task_type,
                custom_estimator=self.custom_estimator,
            )

            # Cross-validation
            if self.config.task_type == "classification" and self.config.stratify:
                cv = RepeatedStratifiedKFold(
                    n_splits=self.config.cv_folds,
                    n_repeats=self.config.n_repeats,
                    random_state=self.config.random_state,
                )
            else:
                cv = RepeatedKFold(
                    n_splits=self.config.cv_folds,
                    n_repeats=self.config.n_repeats,
                    random_state=self.config.random_state,
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

    def _log_to_mlflow(self, key: str, value: Any, step: int | None = None) -> None:
        """Log metrics, parameters, or artifacts to MLflow."""
        if not MLFLOW_AVAILABLE or not self.config.enable_mlflow:
            return

        try:
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
            elif isinstance(value, str):
                mlflow.log_param(key, value)
            elif isinstance(value, dict):
                # Log as JSON artifact
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(value, f, indent=2, default=str)
                    temp_path = f.name
                mlflow.log_artifact(temp_path, f"{key}.json")
                os.unlink(temp_path)
            else:
                mlflow.log_param(key, str(value))
        except Exception as e:
            logger.warning(f"Failed to log {key} to MLflow: {e}")

    def _log_agent_iteration(
        self, iteration: int, agent_type: str, result: dict
    ) -> None:
        """Log agent iteration results to MLflow."""
        iteration_data = {
            "iteration": iteration,
            "agent_type": agent_type,
            "timestamp": pd.Timestamp.now().isoformat(),
            "result": result,
        }

        self.agent_results.append(iteration_data)

        # Log as MLflow artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(iteration_data, f, indent=2, default=str)
            temp_path = f.name

        artifact_path = f"agent_iterations/iteration_{iteration}_{agent_type}.json"
        if MLFLOW_AVAILABLE and self.config.enable_mlflow:
            mlflow.log_artifact(temp_path, artifact_path)
            os.unlink(temp_path)

        # Log key metrics
        if "best_score" in result:
            self._log_to_mlflow(
                f"iteration_{iteration}_best_score",
                result["best_score"],
                step=iteration,
            )
        if "improvement" in result:
            self._log_to_mlflow(
                f"iteration_{iteration}_improvement",
                result["improvement"],
                step=iteration,
            )

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
        """Create comprehensive final summary."""
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
        """Log final results to MLflow as artifacts."""
        # Log final metrics
        if results["best_score"] is not None:
            self._log_to_mlflow("final_best_score", results["best_score"])
        if results["improvement"] is not None:
            self._log_to_mlflow("final_improvement", results["improvement"])

        self._log_to_mlflow("total_iterations", results["iterations"])

        # Save comprehensive results as artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f, indent=2, default=str)
            temp_path = f.name
        if MLFLOW_AVAILABLE and self.config.enable_mlflow:
            mlflow.log_artifact(temp_path, "final_results.json")
            os.unlink(temp_path)

        # Save best parameters as artifact
        if results["best_config"]:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(results["best_config"], f, indent=2, default=str)
                temp_path = f.name
            if MLFLOW_AVAILABLE and self.config.enable_mlflow:
                mlflow.log_artifact(temp_path, "best_parameters.json")
                os.unlink(temp_path)

    def _create_results_table(self, top_trials: str, iteration: int) -> Table:
        """Create a rich table showing trial results."""
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

    def _get_comprehensive_metrics(
        self, y_true, y_pred, y_pred_proba=None
    ) -> dict[str, float]:
        """
        Get comprehensive evaluation metrics based on task type.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_pred_proba: Predicted probabilities (for classification only)

        Returns:
            dict: Dictionary of metric names and values
        """
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            log_loss,
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
        )

        metrics = {}

        if self.config.task_type == "classification":
            # Core classification metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["balanced_accuracy"] = float(
                balanced_accuracy_score(y_true, y_pred)
            )

            # Handle multiclass vs binary
            average_method = "binary" if len(np.unique(y_true)) == 2 else "weighted"

            metrics["precision"] = float(
                precision_score(y_true, y_pred, average=average_method, zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, average=average_method, zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_true, y_pred, average=average_method, zero_division=0)
            )

            # Probability-based metrics (if available)
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_true, y_pred_proba[:, 1])
                        )
                        metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                    else:
                        # Multiclass
                        metrics["roc_auc"] = float(
                            roc_auc_score(
                                y_true,
                                y_pred_proba,
                                multi_class="ovr",
                                average="weighted",
                            )
                        )
                        metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                except (ValueError, IndexError):
                    # Skip if probabilities are not compatible
                    pass

        else:  # regression
            metrics["r2"] = float(r2_score(y_true, y_pred))
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

            # Avoid division by zero for MAPE
            if not np.any(y_true == 0):
                metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))

        return metrics

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
