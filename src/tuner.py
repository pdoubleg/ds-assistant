"""
AutoTuner: A generalized ML model tuner with CLI interface.

This module provides a command-line interface for automated hyperparameter tuning
of machine learning models using Optuna optimization with LLM-guided search space
generation. Supports both classification and regression tasks with any sklearn-compatible
estimator.

Key Features:
- CLI interface with rich console output
- YAML configuration for all parameters
- Support for classification and regression
- LLM-guided search space optimization
- Comprehensive result reporting
- Model persistence and result saving

Author: Auto-generated from tune_optuna.py
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC, SVR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.tune.prompts import (
        get_analysis_and_recommendations_prompt,
        get_initial_search_space_prompt,
        get_refine_search_space_prompt,
    )

    from .tune.agents import (
        get_analysis_and_recommendations_agent,
        get_initial_search_space_agent,
        get_refine_search_space_agent,
    )
    from .tune.schema import AutoMLDependencies, AutoTunerConfig
    from .tune.utils import (
        extract_logs_from_study,
        get_scorer_from_string,
    )
    from .tune.tools import generate_search_space_from_code
except ModuleNotFoundError:
    # Fallback when running as script
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from src.tune.prompts import (
        get_analysis_and_recommendations_prompt,
        get_initial_search_space_prompt,
        get_refine_search_space_prompt,
    )

    from .tune.agents import (
        get_analysis_and_recommendations_agent,
        get_initial_search_space_agent,
        get_refine_search_space_agent,
    )
    from .tune.schema import AutoMLDependencies, AutoTunerConfig
    from .tune.utils import (
        extract_logs_from_study,
        get_scorer_from_string,
    )
# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

# Initialize rich console
console = Console()


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def detect_task_type(y: pd.Series) -> str:
    """Automatically detect if the task is classification or regression."""
    if y.dtype == "object" or len(y.unique()) <= 20:
        return "classification"
    else:
        return "regression"


def get_default_estimator(
    estimator_type: str, task_type: str, random_state: int = 42
) -> BaseEstimator:
    """Get a default estimator based on type and task."""
    if estimator_type == "xgboost":
        if task_type == "classification":
            return xgb.XGBClassifier(
                random_state=random_state,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                use_label_encoder=False,
                verbosity=0,
            )
        else:
            return xgb.XGBRegressor(
                random_state=random_state,
                objective="reg:squarederror",
                n_jobs=-1,
                verbosity=0,
            )
    elif estimator_type == "lightgbm":
        if task_type == "classification":
            return lgb.LGBMClassifier(
                random_state=random_state,
                objective="binary",
                metric="binary_logloss",
                n_jobs=-1,
                verbosity=-1,
                force_row_wise=True,
            )
        else:
            return lgb.LGBMRegressor(
                random_state=random_state,
                objective="regression",
                metric="rmse",
                n_jobs=-1,
                verbosity=-1,
                force_row_wise=True,
            )
    elif estimator_type == "random_forest":
        if task_type == "classification":
            return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        else:
            return RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif estimator_type == "svm":
        if task_type == "classification":
            return SVC(random_state=random_state, probability=True)
        else:
            return SVR()
    elif estimator_type == "logistic_regression" and task_type == "classification":
        return LogisticRegression(random_state=random_state, n_jobs=-1)
    elif estimator_type == "linear_regression" and task_type == "regression":
        return LinearRegression(n_jobs=-1)
    else:
        raise ValueError(
            f"Unsupported estimator_type '{estimator_type}' for task_type '{task_type}'"
        )


# -----------------------------------------------------------------------------
# Main AutoTuner Class
# -----------------------------------------------------------------------------


class AutoTuner:
    """
    A generalized machine learning model tuner with CLI interface.

    This class provides automated hyperparameter tuning for both classification and regression
    tasks using Optuna optimization with LLM-guided search space generation. It supports any
    sklearn-compatible estimator and provides a rich command-line interface.

    Args:
        config_path (str, optional): Path to YAML configuration file. Defaults to None.
        config (AutoTunerConfig, optional): Configuration object. Defaults to None.
        dataset (pd.DataFrame, optional): Training dataset. Defaults to None.
        target (str, optional): Name of target column. Defaults to None.
        estimator (BaseEstimator, optional): Custom estimator to tune. Defaults to None.
        estimator_type (str, optional): Type of estimator ("xgboost", "lightgbm", etc.). Defaults to "auto".
        task_type (str, optional): Task type ("classification", "regression", "auto"). Defaults to "auto".
        metric (str or Callable, optional): Scoring metric for optimization. Defaults to None.

    Example:
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestClassifier
        >>>
        >>> # Initialize with dataset
        >>> tuner = AutoTuner(
        ...     dataset=df,
        ...     target="target_column",
        ...     estimator_type="random_forest",
        ...     metric="accuracy"
        ... )
        >>>
        >>> # Run tuning
        >>> tuner.tune()
        >>>
        >>> # Get results
        >>> best_config = tuner.get_best_config()
        >>> summary = tuner.get_tuning_summary()
    """

    def __init__(
        self,
        config_path: str | None = None,
        config: AutoTunerConfig | None = None,
        dataset: pd.DataFrame | None = None,
        target_column: str | None = None,
        estimator: BaseEstimator | None = None,
        estimator_type: str = "auto",
        task_type: str = "auto",
        metric: str | Callable | None = None,
    ):
        # Load configuration
        if config_path:
            self.config = AutoTunerConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            # Use default config
            self.config = AutoTunerConfig()

        # Set data and task parameters
        self.dataset = dataset
        self.target_column = target_column
        self.custom_estimator = estimator
        self.estimator_type = estimator_type
        self.task_type = task_type
        self.metric = metric
        self.usage = None

        # Set agents
        self.initial_search_space_agent = get_initial_search_space_agent(
            self.config.llm.model
        )
        self.refine_search_space_agent = get_refine_search_space_agent(
            self.config.llm.model
        )
        self.analysis_and_recommendations_agent = (
            get_analysis_and_recommendations_agent(self.config.llm.model)
        )

        # Auto-detect task type if needed
        if (
            self.task_type == "auto"
            and self.dataset is not None
            and self.target_column is not None
        ):
            self.task_type = detect_task_type(self.dataset[self.target_column])

        # Auto-detect estimator type if needed
        if self.estimator_type == "auto" and self.custom_estimator is not None:
            estimator_name = type(self.custom_estimator).__name__.lower()
            if "xgb" in estimator_name:
                self.estimator_type = "xgboost"
            elif "lgbm" in estimator_name or "lightgbm" in estimator_name:
                self.estimator_type = "lightgbm"
            elif "randomforest" in estimator_name:
                self.estimator_type = "random_forest"
            elif (
                "svm" in estimator_name
                or "svc" in estimator_name
                or "svr" in estimator_name
            ):
                self.estimator_type = "svm"
            elif "logistic" in estimator_name:
                self.estimator_type = "logistic_regression"
            elif "linear" in estimator_name:
                self.estimator_type = "linear_regression"
            else:
                self.estimator_type = "custom"

        # Initialize tracking variables
        self.best_configs: list[dict] = []
        self.last_values: list[float] = []
        self.studies: list[optuna.Study] = []
        self.baseline_metrics: dict | None = None
        self.final_metrics: dict | None = None

        # Store verbose level for easy access
        self.verbose = self.config.verbose

        # Initialize dependencies for LLM agents
        self.deps = AutoMLDependencies(
            dataset=self.dataset,
            target_column=self.target_column,
            estimator_type=self.estimator_type,
            custom_estimator=self.custom_estimator,
            use_dataset_analysis=self.config.use_dataset_analysis,
            task_type=self.task_type,
        )
        # Add verbose level to deps for LLM tools
        self.deps.verbose = self.verbose

    def _print_header(self):
        """Print a fancy header for the tuning process."""
        if self.verbose >= 1:
            console.print(
                Panel.fit(
                    "[bold blue]🚀 AutoTuner - Automated ML Hyperparameter Optimization[/bold blue]\n"
                    f"[cyan]Task Type:[/cyan] {self.task_type.title()}\n"
                    f"[cyan]Estimator:[/cyan] {self.estimator_type.title()}\n"
                    f"[cyan]Metric:[/cyan] {self.metric if isinstance(self.metric, str) else 'Custom Function'}\n"
                    f"[cyan]Max Iterations:[/cyan] {self.config.max_iterations}",
                    title="Configuration",
                    border_style="blue",
                )
            )
        elif self.verbose == 0:
            console.print("🚀 Starting AutoTuner optimization...")

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

    def _evaluate_baseline(self) -> dict[str, float]:
        """Evaluate baseline model performance."""
        if self.verbose >= 1:
            console.print("📊 Evaluating baseline model performance...", style="yellow")

        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
            if self.config.stratify and self.task_type == "classification"
            else None,
        )

        # Get default estimator
        if self.custom_estimator is not None:
            estimator = self.custom_estimator
        else:
            estimator = get_default_estimator(
                self.estimator_type, self.task_type, self.config.random_state
            )

        # Fit and evaluate
        estimator.fit(X_train, y_train)

        # Get scorer
        if isinstance(self.metric, str):
            scorer = get_scorer_from_string(self.metric, self.task_type)
        else:
            scorer = self.metric

        # Cross-validation score
        if self.task_type == "classification" and self.config.stratify:
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

        cv_scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scorer,
            n_jobs=self.config.n_jobs,
        )

        # Test score
        test_score = scorer(estimator, X_test, y_test)

        metrics = {
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "test_score": float(test_score),
        }

        # Display baseline results
        if self.verbose >= 2:
            console.print(
                Panel(
                    f"[green]CV Score:[/green] {metrics['cv_mean']:.{self.config.decimal_precision}f} "
                    f"(±{metrics['cv_std']:.{self.config.decimal_precision}f})\n"
                    f"[green]Test Score:[/green] {metrics['test_score']:.{self.config.decimal_precision}f}",
                    title="Baseline Performance",
                    border_style="green",
                )
            )
        elif self.verbose >= 1:
            console.print(
                f"📊 Baseline CV: {metrics['cv_mean']:.{self.config.decimal_precision}f} (±{metrics['cv_std']:.{self.config.decimal_precision}f})"
            )

        return metrics

    def _display_llm_prompt(self, prompt: str, title: str) -> None:
        """Display LLM prompt if verbosity level is 3."""
        if self.verbose >= 3:
            console.print(
                Panel(prompt, title=f"🤖 LLM Prompt: {title}", border_style="magenta")
            )

    def _run_optimization(
        self, define_search_space: Callable, iteration: int
    ) -> optuna.Study:
        """Run a single optimization iteration."""
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        # Get scorer
        if isinstance(self.metric, str):
            scorer = get_scorer_from_string(self.metric, self.task_type)
        else:
            scorer = self.metric

        def objective(trial: optuna.trial.Trial) -> float:
            params = define_search_space(trial)

            # Get estimator
            if self.custom_estimator is not None:
                model = self.custom_estimator
            else:
                model = get_default_estimator(
                    self.estimator_type, self.task_type, self.config.random_state
                )

            # Set parameters
            model.set_params(**params)

            # Cross-validation
            if self.task_type == "classification" and self.config.stratify:
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
        if self.config.show_progress_bar:
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

    def tune(
        self,
        max_iterations: int | None = None,
        user_description: str | None = None,
    ) -> None:
        """
        Run the automated tuning process.

        Args:
            max_iterations (int, optional): Maximum number of iterations. Defaults to config value.
            user_description (str, optional): Custom task description. Defaults to None.
        """
        if self.dataset is None or self.target_column is None:
            raise ValueError("Dataset and target must be provided for tuning")

        if max_iterations is None:
            max_iterations = self.config.max_iterations

        if user_description is not None:
            self.config.task_description = user_description

        # Print header
        self._print_header()

        # Evaluate baseline
        self.baseline_metrics = self._evaluate_baseline()

        # Run analysis and recommendations
        if self.verbose >= 1:
            console.print(
                "🧠 Getting LLM analysis and recommendations...", style="blue"
            )

        prompt = get_analysis_and_recommendations_prompt(
            self.config.task_description,
            self.estimator_type,
            self.task_type,
            self.metric if isinstance(self.metric, str) else None,
        )

        # Display prompt for verbosity level 3
        self._display_llm_prompt(prompt, "Analysis and Recommendations")

        # Temporarily disable dataset for LLM if configured
        if not self.config.use_dataset_analysis:
            self.deps.dataset = None

        analysis = self.analysis_and_recommendations_agent.run_sync(
            prompt, deps=self.deps
        )

        # Reset dataset
        if not self.config.use_dataset_analysis:
            self.deps.dataset = self.dataset

        # Display analysis results based on verbosity
        if self.verbose >= 2:
            console.print(
                Panel(
                    f"[yellow]Domain Analysis:[/yellow]\n{analysis.output.domain_analysis}\n\n"
                    f"[yellow]Dataset Analysis:[/yellow]\n{analysis.output.dataset_analysis}",
                    title="LLM Analysis",
                    border_style="yellow",
                )
            )
        elif self.verbose >= 1:
            console.print("✓ Analysis complete")

        # Generate initial search space
        if self.verbose >= 1:
            console.print("🔍 Generating initial search space...", style="blue")

        # Combine analysis insights for the initial search space
        analysis_text = f"Domain Analysis: {analysis.output.domain_analysis}\n\nDataset Analysis: {analysis.output.dataset_analysis}"

        init_prompt = get_initial_search_space_prompt(
            self.estimator_type, self.task_type, analysis_text
        )

        # Display prompt for verbosity level 3
        self._display_llm_prompt(init_prompt, "Initial Search Space")

        init_sc = self.initial_search_space_agent.run_sync(
            init_prompt,
            deps=self.deps,
            message_history=analysis.all_messages(),
            usage=analysis.usage(),
        )

        # Display initial search space results based on verbosity
        if self.verbose >= 2:
            console.print(
                Panel(
                    f"[green]Reasoning:[/green]\n{init_sc.output.reasoning}\n\n"
                    f"[green]Code:[/green]\n{init_sc.output.code_markdown}",
                    title="Initial Search Space",
                    border_style="green",
                )
            )
        elif self.verbose >= 1:
            console.print("✓ Initial search space generated")

        current_code = init_sc.output.code
        self.usage = init_sc.usage()
        last_history = init_sc.all_messages()

        no_improve = 0

        # Main optimization loop
        for iteration in range(max_iterations):
            if self.verbose >= 1:
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
            if self.verbose >= 2:
                table = self._create_results_table(top_summary, iteration + 1)
                console.print(table)

            if self.verbose >= 1:
                console.print(
                    f"[bold green]Best value this iteration:[/bold green] {best_val:.{self.config.decimal_precision}f}"
                )

            # Update best configurations
            improved = self._update_best(best_val, study.best_params, iteration + 1)
            self.last_values.append(best_val)

            if improved:
                if self.verbose >= 1:
                    console.print(
                        "✅ [bold green]New best configuration found![/bold green]"
                    )
                no_improve = 0
            else:
                no_improve += 1
                if self.verbose >= 1:
                    console.print(
                        f"❌ [bold red]No improvement ({no_improve}/{self.config.max_no_improve})[/bold red]"
                    )

            # Check early stopping
            if no_improve >= self.config.max_no_improve:
                if self.verbose >= 1:
                    console.print(
                        "🛑 [bold yellow]Early stopping triggered[/bold yellow]"
                    )
                break

            if iteration + 1 == max_iterations:
                if self.verbose >= 1:
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
                estimator_type=self.estimator_type,
                task_type=self.task_type,
                all_time_configs=all_time,
                iteration=iteration,
                max_iterations=max_iterations,
            )

            # Get refined search space
            if self.verbose >= 1:
                console.print("🔧 Refining search space...", style="blue")

            # Display refinement prompt for verbosity level 3
            self._display_llm_prompt(
                refine_prompt, f"Search Space Refinement (Iteration {iteration + 1})"
            )

            ref_sc = self.refine_search_space_agent.run_sync(
                refine_prompt,
                message_history=last_history,
                deps=self.deps,
                usage=self.usage,
            )
            # Display refinement results based on verbosity
            if self.verbose >= 2:
                console.print(
                    Panel(
                        f"[cyan]Refinement Reasoning:[/cyan]\n{ref_sc.output.reasoning}\n\n"
                        f"[cyan]Updated Code:[/cyan]\n{ref_sc.output.code_markdown}",
                        title="Search Space Refinement",
                        border_style="cyan",
                    )
                )
            elif self.verbose >= 1:
                console.print("✓ Search space refined")

            current_code = ref_sc.output.code
            last_history = ref_sc.all_messages()
            self.usage = ref_sc.usage()

        # Final summary
        self._print_final_summary()

        # Save results if configured
        if self.config.save_results:
            self._save_results()

    def _print_final_summary(self):
        """Print final tuning summary."""
        if self.verbose == 0:
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
            if self.verbose >= 2:
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
            elif self.verbose >= 1:
                console.print(
                    f"🏆 Best score: {best['score']:.{self.config.decimal_precision}f} (iteration {best['iteration']})"
                )

            # Progress chart
            if len(self.last_values) > 1 and self.verbose >= 2:
                progress_text = "Progress: " + " → ".join(
                    [f"{v:.{self.config.decimal_precision}f}" for v in self.last_values]
                )
                console.print(f"[dim]{progress_text}[/dim]")

        # Improvement comparison if baseline available
        if self.baseline_metrics and self.verbose >= 1:
            best_score = self.best_configs[0]["score"] if self.best_configs else None
            if best_score is not None:
                improvement = best_score - self.baseline_metrics["cv_mean"]
                improvement_pct = (
                    improvement / abs(self.baseline_metrics["cv_mean"])
                ) * 100

                if self.verbose >= 2:
                    console.print(
                        Panel(
                            f"[blue]Baseline CV Score:[/blue] {self.baseline_metrics['cv_mean']:.{self.config.decimal_precision}f}\n"
                            f"[green]Final CV Score:[/green] {best_score:.{self.config.decimal_precision}f}\n"
                            f"[bold {'green' if improvement > 0 else 'red'}]Improvement:[/bold {'green' if improvement > 0 else 'red'}] "
                            f"{improvement:+.{self.config.decimal_precision}f} ({improvement_pct:+.2f}%)",
                            title="📈 Performance Improvement",
                            border_style="blue",
                        )
                    )
                else:
                    console.print(
                        f"📈 Improvement: {improvement:+.{self.config.decimal_precision}f} ({improvement_pct:+.2f}%)"
                    )

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON/YAML serializable format.

        Args:
            obj: The object to make serializable

        Returns:
            A JSON/YAML serializable version of the object
        """
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Handle dataclass objects and other objects with __dict__
            if hasattr(obj, "__dataclass_fields__"):
                # It's a dataclass
                return self._make_serializable(asdict(obj))
            else:
                # Regular object with __dict__
                return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            # Fallback to string representation for unknown types
            return str(obj)

    def get_serializable_summary(self) -> dict:
        """Get a JSON/YAML serializable summary of the tuning process.

        Returns:
            A dictionary containing all tuning results in serializable format
        """
        summary = self.get_tuning_summary()
        return self._make_serializable(summary)

    def _save_results(self):
        """Save tuning results to files."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)

        if self.verbose >= 1:
            console.print(f"💾 Saving results to {results_dir}...", style="blue")

        # Save tuning summary
        summary = self.get_tuning_summary()
        with open(results_dir / "tuning_summary.pkl", "wb") as f:
            pickle.dump(summary, f)

        # Save human-readable formats
        if self.config.export_json or self.config.export_yaml:
            serializable_summary = self.get_serializable_summary()

            if self.config.export_json:
                with open(results_dir / "tuning_summary.json", "w") as f:
                    json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
                if self.verbose >= 2:
                    console.print("📄 Tuning summary saved as JSON", style="dim")

            if self.config.export_yaml:
                with open(results_dir / "tuning_summary.yaml", "w") as f:
                    yaml.dump(
                        serializable_summary,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        indent=2,
                    )
                if self.verbose >= 2:
                    console.print("📄 Tuning summary saved as YAML", style="dim")

        # Save best model parameters separately for easy loading
        if self.best_configs and (self.config.export_json or self.config.export_yaml):
            best_params = self._make_serializable(self.best_configs[0]["config"])

            if self.config.export_json:
                with open(results_dir / "best_model_params.json", "w") as f:
                    json.dump(best_params, f, indent=2, ensure_ascii=False)
                if self.verbose >= 2:
                    console.print("🎯 Best model parameters saved as JSON", style="dim")

            if self.config.export_yaml:
                params_with_example = {
                    "best_parameters": best_params,
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
                with open(results_dir / "best_model_params.yaml", "w") as f:
                    yaml.dump(
                        params_with_example,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        indent=2,
                    )
                if self.verbose >= 2:
                    console.print("🎯 Best model parameters saved as YAML", style="dim")

        # Save best model
        if self.config.save_best_model and self.best_configs:
            if self.custom_estimator is not None:
                best_model = self.custom_estimator
            else:
                best_model = get_default_estimator(
                    self.estimator_type, self.task_type, self.config.random_state
                )

            best_model.set_params(**self.best_configs[0]["config"])

            # Fit on full dataset
            X = self.dataset.drop(columns=[self.target])
            y = self.dataset[self.target]
            best_model.fit(X, y)

            with open(results_dir / "best_model.pkl", "wb") as f:
                pickle.dump(best_model, f)

        # Save studies
        if self.config.save_study:
            with open(results_dir / "optuna_studies.pkl", "wb") as f:
                pickle.dump(self.studies, f)

        if self.verbose >= 1:
            console.print("✅ Results saved successfully!", style="green")

    def get_best_config(self) -> dict | None:
        """Get the best hyperparameter configuration."""
        return self.best_configs[0]["config"] if self.best_configs else None

    def get_best_configs(self, n: int | None = None) -> list[dict]:
        """Get the top N best configurations."""
        return self.best_configs if n is None else self.best_configs[:n]

    def get_tuning_summary(self) -> dict:
        """Get a comprehensive summary of the tuning process."""
        return {
            "config": self.config,
            "task_type": self.task_type,
            "estimator_type": self.estimator_type,
            "metric": self.metric,
            "best_score": self.best_configs[0]["score"] if self.best_configs else None,
            "best_config": self.get_best_config(),
            "top_configs": self.best_configs.copy(),
            "iterations": len(self.last_values),
            "progression": self.last_values.copy(),
            "baseline_metrics": self.baseline_metrics.copy()
            if self.baseline_metrics
            else None,
            "final_metrics": self.final_metrics.copy() if self.final_metrics else None,
        }


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line interface parser."""
    parser = argparse.ArgumentParser(
        description="AutoTuner - Automated ML Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with CSV file
  python tuner.py --data data.csv --target target_col --estimator xgboost --metric accuracy
  
  # Using config file (set data.path and data.target in YAML)
  python tuner.py --config custom_config.yml
  
  # Override config data with CLI args
  python tuner.py --config custom_config.yml --data different_data.csv
  
  # Interactive mode
  python tuner.py --interactive
  
  # Verbosity levels
  python tuner.py --data data.csv --target target --verbose 0  # Minimal output
  python tuner.py --data data.csv --target target --verbose 3  # Full debug with LLM prompts
        """,
    )

    # Data arguments
    parser.add_argument(
        "--data", type=str, help="Path to CSV data file (overrides config)"
    )
    parser.add_argument(
        "--target", type=str, help="Name of target column (overrides config)"
    )

    # Model arguments
    parser.add_argument(
        "--estimator",
        type=str,
        default="auto",
        choices=[
            "auto",
            "xgboost",
            "lightgbm",
            "random_forest",
            "svm",
            "logistic_regression",
            "linear_regression",
        ],
        help="Type of estimator to tune",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Type of ML task",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Scoring metric (e.g., accuracy, f1, roc_auc, r2, mse)",
    )

    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        default="src/tune/config.yml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of tuning iterations",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None, help="Number of trials per iteration"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=start/end only, 1=minimal, 2=current default, 3=includes LLM prompts",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to files"
    )

    # Interactive mode
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    return parser


def interactive_setup() -> tuple[pd.DataFrame, str, str, str, str]:
    """Interactive setup for data and parameters."""
    console.print(
        Panel.fit(
            "[bold blue]🚀 AutoTuner Interactive Setup[/bold blue]", border_style="blue"
        )
    )

    # Get data file
    while True:
        data_path = Prompt.ask("📁 Enter path to your CSV data file")
        try:
            df = pd.read_csv(data_path)
            console.print(f"✅ Loaded dataset with shape: {df.shape}")
            break
        except Exception as e:
            console.print(f"❌ Error loading file: {e}", style="red")

    # Show columns and get target
    console.print("\n📊 Available columns:")
    for i, col in enumerate(df.columns, 1):
        console.print(f"  {i}. {col} ({df[col].dtype})")

    target = Prompt.ask("\n🎯 Enter target column name", choices=list(df.columns))

    # Auto-detect task type
    task_type = detect_task_type(df[target])
    console.print(f"🔍 Detected task type: [bold]{task_type}[/bold]")

    if not Confirm.ask(f"Use {task_type} task type?"):
        task_type = Prompt.ask(
            "Select task type", choices=["classification", "regression"]
        )

    # Get estimator type
    estimator_options = ["xgboost", "lightgbm", "random_forest", "svm"]
    if task_type == "classification":
        estimator_options.append("logistic_regression")
    else:
        estimator_options.append("linear_regression")

    estimator_type = Prompt.ask(
        "🤖 Select estimator type", choices=estimator_options, default="xgboost"
    )

    # Get metric
    if task_type == "classification":
        metric_options = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        default_metric = "accuracy"
    else:
        metric_options = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
        default_metric = "r2"

    metric = Prompt.ask(
        "📊 Select metric", choices=metric_options, default=default_metric
    )

    return df, target, task_type, estimator_type, metric


def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()

    try:
        # Load configuration first to check for data path
        config_path = args.config if os.path.exists(args.config) else None
        if config_path:
            config = AutoTunerConfig.from_yaml(config_path)
            console.print(f"✅ Loaded configuration from {config_path}")
        else:
            config = AutoTunerConfig()
            console.print("⚠️  Using default configuration")

        # Interactive mode
        if args.interactive:
            df, target, task_type, estimator_type, metric = interactive_setup()
        else:
            # Determine data source - CLI args override config
            data_path = args.data or config.data_path
            target = args.target or config.target_column

            # Validate required arguments
            if not data_path or not target:
                console.print(
                    "❌ Data path and target_column are required. Specify via:",
                    style="red",
                )
                console.print(
                    "   • CLI: --data <path> --target_column <column>", style="yellow"
                )
                console.print(
                    "   • Config: Set data.path and data.target_column in YAML",
                    style="yellow",
                )
                console.print("   • Interactive: --interactive", style="yellow")
                return

            # Load data
            try:
                df = pd.read_csv(data_path)
                console.print(
                    f"✅ Loaded dataset from {data_path} with shape: {df.shape}"
                )
            except Exception as e:
                console.print(
                    f"❌ Error loading data from {data_path}: {e}", style="red"
                )
                return

            task_type = args.task
            estimator_type = args.estimator
            metric = args.metric

        # Override config with CLI arguments
        if args.max_iterations:
            config.max_iterations = args.max_iterations
        if args.n_trials:
            config.n_trials = args.n_trials
        if args.output_dir:
            config.results_dir = args.output_dir
        if args.no_save:
            config.save_results = False
        if args.verbose is not None:
            config.verbose = args.verbose

        # Create tuner
        tuner = AutoTuner(
            config=config,
            dataset=df,
            target_column=target,
            estimator_type=estimator_type,
            task_type=task_type,
            metric=metric,
        )

        # Run tuning
        tuner.tune()

    except KeyboardInterrupt:
        console.print("\n👋 Tuning interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise


if __name__ == "__main__":
    main()
