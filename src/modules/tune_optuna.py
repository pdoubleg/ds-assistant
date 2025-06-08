"""
AutoTuneLLM (Optuna edition):
A generalized ML model tuner that supports XGBoost, LightGBM, and custom sklearn estimators,
powered by LLM-generated Optuna search-space functions.
"""

import ast
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import scipy
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.tools import ToolDefinition
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, make_scorer, precision_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split

from src.utils import exploratory_data_analysis, format_eda_for_llm, metric_ppv

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="lightgbm")
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ["PYTHONWARNINGS"] = "ignore"

OPENAI_MODEL = OpenAIModel("gpt-4.1")

# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior data scientist tasked with guiding the use of an AutoML tool to discover \
the best model configurations for a given binary classification dataset. Your role involves \
understanding the dataset characteristics (when available), proposing suitable hyperparameters \
and their search spaces, analyzing results, and iterating on configurations.
"""

DEFAULT_TASK_DESCRIPTION = """\
The classification problem under investigation is based on insurance claims data. \
More specifically, the goal is to predict whether a given claim will be high severity. \
Ultimately, we are interested in optimizing for PPV (Positive Predictive Value) at the top 5% of predicted probabilities.
"""


def get_analysis_and_recommendations_prompt(
    task_description: str = DEFAULT_TASK_DESCRIPTION, estimator_type: str = "xgboost"
) -> str:
    return f"""{task_description}
For this specific inquiry, you are tasked with supporting hyperparameter optimization for a {estimator_type} model.
Given the problem context and dataset characteristics (when available), provide analysis and recommendations
to guide downstream iterative search space exploration."
""".strip()


def get_initial_search_space_prompt(estimator_type: str = "xgboost") -> str:
    """
    Prompt the LLM to output ONLY a function:
        def define_search_space(trial):
            ...
    using trial.suggest_* APIs.
    """
    if estimator_type == "xgboost":
        desc = """
Tunable hyperparameters include:
- n_estimators (int)
- max_depth (int)
- min_child_weight (int or float)
- gamma (float)
- scale_pos_weight (float)
- learning_rate (float)
- subsample (float)
- colsample_bylevel (float)
- colsample_bytree (float)
- reg_alpha (float)
- reg_lambda (float)
"""
    elif estimator_type == "lightgbm":
        desc = """
Tunable hyperparameters include:
- n_estimators (int)
- max_depth (int)
- min_child_samples (int)
- min_split_gain (float)
- scale_pos_weight (float)
- learning_rate (float)
- subsample (float)
- colsample_bytree (float)
- reg_alpha (float)
- reg_lambda (float)
- num_leaves (int)
"""
    else:
        desc = """
For custom estimators, pick the most relevant hyperparameters, e.g. regularization (alpha, lambda),
complexity (max_depth, n_estimators), learning_rate, sampling (subsample), class_balance, etc.
"""

    return f"""{SYSTEM_PROMPT}

Given your understanding of {estimator_type} and general best practices,
along with dataset characteristics (if available), please do the following:

1. Explain your reasoning for an **initial** search space. Focus on casting a sufficiently wide search space that we will refine in subsequent iterations.
2. Then OUTPUT ONLY a Python function with this exact signature:

    def define_search_space(trial):

Within it, use `trial.suggest_int`, `trial.suggest_float`, `trial.suggest_loguniform`,
`trial.suggest_categorical`, etc., to define the full hyperparameter search space.

Avoid any other code outside that function.  

Hyperparameter descriptions:
{desc}
""".strip()


def get_refine_search_space_prompt(
    top_n: str,
    last_value: float,
    best_value: float,
    all_time_configs: Optional[str] = None,
    iteration: Optional[int] = None,
    max_iterations: Optional[int] = None,
    estimator_type: str = "xgboost",
) -> str:
    header = (
        f"--- Iteration {iteration + 1}/{max_iterations} ---\n"
        if iteration is not None and max_iterations is not None
        else ""
    )
    body = f"""
{header}
Previous top trials:
{top_n}

Last iteration best value: {last_value:.4f}
All-time best value: {best_value:.4f}

All-time best configs:
{all_time_configs}

Please explain your refinements, then OUTPUT ONLY a function:

    def define_search_space(trial):

that adjusts your `trial.suggest_*` ranges or categories for the next round.

Given the insights from the search history, your expertise in ML, and the need to further explore the search space, \
please suggest refinements for the search space in the next optimization round. Consider both narrowing and \
expanding the search space for hyperparameters where appropriate.

For each recommendation, please:
1. Explicitly tie back to any general best practices or patterns you are aware of regarding {estimator_type} tuning
2. Then, relate to the insights from the search history and explain how they align or deviate from these practices or patterns.
3. If suggesting an expansion of the search space, please provide a rationale for why a broader range could be beneficial.

Briefly summarize your reasoning for the refinements and then present the adjusted configurations.
"""
    return body.strip()


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------


class AnalysisAndRecommendations(BaseModel):
    """A summary analysis and recommendations for downstream HPO."""

    domain_analysis: str = Field(
        description="Domain-level insights and HPO related recommendations"
    )
    dataset_analysis: str = Field(
        description="Data-driven insights and HPO related recommendations"
    )


class PythonCode(BaseModel):
    """A valid python code block and its reasoning"""

    reasoning: str = Field(description="LLM explanation")
    code: str = Field(description="A Python code block")

    @field_validator("code", mode="after")
    def is_syntax_valid(cls, v: Any) -> bool:
        try:
            ast.parse(v, mode="exec")
            return v
        except SyntaxError as e:
            raise ValueError(f"Code can not be compiled: {e}")

    @field_validator("code", mode="after")
    def is_executable(cls, v: Any) -> bool:
        safe_globals = {"scipy": scipy, "optuna": optuna}
        try:
            exec(v, safe_globals)
            return v
        except Exception as e:
            raise ValueError(f"Code is not executable: {e}")
        
    @property
    def code_markdown(self) -> str:
        return f"```python\n{self.code}\n```"


# -----------------------------------------------------------------------------
# Dependencies dataclass
# -----------------------------------------------------------------------------


@dataclass
class AutoMLDependencies:
    """Dependencies for AutoML agents."""

    dataset: Optional[pd.DataFrame] = None
    target: Optional[str] = None
    estimator_type: str = "xgboost"
    custom_estimator: Optional[BaseEstimator] = None
    use_dataset_analysis: bool = True


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def generate_search_space_from_code(
    code: str,
) -> Callable[[optuna.trial.Trial], dict]:
    """
    Exec the LLM code and return the define_search_space function.
    """
    local_ns: dict[str, Any] = {"optuna": optuna}
    exec(code, local_ns)
    return local_ns["define_search_space"]


def create_estimator(
    estimator_type: str,
    custom_estimator: Optional[BaseEstimator] = None,
    random_state: int = 42,
) -> BaseEstimator:
    if custom_estimator is not None:
        return custom_estimator
    if estimator_type == "xgboost":
        return xgb.XGBClassifier(
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            use_label_encoder=False,
            verbosity=0,
        )
    elif estimator_type == "lightgbm":
        return lgb.LGBMClassifier(
            random_state=random_state,
            objective="binary",
            metric="binary_logloss",
            n_jobs=-1,
            verbosity=-1,
            force_row_wise=True,
        )
    elif estimator_type == "custom":
        if custom_estimator is None:
            raise ValueError("custom_estimator must be provided for 'custom'")
        return custom_estimator
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}")


def evaluate_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
    target_name: str = "target",
    top_p: float = 0.05,
    estimator_type: str = "xgboost",
    custom_estimator: Optional[BaseEstimator] = None,
    random_state: int = 42,
    n_splits: int = 5,
    n_repeats: int = 2,
) -> Dict[str, float]:
    """Evaluates model performance using cross-validation and holdout test set.

    This method performs both cross-validation on the training data and evaluation
    on a holdout test set. It provides more robust performance estimates by:
    1. Using repeated k-fold cross-validation on training data
    2. Evaluating on a separate holdout test set
    3. Computing mean and standard deviation of metrics across folds

    Args:
        df_train (pd.DataFrame): Training data including features and target
        df_test (pd.DataFrame | None): Test data including features and target. If None, a test set will be created using train_test_split.
        target_name (str): Name of target column in dataframes
        top_p (float): Fraction (0 < top_p <= 1) of samples to include in top predictions
        estimator_type (str): Type of estimator to use
        custom_estimator (BaseEstimator): Custom estimator to use
        random_state (int): Random state for reproducibility
        n_splits (int): Number of folds for cross-validation
        n_repeats (int): Number of times to repeat cross-validation

    Returns:
        dict: Dictionary containing metrics for both CV and holdout test:
            - cv_accuracy (float): Mean CV accuracy
            - cv_accuracy_std (float): Std of CV accuracy
            - cv_auc (float): Mean CV AUC
            - cv_auc_std (float): Std of CV AUC
            - cv_ppv (float): Mean CV PPV
            - cv_ppv_std (float): Std of CV PPV
            - test_accuracy (float): Holdout test accuracy
            - test_auc (float): Holdout test AUC
            - test_ppv (float): Holdout test PPV
    """
    df_train, df_test = df_train.copy(), df_test.copy() if df_test is not None else None

    # Prepare data using train_test_split
    X = df_train.drop(columns=[target_name])
    y = df_train[target_name]

    if df_test is None:
        # If no test set provided, create one using train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,  # Use 20% for test set
            random_state=random_state,
            stratify=y,  # Stratify on target variable
        )
        x = X_train
        y = y_train
        test_x = X_test
        test_y = y_test
    else:
        # Use provided test set
        x = X
        test_x = df_test.drop(columns=[target_name])
        test_y = df_test[target_name]

    # Initialize metrics storage
    cv_metrics = {"accuracy": [], "auc": [], "ppv": []}

    # Perform repeated k-fold cross-validation
    rkf = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    for train_idx, val_idx in rkf.split(x):
        # Split data for this fold
        X_train_fold = x.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = x.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train model
        model = create_estimator(estimator_type, custom_estimator, random_state)
        model.fit(X=X_train_fold, y=y_train_fold)

        # Get predictions and probabilities
        probs = model.predict_proba(X_val_fold)
        preds = model.predict(X_val_fold)

        # Calculate metrics for this fold
        cv_metrics["accuracy"].append(float(accuracy_score(y_val_fold, preds)))
        cv_metrics["auc"].append(
            float(roc_auc_score(y_val_fold, probs[:, 1]))
        )  # Use probabilities for AUC

        # Compute PPV at top_p%
        if probs.shape[1] == 2:  # Binary classification
            positive_probs = probs[:, 1]
        else:
            positive_probs = probs.flatten()
        cv_metrics["ppv"].append(
            float(metric_ppv(y_val_fold, positive_probs, top_p=top_p))
        )

    # Calculate mean and std of CV metrics
    cv_results = {
        "cv_accuracy": float(np.mean(cv_metrics["accuracy"])),
        "cv_accuracy_std": float(np.std(cv_metrics["accuracy"])),
        "cv_auc": float(np.mean(cv_metrics["auc"])),
        "cv_auc_std": float(np.std(cv_metrics["auc"])),
        "cv_ppv": float(np.mean(cv_metrics["ppv"])),
        "cv_ppv_std": float(np.std(cv_metrics["ppv"])),
    }

    # Evaluate on holdout test set
    if test_x is not None and test_y is not None:
        model = create_estimator(estimator_type, custom_estimator, random_state)
        model.fit(X=x, y=y)

        # Get predictions and probabilities for test set
        probs = model.predict_proba(test_x)
        preds = model.predict(test_x)

        test_results = {
            "test_accuracy": float(accuracy_score(test_y, preds)),
            "test_auc": float(
                roc_auc_score(test_y, probs[:, 1])
            ),  # Use probabilities for AUC
        }

        # Compute PPV at top_p% for test set
        if probs.shape[1] == 2:
            positive_probs = probs[:, 1]
        else:
            positive_probs = probs.flatten()
        test_results["test_ppv"] = float(
            metric_ppv(test_y, positive_probs, top_p=top_p)
        )

        # Combine CV and test results
        return {**cv_results, **test_results}

    return cv_results


def tune_model(
    define_search_space: Callable[[optuna.trial.Trial], dict],
    X: pd.DataFrame,
    y: pd.Series,
    estimator_type: str = "xgboost",
    custom_estimator: Optional[BaseEstimator] = None,
    top_p: float = 0.05,
    scoring: Union[str, Callable, None] = None,
    n_trials: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    cv: int = 5,
) -> optuna.Study:
    """
    Run an Optuna study with a sklearn CV objective.
    """
    if scoring is None:
        scoring = make_scorer(precision_score, pos_label=1)
    elif scoring == "ppv":
        scoring = make_scorer(metric_ppv, top_p=top_p)
    elif scoring == "auc":
        scoring = make_scorer(roc_auc_score)
    elif scoring == "accuracy":
        scoring = make_scorer(accuracy_score)

    def objective(trial: optuna.trial.Trial) -> float:
        params = define_search_space(trial)
        model = create_estimator(estimator_type, custom_estimator, random_state)
        model.set_params(**params)

        scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    return study


def extract_logs_from_study(study: optuna.Study, top_n: int = 5) -> Tuple[str, float]:
    """
    Summarize the top trials and return (summary, best_value)
    """
    trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )[:top_n]

    lines = []
    for i, t in enumerate(trials, start=1):
        param_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        lines.append(f"Trial {i} (value={t.value:.4f}): {param_str}")

    best_value = trials[0].value if trials else float("-inf")
    return "\n".join(lines), best_value


async def only_use_df_only_if_allowed(
    ctx: RunContext[AutoMLDependencies],
    tool_def: ToolDefinition,
) -> Union[ToolDefinition, None]:
    if ctx.deps.use_dataset_analysis:
        return tool_def


# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------

analysis_and_recommendations_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=AutoMLDependencies,
    output_type=AnalysisAndRecommendations,
    system_prompt=SYSTEM_PROMPT,
)


@analysis_and_recommendations_agent.tool(prepare=only_use_df_only_if_allowed)
async def get_dataset_characteristics(ctx: RunContext[AutoMLDependencies]) -> str:
    """Returns a summary of the user's dataset.

    Args:
        n_sample (int): The number of sample rows to capture for analysis.

    Returns:
        str: A summary of the dataset.
    """
    df_summary = exploratory_data_analysis(
        ctx.deps.dataset, ctx.deps.target, n_sample=20
    )
    summary_string = format_eda_for_llm(df_summary)
    return summary_string


initial_search_space_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=AutoMLDependencies,
    output_type=PythonCode,
    system_prompt=SYSTEM_PROMPT,
)


@initial_search_space_agent.output_validator
def validate_initial_space(
    ctx: RunContext[AutoMLDependencies], python_code: PythonCode
) -> PythonCode:
    # Ensure define_search_space is present
    if "def define_search_space" not in python_code.code:
        raise ModelRetry("Please define `def define_search_space(trial):`")
    return python_code


refine_search_space_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=AutoMLDependencies,
    output_type=PythonCode,
    system_prompt=SYSTEM_PROMPT,
)


# -----------------------------------------------------------------------------
# Main Tuner Class
# -----------------------------------------------------------------------------


class AutoTuneLLM:
    def __init__(
        self,
        use_dataset_analysis: bool = True,
        estimator_type: str = "xgboost",
        custom_estimator: Optional[BaseEstimator] = None,
        scoring: Union[str, Callable, None] = None,
        dataset: Optional[pd.DataFrame] = None,
        target: Optional[str] = None,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        top_p: float = 0.05,
        top_n_configs: int = 5,
        max_no_improve: int = 3,
        n_trials: int = 100,
        random_state: int = 42,
        n_jobs: int = -1,
        cv: int = 5,
        verbose: int = 0,
    ):
        # init
        self.use_dataset_analysis = use_dataset_analysis
        self.estimator_type = estimator_type
        self.custom_estimator = custom_estimator
        self.scoring = scoring
        self.top_p = top_p
        self.dataset = dataset
        self.target = target
        self.task_description = task_description
        self.top_n_configs = top_n_configs
        self.max_no_improve = max_no_improve
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose

        if estimator_type == "custom" and custom_estimator is None:
            raise ValueError("custom_estimator must be provided for 'custom'")
        if estimator_type not in ("xgboost", "lightgbm", "custom"):
            raise ValueError(f"Unsupported estimator_type: {estimator_type}")

        self.deps = AutoMLDependencies(
            dataset=dataset,
            target=target,
            estimator_type=estimator_type,
            custom_estimator=custom_estimator,
            use_dataset_analysis=use_dataset_analysis,
        )
        self.best_configs: list[dict] = []
        self.last_values: list[float] = []
        self.studies: list[optuna.Study] = []

    def _update_best(self, value: float, params: dict, it: int) -> bool:
        entry = {"score": value, "config": params.copy(), "iteration": it}
        self.best_configs.append(entry)
        self.best_configs.sort(key=lambda x: x["score"], reverse=True)
        self.best_configs = self.best_configs[: self.top_n_configs]
        return entry in self.best_configs

    def tune_with_dataset_analysis(self, max_iterations: int = 5) -> None:
        if self.dataset is None or self.target is None:
            raise ValueError("Dataset and target are required")
        self._run_loop(max_iterations, use_data=True)

    def tune_with_user_description(
        self, user_description: str, max_iterations: int = 5
    ) -> None:
        if self.dataset is None or self.target is None:
            raise ValueError("Dataset and target are required")
        self.task_description = user_description
        self._run_loop(max_iterations, use_data=False)

    def _run_loop(self, max_iters: int, use_data: bool) -> None:
        X = self.dataset.drop(columns=[self.target])
        y = self.dataset[self.target]

        # 1) analysis
        # Get train/test split for the baseline evaluation
        from sklearn.model_selection import train_test_split

        # Create train/test split with stratification to maintain class balance
        df_train, df_test = train_test_split(
            self.dataset,
            test_size=0.2,  # Use 20% for test set
            random_state=self.random_state,
            stratify=self.dataset[self.target],  # Stratify on target variable
        )
        metrics = evaluate_dataset(
            df_train=df_train,
            df_test=df_test,
            target_name=self.target,
            n_splits=5,  # Number of CV folds
            n_repeats=2,  # Number of CV repetitions
        )

        # Access CV metrics with standard deviations
        print(
            f"CV Accuracy: {metrics['cv_accuracy']:.3f} (±{metrics['cv_accuracy_std']:.3f})"
        )
        print(f"CV AUC: {metrics['cv_auc']:.3f} (±{metrics['cv_auc_std']:.3f})")
        print(f"CV PPV: {metrics['cv_ppv']:.3f} (±{metrics['cv_ppv_std']:.3f})")

        # Access test set metrics
        print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"Test AUC: {metrics['test_auc']:.3f}")
        print(f"Test PPV: {metrics['test_ppv']:.3f}")

        prompt = get_analysis_and_recommendations_prompt(
            self.task_description, self.estimator_type
        )
        if not use_data:
            self.deps.dataset = None
            self.deps.use_dataset_analysis = False
        else:
            self.deps.use_dataset_analysis = True
        analysis = analysis_and_recommendations_agent.run_sync(prompt, deps=self.deps)
        print(analysis.output.domain_analysis)
        print(analysis.output.dataset_analysis)
        if not use_data:
            self.deps.dataset = self.dataset

        # 2) initial search-space
        init_prompt = get_initial_search_space_prompt(self.estimator_type)
        init_sc = initial_search_space_agent.run_sync(init_prompt, deps=self.deps)
        print(f"Initial code:\n\n```python\n{init_sc.output.code}\n```")
        current_code = init_sc.output.code
        last_history = init_sc.all_messages()

        no_improve = 0

        for iteration in range(max_iters):
            print(f"\n--- Iteration {iteration + 1}/{max_iters} ---")
            define_fn = generate_search_space_from_code(current_code)
            study = tune_model(
                define_fn,
                X,
                y,
                estimator_type=self.estimator_type,
                custom_estimator=self.custom_estimator,
                scoring=self.scoring,
                top_p=self.top_p,
                n_trials=self.n_trials,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                cv=self.cv,
            )
            self.studies.append(study)
            top_summary, best_val = extract_logs_from_study(
                study, top_n=self.top_n_configs
            )
            print("Top Trials:\n", top_summary)
            print(f"Best value this run: {best_val:.4f}")

            improved = self._update_best(best_val, study.best_params, iteration + 1)
            self.last_values.append(best_val)

            if improved:
                print("✅ New top configuration.")
                no_improve = 0
            else:
                no_improve += 1
                print(f"❌ No improvement ({no_improve}/{self.max_no_improve}).")

            # check early stop
            if no_improve >= self.max_no_improve:
                print("Early stopping.")
                break

            if iteration + 1 == max_iters:
                print("Reached max iterations.")
                break

            # refine
            all_time = "\n".join(
                f"Iteration {e['iteration']}: score={e['score']:.4f}, params={e['config']}"
                for e in self.best_configs
            )
            refine_p = get_refine_search_space_prompt(
                top_n=top_summary,
                last_value=best_val,
                best_value=self.best_configs[0]["score"],
                estimator_type=self.estimator_type,
                all_time_configs=all_time,
                iteration=iteration,
                max_iterations=max_iters,
            )
            print("Refinement prompt:\n", refine_p)
            ref_sc = refine_search_space_agent.run_sync(
                refine_p,
                message_history=last_history,
                deps=self.deps,
            )
            print("Refine reasoning:\n", ref_sc.output.reasoning)
            print(f"Code:\n\n```python\n{ref_sc.output.code}\n```")
            current_code = ref_sc.output.code
            last_history = ref_sc.all_messages()

        # final summary
        print("\n=== Tuning Complete ===")
        if self.best_configs:
            best = self.best_configs[0]
            print(f"Best score: {best['score']:.4f} (Iteration {best['iteration']})")
            print("Best params:", best["config"])
        print("Value progression:", self.last_values)

        final_estimator = create_estimator(
            self.estimator_type, self.custom_estimator, self.random_state
        )
        final_estimator.set_params(**self.best_configs[0]["config"])
        final_results = evaluate_dataset(
            custom_estimator=final_estimator,
            df_train=df_train,
            df_test=df_test,
            target_name=self.target,
            n_splits=5,  # Number of CV folds
            n_repeats=2,  # Number of CV repetitions
        )
        print(f"Final accuracy: {final_results['cv_accuracy']:.4f}")
        print(f"Final auc: {final_results['cv_auc']:.4f}")
        print(f"Final ppv: {final_results['cv_ppv']:.4f}")

    def get_best_config(self) -> Optional[dict]:
        return self.best_configs[0]["config"] if self.best_configs else None

    def get_best_configs(self, n: Optional[int] = None) -> list[dict]:
        return self.best_configs if n is None else self.best_configs[:n]

    def get_tuning_summary(self) -> dict:
        return {
            "best_score": self.best_configs[0]["score"] if self.best_configs else None,
            "best_config": self.get_best_config(),
            "top_configs": self.best_configs.copy(),
            "iterations": len(self.last_values),
            "progression": self.last_values.copy(),
        }
