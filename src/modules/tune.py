"""
AutoTuneLLM: Generalized ML model tuning process for LightGBM, XGBoost, and custom sklearn estimators powered by LLMs.

This module provides a flexible AutoML tuning framework that can work with different
estimators and offers two distinct workflows:
1. Dataset-aware workflow: Analyzes dataset characteristics and uses them in LLM prompts
2. User-description workflow: Relies solely on user-provided task descriptions
"""

import ast
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import lightgbm as lgb
import pandas as pd
import scipy
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import HalvingRandomSearchCV

from src.utils import exploratory_data_analysis, format_eda_for_llm

# Comprehensive warning suppression
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="lightgbm")

os.environ["PYTHONWARNINGS"] = "ignore"

OPENAI_MODEL = OpenAIModel("gpt-4.1")


# ============================================================================
# Prompt Templates & Functions
# ============================================================================

DEFAULT_TASK_DESCRIPTION = """\
The classification problem under investigation is based on insurance claims data. \
More specifically, the goal is to predict whether a given claim will be high severity. \
Ultimately, we are interested in optimizing for PPV (Positive Predictive Value) at the top 5% of predicted probabilities.
"""

SYSTEM_PROMPT = """
You are a senior data scientist tasked with guiding the use of an AutoML tool to discover \
the best model configurations for a given binary classification dataset. Your role involves \
understanding the dataset characteristics (when available), proposing suitable hyperparameters \
and their search spaces, analyzing results, and iterating on configurations.

When dataset characteristics are available, use them to carefully analyze the dataset before \
responding to the user's question. When they are not available, rely on the user's description \
of the data and task.
"""


def get_analysis_and_recommendations_prompt(
    task_description: str = DEFAULT_TASK_DESCRIPTION, estimator_type: str = "xgboost"
) -> str:
    """
    Generate analysis and recommendations prompt for the given estimator type.

    Args:
        task_description (str): Description of the machine learning task
        estimator_type (str): Type of estimator ("xgboost", "lightgbm", or "custom")

    Returns:
        str: Formatted prompt for analysis and recommendations
    """
    prompt = f"""{task_description}
For this specific inquiry, you are tasked with supporting hyperparameter optimization for a {estimator_type} model. \
Given the problem context and dataset characteristics (when available), provide analysis and recommendations \
to guide downstream iterative search space exploration.
"""
    return prompt.strip()


def get_initial_search_space_prompt(estimator_type: str = "xgboost") -> str:
    """
    Generate initial search space prompt based on estimator type.

    Args:
        estimator_type (str): Type of estimator ("xgboost", "lightgbm", or "custom")

    Returns:
        str: Formatted prompt for initial search space generation
    """
    if estimator_type == "xgboost":
        hyperparams_description = """
Tunable hyperparameters include:
- n_estimators (integer): Number of boosting rounds or trees to be trained.
- max_depth (integer): Maximum tree depth for base learners.
- min_child_weight (integer or float): Minimum sum of instance weight (hessian) needed in a leaf node. 
- gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree.
- scale_pos_weight (float): Balancing of positive and negative weights.
- learning_rate (float): Step size shrinkage used during each boosting round to prevent overfitting. 
- subsample (float): Fraction of the training data sampled to train each tree. 
- colsample_bylevel (float): Fraction of features that can be randomly sampled for building each level.
- colsample_bytree (float): Fraction of features that can be randomly sampled for building each tree. 
- reg_alpha (float): L1 regularization term on weights. 
- reg_lambda (float): L2 regularization term on weights.
"""
    elif estimator_type == "lightgbm":
        hyperparams_description = """
Tunable hyperparameters include:
- n_estimators (integer): Number of boosting rounds or trees to be trained.
- max_depth (integer): Maximum tree depth for base learners.
- min_child_samples (integer): Minimum number of data points in a leaf node.
- min_split_gain (float): Minimum loss reduction required to make a further partition.
- scale_pos_weight (float): Balancing of positive and negative weights.
- learning_rate (float): Step size shrinkage used during each boosting round.
- subsample (float): Fraction of the training data sampled to train each tree.
- colsample_bytree (float): Fraction of features that can be randomly sampled for building each tree.
- reg_alpha (float): L1 regularization term on weights.
- reg_lambda (float): L2 regularization term on weights.
- num_leaves (integer): Maximum number of leaves in one tree.
"""
    else:  # custom
        hyperparams_description = """
For custom estimators, you should identify the most relevant hyperparameters based on \
the estimator type and machine learning best practices. Common hyperparameters to consider include:
- Regularization parameters (alpha, lambda, C)
- Model complexity parameters (max_depth, n_estimators, num_leaves)
- Learning rate or step size parameters
- Sampling parameters (subsample, colsample)
- Class balancing parameters when applicable

Please analyze the provided estimator and suggest appropriate hyperparameters to tune.
"""

    prompt = f"""\
Given your understanding of {estimator_type} and general best practices in machine learning, along with the \
dataset characteristics (when available), suggest an initial search space for hyperparameters.

{hyperparams_description}

The search space is defined as a dict with keys being hyperparameter names, and values 
are the search space associated with the hyperparameter. For example:
    search_space = {{
        "learning_rate": loguniform(1e-4, 1e-3)
    }}

Available types of domains include: 
- scipy.stats.uniform(loc, scale), it samples values uniformly between loc and loc + scale.
- scipy.stats.loguniform(a, b), it samples values between a and b in a logarithmic scale.
- scipy.stats.randint(low, high), it samples integers uniformly between low (inclusive) and high (exclusive).
- a list of possible discrete value, e.g., ["a", "b", "c"]

Please first explain your reasoning, then provide the configurations of the initial 
search space written in python code.
"""
    return prompt.strip()


def get_refine_search_space_prompt(
    top_n: str,
    last_run_best_score: float,
    all_time_best_score: float,
    estimator_type: str = "xgboost",
    all_time_best_configs: str | None = None,
    iteration: int | None = None,
    max_iterations: int | None = None,
) -> str:
    """
    Generate a prompt for refining the search space based on results.

    Args:
        top_n (str): Top configurations from the last run
        last_run_best_score (float): Best score from the last run
        all_time_best_score (float): Best score across all runs
        estimator_type (str): Type of estimator being tuned
        all_time_best_configs (str | None): Top configurations across all runs (optional)
        iteration (int | None): Current iteration number (optional)
        max_iterations (int | None): Maximum number of iterations (optional)

    Returns:
        str: The formatted prompt for search space refinement
    """

    if iteration is not None and max_iterations is not None:
        prompt_parts = [
            f"--- Iteration {iteration + 1}/{max_iterations} ---",
            "",
        ]
    else:
        prompt_parts = [
            f"Given your previously suggested search space for {estimator_type}, the obtained top configurations from the last run with their test scores:",
            top_n,
            "",
        ]

    # Add all-time best configs section if provided
    if all_time_best_configs:
        prompt_parts.extend(
            ["Top configurations across ALL runs so far:", all_time_best_configs, ""]
        )

    # Get hyperparameter list based on estimator type
    if estimator_type == "xgboost":
        hyperparam_list = "n_estimators, max_depth, min_child_weight, gamma, scale_pos_weight, learning_rate, subsample, colsample_bylevel, colsample_bytree, reg_alpha, and reg_lambda"
    elif estimator_type == "lightgbm":
        hyperparam_list = "n_estimators, max_depth, min_child_samples, min_split_gain, scale_pos_weight, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, and num_leaves"
    else:
        hyperparam_list = "the relevant hyperparameters for your custom estimator"

    prompt_parts.extend(
        [
            f"The best score from the last run was {last_run_best_score}, while the best score ever achieved in all previous runs is {all_time_best_score}",
            "",
            f"Remember, tunable hyperparameters for {estimator_type} are: {hyperparam_list}.",
            "",
            "Given the insights from the search history, your expertise in ML, and the need to further explore the search space, please suggest refinements for the search space in the next optimization round. Consider both narrowing and expanding the search space for hyperparameters where appropriate.",
            "",
            "For each recommendation, please:",
            f"1. Explicitly tie back to any general best practices or patterns you are aware of regarding {estimator_type} tuning",
            "2. Then, relate to the insights from the search history and explain how they align or deviate from these practices or patterns.",
            "3. If suggesting an expansion of the search space, please provide a rationale for why a broader range could be beneficial.",
            "",
            "",
            "Briefly summarize your reasoning for the refinements and then present the adjusted configurations. Enclose your refined configurations between python code fences, and assign your configuration to a variable named search_space.",
        ]
    )

    return "\n".join(prompt_parts)


# ============================================================================
# Data Models
# ============================================================================


class AnalysisAndRecommendations(BaseModel):
    """A summary analysis and recommendations for downstream HPO."""

    domain_analysis: str = Field(
        description="Analysis and recommendations based on real world domain knowledge about the data and the problem at hand."
    )
    dataset_analysis: str = Field(
        description="Analysis and recommendations based on the dataset characteristics or user description."
    )

    def __str__(self) -> str:
        return f"Analysis and Recommendations:\n{self.domain_analysis}\n{self.dataset_analysis}"


class PythonCode(BaseModel):
    """A valid python code block and its reasoning"""

    reasoning: str = Field(description="Explanation of the code block")
    code: str = Field(description="A valid python code block")

    @field_validator("code", mode="after")
    def is_syntax_valid(cls, v: Any) -> bool:
        try:
            ast.parse(v, mode="exec")
            return v
        except SyntaxError as e:
            raise ValueError(f"Code can not be compiled: {e}")

    @field_validator("code", mode="after")
    def is_executable(cls, v: Any) -> bool:
        safe_globals = {"scipy": scipy}
        try:
            exec(v, safe_globals)
            return v
        except Exception as e:
            raise ValueError(f"Code is not executable: {e}")


@dataclass
class AutoMLDependencies:
    """Dependencies for AutoML agents."""

    dataset: Optional[pd.DataFrame] = None
    target: Optional[str] = None
    estimator_type: str = "xgboost"
    custom_estimator: Optional[BaseEstimator] = None


# ============================================================================
# Helper Functions
# ============================================================================


def generate_search_space_from_code(code: str) -> dict:
    """
    Generate a search space object from a python code block.

    Args:
        code (str): Python code string containing search_space definition

    Returns:
        dict: The search space dictionary
    """
    local_ns = {"scipy": scipy}
    exec(code, local_ns)
    search_space = local_ns["search_space"]
    return search_space


def create_estimator(
    estimator_type: str,
    custom_estimator: Optional[BaseEstimator] = None,
    random_state: int = 42,
) -> BaseEstimator:
    """
    Create an estimator based on the specified type.

    Args:
        estimator_type (str): Type of estimator ("xgboost", "lightgbm", or "custom")
        custom_estimator (BaseEstimator, optional): Custom sklearn estimator
        random_state (int): Random state for reproducibility

    Returns:
        BaseEstimator: Configured estimator
    """
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
            raise ValueError(
                "custom_estimator must be provided when estimator_type is 'custom'"
            )
        return custom_estimator
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}")


def extract_logs(search_results: BaseEstimator, top_n: int = 5) -> tuple[str, float]:
    """
    Extract top configurations and best score from search results.

    Args:
        search_results (BaseEstimator): Results from HalvingRandomSearchCV
        top_n (int): Number of top configurations to extract

    Returns:
        tuple[str, float]: Top configuration summary and best score
    """
    df = pd.DataFrame(search_results.cv_results_)

    # Identify top-performing configurations using rank_test_score
    top_configs = df.nsmallest(top_n, "rank_test_score").reset_index(drop=True)

    # Get hyperparameter columns (those starting with 'param_')
    hyperparameter_columns = [col for col in df.columns if col.startswith("param_")]

    # Extract the top-N configurations as strings
    config_strings = []
    for index, row in top_configs.iterrows():
        config_str = ", ".join(
            [f"{col[6:]}: {row[col]}" for col in hyperparameter_columns]
        )
        config_strings.append(
            f"Configuration {index + 1} ({row['mean_test_score']:.4f} test score): {config_str}"
        )

    # Join them together for a complete summary
    top_config_summary = "\n".join(config_strings)

    # Best test score
    last_run_best_score = top_configs.loc[0, "mean_test_score"]

    return top_config_summary, last_run_best_score


def format_best_configs_across_runs(best_configs: list[dict]) -> str:
    """
    Format the best configurations across all runs for inclusion in prompts.

    Args:
        best_configs (list[dict]): List of best config dictionaries with 'score', 'config', and 'iteration' keys

    Returns:
        str: Formatted string showing the top configurations across all runs
    """
    if not best_configs:
        return "No configurations available yet."

    config_strings = []
    for index, config_info in enumerate(best_configs):
        config = config_info["config"]
        score = config_info["score"]
        iteration = config_info["iteration"]

        # Format the hyperparameters
        config_str = ", ".join([f"{key}: {value}" for key, value in config.items()])

        config_strings.append(
            f"Configuration {index + 1} ({score:.4f} test score, from iteration {iteration}): {config_str}"
        )

    return "\n".join(config_strings)


def tune_model(
    search_space: dict,
    X: pd.DataFrame,
    y: pd.Series,
    estimator_type: str = "xgboost",
    custom_estimator: Optional[BaseEstimator] = None,
    scoring: Union[str, Callable] = None,
    n_candidates: int = 500,
    min_resources: str = "smallest",
    random_state: int = 42,
    n_jobs: int = -1,
    cv: int = 5,
    verbose: int = 0,
) -> BaseEstimator:
    """
    Tune a model using the given search space and metric.

    Args:
        search_space (dict): Hyperparameter search space
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        estimator_type (str): Type of estimator to tune
        custom_estimator (BaseEstimator, optional): Custom sklearn estimator
        scoring (Union[str, Callable], optional): Scoring function for evaluation

    Returns:
        BaseEstimator: Fitted HalvingRandomSearchCV object
    """
    # Create the estimator
    clf = create_estimator(estimator_type, custom_estimator)

    # Set default scoring if not provided
    if scoring is None:
        scoring = make_scorer(precision_score, pos_label=1)

    # Perform hyperparameter search
    search = HalvingRandomSearchCV(
        clf,
        search_space,
        scoring=scoring,
        factor=2,
        n_candidates=n_candidates,
        min_resources=min_resources,
        random_state=random_state,
        n_jobs=n_jobs,
        cv=cv,
        verbose=verbose,
    ).fit(X, y)

    return search


# ============================================================================
# Agents
# ============================================================================


async def get_dataset_characteristics(ctx: RunContext[AutoMLDependencies]) -> str:
    """
    Returns a summary of the user's dataset.

    Args:
        ctx (RunContext): Context containing dataset and target information

    Returns:
        str: A summary of the dataset characteristics
    """
    if ctx.deps.dataset is None or ctx.deps.target is None:
        return "Dataset characteristics not available - using user description only."

    df_summary = exploratory_data_analysis(
        ctx.deps.dataset, ctx.deps.target, n_sample=20
    )
    summary_string = format_eda_for_llm(df_summary)
    return summary_string


# Create agents
analysis_and_recommendations_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=AutoMLDependencies,
    output_type=AnalysisAndRecommendations,
    output_retries=3,
    retries=3,
    tools=[Tool(get_dataset_characteristics, takes_ctx=True, max_retries=5)],
    system_prompt=SYSTEM_PROMPT,
)


initial_search_space_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=AutoMLDependencies,
    output_type=PythonCode,
    output_retries=3,
    retries=3,
    system_prompt=SYSTEM_PROMPT,
)


@initial_search_space_agent.output_validator
def check_search_space_name(
    ctx: RunContext[AutoMLDependencies], python_code: PythonCode
) -> PythonCode:
    """Validate that the search space variable is properly named."""
    if "search_space" not in python_code.code:
        raise ModelRetry("The search space object name must be 'search_space'")
    return python_code


refine_search_space_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=AutoMLDependencies,
    output_type=PythonCode,
    output_retries=3,
    retries=3,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================================
# Main Generalized Tuner Class
# ============================================================================


class AutoTuneLLM:
    """
    A generalized ML model tuner that supports XGBoost, LightGBM, and custom sklearn estimators.

    This class provides two main workflows:
    1. Dataset-aware workflow: Analyzes dataset characteristics and uses them in LLM prompts
    2. User-description workflow: Relies solely on user-provided task descriptions

    Example usage:
        # XGBoost with dataset analysis
        tuner = AutoTuneLLM(
            estimator_type="xgboost",
            dataset=df,
            target="target_column",
            scoring="precision"
        )
        tuner.tune_with_dataset_analysis(max_iterations=5)

        # LightGBM with user description only
        tuner = AutoTuneLLM(
            estimator_type="lightgbm",
            scoring=make_scorer(f1_score)
        )
        tuner.tune_with_user_description("Binary classification task with imbalanced data...", max_iterations=5)
    """

    def __init__(
        self,
        estimator_type: str = "xgboost",
        custom_estimator: Optional[BaseEstimator] = None,
        scoring: Union[str, Callable, None] = None,
        dataset: Optional[pd.DataFrame] = None,
        target: Optional[str] = None,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        top_n_configs: int = 5,
        max_consecutive_no_improvement: int = 3,
        n_candidates: int = 500,
        min_resources: str = "smallest",
        random_state: int = 42,
        n_jobs: int = -1,
        cv: int = 5,
        verbose: int = 0,
    ):
        """
        Initialize the GeneralizedTuner.

        Args:
            estimator_type (str): Type of estimator ("xgboost", "lightgbm", or "custom")
            custom_estimator (BaseEstimator, optional): Custom sklearn estimator (required if estimator_type="custom")
            scoring (Union[str, Callable], optional): Scoring function for evaluation
            dataset (pd.DataFrame, optional): The dataset to tune on
            target (str, optional): The target column name
            task_description (str): Description of the task for context
            top_n_configs (int): Number of top configurations to retain
            max_consecutive_no_improvement (int): Maximum consecutive runs without improvement
            n_candidates (int): Number of candidates to evaluate
            min_resources (str): Minimum resources to use
            random_state (int): Random state for reproducibility
            n_jobs (int): Number of jobs to run in parallel
            cv (int): Number of cross-validation folds
            verbose (int): Verbosity level
        """
        self.estimator_type = estimator_type
        self.custom_estimator = custom_estimator
        self.scoring = scoring
        self.dataset = dataset
        self.target = target
        self.task_description = task_description
        self.top_n_configs = top_n_configs
        self.max_consecutive_no_improvement = max_consecutive_no_improvement
        self.n_candidates = n_candidates
        self.min_resources = min_resources
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose

        # Validate inputs
        if estimator_type == "custom" and custom_estimator is None:
            raise ValueError(
                "custom_estimator must be provided when estimator_type is 'custom'"
            )

        if estimator_type not in ["xgboost", "lightgbm", "custom"]:
            raise ValueError(f"Unsupported estimator_type: {estimator_type}")

        # Initialize dependencies
        self.deps = AutoMLDependencies(
            dataset=dataset,
            target=target,
            estimator_type=estimator_type,
            custom_estimator=custom_estimator,
        )

        # Initialize tracking variables
        self.search_space = None
        self.last_run_best_score = []
        self.best_configs = []  # List of {'score': float, 'config': dict, 'iteration': int}

    def _update_best_configs(self, score: float, config: dict, iteration: int) -> bool:
        """
        Update the list of best configurations with a new score and config.

        Args:
            score (float): The score achieved by this configuration
            config (dict): The hyperparameter configuration
            iteration (int): The iteration number when this config was found

        Returns:
            bool: True if this config made it into the top n, False otherwise
        """
        new_entry = {"score": score, "config": config.copy(), "iteration": iteration}

        # Add the new entry
        self.best_configs.append(new_entry)

        # Sort by score in descending order (higher scores are better)
        self.best_configs.sort(key=lambda x: x["score"], reverse=True)

        # Keep only the top n configurations
        was_in_top_n = (
            len(self.best_configs) <= self.top_n_configs
            or new_entry in self.best_configs[: self.top_n_configs]
        )
        self.best_configs = self.best_configs[: self.top_n_configs]

        return was_in_top_n

    def tune_with_dataset_analysis(self, max_iterations: int = 10) -> None:
        """
        Tune model hyperparameters using dataset characteristics in LLM prompts.

        This workflow analyzes the provided dataset and uses its characteristics
        to guide the hyperparameter optimization process.

        Args:
            max_iterations (int): Maximum number of tuning iterations to perform

        Raises:
            ValueError: If dataset or target is not provided
        """
        if self.dataset is None or self.target is None:
            raise ValueError(
                "Dataset and target must be provided for dataset-aware tuning"
            )

        print(f"Starting dataset-aware tuning for {self.estimator_type}")
        self._run_tuning_loop(max_iterations, use_dataset_analysis=True)

    def tune_with_user_description(
        self, user_description: str, max_iterations: int = 10
    ) -> None:
        """
        Tune model hyperparameters using only user-provided task description.

        This workflow relies solely on the user's description of the data and task,
        without passing any actual dataset information to the LLM.

        Args:
            user_description (str): Detailed description of the data and task
            max_iterations (int): Maximum number of tuning iterations to perform

        Raises:
            ValueError: If dataset or target is not provided for actual tuning
        """
        if self.dataset is None or self.target is None:
            raise ValueError(
                "Dataset and target must be provided for model tuning (even if not used in LLM prompts)"
            )

        print(f"Starting user-description-based tuning for {self.estimator_type}")

        # Update task description with user's detailed description
        self.task_description = user_description

        # Run tuning without dataset analysis
        self._run_tuning_loop(max_iterations, use_dataset_analysis=False)

    def _run_tuning_loop(self, max_iterations: int, use_dataset_analysis: bool) -> None:
        """
        Internal method to run the main tuning loop.

        Args:
            max_iterations (int): Maximum number of tuning iterations
            use_dataset_analysis (bool): Whether to use dataset characteristics in prompts
        """
        # Create X and y from dataset and target column
        X = self.dataset.drop(columns=[self.target])
        y = self.dataset[self.target]

        # Get analysis and recommendations
        analysis_and_recommendations_prompt = get_analysis_and_recommendations_prompt(
            self.task_description, self.estimator_type
        )

        # Temporarily disable dataset analysis if not needed
        if not use_dataset_analysis:
            original_dataset = self.deps.dataset
            self.deps.dataset = None

        try:
            analysis_and_recommendations_result = (
                analysis_and_recommendations_agent.run_sync(
                    analysis_and_recommendations_prompt, deps=self.deps
                )
            )
            print(f"{str(analysis_and_recommendations_result.output)}")
        finally:
            # Restore dataset if temporarily disabled
            if not use_dataset_analysis:
                self.deps.dataset = original_dataset

        # Generate initial search space
        initial_search_space_prompt = get_initial_search_space_prompt(
            self.estimator_type
        )
        initial_search_space = initial_search_space_agent.run_sync(
            initial_search_space_prompt,
            message_history=analysis_and_recommendations_result.all_messages(),
            deps=self.deps,
        )
        print(f"Initial Search Space: {initial_search_space.output.reasoning}")

        # Initialize tracking variables
        consecutive_no_improvement = 0
        current_search_space_code = initial_search_space.output.code
        last_message_history = initial_search_space.all_messages()

        # Main tuning loop
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Generate search space from current code
            search_space = generate_search_space_from_code(current_search_space_code)

            # Run hyperparameter tuning
            search_results = tune_model(
                search_space,
                X,
                y,
                estimator_type=self.estimator_type,
                custom_estimator=self.custom_estimator,
                scoring=self.scoring,
                n_candidates=self.n_candidates,
                min_resources=self.min_resources,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                cv=self.cv,
                verbose=self.verbose,
            )

            # Extract results
            top_config_summary, last_run_best_score = extract_logs(search_results)
            print(f"Top Config Summary:\n{top_config_summary}\n")
            print(f"Last Run Best Score: {last_run_best_score}")

            # Update score tracking
            self.last_run_best_score.append(last_run_best_score)

            # Update best configurations
            was_improvement = self._update_best_configs(
                last_run_best_score, search_results.best_params_, iteration + 1
            )

            # Check if this improved our top configurations
            if was_improvement and (
                len(self.best_configs) == 1
                or last_run_best_score > self.best_configs[1]["score"]
            ):
                print(f"New best score: {last_run_best_score}")
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
                print(
                    f"No improvement. Consecutive runs without improvement: {consecutive_no_improvement}"
                )

            # Print current top configurations
            print(f"\nCurrent top {len(self.best_configs)} configurations:")
            for i, config_info in enumerate(self.best_configs):
                print(
                    f"  {i + 1}. Score: {config_info['score']:.4f} (Iteration {config_info['iteration']})"
                )

            # Check early stopping criteria
            if consecutive_no_improvement >= self.max_consecutive_no_improvement:
                print(
                    f"Early stopping: No improvement for {self.max_consecutive_no_improvement} consecutive runs"
                )
                break

            # If this is the last iteration, don't refine search space
            if iteration == max_iterations - 1:
                print("Reached maximum iterations")
                break

            # Refine search space for next iteration
            all_time_best_score = (
                self.best_configs[0]["score"]
                if self.best_configs
                else last_run_best_score
            )

            # Only include all-time best configs if we're past the first iteration
            all_time_best_configs = None
            if iteration > 0:
                all_time_best_configs = format_best_configs_across_runs(
                    self.best_configs
                )

            refine_search_space_prompt = get_refine_search_space_prompt(
                top_config_summary,
                self.last_run_best_score[-1],
                all_time_best_score,
                self.estimator_type,
                all_time_best_configs,
                iteration=iteration,
                max_iterations=max_iterations,
            )

            print(f"Refine Search Space Prompt:\n\n{refine_search_space_prompt}\n\n")

            refine_search_space_result = refine_search_space_agent.run_sync(
                refine_search_space_prompt,
                message_history=last_message_history,
                deps=self.deps,
            )

            print(
                f"\nRefined Search Space Reasoning:\n{refine_search_space_result.output.reasoning}"
            )

            # Update for next iteration
            current_search_space_code = refine_search_space_result.output.code
            last_message_history = refine_search_space_result.all_messages()

        # Print final results
        print(f"\nTuning completed after {len(self.last_run_best_score)} iterations")
        if self.best_configs:
            print(f"Best score achieved: {self.best_configs[0]['score']}")
            print(f"Top {len(self.best_configs)} configurations found:")
            for i, config_info in enumerate(self.best_configs):
                print(
                    f"  {i + 1}. Score: {config_info['score']:.4f} (Iteration {config_info['iteration']})"
                )
        print(f"Score progression: {self.last_run_best_score}")

    def get_best_config(self) -> Optional[dict]:
        """
        Get the best hyperparameter configuration found during tuning.

        Returns:
            dict | None: The best hyperparameter configuration, or None if tuning hasn't been run yet
        """
        return self.best_configs[0]["config"] if self.best_configs else None

    def get_best_configs(self, n: Optional[int] = None) -> list[dict]:
        """
        Get the top n best hyperparameter configurations found during tuning.

        Args:
            n (int | None): Number of configurations to return. If None, returns all stored configs.

        Returns:
            list[dict]: List of configuration dictionaries with 'score', 'config', and 'iteration' keys
        """
        if n is None:
            return self.best_configs.copy()
        return self.best_configs[:n].copy()

    def get_tuning_summary(self) -> dict:
        """
        Get a summary of the tuning process including best scores, best configs, and score progression.

        Returns:
            dict: Summary of tuning results including best_score, best_config, top_configs,
                 total_iterations, score_progression, improvement_over_baseline, configs_retained,
                 and max_configs_to_retain
        """
        return {
            "estimator_type": self.estimator_type,
            "best_score": self.best_configs[0]["score"] if self.best_configs else None,
            "best_config": self.best_configs[0]["config"]
            if self.best_configs
            else None,
            "top_configs": self.best_configs.copy(),
            "total_iterations": len(self.last_run_best_score),
            "score_progression": self.last_run_best_score.copy(),
            "improvement_over_baseline": (
                self.best_configs[0]["score"] - self.last_run_best_score[0]
                if len(self.last_run_best_score) > 0 and self.best_configs
                else None
            ),
            "configs_retained": len(self.best_configs),
            "max_configs_to_retain": self.top_n_configs,
        }

    def create_best_estimator(self) -> Optional[BaseEstimator]:
        """
        Create an estimator with the best configuration found during tuning.

        Returns:
            BaseEstimator | None: Configured estimator with best hyperparameters, or None if no tuning has been performed
        """
        if not self.best_configs:
            return None

        best_config = self.best_configs[0]["config"]
        estimator = create_estimator(self.estimator_type, self.custom_estimator)
        estimator.set_params(**best_config)
        return estimator
