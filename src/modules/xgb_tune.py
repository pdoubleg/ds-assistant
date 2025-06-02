import os
import ast
import warnings
from dataclasses import dataclass
from typing import Any

import pandas as pd
import scipy
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import check_scoring
from sklearn.model_selection import HalvingRandomSearchCV

from src.utils import exploratory_data_analysis, format_eda_for_llm

# Comprehensive warning suppression for XGBoost
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="xgboost")

os.environ['PYTHONWARNINGS'] = 'ignore'

OPENAI_MODEL = OpenAIModel("gpt-4.1")


# ============================================================================
# Prompt Templates & Functions
# ============================================================================

DEFAULT_TASK_DESCRIPTION = """\
The classification problem under investigation is based on insurance claims data. \
Ultimately, we are interested in optimizing for PPV (Positive Predictive Value).
"""

SYSTEM_PROMPT = """
You are a senior data scientist tasked with guiding the use of an AutoML tool  
to discover the best XGBoost model configurations for a given binary classification dataset. 
Your role involves understanding the dataset characteristics, proposing suitable metrics, 
hyperparameters, and their search spaces, analyzing results, and iterating on configurations.

Use the dataset characteristics tool to carefully analyze the dataset before responding to the user's question.
"""

INITIAL_SEARCH_SPACE_PROMPT = """\
Given your understanding of XGBoost and general best practices in machine learning, along with the \
dataset characteristics, suggest an initial search space for hyperparameters. 

Tunable hyperparameters include:
- n_estimators (integer): Number of boosting rounds or trees to be trained.
- max_depth (integer): Maximum tree depth for base learners.
- min_child_weight (integer or float): Minimum sum of instance weight (hessian) needed in a leaf node. 
- gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree.
- scale_pos_weight (float): Balancing of positive and negative weights.
- learning_rate (float): Step size shrinkage used during each boosting round to prevent overfitting. 
- subsample (float): Fraction of the training data sampled to train each tree. 
- colsample_bylevel (float): Fraction of features that can be randomly sampled for building each level (or depth) of the tree.
- colsample_bytree (float): Fraction of features that can be randomly sampled for building each tree. 
- reg_alpha (float): L1 regularization term on weights. 
- reg_lambda (float): L2 regularization term on weights. 

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


def get_metric_prompt(task_description: str = DEFAULT_TASK_DESCRIPTION) -> str:
    prompt = f"""{task_description}
For this specific inquiry, you are tasked with recommending a suitable hyperparameter optimization \
metric for training a XGBoost model. \
Given the problem context and dataset characteristics, suggest only the name of one of the built-in \
metrics using the MetricResult output type.
"""
    return prompt.strip()


def get_refine_search_space_prompt(
    top_n: str, last_run_best_score: float, all_time_best_score: float
) -> str:
    prompt = f"""\
Given your previously suggested search space, the obtained top configurations with their \
test scores:
{top_n}

The best score from the last run was {last_run_best_score}, while the best score ever \
achieved in all previous runs is {all_time_best_score}

Remember, tunable hyperparameters are: n_estimators, max_depth, min_child_samples, gamma, \
scale_pos_weight, learning_rate, subsample, colsample_bylevel, colsample_bytree, reg_alpha, \
and reg_lambda.

Given the insights from the search history, your expertise in ML, and the need to further \
explore the search space, please suggest refinements for the search space in the next optimization round. \
Consider both narrowing and expanding the search space for hyperparameters where appropriate.

For each recommendation, please:
1. Explicitly tie back to any general best practices or patterns you are aware of regarding XGBoost tuning
2. Then, relate to the insights from the search history and explain how they align or deviate from these \
practices or patterns.
3. If suggesting an expansion of the search space, please provide a rationale for why a broader range could \
be beneficial.


Briefly summarize your reasoning for the refinements and then present the adjusted configurations. \
Enclose your refined configurations between python code fences, and assign your \
configuration to a variable named search_space.
"""
    return prompt.strip()


# ============================================================================
# Data Models
# ============================================================================


class MetricResult(BaseModel):
    explanation: str = Field(
        description="An explanation of the metric choice based on best practices and the dataset characteristics."
    )
    metric: str = Field(
        description="The name of the metric to optimize. Must be a valid scikit-learn callable scoring metric."
    )

    @field_validator("metric", mode="after")
    @classmethod
    def validate_metric(cls, value: Any):
        try:
            # Use a dummy estimator for validation
            dummy_estimator = DummyClassifier()

            # Try to create a scorer from the string
            _ = check_scoring(dummy_estimator, scoring=value)

            return value

        except (ValueError, TypeError, AttributeError):
            raise ValueError(
                f"Invalid scoring metric: {value}. Please use a valid scikit-learn callable scoring metric."
            )


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
    dataset: pd.DataFrame
    target: str


# ============================================================================
# Helper Functions
# ============================================================================


def generate_search_space_from_code(code: str) -> dict:
    """
    Generate a search space object from a python code block.
    """
    local_ns = {"scipy": scipy}
    exec(code, local_ns)
    search_space = local_ns["search_space"]
    return search_space


def extract_logs(search_results: BaseEstimator, top_n: int = 5):
    df = pd.DataFrame(search_results.cv_results_)

    # 1. Identify top-performing configurations using rank_test_score
    top_configs = df.nsmallest(top_n, "rank_test_score").reset_index(drop=True)

    hyperparameter_columns = [
        "param_colsample_bylevel",
        "param_colsample_bytree",
        "param_gamma",
        "param_learning_rate",
        "param_max_depth",
        "param_min_child_weight",
        "param_n_estimators",
        "param_reg_alpha",
        "param_reg_lambda",
        "param_scale_pos_weight",
        "param_subsample",
    ]

    # Extracting the top-N configurations as strings
    config_strings = []
    for index, row in top_configs.iterrows():
        config_str = ", ".join(
            [f"{col[6:]}: {row[col]}" for col in hyperparameter_columns]
        )
        config_strings.append(
            f"Configuration {index + 1} ({row['mean_test_score']:.4f} test score): {config_str}"
        )

    # Joining them together for a complete summary
    top_config_summary = "\n".join(config_strings)

    # Best test score
    last_run_best_score = top_configs.loc[0, "mean_test_score"]

    return top_config_summary, last_run_best_score


def tune_xgb_model(
    search_space: dict, metric_result: MetricResult, X: pd.DataFrame, y: pd.Series
) -> BaseEstimator:
    """
    Tune an XGBoost model using the given search space and metric.
    """
    clf = xgb.XGBClassifier(
        seed=42,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        use_label_encoder=False,
        verbosity=0  # Set to silent mode (0 = silent, 1 = warning, 2 = info, 3 = debug)
    )

    search = HalvingRandomSearchCV(
        clf,
        search_space,
        scoring=metric_result.output.metric,
        n_candidates=500,
        cv=5,
        min_resources="exhaust",
        factor=2,
        verbose=0,
    ).fit(X, y)

    return search


# ============================================================================
# Agents
# ============================================================================


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


metric_agent = Agent(
    model=OPENAI_MODEL,
    deps_type=AutoMLDependencies,
    output_type=MetricResult,
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
    system_prompt=INITIAL_SEARCH_SPACE_PROMPT,
)


@initial_search_space_agent.output_validator
def check_search_space_name(
    ctx: RunContext[AutoMLDependencies], python_code: PythonCode
) -> PythonCode:
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


class XGBoostTuner:
    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        top_n_configs: int = 5,
    ):
        """
        Initialize the XGBoost tuner.

        Args:
            dataset (pd.DataFrame): The dataset to tune on
            target (str): The target column name
            task_description (str): Description of the task for context
            top_n_configs (int): Number of top configurations to retain (default: 5)
        """
        self.dataset = dataset
        self.target = target
        self.task_description = task_description
        self.top_n_configs = top_n_configs

        self.deps = AutoMLDependencies(dataset=dataset, target=target)
        self.metric_agent = metric_agent
        self.initial_search_space_agent = initial_search_space_agent
        self.refine_search_space_agent = refine_search_space_agent

        self.search_space = None
        self.last_run_best_score = []
        # Store top n configurations as list of dicts with 'score' and 'config' keys
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
        new_entry = {
            'score': score,
            'config': config.copy(),
            'iteration': iteration
        }
        
        # Add the new entry
        self.best_configs.append(new_entry)
        
        # Sort by score in descending order (higher scores are better)
        self.best_configs.sort(key=lambda x: x['score'], reverse=True)
        
        # Keep only the top n configurations
        was_in_top_n = len(self.best_configs) <= self.top_n_configs or new_entry in self.best_configs[:self.top_n_configs]
        self.best_configs = self.best_configs[:self.top_n_configs]
        
        return was_in_top_n

    def tune(self, max_iterations: int = 10):
        """
        Tune XGBoost hyperparameters iteratively until max_iterations or 3 consecutive runs without improvement.

        Args:
            max_iterations (int): Maximum number of tuning iterations to perform
        """
        # Create X and y from dataset and target column
        X = self.dataset.drop(columns=[self.target])
        y = self.dataset[self.target]

        # Get the metric to optimize
        metric_prompt = get_metric_prompt(self.task_description)
        metric_result = self.metric_agent.run_sync(metric_prompt, deps=self.deps)
        print(f"Metric Explanation: {metric_result.output.explanation}")
        print(f"Metric: {metric_result.output.metric}")

        # Generate initial search space
        initial_search_space = self.initial_search_space_agent.run_sync(
            INITIAL_SEARCH_SPACE_PROMPT, message_history=metric_result.all_messages()
        )
        print(f"Initial Search Space: {initial_search_space.output.reasoning}")

        # Initialize tracking variables
        consecutive_no_improvement = 0
        max_consecutive_no_improvement = 3
        current_search_space_code = initial_search_space.output.code
        last_message_history = initial_search_space.all_messages()

        # Main tuning loop
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Generate search space from current code
            search_space = generate_search_space_from_code(current_search_space_code)

            # Run hyperparameter tuning
            search_results = tune_xgb_model(search_space, metric_result, X, y)

            # Extract results
            top_config_summary, last_run_best_score = extract_logs(search_results)
            print(f"Top Config Summary:\n{top_config_summary}\n")
            print(f"Last Run Best Score: {last_run_best_score}")

            # Update score tracking
            self.last_run_best_score.append(last_run_best_score)

            # Update best configurations
            was_improvement = self._update_best_configs(
                last_run_best_score, 
                search_results.best_params_, 
                iteration + 1
            )

            # Check if this improved our top configurations
            if was_improvement and (len(self.best_configs) == 1 or last_run_best_score > self.best_configs[1]['score']):
                print(f"New best score: {last_run_best_score}")
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
                print(f"No improvement. Consecutive runs without improvement: {consecutive_no_improvement}")

            # Print current top configurations
            print(f"\nCurrent top {len(self.best_configs)} configurations:")
            for i, config_info in enumerate(self.best_configs):
                print(f"  {i+1}. Score: {config_info['score']:.4f} (Iteration {config_info['iteration']})")

            # Check early stopping criteria
            if consecutive_no_improvement >= max_consecutive_no_improvement:
                print(f"Early stopping: No improvement for {max_consecutive_no_improvement} consecutive runs")
                break

            # If this is the last iteration, don't refine search space
            if iteration == max_iterations - 1:
                print("Reached maximum iterations")
                break

            # Refine search space for next iteration
            # Use the overall best score for refinement logic
            all_time_best_score = self.best_configs[0]['score'] if self.best_configs else last_run_best_score
            refine_search_space_prompt = get_refine_search_space_prompt(
                top_config_summary,
                self.last_run_best_score[-1],
                all_time_best_score,
            )

            refine_search_space_result = self.refine_search_space_agent.run_sync(
                refine_search_space_prompt,
                message_history=last_message_history,
            )

            print(f"\nRefined Search Space Reasoning:\n{refine_search_space_result.output.reasoning}")

            # Update for next iteration
            current_search_space_code = refine_search_space_result.output.code
            last_message_history = refine_search_space_result.all_messages()

        print(f"\nTuning completed after {len(self.last_run_best_score)} iterations")
        if self.best_configs:
            print(f"Best score achieved: {self.best_configs[0]['score']}")
            print(f"Top {len(self.best_configs)} configurations found:")
            for i, config_info in enumerate(self.best_configs):
                print(f"  {i+1}. Score: {config_info['score']:.4f} (Iteration {config_info['iteration']})")
        print(f"Score progression: {self.last_run_best_score}")

    def get_best_config(self) -> dict | None:
        """
        Get the best hyperparameter configuration found during tuning.

        Returns:
            dict | None: The best hyperparameter configuration, or None if tuning hasn't been run yet
        """
        return self.best_configs[0]['config'] if self.best_configs else None

    def get_best_configs(self, n: int | None = None) -> list[dict]:
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
            dict: Summary of tuning results
        """
        return {
            "best_score": self.best_configs[0]['score'] if self.best_configs else None,
            "best_config": self.best_configs[0]['config'] if self.best_configs else None,
            "top_configs": self.best_configs.copy(),
            "total_iterations": len(self.last_run_best_score),
            "score_progression": self.last_run_best_score.copy(),
            "improvement_over_baseline": (
                self.best_configs[0]['score'] - self.last_run_best_score[0]
                if len(self.last_run_best_score) > 0 and self.best_configs
                else None
            ),
            "configs_retained": len(self.best_configs),
            "max_configs_to_retain": self.top_n_configs,
        }
