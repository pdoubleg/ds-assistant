from typing import Optional
from sklearn.base import BaseEstimator

from .tools import render_estimator_params


SYSTEM_PROMPT = """
You are a senior data scientist tasked with guiding the use of Optuna to discover \
the best model and/or pipeline steps configurations for a given machine learning dataset. Your role involves \
understanding dataset characteristics, proposing suitable model and/or pipeline steps hyperparameters and their search \
spaces, analyzing results, and iterating on configurations for a given modeling task.
"""


def get_analysis_and_recommendations_prompt(
    task_description: str,
    estimator_type: str = "xgboost",
    task_type: str = "classification",
    metric: str = None,
) -> str:
    """Generate prompt for analysis and recommendations.

    Args:
        task_description (str): Description of the task/problem
        estimator_type (str): Type of estimator being used
        task_type (str, optional): Type of ML task. Defaults to "classification".
        metric (str, optional): The evaluation metric being optimized. Defaults to None.

    Returns:
        str: Formatted prompt for the analysis agent
    """
    metric_info = (
        f" The optimization target is the '{metric}' metric." if metric else ""
    )

    output_string = f"""\
{task_description}

For this specific inquiry, you are tasked with supporting hyperparameter optimization for a {estimator_type} model and/or pipeline steps \
on a {task_type} task.{metric_info} Given the problem context and dataset characteristics (when available), provide analysis \
and recommendations to guide downstream iterative search space exploration.
"""

    return output_string.strip()


def get_initial_search_space_prompt(
    estimator_type: str = "xgboost",
    task_type: str = "classification",
    analysis: Optional[str] = None,
    estimator: Optional[BaseEstimator] = None,
) -> str:
    """Generate prompt for initial search space creation."""

    if estimator is not None:
        model_params = render_estimator_params(estimator)
        model_params_section = f"""\
Here are the parameters of the estimator or Pipeline. Note that these are simply the result of the model.get_params() method. Not all \
parameters can/should be tuned with Optuna, e.g. random_state, verbosity, etc. Pay attention to any transformations that may benefit from \
different search spaces, e.g. StandardScaler, OneHotEncoder, PCA, etc. Ensure you always use precise names for the hyperparameters being mindful \
of nested parameters, i.e., steps and their input data types.

Prior issues to avoid:
InvalidParameterError: The 'max_features' parameter of RandomForestClassifier must be an int in the range [1, inf), a float in the \
range (0.0, 1.0], a str among {'sqrt', 'log2'} or None. Got 'auto' instead.
ValueError: CategoricalDistribution does not support dynamic value space.

{model_params}
"""
    else:
        model_params_section = ""

    analysis_section = ""
    if analysis:
        analysis_section = f"""
ANALYSIS AND RECOMMENDATIONS:
{analysis}"""

    output_string = f"""\
Given your understanding of {estimator_type} for {task_type} tasks and general best practices, \
along with dataset characteristics (if available), please do the following:

1. Explain your reasoning for an **initial** search space. Focus on casting a sufficiently wide search space that we will refine in subsequent iterations.
2. Then OUTPUT ONLY a Python function with this exact signature:

    def define_search_space(trial):

Within it, use `trial.suggest_int`, `trial.suggest_float`, `trial.suggest_loguniform`,
`trial.suggest_categorical`, etc., to define the full hyperparameter search space.

IMPORTANT CONSTRAINTS:
- For `trial.suggest_loguniform()`: The low value MUST be > 0 (not 0). Use trial.suggest_float() for ranges that include 0.
- For regularization parameters that can be 0, use `trial.suggest_float(name, 0.0, upper_bound)` instead of log uniform.
- Use log uniform only for parameters where the minimum meaningful value is > 0 (like learning_rate).

Avoid any other code outside that function.

Hyperparameter descriptions:
{analysis_section}
{model_params_section}
"""
    return output_string.strip()


# def get_initial_search_space_prompt(
#     estimator_type: str = "xgboost",
#     task_type: str = "classification",
#     analysis: Optional[str] = None,
#     estimator: Optional[BaseEstimator] = None,
# ) -> str:
#     """Generate prompt for initial search space creation."""

#     if estimator:
#         model_params = render_estimator_params(estimator)
#         model_params_section = f"""
# Here are the parameters of the model. Note that these are simply the result of the model.get_params() method. Not all \
# parameters can/should be tuned with Optuna, e.g. random_state, verbosity, etc.

# {model_params}
# """
#     else:
#         model_params_section = ""

#     if estimator_type == "xgboost":
#         desc = """
# Tunable hyperparameters include:
# - n_estimators (int): Number of boosting rounds [50-1000]
# - max_depth (int): Maximum tree depth [3-15]
# - min_child_weight (int or float): Minimum sum of instance weight needed in a child [1-10]
# - gamma (float): Minimum loss reduction required to make a further partition [0.0-1.0]
# - learning_rate (float): Step size shrinkage used to prevent overfitting [0.01-0.3]
# - subsample (float): Subsample ratio of the training instances [0.6-1.0]
# - colsample_bytree (float): Subsample ratio of columns when constructing each tree [0.6-1.0]
# - reg_alpha (float): L1 regularization term on weights [0.0-10.0]
# - reg_lambda (float): L2 regularization term on weights [0.0-10.0]
# """
#         if task_type == "classification":
#             desc += "- scale_pos_weight (float): Balancing of positive and negative weights [0.1-10.0]\n"

#     elif estimator_type == "lightgbm":
#         desc = """
# Tunable hyperparameters include:
# - n_estimators (int): Number of boosting iterations [50-1000]
# - max_depth (int): Maximum tree depth [3-15]
# - min_child_samples (int): Minimum number of data needed in a child [10-100]
# - min_split_gain (float): Minimum loss reduction required to make a further partition [0.0-1.0]
# - learning_rate (float): Boosting learning rate [0.01-0.3]
# - subsample (float): Subsample ratio of the training instance [0.6-1.0]
# - colsample_bytree (float): Subsample ratio of columns when constructing each tree [0.6-1.0]
# - reg_alpha (float): L1 regularization term on weights [0.0-10.0]
# - reg_lambda (float): L2 regularization term on weights [0.0-10.0]
# - num_leaves (int): Maximum tree leaves for base learners [10-300]
# """
#         if task_type == "classification":
#             desc += "- scale_pos_weight (float): Balancing of positive and negative weights [0.1-10.0]\n"

#     elif estimator_type == "random_forest":
#         desc = """
# Tunable hyperparameters include:
# - n_estimators (int): Number of trees in the forest
# - max_depth (int): Maximum depth of the tree
# - min_samples_split (int): Minimum number of samples required to split an internal node
# - min_samples_leaf (int): Minimum number of samples required to be at a leaf node
# - max_features (str or int): Number of features to consider when looking for the best split
# - bootstrap (bool): Whether bootstrap samples are used when building trees
# """
#         if task_type == "classification":
#             desc += "- class_weight (str): Weights associated with classes\n"

#     else:
#         desc = f"""
# For {estimator_type} estimators, using your LM domain knowledge, focus on the most relevant hyperparameters such as:
# - Regularization parameters (alpha, lambda, C)
# - Model complexity (max_depth, n_estimators, hidden_layer_sizes)
# - Learning parameters (learning_rate, solver)
# - Sampling parameters (subsample, max_features)
# - Class balance parameters (for classification tasks)
# """

#     analysis_section = ""
#     if analysis:
#         analysis_section = f"""
# ANALYSIS AND RECOMMENDATIONS:
# {analysis}
# """

#     output_string = f"""\
# Given your understanding of {estimator_type} for {task_type} tasks and general best practices, \
# along with dataset characteristics (if available), please do the following:

# 1. Explain your reasoning for an **initial** search space. Focus on casting a sufficiently wide search space that we will refine in subsequent iterations.
# 2. Then OUTPUT ONLY a Python function with this exact signature:

#     def define_search_space(trial):

# Within it, use `trial.suggest_int`, `trial.suggest_float`, `trial.suggest_loguniform`,
# `trial.suggest_categorical`, etc., to define the full hyperparameter search space.

# IMPORTANT CONSTRAINTS:
# - For `trial.suggest_loguniform()`: The low value MUST be > 0 (not 0). Use trial.suggest_float() for ranges that include 0.
# - For regularization parameters that can be 0, use `trial.suggest_float(name, 0.0, upper_bound)` instead of log uniform.
# - Use log uniform only for parameters where the minimum meaningful value is > 0 (like learning_rate).

# Avoid any other code outside that function.

# Hyperparameter descriptions:
# {desc}
# {analysis_section}
# {model_params_section}
# """
#     return output_string.strip()


def get_refine_search_space_prompt(
    top_n: str,
    last_value: float,
    best_value: float,
    all_time_configs: Optional[str] = None,
    iteration: Optional[int] = None,
    max_iterations: Optional[int] = None,
    estimator_type: str = "auto",
    task_type: str = "auto",
    estimator: Optional[BaseEstimator] = None,
) -> str:
    """Generate prompt for search space refinement."""
    header = (
        f"--- Iteration {iteration + 1}/{max_iterations} ---\n"
        if iteration is not None and max_iterations is not None
        else ""
    )
    if estimator is not None:
        model_params = render_estimator_params(estimator)
        model_params_section = f"""
Recall the parameters of the estimator or Pipeline. Note that these are simply the result of the get_params() method. Not all \
parameters can/should be tuned with Optuna, e.g. random_state, verbosity, etc. Pay attention to any transformations that may benefit from \
different search spaces, e.g. StandardScaler, OneHotEncoder, PCA, etc. Ensure you always use precise names for the hyperparameters being mindful \
of nested parameters, i.e., steps and their input data types.

Prior issues to avoid:
InvalidParameterError: The 'max_features' parameter of RandomForestClassifier must be an int in the range [1, inf), a float in the \
range (0.0, 1.0], a str among {'sqrt', 'log2'} or None. Got 'auto' instead.
Optuna ValueError: CategoricalDistribution does not support dynamic value space.

{model_params}
"""
    else:
        model_params_section = ""

    output_string = f"""
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
please suggest refinements for the search space in the next optimization round for this {task_type} task using {estimator_type}.
Consider both narrowing and expanding the search space for hyperparameters where appropriate. If improvement stagnates, be creative!

For each recommendation, please:
1. Explicitly tie back to any general best practices or patterns you are aware of regarding {estimator_type} tuning for {task_type}
2. Then, relate to the insights from the search history and explain how they align or deviate from these practices or patterns.
3. If suggesting an expansion of the search space, please provide a rationale for why a broader range could be beneficial.

Briefly summarize your reasoning for the refinements and then present the adjusted configurations.
{model_params_section}
"""
    return output_string.strip()
