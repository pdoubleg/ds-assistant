# /// script
# dependencies = [
#   "openai>=1.63.0", 
#   "streamlit>=1.31.0",
#   "pydantic>=2.0.0",
#   "pydantic-ai",
#   "xgboost",
#   "pandas",
#   "scikit-learn",
#   "seaborn",
# ]
# ///

import streamlit as st
import ast
import seaborn as sns
import math
import time
import warnings
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import TypeAlias

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.usage import Usage
from pydantic_ai.messages import ModelMessage
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedKFold, train_test_split

from run_llm_code import check_ast, run_llm_code
from utils import (
    accuracy_metric,
    auc_metric,
    get_dataset_summary_with_importance,
    get_X_y,
    make_dataset_numeric,
    make_df_numeric,
    to_code_markdown,
)

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")


def metric_ppv(
    y_true: Union[list, pd.Series], 
    y_pred: Union[list, pd.Series], 
    top_p: float
) -> float:
    """
    Computes PPV (Positive Predictive Value) at the top p% predicted probability scores.

    This metric calculates precision among the samples with the highest predicted probabilities,
    which is useful for scenarios where you care about precision in your most confident predictions.

    Args:
        y_true (Union[list, pd.Series]): Ground truth binary labels (0 or 1).
        y_pred (Union[list, pd.Series]): Predicted probabilities (not hard labels).
        top_p (float): Fraction (0 < top_p <= 1) of samples to include in the top predictions.

    Returns:
        float: Precision/PPV in the top_p most confident predictions.

    Raises:
        ValueError: If top_p is not between 0 and 1, or if y_true and y_pred have different lengths.

    Example:
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0.1, 0.9, 0.8, 0.2, 0.7]
        >>> metric_ppv(y_true, y_pred, top_p=0.4)  # Top 40% (2 samples)
        1.0  # Both top predictions were correct positives
    """
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be between 0 and 1.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")

    top_num = max(1, math.ceil(len(y_true) * top_p))

    ranked = pd.DataFrame({
        "label": pd.Series(y_true).values,
        "predicted_prob": pd.Series(y_pred).values,
    })

    top_ranked = ranked.sort_values("predicted_prob", ascending=False).head(top_num)
    ppv = top_ranked["label"].value_counts(normalize=True).get(1, 0.0)

    return ppv


class PythonCode(BaseModel):
    """A valid python code block and its reasoning"""

    reasoning: str = Field(description="Reasoning for why this code is useful")
    name: str = Field(description="Feature name")
    code: str = Field(description="Python code ready to modify the df")

    @field_validator("code", mode="after")
    def validate_code_syntax(cls, v: Any) -> str:
        """Validate that the code has proper Python syntax."""
        import ast

        try:
            # Check if it's valid Python
            ast.parse(v, mode="exec")
            return v
        except SyntaxError as e:
            # logger.error(f"Invalid Python syntax: {e}")
            raise ValueError(f"Invalid Python syntax: {e}")

    @field_validator("code", mode="after")
    def validate_code_ast(cls, v: Any) -> str:
        """Validate that the code has proper AST based on the allowed specifications."""
        try:
            check_ast(ast.parse(v, mode="exec"))
        except Exception as e:
            # logger.error(f"Invalid AST: {e}")
            raise ValueError(f"Invalid AST: {e}")
        return v

    @field_validator("code", mode="after")
    def validate_code_add_to_df(cls, v: Any) -> str:
        """Validate that the code adds the feature to the df"""
        if "df" not in v:
            # logger.error("Code must operate on a pandas DataFrame called 'df'")
            raise ValueError("Code must operate on a pandas DataFrame called 'df'")
        return v


class DroppedColumns(BaseModel):
    """Represents dropped column(s)."""

    reasoning: str = Field(description="Reason for dropping the column(s)")
    column_names: List[str] = Field(
        description="List of column names to drop", default_factory=list
    )


class FeatureGenerationResult(BaseModel):
    """Result from feature generation including multiple features and/or dropped columns."""

    reasoning: str = Field(
        description="Overall reasoning for the feature engineering decisions"
    )
    new_features: List[PythonCode] = Field(
        description="List of features written in python code", default_factory=list
    )
    dropped_columns: Optional[DroppedColumns] = Field(
        default=None,
        description="Column name(s) to drop",
    )

    @property
    def code_to_run(self) -> str:
        """Code to run the feature engineering result."""
        code_lines = []
        for feature in self.new_features:
            code_lines.append(feature.code)
            code_lines.append("")
        # Add column dropping code
        if self.dropped_columns:
            for col in self.dropped_columns.column_names:
                code_lines.append(f"df.drop(columns=['{col}'], inplace=True)")
            code_lines.append("")
        return "\n".join(code_lines)

    @property
    def feature_count(self) -> int:
        """Count the number of features in the result."""
        return len(self.new_features)

    @property
    def dropped_count(self) -> int:
        """Count the number of columns dropped in the result."""
        if self.dropped_columns:
            return len(self.dropped_columns.column_names)
        return 0

    def to_code(self) -> str:
        """Convert the feature generation result to Python code with comments."""
        code_lines = []

        # Add feature generation code
        for feature in self.new_features:
            code_lines.append(f"# {feature.name}: {feature.reasoning}")
            code_lines.append(feature.code)
            code_lines.append("")

        # Add column dropping code
        if self.dropped_columns:
            code_lines.append(f"# Dropping columns: {self.dropped_columns.reasoning}")
            for col in self.dropped_columns.column_names:
                code_lines.append(f"df.drop(columns=['{col}'], inplace=True)")
            code_lines.append("")

        return "\n".join(code_lines)


@dataclass
class FeatureEngineeringDependencies:
    """Dependencies for feature engineering agents."""

    original_dataset: pd.DataFrame
    dataset: pd.DataFrame
    target_name: str
    dataset_description: str
    current_features: List[str]
    agent_notepad: List[str]
    error_log: List[str]
    tool_output_log: List[str]
    prompt_log: List[str]


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """
You are a senior data scientist and Kaggle grandmaster whose sole mission is to design, implement, and \
iterate on FEATURE ENGINEERING strategies for a binary‑classification dataset that will be modeled using XGBoost.

Users will provide you with a summary of the dataset and the current features, along with \
results from the previous feature engineering iteration when applicable.

Users will also provide a narrative description of the dataset which may also include \
additional instructions on what to focus on, or specific requirements for the feature engineering.

You have deep knowledge of:
- Domain-specific feature engineering techniques
- Statistical transformations and aggregations
- Feature interactions and combinations
- Feature selection principles

When generating features, you always:
1. Focus on the deliverable code that will be run by the user.
2. Consider the real-world meaning of the data
3. Create features that capture important patterns
4. Avoid overfitting by being selective
5. Drop redundant or harmful features when appropriate
6. Only use the following external packages:
    - pandas
    - numpy
    - scipy
    - sklearn
    
Features can include but are not limited to:
    - Numerical: log/Box‑Cox transforms, binning, polynomial & interaction terms, \
      scaling, winsorisation.  
    - Categorical: frequency, target, leave‑one‑out & Helmert encodings; group \
      statistics; rare‑label consolidation.  
    - Text: token counts, TF‑IDF, embeddings, sentiment, key‑phrase flags.  
    - Date/Time: cyclical transforms, lags, rolling windows, period aggregates.  
    - Cross‑feature aggregates and statistical summaries.
    
<instructions>
    <instruction>Start by understanding the task with respect to user query and dataset, then decide on the most impactful subset of features to focus on.</instruction>
    <instruction>If provided, review prior results to inform your feature engineering strategy.</instruction>
    <instruction>Use the provided tools to better understand the subset of interest.</instruction>
    <instruction>Be judicious and purposeful with tool usage. Each tool should only be used once or twice.</instruction>
    <instruction>Think step by step about what information you need.</instruction>
    <instruction>If provided, review prior results to inform your feature engineering strategy. For example, if something did not work do not try it again.</instruction>
    <instruction>If you find your python code returns an error, try to fix the code or pivot to a safer approach.</instruction>
    <instruction>Every tool call should be carefully thought out and have a well defined reason for why you are calling the tool.</instruction>
</instructions>
    
    
IMPORTANT NOTES: 
- Always assume you are operating on a pandas DataFrame called "df".
- Always favor robustness over performance. For example use apply lambda with regex vs str.extract.
- You may review the target variable but do NOT include it in your feature engineering as it is not available at prediction time.

"""

FEATURE_GENERATION_PROMPT = """
Generate new features to improve classification performance.

Dataset description:
{dataset_description}

Target variable: {target_name}

Dataset summary:
{dataset_summary}

Generate up to {max_features} meaningful features that:
1. Add semantic information based on real-world knowledge and df characteristics
2. Capture patterns through combinations, transformations, or aggregations
3. Are likely to improve classification of "{target_name}"

Also identify any existing features that should be dropped because they:
- Are redundant with other features
- May cause overfitting
- Don't contribute to predicting the target

Ensure all generated code uses only existing column names from the df: {current_features}
"""


# ============================================================================
# Agents
# ============================================================================


def get_feature_generation_agent(model: str = "openai:gpt-4.1") -> Agent:
    """Get the feature generation agent."""

    feature_generation_agent = Agent(
        model=model,
        deps_type=FeatureEngineeringDependencies,
        output_type=FeatureGenerationResult,
        retries=5,
        system_prompt=SYSTEM_PROMPT,
    )

    @feature_generation_agent.output_validator
    def is_executable(
        ctx: RunContext[FeatureEngineeringDependencies], result: FeatureGenerationResult
    ) -> FeatureGenerationResult:
        """Validate that the generated code is executable."""

        try:
            run_llm_code(
                result.code_to_run,
                ctx.deps.dataset,
                convert_categorical_to_integer=True,
                fill_na=True,
            )
        except Exception as e:
            ctx.deps.error_log.append(
                f"🚨 Code validation failed:\nError: {str(e)}\nCode:\n{result.code_to_run}\nPhase: code_validation"
            )
            raise ModelRetry(f"Invalid code {result.code_to_run}: {e}") from e
        return result

    @feature_generation_agent.tool
    async def get_column_statistics(
        ctx: RunContext[FeatureEngineeringDependencies],
        reasoning: str,
        column_names: List[str],
    ) -> str:
        """Get detailed statistics for one or more columns. Handles numeric and categorical columns.

        Args:
            reasoning: Reasoning for the column statistics
            column_names: List of column names to get statistics for

        Returns:
            Summary of the column statistics for each column.

        """
        st.write(f"\n🤖 Agent: {reasoning}")
        st.write(f"\n🛠️ Tool Call: `get_column_statistics(column_names={column_names})`\n")

        df = ctx.deps.dataset
        valid_columns = [col for col in column_names if col in df.columns]

        for col in column_names:
            if col not in valid_columns:
                st.write(f"Column '{col}' not found in dataset. Retrying...")
                ctx.deps.error_log.append(f"Column '{col}' not found in dataset")
                raise ModelRetry(
                    f"Column '{col}' not found in dataset. Please select from the following columns: {valid_columns}"
                )

        summaries = []
        for column_name in column_names:
            if column_name not in df.columns:
                summaries.append(f"Column '{column_name}' not found in dataset")
                continue

            col = df[column_name]
            stats = {
                "dtype": str(col.dtype),
                "non_null_count": col.notna().sum(),
                "null_count": col.isna().sum(),
                "null_percentage": f"{col.isna().mean() * 100:.2f}%",
            }

            if pd.api.types.is_numeric_dtype(col):
                stats.update(
                    {
                        "mean": f"{col.mean():.4f}",
                        "std": f"{col.std():.4f}",
                        "min": f"{col.min():.4f}",
                        "25%": f"{col.quantile(0.25):.4f}",
                        "50%": f"{col.quantile(0.50):.4f}",
                        "75%": f"{col.quantile(0.75):.4f}",
                        "max": f"{col.max():.4f}",
                        "skew": f"{col.skew():.4f}",
                        "kurtosis": f"{col.kurtosis():.4f}",
                    }
                )
            else:
                value_counts = col.value_counts().head(10)
                stats["top_values"] = value_counts.to_dict()
                stats["unique_count"] = col.nunique()

            # Format nicely for each column
            summary = f"Column: {column_name}\n" + str(stats)
            summaries.append(summary)

        # Join summaries with a separator for readability
        summary_str = "\n---\n".join(summaries)
        tool_log_entry = f"```python\nget_column_statistics(column_names={column_names})\n```\n"
        ctx.deps.tool_output_log.append(tool_log_entry + "\n" + summary_str)
        with st.expander("Tool Details"):
            st.markdown(tool_log_entry)
            st.code(summary_str)
        time.sleep(2)
        return summary_str

    @feature_generation_agent.tool
    async def get_feature_target_correlations(
        ctx: RunContext[FeatureEngineeringDependencies],
        reasoning: str,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Get correlation matrix for numeric features with the target.

        Args:
            reasoning: Reasoning for the correlation analysis
            columns: Optional list of columns to analyze. If None, all numeric columns will be analyzed.

        Returns:
            String summary of the correlation matrix
        """
        st.write(f"\n🤖 Agent: {reasoning}")
        st.write(f"\n🛠️ Tool Call: `get_feature_target_correlations(columns={columns})`\n")

        df = ctx.deps.dataset
        target_name = ctx.deps.target_name
        if columns is not None:
            valid_columns = [col for col in columns if col in df.columns]
            for col in columns:
                if col not in valid_columns:
                    st.write(f"Column '{col}' not found in dataset. Retrying...")
                    ctx.deps.error_log.append(f"Column '{col}' not found in dataset")
                    raise ModelRetry(
                        f"Column '{col}' not found in dataset. Please select from the following columns: {valid_columns}"
                    )

        if columns is not None:
            numeric_cols = [
                col for col in columns if pd.api.types.is_numeric_dtype(df[col])
            ]
        else:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if target_name not in numeric_cols:
            return "Target variable is not numeric, correlation analysis not applicable"

        if len(numeric_cols) == 0:
            return "No numeric features found, correlation analysis not applicable"

        corr_columns = [col for col in numeric_cols if col != target_name]

        # Calculate correlations with target
        correlations = (
            df[corr_columns]
            .corrwith(df[target_name])
            .sort_values(ascending=False)
            .round(4)
        )
        tool_log_entry = f"```python\nget_feature_target_correlations(columns={columns})\n```\n"
        ctx.deps.tool_output_log.append(tool_log_entry + "\n" + correlations.to_string())
        with st.expander("Tool Details"):
            st.markdown(tool_log_entry)
            st.code(correlations.to_string())
        time.sleep(2)
        return f"Correlation with target '{target_name}':\n" + correlations.to_string()

    @feature_generation_agent.tool
    async def get_correlation_pairs_summary(
        ctx: RunContext[FeatureEngineeringDependencies],
        reasoning: str,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Get summary of highly correlated feature pairs.

        Args:
            reasoning: Reasoning for the correlation analysis
            columns: Optional list of columns to analyze. If None, all numeric columns will be analyzed.

        Returns:
            String summary of the highly correlated feature pairs
        """
        st.write(f"\n🤖 Agent: {reasoning}")
        st.write(f"\n🛠️ Tool Call: `get_correlation_pairs_summary(columns={columns})`\n")

        df = ctx.deps.dataset
        if columns is not None:
            valid_columns = [col for col in columns if col in df.columns]
            for col in columns:
                if col not in valid_columns:
                    st.write(f"Column '{col}' not found in dataset. Retrying...")
                    ctx.deps.error_log.append(f"Column '{col}' not found in dataset")
                    raise ModelRetry(
                        f"Column '{col}' not found in dataset. Please select from the following columns: {valid_columns}"
                    )

        columns = columns or df.select_dtypes(include=np.number).columns.tolist()
        corr_matrix = df[columns].corr().abs()
        pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
            .rename(
                columns={"level_0": "Feature1", "level_1": "Feature2", 0: "Correlation"}
            )
        )
        high_corr_pairs = pairs[pairs["Correlation"] >= 0.8]
        if high_corr_pairs.empty:
            return "No feature pairs found with correlation above threshold."

        output_string = high_corr_pairs.sort_values(
            by="Correlation", ascending=False
        ).to_string(index=False)
        tool_log_entry = f"```python\nget_correlation_pairs_summary(columns={columns})\n```\n"
        ctx.deps.tool_output_log.append(tool_log_entry + "\n" + output_string)
        with st.expander("Tool Details"):
            st.markdown(tool_log_entry)
            st.code(output_string)
        time.sleep(2)
        return output_string

    @feature_generation_agent.tool
    async def check_for_outliers(
        ctx: RunContext[FeatureEngineeringDependencies], reasoning: str
    ) -> str:
        """Detect outliers in numeric columns using IQR method.

        Args:
            reasoning: Reasoning for the outlier detection

        Returns:
            String summary of outlier counts and percentages for columns with outliers,
            or "No outliers detected" if none are found.
        """
        st.write(f"\n🤖 Agent: {reasoning}")
        st.write("\n🛠️ Tool Call: `check_for_outliers()`\n")

        df = ctx.deps.dataset
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out bool columns and int columns with only 0s and 1s
        numeric_cols = [
            col
            for col in numeric_cols
            if not (
                df[col].dtype == bool
                or (
                    df[col].dtype in ["int32", "int64"]
                    and set(df[col].unique()).issubset({0, 1})
                )
            )
        ]

        summary = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = (df[col] < lower) | (df[col] > upper)
            count = outliers.sum()
            if count > 0:
                pct = (count / len(df) * 100).round(2)
                summary.append(f"{col}: {count} outliers ({pct}% of values)")

        if not summary:
            return "No outliers detected"
        time.sleep(2)
        tool_log_entry = "```python\ncheck_for_outliers()\n```\n"
        ctx.deps.tool_output_log.append(tool_log_entry + "\n" + "\n".join(summary))
        with st.expander("Tool Details"):
            st.markdown(tool_log_entry)
            st.code("\n".join(summary))
        return "\n".join(summary)

    @feature_generation_agent.tool
    async def get_mutual_information_summary(
        ctx: RunContext[FeatureEngineeringDependencies],
        reasoning: str,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Get mutual information (classification) summary for features with the target.

        Args:
            reasoning: Reasoning for the mutual information analysis
            columns: Optional list of columns to analyze. If None, all columns will be analyzed.

        Returns:
            String summary of the mutual information
        """
        st.write(f"\n🤖 Agent: {reasoning}")
        st.write(f"\n🛠️ Tool Call: `get_mutual_information_summary(columns={columns})`\n")

        df = ctx.deps.dataset
        target = ctx.deps.target_name
        if columns is not None:
            valid_columns = [col for col in columns if col in df.columns]
            for col in columns:
                if col not in valid_columns:
                    st.write(f"Column '{col}' not found in dataset. Retrying...")
                    ctx.deps.error_log.append(f"Column '{col}' not found in dataset")
                    raise ModelRetry(
                        f"Column '{col}' not found in dataset. Please select from the following columns: {valid_columns}"
                    )

        columns = columns or df.drop(columns=[target]).columns.tolist()
        X = df[columns].copy()
        for col in X.select_dtypes(include=["object", "category"]):
            X[col] = pd.factorize(X[col])[0]
        X = X.fillna(-999)

        y = df[target]

        mi = mutual_info_classif(X, y)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False).round(4)

        output_string = (
            f"Mutual Information with target '{target}':\n" + mi_series.to_string()
        )
        tool_log_entry = f"```python\nget_mutual_information_summary(columns={columns})\n```\n"
        ctx.deps.tool_output_log.append(tool_log_entry + "\n" + output_string)
        with st.expander("Tool Details"):
            st.markdown(tool_log_entry)
            st.code(output_string)
        time.sleep(2)
        return output_string

    return feature_generation_agent


class CAAFETransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn–compatible transformer that uses an LLM (e.g. GPT-4o)
    to iteratively generate new features (CAAFE algorithm), evaluating each batch
    via RepeatedKFold and keeping only those that show improvement.
    
    Supports:

    - Logging (no direct prints/displays)
    - agent_notepad to record each iteration's summary
    - Saving/loading the final feature-generation code (as .py or .md).
    - In-fit option to "load prior code" and skip regeneration.

    Parameters
    ----------
    target_name : str
        Name of the target column in your DataFrame.

    dataset_description : Optional[str]
        A textual description of the dataset, used in the LLM prompt.
        If None, you must pass dataset_description later in fit().

    max_features : int, default=10
        Maximum number of new features to request from the LLM each iteration.

    base_classifier : Optional[object], default=None
        A scikit-learn–compatible classifier used during fold-based evaluation.
        If None, defaults to XGBClassifier(use_label_encoder=False, eval_metric="logloss").
        Note: The transformer does *not* fit this classifier for final predictions;
        it only uses it to compute fold‐by‐fold metrics when evaluating new features.

    optimization_metric : str, {"accuracy", "auc", "ppv"}, default="accuracy"
        Which metric to optimize when comparing baseline vs. enhanced:
        - "accuracy": uses accuracy_score
        - "auc": uses roc_auc_score (binary or multiclass via ovr)
        - "ppv": uses PPV at top_p% of predictions

    iterations : int, default=10
        Maximum number of LLM‐driven feature‐generation iterations.

    n_splits : int, default=10
        Number of folds in RepeatedKFold.

    n_repeats : int, default=2
        Number of repeats in RepeatedKFold.

    random_state : int, default=42
        Random seed for reproducibility.

    n_samples : int, default=10
        Number of sample rows to include in dataset summary for LLM.

    cv_folds : int, default=5
        Number of cross-validation folds for feature importance calculation.

    top_p : float, default=0.05
        Fraction of top predictions to use for PPV calculation (only used when optimization_metric="ppv").

    llm_model : str, default="gpt-4o-mini"
        Name of the OpenAI (LLM) model to invoke.

    """

    def __init__(
        self,
        target_name: str,
        dataset_description: Optional[str] = None,
        max_features: int = 10,
        base_classifier: Optional[Any] = None,
        optimization_metric: str = "accuracy",
        iterations: int = 10,
        llm_model: str = "gpt-4o",
        n_splits: int = 10,
        n_repeats: int = 2,
        random_state: int = 42,
        n_samples: int = 10,
        cv_folds: int = 5,
        top_p: float = 0.05,
    ) -> None:
        """
        Initialize CAAFETransformer with structured logging capabilities.

        Parameters
        ----------
        target_name : str
            Name of the target column in your DataFrame.
        dataset_description : Optional[str]
            A textual description of the dataset, used in the LLM prompt.
        max_features : int, default=10
            Maximum number of new features to request from the LLM each iteration.
        base_classifier : Optional[object], default=None
            A scikit-learn–compatible classifier used during fold-based evaluation.
        optimization_metric : str, {"accuracy", "auc", "ppv"}, default="accuracy"
            Which metric to optimize when comparing baseline vs. enhanced:
            - "accuracy": uses accuracy_score
            - "auc": uses roc_auc_score (binary or multiclass via ovr)
            - "ppv": uses PPV at top_p% of predictions
        iterations : int, default=10
            Maximum number of LLM‐driven feature‐generation iterations.
        llm_model : str, default="gpt-4o"
            Name of the OpenAI (LLM) model to invoke.
        n_splits : int, default=10
            Number of folds in RepeatedKFold.
        n_repeats : int, default=2
            Number of repeats in RepeatedKFold.
        random_state : int, default=42
            Random seed for reproducibility.
        n_samples : int, default=10
            Number of sample rows to include in dataset summary for LLM.
        cv_folds : int, default=5
            Number of cross-validation folds for feature importance calculation.
        top_p : float, default=0.05
            Fraction of top predictions to use for PPV calculation (only used when optimization_metric="ppv").
        """
        self.target_name = target_name
        self.dataset_description = dataset_description or ""
        self.max_features = max_features
        self.optimization_metric = optimization_metric.lower()
        
        # Validate optimization metric
        if self.optimization_metric not in ["accuracy", "auc", "ppv"]:
            raise ValueError("optimization_metric must be one of: 'accuracy', 'auc', 'ppv'")
            
        self.iterations = iterations
        self.llm_model = llm_model
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_samples = n_samples
        self.cv_folds = cv_folds
        self.top_p = top_p

        # If no base classifier is given, default to XGBClassifier
        if base_classifier is None:
            self.base_classifier = xgb.XGBClassifier(
                objective="binary:logistic",
                use_label_encoder=False,
                eval_metric="logloss",
                enable_categorical=False,
                random_state=self.random_state,
            )
        else:
            self.base_classifier = base_classifier

        self.deps: FeatureEngineeringDependencies = None
        self.feature_agent = get_feature_generation_agent(model=self.llm_model)

        # Will store the final code (concatenated accepted iterations)
        self.code: str = ""
        self.full_code: str = ""

        # Each iteration's code blocks
        self.feature_code_history: List[str] = []

        # After fit, this becomes True
        self._is_fitted = False

        # agent_notepad is inside deps; we'll initialize in fit()
        # The format will be a list of dicts, each summarizing one iteration of evaluation
        self.agent_notepad: List[Dict[str, Any]] = []

        # Store fold‐by‐fold and summary stats for each iteration
        # (baseline vs. enhanced, t-stat, p-value, improvement, significance, etc.)
        self.evaluation_history: List[Dict[str, Any]] = []

        # Keep track of best overall score (primary metric) during fit
        self.baseline_auc: float = -np.inf
        self.baseline_acc: float = -np.inf
        self.baseline_ppv: float = -np.inf
        self.best_score: float = -np.inf
        self.best_acc: float = -np.inf
        self.best_auc: float = -np.inf
        self.best_ppv: float = -np.inf

        # Performance tracking
        self._start_time: Optional[float] = None
        self._features_accepted: int = 0
        self._features_dropped: int = 0
        self._features_rejected: int = 0
        self.accepted_features: List[str] = []
        self.rejected_features: List[str] = []
        self.features_dropped: List[str] = []

        # Cost tracking
        self.usages: List[Usage] = []
        

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_description: Optional[str] = None,
        load_code_path: Optional[str] = None,
        show_prompts: bool = False,
        **kwargs,
    ) -> "CAAFETransformer":
        """
        Fit the transformer with comprehensive structured logging.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame (does not include target).
        y : pd.Series
            Target variable (used for evaluating each candidate set of features).
        dataset_description : Optional[str]
            If provided, overrides `self.dataset_description`.
        load_code_path : Optional[str]
            Path to a .py or .md file containing previously‐generated feature code.
        show_prompts : bool, default=False
            If True, prints the prompts sent to the LLM.
        """
        # Start overall timing
        self._start_time = time.time()
        st.write(
            "Starting CAAFETransformer.fit(): running iterative feature engineering."
        )

        # Update dataset_description if the user passed a new one
        if dataset_description is not None:
            self.dataset_description = dataset_description

        # Combine X + y into a single DataFrame for in‐memory operations
        combined = X.copy()
        combined[self.target_name] = y.values
        combined_df = combined

        self.deps = FeatureEngineeringDependencies(
            original_dataset=combined_df,
            dataset=combined_df,
            target_name=self.target_name,
            dataset_description=self.dataset_description,
            current_features=[col for col in X.columns],
            agent_notepad=[],
            error_log=[],
            tool_output_log=[],
            prompt_log=[],
        )

        # Log initialization
        st.code(
            f"CAAFE transformer initialization completed:\n"
            f"  Target: {self.target_name}\n"
            f"  Dataset shape: {combined_df.shape}\n"
            f"  Original features: {len(X.columns)}\n"
            f"  Max features per iteration: {self.max_features}\n"
            f"  Max iterations: {self.iterations}\n"
            f"  Optimization metric: {self.optimization_metric}\n"
            f"  LLM model: {self.llm_model}\n"
            f"  CV splits: {self.n_splits}\n"
            f"  CV repeats: {self.n_repeats}"
        )

        # If load_code_path is provided, read code from disk, skip iteration loop
        if load_code_path:
            st.write(
                f"Loading existing feature-generation code from {load_code_path}"
            )
            try:
                with open(load_code_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                    if load_code_path.lower().endswith(".md"):
                        # strip fenced code if present
                        if raw.startswith("```") and raw.strip().endswith("```"):
                            # Remove leading/trailing ```python and ```
                            lines = raw.splitlines()
                            if lines[0].startswith("```") and lines[-1].startswith(
                                "```"
                            ):
                                # remove first and last line
                                raw = "\n".join(lines[1:-1])
                    self.code = raw
                    self.feature_code_history = [raw]  # at least one entry
                    self.deps.agent_notepad.append(
                        {
                            "iteration": "loaded_from_disk",
                            "source_path": load_code_path,
                            "notes": "Code loaded; no LLM generation performed",
                        }
                    )
                self._is_fitted = True

                # Log successful code loading
                st.write(
                    f"Feature code loaded successfully (took {time.time() - self._start_time:.2f}s)"
                )
                return self
            except Exception as e:
                st.write(
                    f"Failed to load code from {load_code_path}: {e}"
                )
                raise IOError(f"Failed to load code from {load_code_path}: {e}")

        # Otherwise, run full iterative feature-engineering process
        st.write("Starting iterative feature engineering process...")

        # 1) Evaluate baseline stats (no new features)
        st.write(
            "\n\n→ Evaluating baseline performance (no added features)...\n"
        )
        baseline_scores, baseline_stats = self.evaluate_features(
            full_code="",
            code="",
        )

        # Extract baseline metric (primary) and record
        # Calculate mean and std from the lists returned by evaluate_features
        self.baseline_auc = (
            np.mean(baseline_stats["auc"]) if baseline_stats["auc"] else 0.0
        )
        baseline_auc_std = (
            np.std(baseline_stats["auc"]) if baseline_stats["auc"] else 0.0
        )
        self.baseline_acc = (
            np.mean(baseline_stats["accuracy"]) if baseline_stats["accuracy"] else 0.0
        )
        baseline_acc_std = (
            np.std(baseline_stats["accuracy"]) if baseline_stats["accuracy"] else 0.0
        )
        self.baseline_ppv = (
            np.mean(baseline_stats["ppv"]) if baseline_stats["ppv"] else 0.0
        )
        baseline_ppv_std = (
            np.std(baseline_stats["ppv"]) if baseline_stats["ppv"] else 0.0
        )

        baseline_primary = (
            self.baseline_auc
            if self.optimization_metric == "auc"
            else self.baseline_acc
            if self.optimization_metric == "accuracy" 
            else self.baseline_ppv
        )
        st.info(
            f"\nBaseline ROC AUC: {self.baseline_auc:.3f} (±{baseline_auc_std:.3f})"
        )
        st.info(
            f"\nBaseline Accuracy: {self.baseline_acc:.3f} (±{baseline_acc_std:.3f})"
        )
        st.info(
            f"\nBaseline PPV@{self.top_p:.1%}: {self.baseline_ppv:.3f} (±{baseline_ppv_std:.3f})"
        )

        self.best_score = baseline_primary
        self.best_acc = self.baseline_acc
        self.best_auc = self.baseline_auc
        self.best_ppv = self.baseline_ppv

        # Seed the iteration loop
        consecutive_no_improvement = 0
        previous_iteration_reasoning = ""

        for itr in range(self.iterations):
            # Log iteration start
            st.write(f"\n\n--- Iteration {itr + 1}/{self.iterations} ---\n")

            # Update deps with the current dataset and feature list
            self.deps.current_features = [
                col for col in self.deps.dataset.columns if col != self.target_name
            ]
            current_df = self.deps.dataset

            # Summarize dataset for the LLM (10 sample rows)
            dataset_summary = get_dataset_summary_with_importance(
                df_train=self.deps.dataset,
                target_name=self.target_name,
                n_samples=self.n_samples,
                cv_folds=self.cv_folds,
                model=self.base_classifier,
            )

            # Build the prompt
            prompt = FEATURE_GENERATION_PROMPT.format(
                dataset_description=self.dataset_description,
                target_name=self.target_name,
                current_features=", ".join(self.deps.current_features),
                dataset_summary=dataset_summary,
                max_features=self.max_features,
            )

            prompt = f"---ITERATION {itr + 1}---\n\n{prompt}"

            # Include previous iteration results if available (skip first iteration)
            if itr > 0 and self.deps.agent_notepad:
                prompt += f"\nPrevious iteration results to take into consideration:\n{self.get_formatted_agent_notepad(n=1)}\n"
                prompt += (
                    f"Previous iteration reasoning: {previous_iteration_reasoning}\n"
                )

            if show_prompts:
                st.write(f"\n\nPrompt for iteration {itr + 1}:\n{prompt}\n")
            self.deps.prompt_log.append(f"\n\nPrompt for iteration {itr + 1}:\n{prompt}\n")
            try:
                # 2) Ask the LLM agent to propose new features
                llm_start_time = time.time()
                st.write("\n\n→ Invoking LLM for new feature generation...\n")

                result = self.feature_agent.run_sync(prompt, deps=self.deps)
                llm_duration = time.time() - llm_start_time

                feature_result = result.output
                self.usages.append(result.usage())
                previous_iteration_reasoning = feature_result.reasoning

                # Log LLM interaction
                st.write(
                    f"\nLLM Feature Engineering Reasoning:\n🤖 Agent: {feature_result.reasoning}"
                )
                st.write(
                    f"\nProposing {len(feature_result.new_features)} new features"
                    f"\n\nDropping {feature_result.dropped_count} existing columns."
                    f"\n\n✅ LLM interaction completed - Iteration {itr + 1} took {llm_duration:.2f}s"
                )

                # Convert to executable Python code
                code = feature_result.code_to_run
                st.write(
                    "\nGenerated code snippet:\n\n"
                    + to_code_markdown(feature_result.to_code())
                )

                # 3) Use incremental evaluation pattern
                try:
                    # 4) Incremental evaluation: (original + previous) vs (original + previous + new)
                    st.write(
                        "\n\n→ Evaluating incremental benefit of proposed features...\n"
                    )

                    eval_start_time = time.time()
                    old_results, new_results = self.evaluate_features(
                        full_code=self.full_code, code=code
                    )
                    eval_duration = time.time() - eval_start_time

                    # Record this iteration's metrics into self.evaluation_history
                    iteration_record = {
                        "iteration": itr + 1,
                        "old_results": old_results,
                        "new_results": new_results,
                        "evaluation_duration": eval_duration,
                    }
                    self.evaluation_history.append(iteration_record)

                    # Also append to agent_notepad for LLM memory
                    # Extract feature names from the feature_result
                    feature_names = [
                        feature.name for feature in feature_result.new_features
                    ]
                    if feature_result.dropped_columns:
                        dropped_feature_names = (
                            feature_result.dropped_columns.column_names
                        )
                    else:
                        dropped_feature_names = []

                    # 5) Decide if we keep or discard these features
                    improvement_roc = np.nanmean(new_results["auc"]) - np.nanmean(
                        old_results["auc"]
                    )
                    improvement_acc = np.nanmean(new_results["accuracy"]) - np.nanmean(
                        old_results["accuracy"]
                    )
                    improvement_ppv = np.nanmean(new_results["ppv"]) - np.nanmean(
                        old_results["ppv"]
                    )
                    
                    # Determine significance based on the primary optimization metric
                    primary_improvement = (
                        improvement_roc if self.optimization_metric == "auc"
                        else improvement_acc if self.optimization_metric == "accuracy"
                        else improvement_ppv
                    )
                    is_significant = primary_improvement > 0

                    # Get all three metrics for comprehensive reporting
                    baseline_acc = np.nanmean(old_results["accuracy"])
                    enhanced_acc = np.nanmean(new_results["accuracy"])
                    baseline_auc = np.nanmean(old_results["auc"])
                    enhanced_auc = np.nanmean(new_results["auc"])
                    baseline_ppv = np.nanmean(old_results["ppv"])
                    enhanced_ppv = np.nanmean(new_results["ppv"])
                    acc_improvement = improvement_acc
                    auc_improvement = improvement_roc
                    ppv_improvement = improvement_ppv

                    # Log detailed feature evaluation
                    # Log a human-readable summary of feature evaluation results
                    eval_msg = (
                        f"\nFeature Evaluation Results (Iteration {itr + 1}):\n"
                        f"  Features Added: {', '.join(feature_names) if feature_names else 'None'}\n"
                        f"  Features Dropped: {', '.join(dropped_feature_names) if dropped_feature_names else 'None'}\n"
                        f"  Baseline Metrics: ACC {baseline_acc:.4}, ROC AUC {baseline_auc:.4}, PPV@{self.top_p:.1%} {baseline_ppv:.4}\n"
                        f"  Updated Metrics: ACC {enhanced_acc:.4}, ROC AUC {enhanced_auc:.4}, PPV@{self.top_p:.1%} {enhanced_ppv:.4}\n"
                        f"  Improvements: ACC {acc_improvement:+.4}, ROC AUC {auc_improvement:+.4}, PPV {ppv_improvement:+.4}\n"
                        f"  Primary Metric ({self.optimization_metric}): {primary_improvement:+.4}\n"
                        f"  Significant: {is_significant}\n"
                        f"  Evaluation Time: {eval_duration:.2f}s"
                    )
                    st.code(eval_msg)

                    # Create formatted summary string for agent memory
                    performance_summary = (
                        f"Iteration {itr + 1}\n"
                        f"Features created: {', '.join(feature_names) if feature_names else 'None'}\n"
                        f"Features dropped: {', '.join(dropped_feature_names) if dropped_feature_names else 'None'}\n"
                        f"Performance before adding features ROC {baseline_auc:.4}, ACC {baseline_acc:.4}, PPV@{self.top_p:.1%} {baseline_ppv:.4}.\n"
                        f"Performance after adding features ROC {enhanced_auc:.4}, ACC {enhanced_acc:.4}, PPV@{self.top_p:.1%} {enhanced_ppv:.4}.\n"
                        f"Improvement ROC {auc_improvement:+.4}, ACC {acc_improvement:+.4}, PPV {ppv_improvement:+.4}.\n"
                        f"Primary optimization metric ({self.optimization_metric}): {primary_improvement:+.4}.\n"
                        f"Note: {'Code was ACCEPTED and applied to the dataset. Columns were successfully added/dropped.' if is_significant else 'Code was REJECTED and NOT applied to the dataset.'}"
                    )

                    summary_record = {
                        "iteration": itr + 1,
                        "feature_names": feature_names,
                        "dropped_features": dropped_feature_names,
                        "baseline_roc": baseline_auc,
                        "enhanced_roc": enhanced_auc,
                        "baseline_acc": baseline_acc,
                        "enhanced_acc": enhanced_acc,
                        "baseline_ppv": baseline_ppv,
                        "enhanced_ppv": enhanced_ppv,
                        "roc_improvement": auc_improvement,
                        "acc_improvement": acc_improvement,
                        "ppv_improvement": ppv_improvement,
                        "primary_improvement": primary_improvement,
                        "optimization_metric": self.optimization_metric,
                        "significant": str(is_significant),
                        "code_retained": str(is_significant),
                        "formatted_summary": performance_summary,
                    }

                    self.deps.agent_notepad.append(summary_record)

                    # Continue with processing the decision
                    if is_significant:
                        st.success(
                            "\n✓ Proposed features show improvement: Keeping them. ",
                            icon="✅",
                        )
                        if acc_improvement > 0:
                            st.info(f"\nAccuracy +{acc_improvement:.4}")
                        if auc_improvement > 0:
                            st.info(f"\nROC AUC +{auc_improvement:.4}")
                        if ppv_improvement > 0:
                            st.info(f"\nPPV@{self.top_p:.1%} +{ppv_improvement:.4}")

                        # Update our accumulated feature code
                        self.full_code += code
                        self.feature_code_history.append(code)
                        self._features_accepted += len(feature_names)
                        self._features_dropped += len(dropped_feature_names)
                        self.accepted_features.extend(feature_names)
                        self.features_dropped.extend(dropped_feature_names)

                        # Apply the complete feature code to get the enhanced dataset for next iteration context
                        enhanced_df = run_llm_code(
                            self.full_code,
                            self.deps.original_dataset,
                            convert_categorical_to_integer=True,
                        )
                        self.deps.dataset = enhanced_df

                        # Update current_df to be the enhanced version for next iteration
                        prev_shape = current_df.shape
                        current_df = enhanced_df
                        new_shape = current_df.shape

                        st.write(
                            f"\n\n→ Updated dataset shape: {prev_shape} → {new_shape}\n"
                        )

                        # Update the best_score if this iteration's metrics are better
                        current_primary = np.nanmean(
                            new_results[self.optimization_metric]
                        )
                        if current_primary > self.best_score:
                            self.best_score = current_primary

                        if self.best_acc < np.nanmean(new_results["accuracy"]):
                            self.best_acc = np.nanmean(new_results["accuracy"])
                        if self.best_auc < np.nanmean(new_results["auc"]):
                            self.best_auc = np.nanmean(new_results["auc"])
                        if self.best_ppv < np.nanmean(new_results["ppv"]):
                            self.best_ppv = np.nanmean(new_results["ppv"])

                        # Reset counter of "no improvement"
                        consecutive_no_improvement = 0

                    else:
                        st.warning(
                            "\n✗ Proposed features did NOT show improvement: Discarding.",
                            icon="❌",
                        )
                        consecutive_no_improvement += 1
                        self._features_rejected += len(feature_names)
                        self.rejected_features.extend(feature_names)

                except Exception as e:
                    st.write(
                        f"Error during feature evaluation (iteration {itr + 1}): {str(e)}\n"
                        f"Code length: {len(self.full_code)} chars\n"
                        f"Full error: {type(e).__name__}: {str(e)}"
                    )
                    consecutive_no_improvement += 1

            except Exception as e:
                st.write(
                    f"Error during feature evaluation (iteration {itr + 1}): {str(e)}\n"
                    f"Code length: {len(self.full_code)} chars\n"
                    f"Full error: {type(e).__name__}: {str(e)}"
                )
                consecutive_no_improvement += 1

            # Log iteration end
            st.code(
                f"\nIteration {itr + 1} completed: "
                f"\nConsecutive no improvement: {consecutive_no_improvement}, "
                f"\nCurrent best primary metric ({self.optimization_metric}) score: {self.best_score:.3f}"
                f"\nCurrent best accuracy score: {self.best_acc:.3f}"
                f"\nCurrent best ROC AUC score: {self.best_auc:.3f}"
                f"\nCurrent best PPV@{self.top_p:.1%} score: {self.best_ppv:.3f}"
            )

            # Early‐stopping condition
            if consecutive_no_improvement >= 3:
                st.warning(
                    f"\nNo improvement for {consecutive_no_improvement} consecutive iterations → early stopping."
                    f"\nCompleted iterations: {itr + 1}, "
                    f"\nTotal planned iterations: {self.iterations}"
                )
                break

        # End of iteration loop - log final summary
        total_duration = time.time() - self._start_time

        st.code(
            f"\nFinal summary: "
            f"\nTotal iterations={itr + 1}, "
            f"\nFeatures created, {', '.join(self.accepted_features)}"
            f"\nFeatures dropped, {', '.join(self.features_dropped)}"
            f"\nFeatures rejected, {', '.join(self.rejected_features)}"
            f"\nOriginal ROC AUC: {float(self.baseline_auc)}, "
            f"\nOriginal accuracy: {float(self.baseline_acc)}, "
            f"\nOriginal PPV@{self.top_p:.1%}: {float(self.baseline_ppv)}, "
            f"\nFinal ROC AUC: {float(self.best_auc)}, "
            f"\nFinal accuracy: {float(self.best_acc)}, "
            f"\nFinal PPV@{self.top_p:.1%}: {float(self.best_ppv)}, "
            f"\nTotal duration={total_duration}"
        )

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame, convert_categorical_to_integer: bool = False) -> pd.DataFrame:
        """
        Apply the LLM‐generated feature code (self.full_code) to a new DataFrame X.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain the same original feature columns that were present when fit() was called.
            Does NOT include the target column.
        convert_categorical_to_integer : bool, default=False
            If True, convert categorical columns to integer.

        Returns
        -------
        pd.DataFrame
            A new DataFrame containing the original columns plus any new columns created by running
            `run_llm_code(self.full_code, ...)`.  If the code references columns not present, you'll get
            an error.  Conversely, if new categories appear, they will appear in the output.

        Raises
        ------
        NotFittedError
            If fit() has not been called yet.
        """
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError(
                "CAAFETransformer not fitted yet; call fit() before transform()."
            )

        # Make a copy so we don't overwrite the user's X
        df_in = X.copy()
        try:
            df_out = run_llm_code(self.full_code, df_in, convert_categorical_to_integer=convert_categorical_to_integer)
        except Exception as e:
            st.warning(
                f"Error applying self.full_code in transform: {type(e).__name__}: {e}"
            )
            raise

        return df_out

    def save_code(self, filepath: str) -> None:
        """
        Save self.full_code to disk as either a .py or .md.

        - If filepath ends in '.py', writes raw Python code.
        - If filepath ends in '.md', wraps self.full_code in a triple‐backtick fence labeled 'python'.

        After writing, logs the location to self.logger.
        """
        if not self.full_code:
            st.warning("No feature-generation code available to save.")
            return

        if filepath.lower().endswith(".py"):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.full_code)
                st.success(f"Feature-generation code saved to {filepath}")
            except Exception as e:
                st.warning(f"Failed to save code to {filepath}: {e}")
        elif filepath.lower().endswith(".md"):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("```python\n")
                    f.write(self.full_code)
                    f.write("\n```")
                st.write(
                    f"Feature-generation code saved (as Markdown) to {filepath}"
                )
            except Exception as e:
                st.write(f"Failed to save code to {filepath}: {e}")
        else:
            st.write(
                "Unrecognized extension for save_code: use '.py' or '.md'."
            )

    def evaluate_features(
        self,
        full_code: str,
        code: str,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        old_results = {"accuracy": [], "auc": [], "ppv": []}
        new_results = {"accuracy": [], "auc": [], "ppv": []}

        rskf = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        # Use original dataset for splitting
        original_df = self.deps.original_dataset

        for train_idx, valid_idx in rskf.split(original_df):
            df_train, df_valid = (
                original_df.iloc[train_idx],
                original_df.iloc[valid_idx],
            )

            target_train = df_train[self.target_name]
            target_valid = df_valid[self.target_name]
            df_train = df_train.drop(columns=[self.target_name])
            df_valid = df_valid.drop(columns=[self.target_name])

            df_train_extended = df_train.copy()
            df_valid_extended = df_valid.copy()

            try:
                df_train = run_llm_code(
                    full_code,
                    df_train,
                )
                df_valid = run_llm_code(
                    full_code,
                    df_valid,
                )
                df_train_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_train_extended,
                )
                df_valid_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_valid_extended,
                )

            except Exception as e:
                st.write(
                    f"Error during fold evaluation: {type(e).__name__}: {e}"
                )
                continue
            # Add the target column back to df_train
            df_train[self.target_name] = target_train
            df_valid[self.target_name] = target_valid
            df_train_extended[self.target_name] = target_train
            df_valid_extended[self.target_name] = target_valid

            old_result = self._evaluate_dataset(df_train, df_valid, self.target_name)
            new_result = self._evaluate_dataset(
                df_train_extended, df_valid_extended, self.target_name
            )

            for metric in ["accuracy", "auc", "ppv"]:
                old_results[metric].append(old_result[metric])
                new_results[metric].append(new_result[metric])

        return old_results, new_results

    def _evaluate_dataset(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_name: str
    ):
        """Evaluates model performance on train and test datasets.

        This method takes training and test dataframes, preprocesses them by converting categorical 
        features to numeric, trains the base classifier, and evaluates performance using accuracy
        and AUC metrics.

        Args:
            df_train (pd.DataFrame): Training data including features and target
            df_test (pd.DataFrame): Test data including features and target  
            target_name (str): Name of target column in dataframes

        Returns:
            dict: Dictionary containing accuracy and AUC scores
                - accuracy (float): Classification accuracy on test set
                - auc (float): Area under ROC curve on test set

        """
        df_train, df_test = df_train.copy(), df_test.copy()
        feature_names = list(df_train.drop(target_name, axis=1).columns)
        df_train, _, encoder = make_dataset_numeric(
            df_train,
            None,
            target_name,
            categorical_cols=feature_names,
            return_encoder=True,
        )

        df_test = make_df_numeric(
            df_test, encoder=encoder, categorical_cols=feature_names
        )

        if df_test is not None:
            test_x, test_y = get_X_y(df_test, target_name=target_name)

        x, y = get_X_y(df_train, target_name=target_name)

        self.base_classifier.fit(X=x, y=y)
        probs = self.base_classifier.predict_proba(test_x)
        acc = float(accuracy_metric(test_y, probs))
        auc = float(auc_metric(test_y, probs))
        
        # Compute PPV at top_p% - extract positive class probabilities
        if probs.shape[1] == 2:  # Binary classification
            positive_probs = probs[:, 1]  # Probabilities for positive class
        else:  # Single class (should not happen in binary classification)
            positive_probs = probs.flatten()
            
        ppv = float(metric_ppv(test_y, positive_probs, top_p=self.top_p))

        return {"accuracy": acc, "auc": auc, "ppv": ppv}

    def get_formatted_agent_notepad(self, n: int = 2) -> str:
        """
        Get the agent_notepad formatted as a string suitable for including in prompts.

        Parameters
        ----------
        n : int, default=2
            Number of most recent iterations to include in the formatted output.
            If n <= 0, all iterations will be included.

        Returns
        -------
        str
            A formatted string containing the most recent n iteration summaries,
            suitable for including in LLM prompts to provide context about previous iterations.

        Example output:
            "Iteration 1
            Features created: feature_1, feature_2
            Features dropped: old_feature
            Performance before adding features ROC 0.888, ACC 0.700.
            Performance after adding features ROC 0.987, ACC 0.980.
            Improvement ROC +0.099, ACC +0.280. Code was executed and changes to df retained.

            Iteration 2
            ..."
        """
        if not hasattr(self, "deps") or not self.deps or not self.deps.agent_notepad:
            return "No iteration history available."

        formatted_summaries = []
        for record in self.deps.agent_notepad:
            if "formatted_summary" in record:
                formatted_summaries.append(record["formatted_summary"])
            else:
                # Fallback for older format
                iteration = record.get("iteration", "Unknown")
                formatted_summaries.append(f"Iteration Results: {iteration}")

        # Limit to the most recent n entries if n > 0
        if n > 0 and len(formatted_summaries) > n:
            formatted_summaries = formatted_summaries[-n:]

        return "\n\n".join(formatted_summaries)


def get_demo_datasets() -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    Get a collection of popular binary classification datasets for demonstration.
    
    Returns:
        Dict[str, Tuple[pd.DataFrame, str]]: Dictionary mapping dataset names to 
        (DataFrame, target_column_name) tuples.
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import (
        load_breast_cancer,
        load_wine,
        fetch_california_housing,
        make_classification
    )
    from sklearn.preprocessing import KBinsDiscretizer
    import seaborn as sns
    
    demo_datasets = {}
    
    # Breast Cancer Dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    demo_datasets['Breast Cancer'] = (df, 'target')
    
    # Wine Dataset (converted to binary classification)
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = (data.target > 0).astype(int)  # Convert to binary
    demo_datasets['Wine Quality'] = (df, 'target')
    
    # California Housing (converted to binary classification)
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # Convert continuous target to binary using median split
    df['target'] = (data.target > np.median(data.target)).astype(int)
    demo_datasets['California Housing'] = (df, 'target')
    
    # Titanic Dataset
    try:
        titanic_df = sns.load_dataset('titanic')
        # Clean and prepare the dataset
        titanic_df = titanic_df.drop(['deck', 'embark_town'], axis=1)  # Drop columns with too many missing values
        titanic_df['age'] = titanic_df['age'].fillna(titanic_df['age'].median())
        titanic_df['embarked'] = titanic_df['embarked'].fillna(titanic_df['embarked'].mode()[0])
        # Convert categorical variables to numeric
        titanic_df['sex'] = titanic_df['sex'].map({'male': 0, 'female': 1})
        titanic_df['embarked'] = titanic_df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        titanic_df['class'] = titanic_df['class'].map({'First': 0, 'Second': 1, 'Third': 2})
        # Rename target column
        titanic_df = titanic_df.rename(columns={'survived': 'target'})
        demo_datasets['Titanic Survival'] = (titanic_df, 'target')
    except Exception as e:
        st.warning(f"Could not load Titanic dataset: {str(e)}")
    
    # Synthetic Classification Dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    demo_datasets['Synthetic Classification'] = (df, 'target')
    
    return demo_datasets

def get_available_datasets() -> Dict[str, Union[str, Tuple[pd.DataFrame, str]]]:
    """
    Get all available datasets, including both local files and demo datasets.
    
    Returns:
        Dict[str, Union[str, Tuple[pd.DataFrame, str]]]: Dictionary mapping dataset names to either
        file paths (for local datasets) or (DataFrame, target_column_name) tuples (for demo datasets).
    """
    # Get local datasets
    data_dir = "data"
    dataset_files = {}
    
    # Supported file extensions
    supported_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
    
    try:
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                # Get file extension
                _, ext = os.path.splitext(file)
                if ext.lower() in supported_extensions:
                    # Create a nice display name from the filename
                    # Remove extension and replace underscores with spaces
                    display_name = os.path.splitext(file)[0].replace('_', ' ').title()
                    dataset_files[display_name] = file_path
    except Exception as e:
        st.error(f"Error scanning data directory: {str(e)}")
    
    # Add demo datasets
    demo_datasets = get_demo_datasets()
    dataset_files.update(demo_datasets)
    
    return dataset_files

def load_dataset(file_path: Union[str, Tuple[pd.DataFrame, str]]) -> Optional[pd.DataFrame]:
    """
    Load a dataset from file or use a demo dataset.
    
    Args:
        file_path (Union[str, Tuple[pd.DataFrame, str]]): Either a path to a dataset file
            or a tuple of (DataFrame, target_column_name) for demo datasets.
        
    Returns:
        Optional[pd.DataFrame]: Loaded dataframe or None if loading fails
    """
    try:
        if isinstance(file_path, tuple):
            # This is a demo dataset
            return file_path[0]
        else:
            # This is a file path
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.csv':
                return pd.read_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif ext == '.json':
                return pd.read_json(file_path)
            elif ext == '.parquet':
                return pd.read_parquet(file_path)
            else:
                st.error(f"Unsupported file format: {ext}")
                return None
                
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None
    
    
class CAAFEInputAdapter(BaseModel):
    """
    Input adapter for CAAFE - Context-Aware Automated Feature Engineering.
    """
    dataset_description: str = Field("A description of the dataset.")
    target_name: str = Field("The name of the target column.")
    optimization_metric: str = Field("The metric to optimize. One of 'accuracy', 'auc', 'ppv'.")
    max_features: int = Field(description="The maximum number of features to create.", gt=0, le=10)
    iterations: int = Field(description="The number of iterations to run.", gt=0, le=10)
    n_splits: int = Field(default=3, description="The number of splits to use for cross-validation.", gt=0, le=10)
    n_repeats: int = Field(default=2, description="The number of times to repeat the cross-validation.", gt=0, le=10)
    top_p: Optional[float] = Field(default=0.05, description="The top p value for the PPV metric. Only used if optimization_metric is 'ppv'.")
    
    
class NeedMoreInfo(BaseModel):
    """Use when the user needs to provide more information."""
    
    message: str = Field("A conversational follow up message &/or question to the user.")
    
    
AdapterResponse: TypeAlias = Union[CAAFEInputAdapter, NeedMoreInfo]

def main():
    """Streamlit app main function"""
    
    if st.session_state.get("message_history", None) is None:
        st.session_state["message_history"] = []
    
    agent = Agent(
        model="openai:gpt-4.1",
        result_type=AdapterResponse,
        deps_type=None,
        system_prompt="""
        You are CAAFE - A Context-Aware Automated Feature Engineering AI Assistant.
        Your goal is to help users run a feature engineering task by obtaining the necessary input parameters.
        It's okay to use default values when appropriate, but we will need the user to provide the following information:
        - dataset_description: A description of the dataset.
        - target_name: The name of the target column.
        - optimization_metric: The metric to optimize. One of 'accuracy', 'auc', 'ppv'.
        - max_features: The maximum number of features to create.
        - iterations: The number of iterations to run.
        
        If users give a simple greeting, reply with a friendly greeting, explain that you are CAAFE and tell them the information you need.
        Try to get as much info as possible but don't be too pushy.
        """
    )
    
    # fe = CAAFETransformer(
    #     llm_model='gpt-4.1-mini',
    #     optimization_metric="accuracy",
    #     top_p=0.05,
    #     target_name="claim_status",
    #     dataset_description="",
    #     max_features=5,
    #     iterations=2,
    #     n_splits=3,
    #     n_repeats=2,
    #     random_state=123,
    # )
    
    st.title("CAAFE - Context-Aware Automated Feature Engineering")
    st.markdown("Inspired by the paper: [CAAFE: Context-Aware Automated Feature Engineering](https://arxiv.org/pdf/2305.03403)")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    st.sidebar.subheader("Dataset Selection")
    
    # Get available datasets dynamically
    available_datasets = get_available_datasets()
    
    if not available_datasets:
        st.sidebar.warning("No datasets found in the data folder. Please add some datasets.")
    else:
        selected_dataset = st.sidebar.selectbox(
            "Choose a dataset",
            options=list(available_datasets.keys()),
            index=0
        )
        
        # Load the selected dataset
        dataset_info = available_datasets[selected_dataset]
        if isinstance(dataset_info, tuple):
            # This is a demo dataset
            df = dataset_info[0]
            target_name = dataset_info[1]
        else:
            # This is a file path
            df = load_dataset(dataset_info)
            target_name = "claim_status"  # Default target name for local datasets
        
        if df is not None:
            st.sidebar.success(f"Successfully loaded {selected_dataset}")
            
            # Display dataset info in sidebar
            st.sidebar.subheader("Dataset Info")
            st.sidebar.write(f"Shape: {df.shape}")
            st.sidebar.write(f"Target column: {target_name}")
            
            # Show column types and sample values
            with st.sidebar.expander("Dataset Preview"):
                st.dataframe(df.head(10))
            
            # Store the loaded dataframe in session state
            st.session_state['df'] = df
            st.session_state['target_name'] = target_name
               
    if user_prompt := st.chat_input("Enter your query", accept_file=True):
        with st.chat_message("user"):
            st.write(user_prompt.text)
        response = agent.run_sync(user_prompt.text, message_history=st.session_state["message_history"])
        st.session_state["message_history"] = response.all_messages()
        
        if isinstance(response.output, NeedMoreInfo):
            with st.chat_message("assistant"):
                st.write(response.output.message)
        elif isinstance(response.output, CAAFEInputAdapter):
            with st.chat_message("assistant"):
                st.success("CAAFE has been launched successfully.")
                if df is None:
                    st.error("No dataset loaded. Please load a dataset first.")
                    return
                X = df.drop(response.output.target_name, axis=1)
                y = df[response.output.target_name] 
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, 
                    y,
                    test_size=0.2,
                    random_state=42, # For reproducibility
                    stratify=y # Maintain class distribution
                )
                fe = CAAFETransformer(
                    llm_model='gpt-4.1-mini',
                    optimization_metric=response.output.optimization_metric,
                    top_p=response.output.top_p,
                    target_name=response.output.target_name,
                    dataset_description=response.output.dataset_description,
                    max_features=response.output.max_features,
                    iterations=response.output.iterations,
                    n_splits=response.output.n_splits,
                    n_repeats=response.output.n_repeats,
                    random_state=123,
                )
                fe.fit(X_train, y_train, show_prompts=False)
                if fe.full_code:
                    st.write("Final Code:")
                    st.code(fe.full_code)
                with st.expander("Tool Output"):
                    st.code("\n\n".join(fe.deps.tool_output_log))
                with st.expander("Prompt Log"):
                    st.code("\n\n".join(fe.deps.prompt_log))
                    
                follow_up_response = agent.run_sync(
                    f"Please summarize these results for the user:\n\n{fe.get_formatted_agent_notepad(n=-1)}", 
                    message_history=st.session_state["message_history"],
                )
                st.session_state["message_history"] = follow_up_response.all_messages()
                st.write(follow_up_response.output.message)
        else:
            with st.chat_message("assistant"):
                st.write(response.output.message)

if __name__ == "__main__":
    # See: https://bartbroere.eu/2023/06/17/adding-a-main-to-streamlit/

    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap

        streamlit.web.bootstrap.run(__file__, False, [], {})

    main()
