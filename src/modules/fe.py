import ast
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.usage import Usage
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedKFold

from src.run_llm_code import check_ast, run_llm_code
from src.utils import (
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

logger = logging.getLogger(f"{__name__}.CAAFETransformer")
if not logger.handlers:
    # If no handlers attached, add a basic console handler at INFO level
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
            logger.error(f"Invalid Python syntax: {e}")
            raise ValueError(f"Invalid Python syntax: {e}")

    @field_validator("code", mode="after")
    def validate_code_ast(cls, v: Any) -> str:
        """Validate that the code has proper AST based on the allowed specifications."""
        try:
            check_ast(ast.parse(v, mode="exec"))
        except Exception as e:
            logger.error(f"Invalid AST: {e}")
            raise ValueError(f"Invalid AST: {e}")
        return v

    @field_validator("code", mode="after")
    def validate_code_add_to_df(cls, v: Any) -> str:
        """Validate that the code adds the feature to the df"""
        if "df" not in v:
            logger.error("Code must operate on a pandas DataFrame called 'df'")
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
    agent_notepad: List[str] = Field(default_factory=list)


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
            logger.error(
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
        logger.info(
            f"\n🤖 Agent: {reasoning}"
            f"\n🛠️ Tool Call: `get_column_statistics(column_names={column_names})`\n"
        )

        df = ctx.deps.dataset
        valid_columns = [col for col in column_names if col in df.columns]

        for col in column_names:
            if col not in valid_columns:
                logger.error(f"Column '{col}' not found in dataset")
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
        return "\n---\n".join(summaries)

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
        logger.info(
            f"\n🤖 Agent: {reasoning}"
            f"\n🛠️ Tool Call: `get_feature_target_correlations(columns={columns})`\n"
        )

        df = ctx.deps.dataset
        target_name = ctx.deps.target_name
        if columns is not None:
            valid_columns = [col for col in columns if col in df.columns]
            for col in columns:
                if col not in valid_columns:
                    logger.error(f"Column '{col}' not found in dataset")
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
        logger.info(
            f"\n🤖 Agent: {reasoning}"
            f"\n🛠️ Tool Call: `get_correlation_pairs_summary(columns={columns})`\n"
        )

        df = ctx.deps.dataset
        if columns is not None:
            valid_columns = [col for col in columns if col in df.columns]
            for col in columns:
                if col not in valid_columns:
                    logger.error(f"Column '{col}' not found in dataset")
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
        logger.info(f"\n🤖 Agent: {reasoning}\n🛠️ Tool Call: `check_for_outliers()`\n")

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
        logger.info(
            f"\n🤖 Agent: {reasoning}"
            f"\n🛠️ Tool Call: `get_mutual_information_summary(columns={columns})`\n"
        )

        df = ctx.deps.dataset
        target = ctx.deps.target_name
        if columns is not None:
            valid_columns = [col for col in columns if col in df.columns]
            for col in columns:
                if col not in valid_columns:
                    logger.error(f"Column '{col}' not found in dataset")
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

        return output_string

    return feature_generation_agent


class CAAFETransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn–compatible transformer that uses an LLM (e.g. GPT-4o)
    to iteratively generate new features (CAAFE algorithm), evaluating each batch
    via RepeatedKFold and keeping only those that show statistically
    significant improvement.  Supports:

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

    optimization_metric : str, {"accuracy", "auc"}, default="accuracy"
        Which metric to optimize when comparing baseline vs. enhanced:
        - "accuracy": uses accuracy_score
        - "auc": uses roc_auc_score (binary or multiclass via ovr)

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

    llm_model : str, default="gpt-4o-mini"
        Name of the OpenAI (LLM) model to invoke.

    logger : Optional[logging.Logger], default=None
        If provided, uses this logger; otherwise, creates a new one under
        `__name__ + ".CAAFETransformer"`.
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
        logger: Optional[logging.Logger] = None,
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
        optimization_metric : str, {"accuracy", "auc"}, default="accuracy"
            Which metric to optimize when comparing baseline vs. enhanced.
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
        logger : Optional[logging.Logger], default=None
            If provided, uses this structured logger; otherwise, creates a new one.

        """
        # Set up structured logger
        if logger is None:
            self.logger = logging.getLogger(f"{__name__}.CAAFETransformer")
            if not self.logger.handlers:
                # If no handlers attached, add a basic console handler at INFO level
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        self.target_name = target_name
        self.dataset_description = dataset_description or ""
        self.max_features = max_features
        self.optimization_metric = optimization_metric.lower()
        self.iterations = iterations
        self.llm_model = llm_model
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_samples = n_samples
        self.cv_folds = cv_folds

        # If no base classifier is given, default to XGBClassifier
        if base_classifier is None:
            self.base_classifier = xgb.XGBClassifier(
                objective="binary:logistic",
                use_label_encoder=False,
                eval_metric="logloss",
                enable_categorical=True,
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
        self.best_score: float = -np.inf
        self.best_acc: float = -np.inf
        self.best_auc: float = -np.inf

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
        self.logger.info(
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
        )

        # Log initialization
        self.logger.info(
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
            self.logger.info(
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
                self.logger.log_info(
                    f"Feature code loaded successfully (took {time.time() - self._start_time:.2f}s)"
                )
                return self
            except Exception as e:
                self.logger.log_error(
                    e, {"phase": "code_loading", "source_path": load_code_path}
                )
                raise IOError(f"Failed to load code from {load_code_path}: {e}")

        # Otherwise, run full iterative feature-engineering process
        self.logger.info("Starting iterative feature engineering process...")

        # 1) Evaluate baseline stats (no new features)
        self.logger.info(
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

        baseline_primary = (
            self.baseline_auc
            if self.optimization_metric == "auc"
            else self.baseline_acc
        )
        self.logger.info(
            f"\nBaseline ROC AUC: {self.baseline_auc:.3f} (±{baseline_auc_std:.3f})"
        )
        self.logger.info(
            f"\nBaseline Accuracy: {self.baseline_acc:.3f} (±{baseline_acc_std:.3f})"
        )

        self.best_score = baseline_primary
        self.best_acc = self.baseline_acc
        self.best_auc = self.baseline_auc

        # Seed the iteration loop
        consecutive_no_improvement = 0
        previous_iteration_reasoning = ""

        for itr in range(self.iterations):
            # Log iteration start
            self.logger.info(f"\n\n--- Iteration {itr + 1}/{self.iterations} ---\n")

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
                self.logger.info(f"\n\nPrompt for iteration {itr + 1}:\n{prompt}\n")

            try:
                # 2) Ask the LLM agent to propose new features
                llm_start_time = time.time()
                self.logger.info("\n\n→ Invoking LLM for new feature generation...\n")

                result = self.feature_agent.run_sync(prompt, deps=self.deps)
                llm_duration = time.time() - llm_start_time

                feature_result = result.output
                self.usages.append(result.usage())
                previous_iteration_reasoning = feature_result.reasoning

                # Log LLM interaction
                self.logger.info(
                    f"\nLLM Feature Engineering Reasoning:\n🤖 Agent: {feature_result.reasoning}"
                )
                self.logger.info(
                    f"\nProposing {len(feature_result.new_features)} new features"
                    f"\nDropping {feature_result.dropped_count} existing columns."
                    f"\n✅ LLM interaction completed - Iteration {itr + 1} took {llm_duration:.2f}s"
                )

                # Convert to executable Python code
                code = feature_result.code_to_run
                self.logger.info(
                    "\nGenerated code snippet:\n\n"
                    + to_code_markdown(feature_result.to_code())
                )

                # 3) Use incremental evaluation pattern
                try:
                    # 4) Incremental evaluation: (original + previous) vs (original + previous + new)
                    self.logger.info(
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
                    is_significant = improvement_roc > 0 or improvement_acc > 0

                    # Get both accuracy and AUC metrics for comprehensive reporting
                    baseline_acc = np.nanmean(old_results["accuracy"])
                    enhanced_acc = np.nanmean(new_results["accuracy"])
                    baseline_auc = np.nanmean(old_results["auc"])
                    enhanced_auc = np.nanmean(new_results["auc"])
                    acc_improvement = improvement_acc
                    auc_improvement = improvement_roc

                    # Log detailed feature evaluation
                    # Log a human-readable summary of feature evaluation results
                    eval_msg = (
                        f"\nFeature Evaluation Results (Iteration {itr + 1}):\n"
                        f"  Features Added: {', '.join(feature_names) if feature_names else 'None'}\n"
                        f"  Features Dropped: {', '.join(dropped_feature_names) if dropped_feature_names else 'None'}\n"
                        f"  Baseline Metrics: ACC {baseline_acc:.4}, ROC AUC {baseline_auc:.4}\n"
                        f"  Updated Metrics: ACC {enhanced_acc:.4}, ROC AUC {enhanced_auc:.4}\n"
                        f"  Improvements: ACC {acc_improvement:+.4}, ROC AUC {auc_improvement:+.4}\n"
                        f"  Significant: {is_significant}\n"
                        f"  Evaluation Time: {eval_duration:.2f}s"
                    )
                    self.logger.info(eval_msg)

                    # Create formatted summary string for agent memory
                    performance_summary = (
                        f"Iteration {itr + 1}\n"
                        f"Features created: {', '.join(feature_names) if feature_names else 'None'}\n"
                        f"Features dropped: {', '.join(dropped_feature_names) if dropped_feature_names else 'None'}\n"
                        f"Performance before adding features ROC {baseline_auc:.4}, ACC {baseline_acc:.4}.\n"
                        f"Performance after adding features ROC {enhanced_auc:.4}, ACC {enhanced_acc:.4}.\n"
                        f"Improvement ROC {auc_improvement:+.4}, ACC {acc_improvement:+.4}.\n"
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
                        "roc_improvement": auc_improvement,
                        "acc_improvement": acc_improvement,
                        "significant": str(is_significant),
                        "code_retained": str(is_significant),
                        "formatted_summary": performance_summary,
                    }

                    self.deps.agent_notepad.append(summary_record)

                    # Continue with processing the decision
                    if is_significant:
                        self.logger.info(
                            "\n✓ Proposed features show improvement: Keeping them. "
                        )
                        if acc_improvement > 0:
                            self.logger.info(f"\nAccuracy +{acc_improvement:.4}")
                        if auc_improvement > 0:
                            self.logger.info(f"\nROC AUC +{auc_improvement:.4}")

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

                        self.logger.info(
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

                        # Reset counter of "no improvement"
                        consecutive_no_improvement = 0

                    else:
                        self.logger.info(
                            "\n✗ Proposed features did NOT show improvement: Discarding."
                        )
                        consecutive_no_improvement += 1
                        self._features_rejected += len(feature_names)
                        self.rejected_features.extend(feature_names)

                except Exception as e:
                    self.logger.error(
                        f"Error during feature evaluation (iteration {itr + 1}): {str(e)}\n"
                        f"Code length: {len(self.full_code)} chars\n"
                        f"Full error: {type(e).__name__}: {str(e)}"
                    )
                    consecutive_no_improvement += 1

            except Exception as e:
                self.logger.error(
                    f"Error during feature evaluation (iteration {itr + 1}): {str(e)}\n"
                    f"Code length: {len(self.full_code)} chars\n"
                    f"Full error: {type(e).__name__}: {str(e)}"
                )
                consecutive_no_improvement += 1

            # Log iteration end
            self.logger.info(
                f"\nIteration {itr + 1} completed: "
                f"\nConsecutive no improvement: {consecutive_no_improvement}, "
                f"\nCurrent best primary metric ({self.optimization_metric}) score: {self.best_score:.3f}"
                f"\nCurrent best accuracy score: {self.best_acc:.3f}"
                f"\nCurrent best ROC AUC score: {self.best_auc:.3f}"
            )

            # Early‐stopping condition
            if consecutive_no_improvement >= 3:
                self.logger.info(
                    f"\nNo improvement for {consecutive_no_improvement} consecutive iterations → early stopping."
                    f"\nCompleted iterations: {itr + 1}, "
                    f"\nTotal planned iterations: {self.iterations}"
                )
                break

        # End of iteration loop - log final summary
        total_duration = time.time() - self._start_time

        self.logger.info(
            f"\nFinal summary: "
            f"\nTotal iterations={itr + 1}, "
            f"\nFeatures created, {', '.join(self.accepted_features)}"
            f"\nFeatures dropped, {', '.join(self.features_dropped)}"
            f"\nFeatures rejected, {', '.join(self.rejected_features)}"
            f"\nOriginal ROC AUC: {float(self.baseline_auc)}, "
            f"\nOriginal accuracy: {float(self.baseline_acc)}, "
            f"\nFinal ROC AUC: {float(self.best_auc)}, "
            f"\nFinal accuracy: {float(self.best_acc)}, "
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
            self.logger.error(
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
            self.logger.warning("No feature-generation code available to save.")
            return

        if filepath.lower().endswith(".py"):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.full_code)
                self.logger.info(f"Feature-generation code saved to {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to save code to {filepath}: {e}")
        elif filepath.lower().endswith(".md"):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("```python\n")
                    f.write(self.full_code)
                    f.write("\n```")
                self.logger.info(
                    f"Feature-generation code saved (as Markdown) to {filepath}"
                )
            except Exception as e:
                self.logger.error(f"Failed to save code to {filepath}: {e}")
        else:
            self.logger.warning(
                "Unrecognized extension for save_code: use '.py' or '.md'."
            )

    def evaluate_features(
        self,
        full_code: str,
        code: str,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        old_results = {"accuracy": [], "auc": []}
        new_results = {"accuracy": [], "auc": []}

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
                self.logger.warning(
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

            for metric in ["accuracy", "auc"]:
                old_results[metric].append(old_result[metric])
                new_results[metric].append(new_result[metric])

        return old_results, new_results

    def _evaluate_dataset(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_name: str
    ):
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

        self.base_classifier.fit(X=x, y=y.long())
        probs = self.base_classifier.predict_proba(test_x)
        acc = float(accuracy_metric(test_y, probs))
        auc = float(auc_metric(test_y, probs))

        return {"accuracy": acc, "auc": auc}

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
