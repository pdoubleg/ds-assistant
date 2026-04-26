import ast
import logging
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.messages import ModelMessage
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold

from src.run_llm_code import (
    check_ast,
    run_llm_code,
    run_llm_encoder_code,
)
from src.utils import (
    accuracy_metric,
    auc_metric,
    get_dataset_summary_with_importance,
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

FEATURE_CODE_SECTION_HEADER = "# === FEATURE_ENGINEERING_CODE ==="
ENCODING_CODE_SECTION_HEADER = "# === FEATURE_ENCODING_CODE ==="
DEFAULT_ENCODING_STRATEGY_NAME = "default_tabular_encoder"
ESTIMATOR_TYPE_ALIASES = {
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "custom": "custom",
}
MODEL_PROMPT_DETAILS = {
    "xgboost": {
        "display_name": "XGBoost",
        "system_guidance": """
- Favor robust nonlinear interactions, monotonic-safe transforms, and well-behaved numeric inputs.
- Ensure the final encoded matrix is fully numeric; never leave raw object or string columns unresolved.
- One-hot low-cardinality nominal categoricals when it adds signal, but prefer more compact encodings for higher-cardinality features.
""".strip(),
        "encoding_guidance": """
- Produce a fully numeric sklearn-compatible feature matrix for XGBoost.
- One-hot encode low-cardinality nominal features when appropriate.
- Use ordinal encoders only when categories have a real or operational ordering.
- For higher-cardinality categoricals, prefer compact encodings such as frequency, count, grouped-category, or other dense representations over blindly exploding dimensionality.
""".strip(),
    },
    "lightgbm": {
        "display_name": "LightGBM",
        "system_guidance": """
- Favor compact, information-dense engineered features that work well with histogram-based gradient boosting.
- Preserve missingness signal intentionally and avoid needlessly wide sparse expansions unless they are clearly justified.
- Treat categoricals pragmatically: grouped categories, count/frequency features, and careful ordinal-style encodings are often stronger than broad one-hot expansions.
""".strip(),
        "encoding_guidance": """
- Produce model-ready sklearn-compatible features for LightGBM, favoring compact dense encodings when possible.
- Prefer ordinal, count, frequency, or grouped-category encodings for moderate or high-cardinality categoricals unless a low-cardinality one-hot representation is clearly beneficial.
- Use one-hot encoding selectively for genuinely low-cardinality nominal features.
- Preserve missing values explicitly through imputers, missing buckets, indicators, or other robust preprocessing choices.
""".strip(),
    },
    "custom": {
        "display_name": "custom estimator",
        "system_guidance": """
- Align feature engineering and encoding choices with the estimator's likely bias toward numeric, dense, sparse, linear, or tree-based inputs.
- Prefer safe, general-purpose transformations when the estimator's behavior is uncertain.
- Keep the final encoded representation robust, reproducible, and fully sklearn-compatible.
""".strip(),
        "encoding_guidance": """
- Produce a fully sklearn-compatible encoded feature matrix.
- Choose dense versus sparse encodings based on what is most likely to work well for the estimator.
- Use missing-value handling, category grouping, and feature scaling only when they improve robustness for the model family.
""".strip(),
    },
}


def metric_ppv(
    y_true: Union[list, pd.Series], y_pred: Union[list, pd.Series], top_p: float
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

    ranked = pd.DataFrame(
        {
            "label": pd.Series(y_true).values,
            "predicted_prob": pd.Series(y_pred).values,
        }
    )

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


class FeatureEncodingCode(BaseModel):
    """A valid sklearn encoding pipeline code block and its reasoning."""

    reasoning: str = Field(description="Reasoning for why this encoding is useful")
    name: str = Field(description="Encoding strategy name")
    code: str = Field(
        description="Python code that assigns a sklearn-compatible transformer to `encoder`"
    )

    @field_validator("code", mode="after")
    def validate_code_syntax(cls, v: Any) -> str:
        """Validate that the encoding code has proper Python syntax."""
        try:
            ast.parse(v, mode="exec")
        except SyntaxError as exc:
            logger.error(f"Invalid encoding Python syntax: {exc}")
            raise ValueError(f"Invalid encoding Python syntax: {exc}") from exc
        return v

    @field_validator("code", mode="after")
    def validate_code_ast(cls, v: Any) -> str:
        """Validate that the encoding code follows the allowed AST rules."""
        try:
            check_ast(ast.parse(v, mode="exec"))
        except Exception as exc:
            logger.error(f"Invalid encoding AST: {exc}")
            raise ValueError(f"Invalid encoding AST: {exc}") from exc
        return v

    @field_validator("code", mode="after")
    def validate_code_assigns_encoder(cls, v: Any) -> str:
        """Validate that the encoding code assigns a transformer to `encoder`."""
        parsed = ast.parse(v, mode="exec")
        assigns_encoder = False

        for node in ast.walk(parsed):
            if isinstance(node, ast.Assign):
                assigns_encoder = assigns_encoder or any(
                    isinstance(target, ast.Name) and target.id == "encoder"
                    for target in node.targets
                )
            elif isinstance(node, ast.AnnAssign):
                assigns_encoder = assigns_encoder or (
                    isinstance(node.target, ast.Name) and node.target.id == "encoder"
                )

        if not assigns_encoder:
            logger.error("Encoding code must assign a transformer to `encoder`")
            raise ValueError(
                "Encoding code must assign a sklearn-compatible transformer to `encoder`"
            )
        return v


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
    feature_encoding: FeatureEncodingCode = Field(
        description="Sklearn encoding pipeline code for the final engineered dataframe"
    )

    @property
    def feature_code_to_run(self) -> str:
        """Return feature-engineering code ready for execution."""
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
    def encoding_code_to_run(self) -> str:
        """Return the encoding code ready for execution."""
        return self.feature_encoding.code

    @property
    def code_to_run(self) -> str:
        """Return feature-engineering code for backward compatibility."""
        return self.feature_code_to_run

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
        """Convert the feature and encoding result to Python code with comments."""
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

        code_lines.append(
            f"# Encoding: {self.feature_encoding.name}: {self.feature_encoding.reasoning}"
        )
        code_lines.append(self.feature_encoding.code)

        return "\n".join(code_lines)


@dataclass
class FeatureEngineeringDependencies:
    """Dependencies for feature engineering agents."""

    original_dataset: pd.DataFrame
    dataset: pd.DataFrame
    target_name: str
    dataset_description: str
    current_features: List[str]
    agent_notepad: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """
You are a senior data scientist and Kaggle grandmaster whose sole mission is to design, implement, and \
**iterate** on FEATURE ENGINEERING and FEATURE ENCODING strategies for a binary-classification dataset that will be modeled using {model_name}.

Do not try to solve everything at once. Instead work incrementally while reflecting on past iterations. Aim to increase your level of \
creativity and complexity as you go. For example start with simple best practices and then gradually add more advanced techniques. If \
progress stalls pivot your approach; towards the end of the process try to use more advanced techniques.

Users will provide you with a summary of the dataset and the current features, along with \
results from the previous feature engineering iteration when applicable.

Users will also provide a narrative description of the dataset which may also include \
additional instructions on what to focus on, or specific requirements for the feature engineering.

You have deep knowledge of:
- Domain-specific feature engineering techniques
- Statistical transformations and aggregations
- Feature interactions and combinations
- Feature selection principles
- Sklearn preprocessing, column transformers, pipelines, missing-value handling, and text vectorizers

Model-specific guidance for {model_name}:
{model_specific_guidance}

When generating features, you always:
1. Focus on the deliverable code that will be run by the user.
2. Consider the real-world meaning of the data
3. Create features that capture important patterns
4. Avoid overfitting by being selective
5. Drop redundant or harmful features when appropriate
6. Pair your feature changes with a sklearn-compatible encoding pipeline.
7. Only use the following external packages:
    - pandas
    - numpy
    - scipy
    - sklearn
    
Features can include but are not limited to:
    - Numerical: log/Box‑Cox transforms, binning, polynomial & interaction terms, \
      scaling, winsorisation.  
    - Categorical: frequency, target, leave‑one‑out, Helmert, ordinal, label, \
      and one-hot encodings; group statistics; rare-label consolidation.  
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
- Do not import additional packages. Use the already-available `pd`, `np`, `scipy`, and `sklearn` objects.
- Always assign your sklearn-compatible encoding transformer to a variable named `encoder`.
- The encoding pipeline must be fit only on the training dataframe and later reused to transform validation or inference data.
- Choose encoding strategies based on semantic meaning, cardinality, and modeling needs. For example, prefer ordinal-style encoders order-like features, one-hot for low-cardinality nominal features, label encoding only when it is operationally safe, and vectorizers or hashing approaches for free text.
- Handle missing values explicitly inside the encoding pipeline. Use simple imputers, constant fill values, missing-category buckets, indicator features, or other robust sklearn-compatible strategies when appropriate.
- Consider more advanced preprocessing when justified, such as rare-category grouping, binning, scaling, interaction-safe pipelines, text vectorization, feature unions, or column-specific sub-pipelines.
- Always be mindful of the relationship between the features and their respective encoding strategies.
"""

FEATURE_GENERATION_PROMPT = """
Generate new features to improve classification performance.

Dataset description:
{dataset_description}

Target variable: {target_name}

Dataset summary:
{dataset_summary}

Generate up to {max_features} meaningful feature(s) that:
1. Add semantic information based on real-world knowledge and df characteristics
2. Capture patterns through combinations, transformations, or aggregations
3. Are likely to improve classification of "{target_name}"

Also identify any existing features that should be dropped because they:
- Are redundant with other features
- May cause overfitting
- Don't contribute to predicting the target

Also generate sklearn feature-encoding code that:
1. Assumes your feature code has already been run on `df`
2. Assigns the final sklearn-compatible transformer to a variable named `encoder`
3. Produces model-ready features for {model_name}
4. Uses the best encoding choice for the current feature set, such as one-hot, ordinal, label encoding, imputers, column transformers, pipelines, feature unions, or vectorizers where appropriate
5. Handles missing values intentionally and robustly, rather than relying on accidental defaults
6. Uses ordinal encoders when it makes sense, i.e., inherent or perceived ordering, levels of intensity, etc.
7. Does not reference the target column

Model-specific encoding instructions for {model_name}:
{model_specific_encoding_guidance}

Ensure all generated code uses only existing column names from the df: {current_features}
"""


# ============================================================================
# Agents
# ============================================================================


def normalize_estimator_type(estimator_type: str) -> str:
    """Normalize estimator family names used by the transformer.

    Args:
        estimator_type: Estimator family label supplied by the caller.

    Returns:
        Normalized estimator family name.

    Raises:
        ValueError: If the estimator family is unsupported.
    """
    normalized = ESTIMATOR_TYPE_ALIASES.get(estimator_type.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(ESTIMATOR_TYPE_ALIASES))
        raise ValueError(
            f"Unsupported estimator_type '{estimator_type}'. Supported values: {supported}"
        )
    return normalized


def infer_estimator_type(base_classifier: Optional[Any]) -> str:
    """Infer the estimator family from a classifier instance.

    Args:
        base_classifier: Optional classifier instance used for evaluation.

    Returns:
        Estimator family label compatible with prompt generation.
    """
    if base_classifier is None:
        return "xgboost"

    classifier_module = base_classifier.__class__.__module__.lower()
    if isinstance(base_classifier, xgb.XGBModel) or classifier_module.startswith(
        "xgboost"
    ):
        return "xgboost"
    if isinstance(base_classifier, lgb.LGBMModel) or classifier_module.startswith(
        "lightgbm"
    ):
        return "lightgbm"
    return "custom"


def get_model_prompt_context(
    estimator_type: str,
    base_classifier: Optional[Any] = None,
) -> Dict[str, str]:
    """Build prompt metadata for the requested estimator family.

    Args:
        estimator_type: Normalized or alias estimator family.
        base_classifier: Optional classifier instance for custom naming.

    Returns:
        Prompt context containing model labels and model-specific instructions.
    """
    normalized_type = normalize_estimator_type(estimator_type)
    context = MODEL_PROMPT_DETAILS[normalized_type].copy()
    if normalized_type == "custom" and base_classifier is not None:
        context["display_name"] = base_classifier.__class__.__name__
    return context


def get_system_prompt(
    estimator_type: str,
    base_classifier: Optional[Any] = None,
) -> str:
    """Render the system prompt for the selected estimator family.

    Args:
        estimator_type: Estimator family used for evaluation.
        base_classifier: Optional classifier instance for custom naming.

    Returns:
        Fully rendered system prompt.
    """
    prompt_context = get_model_prompt_context(estimator_type, base_classifier)
    return SYSTEM_PROMPT.format(
        model_name=prompt_context["display_name"],
        model_specific_guidance=prompt_context["system_guidance"],
    )


def get_feature_generation_prompt(
    dataset_description: str,
    target_name: str,
    dataset_summary: str,
    max_features: int,
    current_features: str,
    estimator_type: str,
    base_classifier: Optional[Any] = None,
) -> str:
    """Render the per-iteration feature engineering prompt.

    Args:
        dataset_description: User-provided dataset narrative.
        target_name: Name of the target column.
        dataset_summary: Generated dataset profile for the current iteration.
        max_features: Maximum number of new features to request.
        current_features: Comma-separated list of current features.
        estimator_type: Estimator family used for evaluation.
        base_classifier: Optional classifier instance for custom naming.

    Returns:
        Fully rendered feature-generation prompt.
    """
    prompt_context = get_model_prompt_context(estimator_type, base_classifier)
    return FEATURE_GENERATION_PROMPT.format(
        dataset_description=dataset_description,
        target_name=target_name,
        current_features=current_features,
        dataset_summary=dataset_summary,
        max_features=max_features,
        model_name=prompt_context["display_name"],
        model_specific_encoding_guidance=prompt_context["encoding_guidance"],
    )


def get_feature_generation_agent(
    model: str = "openai:gpt-4.1",
    estimator_type: str = "xgboost",
    base_classifier: Optional[Any] = None,
) -> Agent:
    """Get the feature generation agent."""

    feature_generation_agent = Agent(
        model=model,
        deps_type=FeatureEngineeringDependencies,
        output_type=FeatureGenerationResult,
        retries=5,
        system_prompt=get_system_prompt(estimator_type, base_classifier),
    )

    @feature_generation_agent.output_validator
    def is_executable(
        ctx: RunContext[FeatureEngineeringDependencies], result: FeatureGenerationResult
    ) -> FeatureGenerationResult:
        """Validate that the generated code is executable."""

        try:
            feature_df = run_llm_code(
                result.feature_code_to_run,
                ctx.deps.dataset,
                fill_na=True,
            )
            feature_df = feature_df.drop(
                columns=[ctx.deps.target_name], errors="ignore"
            )
            run_llm_encoder_code(result.encoding_code_to_run, feature_df)
        except Exception as e:
            logger.error(
                "🚨 Code validation failed:\n"
                f"Error: {str(e)}\n"
                f"Feature code:\n{result.feature_code_to_run}\n"
                f"Encoding code:\n{result.encoding_code_to_run}\n"
                "Phase: code_validation"
            )
            raise ModelRetry(f"Invalid feature/encoding code: {e}") from e
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
    via RepeatedStratifiedKFold and keeping only those that show improvement.

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
        Number of folds in RepeatedStratifiedKFold.

    n_repeats : int, default=2
        Number of repeats in RepeatedStratifiedKFold.

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

    logger : Optional[logging.Logger], default=None
        If provided, uses this logger; otherwise, creates a new one under
        `__name__ + ".CAAFETransformer"`.
    estimator_type : Optional[str], default=None
        Type of estimator to use. If None, will be inferred from base_classifier.
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
        logger: Optional[logging.Logger] = None,
        estimator_type: Optional[str] = None,
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
            Number of folds in RepeatedStratifiedKFold.
        n_repeats : int, default=2
            Number of repeats in RepeatedStratifiedKFold.
        random_state : int, default=42
            Random seed for reproducibility.
        n_samples : int, default=10
            Number of sample rows to include in dataset summary for LLM.
        cv_folds : int, default=5
            Number of cross-validation folds for feature importance calculation.
        top_p : float, default=0.05
            Fraction of top predictions to use for PPV calculation (only used when optimization_metric="ppv").
        logger : Optional[logging.Logger], default=None
            If provided, uses this structured logger; otherwise, creates a new one.
        estimator_type : Optional[str], default=None
            Type of estimator to use. If None, will be inferred from base_classifier.

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

        # Validate optimization metric
        if self.optimization_metric not in ["accuracy", "auc", "ppv"]:
            raise ValueError(
                "optimization_metric must be one of: 'accuracy', 'auc', 'ppv'"
            )

        self.iterations = iterations
        self.llm_model = llm_model
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_samples = n_samples
        self.cv_folds = cv_folds
        self.top_p = top_p
        inferred_estimator_type = infer_estimator_type(base_classifier)
        self.estimator_type = (
            normalize_estimator_type(estimator_type)
            if estimator_type is not None
            else inferred_estimator_type
        )

        if (
            base_classifier is not None
            and inferred_estimator_type != "custom"
            and self.estimator_type != inferred_estimator_type
        ):
            raise ValueError(
                "The provided estimator_type does not match the supplied base_classifier family."
            )

        # If no base classifier is given, choose a default tree booster for the selected family.
        if base_classifier is None:
            if self.estimator_type == "xgboost":
                self.base_classifier = xgb.XGBClassifier(
                    objective="binary:logistic",
                    use_label_encoder=False,
                    eval_metric="logloss",
                    enable_categorical=False,
                    random_state=self.random_state,
                )
            elif self.estimator_type == "lightgbm":
                self.base_classifier = lgb.LGBMClassifier(
                    objective="binary",
                    metric="binary_logloss",
                    random_state=self.random_state,
                    verbosity=-1,
                    force_row_wise=True,
                    n_jobs=1,
                )
            else:
                raise ValueError(
                    "A base_classifier must be provided when estimator_type='custom'."
                )
        else:
            self.base_classifier = base_classifier
        self.estimator_name = get_model_prompt_context(
            self.estimator_type,
            self.base_classifier,
        )["display_name"]

        self.deps: FeatureEngineeringDependencies = None
        self.feature_agent = get_feature_generation_agent(
            model=self.llm_model,
            estimator_type=self.estimator_type,
            base_classifier=self.base_classifier,
        )

        # Will store the final code (concatenated accepted iterations)
        self.code: str = ""
        self.full_code: str = ""
        self.full_encoding_code: str = ""

        # Each iteration's code blocks
        self.feature_code_history: List[str] = []
        self.encoding_code_history: List[str] = []
        self.fitted_encoder: Optional[Any] = None

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
        self.usages: List[RunUsage] = []

    def _combine_code_blocks(self, *code_blocks: str) -> str:
        """Join executable code blocks while skipping empty entries.

        Args:
            *code_blocks: Code snippets to concatenate.

        Returns:
            Combined executable code.
        """
        return "\n\n".join(
            block.strip() for block in code_blocks if block and block.strip()
        )

    def _describe_encoding_strategy(self, encoding_code: str) -> str:
        """Return a human-readable encoding strategy label.

        Args:
            encoding_code: Raw encoding code block.

        Returns:
            Strategy label used in logs and evaluation history.
        """
        return encoding_code.strip() or DEFAULT_ENCODING_STRATEGY_NAME

    def _serialize_code_bundle(
        self,
        feature_code: Optional[str] = None,
        encoding_code: Optional[str] = None,
    ) -> str:
        """Serialize feature and encoding code into one portable artifact.

        Args:
            feature_code: Optional feature-engineering code override.
            encoding_code: Optional encoding code override.

        Returns:
            Combined code artifact ready to save to disk.
        """
        feature_code = self.full_code if feature_code is None else feature_code
        encoding_code = (
            self.full_encoding_code if encoding_code is None else encoding_code
        )

        sections = [FEATURE_CODE_SECTION_HEADER]
        if feature_code.strip():
            sections.append(feature_code.strip())

        sections.extend(["", ENCODING_CODE_SECTION_HEADER])
        if encoding_code.strip():
            sections.append(encoding_code.strip())

        return "\n".join(sections).strip() + "\n"

    def _parse_code_bundle(self, raw_code: str) -> Tuple[str, str]:
        """Parse a saved code artifact into feature and encoding sections.

        Args:
            raw_code: Raw file contents loaded from disk.

        Returns:
            Tuple of ``(feature_code, encoding_code)``.
        """
        if (
            FEATURE_CODE_SECTION_HEADER not in raw_code
            and ENCODING_CODE_SECTION_HEADER not in raw_code
        ):
            return raw_code.strip(), ""

        payload = raw_code
        if FEATURE_CODE_SECTION_HEADER in payload:
            _, _, payload = payload.partition(FEATURE_CODE_SECTION_HEADER)

        feature_part, _, encoding_part = payload.partition(ENCODING_CODE_SECTION_HEADER)
        return feature_part.strip(), encoding_part.strip()

    def _fit_encoder_for_dataframe(
        self,
        df: pd.DataFrame,
        encoding_code: str,
    ) -> Any:
        """Fit the current encoding strategy on a feature dataframe.

        Args:
            df: Feature dataframe without the target column.
            encoding_code: Encoding code to fit. Empty code triggers the default encoder.

        Returns:
            Fitted sklearn-compatible transformer.
        """
        _, _, encoder = run_llm_encoder_code(encoding_code, df)
        return encoder

    def _refresh_fitted_encoder(self) -> None:
        """Refresh the persisted encoder using the current engineered dataset."""
        if self.deps is None:
            self.fitted_encoder = None
            return

        feature_df = self.deps.dataset.drop(columns=[self.target_name], errors="ignore")
        self.fitted_encoder = self._fit_encoder_for_dataframe(
            feature_df,
            self.full_encoding_code,
        )

    def _apply_feature_code(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """Apply feature code to a dataframe.

        Args:
            df: Input dataframe.
            code: Feature-engineering code to execute.

        Returns:
            Engineered dataframe.
        """
        if not code.strip():
            return df.copy()
        return run_llm_code(code, df)

    def _apply_fitted_encoding(self, df: pd.DataFrame) -> Any:
        """Transform a feature dataframe with the persisted encoder.

        Args:
            df: Feature dataframe without the target column.

        Returns:
            Encoded feature matrix for inference.

        Raises:
            RuntimeError: If no encoder has been fitted yet.
        """
        if self.fitted_encoder is None:
            raise RuntimeError("Encoding pipeline not fitted yet; call fit() first.")

        encoded_df, _, _ = run_llm_encoder_code(
            "",
            df_train=df,
            encoder=self.fitted_encoder,
            fit_encoder=False,
        )
        return encoded_df

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
            f"  Estimator family: {self.estimator_type}\n"
            f"  Base classifier: {self.estimator_name}\n"
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
                    feature_code, encoding_code = self._parse_code_bundle(raw)
                    self.code = feature_code
                    self.full_code = feature_code
                    self.full_encoding_code = encoding_code
                    self.feature_code_history = [feature_code] if feature_code else []
                    self.encoding_code_history = (
                        [encoding_code] if encoding_code else []
                    )
                    if self.full_code:
                        self.deps.dataset = self._apply_feature_code(
                            self.deps.original_dataset,
                            self.full_code,
                        )
                    self._refresh_fitted_encoder()
                    self.deps.agent_notepad.append(
                        {
                            "iteration": "loaded_from_disk",
                            "source_path": load_code_path,
                            "notes": "Code loaded; no LLM generation performed",
                            "encoding_strategy": self._describe_encoding_strategy(
                                self.full_encoding_code
                            ),
                        }
                    )
                self._is_fitted = True

                # Log successful code loading
                self.logger.info(
                    f"Feature code loaded successfully (took {time.time() - self._start_time:.2f}s)"
                )
                return self
            except Exception as e:
                self.logger.error(f"Failed to load code from {load_code_path}: {e}")
                raise IOError(f"Failed to load code from {load_code_path}: {e}")

        # Otherwise, run full iterative feature-engineering process
        self.logger.info("Starting iterative feature engineering process...")

        # 1) Evaluate baseline stats (no new features)
        self.logger.info(
            "\n\n→ Evaluating baseline performance (no added features)...\n"
        )
        _, baseline_stats = self.evaluate_features(
            full_code="",
            code="",
            full_encoding_code="",
            encoding_code="",
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
        self.logger.info(
            f"\nBaseline ROC AUC: {self.baseline_auc:.3f} (±{baseline_auc_std:.3f})"
        )
        self.logger.info(
            f"\nBaseline Accuracy: {self.baseline_acc:.3f} (±{baseline_acc_std:.3f})"
        )
        self.logger.info(
            f"\nBaseline PPV@{self.top_p:.1%}: {self.baseline_ppv:.3f} (±{baseline_ppv_std:.3f})"
        )

        self.best_score = baseline_primary
        self.best_acc = self.baseline_acc
        self.best_auc = self.baseline_auc
        self.best_ppv = self.baseline_ppv

        # Seed the iteration loop
        consecutive_no_improvement = 0
        previous_iteration_reasoning = ""
        messages: List[ModelMessage] = []

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
            prompt = get_feature_generation_prompt(
                dataset_description=self.dataset_description,
                target_name=self.target_name,
                current_features=", ".join(self.deps.current_features),
                dataset_summary=dataset_summary,
                max_features=self.max_features,
                estimator_type=self.estimator_type,
                base_classifier=self.base_classifier,
            )

            prompt = f"---ITERATION {itr + 1}/{self.iterations}---\n\n{prompt}"

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

                result = self.feature_agent.run_sync(
                    prompt, deps=self.deps, message_history=messages or None
                )
                llm_duration = time.time() - llm_start_time

                feature_result = result.output
                self.usages.append(result.usage())
                messages.extend(result.new_messages())
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
                code = feature_result.feature_code_to_run
                encoding_code = feature_result.encoding_code_to_run
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
                        full_code=self.full_code,
                        code=code,
                        full_encoding_code=self.full_encoding_code,
                        encoding_code=encoding_code,
                    )
                    eval_duration = time.time() - eval_start_time

                    # Record this iteration's metrics into self.evaluation_history
                    iteration_record = {
                        "iteration": itr + 1,
                        "old_feature_code": self.full_code,
                        "old_encoding_code": self.full_encoding_code,
                        "new_feature_code": self._combine_code_blocks(
                            self.full_code,
                            code,
                        ),
                        "new_encoding_code": encoding_code,
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
                    encoding_name = feature_result.feature_encoding.name

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
                        improvement_roc
                        if self.optimization_metric == "auc"
                        else improvement_acc
                        if self.optimization_metric == "accuracy"
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
                        f"  Encoding Strategy Tested: {encoding_name}\n"
                        f"  Baseline Metrics: ACC {baseline_acc:.4}, ROC AUC {baseline_auc:.4}, PPV@{self.top_p:.1%} {baseline_ppv:.4}\n"
                        f"  Updated Metrics: ACC {enhanced_acc:.4}, ROC AUC {enhanced_auc:.4}, PPV@{self.top_p:.1%} {enhanced_ppv:.4}\n"
                        f"  Improvements: ACC {acc_improvement:+.4}, ROC AUC {auc_improvement:+.4}, PPV {ppv_improvement:+.4}\n"
                        f"  Primary Metric ({self.optimization_metric}): {primary_improvement:+.4}\n"
                        f"  Significant: {is_significant}\n"
                        f"  Evaluation Time: {eval_duration:.2f}s"
                    )
                    self.logger.info(eval_msg)

                    # Create formatted summary string for agent memory
                    performance_summary = (
                        f"Iteration {itr + 1}\n"
                        f"Features created: {', '.join(feature_names) if feature_names else 'None'}\n"
                        f"Features dropped: {', '.join(dropped_feature_names) if dropped_feature_names else 'None'}\n"
                        f"Encoding strategy tested: {encoding_name}\n"
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
                        "encoding_name": encoding_name,
                        "encoding_code": encoding_code,
                        "baseline_encoding_code": self.full_encoding_code,
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
                        self.logger.info(
                            "\n✓ Proposed features show improvement: Keeping them. "
                        )
                        if acc_improvement > 0:
                            self.logger.info(f"\nAccuracy +{acc_improvement:.4}")
                        if auc_improvement > 0:
                            self.logger.info(f"\nROC AUC +{auc_improvement:.4}")
                        if ppv_improvement > 0:
                            self.logger.info(
                                f"\nPPV@{self.top_p:.1%} +{ppv_improvement:.4}"
                            )

                        # Update our accumulated feature code
                        self.full_code = self._combine_code_blocks(self.full_code, code)
                        self.full_encoding_code = encoding_code
                        self.feature_code_history.append(code)
                        self.encoding_code_history.append(encoding_code)
                        self._features_accepted += len(feature_names)
                        self._features_dropped += len(dropped_feature_names)
                        self.accepted_features.extend(feature_names)
                        self.features_dropped.extend(dropped_feature_names)

                        # Apply the complete feature code to get the enhanced dataset for next iteration context
                        enhanced_df = run_llm_code(
                            self.full_code,
                            self.deps.original_dataset,
                        )
                        self.deps.dataset = enhanced_df
                        self._refresh_fitted_encoder()

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
                        if self.best_ppv < np.nanmean(new_results["ppv"]):
                            self.best_ppv = np.nanmean(new_results["ppv"])

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
                f"\nCurrent best PPV@{self.top_p:.1%} score: {self.best_ppv:.3f}"
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
            f"\nActive encoding strategy: {self._describe_encoding_strategy(self.full_encoding_code)}"
            f"\nOriginal ROC AUC: {float(self.baseline_auc)}, "
            f"\nOriginal accuracy: {float(self.baseline_acc)}, "
            f"\nOriginal PPV@{self.top_p:.1%}: {float(self.baseline_ppv)}, "
            f"\nFinal ROC AUC: {float(self.best_auc)}, "
            f"\nFinal accuracy: {float(self.best_acc)}, "
            f"\nFinal PPV@{self.top_p:.1%}: {float(self.best_ppv)}, "
            f"\nTotal duration={total_duration}"
        )

        self._refresh_fitted_encoder()
        self._is_fitted = True
        return self

    def transform(
        self, X: pd.DataFrame, convert_categorical_to_integer: bool = False
    ) -> pd.DataFrame:
        """
        Apply the accepted feature code and encoding pipeline to a new DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain the same original feature columns that were present when fit() was called.
            Does NOT include the target column.
        convert_categorical_to_integer : bool, default=False
            Backward-compatible fallback used only when no encoding pipeline is available.

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
            df_out = run_llm_code(
                self.full_code,
                df_in,
                convert_categorical_to_integer=(
                    convert_categorical_to_integer and not self.full_encoding_code
                ),
            )
            if self.fitted_encoder is not None:
                df_out = self._apply_fitted_encoding(df_out)
        except Exception as e:
            self.logger.error(
                f"Error applying self.full_code in transform: {type(e).__name__}: {e}"
            )
            raise

        return df_out

    def save_code(self, filepath: str) -> None:
        """
        Save feature and encoding code to disk as either a .py or .md.

        - If filepath ends in '.py', writes the combined artifact directly.
        - If filepath ends in '.md', wraps the combined artifact in a triple-backtick
          fence labeled 'python'.

        After writing, logs the location to self.logger.
        """
        if not self.full_code and not self.full_encoding_code:
            self.logger.warning(
                "No feature-generation or encoding code available to save."
            )
            return

        bundled_code = self._serialize_code_bundle()

        if filepath.lower().endswith(".py"):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(bundled_code)
                self.logger.info(f"Feature-generation code saved to {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to save code to {filepath}: {e}")
        elif filepath.lower().endswith(".md"):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("```python\n")
                    f.write(bundled_code)
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
        full_encoding_code: str,
        encoding_code: str,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Compare the accepted feature/encoding stack against a new proposal.

        Args:
            full_code: Accepted feature-engineering code accumulated so far.
            code: Candidate feature-engineering code for this iteration.
            full_encoding_code: Accepted encoding pipeline code accumulated so far.
            encoding_code: Candidate encoding pipeline code for this iteration.

        Returns:
            Tuple of metric dictionaries for the current stack and proposed stack.
        """
        old_results = {"accuracy": [], "auc": [], "ppv": []}
        new_results = {"accuracy": [], "auc": [], "ppv": []}
        old_feature_code = full_code
        new_feature_code = self._combine_code_blocks(full_code, code)
        old_encoding_code = full_encoding_code
        new_encoding_code = encoding_code

        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        # Use original dataset for splitting
        original_df = self.deps.original_dataset
        original_target = original_df[self.target_name].astype(int).to_numpy()
        original_features = original_df.drop(columns=[self.target_name])

        # Preserve label balance across folds during repeated evaluation.
        for train_idx, valid_idx in rskf.split(original_features, original_target):
            df_train, df_valid = (
                original_df.iloc[train_idx],
                original_df.iloc[valid_idx],
            )

            target_train = df_train[self.target_name].astype(int).to_numpy()
            target_valid = df_valid[self.target_name].astype(int).to_numpy()
            df_train = df_train.drop(columns=[self.target_name])
            df_valid = df_valid.drop(columns=[self.target_name])

            try:
                df_train_current = self._apply_feature_code(
                    df_train,
                    old_feature_code,
                )
                df_valid_current = self._apply_feature_code(
                    df_valid,
                    old_feature_code,
                )
                df_train_extended = self._apply_feature_code(
                    df_train,
                    new_feature_code,
                )
                df_valid_extended = self._apply_feature_code(
                    df_valid,
                    new_feature_code,
                )

            except Exception as e:
                self.logger.warning(
                    f"Error during fold evaluation: {type(e).__name__}: {e}"
                )
                continue
            old_result = self._evaluate_dataset(
                df_train_current,
                df_valid_current,
                target_train,
                target_valid,
                old_encoding_code,
            )
            new_result = self._evaluate_dataset(
                df_train_extended,
                df_valid_extended,
                target_train,
                target_valid,
                new_encoding_code,
            )

            for metric in ["accuracy", "auc", "ppv"]:
                old_results[metric].append(old_result[metric])
                new_results[metric].append(new_result[metric])

        return old_results, new_results

    def _evaluate_dataset(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_train: np.ndarray,
        target_test: np.ndarray,
        encoding_code: str,
    ) -> Dict[str, float]:
        """Evaluate model performance on encoded train and test feature sets.

        This method applies the requested encoding pipeline to the engineered feature
        dataframes, trains the base classifier, and evaluates performance using
        accuracy, AUC, and PPV metrics.

        Args:
            df_train: Training features after feature engineering.
            df_test: Validation features after feature engineering.
            target_train: Training targets.
            target_test: Validation targets.
            encoding_code: Encoding pipeline code for the engineered features.

        Returns:
            Dictionary containing accuracy, AUC, and PPV scores.
        """
        df_train = df_train.copy().replace([np.inf, -np.inf], np.nan)
        df_test = df_test.copy().replace([np.inf, -np.inf], np.nan)
        train_x, test_x, _ = run_llm_encoder_code(
            encoding_code,
            df_train,
            df_test,
        )
        self.base_classifier.fit(X=train_x, y=target_train)
        probs = self.base_classifier.predict_proba(test_x)
        acc = float(accuracy_metric(target_test, probs))
        auc = float(auc_metric(target_test, probs))

        # Compute PPV at top_p% - extract positive class probabilities
        if probs.shape[1] == 2:  # Binary classification
            positive_probs = probs[:, 1]  # Probabilities for positive class
        else:  # Single class (should not happen in binary classification)
            positive_probs = probs.flatten()

        ppv = float(metric_ppv(target_test, positive_probs, top_p=self.top_p))

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
