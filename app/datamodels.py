import ast
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass
import pandas as pd
from typing import Any, List, Optional

from run_llm_code import check_ast


@dataclass
class AutoMLDependencies:
    """Dependencies for AutoML agents."""

    original_dataset: pd.DataFrame
    dataset: pd.DataFrame
    target_name: str
    dataset_description: str
    current_features: List[str]
    agent_notepad: List[str]
    error_log: List[str]
    tool_output_log: List[str]
    prompt_log: List[str]
    
    
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