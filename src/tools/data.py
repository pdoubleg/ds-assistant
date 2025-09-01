from dataclasses import dataclass, field

import datasets
import duckdb
import pandas as pd

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from src.utils import exploratory_data_analysis, format_eda_for_llm
from .logging import LoggingToolset


@dataclass
class AnalystAgentDeps:
    """Dependencies for the AnalystAgent."""
    
    datasets: dict[str, pd.DataFrame] = field(default_factory=dict)

    def store(self, value: pd.DataFrame) -> str:
        """Store the output in deps and return the reference such as dataframe_1.csv to be used by the LLM."""
        ref = f'dataframe_{len(self.datasets) + 1}.csv'
        self.datasets[ref] = value
        value.to_csv(ref, index=False)
        return ref

    def get(self, ref: str) -> pd.DataFrame:
        if ref not in self.datasets:
            raise ModelRetry(
                f'Error: {ref} is not a valid variable reference. Check the previous messages and try again.'
            )
        return self.datasets[ref]
    
    def list_datasets(self) -> list[str]:
        return list(self.datasets.keys())
    

def load_huggingface_dataset(
    ctx: RunContext[AnalystAgentDeps],
    path: str,
    split: str = 'train',
) -> str:
    """Load the dataset `dataset_name` from huggingface.

    Args:
        ctx: Pydantic AI agent RunContext
        path: name of the dataset in the form of `<user_name>/<dataset_name>`
        split: load the split of the dataset (default: "train")
    """
    # begin load data from hf
    builder = datasets.load_dataset_builder(path)  # pyright: ignore[reportUnknownMemberType]
    splits: dict[str, datasets.SplitInfo] = builder.info.splits or {}  # pyright: ignore[reportUnknownMemberType]
    if split not in splits:
        raise ModelRetry(
            f'{split} is not valid for dataset {path}. Valid splits are {",".join(splits.keys())}'
        )

    builder.download_and_prepare()  # pyright: ignore[reportUnknownMemberType]
    dataset = builder.as_dataset(split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)
    # end load data from hf

    # store the dataframe in the deps and get a ref like "dataframe_1.csv"
    ref = ctx.deps.store(dataframe)
    # construct a summary of the loaded dataset
    output = [
        f'Loaded the dataset as `{ref}`.',
        f'Description: {dataset.info.description}'
        if dataset.info.description
        else None,
        f'Features: {dataset.info.features!r}' if dataset.info.features else None,
    ]
    return '\n'.join(filter(None, output))


def run_duckdb(ctx: RunContext[AnalystAgentDeps], dataset_ref: str, sql: str) -> str:
    """Run DuckDB SQL query on the DataFrame.

    Note that the virtual table name used in DuckDB SQL must be `dataset`.

    Args:
        ctx: Pydantic AI agent RunContext
        dataset_ref: reference string to the DataFrame, e.g. "dataframe_1.csv" for creating a virtual table named "dataset"
        sql: the query to be executed using DuckDB
    """
    try:
        data = ctx.deps.get(dataset_ref)
    except ModelRetry:
        raise ModelRetry(f"Dataset '{dataset_ref}' not found in context. Please load the dataset using `load_huggingface_dataset` \
tool, or pick from the following datasets: {ctx.deps.list_datasets()}")
        
    try:
        result = duckdb.query_df(df=data, virtual_table_name='dataset', sql_query=sql)
    except Exception as e:
        raise ModelRetry(f"Error executing SQL query: {e}") from e
    
    ref = ctx.deps.store(result.df())
    return f'Executed SQL, result is `{ref}`'


def get_eda_analysis(ctx: RunContext[AnalystAgentDeps], dataset_ref: str, target: str | None = None) -> str:
    """Exploratory data analysis.
    
    Args:
        ctx: Pydantic AI agent RunContext
        dataset_ref: reference string to the DataFrame, e.g. "dataframe_1.csv"
        target: the target column name (optional)

    Returns:
        str: Exploratory data analysis results
    """
    try:
        df = ctx.deps.get(dataset_ref)
    except ModelRetry:
        raise ModelRetry(f"Dataset '{dataset_ref}' not found in context. Please load the dataset using `load_huggingface_dataset` \
tool, or pick from the following datasets: {ctx.deps.list_datasets()}")
    
    if target is not None and target not in df.columns:
        valid_columns = df.columns.tolist()
        raise ModelRetry(f"Target column '{target}' not found in dataset. Please select from the following columns: {valid_columns}")

    result_dict = exploratory_data_analysis(df, target=target)
    return format_eda_for_llm(result_dict)


data_tools = FunctionToolset(tools=[load_huggingface_dataset, get_eda_analysis, run_duckdb], max_retries=5, id="data_toolset")
