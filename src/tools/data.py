from dataclasses import dataclass, field
import os

import datasets
import pandas as pd

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from src.utils import exploratory_data_analysis, format_eda_for_llm


@dataclass
class AnalystAgentDeps:
    """Dependencies for the AnalystAgent."""
    
    data_directory: str = field(default="data")
    _directory_list: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure the data directory exists."""
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
                    
    def list_files_in_data_directory(self) -> list[str]:
        """Get a list of files in the data directory.

        Returns:
            str: A string containing the list of files in the data directory, one per line.
        """
        # Get list of files in data directory
        files = os.listdir(self.data_directory)
        # Filter out hidden files and directories
        files = [f for f in files if not f.startswith('.') or f not in ['__pycache__', 'node_modules', '.git']]
        # Return as newline-separated string
        return files
    

def load_huggingface_dataset(
    ctx: RunContext[AnalystAgentDeps],
    path: str,
    split: str = 'train',
) -> str:
    """Load a dataset from Hugging Face. Note the path follows the format `<user_name>/<dataset_name>`.

    Args:
        ctx: Pydantic AI agent RunContext
        path: The name of the dataset in the form of `<user_name>/<dataset_name>`. 
        split: load the split of the dataset (default: "train")
    """
    if os.path.exists(path):
        return f'Dataset loaded to `{path}`.'
    
    clean_name = path.replace('/', '_').replace('-', '_').lower()
    suffix = ".csv" if not path.endswith(".csv") else ""
    
    clean_path = f"{ctx.deps.data_directory}/{clean_name}_{split}{suffix}"
    if os.path.exists(clean_path):
        return f'Dataset loaded to `{clean_path}`.'
    
    # begin load data from hf
    try:
        builder = datasets.load_dataset_builder(path)  # pyright: ignore[reportUnknownMemberType]
        splits: dict[str, datasets.SplitInfo] = builder.info.splits or {}  # pyright: ignore[reportUnknownMemberType]
        if split not in splits:
            raise ModelRetry(
                f'{split} is not valid for dataset {path}. Valid splits are {",".join(splits.keys())}'
            )
        builder.download_and_prepare()  # pyright: ignore[reportUnknownMemberType]
    except Exception as e:
        raise ModelRetry(f"Error loading dataset from Hugging Face: {str(e)}")
    
    dataset = builder.as_dataset(split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)
    # end load data from hf
        
    # Save the dataframe to the data directory
    dataframe.to_csv(clean_path, index=False)

    # construct a summary of the loaded dataset
    output = [
        f'Dataset loaded to `{clean_path}`.',
        f'Description: {dataset.info.description}'
        if dataset.info.description
        else None,
        f'Features: {dataset.info.features!r}' if dataset.info.features else None,
    ]
    return '\n'.join(filter(None, output))



def get_eda_report(ctx: RunContext[AnalystAgentDeps], dataset_path: str, target: str | None = None) -> str:
    """Get a comprehensive exploratory data analysis of a given dataset.
    
    Args:
        ctx: Pydantic AI agent RunContext
        dataset_path: Path to the dataset, e.g. "data/dataset_name.csv"
        target: The target column name (optional)

    Returns:
        str: A comprehensive exploratory data analysis of the given dataset
    """
    try:
        df = pd.read_csv(dataset_path)
    except ModelRetry:
        raise ModelRetry(f"Dataset '{dataset_path}' not found in context. Please load the dataset using `load_huggingface_dataset` \
tool, or pick from the following datasets: {str(ctx.deps.list_files_in_data_directory())}")
    
    if target is not None and target not in df.columns:
        valid_columns = df.columns.tolist()
        raise ModelRetry(f"Target column '{target}' not found in dataset. Please select from the following columns: {valid_columns}")

    result_dict = exploratory_data_analysis(df, target=target)
    return format_eda_for_llm(result_dict)


data_tools = FunctionToolset(tools=[load_huggingface_dataset, get_eda_report], max_retries=5, id="data_toolset")
