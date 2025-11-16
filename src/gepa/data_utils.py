"""Utility functions for loading and transforming data into GEPA-compatible formats.

This module provides helper functions to convert common data formats (DataFrames, JSON)
into the DataInstWithInput format required by GEPA optimization.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .types import DataInstWithInput

# Type variable for input models
InputModelT = TypeVar("InputModelT", bound=BaseModel)


def dataframe_to_dataset(
    df: Any,
    row_mapper: Callable[[Any], InputModelT],
    metadata_cols: list[str] | None = None,
    case_id_col: str | None = None,
) -> list[DataInstWithInput[InputModelT]]:
    """Convert a pandas DataFrame to a list of DataInstWithInput instances.

    This function provides a flexible way to transform tabular data into the format
    required by GEPA optimization. Users provide a custom mapping function to convert
    DataFrame rows into their input model instances.

    Args:
        df: pandas DataFrame containing the data.
        row_mapper: Function that takes a DataFrame row and returns an input model instance.
            The row is passed as a pandas Series with column names as keys.
        metadata_cols: List of column names to include in metadata dict. If None, all
            columns not used in the input model are included. Default: None.
        case_id_col: Column name to use for case IDs. If None, uses row index. Default: None.

    Returns:
        List of DataInstWithInput instances ready for GEPA optimization.

    Raises:
        ValueError: If DataFrame is empty or required columns are missing.

    Example:
        >>> import pandas as pd
        >>> from pydantic import BaseModel, Field
        >>> from src.gepa.data_utils import dataframe_to_dataset
        >>>
        >>> class SentimentInput(BaseModel):
        ...     text: str = Field(description="Text to classify")
        ...     context: str = Field(description="Context information")
        >>>
        >>> # Load data
        >>> df = pd.DataFrame({
        ...     'text': ['Great product!', 'Terrible service'],
        ...     'context': ['Product review', 'Service feedback'],
        ...     'label': ['positive', 'negative'],
        ...     'confidence': [0.9, 0.8]
        ... })
        >>>
        >>> # Define mapping function
        >>> def row_to_input(row):
        ...     return SentimentInput(text=row['text'], context=row['context'])
        >>>
        >>> # Convert to dataset
        >>> dataset = dataframe_to_dataset(
        ...     df,
        ...     row_mapper=row_to_input,
        ...     metadata_cols=['label', 'confidence']
        ... )
        >>>
        >>> print(f"Created {len(dataset)} examples")
        >>> print(f"First example metadata: {dataset[0].metadata}")
    """
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    dataset: list[DataInstWithInput[InputModelT]] = []

    for idx, row in df.iterrows():
        # Convert row to input model using user-provided mapper
        try:
            input_instance = row_mapper(row)
        except Exception as e:
            raise ValueError(f"Error mapping row {idx} to input model: {e}") from e

        # Extract metadata
        if metadata_cols is not None:
            metadata = {col: row[col] for col in metadata_cols if col in row}
        else:
            # Include all columns as metadata
            metadata = row.to_dict()

        # Determine case ID
        if case_id_col and case_id_col in row:
            case_id = str(row[case_id_col])
        else:
            case_id = f"row-{idx}"

        # Create DataInstWithInput
        data_inst = DataInstWithInput[InputModelT](
            input=input_instance,
            message_history=None,
            metadata=metadata,
            case_id=case_id,
        )
        dataset.append(data_inst)

    return dataset


def json_to_dataset(
    json_path: str | Path,
    input_mapper: Callable[[dict[str, Any]], InputModelT],
    metadata_keys: list[str] | None = None,
    case_id_key: str | None = None,
) -> list[DataInstWithInput[InputModelT]]:
    """Load data from a JSON file and convert to DataInstWithInput instances.

    This function reads a JSON file containing a list of objects and converts each
    object into a DataInstWithInput instance using a user-provided mapping function.

    Args:
        json_path: Path to the JSON file. Should contain a list of objects.
        input_mapper: Function that takes a dict and returns an input model instance.
        metadata_keys: List of keys to include in metadata dict. If None, all keys
            not used in the input model are included. Default: None.
        case_id_key: Key to use for case IDs. If None, uses array index. Default: None.

    Returns:
        List of DataInstWithInput instances ready for GEPA optimization.

    Raises:
        ValueError: If JSON file is invalid or empty.
        FileNotFoundError: If JSON file doesn't exist.

    Example:
        >>> from pydantic import BaseModel, Field
        >>> from src.gepa.data_utils import json_to_dataset
        >>>
        >>> class QueryInput(BaseModel):
        ...     question: str = Field(description="User question")
        ...     domain: str = Field(description="Question domain")
        >>>
        >>> # JSON file contains:
        >>> # [
        >>> #   {"question": "What is Python?", "domain": "programming", "label": "definition"},
        >>> #   {"question": "How to sort a list?", "domain": "programming", "label": "how-to"}
        >>> # ]
        >>>
        >>> def dict_to_input(data):
        ...     return QueryInput(question=data['question'], domain=data['domain'])
        >>>
        >>> dataset = json_to_dataset(
        ...     'questions.json',
        ...     input_mapper=dict_to_input,
        ...     metadata_keys=['label'],
        ...     case_id_key='question'
        ... )
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects")

    if len(data) == 0:
        raise ValueError("JSON file contains an empty list")

    dataset: list[DataInstWithInput[InputModelT]] = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a dict: {type(item)}")

        # Convert dict to input model using user-provided mapper
        try:
            input_instance = input_mapper(item)
        except Exception as e:
            raise ValueError(f"Error mapping item {idx} to input model: {e}") from e

        # Extract metadata
        if metadata_keys is not None:
            metadata = {key: item[key] for key in metadata_keys if key in item}
        else:
            # Include all keys as metadata
            metadata = item.copy()

        # Determine case ID
        if case_id_key and case_id_key in item:
            case_id = str(item[case_id_key])
        else:
            case_id = f"item-{idx}"

        # Create DataInstWithInput
        data_inst = DataInstWithInput[InputModelT](
            input=input_instance,
            message_history=None,
            metadata=metadata,
            case_id=case_id,
        )
        dataset.append(data_inst)

    return dataset


def split_dataset(
    dataset: list[DataInstWithInput[InputModelT]],
    train_ratio: float = 0.7,
    shuffle: bool = False,
    random_seed: int | None = None,
) -> tuple[list[DataInstWithInput[InputModelT]], list[DataInstWithInput[InputModelT]]]:
    """Split a dataset into training and validation sets.

    Args:
        dataset: List of DataInstWithInput instances to split.
        train_ratio: Ratio of data to use for training (default: 0.7).
        shuffle: Whether to shuffle the dataset before splitting (default: False).
        random_seed: Random seed for reproducible shuffling (default: None).

    Returns:
        Tuple of (trainset, valset) as lists of DataInstWithInput instances.

    Raises:
        ValueError: If train_ratio is not between 0 and 1, or if dataset is too small.

    Example:
        >>> from src.gepa.data_utils import split_dataset
        >>>
        >>> # Assume we have a dataset with 100 examples
        >>> trainset, valset = split_dataset(dataset, train_ratio=0.8, shuffle=True, random_seed=42)
        >>> print(f"Training: {len(trainset)}, Validation: {len(valset)}")
        Training: 80, Validation: 20
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 examples for splitting")

    # Shuffle if requested
    if shuffle:
        import random

        if random_seed is not None:
            random.seed(random_seed)
        dataset = dataset.copy()
        random.shuffle(dataset)

    # Calculate split index
    split_index = int(len(dataset) * train_ratio)

    if split_index == 0 or split_index == len(dataset):
        raise ValueError(
            f"Invalid train_ratio {train_ratio} for dataset size {len(dataset)}. "
            f"Results in empty train or validation set."
        )

    trainset = dataset[:split_index]
    valset = dataset[split_index:]

    return trainset, valset


def create_dataset_from_dicts(
    data: list[dict[str, Any]],
    input_model: type[InputModelT],
    input_keys: list[str],
    metadata_keys: list[str] | None = None,
    case_id_key: str | None = None,
) -> list[DataInstWithInput[InputModelT]]:
    """Create a dataset from a list of dictionaries with automatic field mapping.

    This is a convenience function that automatically maps dictionary keys to
    input model fields, useful when the dictionary structure closely matches
    the input model.

    Args:
        data: List of dictionaries containing the data.
        input_model: The Pydantic model class for structured inputs.
        input_keys: List of dictionary keys to use for creating input model instances.
            These keys will be passed as kwargs to the input model constructor.
        metadata_keys: List of keys to include in metadata dict. If None, all keys
            not in input_keys are included. Default: None.
        case_id_key: Key to use for case IDs. If None, uses array index. Default: None.

    Returns:
        List of DataInstWithInput instances ready for GEPA optimization.

    Raises:
        ValueError: If data is empty or required keys are missing.

    Example:
        >>> from pydantic import BaseModel, Field
        >>> from src.gepa.data_utils import create_dataset_from_dicts
        >>>
        >>> class TaskInput(BaseModel):
        ...     instruction: str = Field(description="Task instruction")
        ...     context: str = Field(description="Task context")
        >>>
        >>> data = [
        ...     {'instruction': 'Summarize this', 'context': 'Long text...', 'label': 'summary'},
        ...     {'instruction': 'Translate this', 'context': 'English text...', 'label': 'translation'},
        ... ]
        >>>
        >>> dataset = create_dataset_from_dicts(
        ...     data,
        ...     input_model=TaskInput,
        ...     input_keys=['instruction', 'context'],
        ...     metadata_keys=['label']
        ... )
    """
    if len(data) == 0:
        raise ValueError("Data list is empty")

    dataset: list[DataInstWithInput[InputModelT]] = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a dict: {type(item)}")

        # Extract input fields
        input_kwargs = {}
        for key in input_keys:
            if key not in item:
                raise ValueError(f"Required key '{key}' not found in item {idx}")
            input_kwargs[key] = item[key]

        # Create input model instance
        try:
            input_instance = input_model(**input_kwargs)
        except Exception as e:
            raise ValueError(f"Error creating input model for item {idx}: {e}") from e

        # Extract metadata
        if metadata_keys is not None:
            metadata = {key: item[key] for key in metadata_keys if key in item}
        else:
            # Include all keys not used in input
            metadata = {k: v for k, v in item.items() if k not in input_keys}

        # Determine case ID
        if case_id_key and case_id_key in item:
            case_id = str(item[case_id_key])
        else:
            case_id = f"item-{idx}"

        # Create DataInstWithInput
        data_inst = DataInstWithInput[InputModelT](
            input=input_instance,
            message_history=None,
            metadata=metadata,
            case_id=case_id,
        )
        dataset.append(data_inst)

    return dataset


def prepare_train_val_sets(
    data: list[dict[str, Any]],
    input_model: type[InputModelT],
    input_keys: list[str],
    train_ratio: float = 0.7,
    shuffle: bool = True,
    random_seed: int | None = 42,
    metadata_keys: list[str] | None = None,
    case_id_key: str | None = None,
) -> tuple[list[DataInstWithInput[InputModelT]], list[DataInstWithInput[InputModelT]]]:
    """Create and split a dataset from dictionaries into train and validation sets.

    This is a convenience function that combines dataset creation and splitting
    into a single step, making it easy to prepare data for GEPA optimization.

    Args:
        data: List of dictionaries containing the data.
        input_model: The Pydantic model class for structured inputs.
        input_keys: List of dictionary keys to use for creating input model instances.
        train_ratio: Ratio of data to use for training (default: 0.7).
        shuffle: Whether to shuffle the dataset before splitting (default: True).
        random_seed: Random seed for reproducible shuffling (default: 42).
        metadata_keys: List of keys to include in metadata dict. If None, all keys
            not in input_keys are included. Default: None.
        case_id_key: Key to use for case IDs. If None, uses array index. Default: None.

    Returns:
        Tuple of (trainset, valset) as lists of DataInstWithInput instances.

    Raises:
        ValueError: If data is empty, train_ratio is invalid, or dataset is too small.

    Example:
        >>> from pydantic import BaseModel, Field
        >>> from src.gepa.data_utils import prepare_train_val_sets
        >>>
        >>> class SentimentInput(BaseModel):
        ...     text: str = Field(description="Text to classify")
        >>>
        >>> data = [
        ...     {'text': 'Great product!', 'label': 'positive'},
        ...     {'text': 'Terrible service', 'label': 'negative'},
        ...     # ... more examples
        ... ]
        >>>
        >>> trainset, valset = prepare_train_val_sets(
        ...     data,
        ...     input_model=SentimentInput,
        ...     input_keys=['text'],
        ...     metadata_keys=['label'],
        ...     train_ratio=0.8,
        ...     shuffle=True,
        ...     random_seed=42
        ... )
        >>> print(f"Training: {len(trainset)}, Validation: {len(valset)}")
    """
    # Create the full dataset
    dataset = create_dataset_from_dicts(
        data=data,
        input_model=input_model,
        input_keys=input_keys,
        metadata_keys=metadata_keys,
        case_id_key=case_id_key,
    )

    # Split into train and validation sets
    trainset, valset = split_dataset(
        dataset=dataset,
        train_ratio=train_ratio,
        shuffle=shuffle,
        random_seed=random_seed,
    )

    return trainset, valset