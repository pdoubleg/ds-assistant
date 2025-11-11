"""Configuration-based scaffolding for GEPA optimization setup.

This module provides a simplified interface for setting up GEPA prompt optimization
through a configuration-based approach, reducing boilerplate and setup complexity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from .runner import GepaOptimizationResult, optimize_agent_prompts
from .signature_agent import SignatureAgent
from .types import DataInstWithInput, RolloutOutput

# Type variables
InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


@dataclass
class GepaConfig:
    """Configuration for GEPA optimization setup.

    This class captures all parameters needed to run GEPA optimization,
    providing a clean interface for users to configure their optimization runs.

    Args:
        agent_model: The model name or Model instance for the agent (e.g., "openai:gpt-4.1-mini").
        agent_instructions: System instructions for the agent.
        input_type: Pydantic model class defining the structured input format.
        output_type: Pydantic model class defining the expected output format.
        dataset: List of DataInstWithInput instances for training and validation.
        metric: Function that evaluates agent outputs, returning (score, feedback).
        train_ratio: Ratio of dataset to use for training (default: 0.7).
        agent_tools: Optional list of tool functions to register with the agent.
        optimize_tools: Whether to optimize tool descriptions (default: True).
        seed_candidate: Optional initial candidate prompts to start optimization from.
        reflection_model: Model to use for reflection/mutation (default: None, uses agent_model).
        max_metric_calls: Maximum number of metric evaluations (default: 100).
        module_selector: Which components to optimize - "all", "instructions", etc. (default: "all").
        display_progress_bar: Whether to show progress bar during optimization (default: True).
        track_best_outputs: Whether to track best outputs for analysis (default: True).
        enable_cache: Whether to enable caching of metric results (default: False).
        cache_dir: Directory for cache storage (default: ".gepa_cache").
        cache_verbose: Whether to print cache statistics (default: False).
        output_dir: Directory to save optimization results (default: "optimization_results").
        save_result: Whether to automatically save results to JSON (default: True).

    Example:
        >>> from pydantic import BaseModel, Field
        >>> from src.gepa.scaffold import GepaConfig, setup_optimization
        >>> from src.gepa.data_utils import dataframe_to_dataset
        >>>
        >>> class MyInput(BaseModel):
        ...     text: str = Field(description="Input text")
        >>>
        >>> class MyOutput(BaseModel):
        ...     category: str = Field(description="Classification category")
        >>>
        >>> def my_metric(data_inst, output):
        ...     if output.success and output.result:
        ...         return 1.0 if output.result.category == data_inst.metadata["label"] else 0.0, None
        ...     return 0.0, "Failed to produce output"
        >>>
        >>> config = GepaConfig(
        ...     agent_model="openai:gpt-4.1-mini",
        ...     agent_instructions="Classify the input text",
        ...     input_type=MyInput,
        ...     output_type=MyOutput,
        ...     dataset=my_dataset,
        ...     metric=my_metric,
        ... )
        >>> result = setup_optimization(config)
    """

    # Core agent configuration
    agent_model: Model | KnownModelName | str
    agent_instructions: str
    input_type: type[BaseModel]
    output_type: type[BaseModel]

    # Dataset and evaluation
    dataset: Sequence[DataInstWithInput[Any]]
    metric: Callable[
        [DataInstWithInput[Any], RolloutOutput[Any]], tuple[float, str | None]
    ]
    train_ratio: float = 0.7

    # Agent tools configuration
    agent_tools: list[Callable[..., Any]] | None = None
    optimize_tools: bool = True

    # Optimization parameters
    seed_candidate: dict[str, str] | None = None
    reflection_model: Model | KnownModelName | str | None = None
    max_metric_calls: int = 100
    module_selector: ReflectionComponentSelector | str = "all" # "round_robin"

    # Runtime options
    display_progress_bar: bool = True
    track_best_outputs: bool = True

    # Caching options
    enable_cache: bool = False
    cache_dir: str = ".gepa_cache"
    cache_verbose: bool = False

    # Output options
    output_dir: str | Path = "optimization_results"
    save_result: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0 < self.train_ratio < 1:
            raise ValueError(
                f"train_ratio must be between 0 and 1, got {self.train_ratio}"
            )

        if len(self.dataset) == 0:
            raise ValueError("dataset cannot be empty")

        if self.max_metric_calls < 1:
            raise ValueError(
                f"max_metric_calls must be positive, got {self.max_metric_calls}"
            )


def run_optimization_pipeline(config: GepaConfig) -> GepaOptimizationResult:
    """Set up and run GEPA optimization based on configuration.

    This function orchestrates the complete optimization workflow:
    1. Validates the configuration
    2. Splits dataset into training and validation sets
    3. Creates the agent with specified configuration
    4. Wraps agent with SignatureAgent for structured input support
    5. Runs GEPA optimization
    6. Optionally saves results to disk

    Args:
        config: GepaConfig instance with all optimization parameters.

    Returns:
        GepaOptimizationResult containing the best candidate, scores, and metadata.

    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If optimization fails.

    Example:
        >>> config = GepaConfig(
        ...     agent_model="openai:gpt-4.1-mini",
        ...     agent_instructions="Classify sentiment as positive, negative, or neutral",
        ...     input_type=SentimentInput,
        ...     output_type=SentimentOutput,
        ...     dataset=training_data,
        ...     metric=sentiment_metric,
        ...     train_ratio=0.7,
        ...     max_metric_calls=50,
        ... )
        >>> result = setup_optimization(config)
        >>> print(f"Best score: {result.best_score:.4f}")
        >>> print(f"Improvement: {result.improvement_ratio():.2%}")
        >>>
        >>> # Apply best candidate to agent
        >>> with result.apply_best(agent):
        ...     output = agent.run_sync("This is great!")
    """
    import json
    from datetime import datetime

    # Validate configuration
    if len(config.dataset) < 2:
        raise ValueError("Dataset must contain at least 2 examples for train/val split")

    # Split dataset into train and validation sets
    split_index = int(len(config.dataset) * config.train_ratio)
    if split_index == 0 or split_index == len(config.dataset):
        raise ValueError(
            f"Invalid train_ratio {config.train_ratio} for dataset size {len(config.dataset)}. "
            f"Results in empty train or validation set."
        )

    trainset = config.dataset[:split_index]
    valset = config.dataset[split_index:]

    print(f"Dataset split: {len(trainset)} training, {len(valset)} validation examples")

    # Create the base agent
    agent = Agent(
        model=config.agent_model,
        instructions=config.agent_instructions,
        output_type=config.output_type,
    )

    # Register tools if provided
    if config.agent_tools:
        for tool_func in config.agent_tools:
            agent.tool(tool_func)

    # Wrap with SignatureAgent for structured input support
    signature_agent = SignatureAgent(
        agent,
        input_type=config.input_type,
        optimize_tools=config.optimize_tools,
    )

    # Run optimization
    print("Starting GEPA optimization...")
    result = optimize_agent_prompts(
        agent=signature_agent,
        seed_candidate=config.seed_candidate,
        trainset=trainset,
        valset=valset,
        module_selector=config.module_selector,
        metric=config.metric,
        input_type=config.input_type,
        reflection_model=config.reflection_model,
        max_metric_calls=config.max_metric_calls,
        display_progress_bar=config.display_progress_bar,
        track_best_outputs=config.track_best_outputs,
        enable_cache=config.enable_cache,
        cache_dir=config.cache_dir,
        cache_verbose=config.cache_verbose,
    )

    # Save result if requested
    if config.save_result:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"optimization_{timestamp}.json"

        result_dict = result.model_dump()
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"\n✅ Optimization result saved to: {output_file}")

    # Print summary
    print(f"   Best score: {result.best_score:.4f}")
    print(f"   Iterations: {result.num_iterations}")
    print(f"   Metric calls: {result.num_metric_calls}")

    improvement = result.improvement_ratio()
    if improvement is not None:
        print(f"   Improvement: {improvement:.2%}")

    return result
