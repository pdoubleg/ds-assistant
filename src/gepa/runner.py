"""High-level API for GEPA optimization of pydantic-ai agents."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import gepa.api
from gepa.core.result import GEPAResult
from gepa.logging.logger import LoggerProtocol
from gepa.utils import StopperProtocol
from gepa.gepa_utils import find_dominator_programs
from gepa.proposer.reflective_mutation.base import (
    ReflectionComponentSelector,
    CandidateSelector,
)
from pydantic import BaseModel, ConfigDict, Field

from .adapter import PydanticAIGEPAAdapter, ReflectionSampler
from .cache import CacheManager
from .components import (
    apply_candidate_to_agent,
    apply_candidate_to_agent_and_signature,
    extract_seed_candidate_with_signature,
    normalize_component_text,
)
from .signature import InputSpec
from .types import DataInst, RolloutOutput
from .lm import GEPALanguageModel
from pydantic_ai import usage as _usage
# Type variable for the DataInst type
DataInstT = TypeVar("DataInstT", bound=DataInst)

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent
    from .lm import GEPALanguageModel
    
    
AUTO_RUN_SETTINGS = {
    "light": {"n": 6},
    "medium": {"n": 12},
    "heavy": {"n": 18},
}


def auto_budget(num_candidates: int, valset_size: int, minibatch_size: int = 35, full_eval_steps: int = 5) -> int:
    """Calculate the total budget for GEPA optimization based on configuration parameters.
    
    This function estimates the total number of metric calls needed for a GEPA optimization
    run based on the number of candidates, validation set size, minibatch size, and 
    full evaluation frequency.
    
    Note: Logic is based on this DSPy version: https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa.py
        
    Args:
        num_candidates: Number of candidate programs to generate.
        valset_size: Size of the validation set.
        minibatch_size: Size of minibatches for evaluation. Defaults to 35.
        full_eval_steps: Number of steps between full evaluations. Defaults to 5.
        
    Returns:
        Total estimated number of metric calls needed.
        
    Raises:
        ValueError: If num_trials, valset_size, or minibatch_size are negative,
                   or if full_eval_steps is less than 1.
    """
    import numpy as np
    num_trials = int(max(2 * (1 * 2) * np.log2(num_candidates), 1.5 * num_candidates))
    if num_trials < 0 or valset_size < 0 or minibatch_size < 0:
        raise ValueError("num_trials, valset_size, and minibatch_size must be >= 0.")
    if full_eval_steps < 1:
        raise ValueError("full_eval_steps must be >= 1.")

    V = valset_size
    N = num_trials
    M = minibatch_size
    m = full_eval_steps

    # Initial full evaluation on the default program
    total = V

    # Assume upto 5 trials for each candidate
    total += num_candidates * 5

    # N minibatch evaluations
    total += N * M
    if N == 0:
        return total  # no periodic/full evals inside the loop
    # Periodic full evals occur when trial_num % (m+1) == 0, where trial_num runs 2..N+1
    periodic_fulls = (N + 1) // (m) + 1
    # If 1 <= N < m, the code triggers one final full eval at the end
    extra_final = 1 if N < m else 0

    total += (periodic_fulls + extra_final) * V
    return total
    
    
def _normalize_candidate(
    candidate: dict[str, Any] | None,
) -> dict[str, str]:
    if not candidate:
        return {}
    return {
        key: normalize_component_text(value)
        for key, value in candidate.items()
    }
    

def dag_to_dot(
    parent_program_for_candidate: list[list[int | None]], 
    dominator_program_ids: set[int], 
    best_program_idx: int, 
    full_eval_scores: list[float]
) -> str:
    """Generate a DOT graph representation of the program evolution DAG.
    
    Creates a directed acyclic graph (DAG) visualization showing the evolution
    of programs during optimization, with special highlighting for the best
    program and dominator programs.
    
    Args:
        parent_program_for_candidate: List where each index represents a program
            and contains a list of parent program indices (or None for no parent).
        dominator_program_ids: Set of program indices that are dominators in the DAG.
        best_program_idx: Index of the best performing program.
        full_eval_scores: List of evaluation scores for each program.
        
    Returns:
        DOT format string representing the program evolution graph.
    """
    dot_lines = [
        "digraph G {",
        "    node [style=filled, shape=circle, fontsize=50];"
    ]
    n = len(parent_program_for_candidate)
    # Set up nodes with colors and scores in labels
    for idx in range(n):
        score = full_eval_scores[idx]
        label = f"{idx}\\n({score:.2f})"
        if idx == best_program_idx:
            dot_lines.append(f'    {idx} [label="{label}", fillcolor=cyan, fontcolor=black];')
        elif idx in dominator_program_ids:
            dot_lines.append(f'    {idx} [label="{label}", fillcolor=orange, fontcolor=black];')
        else:
            dot_lines.append(f'    {idx} [label="{label}"];')
    
    # Set up edges
    for child, parents in enumerate(parent_program_for_candidate):
        for parent in parents:
            if parent is not None:
                dot_lines.append(f'    {parent} -> {child};')
    
    dot_lines.append("}")
    return "\n".join(dot_lines)


class GepaOptimizationResult(BaseModel):
    """Result from GEPA optimization."""

    best_candidate: dict[str, str]
    """The best candidate found during optimization."""

    best_score: float
    """The validation score of the best candidate."""

    original_candidate: dict[str, str]
    """The original candidate before optimization."""

    original_score: float | None
    """The validation score of the original candidate (if evaluated)."""

    num_iterations: int
    """Number of optimization iterations performed."""

    num_metric_calls: int
    """Total number of metric evaluations performed."""

    raw_result: GEPAResult[dict[str, str], RolloutOutput[Any]] | None = Field(
        default=None, exclude=True, repr=False
    )
    """The raw GEPA optimization result (for advanced users)."""

    gepa_usage: _usage.RunUsage
    """The GEPA usage of the optimization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @contextmanager
    def apply_best(self, agent: AbstractAgent[Any, Any]) -> Iterator[None]:
        """Apply the best candidate to an agent as a context manager.

        Args:
            agent: The agent to apply the best candidate to.

        Yields:
            None while the context is active.
        """
        with apply_candidate_to_agent(agent, self.best_candidate):
            yield

    def improvement_ratio(self) -> float | None:
        """Calculate the improvement ratio from original to best.

        Returns:
            The ratio of improvement, or None if original score is not available.
        """
        if self.original_score is not None and self.original_score > 0:
            return (self.best_score - self.original_score) / self.original_score
        return None

    @contextmanager
    def apply_best_to(
        self,
        *,
        agent: AbstractAgent[Any, Any],
        input_type: InputSpec[BaseModel] | None = None,
    ) -> Iterator[None]:
        """Apply the best candidate to an agent and optional signature.

        Args:
            agent: The agent to apply the best candidate to.
            input_type: Optional structured input specification to also apply the candidate to.

        Yields:
            None while the context is active.
        """
        with apply_candidate_to_agent_and_signature(
            self.best_candidate, agent=agent, input_type=input_type
        ):
            yield

    @property
    def graphviz_dag(self) -> str:
        """Return the Graphviz DOT string of the optimization DAG."""
        if self.raw_result is None:
            raise ValueError("Raw result is not available")
        pareto_front_programs = find_dominator_programs(self.raw_result.per_val_instance_best_candidates, self.raw_result.val_aggregate_scores)
        return dag_to_dot(self.raw_result.parents, pareto_front_programs, self.raw_result.best_idx, self.raw_result.val_aggregate_scores)


def optimize_agent_prompts(
    agent: AbstractAgent[Any, Any],
    trainset: Sequence[DataInstT],
    *,
    metric: Callable[[DataInstT, RolloutOutput[Any]], tuple[float, str | None]],
    valset: Sequence[DataInstT] | None = None,
    input_type: InputSpec[BaseModel] | None = None,
    seed_candidate: dict[str, str] | None = None,
    # Budget
    max_metric_calls: int | None = None,
    max_full_evals: int | None = None,
    auto: Literal["light", "medium", "heavy"] | None = None,
    # Reflection-based configuration
    reflection_model: str | None = None,
    candidate_selection_strategy: CandidateSelector | Literal['pareto', 'current_best', 'epsilon_greedy'] = "pareto",
    skip_perfect_score: bool = True,
    reflection_minibatch_size: int = 3,
    perfect_score: int = 1,
    # Component selection configuration
    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = "all",
    # Merge-based configuration
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    # Stopping
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None,
    # Caching configuration
    enable_cache: bool = False,
    cache_dir: str | None = None,
    cache_verbose: bool = False,
    # Logging
    logger: LoggerProtocol | None = None,
    run_dir: str | None = None,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    track_best_outputs: bool = False,
    display_progress_bar: bool = False,
    # Reproducibility
    seed: int = 0,
    raise_on_exception: bool = True,
    # Reflection sampler
    reflection_sampler: ReflectionSampler | None = None,
) -> GepaOptimizationResult:
    """Optimize agent (and optional signature) prompts using GEPA.

    This is the main entry point for prompt optimization. It takes a pydantic-ai
    agent and a dataset, and returns optimized prompts.

    Args:
        agent: The pydantic-ai agent to optimize.
        trainset: Training dataset (pydantic-evals Dataset or list of DataInst).
        metric: Function that computes (score, feedback) for each instance.
                The feedback (second element of tuple) is optional but recommended.
                If provided, it will be used to guide the optimization process.
        valset: Optional validation dataset. If not provided, trainset is used.
        input_type: Optional structured input specification whose instructions and
            field descriptions should be optimized alongside the agent's prompts.

        # Reflection-based configuration
        reflection_model: Model name to use for reflection (proposing new prompts).
        candidate_selection_strategy: Strategy for selecting candidates ('pareto' or 'current_best').
        skip_perfect_score: Whether to skip updating if perfect score achieved on minibatch.
        reflection_minibatch_size: Number of examples to use for reflection in each proposal.
        perfect_score: The perfect score value to achieve (integer).

        # Component selection configuration
        module_selector: Component selection strategy. Can be a ReflectionComponentSelector
                        instance or a string ('round_robin', 'all').

        # Merge-based configuration
        use_merge: Whether to use the merge strategy for combining candidates.
        max_merge_invocations: Maximum number of merge invocations to perform.
        merge_val_overlap_floor: Minimum number of validation examples to overlap between merge candidates.

        # Budget
        max_metric_calls: Maximum number of metric evaluations (budget).
        max_full_evals: Maximum number of full evaluations (budget).
        auto: Automatically set the budget based on the dataset size. Can be 'light', 'medium', or 'heavy'.

        # Caching configuration
        enable_cache: Whether to enable caching of metric results for resumable runs.
        cache_dir: Directory to store cache files. If None, uses '.gepa_cache' in current directory.
        cache_verbose: Whether to log cache hits and misses.

        # Logging
        logger: Logger instance for tracking progress.
        run_dir: Directory to save results to.
        use_wandb: Whether to use Weights and Biases for logging.
        wandb_api_key: API key for Weights and Biases.
        wandb_init_kwargs: Additional kwargs for wandb initialization.
        use_mlflow: Whether to use MLflow for logging.
        mlflow_tracking_uri: Tracking URI for MLflow.
        mlflow_experiment_name: Experiment name for MLflow.
        track_best_outputs: Whether to track best outputs on validation set.
        display_progress_bar: Whether to display a progress bar.

        # Reproducibility
        seed: Random seed for reproducibility.
        raise_on_exception: Whether to raise exceptions or continue on errors.

        # Reflection sampler
        reflection_sampler: Optional sampler for reflection records. If provided,
                               it will be called to sample records when needed. If None,
                               all reflection records are kept without sampling.

    Returns:
        GepaOptimizationResult with the best candidate and metadata.
    """    
    
    # Convert datasets if needed
    train_instances = list(trainset)

    if valset is not None:
        val_instances = list(valset)
    else:
        # If None we will use the trainset as the validation set
        val_instances = list(trainset)
        print("No valset provided; Using trainset as valset. This is useful as an inference-time scaling strategy where you want GEPA to find the best solutions for the provided tasks in the trainset, as it makes GEPA overfit prompts to the provided trainset. In order to ensure generalization and perform well on unseen tasks, please provide separate trainset and valset. Provide the smallest valset that is just large enough to match the downstream task distribution, while keeping trainset as large as possible.")

    # Extract seed candidate from agent and optional signature
    extracted_seed_candidate = _normalize_candidate(
        extract_seed_candidate_with_signature(
            agent=agent,
            input_type=input_type,
        )
    )
    if seed_candidate is None:
        seed_candidate = extracted_seed_candidate
    else:
        seed_candidate = _normalize_candidate(seed_candidate)
        if sorted(extracted_seed_candidate.keys()) != sorted(seed_candidate.keys()):
            raise ValueError(
                "Seed candidate keys do not match extracted seed candidate keys"
            )
            
    # Set budget
    if auto is not None:
        max_metric_calls = auto_budget(
            num_candidates=AUTO_RUN_SETTINGS[auto]["n"],
            valset_size=len(valset) if valset is not None else len(trainset),
        )
    
    elif max_full_evals is not None:
        max_metric_calls = max_full_evals * (len(trainset) + (len(valset) if valset is not None else 0))
        
    print(f"Running GEPA for approx {max_metric_calls} metric calls of the program. This amounts to {max_metric_calls / len(trainset) if valset is None else max_metric_calls / (len(trainset) + len(valset)):.2f} full evals on the {'train' if valset is None else 'train+val'} set.")

    # Create cache manager if caching is enabled
    cache_manager = None
    if enable_cache:
        cache_manager = CacheManager(
            cache_dir=cache_dir,
            enabled=True,
            verbose=cache_verbose,
        )

    # Create adapter
    adapter = PydanticAIGEPAAdapter(
        agent=agent,
        metric=metric,
        input_type=input_type,
        reflection_sampler=reflection_sampler,
        reflection_model=reflection_model,
        cache_manager=cache_manager,
    )
    # If no reflection model is provided, use the agent's model
    if not reflection_model:
        reflection_model = agent.model.model_name
        
    # Create a default language model for GEPA reflection
    reflection_lm = GEPALanguageModel(reflection_model)

    # Adjust module_selector based on number of components if needed
    # If only one component and module_selector is still default, use 'all'
    if module_selector == "round_robin" and len(seed_candidate) == 1:
        module_selector = "all"

    # Run optimization
    raw_result: GEPAResult[RolloutOutput[Any]] = gepa.api.optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=train_instances,
        valset=val_instances,
        # Budget
        max_metric_calls=max_metric_calls,
        # Reflection-based configuration
        reflection_lm=reflection_lm,
        candidate_selection_strategy=candidate_selection_strategy,
        reflection_minibatch_size=reflection_minibatch_size,
        perfect_score=perfect_score,
        skip_perfect_score=skip_perfect_score,
        # Component selection configuration
        module_selector=module_selector,
        # Merge-based configuration
        use_merge=use_merge,
        max_merge_invocations=max_merge_invocations,
        merge_val_overlap_floor=merge_val_overlap_floor,
        # Stopping
        stop_callbacks=stop_callbacks,
        # Logging
        logger=logger,
        run_dir=run_dir,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        track_best_outputs=track_best_outputs,
        display_progress_bar=display_progress_bar,
        # Reproducibility
        seed=seed,
        raise_on_exception=raise_on_exception,
    )

    # Extract results
    best_candidate = raw_result.best_candidate
    normalized_best_candidate = _normalize_candidate(best_candidate)
    normalized_seed_candidate = _normalize_candidate(seed_candidate)
    best_score = (
        raw_result.val_aggregate_scores[raw_result.best_idx]
        if raw_result.val_aggregate_scores
        else 0.0
    )

    # Get original score if available (assuming the first candidate is the seed)
    original_score = None
    if raw_result.candidates and len(raw_result.candidates) > 0:
        # Check if the first candidate is the seed candidate
        if (
            _normalize_candidate(raw_result.candidates[0])
            == normalized_seed_candidate
        ):
            original_score = raw_result.val_aggregate_scores[0]
        else:
            # Search through all candidates for the seed
            for i, candidate in enumerate(raw_result.candidates):
                if (
                    _normalize_candidate(candidate)
                    == normalized_seed_candidate
                ):
                    original_score = raw_result.val_aggregate_scores[i]
                    break

    result = GepaOptimizationResult(
        best_candidate=normalized_best_candidate,
        best_score=best_score,
        original_candidate=normalized_seed_candidate,
        original_score=original_score,
        num_iterations=raw_result.num_full_val_evals or len(raw_result.candidates),
        num_metric_calls=raw_result.total_metric_calls or 0,
        raw_result=raw_result,
        gepa_usage=adapter.gepa_usage,
    )

    # Log cache stats if caching was enabled
    if cache_manager and cache_verbose:
        stats = cache_manager.get_cache_stats()
        if logger:
            logger.log(f"Cache stats: {stats}")

    return result
