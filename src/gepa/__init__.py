"""GEPA optimization integration for pydantic-ai."""

from __future__ import annotations

from .adapter import PydanticAIGEPAAdapter, ReflectionSampler
from .cache import CacheManager, create_cached_metric
from .data_utils import (
    create_dataset_from_dicts,
    dataframe_to_dataset,
    json_to_dataset,
    split_dataset,
)
from .runner import GepaOptimizationResult, optimize_agent_prompts
from .scaffold import GepaConfig, run_optimization_pipeline
from .signature import (
    BoundInputSpec,
    InputSpec,
    SignatureSuffix,
    apply_candidate_to_input_model,
    build_input_spec,
    generate_system_instructions,
    generate_user_content,
    get_gepa_components,
)
from .signature_agent import SignatureAgent
from .types import DataInst, RolloutOutput, Trajectory, OutputT

__all__ = [
    # Core optimization
    "optimize_agent_prompts",
    "GepaOptimizationResult",
    # Scaffolding
    "GepaConfig",
    "run_optimization_pipeline",
    # Data utilities
    "dataframe_to_dataset",
    "json_to_dataset",
    "create_dataset_from_dicts",
    "split_dataset",
    # Templates
    # Advanced
    "PydanticAIGEPAAdapter",
    "ReflectionSampler",
    "CacheManager",
    "create_cached_metric",
    # Types
    "DataInst",
    "Trajectory",
    "RolloutOutput",
    "OutputT",
    # Signature utilities
    "BoundInputSpec",
    "InputSpec",
    "generate_system_instructions",
    "generate_user_content",
    "get_gepa_components",
    "apply_candidate_to_input_model",
    "build_input_spec",
    "SignatureSuffix",
    "SignatureAgent",
]

__version__ = "0.1.0"
