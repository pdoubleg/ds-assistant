"""
Model Configuration - Centralized LLM model settings for pydantic-ai agents.

Uses OpenAI as the default provider. Model names follow pydantic-ai's
model string format (e.g., 'openai:gpt-4o', 'openai:gpt-4o-mini').
Set OPENAI_API_KEY in the environment for authentication.

Example usage:
    >>> from model_config import get_agent_model, get_orchestrator_model
    >>> agent = Agent(model=get_agent_model(), ...)
    >>> orchestrator = Agent(model=get_orchestrator_model(), ...)
"""

import os
from dotenv import load_dotenv

from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings


load_dotenv()

# Higher-capability model for the main AG-UI conversational agent
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "openai:gpt-5.4")

# Faster, cheaper model for orchestration sub-agents (analysis, component generation)
ORCHESTRATOR_MODEL: str = os.getenv("ORCHESTRATOR_MODEL", "openai:gpt-5.4-nano")


def get_agent_model() -> str:
    """Get the model string for the main AG-UI conversational agent.

    Returns:
        pydantic-ai model string (e.g., 'openai:gpt-5-mini')
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="concise",
        parallel_tool_calls=False,
    )
    model = OpenAIResponsesModel(
        model_name="gpt-5.4",
        settings=settings,
    )
    return model


def get_orchestrator_model() -> str:
    """Get the model string for orchestration tasks (analysis, form generation).

    Returns:
        pydantic-ai model string (e.g., 'openai:gpt-5-mini')
    """
    return ORCHESTRATOR_MODEL
