from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def get_openai_model(model_name: str) -> Model:
    """Get an OpenAI model from a model name."""
    
    client = AsyncOpenAI()
    model = OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client))
    return model


class GEPALanguageModel:
    """Simple LanguageModel wrapper using a pydantic-ai Agent returning text."""

    def __init__(self, model: str | None):
        self._agent = Agent(get_openai_model(model), output_type=str)

    def __call__(self, prompt: str) -> str:
        result = self._agent.run_sync(prompt)
        return result.output
    
    