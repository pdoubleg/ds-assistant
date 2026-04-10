"""Automatic conversation summarization for pydantic-ai agents.

This package provides history processors for automatic conversation summarization,
helping manage context window limits in long-running agent conversations.

Example:
    ```python
    from pydantic_ai import Agent
    from .processor import create_summarization_processor

    # Create a processor that triggers at 100k tokens and keeps 20 messages
    processor = create_summarization_processor(
        trigger=("tokens", 100000),
        keep=("messages", 20),
    )

    agent = Agent(
        "openai:gpt-4.1",
        history_processors=[processor],
    )

    # The processor will automatically summarize older messages
    # when the conversation grows too long
    result = await agent.run("Hello!")
    ```
"""

from .cutoff import async_count_tokens
from .processor import (
    DEFAULT_SUMMARY_PROMPT,
    SummarizationProcessor,
    count_tokens_approximately,
    count_tokens_tiktoken,
    create_summarization_processor,
    format_messages_for_summary,
)
from .sliding_window import (
    SlidingWindowProcessor,
    create_sliding_window_processor,
)
from .types import (
    ContextFraction,
    ContextMessages,
    ContextSize,
    ContextTokens,
    ModelType,
    TokenCounter,
)


__all__ = [
    # Main exports - Summarization
    "SummarizationProcessor",
    "create_summarization_processor",
    # Main exports - Sliding Window
    "SlidingWindowProcessor",
    "create_sliding_window_processor",
    # Utilities
    "count_tokens_approximately",
    "count_tokens_tiktoken",
    "format_messages_for_summary",
    # Types
    "ContextSize",
    "ContextFraction",
    "ContextTokens",
    "ContextMessages",
    "ModelType",
    "TokenCounter",
    # Constants
    "DEFAULT_SUMMARY_PROMPT",
    # Utilities
    "async_count_tokens",
]
