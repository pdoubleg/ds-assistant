"""pydantic-ai Agent definitions for all LLM tasks in the indexing pipeline.

Each agent uses a typed ``output_type`` so that pydantic-ai handles JSON
parsing and validation automatically.  Where appropriate, ``@output_validator``
decorators raise :class:`ModelRetry` to self-correct malformed responses.

All agents are created via factory functions so callers can supply a custom
OpenAI model name.

Example:
    >>> from src.policy_index_v2.agents import create_toc_detector_agent
    >>> agent = create_toc_detector_agent("gpt-4.1-mini")
    >>> result = await agent.run("page text here...")
    >>> print(result.output.is_toc_page)
"""

from __future__ import annotations

import json

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel

from .models import (
    CompletenessCheckResult,
    DocDescription,
    NodeSummary,
    PageIndexDetectionResult,
    PageIndexEntry,
    SingleItemFixResult,
    TitleAppearanceResult,
    TitleStartResult,
    TocDetectionResult,
    TocGeneratorEntry,
    TocTransformResult,
)

# ---------------------------------------------------------------------------
# 1. TOC Detection Agent
# ---------------------------------------------------------------------------

_TOC_DETECTOR_INSTRUCTIONS = (
    "You detect whether a given page of text contains a Table of Contents. "
    "Abstract, summary, notation lists, figure lists, and table lists are NOT tables of contents."
)


def create_toc_detector_agent(model: str = "gpt-4.1-mini") -> Agent[None, TocDetectionResult]:
    """Create an agent that determines whether a single page is a TOC page.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, TocDetectionResult] = Agent(
        model=OpenAIChatModel(model),
        output_type=TocDetectionResult,
        instructions=_TOC_DETECTOR_INSTRUCTIONS,
        retries=3,
    )
    return agent


def toc_detector_prompt(page_text: str) -> str:
    """Build the user prompt for TOC detection.

    Args:
        page_text: Raw text extracted from a single PDF page.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Determine whether the following page text contains a table of contents.\n\n"
        f"Page text:\n{page_text}"
    )


# ---------------------------------------------------------------------------
# 2. Page-Index Detection Agent
# ---------------------------------------------------------------------------

_PAGE_INDEX_DETECTION_INSTRUCTIONS = (
    "You determine whether a table of contents contains explicit page numbers or indices."
)


def create_page_index_detector_agent(model: str = "gpt-4.1-mini") -> Agent[None, PageIndexDetectionResult]:
    """Create an agent that checks if a TOC contains page numbers.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, PageIndexDetectionResult] = Agent(
        model=OpenAIChatModel(model),
        output_type=PageIndexDetectionResult,
        instructions=_PAGE_INDEX_DETECTION_INSTRUCTIONS,
        retries=3,
    )
    return agent


def page_index_detection_prompt(toc_text: str) -> str:
    """Build the user prompt for page-index detection.

    Args:
        toc_text: The extracted table of contents text.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Determine whether the following table of contents contains page numbers or indices.\n\n"
        f"Table of contents:\n{toc_text}"
    )


# ---------------------------------------------------------------------------
# 3. TOC Transformation Agent
# ---------------------------------------------------------------------------

_TOC_TRANSFORM_INSTRUCTIONS = (
    "You are an expert at transforming raw table-of-contents text into structured JSON. "
    "The 'structure' field is a hierarchical numbering system (e.g., '1', '1.1', '1.2', '2'). "
    "The 'page' field should be an integer if the TOC contains page numbers, otherwise null. "
    "Transform the COMPLETE table of contents in one response."
)


def create_toc_transform_agent(model: str = "gpt-4.1-mini") -> Agent[None, TocTransformResult]:
    """Create an agent that transforms raw TOC text into structured entries.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, TocTransformResult] = Agent(
        model=OpenAIChatModel(model),
        output_type=TocTransformResult,
        instructions=_TOC_TRANSFORM_INSTRUCTIONS,
        retries=3,
    )

    @agent.output_validator
    def _check_non_empty(ctx: RunContext, result: TocTransformResult) -> TocTransformResult:
        """Ensure at least one entry was extracted."""
        if not result.table_of_contents:
            raise ModelRetry("The table of contents is empty. Please extract all entries.")
        return result

    return agent


def toc_transform_prompt(toc_text: str) -> str:
    """Build the user prompt for TOC transformation.

    Args:
        toc_text: Raw table of contents text.

    Returns:
        Formatted prompt string.
    """
    return f"Transform the following table of contents into structured JSON.\n\n{toc_text}"


# ---------------------------------------------------------------------------
# 4. Title Appearance Checker Agent
# ---------------------------------------------------------------------------

_TITLE_APPEARANCE_INSTRUCTIONS = (
    "You check whether a given section title appears or starts in the given page text. "
    "Use fuzzy matching and ignore any whitespace inconsistencies."
)


def create_title_checker_agent(model: str = "gpt-4.1-mini") -> Agent[None, TitleAppearanceResult]:
    """Create an agent that checks if a section title appears on a page.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, TitleAppearanceResult] = Agent(
        model=OpenAIChatModel(model),
        output_type=TitleAppearanceResult,
        instructions=_TITLE_APPEARANCE_INSTRUCTIONS,
        retries=3,
    )
    return agent


def title_appearance_prompt(title: str, page_text: str) -> str:
    """Build the user prompt for title-appearance checking.

    Args:
        title: The section title to look for.
        page_text: Text of the page to search.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Check if the following section title appears or starts in the given page text.\n\n"
        f"Section title: {title}\n\n"
        f"Page text:\n{page_text}"
    )


# ---------------------------------------------------------------------------
# 5. Title Start Checker Agent
# ---------------------------------------------------------------------------

_TITLE_START_INSTRUCTIONS = (
    "You check whether a given section starts at the BEGINNING of a page. "
    "If other content precedes the section title, the answer is 'no'. "
    "If the section title is the first content on the page, the answer is 'yes'. "
    "Use fuzzy matching and ignore whitespace inconsistencies."
)


def create_title_start_checker_agent(model: str = "gpt-4.1-mini") -> Agent[None, TitleStartResult]:
    """Create an agent that checks if a section starts at the beginning of a page.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, TitleStartResult] = Agent(
        model=OpenAIChatModel(model),
        output_type=TitleStartResult,
        instructions=_TITLE_START_INSTRUCTIONS,
        retries=3,
    )
    return agent


def title_start_prompt(title: str, page_text: str) -> str:
    """Build the user prompt for title-start checking.

    Args:
        title: The section title.
        page_text: Text of the page.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Check if the following section starts at the beginning of the page text.\n\n"
        f"Section title: {title}\n\n"
        f"Page text:\n{page_text}"
    )


# ---------------------------------------------------------------------------
# 6. Summary Agent
# ---------------------------------------------------------------------------

_SUMMARY_INSTRUCTIONS = (
    "You generate concise summaries of document sections. "
    "Focus on the main points covered in the text."
)


def create_summary_agent(model: str = "gpt-4.1-mini") -> Agent[None, NodeSummary]:
    """Create an agent that generates a concise summary for a section.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, NodeSummary] = Agent(
        model=OpenAIChatModel(model),
        output_type=NodeSummary,
        instructions=_SUMMARY_INSTRUCTIONS,
        retries=2,
    )
    return agent


def summary_prompt(section_text: str) -> str:
    """Build the user prompt for summary generation.

    Args:
        section_text: The full text of the document section.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Generate a concise description of the main points covered in this document section.\n\n"
        f"Section text:\n{section_text}"
    )


# ---------------------------------------------------------------------------
# 7. Document Description Agent
# ---------------------------------------------------------------------------

_DESCRIPTION_INSTRUCTIONS = (
    "You are an expert at generating distinguishing one-sentence descriptions for documents. "
    "Given a document's structure (titles and summaries), produce a single sentence that "
    "makes it easy to tell this document apart from others."
)


def create_description_agent(model: str = "gpt-4.1-mini") -> Agent[None, DocDescription]:
    """Create an agent that generates a one-sentence document description.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, DocDescription] = Agent(
        model=OpenAIChatModel(model),
        output_type=DocDescription,
        instructions=_DESCRIPTION_INSTRUCTIONS,
        retries=2,
    )
    return agent


def description_prompt(structure_summary: str) -> str:
    """Build the user prompt for document-description generation.

    Args:
        structure_summary: A text summary of the document structure
            (e.g. titles, node IDs, summaries).

    Returns:
        Formatted prompt string.
    """
    return (
        f"Generate a one-sentence description for the following document.\n\n"
        f"Document structure:\n{structure_summary}"
    )


# ---------------------------------------------------------------------------
# 8. TOC Index Extractor Agent
# ---------------------------------------------------------------------------

_TOC_INDEX_EXTRACTOR_INSTRUCTIONS = (
    "You are given a table of contents in JSON format and several pages of a document. "
    "Your job is to add the physical_index (the physical page number where each section starts) "
    "to the table of contents entries. "
    "The provided pages contain tags like <physical_index_X> to indicate the physical location of page X. "
    "Only add physical_index to sections that appear in the provided pages. "
    "If a section is not in the provided pages, leave physical_index as null."
)


def create_toc_index_extractor_agent(model: str = "gpt-4.1-mini") -> Agent[None, list[PageIndexEntry]]:
    """Create an agent that maps TOC entries to physical page indices.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, list[PageIndexEntry]] = Agent(
        model=OpenAIChatModel(model),
        output_type=list[PageIndexEntry],
        instructions=_TOC_INDEX_EXTRACTOR_INSTRUCTIONS,
        retries=3,
    )

    @agent.output_validator
    def _check_monotonic_indices(ctx: RunContext, result: list[PageIndexEntry]) -> list[PageIndexEntry]:
        """Ensure physical_index values are monotonically non-decreasing."""
        prev_idx: int | None = None
        for entry in result:
            if entry.physical_index is not None:
                if prev_idx is not None and entry.physical_index < prev_idx:
                    raise ModelRetry(
                        f"physical_index values must be in non-decreasing order. "
                        f"Entry '{entry.title}' has physical_index={entry.physical_index} "
                        f"but the previous entry had physical_index={prev_idx}. "
                        f"Please correct the ordering."
                    )
                prev_idx = entry.physical_index
        return result

    return agent


def toc_index_extractor_prompt(toc_json: list[dict[str, object]], page_content: str) -> str:
    """Build the user prompt for TOC index extraction.

    Args:
        toc_json: The TOC entries (without physical indices).
        page_content: Tagged page text (with ``<physical_index_X>`` markers).

    Returns:
        Formatted prompt string.
    """
    return (
        f"Map the following TOC entries to physical page indices using the document pages.\n\n"
        f"Table of contents:\n{json.dumps(toc_json, indent=2)}\n\n"
        f"Document pages:\n{page_content}"
    )


# ---------------------------------------------------------------------------
# 9. TOC Generator (initial) Agent
# ---------------------------------------------------------------------------

_TOC_GENERATOR_INIT_INSTRUCTIONS = (
    "You are an expert at extracting hierarchical tree structures from document text. "
    "Generate a table of contents structure from the given document pages. "
    "The 'structure' field is the hierarchical numbering (e.g. '1', '1.1', '1.2'). "
    "For the title, extract the original title from the text, only fixing space inconsistencies. "
    "The provided text contains tags like <physical_index_X> to indicate page locations. "
    "Extract the physical_index as an integer from the tags."
)


def create_toc_generator_init_agent(model: str = "gpt-4.1-mini") -> Agent[None, list[TocGeneratorEntry]]:
    """Create an agent that generates an initial TOC from page text.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, list[TocGeneratorEntry]] = Agent(
        model=OpenAIChatModel(model),
        output_type=list[TocGeneratorEntry],
        instructions=_TOC_GENERATOR_INIT_INSTRUCTIONS,
        retries=3,
    )

    @agent.output_validator
    def _check_non_empty(ctx: RunContext, result: list[TocGeneratorEntry]) -> list[TocGeneratorEntry]:
        """Ensure at least one entry was generated."""
        if not result:
            raise ModelRetry("No TOC entries were generated. Please extract the document structure.")
        return result

    @agent.output_validator
    def _check_monotonic_indices(ctx: RunContext, result: list[TocGeneratorEntry]) -> list[TocGeneratorEntry]:
        """Ensure physical_index values are monotonically non-decreasing.

        When the LLM produces out-of-order page indices, we ask it to
        self-correct via ModelRetry rather than letting the error propagate
        into tree construction.
        """
        prev_idx: int | None = None
        for entry in result:
            if entry.physical_index is not None:
                if prev_idx is not None and entry.physical_index < prev_idx:
                    raise ModelRetry(
                        f"physical_index values must be in non-decreasing order. "
                        f"Entry '{entry.title}' has physical_index={entry.physical_index} "
                        f"but the previous entry had physical_index={prev_idx}. "
                        f"Please correct the ordering."
                    )
                prev_idx = entry.physical_index
        return result

    return agent


def toc_generator_init_prompt(page_text: str) -> str:
    """Build the user prompt for initial TOC generation.

    Args:
        page_text: Tagged page text.

    Returns:
        Formatted prompt string.
    """
    return f"Generate the hierarchical tree structure for the following document pages.\n\n{page_text}"


# ---------------------------------------------------------------------------
# 10. TOC Generator (continue) Agent
# ---------------------------------------------------------------------------

_TOC_GENERATOR_CONTINUE_INSTRUCTIONS = (
    "You are an expert at extracting hierarchical tree structures. "
    "You are given a tree structure from the previous part and text from the current part. "
    "Continue the tree structure to include the current part. "
    "The 'structure' field is the hierarchical numbering (e.g. '1', '1.1', '1.2'). "
    "For the title, extract the original title from the text, only fixing space inconsistencies. "
    "The provided text contains tags like <physical_index_X> to indicate page locations. "
    "Extract the physical_index as an integer from the tags. "
    "Return ONLY the additional entries for the current part."
)


def create_toc_generator_continue_agent(model: str = "gpt-4.1-mini") -> Agent[None, list[TocGeneratorEntry]]:
    """Create an agent that continues TOC generation from additional pages.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, list[TocGeneratorEntry]] = Agent(
        model=OpenAIChatModel(model),
        output_type=list[TocGeneratorEntry],
        instructions=_TOC_GENERATOR_CONTINUE_INSTRUCTIONS,
        retries=3,
    )

    @agent.output_validator
    def _check_monotonic_indices(ctx: RunContext, result: list[TocGeneratorEntry]) -> list[TocGeneratorEntry]:
        """Ensure physical_index values are monotonically non-decreasing."""
        prev_idx: int | None = None
        for entry in result:
            if entry.physical_index is not None:
                if prev_idx is not None and entry.physical_index < prev_idx:
                    raise ModelRetry(
                        f"physical_index values must be in non-decreasing order. "
                        f"Entry '{entry.title}' has physical_index={entry.physical_index} "
                        f"but the previous entry had physical_index={prev_idx}. "
                        f"Please correct the ordering."
                    )
                prev_idx = entry.physical_index
        return result

    return agent


def toc_generator_continue_prompt(page_text: str, previous_toc: list[dict[str, object]]) -> str:
    """Build the user prompt for continuing TOC generation.

    Args:
        page_text: Tagged page text for the current chunk.
        previous_toc: Previously generated TOC entries (as dicts).

    Returns:
        Formatted prompt string.
    """
    return (
        f"Continue the tree structure for the following document pages.\n\n"
        f"Current pages:\n{page_text}\n\n"
        f"Previous structure:\n{json.dumps(previous_toc, indent=2)}"
    )


# ---------------------------------------------------------------------------
# 11. Single Item Fixer Agent
# ---------------------------------------------------------------------------

_SINGLE_ITEM_FIXER_INSTRUCTIONS = (
    "You are given a section title and several pages of a document. "
    "Your job is to find the physical page index where the section starts. "
    "The provided pages contain tags like <physical_index_X> to indicate page locations. "
    "Return the physical_index as an integer."
)


def create_single_item_fixer_agent(model: str = "gpt-4.1-mini") -> Agent[None, SingleItemFixResult]:
    """Create an agent that fixes the page index for a single TOC entry.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, SingleItemFixResult] = Agent(
        model=OpenAIChatModel(model),
        output_type=SingleItemFixResult,
        instructions=_SINGLE_ITEM_FIXER_INSTRUCTIONS,
        retries=3,
    )
    return agent


def single_item_fixer_prompt(section_title: str, page_content: str) -> str:
    """Build the user prompt for single-item fixing.

    Args:
        section_title: Title of the section to locate.
        page_content: Tagged page text to search.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Find the physical page index where the following section starts.\n\n"
        f"Section title: {section_title}\n\n"
        f"Document pages:\n{page_content}"
    )


# ---------------------------------------------------------------------------
# 12. Completeness Checker Agent
# ---------------------------------------------------------------------------

_COMPLETENESS_INSTRUCTIONS = (
    "You check whether a transformed table of contents is complete "
    "compared to the raw source material."
)


def create_completeness_checker_agent(model: str = "gpt-4.1-mini") -> Agent[None, CompletenessCheckResult]:
    """Create an agent that checks TOC transformation completeness.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, CompletenessCheckResult] = Agent(
        model=OpenAIChatModel(model),
        output_type=CompletenessCheckResult,
        instructions=_COMPLETENESS_INSTRUCTIONS,
        retries=2,
    )
    return agent


def completeness_check_prompt(raw_toc: str, transformed_toc: str) -> str:
    """Build the user prompt for completeness checking.

    Args:
        raw_toc: The original raw TOC text.
        transformed_toc: The transformed/cleaned TOC representation.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Check if the following transformed TOC is complete compared to the raw source.\n\n"
        f"Raw TOC:\n{raw_toc}\n\n"
        f"Transformed TOC:\n{transformed_toc}"
    )


# ---------------------------------------------------------------------------
# 13. Page Number Adder Agent (for TOC without page numbers)
# ---------------------------------------------------------------------------

_PAGE_NUMBER_ADDER_INSTRUCTIONS = (
    "You are given a JSON structure of a document and a partial part of the document. "
    "Your task is to check if each title in the structure starts in the given document text. "
    "The text contains tags like <physical_index_X> to indicate page locations. "
    "If the section starts in the text, set physical_index to the integer page number. "
    "If not, leave physical_index as null. "
    "Do not change previous results that already have a physical_index."
)


def create_page_number_adder_agent(model: str = "gpt-4.1-mini") -> Agent[None, list[PageIndexEntry]]:
    """Create an agent that adds page numbers to TOC entries without them.

    Args:
        model: OpenAI model identifier.

    Returns:
        A configured pydantic-ai ``Agent``.
    """
    agent: Agent[None, list[PageIndexEntry]] = Agent(
        model=OpenAIChatModel(model),
        output_type=list[PageIndexEntry],
        instructions=_PAGE_NUMBER_ADDER_INSTRUCTIONS,
        retries=3,
    )

    @agent.output_validator
    def _check_monotonic_indices(ctx: RunContext, result: list[PageIndexEntry]) -> list[PageIndexEntry]:
        """Ensure physical_index values are monotonically non-decreasing."""
        prev_idx: int | None = None
        for entry in result:
            if entry.physical_index is not None:
                if prev_idx is not None and entry.physical_index < prev_idx:
                    raise ModelRetry(
                        f"physical_index values must be in non-decreasing order. "
                        f"Entry '{entry.title}' has physical_index={entry.physical_index} "
                        f"but the previous entry had physical_index={prev_idx}. "
                        f"Please correct the ordering."
                    )
                prev_idx = entry.physical_index
        return result

    return agent


def page_number_adder_prompt(page_text: str, structure_json: list[dict[str, object]]) -> str:
    """Build the user prompt for page-number addition.

    Args:
        page_text: Tagged partial document text.
        structure_json: Current state of the TOC structure.

    Returns:
        Formatted prompt string.
    """
    return (
        f"Add physical page indices to the following TOC structure using the document text.\n\n"
        f"Document text:\n{page_text}\n\n"
        f"Current structure:\n{json.dumps(structure_json, indent=2)}"
    )
