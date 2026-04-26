"""Pydantic data models for policy_index_v2.

This module defines all typed data models used throughout the indexing pipeline:
- Configuration (``IndexConfig``)
- Core domain objects (``PageContent``, ``IndexNode``, ``DocumentIndex``)
- LLM structured-output response models consumed by pydantic-ai agents

Example:
    >>> from src.policy_index_v2.models import IndexConfig, IndexNode
    >>> cfg = IndexConfig(model="gpt-4.1-mini")
    >>> node = IndexNode(
    ...     node_id="0001",
    ...     title="Introduction",
    ...     start_page=1,
    ...     end_page=3,
    ... )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class IndexConfig(BaseModel):
    """Configuration for the indexing pipeline.

    Attributes:
        model: OpenAI model identifier used for all LLM calls.
        toc_check_pages: Maximum number of leading pages to scan for a TOC.
        max_pages_per_node: A node exceeding this page count is recursively split.
        max_tokens_per_node: A node exceeding this token count is recursively split.
        add_node_ids: Whether to assign sequential node IDs.
        add_summaries: Whether to generate per-node summaries via LLM.
        add_descriptions: Whether to generate a one-sentence document description.
        add_text: Whether to attach the full extracted text to each node.
        max_concurrent_llm_calls: Semaphore limit for concurrent LLM requests.
    """

    model: str = Field(default="gpt-4.1-mini", description="OpenAI model name")
    toc_check_pages: int = Field(default=1, gt=0, description="Pages to scan for TOC")
    max_pages_per_node: int = Field(
        default=5, gt=0, description="Max pages per node before splitting"
    )
    max_tokens_per_node: int = Field(
        default=12_000, gt=0, description="Max tokens per node before splitting"
    )
    add_node_ids: bool = Field(default=True, description="Assign sequential node IDs")
    add_summaries: bool = Field(
        default=True, description="Generate per-node LLM summaries"
    )
    add_descriptions: bool = Field(
        default=True, description="Generate document description"
    )
    add_text: bool = Field(default=True, description="Attach extracted text to nodes")
    max_concurrent_llm_calls: int = Field(
        default=10, gt=0, description="Semaphore limit for concurrent LLM calls"
    )


# ---------------------------------------------------------------------------
# PDF page content
# ---------------------------------------------------------------------------


class PageContent(BaseModel):
    """Extracted content for a single PDF page.

    Attributes:
        page_number: 1-based page number within the PDF.
        text: Raw extracted text content.
        token_count: Token count according to the configured model's tokenizer.
    """

    page_number: int = Field(ge=1, description="1-based page number")
    text: str = Field(description="Extracted page text")
    token_count: int = Field(ge=0, description="Token count for this page")


# ---------------------------------------------------------------------------
# Tree / index nodes
# ---------------------------------------------------------------------------


class IndexNode(BaseModel):
    """A single node within the hierarchical document index.

    Nodes form a recursive tree via the ``children`` field.  Leaf nodes
    typically correspond to individual document sections; inner nodes to
    logical groupings (parts, chapters, etc.).

    Attributes:
        node_id: Zero-padded 4-digit unique identifier (e.g. ``"0001"``).
        title: Section title extracted from the document.
        start_page: 1-based inclusive start page.
        end_page: 1-based inclusive end page.
        text: Full extracted text for this node's page range (optional).
        summary: LLM-generated summary (optional).
        children: Nested child nodes forming the tree.
    """

    node_id: str = Field(default="", description="Zero-padded 4-digit ID")
    title: str = Field(description="Section title")
    start_page: int = Field(ge=1, description="Inclusive start page (1-based)")
    end_page: int = Field(ge=1, description="Inclusive end page (1-based)")
    text: str | None = Field(default=None, description="Full extracted text")
    summary: str | None = Field(default=None, description="LLM-generated summary")
    children: list[IndexNode] = Field(default_factory=list, description="Child nodes")

    @model_validator(mode="after")
    def _clamp_start_le_end(self) -> IndexNode:
        """Clamp end_page to at least start_page.

        LLM-generated page ranges occasionally produce inverted ranges
        (e.g. two adjacent sections mapped to the same page).  Rather than
        crashing, we silently clamp so downstream code always sees a valid
        range.  The root cause is addressed upstream via ``ModelRetry``
        validators on the TOC-generation agents.
        """
        if self.start_page > self.end_page:
            self.end_page = self.start_page
        return self


# ---------------------------------------------------------------------------
# Document-level index
# ---------------------------------------------------------------------------


class DocumentIndex(BaseModel):
    """Top-level result for a single indexed document.

    Attributes:
        doc_name: The filename (or derived name) of the indexed document.
        doc_description: A one-sentence LLM-generated description (optional).
        root_nodes: The top-level nodes of the hierarchical index tree.
    """

    doc_name: str = Field(description="Document file name")
    doc_description: str | None = Field(
        default=None, description="One-sentence document description"
    )
    root_nodes: list[IndexNode] = Field(
        default_factory=list, description="Top-level tree nodes"
    )


# ---------------------------------------------------------------------------
# LLM structured-output response models
# ---------------------------------------------------------------------------


class TocDetectionResult(BaseModel):
    """Response from the TOC detection agent.

    Attributes:
        reasoning: Chain-of-thought explanation.
        is_toc_page: Whether the page contains a table of contents.
    """

    reasoning: str = Field(description="Why the page is or is not a TOC page")
    is_toc_page: bool = Field(
        description="True if the page contains a table of contents"
    )


class TocEntry(BaseModel):
    """A single entry extracted from a table of contents.

    Attributes:
        structure: Hierarchical numbering string (e.g. ``"1.2.3"``).
        title: Section title text.
        page: Page number as printed in the TOC, if present.
    """

    structure: str = Field(description="Hierarchical numbering (e.g. '1.2.3')")
    title: str = Field(description="Section title")
    page: int | None = Field(
        default=None, description="Page number from TOC (if present)"
    )


class TocTransformResult(BaseModel):
    """Wrapper for the full table of contents transformation.

    Attributes:
        table_of_contents: List of extracted TOC entries.
    """

    table_of_contents: list[TocEntry] = Field(description="Extracted TOC entries")


class TitleAppearanceResult(BaseModel):
    """Response from the title-appearance checker agent.

    Attributes:
        thinking: Chain-of-thought explanation.
        answer: ``"yes"`` if the section title appears on the page, ``"no"`` otherwise.
    """

    thinking: str = Field(description="Reasoning about title appearance")
    answer: Literal["yes", "no"] = Field(
        description="Whether the title appears on the page"
    )


class TitleStartResult(BaseModel):
    """Response from the title-start checker agent.

    Attributes:
        thinking: Chain-of-thought explanation.
        start_begin: ``"yes"`` if the section starts at the beginning of the page.
    """

    thinking: str = Field(description="Reasoning about where the section starts")
    start_begin: Literal["yes", "no"] = Field(
        description="Whether the section starts at the beginning of the page"
    )


class NodeSummary(BaseModel):
    """Summary generated for a single document section.

    Attributes:
        summary: Concise description of the section's main points.
    """

    summary: str = Field(description="Concise summary of the section")


class DocDescription(BaseModel):
    """One-sentence description generated for an entire document.

    Attributes:
        description: A distinguishing one-sentence description.
    """

    description: str = Field(description="One-sentence document description")


class PageIndexEntry(BaseModel):
    """Maps a TOC entry to its physical page location.

    Attributes:
        structure: Hierarchical numbering of the section.
        title: Section title.
        physical_index: 1-based physical page number in the PDF.
    """

    structure: str | None = Field(default=None, description="Hierarchical numbering")
    title: str = Field(description="Section title")
    physical_index: int | None = Field(
        default=None, ge=1, description="1-based physical page index"
    )


class TocGeneratorEntry(BaseModel):
    """A single entry produced when generating a TOC from raw page text.

    Attributes:
        structure: Hierarchical numbering string.
        title: Section title extracted from the text.
        physical_index: Physical page tag value (e.g. extracted from ``<physical_index_X>``).
    """

    structure: str = Field(description="Hierarchical numbering (e.g. '1.2.3')")
    title: str = Field(description="Section title")
    physical_index: int | None = Field(
        default=None, ge=1, description="Physical page index"
    )


class SingleItemFixResult(BaseModel):
    """Response from the single-item TOC fixer agent.

    Attributes:
        thinking: Chain-of-thought explanation.
        physical_index: Corrected physical page index.
    """

    thinking: str = Field(description="Reasoning for the corrected page index")
    physical_index: int | None = Field(
        default=None, ge=1, description="Corrected physical page index"
    )


class PageIndexDetectionResult(BaseModel):
    """Response from the page-index detection agent.

    Attributes:
        thinking: Chain-of-thought explanation.
        page_index_given_in_toc: Whether page numbers are present in the TOC.
    """

    thinking: str = Field(description="Reasoning about page index presence")
    page_index_given_in_toc: bool = Field(
        description="True if the TOC contains page numbers"
    )


class CompletenessCheckResult(BaseModel):
    """Response from the completeness checker agent.

    Attributes:
        thinking: Chain-of-thought explanation.
        completed: Whether the content is complete.
    """

    thinking: str = Field(description="Reasoning about completeness")
    completed: bool = Field(description="True if the content is complete")
