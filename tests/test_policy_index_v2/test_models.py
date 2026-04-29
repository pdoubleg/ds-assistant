"""Unit tests for policy_index_v2 Pydantic models.

Tests cover:
- Field defaults and validation rules
- Serialization round-trips
- Model validators (e.g. start_page <= end_page)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.policy_index_v2.models import (
    DocumentIndex,
    IndexConfig,
    IndexNode,
    NodeSummary,
    PageContent,
    TitleAppearanceResult,
    TocDetectionResult,
    TocEntry,
)


# ---------------------------------------------------------------------------
# IndexConfig
# ---------------------------------------------------------------------------


class TestIndexConfig:
    """Tests for the IndexConfig model."""

    def test_defaults(self) -> None:
        """Default values should match the v1 config.yaml defaults."""
        cfg = IndexConfig()
        assert cfg.model == "gpt-4.1-mini"
        assert cfg.toc_check_pages == 1
        assert cfg.max_pages_per_node == 5
        assert cfg.max_tokens_per_node == 12_000
        assert cfg.add_node_ids is True
        assert cfg.add_summaries is True
        assert cfg.add_descriptions is True
        assert cfg.add_text is True
        assert cfg.max_concurrent_llm_calls == 10

    def test_custom_values(self) -> None:
        """Custom values should override defaults."""
        cfg = IndexConfig(model="gpt-4o", toc_check_pages=3, add_summaries=False)
        assert cfg.model == "gpt-4o"
        assert cfg.toc_check_pages == 3
        assert cfg.add_summaries is False

    def test_invalid_toc_check_pages(self) -> None:
        """toc_check_pages must be > 0."""
        with pytest.raises(ValidationError):
            IndexConfig(toc_check_pages=0)

    def test_invalid_max_pages(self) -> None:
        """max_pages_per_node must be > 0."""
        with pytest.raises(ValidationError):
            IndexConfig(max_pages_per_node=-1)

    def test_serialization_roundtrip(self) -> None:
        """Config should serialize and deserialize identically."""
        cfg = IndexConfig(model="gpt-4o", toc_check_pages=5)
        data = cfg.model_dump()
        restored = IndexConfig(**data)
        assert restored == cfg


# ---------------------------------------------------------------------------
# PageContent
# ---------------------------------------------------------------------------


class TestPageContent:
    """Tests for the PageContent model."""

    def test_valid_page(self) -> None:
        """A valid page should be created without errors."""
        page = PageContent(page_number=1, text="Hello world", token_count=2)
        assert page.page_number == 1
        assert page.text == "Hello world"
        assert page.token_count == 2

    def test_page_number_must_be_positive(self) -> None:
        """page_number must be >= 1."""
        with pytest.raises(ValidationError):
            PageContent(page_number=0, text="", token_count=0)

    def test_token_count_cannot_be_negative(self) -> None:
        """token_count must be >= 0."""
        with pytest.raises(ValidationError):
            PageContent(page_number=1, text="x", token_count=-1)


# ---------------------------------------------------------------------------
# IndexNode
# ---------------------------------------------------------------------------


class TestIndexNode:
    """Tests for the IndexNode model."""

    def test_minimal_node(self) -> None:
        """A node with only required fields should work."""
        node = IndexNode(title="Section A", start_page=1, end_page=5)
        assert node.title == "Section A"
        assert node.start_page == 1
        assert node.end_page == 5
        assert node.node_id == ""
        assert node.text is None
        assert node.summary is None
        assert node.children == []

    def test_node_with_children(self) -> None:
        """Nodes should support recursive children."""
        child = IndexNode(title="Subsection", start_page=2, end_page=3)
        parent = IndexNode(title="Section", start_page=1, end_page=5, children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].title == "Subsection"

    def test_start_greater_than_end_clamped(self) -> None:
        """start_page > end_page should be silently clamped."""
        node = IndexNode(title="Clamped", start_page=10, end_page=5)
        # The model_validator clamps end_page up to start_page
        assert node.start_page == 10
        assert node.end_page == 10

    def test_equal_start_end(self) -> None:
        """start_page == end_page is valid (single-page section)."""
        node = IndexNode(title="Single", start_page=3, end_page=3)
        assert node.start_page == node.end_page

    def test_serialization_roundtrip(self) -> None:
        """Node should survive a model_dump/model_validate cycle."""
        node = IndexNode(
            node_id="0042",
            title="Coverage A",
            start_page=4,
            end_page=8,
            summary="Covers dwelling",
            children=[
                IndexNode(title="Sub A", start_page=4, end_page=6),
            ],
        )
        data = node.model_dump()
        restored = IndexNode.model_validate(data)
        assert restored == node


# ---------------------------------------------------------------------------
# DocumentIndex
# ---------------------------------------------------------------------------


class TestDocumentIndex:
    """Tests for the DocumentIndex model."""

    def test_minimal_document(self) -> None:
        """A document with just a name and no nodes."""
        doc = DocumentIndex(doc_name="test.pdf")
        assert doc.doc_name == "test.pdf"
        assert doc.doc_description is None
        assert doc.root_nodes == []

    def test_document_with_nodes(self) -> None:
        """A document with nodes should serialize correctly."""
        node = IndexNode(title="Intro", start_page=1, end_page=3)
        doc = DocumentIndex(
            doc_name="policy.pdf",
            doc_description="A homeowners policy",
            root_nodes=[node],
        )
        assert len(doc.root_nodes) == 1
        assert doc.doc_description == "A homeowners policy"


# ---------------------------------------------------------------------------
# LLM response models
# ---------------------------------------------------------------------------


class TestLLMResponseModels:
    """Tests for LLM structured-output models."""

    def test_toc_detection_result(self) -> None:
        result = TocDetectionResult(
            reasoning="It has numbered sections", is_toc_page=True
        )
        assert result.is_toc_page is True

    def test_toc_entry_with_page(self) -> None:
        entry = TocEntry(structure="1.1", title="Introduction", page=3)
        assert entry.page == 3

    def test_toc_entry_without_page(self) -> None:
        entry = TocEntry(structure="2", title="Coverage", page=None)
        assert entry.page is None

    def test_title_appearance_result_valid_values(self) -> None:
        """answer must be 'yes' or 'no'."""
        result = TitleAppearanceResult(thinking="Found it", answer="yes")
        assert result.answer == "yes"

    def test_title_appearance_result_invalid_value(self) -> None:
        """answer must be exactly 'yes' or 'no'."""
        with pytest.raises(ValidationError):
            TitleAppearanceResult(thinking="Maybe", answer="maybe")

    def test_node_summary(self) -> None:
        summary = NodeSummary(summary="Covers dwelling protection")
        assert summary.summary == "Covers dwelling protection"
