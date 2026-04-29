"""Integration tests for the PolicyIndex public API.

These tests require:
1. ``data/HO3_sample.pdf`` to be present.
2. A valid ``CHATGPT_API_KEY`` (or ``OPENAI_API_KEY``) environment variable.

All tests are marked with ``@pytest.mark.integration`` so they can be excluded
in CI environments without API keys::

    pytest -m "not integration"

To run *only* integration tests::

    pytest -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.policy_index_v2 import DocumentIndex, IndexConfig, IndexNode, PolicyIndex
from src.policy_index_v2.tree import flatten_nodes

# All tests in this module are async and require API access
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config() -> IndexConfig:
    """Return a minimal config for integration tests.

    Uses defaults but turns off descriptions to speed things up slightly.
    """
    return IndexConfig(
        model="gpt-4.1-mini",
        add_descriptions=True,
        add_summaries=True,
        add_text=True,
        add_node_ids=True,
    )


# ---------------------------------------------------------------------------
# get_or_create
# ---------------------------------------------------------------------------


class TestGetOrCreate:
    """Integration tests for PolicyIndex.get_or_create."""

    async def test_creates_valid_document_index(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """get_or_create should produce a valid DocumentIndex."""
        pi = PolicyIndex(config)
        doc = await pi.get_or_create(ho3_pdf_path)

        assert isinstance(doc, DocumentIndex)
        assert doc.doc_name == "HO3_sample.pdf"
        assert len(doc.root_nodes) > 0

    async def test_nodes_have_ids(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """All nodes should have non-empty node_id strings."""
        pi = PolicyIndex(config)
        doc = await pi.get_or_create(ho3_pdf_path)

        all_nodes = flatten_nodes(doc.root_nodes)
        for node in all_nodes:
            assert node.node_id != ""
            assert len(node.node_id) == 4  # zero-padded

    async def test_nodes_have_text(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """With add_text=True, all nodes should have non-empty text."""
        pi = PolicyIndex(config)
        doc = await pi.get_or_create(ho3_pdf_path)

        all_nodes = flatten_nodes(doc.root_nodes)
        for node in all_nodes:
            assert node.text is not None
            assert len(node.text) > 0

    async def test_nodes_have_summaries(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """With add_summaries=True, all nodes should have summaries."""
        pi = PolicyIndex(config)
        doc = await pi.get_or_create(ho3_pdf_path)

        all_nodes = flatten_nodes(doc.root_nodes)
        for node in all_nodes:
            assert node.summary is not None
            assert len(node.summary) > 0

    async def test_doc_description_generated(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """With add_descriptions=True, the doc should have a description."""
        pi = PolicyIndex(config)
        doc = await pi.get_or_create(ho3_pdf_path)

        assert doc.doc_description is not None
        assert len(doc.doc_description) > 0

    async def test_page_ranges_valid(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """Every node's start_page should be <= end_page."""
        pi = PolicyIndex(config)
        doc = await pi.get_or_create(ho3_pdf_path)

        all_nodes = flatten_nodes(doc.root_nodes)
        for node in all_nodes:
            assert node.start_page <= node.end_page

    async def test_caching_returns_same_object(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """Calling get_or_create twice should return the cached result."""
        pi = PolicyIndex(config)
        doc1 = await pi.get_or_create(ho3_pdf_path)
        doc2 = await pi.get_or_create(ho3_pdf_path)
        # Should be the exact same object (cached)
        assert doc1 is doc2


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


class TestRenderingHelpers:
    """Tests for list_documents, tree, get_node, get_nodes."""

    async def test_list_documents(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """list_documents should return a markdown bullet list."""
        pi = PolicyIndex(config)
        await pi.get_or_create(ho3_pdf_path)

        md = pi.list_documents()
        assert "HO3_sample.pdf" in md
        assert md.startswith("- ")

    async def test_list_documents_empty(self, config: IndexConfig) -> None:
        """list_documents on empty store should return a message."""
        pi = PolicyIndex(config)
        md = pi.list_documents()
        assert "No documents" in md

    async def test_tree_output(self, ho3_pdf_path: Path, config: IndexConfig) -> None:
        """tree() should return readable markdown with node IDs and page ranges."""
        pi = PolicyIndex(config)
        await pi.get_or_create(ho3_pdf_path)

        md = pi.tree("HO3_sample.pdf")
        assert "HO3_sample.pdf" in md
        assert "[0000]" in md  # first node ID
        assert "pp." in md  # page range

    async def test_tree_all_documents(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """tree() with no args should render all indexed documents."""
        pi = PolicyIndex(config)
        await pi.get_or_create(ho3_pdf_path)

        md = pi.tree()  # no args = all
        assert "HO3_sample.pdf" in md

    async def test_tree_unknown_doc_raises(self, config: IndexConfig) -> None:
        """tree() for an unknown document should raise KeyError."""
        pi = PolicyIndex(config)
        with pytest.raises(KeyError, match="Document not found"):
            pi.tree("nonexistent.pdf")

    async def test_get_node(self, ho3_pdf_path: Path, config: IndexConfig) -> None:
        """get_node should find a node by its ID."""
        pi = PolicyIndex(config)
        await pi.get_or_create(ho3_pdf_path)

        node = pi.get_node("0000")
        assert isinstance(node, IndexNode)
        assert node.node_id == "0000"

    async def test_get_node_missing_raises(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """get_node for a non-existent ID should raise KeyError."""
        pi = PolicyIndex(config)
        await pi.get_or_create(ho3_pdf_path)

        with pytest.raises(KeyError, match="Node not found"):
            pi.get_node("9999")

    async def test_get_nodes_multiple(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        """get_nodes should return nodes in the requested order."""
        pi = PolicyIndex(config)
        await pi.get_or_create(ho3_pdf_path)

        nodes = pi.get_nodes("0000", "0001")
        assert len(nodes) == 2
        assert nodes[0].node_id == "0000"
        assert nodes[1].node_id == "0001"


# ---------------------------------------------------------------------------
# get / remove
# ---------------------------------------------------------------------------


class TestGetAndRemove:
    """Tests for the get() and remove() methods."""

    async def test_get_returns_none_when_empty(self, config: IndexConfig) -> None:
        pi = PolicyIndex(config)
        assert pi.get("anything.pdf") is None

    async def test_get_returns_cached(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        pi = PolicyIndex(config)
        await pi.get_or_create(ho3_pdf_path)
        doc = pi.get("HO3_sample.pdf")
        assert doc is not None
        assert doc.doc_name == "HO3_sample.pdf"

    async def test_remove_returns_true(
        self, ho3_pdf_path: Path, config: IndexConfig
    ) -> None:
        pi = PolicyIndex(config)
        await pi.get_or_create(ho3_pdf_path)
        assert pi.remove("HO3_sample.pdf") is True
        assert pi.get("HO3_sample.pdf") is None

    async def test_remove_returns_false_when_missing(self, config: IndexConfig) -> None:
        pi = PolicyIndex(config)
        assert pi.remove("nothing.pdf") is False
