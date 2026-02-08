"""Unit tests for policy_index_v2 tree manipulation utilities.

All tests use hand-crafted IndexNode fixtures -- no LLM calls required.
"""

from __future__ import annotations

from src.policy_index_v2.models import IndexNode
from src.policy_index_v2.tree import (
    assign_node_ids,
    attach_text_to_nodes,
    build_tree_from_flat,
    find_node,
    flatten_nodes,
    post_process_to_tree,
    propagate_page_ranges,
    render_tree_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tree() -> list[IndexNode]:
    """Build a small sample tree for testing.

    Structure::

        Part I (pages 1-10)
          Ch 1 (pages 1-5)
            Sec 1.1 (pages 1-3)
          Ch 2 (pages 6-10)
        Part II (pages 11-15)
    """
    sec_1_1 = IndexNode(title="Sec 1.1", start_page=1, end_page=3)
    ch_1 = IndexNode(title="Ch 1", start_page=1, end_page=5, children=[sec_1_1])
    ch_2 = IndexNode(title="Ch 2", start_page=6, end_page=10)
    part_1 = IndexNode(title="Part I", start_page=1, end_page=10, children=[ch_1, ch_2])
    part_2 = IndexNode(title="Part II", start_page=11, end_page=15)
    return [part_1, part_2]


# ---------------------------------------------------------------------------
# flatten_nodes
# ---------------------------------------------------------------------------


class TestFlattenNodes:
    """Tests for flatten_nodes."""

    def test_empty_list(self) -> None:
        assert flatten_nodes([]) == []

    def test_single_leaf(self) -> None:
        node = IndexNode(title="Leaf", start_page=1, end_page=1)
        result = flatten_nodes([node])
        assert len(result) == 1
        assert result[0].title == "Leaf"

    def test_nested_tree(self) -> None:
        """All 5 nodes in the sample tree should appear in depth-first order."""
        tree = _make_tree()
        flat = flatten_nodes(tree)
        titles = [n.title for n in flat]
        assert titles == ["Part I", "Ch 1", "Sec 1.1", "Ch 2", "Part II"]

    def test_children_preserved(self) -> None:
        """flatten_nodes should NOT strip children from the original nodes."""
        tree = _make_tree()
        flat = flatten_nodes(tree)
        # Part I (index 0) should still have its children
        assert len(flat[0].children) == 2


# ---------------------------------------------------------------------------
# find_node
# ---------------------------------------------------------------------------


class TestFindNode:
    """Tests for find_node."""

    def test_find_root(self) -> None:
        tree = _make_tree()
        assign_node_ids(tree)
        node = find_node(tree, "0000")
        assert node is not None
        assert node.title == "Part I"

    def test_find_nested(self) -> None:
        tree = _make_tree()
        assign_node_ids(tree)
        node = find_node(tree, "0002")  # Sec 1.1
        assert node is not None
        assert node.title == "Sec 1.1"

    def test_find_missing(self) -> None:
        tree = _make_tree()
        assign_node_ids(tree)
        assert find_node(tree, "9999") is None

    def test_find_in_empty_tree(self) -> None:
        assert find_node([], "0000") is None


# ---------------------------------------------------------------------------
# assign_node_ids
# ---------------------------------------------------------------------------


class TestAssignNodeIds:
    """Tests for assign_node_ids."""

    def test_sequential_ids(self) -> None:
        tree = _make_tree()
        next_id = assign_node_ids(tree)
        flat = flatten_nodes(tree)
        ids = [n.node_id for n in flat]
        assert ids == ["0000", "0001", "0002", "0003", "0004"]
        assert next_id == 5

    def test_zero_padded(self) -> None:
        tree = _make_tree()
        assign_node_ids(tree)
        for node in flatten_nodes(tree):
            assert len(node.node_id) == 4

    def test_custom_start(self) -> None:
        tree = _make_tree()
        assign_node_ids(tree, start=10)
        flat = flatten_nodes(tree)
        assert flat[0].node_id == "0010"


# ---------------------------------------------------------------------------
# build_tree_from_flat
# ---------------------------------------------------------------------------


class TestBuildTreeFromFlat:
    """Tests for build_tree_from_flat."""

    def test_simple_tree(self) -> None:
        items = [
            {"structure": "1", "title": "Part I", "start_index": 1, "end_index": 10},
            {"structure": "1.1", "title": "Ch 1", "start_index": 1, "end_index": 5},
            {"structure": "1.2", "title": "Ch 2", "start_index": 6, "end_index": 10},
            {"structure": "2", "title": "Part II", "start_index": 11, "end_index": 15},
        ]
        tree = build_tree_from_flat(items)
        assert len(tree) == 2
        assert tree[0].title == "Part I"
        assert len(tree[0].children) == 2
        assert tree[0].children[0].title == "Ch 1"
        assert tree[1].title == "Part II"

    def test_flat_list_no_hierarchy(self) -> None:
        """Items without '.' in structure should all be roots."""
        items = [
            {"structure": "1", "title": "A", "start_index": 1, "end_index": 5},
            {"structure": "2", "title": "B", "start_index": 6, "end_index": 10},
        ]
        tree = build_tree_from_flat(items)
        assert len(tree) == 2

    def test_empty_list(self) -> None:
        assert build_tree_from_flat([]) == []


# ---------------------------------------------------------------------------
# render_tree_markdown
# ---------------------------------------------------------------------------


class TestRenderTreeMarkdown:
    """Tests for render_tree_markdown."""

    def test_basic_rendering(self) -> None:
        tree = _make_tree()
        assign_node_ids(tree)
        md = render_tree_markdown(tree)
        assert "Part I" in md
        assert "Sec 1.1" in md
        # Check indentation increases for children
        lines = md.split("\n")
        # The child line should have more leading spaces
        part_i_line = next(line for line in lines if "Part I" in line)
        sec_line = next(line for line in lines if "Sec 1.1" in line)
        assert len(sec_line) - len(sec_line.lstrip()) > len(part_i_line) - len(part_i_line.lstrip())

    def test_node_id_in_output(self) -> None:
        tree = _make_tree()
        assign_node_ids(tree)
        md = render_tree_markdown(tree)
        assert "[0000]" in md
        assert "[0004]" in md

    def test_page_range_in_output(self) -> None:
        tree = _make_tree()
        md = render_tree_markdown(tree)
        assert "pp. 1-10" in md

    def test_empty_tree(self) -> None:
        assert render_tree_markdown([]) == ""


# ---------------------------------------------------------------------------
# post_process_to_tree
# ---------------------------------------------------------------------------


class TestPostProcessToTree:
    """Tests for post_process_to_tree."""

    def test_basic_conversion(self) -> None:
        flat = [
            {"structure": "1", "title": "A", "physical_index": 1},
            {"structure": "2", "title": "B", "physical_index": 5},
        ]
        tree = post_process_to_tree(flat, total_pages=10)
        assert len(tree) == 2
        assert tree[0].title == "A"
        assert tree[0].start_page == 1
        assert tree[1].title == "B"
        assert tree[1].end_page == 10

    def test_appear_start_affects_end_page(self) -> None:
        """If the next item 'appear_start' is 'yes', the current item's
        end_page should be next_physical_index - 1."""
        flat = [
            {"structure": "1", "title": "A", "physical_index": 1},
            {"structure": "2", "title": "B", "physical_index": 5, "appear_start": "yes"},
        ]
        tree = post_process_to_tree(flat, total_pages=10)
        assert tree[0].end_page == 4  # 5 - 1

    def test_inverted_range_clamped(self) -> None:
        """When appear_start='yes' and two items share a page, the range
        should be clamped so start_page <= end_page (not crash)."""
        flat = [
            {"structure": "1", "title": "A", "physical_index": 5},
            {"structure": "2", "title": "B", "physical_index": 5, "appear_start": "yes"},
        ]
        tree = post_process_to_tree(flat, total_pages=10)
        # 5 - 1 = 4 would be < 5, so the clamp sets end = start = 5
        assert tree[0].start_page == 5
        assert tree[0].end_page == 5

    def test_empty_list(self) -> None:
        assert post_process_to_tree([], total_pages=10) == []


# ---------------------------------------------------------------------------
# propagate_page_ranges
# ---------------------------------------------------------------------------


class TestPropagatePageRanges:
    """Tests for propagate_page_ranges."""

    def test_parent_expanded_to_cover_children(self) -> None:
        """Parent end_page should expand to cover its children."""
        child_a = IndexNode(title="A", start_page=8, end_page=10)
        child_b = IndexNode(title="B", start_page=10, end_page=11)
        parent = IndexNode(title="Section I", start_page=8, end_page=8, children=[child_a, child_b])
        propagate_page_ranges([parent])
        assert parent.end_page == 11
        assert parent.start_page == 8

    def test_nested_propagation(self) -> None:
        """Propagation should work across multiple nesting levels."""
        grandchild = IndexNode(title="GC", start_page=5, end_page=15)
        child = IndexNode(title="C", start_page=5, end_page=5, children=[grandchild])
        root = IndexNode(title="R", start_page=1, end_page=4, children=[child])
        propagate_page_ranges([root])
        # child should expand to cover grandchild
        assert child.end_page == 15
        # root should expand to cover child
        assert root.end_page == 15

    def test_parent_start_shrinks_to_cover_children(self) -> None:
        """Parent start_page should shrink if a child starts earlier."""
        child = IndexNode(title="Early", start_page=2, end_page=5)
        parent = IndexNode(title="Parent", start_page=3, end_page=3, children=[child])
        propagate_page_ranges([parent])
        assert parent.start_page == 2

    def test_no_change_when_parent_already_covers(self) -> None:
        """No change needed when parent already spans its children."""
        child = IndexNode(title="C", start_page=3, end_page=5)
        parent = IndexNode(title="P", start_page=1, end_page=10, children=[child])
        propagate_page_ranges([parent])
        assert parent.start_page == 1
        assert parent.end_page == 10

    def test_leaf_nodes_unchanged(self) -> None:
        """Leaf nodes should not be altered."""
        leaf = IndexNode(title="Leaf", start_page=3, end_page=7)
        propagate_page_ranges([leaf])
        assert leaf.start_page == 3
        assert leaf.end_page == 7

    def test_empty_list(self) -> None:
        """Empty list should be a no-op."""
        propagate_page_ranges([])  # should not raise

    def test_matches_user_reported_scenario(self) -> None:
        """Reproduces the exact scenario from the user's error report."""
        cov_a = IndexNode(title="Coverage A", start_page=8, end_page=10)
        cov_c = IndexNode(title="Coverage C", start_page=10, end_page=11)
        section = IndexNode(
            title="SECTION I – PERILS INSURED AGAINST",
            start_page=8,
            end_page=8,
            children=[cov_a, cov_c],
        )
        propagate_page_ranges([section])
        assert section.start_page == 8
        assert section.end_page == 11


# ---------------------------------------------------------------------------
# attach_text_to_nodes
# ---------------------------------------------------------------------------


class TestAttachTextToNodes:
    """Tests for attach_text_to_nodes."""

    def test_text_attached(self) -> None:
        """Each node should get concatenated text from its page range."""
        from src.policy_index_v2.models import PageContent

        pages = [
            PageContent(page_number=1, text="Page 1. ", token_count=2),
            PageContent(page_number=2, text="Page 2. ", token_count=2),
            PageContent(page_number=3, text="Page 3. ", token_count=2),
        ]
        node = IndexNode(title="Test", start_page=1, end_page=2)
        attach_text_to_nodes([node], pages)
        assert node.text == "Page 1. Page 2. "

    def test_children_get_text(self) -> None:
        from src.policy_index_v2.models import PageContent

        pages = [
            PageContent(page_number=1, text="A", token_count=1),
            PageContent(page_number=2, text="B", token_count=1),
            PageContent(page_number=3, text="C", token_count=1),
        ]
        child = IndexNode(title="Child", start_page=2, end_page=3)
        parent = IndexNode(title="Parent", start_page=1, end_page=3, children=[child])
        attach_text_to_nodes([parent], pages)
        assert parent.text == "ABC"
        assert child.text == "BC"
