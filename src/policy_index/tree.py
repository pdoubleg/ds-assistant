"""Pure tree-manipulation utilities for ``IndexNode`` structures.

All functions in this module are synchronous and free of LLM calls, operating
purely on :class:`~src.policy_index_v2.models.IndexNode` instances.

Example:
    >>> from src.policy_index_v2.models import IndexNode
    >>> from src.policy_index_v2.tree import flatten_nodes, render_tree_markdown
    >>> root = IndexNode(
    ...     title="Part I", start_page=1, end_page=10,
    ...     children=[IndexNode(title="Ch 1", start_page=1, end_page=5)],
    ... )
    >>> flat = flatten_nodes([root])
    >>> print(render_tree_markdown([root]))
"""

from __future__ import annotations

from .models import IndexNode


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


def flatten_nodes(nodes: list[IndexNode]) -> list[IndexNode]:
    """Recursively flatten a tree of nodes into a depth-first ordered list.

    The returned list includes *all* nodes (inner and leaf).  Each node's
    ``children`` field is left intact so the tree structure is preserved.

    Args:
        nodes: Top-level list of tree nodes.

    Returns:
        A flat list of every node in depth-first order.
    """
    result: list[IndexNode] = []
    for node in nodes:
        result.append(node)
        if node.children:
            result.extend(flatten_nodes(node.children))
    return result


# ---------------------------------------------------------------------------
# Find
# ---------------------------------------------------------------------------


def find_node(nodes: list[IndexNode], node_id: str) -> IndexNode | None:
    """Search a tree for the first node matching *node_id*.

    Args:
        nodes: Top-level list of tree nodes to search.
        node_id: The ``node_id`` to look for.

    Returns:
        The matching :class:`IndexNode`, or ``None`` if not found.
    """
    for node in nodes:
        if node.node_id == node_id:
            return node
        if node.children:
            found = find_node(node.children, node_id)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# ID assignment
# ---------------------------------------------------------------------------


def assign_node_ids(nodes: list[IndexNode], start: int = 0) -> int:
    """Assign sequential zero-padded 4-digit IDs to every node in the tree.

    IDs are assigned in depth-first pre-order, matching the v1 ``write_node_id``
    behaviour.

    Args:
        nodes: Top-level list of tree nodes.
        start: The first ID integer to assign.

    Returns:
        The next available ID integer (so callers can continue numbering).
    """
    counter = start
    for node in nodes:
        node.node_id = str(counter).zfill(4)
        counter += 1
        if node.children:
            counter = assign_node_ids(node.children, counter)
    return counter


# ---------------------------------------------------------------------------
# Build tree from flat TOC items
# ---------------------------------------------------------------------------


def _parent_structure(structure: str) -> str | None:
    """Return the parent structure key, or ``None`` for top-level entries.

    Example:
        >>> _parent_structure("1.2.3")
        '1.2'
        >>> _parent_structure("1") is None
        True
    """
    parts = structure.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def build_tree_from_flat(
    items: list[dict[str, object]],
) -> list[IndexNode]:
    """Convert a flat list of TOC dicts into a tree of :class:`IndexNode`.

    Each dict must contain at least ``structure``, ``title``, ``start_index``,
    and ``end_index`` keys (matching the v1 post-processing output).

    Args:
        items: Flat list of section dictionaries.

    Returns:
        A list of root-level :class:`IndexNode` objects.
    """
    # Intermediate storage keyed by structure string
    node_map: dict[str, IndexNode] = {}
    roots: list[IndexNode] = []

    for item in items:
        structure = str(item.get("structure", ""))
        node = IndexNode(
            title=str(item.get("title", "")),
            start_page=int(item.get("start_index", 1)),  # type: ignore[arg-type]
            end_page=int(item.get("end_index", 1)),  # type: ignore[arg-type]
        )
        node_map[structure] = node

        parent_key = _parent_structure(structure)
        if parent_key and parent_key in node_map:
            node_map[parent_key].children.append(node)
        else:
            roots.append(node)

    return roots


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_tree_markdown(nodes: list[IndexNode], indent: int = 0) -> str:
    """Render a tree of nodes as an indented Markdown list.

    Each entry shows the node ID (if assigned), title, and page range.

    Args:
        nodes: Top-level tree nodes to render.
        indent: Current indentation level (used for recursion).

    Returns:
        A multi-line Markdown string.

    Example:
        Output might look like::

            - **[0000]** Declarations (pp. 1-2)
              - **[0001]** Section I - Definitions (pp. 3-5)
    """
    lines: list[str] = []
    prefix = "  " * indent + "- "
    for node in nodes:
        id_part = f"**[{node.node_id}]** " if node.node_id else ""
        lines.append(
            f"{prefix}{id_part}{node.title} (pp. {node.start_page}-{node.end_page})"
        )
        if node.children:
            lines.append(render_tree_markdown(node.children, indent + 1))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page-range propagation
# ---------------------------------------------------------------------------


def propagate_page_ranges(nodes: list[IndexNode]) -> None:
    """Expand each parent's page range to encompass all of its descendants.

    When the tree is built from a flat TOC, a parent's ``end_page`` is derived
    from the *next sibling's* start page -- which may be smaller than its last
    child's ``end_page``.  This bottom-up pass corrects that by setting::

        parent.end_page = max(parent.end_page, max(child.end_page for child))
        parent.start_page = min(parent.start_page, min(child.start_page for child))

    The function mutates nodes in place.

    Args:
        nodes: Top-level list of tree nodes.

    Example:
        Before::

            Section I (pp. 8-8)
              Coverage A (pp. 8-10)
              Coverage B (pp. 10-11)

        After::

            Section I (pp. 8-11)
              Coverage A (pp. 8-10)
              Coverage B (pp. 10-11)
    """
    for node in nodes:
        if node.children:
            # Recurse into children first (bottom-up)
            propagate_page_ranges(node.children)

            # Expand parent range to cover all children
            child_min = min(c.start_page for c in node.children)
            child_max = max(c.end_page for c in node.children)
            node.start_page = min(node.start_page, child_min)
            node.end_page = max(node.end_page, child_max)


# ---------------------------------------------------------------------------
# Text attachment helpers
# ---------------------------------------------------------------------------


def attach_text_to_nodes(
    nodes: list[IndexNode],
    pages: list[object],  # list[PageContent] but we accept duck-typed
) -> None:
    """Populate each node's ``text`` field from the extracted *pages*.

    Pages are indexed 1-based via ``start_page`` / ``end_page``.

    Args:
        nodes: Tree nodes to populate.
        pages: Ordered list of :class:`~.models.PageContent`-like objects
            with a ``.text`` attribute.
    """
    for node in nodes:
        # pages list is 0-indexed, node pages are 1-indexed
        start_idx = node.start_page - 1
        end_idx = node.end_page
        node.text = "".join(p.text for p in pages[start_idx:end_idx])  # type: ignore[union-attr]
        if node.children:
            attach_text_to_nodes(node.children, pages)


# ---------------------------------------------------------------------------
# Post-processing: flat TOC items -> tree with page ranges
# ---------------------------------------------------------------------------


def post_process_to_tree(
    flat_items: list[dict[str, object]],
    total_pages: int,
) -> list[IndexNode]:
    """Convert a flat TOC list with ``physical_index`` into a proper tree.

    This mirrors the v1 ``post_processing`` function: it assigns ``start_index``
    and ``end_index`` based on neighbouring items' ``physical_index`` and
    ``appear_start`` fields, then builds the tree.

    Args:
        flat_items: Flat list of TOC dicts (must have ``physical_index``,
            ``structure``, ``title``, and optionally ``appear_start``).
        total_pages: Total page count in the document (used as end for
            the last item).

    Returns:
        A list of root-level :class:`IndexNode` objects.
    """
    if not flat_items:
        return []

    # Assign start_index / end_index from physical_index and neighbours
    for i, item in enumerate(flat_items):
        item["start_index"] = item.get("physical_index")
        if i < len(flat_items) - 1:
            next_item = flat_items[i + 1]
            if next_item.get("appear_start") == "yes":
                item["end_index"] = int(next_item["physical_index"]) - 1  # type: ignore[arg-type]
            else:
                item["end_index"] = int(next_item["physical_index"])  # type: ignore[arg-type]
        else:
            item["end_index"] = total_pages

        # Defensive clamp: if the arithmetic above produced an inverted range
        # (e.g. two adjacent sections mapped to the same page with appear_start),
        # ensure end_index is at least start_index.
        start = int(item["start_index"])  # type: ignore[arg-type]
        end = int(item["end_index"])  # type: ignore[arg-type]
        if end < start:
            item["end_index"] = start

    return build_tree_from_flat(flat_items)
