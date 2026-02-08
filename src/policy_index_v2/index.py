"""PolicyIndex -- the public API for policy_index_v2.

This module provides a vector-db-style interface for creating, caching,
and querying hierarchical document indexes built from PDFs.

Example:
    >>> import asyncio
    >>> from src.policy_index_v2 import PolicyIndex, IndexConfig
    >>>
    >>> async def main():
    ...     pi = PolicyIndex(IndexConfig(model="gpt-4.1-mini"))
    ...     doc = await pi.get_or_create("data/HO3_sample.pdf")
    ...     print(pi.list_documents())
    ...     print(pi.tree("HO3_sample.pdf"))
    ...     node = pi.get_node("0001")
    ...     print(node.title)
    >>>
    >>> asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from .agents import (
    create_description_agent,
    create_page_index_detector_agent,
    create_page_number_adder_agent,
    create_single_item_fixer_agent,
    create_summary_agent,
    create_title_checker_agent,
    create_title_start_checker_agent,
    create_toc_detector_agent,
    create_toc_generator_continue_agent,
    create_toc_generator_init_agent,
    create_toc_index_extractor_agent,
    create_toc_transform_agent,
    description_prompt,
    page_index_detection_prompt,
    page_number_adder_prompt,
    single_item_fixer_prompt,
    summary_prompt,
    title_appearance_prompt,
    title_start_prompt,
    toc_detector_prompt,
    toc_generator_continue_prompt,
    toc_generator_init_prompt,
    toc_index_extractor_prompt,
    toc_transform_prompt,
)
from .models import DocumentIndex, IndexConfig, IndexNode, PageContent
from .pdf import extract_pages, get_pdf_name, group_pages_by_tokens
from .tree import (
    assign_node_ids,
    attach_text_to_nodes,
    find_node,
    flatten_nodes,
    post_process_to_tree,
    propagate_page_ranges,
    render_tree_markdown,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PolicyIndex -- the public API
# ---------------------------------------------------------------------------


class PolicyIndex:
    """A vector-db-style manager for hierarchical PDF document indexes.

    Maintains an in-memory cache of :class:`DocumentIndex` objects keyed by
    document name.  Use :meth:`get_or_create` to index a PDF (or retrieve a
    cached result), then query via :meth:`list_documents`, :meth:`tree`,
    :meth:`get_node`, and :meth:`get_nodes`.

    Args:
        config: Optional configuration.  Defaults to :class:`IndexConfig`
            with all default values.

    Attributes:
        config: The active :class:`IndexConfig`.
    """

    def __init__(self, config: IndexConfig | None = None) -> None:
        self.config: IndexConfig = config or IndexConfig()
        self._store: dict[str, DocumentIndex] = {}
        # Create a semaphore for throttling concurrent LLM calls
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def _sem(self) -> asyncio.Semaphore:
        """Lazy-init semaphore (must be created inside an event loop)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_llm_calls)
        return self._semaphore

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    async def get_or_create(self, file_path: str | Path) -> DocumentIndex:
        """Return a cached index for the document, or build one from scratch.

        If the document has already been indexed (keyed by filename), the
        cached :class:`DocumentIndex` is returned immediately.  Otherwise the
        full async pipeline runs: PDF extraction -> TOC detection -> structure
        building -> optional summaries and descriptions.

        Args:
            file_path: Path to a PDF file.

        Returns:
            The :class:`DocumentIndex` for the document.

        Raises:
            ValueError: If *file_path* is not a valid PDF.
        """
        doc_name = get_pdf_name(file_path)
        if doc_name in self._store:
            logger.info("Returning cached index for %s", doc_name)
            return self._store[doc_name]

        logger.info("Building index for %s", doc_name)
        doc_index = await self._build_index(file_path)
        self._store[doc_name] = doc_index
        return doc_index

    def get(self, doc_name: str) -> DocumentIndex | None:
        """Retrieve a cached index by document name.

        Args:
            doc_name: Exact document name (e.g. ``"HO3_sample.pdf"``).

        Returns:
            The cached :class:`DocumentIndex`, or ``None`` if not found.
        """
        return self._store.get(doc_name)

    def remove(self, doc_name: str) -> bool:
        """Remove a cached index.

        Args:
            doc_name: Document name to remove.

        Returns:
            ``True`` if the document was found and removed.
        """
        if doc_name in self._store:
            del self._store[doc_name]
            return True
        return False

    # ------------------------------------------------------------------
    # Rendering helpers (return Markdown strings for LLM consumption)
    # ------------------------------------------------------------------

    def list_documents(self) -> str:
        """Return a Markdown bullet list of all indexed document names.

        Returns:
            Multi-line Markdown string, or a message if the store is empty.
        """
        if not self._store:
            return "_No documents indexed._"
        lines = [f"- {name}" for name in sorted(self._store)]
        return "\n".join(lines)

    def tree(self, *doc_names: str) -> str:
        """Render Markdown tree view(s) for the given document(s).

        If no names are provided, trees for *all* indexed documents are
        returned.

        Args:
            *doc_names: Zero or more document names.  An empty call means
                "all documents".

        Returns:
            Multi-line Markdown string.

        Raises:
            KeyError: If a requested document name is not in the store.
        """
        names = doc_names or tuple(sorted(self._store.keys()))
        sections: list[str] = []
        for name in names:
            if name not in self._store:
                raise KeyError(f"Document not found: {name!r}")
            doc = self._store[name]
            header = f"## {doc.doc_name}"
            if doc.doc_description:
                header += f"\n\n_{doc.doc_description}_"
            body = render_tree_markdown(doc.root_nodes)
            sections.append(f"{header}\n\n{body}")
        return "\n\n".join(sections)

    def get_node(self, node_id: str) -> IndexNode:
        """Look up a single node by ID across all indexed documents.

        Args:
            node_id: The zero-padded 4-digit node ID.

        Returns:
            The matching :class:`IndexNode`.

        Raises:
            KeyError: If no node with that ID exists in any document.
        """
        for doc in self._store.values():
            node = find_node(doc.root_nodes, node_id)
            if node is not None:
                return node
        raise KeyError(f"Node not found: {node_id!r}")

    def get_nodes(self, node_ids: list[str]) -> list[IndexNode]:
        """Look up multiple nodes by ID.

        Args:
            node_ids: List of node IDs.

        Returns:
            List of matching :class:`IndexNode` objects (order matches input).

        Raises:
            KeyError: If any node ID is not found.
        """
        return [self.get_node(node) for node in node_ids]

    # ------------------------------------------------------------------
    # Internal: full indexing pipeline
    # ------------------------------------------------------------------

    async def _build_index(self, source: str | Path | BytesIO) -> DocumentIndex:
        """Run the full async indexing pipeline for a single PDF.

        This mirrors the v1 ``page_index_main`` function but is fully async
        and uses pydantic-ai agents for all LLM calls.

        Args:
            source: PDF file path or BytesIO stream.

        Returns:
            A complete :class:`DocumentIndex`.
        """
        cfg = self.config
        doc_name = get_pdf_name(source)

        # 1. Extract pages
        pages = extract_pages(source, model=cfg.model)
        logger.info("Extracted %d pages from %s", len(pages), doc_name)

        # 2. Build tree structure
        root_nodes = await self._tree_parser(pages)

        # 3. Assign node IDs
        if cfg.add_node_ids:
            assign_node_ids(root_nodes)

        # 4. Attach text
        if cfg.add_text:
            attach_text_to_nodes(root_nodes, pages)

        # 5. Generate summaries
        doc_description: str | None = None
        if cfg.add_summaries:
            # If text wasn't already attached, temporarily attach it for summary generation
            text_was_attached = cfg.add_text
            if not text_was_attached:
                attach_text_to_nodes(root_nodes, pages)

            await self._generate_all_summaries(root_nodes)

            # 6. Generate document description
            if cfg.add_descriptions:
                doc_description = await self._generate_doc_description(root_nodes)

            # Remove text if it wasn't supposed to be kept
            if not text_was_attached:
                self._strip_text(root_nodes)

        return DocumentIndex(
            doc_name=doc_name,
            doc_description=doc_description,
            root_nodes=root_nodes,
        )

    # ------------------------------------------------------------------
    # Internal: tree parsing pipeline
    # ------------------------------------------------------------------

    async def _tree_parser(self, pages: list[PageContent]) -> list[IndexNode]:
        """Orchestrate the full tree-extraction pipeline.

        Steps:
        1. Detect TOC pages
        2. Route to the appropriate processing mode
        3. Add preface if needed
        4. Check title-start positions concurrently
        5. Post-process into tree
        6. Recursively split large nodes

        Args:
            pages: Extracted PDF pages.

        Returns:
            List of root-level :class:`IndexNode` objects.
        """
        # Step 1: Check for TOC
        toc_result = await self._check_toc(pages)
        toc_content: str | None = toc_result.get("toc_content")
        toc_page_list: list[int] = toc_result.get("toc_page_list", [])
        has_page_index: bool = toc_result.get("page_index_given_in_toc", False)

        # Step 2: Route to processing mode
        if toc_content and toc_content.strip() and has_page_index:
            flat_items = await self._meta_processor(
                pages,
                mode="process_toc_with_page_numbers",
                toc_content=toc_content,
                toc_page_list=toc_page_list,
            )
        else:
            flat_items = await self._meta_processor(
                pages,
                mode="process_no_toc",
            )

        # Step 3: Add preface if document starts before first section
        flat_items = self._add_preface_if_needed(flat_items)

        # Step 4: Check title-start positions concurrently
        flat_items = await self._check_title_starts_concurrent(flat_items, pages)

        # Step 5: Filter valid items and build tree
        valid_items = [item for item in flat_items if item.get("physical_index") is not None]
        root_nodes = post_process_to_tree(valid_items, len(pages))

        # Step 6: Recursively split large nodes
        tasks = [
            self._process_large_node(node, pages)
            for node in root_nodes
        ]
        await asyncio.gather(*tasks)

        # Step 7: Propagate page ranges bottom-up so parents span their children
        propagate_page_ranges(root_nodes)

        return root_nodes

    # ------------------------------------------------------------------
    # Internal: TOC detection
    # ------------------------------------------------------------------

    async def _check_toc(self, pages: list[PageContent]) -> dict[str, Any]:
        """Detect whether the document has a table of contents.

        Args:
            pages: All extracted PDF pages.

        Returns:
            Dict with keys ``toc_content``, ``toc_page_list``,
            ``page_index_given_in_toc``.
        """
        cfg = self.config
        toc_page_list = await self._find_toc_pages(pages, start_index=0)

        if not toc_page_list:
            logger.info("No TOC found")
            return {"toc_content": None, "toc_page_list": [], "page_index_given_in_toc": False}

        logger.info("TOC pages found: %s", toc_page_list)
        toc_result = await self._extract_and_detect_toc(pages, toc_page_list)

        if toc_result["page_index_given_in_toc"]:
            return {
                "toc_content": toc_result["toc_content"],
                "toc_page_list": toc_page_list,
                "page_index_given_in_toc": True,
            }

        # Search for additional TOC sections with page numbers
        current_start = toc_page_list[-1] + 1
        while current_start < len(pages) and current_start < cfg.toc_check_pages:
            additional_pages = await self._find_toc_pages(pages, start_index=current_start)
            if not additional_pages:
                break
            additional_result = await self._extract_and_detect_toc(pages, additional_pages)
            if additional_result["page_index_given_in_toc"]:
                return {
                    "toc_content": additional_result["toc_content"],
                    "toc_page_list": additional_pages,
                    "page_index_given_in_toc": True,
                }
            current_start = additional_pages[-1] + 1

        # Fall back -- TOC found but no page indices
        return {
            "toc_content": toc_result["toc_content"],
            "toc_page_list": toc_page_list,
            "page_index_given_in_toc": False,
        }

    async def _find_toc_pages(
        self, pages: list[PageContent], start_index: int = 0
    ) -> list[int]:
        """Scan pages sequentially to find TOC pages.

        Args:
            pages: All extracted pages.
            start_index: 0-based page index to start scanning from.

        Returns:
            List of 0-based page indices that are TOC pages.
        """
        cfg = self.config
        agent = create_toc_detector_agent(cfg.model)
        toc_page_list: list[int] = []
        last_was_toc = False
        i = start_index

        while i < len(pages):
            # Stop scanning beyond toc_check_pages unless we're mid-streak
            if i >= cfg.toc_check_pages and not last_was_toc:
                break

            async with self._sem:
                result = await agent.run(toc_detector_prompt(pages[i].text))

            if result.output.is_toc_page:
                toc_page_list.append(i)
                last_was_toc = True
            elif last_was_toc:
                # End of TOC streak
                break
            i += 1

        return toc_page_list

    async def _extract_and_detect_toc(
        self, pages: list[PageContent], toc_page_list: list[int]
    ) -> dict[str, Any]:
        """Extract TOC text from pages and check for page numbers.

        Args:
            pages: All extracted pages.
            toc_page_list: 0-based indices of TOC pages.

        Returns:
            Dict with ``toc_content`` and ``page_index_given_in_toc``.
        """
        cfg = self.config

        # Concatenate TOC page text, replacing dot leaders with colons
        toc_text = ""
        for idx in toc_page_list:
            toc_text += pages[idx].text
        toc_text = re.sub(r"\.{5,}", ": ", toc_text)
        toc_text = re.sub(r"(?:\. ){5,}\.?", ": ", toc_text)

        # Check if page numbers are present
        detector_agent = create_page_index_detector_agent(cfg.model)
        async with self._sem:
            result = await detector_agent.run(page_index_detection_prompt(toc_text))

        return {
            "toc_content": toc_text,
            "page_index_given_in_toc": result.output.page_index_given_in_toc,
        }

    # ------------------------------------------------------------------
    # Internal: meta-processor (routing)
    # ------------------------------------------------------------------

    async def _meta_processor(
        self,
        pages: list[PageContent],
        mode: str,
        toc_content: str | None = None,
        toc_page_list: list[int] | None = None,
        start_index: int = 1,
    ) -> list[dict[str, Any]]:
        """Route to the correct processing mode and verify/fix results.

        Args:
            pages: Extracted pages.
            mode: One of ``"process_toc_with_page_numbers"``,
                ``"process_toc_no_page_numbers"``, ``"process_no_toc"``.
            toc_content: Raw TOC text (for modes that need it).
            toc_page_list: 0-based TOC page indices.
            start_index: 1-based start index for physical page numbering.

        Returns:
            Flat list of TOC item dicts with ``physical_index``.
        """
        logger.info("Processing mode: %s, start_index: %d", mode, start_index)

        if mode == "process_toc_with_page_numbers":
            items = await self._process_toc_with_page_numbers(
                toc_content or "", toc_page_list or [], pages, start_index
            )
        elif mode == "process_toc_no_page_numbers":
            items = await self._process_toc_no_page_numbers(
                toc_content or "", toc_page_list or [], pages, start_index
            )
        else:
            items = await self._process_no_toc(pages, start_index)

        # Filter out items without physical_index
        items = [item for item in items if item.get("physical_index") is not None]

        # Validate indices are within document bounds
        items = self._validate_indices(items, len(pages), start_index)

        # Verify accuracy
        accuracy, incorrect = await self._verify_toc(pages, items, start_index)
        logger.info("Verification accuracy: %.2f%%, %d incorrect", accuracy * 100, len(incorrect))

        if accuracy == 1.0 and not incorrect:
            return items

        if accuracy > 0.6 and incorrect:
            items, _ = await self._fix_incorrect_with_retries(
                items, pages, incorrect, start_index, max_attempts=3
            )
            return items

        # Fall back to simpler modes
        if mode == "process_toc_with_page_numbers":
            return await self._meta_processor(
                pages, "process_toc_no_page_numbers",
                toc_content=toc_content, toc_page_list=toc_page_list,
                start_index=start_index,
            )
        elif mode == "process_toc_no_page_numbers":
            return await self._meta_processor(
                pages, "process_no_toc", start_index=start_index,
            )

        raise RuntimeError("All processing modes failed")

    # ------------------------------------------------------------------
    # Internal: process_no_toc
    # ------------------------------------------------------------------

    async def _process_no_toc(
        self,
        pages: list[PageContent],
        start_index: int = 1,
    ) -> list[dict[str, Any]]:
        """Generate a TOC from scratch when no TOC pages exist.

        Args:
            pages: Extracted pages.
            start_index: 1-based start index.

        Returns:
            Flat list of TOC item dicts.
        """
        cfg = self.config
        group_texts = group_pages_by_tokens(pages, start_index=start_index)
        logger.info("Divided pages into %d groups (no TOC mode)", len(group_texts))

        # Generate initial TOC from first group
        init_agent = create_toc_generator_init_agent(cfg.model)
        async with self._sem:
            init_result = await init_agent.run(toc_generator_init_prompt(group_texts[0]))

        toc_items = [entry.model_dump() for entry in init_result.output]

        # Continue for remaining groups
        continue_agent = create_toc_generator_continue_agent(cfg.model)
        for group_text in group_texts[1:]:
            async with self._sem:
                cont_result = await continue_agent.run(
                    toc_generator_continue_prompt(group_text, toc_items)
                )
            toc_items.extend(entry.model_dump() for entry in cont_result.output)

        return toc_items

    # ------------------------------------------------------------------
    # Internal: process_toc_with_page_numbers
    # ------------------------------------------------------------------

    async def _process_toc_with_page_numbers(
        self,
        toc_content: str,
        toc_page_list: list[int],
        pages: list[PageContent],
        start_index: int = 1,
    ) -> list[dict[str, Any]]:
        """Process a TOC that includes printed page numbers.

        Uses page-offset calculation to map printed page numbers to physical
        indices.

        Args:
            toc_content: Raw TOC text.
            toc_page_list: 0-based TOC page indices.
            pages: All extracted pages.
            start_index: 1-based start index.

        Returns:
            Flat list of TOC item dicts with ``physical_index``.
        """
        cfg = self.config

        # Transform TOC to structured entries
        transform_agent = create_toc_transform_agent(cfg.model)
        async with self._sem:
            transform_result = await transform_agent.run(toc_transform_prompt(toc_content))
        toc_entries = [e.model_dump() for e in transform_result.output.table_of_contents]

        # Prepare a version without page numbers for physical-index extraction
        toc_no_page = [
            {"structure": e.get("structure"), "title": e.get("title")}
            for e in toc_entries
        ]

        # Extract physical indices from a few pages after the TOC
        content_start = toc_page_list[-1] + 1
        content_end = min(content_start + cfg.toc_check_pages, len(pages))
        main_content = ""
        for p in pages[content_start:content_end]:
            main_content += (
                f"<physical_index_{p.page_number}>\n{p.text}\n"
                f"<physical_index_{p.page_number}>\n\n"
            )

        extractor_agent = create_toc_index_extractor_agent(cfg.model)
        async with self._sem:
            extract_result = await extractor_agent.run(
                toc_index_extractor_prompt(toc_no_page, main_content)
            )
        physical_entries = [e.model_dump() for e in extract_result.output]

        # Calculate page offset from matching pairs
        pairs = self._extract_matching_pairs(toc_entries, physical_entries, content_start + 1)
        offset = self._calculate_page_offset(pairs)

        if offset is not None:
            # Apply offset to all entries that have a page number
            for entry in toc_entries:
                if entry.get("page") is not None:
                    entry["physical_index"] = entry["page"] + offset
                    del entry["page"]
        else:
            # Fallback: use physical entries directly
            for entry in toc_entries:
                # Try to match by title
                for pe in physical_entries:
                    if pe.get("title") == entry.get("title") and pe.get("physical_index"):
                        entry["physical_index"] = pe["physical_index"]
                        break

        # Fix entries that still don't have a physical_index
        toc_entries = await self._fill_missing_page_numbers(toc_entries, pages, start_index)

        return toc_entries

    # ------------------------------------------------------------------
    # Internal: process_toc_no_page_numbers
    # ------------------------------------------------------------------

    async def _process_toc_no_page_numbers(
        self,
        toc_content: str,
        toc_page_list: list[int],
        pages: list[PageContent],
        start_index: int = 1,
    ) -> list[dict[str, Any]]:
        """Process a TOC without printed page numbers.

        Transforms the TOC into entries, then uses page content to locate
        where each section starts.

        Args:
            toc_content: Raw TOC text.
            toc_page_list: 0-based TOC page indices.
            pages: All extracted pages.
            start_index: 1-based start index.

        Returns:
            Flat list of TOC item dicts with ``physical_index``.
        """
        cfg = self.config

        # Transform TOC to structured entries
        transform_agent = create_toc_transform_agent(cfg.model)
        async with self._sem:
            transform_result = await transform_agent.run(toc_transform_prompt(toc_content))
        toc_entries = [e.model_dump() for e in transform_result.output.table_of_contents]

        # Group pages for page-number addition
        group_texts = group_pages_by_tokens(pages, start_index=start_index)

        # Add page numbers using document content
        adder_agent = create_page_number_adder_agent(cfg.model)
        current_structure = copy.deepcopy(toc_entries)
        for group_text in group_texts:
            async with self._sem:
                result = await adder_agent.run(
                    page_number_adder_prompt(group_text, current_structure)
                )
            # Update structure with new physical indices
            result_dicts = [e.model_dump() for e in result.output]
            for orig, updated in zip(current_structure, result_dicts):
                if updated.get("physical_index") is not None:
                    orig["physical_index"] = updated["physical_index"]

        return current_structure

    # ------------------------------------------------------------------
    # Internal: verification and fixing
    # ------------------------------------------------------------------

    async def _verify_toc(
        self,
        pages: list[PageContent],
        items: list[dict[str, Any]],
        start_index: int = 1,
        sample_size: int | None = None,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Verify TOC accuracy by checking title appearances on mapped pages.

        Args:
            pages: Extracted pages.
            items: Flat TOC items with ``physical_index``.
            start_index: 1-based start index.
            sample_size: Number of items to sample (None = check all).

        Returns:
            Tuple of (accuracy, list of incorrect items).
        """
        # Find last valid physical_index
        last_idx = None
        for item in reversed(items):
            if item.get("physical_index") is not None:
                last_idx = item["physical_index"]
                break

        if last_idx is None or last_idx < len(pages) / 2:
            return 0.0, []

        # Select items to check
        valid_items = [
            (i, item) for i, item in enumerate(items)
            if item.get("physical_index") is not None
        ]

        if sample_size is not None:
            sample_size = min(sample_size, len(valid_items))
            valid_items = random.sample(valid_items, sample_size)

        # Run checks concurrently
        checker_agent = create_title_checker_agent(self.config.model)

        async def _check_one(list_idx: int, item: dict[str, Any]) -> dict[str, Any]:
            page_num = item["physical_index"]
            page_idx = page_num - start_index
            if page_idx < 0 or page_idx >= len(pages):
                return {"list_index": list_idx, "answer": "no", "title": item["title"], "page_number": page_num}

            async with self._sem:
                result = await checker_agent.run(
                    title_appearance_prompt(item["title"], pages[page_idx].text)
                )
            return {
                "list_index": list_idx,
                "answer": result.output.answer,
                "title": item["title"],
                "page_number": page_num,
            }

        tasks = [_check_one(idx, item) for idx, item in valid_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        correct = 0
        incorrect: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Verification check failed: %s", r)
                continue
            if r["answer"] == "yes":
                correct += 1
            else:
                incorrect.append(r)

        total_checked = sum(1 for r in results if not isinstance(r, Exception))
        accuracy = correct / total_checked if total_checked > 0 else 0.0
        return accuracy, incorrect

    async def _fix_incorrect_with_retries(
        self,
        items: list[dict[str, Any]],
        pages: list[PageContent],
        incorrect: list[dict[str, Any]],
        start_index: int = 1,
        max_attempts: int = 3,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Attempt to fix incorrectly-mapped TOC entries with retries.

        Args:
            items: Current flat TOC items.
            pages: Extracted pages.
            incorrect: List of incorrectly-mapped items.
            start_index: 1-based start index.
            max_attempts: Maximum number of fix attempts.

        Returns:
            Tuple of (updated items, remaining incorrect items).
        """
        current_items = items
        current_incorrect = incorrect

        for attempt in range(max_attempts):
            if not current_incorrect:
                break
            logger.info("Fix attempt %d: %d items to fix", attempt + 1, len(current_incorrect))
            current_items, current_incorrect = await self._fix_incorrect_toc(
                current_items, pages, current_incorrect, start_index
            )

        return current_items, current_incorrect

    async def _fix_incorrect_toc(
        self,
        items: list[dict[str, Any]],
        pages: list[PageContent],
        incorrect: list[dict[str, Any]],
        start_index: int = 1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fix a batch of incorrect TOC entries concurrently.

        Args:
            items: Current flat TOC items.
            pages: Extracted pages.
            incorrect: Incorrect items to fix.
            start_index: 1-based start index.

        Returns:
            Tuple of (updated items, still-incorrect items).
        """
        cfg = self.config
        incorrect_indices = {r["list_index"] for r in incorrect}
        end_index = len(pages) + start_index - 1
        fixer_agent = create_single_item_fixer_agent(cfg.model)
        checker_agent = create_title_checker_agent(cfg.model)

        async def _fix_one(inc_item: dict[str, Any]) -> dict[str, Any]:
            list_idx = inc_item["list_index"]

            # Find surrounding valid boundaries
            prev_idx = start_index - 1
            for j in range(list_idx - 1, -1, -1):
                if j not in incorrect_indices and 0 <= j < len(items):
                    pi = items[j].get("physical_index")
                    if pi is not None:
                        prev_idx = pi
                        break

            next_idx = end_index
            for j in range(list_idx + 1, len(items)):
                if j not in incorrect_indices and 0 <= j < len(items):
                    pi = items[j].get("physical_index")
                    if pi is not None:
                        next_idx = pi
                        break

            # Build page content for the range
            content_parts: list[str] = []
            for page_num in range(prev_idx, next_idx + 1):
                page_list_idx = page_num - start_index
                if 0 <= page_list_idx < len(pages):
                    p = pages[page_list_idx]
                    content_parts.append(
                        f"<physical_index_{page_num}>\n{p.text}\n"
                        f"<physical_index_{page_num}>\n\n"
                    )

            content_range = "".join(content_parts)

            # Fix the item
            async with self._sem:
                fix_result = await fixer_agent.run(
                    single_item_fixer_prompt(inc_item["title"], content_range)
                )
            new_idx = fix_result.output.physical_index

            # Verify the fix
            is_valid = False
            if new_idx is not None:
                page_list_idx = new_idx - start_index
                if 0 <= page_list_idx < len(pages):
                    async with self._sem:
                        check = await checker_agent.run(
                            title_appearance_prompt(inc_item["title"], pages[page_list_idx].text)
                        )
                    is_valid = check.output.answer == "yes"

            return {
                "list_index": list_idx,
                "title": inc_item["title"],
                "physical_index": new_idx,
                "is_valid": is_valid,
            }

        tasks = [_fix_one(inc) for inc in incorrect]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        still_incorrect: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Fix failed: %s", r)
                continue
            if r["is_valid"]:
                idx = r["list_index"]
                if 0 <= idx < len(items):
                    items[idx]["physical_index"] = r["physical_index"]
            else:
                still_incorrect.append({
                    "list_index": r["list_index"],
                    "title": r["title"],
                    "physical_index": r["physical_index"],
                })

        return items, still_incorrect

    # ------------------------------------------------------------------
    # Internal: title-start checking
    # ------------------------------------------------------------------

    async def _check_title_starts_concurrent(
        self,
        flat_items: list[dict[str, Any]],
        pages: list[PageContent],
    ) -> list[dict[str, Any]]:
        """Check whether each section starts at the beginning of its page.

        Sets the ``appear_start`` field on each item (``"yes"`` or ``"no"``).

        Args:
            flat_items: Flat TOC items with ``physical_index``.
            pages: All extracted pages.

        Returns:
            The same list, with ``appear_start`` populated.
        """
        cfg = self.config
        agent = create_title_start_checker_agent(cfg.model)

        # Items without physical_index default to "no"
        for item in flat_items:
            if item.get("physical_index") is None:
                item["appear_start"] = "no"

        valid_items = [
            item for item in flat_items
            if item.get("physical_index") is not None
        ]

        async def _check(item: dict[str, Any]) -> tuple[dict[str, Any], str]:
            page_idx = item["physical_index"] - 1
            if page_idx < 0 or page_idx >= len(pages):
                return item, "no"
            async with self._sem:
                result = await agent.run(
                    title_start_prompt(item["title"], pages[page_idx].text)
                )
            return item, result.output.start_begin

        tasks = [_check(item) for item in valid_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.warning("Title-start check failed: %s", r)
                continue
            item, start_begin = r
            item["appear_start"] = start_begin

        return flat_items

    # ------------------------------------------------------------------
    # Internal: large-node recursive splitting
    # ------------------------------------------------------------------

    async def _process_large_node(
        self,
        node: IndexNode,
        pages: list[PageContent],
    ) -> None:
        """Recursively split nodes that exceed size limits.

        Modifies *node* in place by adding children.

        Args:
            node: The node to potentially split.
            pages: All extracted pages.
        """
        cfg = self.config

        # Calculate token count for this node's page range
        node_pages = pages[node.start_page - 1 : node.end_page]
        token_count = sum(p.token_count for p in node_pages)

        page_span = node.end_page - node.start_page
        if page_span > cfg.max_pages_per_node and token_count >= cfg.max_tokens_per_node:
            logger.info(
                "Splitting large node '%s' (pages %d-%d, %d tokens)",
                node.title, node.start_page, node.end_page, token_count,
            )

            # Generate sub-structure for this node
            sub_items = await self._meta_processor(
                node_pages,
                mode="process_no_toc",
                start_index=node.start_page,
            )
            sub_items = await self._check_title_starts_concurrent(sub_items, pages)
            valid_sub = [i for i in sub_items if i.get("physical_index") is not None]

            if valid_sub:
                # If the first sub-item matches the node title, skip it
                if valid_sub[0]["title"].strip() == node.title.strip():
                    child_nodes = post_process_to_tree(valid_sub[1:], node.end_page)
                    if len(valid_sub) > 1:
                        node.end_page = valid_sub[1].get("physical_index", node.end_page)  # type: ignore[assignment]
                else:
                    child_nodes = post_process_to_tree(valid_sub, node.end_page)
                    node.end_page = valid_sub[0].get("physical_index", node.end_page)  # type: ignore[assignment]
                node.children = child_nodes

        # Recurse into children
        if node.children:
            tasks = [self._process_large_node(child, pages) for child in node.children]
            await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Internal: summary generation
    # ------------------------------------------------------------------

    async def _generate_all_summaries(self, nodes: list[IndexNode]) -> None:
        """Generate summaries for all nodes concurrently.

        Modifies nodes in place.

        Args:
            nodes: Root-level tree nodes.
        """
        flat = flatten_nodes(nodes)
        agent = create_summary_agent(self.config.model)

        async def _summarize(node: IndexNode) -> None:
            if node.text:
                async with self._sem:
                    result = await agent.run(summary_prompt(node.text))
                node.summary = result.output.summary

        tasks = [_summarize(n) for n in flat]
        await asyncio.gather(*tasks)

    async def _generate_doc_description(self, nodes: list[IndexNode]) -> str:
        """Generate a one-sentence document description.

        Args:
            nodes: Root-level tree nodes (with summaries).

        Returns:
            The generated description string.
        """
        # Build a clean structure summary for the description agent
        structure_repr = self._clean_structure_for_description(nodes)
        agent = create_description_agent(self.config.model)
        async with self._sem:
            result = await agent.run(description_prompt(str(structure_repr)))
        return result.output.description

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_preface_if_needed(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add a 'Preface' entry if the document starts before the first section.

        Args:
            items: Flat TOC items.

        Returns:
            Updated list (potentially with a prepended Preface entry).
        """
        if not items:
            return items
        first_idx = items[0].get("physical_index")
        if first_idx is not None and first_idx > 1:
            items.insert(0, {
                "structure": "0",
                "title": "Preface",
                "physical_index": 1,
            })
        return items

    @staticmethod
    def _validate_indices(
        items: list[dict[str, Any]],
        total_pages: int,
        start_index: int = 1,
    ) -> list[dict[str, Any]]:
        """Remove entries whose physical_index exceeds the document length.

        Args:
            items: Flat TOC items.
            total_pages: Total pages in the PDF.
            start_index: 1-based start index.

        Returns:
            The cleaned list.
        """
        max_allowed = total_pages + start_index - 1
        for item in items:
            idx = item.get("physical_index")
            if idx is not None and idx > max_allowed:
                logger.warning(
                    "Removed out-of-bounds index for '%s' (was %d, max %d)",
                    item.get("title", "?"), idx, max_allowed,
                )
                item["physical_index"] = None
        return items

    @staticmethod
    def _extract_matching_pairs(
        toc_entries: list[dict[str, Any]],
        physical_entries: list[dict[str, Any]],
        min_page: int,
    ) -> list[dict[str, Any]]:
        """Find TOC entries that match physical entries by title.

        Args:
            toc_entries: Entries from TOC transformation (with ``page``).
            physical_entries: Entries from physical-index extraction.
            min_page: Minimum valid page number.

        Returns:
            List of matching pairs.
        """
        pairs: list[dict[str, Any]] = []
        for pe in physical_entries:
            for te in toc_entries:
                if pe.get("title") == te.get("title"):
                    pi = pe.get("physical_index")
                    if pi is not None and int(pi) >= min_page:
                        pairs.append({
                            "title": pe["title"],
                            "page": te.get("page"),
                            "physical_index": pi,
                        })
        return pairs

    @staticmethod
    def _calculate_page_offset(pairs: list[dict[str, Any]]) -> int | None:
        """Calculate the most common offset between printed and physical pages.

        Args:
            pairs: Matching pairs with ``page`` and ``physical_index``.

        Returns:
            The most common offset, or ``None`` if no pairs exist.
        """
        diffs: list[int] = []
        for pair in pairs:
            try:
                diffs.append(int(pair["physical_index"]) - int(pair["page"]))
            except (KeyError, TypeError, ValueError):
                continue
        if not diffs:
            return None
        counts: dict[int, int] = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    async def _fill_missing_page_numbers(
        self,
        items: list[dict[str, Any]],
        pages: list[PageContent],
        start_index: int = 1,
    ) -> list[dict[str, Any]]:
        """Fill in physical_index for items that still don't have one.

        Uses surrounding valid entries to narrow the search range.

        Args:
            items: Flat TOC items (some may lack ``physical_index``).
            pages: Extracted pages.
            start_index: 1-based start index.

        Returns:
            Updated list.
        """
        cfg = self.config
        adder_agent = create_page_number_adder_agent(cfg.model)

        for i, item in enumerate(items):
            if "physical_index" in item and item["physical_index"] is not None:
                continue

            # Find surrounding valid boundaries
            prev_idx = 0
            for j in range(i - 1, -1, -1):
                if items[j].get("physical_index") is not None:
                    prev_idx = items[j]["physical_index"]
                    break

            next_idx = len(pages) + start_index - 1
            for j in range(i + 1, len(items)):
                if items[j].get("physical_index") is not None:
                    next_idx = items[j]["physical_index"]
                    break

            # Build page content for the range
            content_parts: list[str] = []
            for page_num in range(prev_idx, next_idx + 1):
                page_list_idx = page_num - start_index
                if 0 <= page_list_idx < len(pages):
                    p = pages[page_list_idx]
                    content_parts.append(
                        f"<physical_index_{page_num}>\n{p.text}\n"
                        f"<physical_index_{page_num}>\n\n"
                    )

            item_for_lookup = {"structure": item.get("structure"), "title": item["title"]}
            async with self._sem:
                result = await adder_agent.run(
                    page_number_adder_prompt("".join(content_parts), [item_for_lookup])
                )
            if result.output and result.output[0].physical_index is not None:
                item["physical_index"] = result.output[0].physical_index
                # Remove the old 'page' key if present
                item.pop("page", None)

        return items

    @staticmethod
    def _clean_structure_for_description(nodes: list[IndexNode]) -> list[dict[str, Any]]:
        """Build a lightweight structure dict for description generation.

        Args:
            nodes: Tree nodes (with summaries).

        Returns:
            Nested list of dicts with ``title``, ``node_id``, ``summary``, and ``children``.
        """
        result: list[dict[str, Any]] = []
        for node in nodes:
            entry: dict[str, Any] = {"title": node.title}
            if node.node_id:
                entry["node_id"] = node.node_id
            if node.summary:
                entry["summary"] = node.summary
            if node.children:
                entry["children"] = PolicyIndex._clean_structure_for_description(node.children)
            result.append(entry)
        return result

    @staticmethod
    def _strip_text(nodes: list[IndexNode]) -> None:
        """Remove ``text`` from all nodes in the tree.

        Args:
            nodes: Tree nodes to strip.
        """
        for node in nodes:
            node.text = None
            if node.children:
                PolicyIndex._strip_text(node.children)
