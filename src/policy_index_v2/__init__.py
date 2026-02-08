"""policy_index_v2 -- Production-quality hierarchical PDF indexing.

This package provides a clean, async-first API for extracting hierarchical
structure from PDF documents using LLM-powered analysis via pydantic-ai.

Quick Start:
    >>> import asyncio
    >>> from src.policy_index_v2 import PolicyIndex, IndexConfig
    >>>
    >>> async def main():
    ...     pi = PolicyIndex(IndexConfig(model="gpt-4.1-mini"))
    ...     doc = await pi.get_or_create("data/HO3_sample.pdf")
    ...     print(pi.list_documents())
    ...     print(pi.tree())
    >>>
    >>> asyncio.run(main())

Public API:
    - :class:`PolicyIndex` -- Top-level manager (get_or_create, list, tree, get_node)
    - :class:`IndexConfig` -- Pipeline configuration
    - :class:`IndexNode` -- Tree node model
    - :class:`DocumentIndex` -- Per-document index result
    - :class:`PageContent` -- Extracted page data
"""

from .index import PolicyIndex
from .models import DocumentIndex, IndexConfig, IndexNode, PageContent

__all__ = [
    "PolicyIndex",
    "IndexConfig",
    "IndexNode",
    "DocumentIndex",
    "PageContent",
]
