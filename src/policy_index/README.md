# policy_index_v2

Hierarchical PDF indexing powered by **pydantic-ai** agents.

`policy_index_v2` extracts a structured, navigable table-of-contents tree from
any PDF using LLM analysis, then exposes it through a vector-db-style API for
listing, browsing, and retrieving individual sections.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python >= 3.11 | Uses `X \| Y` union syntax |
| `CHATGPT_API_KEY` env var | Or `OPENAI_API_KEY` -- loaded via `dotenv` |
| Project dependencies | `uv sync` from the repo root |

---

## Quick Start

```python
import asyncio
from src.policy_index_v2 import PolicyIndex

async def main():
    pi = PolicyIndex()                                  # default config
    doc = await pi.get_or_create("data/HO3_sample.pdf") # ingest + index
    print(pi.tree())                                     # explore the tree

asyncio.run(main())
```

---

## End-to-End Walkthrough

The workflow below ingests the HO3 sample homeowners policy, builds the index,
and demonstrates every public method.

### 1. Create a PolicyIndex and ingest a PDF

```python
import asyncio
from src.policy_index_v2 import PolicyIndex, IndexConfig

async def main():
    # Configure the pipeline (all fields have sensible defaults)
    config = IndexConfig(
        model="gpt-4.1-mini",       # LLM model for all calls
        toc_check_pages=1,           # pages to scan for a TOC
        max_pages_per_node=5,        # split nodes larger than this
        max_tokens_per_node=12_000,  # split nodes exceeding this token count
        add_node_ids=True,           # assign "0000"-style IDs
        add_summaries=True,          # generate per-node LLM summaries
        add_descriptions=True,       # generate a one-sentence doc description
        add_text=True,               # attach full text to every node
        max_concurrent_llm_calls=10, # throttle parallel API requests
    )

    pi = PolicyIndex(config)

    # Ingest -- runs the full async pipeline (TOC detection, structure
    # extraction, verification, summaries, description).
    doc = await pi.get_or_create("data/HO3_sample.pdf")

    print(f"Document:    {doc.doc_name}")
    print(f"Description: {doc.doc_description}")
    print(f"Root nodes:  {len(doc.root_nodes)}")

asyncio.run(main())
```

Output (abbreviated):

```
Document:    HO3_sample.pdf
Description: An HO-3 Special Form homeowners insurance policy detailing ...
Root nodes:  8
```

### 2. List indexed documents

```python
# Works like a vector DB -- list everything in the store
print(pi.list_documents())
```

```markdown
- HO3_sample.pdf
```

### 3. Render a Markdown tree

```python
# Tree for a specific document (or omit args for all documents)
print(pi.tree("HO3_sample.pdf"))
```

```markdown
## HO3_sample.pdf

_An HO-3 Special Form homeowners insurance policy ..._

- **[0000]** Declarations (pp. 1-2)
- **[0001]** Section I - Definitions (pp. 3-5)
  - **[0002]** Coverage A - Dwelling (pp. 3-4)
  - **[0003]** Coverage B - Other Structures (pp. 4-5)
- **[0004]** Section II - Exclusions (pp. 6-10)
  - **[0005]** General Exclusions (pp. 6-8)
  - **[0006]** Specific Exclusions (pp. 8-10)
...
```

### 4. Retrieve nodes by ID

```python
# Single node
node = pi.get_node("0001")
print(f"[{node.node_id}] {node.title}  (pp. {node.start_page}-{node.end_page})")
print(f"Summary: {node.summary}")
print(f"Text length: {len(node.text)} chars")
print(f"Children: {len(node.children)}")
```

```
[0001] Section I - Definitions  (pp. 3-5)
Summary: Defines key terms used throughout the policy including ...
Text length: 4832 chars
Children: 2
```

```python
# Multiple nodes at once
nodes = pi.get_nodes("0002", "0003")
for n in nodes:
    print(f"  [{n.node_id}] {n.title}")
```

```
  [0002] Coverage A - Dwelling
  [0003] Coverage B - Other Structures
```

### 5. Caching -- second call is instant

```python
# The index is cached in memory; calling get_or_create again
# returns the same object with zero LLM calls.
doc_again = await pi.get_or_create("data/HO3_sample.pdf")
assert doc_again is doc  # exact same object
```

### 6. Remove a document from the store

```python
pi.remove("HO3_sample.pdf")
print(pi.list_documents())  # "_No documents indexed._"
```

### 7. Working with the DocumentIndex directly

The `DocumentIndex` and `IndexNode` objects are standard Pydantic models, so
they serialize cleanly:

```python
import json

doc = await pi.get_or_create("data/HO3_sample.pdf")

# Serialize to JSON (excludes text to keep it short)
lightweight = doc.model_dump(exclude={"root_nodes": {"__all__": {"text"}}})
print(json.dumps(lightweight, indent=2)[:500])

# Flatten the tree into a list
from src.policy_index_v2.tree import flatten_nodes

all_nodes = flatten_nodes(doc.root_nodes)
print(f"Total nodes: {len(all_nodes)}")
```

---

## Configuration Reference

All fields on `IndexConfig` are optional and have defaults:

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"gpt-4.1-mini"` | OpenAI model for all LLM calls |
| `toc_check_pages` | `int` | `1` | Leading pages to scan for a TOC |
| `max_pages_per_node` | `int` | `5` | Page threshold for splitting large nodes |
| `max_tokens_per_node` | `int` | `12000` | Token threshold for splitting large nodes |
| `add_node_ids` | `bool` | `True` | Assign zero-padded 4-digit IDs |
| `add_summaries` | `bool` | `True` | Generate per-node LLM summaries |
| `add_descriptions` | `bool` | `True` | Generate a one-sentence doc description |
| `add_text` | `bool` | `True` | Attach extracted text to each node |
| `max_concurrent_llm_calls` | `int` | `10` | Semaphore limit for parallel API calls |

---

## Running Tests

```bash
# Unit tests only (no API key needed, fast)
uv run python -m pytest tests/test_policy_index_v2/ -m "not integration" -v

# Integration tests (requires CHATGPT_API_KEY and data/HO3_sample.pdf)
uv run python -m pytest tests/test_policy_index_v2/ -m integration -v

# Everything
uv run python -m pytest tests/test_policy_index_v2/ -v
```

---

## Architecture

```
src/policy_index_v2/
  __init__.py   Re-exports PolicyIndex, IndexConfig, IndexNode, DocumentIndex, PageContent
  models.py     Pydantic data models and LLM response schemas
  agents.py     pydantic-ai Agent factories (one per LLM task)
  pdf.py        PDF extraction, token counting, page grouping
  tree.py       Pure tree manipulation (flatten, build, render, find)
  index.py      PolicyIndex class -- public API + async pipeline orchestration
```

### Pipeline Flow

```
PDF file
  |
  v
extract_pages()          -- PyPDF2 text extraction + tiktoken counting
  |
  v
_check_toc()             -- Detect TOC pages via toc_detector_agent
  |
  +-- TOC with page numbers  --> _process_toc_with_page_numbers()
  +-- TOC without numbers    --> _process_toc_no_page_numbers()
  +-- No TOC                 --> _process_no_toc()
  |
  v
_verify_toc()            -- Concurrent title-appearance checks
  |
  v
_fix_incorrect_*()       -- Retry loop for misaligned entries
  |
  v
post_process_to_tree()   -- Flat TOC items --> IndexNode tree
  |
  v
_process_large_node()    -- Recursively split oversized nodes
  |
  v
assign_node_ids()        -- Sequential "0000"-style IDs
attach_text_to_nodes()   -- Full text from page ranges
_generate_all_summaries()-- Concurrent per-node summaries
_generate_doc_description()
  |
  v
DocumentIndex            -- Cached in PolicyIndex._store
```
