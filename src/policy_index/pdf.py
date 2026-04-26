"""PDF text extraction and page tokenization utilities.

This module provides clean, typed wrappers around PDF parsing libraries
(PyPDF2 and PyMuPDF) and token counting via ``tiktoken``.

Example:
    >>> from pathlib import Path
    >>> from src.policy_index_v2.pdf import extract_pages, get_pdf_name
    >>> pages = extract_pages(Path("data/HO3_sample.pdf"))
    >>> print(f"{len(pages)} pages, first page has {pages[0].token_count} tokens")
    >>> print(get_pdf_name(Path("data/HO3_sample.pdf")))
    HO3_sample.pdf
"""

from __future__ import annotations

import math
import os
import re
from io import BytesIO
from pathlib import Path

import PyPDF2
import tiktoken

from .models import PageContent


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(text: str, model: str = "gpt-4.1-mini") -> int:
    """Count the number of tokens in *text* using the tokenizer for *model*.

    Args:
        text: The string to tokenize.
        model: OpenAI model name whose tokenizer to use.

    Returns:
        Token count (0 for empty/falsy input).
    """
    if not text:
        return 0
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# PDF page extraction
# ---------------------------------------------------------------------------


def extract_pages(
    source: str | Path | BytesIO,
    model: str = "gpt-4.1-mini",
) -> list[PageContent]:
    """Extract text and token counts from every page in a PDF.

    Args:
        source: File path (``str`` or ``Path``) or an in-memory ``BytesIO``
            stream containing PDF bytes.
        model: OpenAI model name used for token counting.

    Returns:
        Ordered list of :class:`PageContent` instances (one per page).

    Raises:
        ValueError: If *source* is a file path that doesn't exist or isn't a PDF.
    """
    # Validate file-path inputs
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    enc = tiktoken.encoding_for_model(model)
    reader = PyPDF2.PdfReader(source)  # type: ignore[arg-type]

    pages: list[PageContent] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        token_count = len(enc.encode(text))
        pages.append(PageContent(page_number=idx, text=text, token_count=token_count))

    return pages


# ---------------------------------------------------------------------------
# PDF metadata helpers
# ---------------------------------------------------------------------------


def get_pdf_name(source: str | Path | BytesIO) -> str:
    """Derive a human-readable document name from a PDF source.

    For file paths the basename is returned.  For ``BytesIO`` objects the
    PDF metadata title is used (falling back to ``"Untitled"``).

    Args:
        source: PDF file path or in-memory stream.

    Returns:
        Sanitized document name string.
    """
    if isinstance(source, (str, Path)):
        return Path(source).name

    # BytesIO -- attempt to read the title from metadata
    reader = PyPDF2.PdfReader(source)
    meta = reader.metadata
    title = meta.title if meta and meta.title else "Untitled"
    # Sanitize: replace path separators
    return title.replace("/", "-").replace("\\", "-")


def get_text_of_pages(
    pages: list[PageContent],
    start_page: int,
    end_page: int,
    *,
    with_labels: bool = False,
) -> str:
    """Concatenate text from a range of pages.

    Args:
        pages: Full list of extracted pages.
        start_page: 1-based inclusive start.
        end_page: 1-based inclusive end.
        with_labels: If ``True``, wrap each page's text in
            ``<physical_index_X>`` tags (used by LLM prompts).

    Returns:
        Concatenated text string.
    """
    parts: list[str] = []
    for p in pages[start_page - 1 : end_page]:
        if with_labels:
            parts.append(
                f"<physical_index_{p.page_number}>\n{p.text}\n"
                f"<physical_index_{p.page_number}>\n"
            )
        else:
            parts.append(p.text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Page grouping (token-aware chunking)
# ---------------------------------------------------------------------------


def group_pages_by_tokens(
    pages: list[PageContent],
    max_tokens: int = 20_000,
    overlap: int = 1,
    start_index: int = 1,
) -> list[str]:
    """Split pages into token-bounded groups with physical-index tags.

    Each group is a single string of concatenated page texts wrapped in
    ``<physical_index_X>`` tags, suitable for LLM consumption.

    Args:
        pages: Extracted pages to group.
        max_tokens: Soft upper limit on tokens per group.
        overlap: Number of overlapping pages between consecutive groups.
        start_index: The 1-based physical page number for the first page.

    Returns:
        List of grouped text strings.
    """
    if not pages:
        return []

    # Build tagged text + token length for each page
    tagged_texts: list[str] = []
    token_lengths: list[int] = []
    for page in pages:
        idx = (
            page.page_number
            if start_index == 1
            else start_index + (page.page_number - pages[0].page_number)
        )
        tagged = f"<physical_index_{idx}>\n{page.text}\n<physical_index_{idx}>\n\n"
        tagged_texts.append(tagged)
        token_lengths.append(count_tokens(tagged, "gpt-4.1-mini"))

    total_tokens = sum(token_lengths)

    # If everything fits in one chunk, return immediately
    if total_tokens <= max_tokens:
        return ["".join(tagged_texts)]

    # Compute a balanced average target per chunk
    expected_parts = math.ceil(total_tokens / max_tokens)
    target = math.ceil(((total_tokens / expected_parts) + max_tokens) / 2)

    subsets: list[str] = []
    current_parts: list[str] = []
    current_count = 0

    for i, (tagged, tlen) in enumerate(zip(tagged_texts, token_lengths)):
        if current_count + tlen > target and current_parts:
            subsets.append("".join(current_parts))
            # Overlap: start new chunk from a few pages back
            overlap_start = max(i - overlap, 0)
            current_parts = list(tagged_texts[overlap_start:i])
            current_count = sum(token_lengths[overlap_start:i])

        current_parts.append(tagged)
        current_count += tlen

    if current_parts:
        subsets.append("".join(current_parts))

    return subsets


# ---------------------------------------------------------------------------
# Physical-index tag helpers
# ---------------------------------------------------------------------------


def parse_physical_index(value: str | int | None) -> int | None:
    """Convert a ``<physical_index_X>`` tag string to an integer.

    Args:
        value: A string like ``"<physical_index_5>"`` or an int.

    Returns:
        The integer page number, or ``None`` if parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None
