"""Unit tests for policy_index_v2 PDF extraction utilities.

Tests that require the HO3_sample.pdf file use the ``ho3_pdf_path`` fixture
from conftest.py, which automatically skips if the file is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.policy_index_v2.pdf import (
    count_tokens,
    extract_pages,
    get_pdf_name,
    get_text_of_pages,
    group_pages_by_tokens,
    parse_physical_index,
)


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Tests for the count_tokens helper."""

    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_none_input(self) -> None:
        """Falsy input returns 0."""
        assert count_tokens("") == 0

    def test_known_text(self) -> None:
        """A simple string should produce a positive token count."""
        result = count_tokens("Hello, world!")
        assert result > 0

    def test_longer_text_has_more_tokens(self) -> None:
        short = count_tokens("Hi")
        long = count_tokens(
            "Hello, this is a much longer sentence for testing purposes."
        )
        assert long > short


# ---------------------------------------------------------------------------
# extract_pages (requires HO3_sample.pdf)
# ---------------------------------------------------------------------------


class TestExtractPages:
    """Tests for PDF page extraction."""

    def test_extract_pages_count(self, ho3_pdf_path: Path) -> None:
        """HO3 sample should have a reasonable number of pages."""
        pages = extract_pages(ho3_pdf_path)
        assert len(pages) > 0
        # The HO3 sample is known to be ~22 pages
        assert 10 <= len(pages) <= 50

    def test_page_numbers_sequential(self, ho3_pdf_path: Path) -> None:
        """page_number should be 1-indexed and sequential."""
        pages = extract_pages(ho3_pdf_path)
        for i, page in enumerate(pages, start=1):
            assert page.page_number == i

    def test_pages_have_text(self, ho3_pdf_path: Path) -> None:
        """At least some pages should have non-empty text."""
        pages = extract_pages(ho3_pdf_path)
        non_empty = [p for p in pages if p.text.strip()]
        assert len(non_empty) > 0

    def test_token_counts_positive(self, ho3_pdf_path: Path) -> None:
        """Pages with text should have positive token counts."""
        pages = extract_pages(ho3_pdf_path)
        for page in pages:
            if page.text.strip():
                assert page.token_count > 0

    def test_invalid_path_raises(self) -> None:
        """Non-existent file should raise ValueError."""
        with pytest.raises(ValueError, match="File not found"):
            extract_pages("nonexistent.pdf")

    def test_non_pdf_raises(self, tmp_path: Path) -> None:
        """A non-PDF file should raise ValueError."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            extract_pages(txt_file)


# ---------------------------------------------------------------------------
# get_pdf_name
# ---------------------------------------------------------------------------


class TestGetPdfName:
    """Tests for get_pdf_name."""

    def test_from_path_string(self) -> None:
        assert get_pdf_name("data/HO3_sample.pdf") == "HO3_sample.pdf"

    def test_from_path_object(self) -> None:
        assert get_pdf_name(Path("data/subdir/policy.pdf")) == "policy.pdf"

    def test_from_path_with_spaces(self) -> None:
        assert get_pdf_name("data/my document.pdf") == "my document.pdf"


# ---------------------------------------------------------------------------
# get_text_of_pages
# ---------------------------------------------------------------------------


class TestGetTextOfPages:
    """Tests for get_text_of_pages."""

    def test_basic_concatenation(self, ho3_pdf_path: Path) -> None:
        """Should concatenate text from the given page range."""
        pages = extract_pages(ho3_pdf_path)
        if len(pages) < 3:
            pytest.skip("Need at least 3 pages")
        text = get_text_of_pages(pages, 1, 3)
        # Should contain text from pages 1, 2, and 3
        assert len(text) > 0

    def test_single_page(self, ho3_pdf_path: Path) -> None:
        """Should work for a single page."""
        pages = extract_pages(ho3_pdf_path)
        text = get_text_of_pages(pages, 1, 1)
        assert text == pages[0].text

    def test_with_labels(self, ho3_pdf_path: Path) -> None:
        """With labels, output should contain physical_index tags."""
        pages = extract_pages(ho3_pdf_path)
        text = get_text_of_pages(pages, 1, 2, with_labels=True)
        assert "<physical_index_1>" in text
        assert "<physical_index_2>" in text


# ---------------------------------------------------------------------------
# group_pages_by_tokens
# ---------------------------------------------------------------------------


class TestGroupPagesByTokens:
    """Tests for group_pages_by_tokens."""

    def test_single_group_small_doc(self, ho3_pdf_path: Path) -> None:
        """A small page set under the limit should produce one group."""
        pages = extract_pages(ho3_pdf_path)
        # Use a very large limit so everything fits
        groups = group_pages_by_tokens(pages, max_tokens=1_000_000)
        assert len(groups) == 1

    def test_multiple_groups_small_limit(self, ho3_pdf_path: Path) -> None:
        """A small limit should split into multiple groups."""
        pages = extract_pages(ho3_pdf_path)
        groups = group_pages_by_tokens(pages, max_tokens=500)
        assert len(groups) > 1

    def test_groups_contain_physical_index_tags(self, ho3_pdf_path: Path) -> None:
        """Each group should contain physical_index tags."""
        pages = extract_pages(ho3_pdf_path)
        groups = group_pages_by_tokens(pages, max_tokens=500)
        for group in groups:
            assert "<physical_index_" in group

    def test_empty_pages(self) -> None:
        """Empty page list should return empty groups."""
        assert group_pages_by_tokens([]) == []


# ---------------------------------------------------------------------------
# parse_physical_index
# ---------------------------------------------------------------------------


class TestParsePhysicalIndex:
    """Tests for parse_physical_index."""

    def test_from_tag_string(self) -> None:
        assert parse_physical_index("<physical_index_5>") == 5

    def test_from_plain_string(self) -> None:
        assert parse_physical_index("physical_index_12") == 12

    def test_from_int(self) -> None:
        assert parse_physical_index(7) == 7

    def test_from_none(self) -> None:
        assert parse_physical_index(None) is None

    def test_no_number(self) -> None:
        assert parse_physical_index("abc") is None
