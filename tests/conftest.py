"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def ho3_pdf_path() -> Path:
    """Return the absolute path to the HO3 sample PDF.

    Skips the test if the file is missing.
    """
    path = PROJECT_ROOT / "data" / "HO3_sample.pdf"
    if not path.is_file():
        pytest.skip("HO3_sample.pdf not found in data/")
    return path


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require LLM API access (deselect with '-m \"not integration\"')",
    )
