"""Lazy factory for the shared Doc Lens service instance."""

import os
from typing import Any

from doc_lens import DocLensService, DuckDBStore, FastEmbedCLIPEmbedder, PDFExtractor, Settings
from doc_lens.db import _is_fatal_duckdb_error

_doc_lens_service: DocLensService | None = None

DOC_LENS_PDF_MIMES = {"application/pdf"}
DOC_LENS_IMAGE_MIMES = {"image/jpeg", "image/png"}
DOC_LENS_ELIGIBLE_MIMES = DOC_LENS_PDF_MIMES | DOC_LENS_IMAGE_MIMES


def get_doc_lens_service() -> DocLensService:
    """Return the shared Doc Lens service singleton.

    Returns:
        Fully initialized `DocLensService` instance.
    """
    global _doc_lens_service
    if _doc_lens_service is not None:
        return _doc_lens_service

    settings = Settings()
    settings.ensure_dirs()

    db = DuckDBStore(settings.duckdb_path, embedding_dim=settings.embedding_dim)
    extractor = PDFExtractor(
        render_dpi=settings.render_dpi,
        min_area_ratio=settings.min_area_ratio,
        max_area_ratio=settings.max_area_ratio,
        crop_padding_px=settings.crop_padding_px,
    )
    embedder = FastEmbedCLIPEmbedder(
        model_key=settings.model_key,
        text_model_name=settings.text_model_name,
        image_model_name=settings.image_model_name,
        cache_dir=str(settings.fastembed_cache_dir),
    )

    _doc_lens_service = DocLensService(
        settings=settings,
        db=db,
        extractor=extractor,
        embedder=embedder,
    )
    return _doc_lens_service


def reset_doc_lens_service_if_fatal(exc: Exception) -> bool:
    """Reset the singleton when a fatal DuckDB error is detected.

    Args:
        exc: Raised exception from Doc Lens processing.

    Returns:
        `True` when the singleton was reset, otherwise `False`.
    """
    global _doc_lens_service
    if _is_fatal_duckdb_error(exc):
        _doc_lens_service = None
        return True
    return False


def get_doc_lens_asset_dir(base_dir: str) -> str:
    """Return and ensure the public Doc Lens asset directory.

    Args:
        base_dir: Backend root directory.

    Returns:
        Absolute asset directory path.
    """
    asset_dir = os.path.join(base_dir, ".cache", "doc_lens_cache", "assets")
    os.makedirs(asset_dir, exist_ok=True)
    return asset_dir
