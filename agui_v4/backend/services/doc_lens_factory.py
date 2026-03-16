"""Lazy factory for the shared Doc Lens service instance."""

import os
from pathlib import Path

from doc_lens import DocLensService, DuckDBStore, FastEmbedCLIPEmbedder, PDFExtractor, Settings
from doc_lens.db import _is_fatal_duckdb_error
from services.runtime_storage import RuntimeStorageService

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

    backend_dir = Path(__file__).resolve().parent.parent
    doc_lens_cache_root = backend_dir / "tmp" / "doc_lens_cache"
    settings = Settings(
        cache_root=doc_lens_cache_root,
        duckdb_path=doc_lens_cache_root / "doc_lens.duckdb",
    )
    settings.ensure_dirs()
    settings.hydrate_fastembed_runtime()

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


def close_doc_lens_service() -> None:
    """Close the shared Doc Lens singleton if it exists."""
    global _doc_lens_service
    if _doc_lens_service is None:
        return
    _doc_lens_service.close()
    _doc_lens_service = None


def reset_doc_lens_service() -> None:
    """Reset the shared Doc Lens singleton."""
    close_doc_lens_service()


def reset_doc_lens_service_if_fatal(exc: Exception) -> bool:
    """Reset the singleton when a fatal DuckDB error is detected.

    Args:
        exc: Raised exception from Doc Lens processing.

    Returns:
        `True` when the singleton was reset, otherwise `False`.
    """
    if _is_fatal_duckdb_error(exc):
        close_doc_lens_service()
        return True
    return False


def get_doc_lens_asset_dir(
    base_dir: str,
    runtime_storage: RuntimeStorageService | None = None,
) -> str:
    """Return and ensure the public Doc Lens asset directory.

    Args:
        base_dir: Backend root directory.
        runtime_storage: Optional runtime storage helper for temp paths.

    Returns:
        Absolute asset directory path.
    """
    if runtime_storage is not None:
        asset_dir = runtime_storage.doc_lens_cache_dir / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        return str(asset_dir)

    asset_dir = os.path.join(base_dir, "tmp", "doc_lens_cache", "assets")
    os.makedirs(asset_dir, exist_ok=True)
    return asset_dir
