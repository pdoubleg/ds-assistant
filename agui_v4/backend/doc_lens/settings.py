import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application-wide configuration for the doc-lens pipeline.

    All fields can be overridden via environment variables with the
    ``DOC_LENS_`` prefix. Paths are resolved relative to the process
    working directory unless absolute.

    Example:
        >>> settings = Settings()
        >>> settings.model_key
        'fastembed__Qdrant-clip-ViT-B-32-text__Qdrant-clip-ViT-B-32-vision'
    """

    # ------------------------------------------------------------------
    # Storage paths
    # ------------------------------------------------------------------
    cache_root: Path = Field(
        default=Path(os.getenv("DOC_LENS_CACHE_ROOT", ".cache/doc_lens_cache"))
    )
    duckdb_path: Path = Field(
        default=Path(os.getenv("DOC_LENS_DUCKDB_PATH", ".cache/doc_lens_cache/doc_lens.duckdb"))
    )
    asset_dir_name: str = Field(default="assets")
    upload_dir_name: str = Field(default="uploads")

    # Subdirectory under cache_root where FastEmbed downloads ONNX model files.
    # Keeps model weights separate from document assets.
    fastembed_cache_dir_name: str = Field(
        default=os.getenv("DOC_LENS_FASTEMBED_CACHE_DIR_NAME", "fastembed_models")
    )

    # ------------------------------------------------------------------
    # PDF extraction settings
    # ------------------------------------------------------------------
    render_dpi: int = Field(default=int(os.getenv("DOC_LENS_RENDER_DPI", "300")))
    min_area_ratio: float = Field(default=float(os.getenv("DOC_LENS_MIN_AREA_RATIO", "0.05")))
    max_area_ratio: float = Field(default=float(os.getenv("DOC_LENS_MAX_AREA_RATIO", "0.95")))
    crop_padding_px: int = Field(default=int(os.getenv("DOC_LENS_CROP_PADDING_PX", "8")))

    # ------------------------------------------------------------------
    # Embedding settings
    # ------------------------------------------------------------------
    embedding_backend: str = Field(default=os.getenv("DOC_LENS_EMBEDDING_BACKEND", "fastembed"))

    # FastEmbed uses separate ONNX models for text and vision towers.
    # These names are passed directly to fastembed.TextEmbedding /
    # fastembed.ImageEmbedding and trigger auto-download on first use.
    text_model_name: str = Field(
        default=os.getenv("DOC_LENS_TEXT_MODEL_NAME", "Qdrant/clip-ViT-B-32-text")
    )
    image_model_name: str = Field(
        default=os.getenv("DOC_LENS_IMAGE_MODEL_NAME", "Qdrant/clip-ViT-B-32-vision")
    )

    # Expected output dimensionality. Used for schema/index setup before the
    # embedder warmup pass runs. ViT-B/32 produces 512-dim vectors.
    embedding_dim: int = Field(default=int(os.getenv("DOC_LENS_EMBEDDING_DIM", "512")))

    # ------------------------------------------------------------------
    # Pipeline behaviour flags
    # ------------------------------------------------------------------
    always_include_page_assets: bool = Field(
        default=os.getenv("DOC_LENS_ALWAYS_INCLUDE_PAGE_ASSETS", "true").lower() == "true"
    )
    enable_segmentation_fallback: bool = Field(
        default=os.getenv("DOC_LENS_ENABLE_SEGMENTATION_FALLBACK", "true").lower() == "true"
    )

    # ------------------------------------------------------------------
    # Derived paths (computed properties, not stored fields)
    # ------------------------------------------------------------------

    @property
    def asset_root(self) -> Path:
        """Absolute path to the extracted-asset storage directory."""
        return self.cache_root / self.asset_dir_name

    @property
    def upload_root(self) -> Path:
        """Absolute path to the user-upload staging directory."""
        return self.cache_root / self.upload_dir_name

    @property
    def fastembed_cache_dir(self) -> Path:
        """Directory where FastEmbed stores downloaded ONNX model files."""
        return self.cache_root / self.fastembed_cache_dir_name

    @property
    def model_key(self) -> str:
        """Stable cache / index key encoding the full embedding configuration.

        Slashes in FastEmbed model names (e.g. ``"Qdrant/clip-ViT-B-32-text"``)
        are replaced with hyphens so the key is safe to use as a directory name
        or database identifier.

        Returns:
            A string like
            ``'fastembed__Qdrant-clip-ViT-B-32-text__Qdrant-clip-ViT-B-32-vision'``.
        """
        # Sanitize model names: replace path-unsafe characters with hyphens
        safe_text = self.text_model_name.replace("/", "-")
        safe_image = self.image_model_name.replace("/", "-")
        return f"{self.embedding_backend}__{safe_text}__{safe_image}"

    def ensure_dirs(self) -> None:
        """Create all configured cache/data directories if needed.

        Example:
            >>> settings = Settings()
            >>> settings.ensure_dirs()
        """
        # Keep directory setup explicit to avoid side effects at import time.
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.asset_root.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.fastembed_cache_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
