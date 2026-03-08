from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import tempfile
from pathlib import Path


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def model_key(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> list[np.ndarray]:
        raise NotImplementedError


class FastEmbedCLIPEmbedder(BaseEmbedder):
    """CLIP embedder backed by FastEmbed's ONNX runtime.

    Uses separate text and vision ONNX models. Embeddings are L2-normalized
    internally by FastEmbed, matching the behavior of OpenCLIP's encode_text /
    encode_image with manual normalization.

    Args:
        model_key: Logical key used to identify this embedder instance.
        text_model_name: FastEmbed model name for the text tower.
        image_model_name: FastEmbed model name for the vision tower.
        cache_dir: Optional directory to cache downloaded ONNX models.

    Example:
        >>> embedder = FastEmbedCLIPEmbedder(model_key="clip-vit-b32")
        >>> text_vec = embedder.embed_text("a dog running in a field")
        >>> image_vec = embedder.embed_image(some_pil_image)
        >>> print(text_vec.shape)  # (512,)
    """

    def __init__(
        self,
        model_key: str,
        text_model_name: str = "Qdrant/clip-ViT-B-32-text",
        image_model_name: str = "Qdrant/clip-ViT-B-32-vision",
        cache_dir: str | None = None,
    ):
        from fastembed import TextEmbedding, ImageEmbedding

        self._model_key = model_key

        # Instantiate separate ONNX models for each modality.
        # Models are auto-downloaded to cache_dir (or default ~/.cache/fastembed)
        # on first use.
        self._text_model = TextEmbedding(
            model_name=text_model_name,
            cache_dir=cache_dir,
        )
        self._image_model = ImageEmbedding(
            model_name=image_model_name,
            cache_dir=cache_dir,
        )

        # Probe embedding dimensionality via a warmup pass
        warmup = self.embed_text("warmup")
        self._embedding_dim = int(warmup.shape[0])

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def model_key(self) -> str:
        """Logical identifier for this embedder instance."""
        return self._model_key

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of output embedding vectors."""
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Embedding methods
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalized float32 vector.

        FastEmbed returns a generator; we consume the first (and only) result.

        Args:
            text: The input string to embed.

        Returns:
            A 1-D float32 numpy array of shape (embedding_dim,).

        Example:
            >>> vec = embedder.embed_text("insurance policy")
            >>> vec.shape
            (512,)
        """
        # embed() accepts a list and returns a generator of np.ndarray
        embeddings = list(self._text_model.embed([text]))
        return embeddings[0].astype(np.float32)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a PIL image into a normalized float32 vector.

        FastEmbed's ImageEmbedding.embed() accepts PIL Images directly in
        recent versions. If your installed version only supports file paths,
        the image is serialized to a temporary PNG file as a fallback.

        Args:
            image: A PIL Image (any mode; converted to RGB internally).

        Returns:
            A 1-D float32 numpy array of shape (embedding_dim,).

        Example:
            >>> from PIL import Image
            >>> img = Image.open("photo.jpg")
            >>> vec = embedder.embed_image(img)
            >>> vec.shape
            (512,)
        """
        rgb_image = image if image.mode == "RGB" else image.convert("RGB")

        try:
            # Prefer passing PIL directly (fastembed >= 0.3.0)
            embeddings = list(self._image_model.embed([rgb_image]))
        except TypeError:
            # Older versions only accept file paths — write to a temp PNG
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                rgb_image.save(tmp_path)
            try:
                embeddings = list(self._image_model.embed([str(tmp_path)]))
            finally:
                tmp_path.unlink(missing_ok=True)  # always clean up

        return embeddings[0].astype(np.float32)

    def embed_images(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Embed a batch of images using FastEmbed's vectorized API.

        Args:
            images: Input PIL images.

        Returns:
            List of float32 embedding vectors, one per image.
        """
        if not images:
            return []

        # Avoid allocating a new image object when the mode is already RGB.
        rgb_images = [img if img.mode == "RGB" else img.convert("RGB") for img in images]

        try:
            embeddings = list(self._image_model.embed(rgb_images))
        except TypeError:
            temp_paths: list[Path] = []
            try:
                for rgb_image in rgb_images:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                    rgb_image.save(tmp_path)
                    temp_paths.append(tmp_path)
                embeddings = list(self._image_model.embed([str(p) for p in temp_paths]))
            finally:
                for tmp_path in temp_paths:
                    tmp_path.unlink(missing_ok=True)

        return [vec.astype(np.float32) for vec in embeddings]
