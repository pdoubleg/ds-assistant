"""Image extraction helpers for Doc Lens PDF ingestion.

This module is responsible for turning a PDF page into a set of reusable image
assets that later stages can hash, store, embed, and search. The extractor uses
two complementary approaches because PDFs vary widely in how visual content is
encoded.

Extraction Approaches:
    1. Embedded image extraction:
        - `extract_embedded_images(...)` asks PyMuPDF for image xrefs already
          present in the PDF content stream.
        - This is the preferred path when a page contains true embedded raster
          images because it preserves the original image bytes instead of
          re-rendering the page.
        - The method filters out tiny or page-filling candidates using area
          thresholds and records a normalized bounding box from the PDF-reported
          placement rectangle.

    2. Page segmentation:
        - `segment_photos_from_page(...)` operates on a rendered page image
          using OpenCV, rather than PDF internals.
        - This path is useful when the visual content is flattened into the page
          rendering, when image metadata is unavailable, or when a page contains
          photo-like regions that are not exposed as clean embedded-image xrefs.
        - The method applies grayscale conversion, blur, adaptive thresholding,
          morphological closing, contour detection, box filtering, and overlap
          deduplication to isolate likely photo regions.

    3. Full-page rendering:
        - `render_page(...)` produces the raster page image used by segmentation
          and can also serve as a fallback artifact when downstream code needs a
          page-level image representation.

In practice, the embedded-image path yields the highest-fidelity crops when it
works, while segmentation provides a more heuristic fallback for visually rich
pages whose structure is not recoverable directly from the PDF object model.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

from .utils import normalize_bbox


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute IoU between two axis-aligned boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / max(area_a + area_b - inter, 1)


@dataclass
class ExtractedAsset:
    """One extracted visual asset plus metadata needed downstream.

    Attributes:
        asset_type: Logical asset category, typically `photo`.
        extraction_method: Strategy that produced the image crop.
        image: PIL image for the extracted asset.
        bbox_norm: Optional normalized `(x0, y0, x1, y1)` page coordinates.
        ordinal: Stable ordering within a page for deterministic persistence.
    """

    asset_type: str
    extraction_method: str
    image: Image.Image
    bbox_norm: tuple[float, float, float, float] | None
    ordinal: int


class PDFExtractor:
    """Extract image assets from PDF pages using direct and heuristic methods."""

    def __init__(
        self,
        render_dpi: int = 200,
        min_area_ratio: float = 0.10,
        max_area_ratio: float = 0.95,
        crop_padding_px: int = 8,
    ):
        """Configure extraction thresholds and render settings.

        Args:
            render_dpi: Resolution used when rasterizing pages.
            min_area_ratio: Smallest allowed candidate area as a page fraction.
            max_area_ratio: Largest allowed candidate area as a page fraction.
            crop_padding_px: Pixel padding added around segmented crops.
        """
        self.render_dpi = render_dpi
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.crop_padding_px = crop_padding_px

    def open_document(self, pdf_path: str | Path) -> fitz.Document:
        """Open a PDF document with PyMuPDF.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            Open `fitz.Document` instance.
        """
        return fitz.open(str(pdf_path))

    def render_page(self, page: fitz.Page, dpi: int | None = None) -> Image.Image:
        """Render a PDF page into an RGB PIL image.

        Args:
            page: PyMuPDF page object to rasterize.
            dpi: Optional override for the configured render DPI.

        Returns:
            Rendered page image in RGB format.
        """
        dpi = dpi or self.render_dpi
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def extract_embedded_images(self, page: fitz.Page) -> list[ExtractedAsset]:
        """Extract embedded raster images directly from a PDF page.

        This method uses PDF-native image metadata and bytes when available,
        which usually gives better fidelity than cropping a rendered page.

        Args:
            page: PyMuPDF page object to inspect.

        Returns:
            Extracted assets ordered by discovery order on the page.
        """
        results: list[ExtractedAsset] = []
        page_rect = page.rect
        page_width = float(page_rect.width)
        page_height = float(page_rect.height)

        image_infos = page.get_image_info(xrefs=True)
        ordinal = 0

        for info in image_infos:
            xref = info.get("xref", 0)
            bbox = info.get("bbox")

            if not xref or bbox is None:
                continue

            rect = fitz.Rect(bbox)
            bbox_width = max(1.0, float(rect.width))
            bbox_height = max(1.0, float(rect.height))
            area_ratio = (bbox_width * bbox_height) / max(page_width * page_height, 1.0)

            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            extracted = page.parent.extract_image(xref)
            image_bytes = extracted.get("image")
            if not image_bytes:
                continue

            try:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception:
                continue

            bbox_norm = (
                max(0.0, min(1.0, rect.x0 / page_width)),
                max(0.0, min(1.0, rect.y0 / page_height)),
                max(0.0, min(1.0, rect.x1 / page_width)),
                max(0.0, min(1.0, rect.y1 / page_height)),
            )

            results.append(
                ExtractedAsset(
                    asset_type="photo",
                    extraction_method="pdf_embedded_image",
                    image=image,
                    bbox_norm=bbox_norm,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

        return results

    def segment_photos_from_page(self, page_image: Image.Image) -> list[ExtractedAsset]:
        """Heuristically segment photo-like regions from a rendered page image.

        The pipeline uses adaptive thresholding and contour detection to locate
        large rectangular visual regions that likely correspond to photos.

        Args:
            page_image: Rendered page image, typically from `render_page(...)`.

        Returns:
            Extracted assets sorted top-to-bottom then left-to-right.
        """
        rgb = np.array(page_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        page_area = float(w * h)
        boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area_ratio = (cw * ch) / max(page_area, 1.0)

            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            aspect = cw / max(ch, 1)
            if aspect < 0.2 or aspect > 5.0:
                continue

            pad = self.crop_padding_px
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + cw + pad)
            y1 = min(h, y + ch + pad)
            boxes.append((x0, y0, x1, y1))

        boxes = self._dedupe_boxes(boxes)
        boxes.sort(key=lambda b: (b[1], b[0]))

        assets: list[ExtractedAsset] = []
        for idx, (x0, y0, x1, y1) in enumerate(boxes):
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop_img = Image.fromarray(crop)
            bbox_norm = normalize_bbox(x0, y0, x1, y1, w, h)
            assets.append(
                ExtractedAsset(
                    asset_type="photo",
                    extraction_method="page_segmentation",
                    image=crop_img,
                    bbox_norm=bbox_norm,
                    ordinal=idx,
                )
            )

        return assets

    def _dedupe_boxes(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Remove heavily overlapping candidate boxes.

        Args:
            boxes: Candidate `(x0, y0, x1, y1)` pixel boxes.

        Returns:
            Reduced list of boxes after suppressing near-duplicates.
        """
        # In-place sort avoids a copy; O(n^2) overlap checks are fine for
        # expected contour counts per page.
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if any(_box_iou(box, existing) > 0.85 for existing in kept):
                continue
            kept.append(box)

        return kept
