import hashlib
import io
import uuid
from pathlib import Path

from PIL import Image


def normalize_bbox(
    x0: int, y0: int, x1: int, y1: int, width: int, height: int
) -> tuple[float, float, float, float]:
    return (
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def image_to_png_bytes(image: Image.Image) -> bytes:
    normalized = image if image.mode == "RGB" else image.convert("RGB")
    buf = io.BytesIO()
    normalized.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def image_hash(image: Image.Image) -> str:
    """Compute a stable hash from raw RGB pixel buffer.

    Hashing the normalized pixel buffer avoids PNG encoding overhead while still
    producing a deterministic content hash for visual data.
    """
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    hasher = hashlib.sha256()
    # Include image geometry and mode to disambiguate equal raw buffers.
    hasher.update(rgb_image.mode.encode("ascii"))
    hasher.update(rgb_image.width.to_bytes(4, byteorder="little", signed=False))
    hasher.update(rgb_image.height.to_bytes(4, byteorder="little", signed=False))
    hasher.update(rgb_image.tobytes())
    return hasher.hexdigest()


def save_image_content_addressed(
    image: Image.Image, root: str | Path, subdir: str, asset_hash: str
) -> str:
    out_dir = ensure_dir(Path(root) / subdir)
    out_path = out_dir / f"{asset_hash}.png"
    if not out_path.exists():
        normalized = image if image.mode == "RGB" else image.convert("RGB")
        normalized.save(out_path, format="PNG", optimize=False)
    return str(out_path)


def make_document_id(document_hash: str) -> str:
    return f"doc_{document_hash[:24]}"


def make_row_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def make_session_asset_row_id(
    session_id: str,
    document_id: str,
    page_number: int,
    asset_hash: str,
    extraction_method: str,
    ordinal: int,
) -> str:
    """Build a deterministic row id for session_assets dedupe.

    Using a deterministic key allows `INSERT OR IGNORE` to safely avoid
    duplicates without a separate existence lookup.
    """
    key = (
        f"{session_id}|{document_id}|{page_number}|"
        f"{asset_hash}|{extraction_method}|{ordinal}"
    )
    return f"sa_{sha256_bytes(key.encode('utf-8'))[:32]}"
