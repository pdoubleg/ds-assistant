import base64
import io
import os
import tempfile
from urllib.parse import urlparse

from PIL import Image as PILImage


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_url(image_path: str) -> bool:
    """
    Check if the given string is a valid URL.

    Args:
        image_path (str): The string to check.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    try:
        result = urlparse(image_path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def base64_to_image(base64_image: str, base_dir: str = ".") -> str:
    """
    Convert a base64 encoded image to a temporary image file and return its full file path.

    Args:
        base64_image (str): The base64 encoded image string.
        base_dir (str): Base directory to save the temporary image file. Defaults to current directory.

    Returns:
        str: The full file path to the temporary image file.
    """
    image_data = base64.b64decode(base64_image)
    image = PILImage.open(io.BytesIO(image_data))
    if image.mode == "RGBA":
        image = image.convert("RGB")

    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=temp_dir, delete=False, suffix=".jpg"
    ) as tmpfile:
        image_path = tmpfile.name
        image.save(image_path)

    return image_path
