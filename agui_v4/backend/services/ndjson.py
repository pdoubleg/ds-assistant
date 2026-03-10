"""Helpers for NDJSON streaming responses."""

import json
from typing import Any


NDJSON_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def encode_ndjson_line(payload: dict[str, Any]) -> str:
    """Serialize one NDJSON payload line.

    Args:
        payload: JSON-serializable payload dictionary.

    Returns:
        NDJSON-formatted string terminated with a newline.
    """
    return json.dumps(payload) + "\n"
