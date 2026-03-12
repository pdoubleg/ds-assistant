"""DocLens is a text-to-image retrieval pipeline that for PDF images."""

from .service import DocLensService
from .db import DuckDBStore
from .embedder import BaseEmbedder, FastEmbedCLIPEmbedder
from .extractor import PDFExtractor
from .settings import Settings
from .models import IngestResponse, QueryRequest, QueryResponse, SearchMode, SessionSummary

__all__ = [
    "DocLensService",
    "Settings",
    "DuckDBStore",
    "BaseEmbedder",
    "FastEmbedCLIPEmbedder",
    "PDFExtractor",
    "QueryRequest",
    "IngestResponse",
    "QueryResponse",
    "SearchMode",
    "SessionSummary",
]
