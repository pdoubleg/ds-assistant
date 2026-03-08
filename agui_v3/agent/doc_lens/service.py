"""High-level orchestration for the Doc Lens ingestion and search pipeline.

Doc Lens lets the application ingest PDFs and standalone images, extract or
render visual assets, cache those assets in local storage, generate embeddings,
and query the resulting session-specific corpus with natural-language search.

This module sits above the lower-level extractor, embedder, and DuckDB storage
layers. Its job is to coordinate those components into a small set of
application-facing workflows:
    - accept uploaded files and persist them into the workspace
    - ingest PDFs into document, asset, and embedding records
    - ingest standalone images as single-asset documents
    - run semantic search over a session's cached visual assets
    - expose document browsing and session summary helpers for the UI

The service keeps the operational details of hashing, deduplication, content-
addressed image storage, and embedding cache reuse out of API and frontend
code, so other layers can interact with Doc Lens through typed responses.
"""

from pathlib import Path

from PIL import Image

from .settings import Settings
from .db import DuckDBStore
from .embedder import BaseEmbedder
from .extractor import PDFExtractor
from .models import BBoxNorm, IngestResponse, QueryHit, QueryResponse, SessionSummary
from .utils import (
    image_hash,
    make_document_id,
    make_session_asset_row_id,
    save_image_content_addressed,
    sha256_file,
)


class DocLensService:
    """Coordinate ingestion, storage, embedding, and retrieval for Doc Lens."""

    def __init__(
        self,
        settings: Settings,
        db: DuckDBStore,
        extractor: PDFExtractor,
        embedder: BaseEmbedder,
    ):
        """Construct the service from its storage and ML dependencies.

        Args:
            settings: Filesystem and feature-toggle configuration.
            db: Persistence layer for documents, assets, and embeddings.
            extractor: PDF/image extraction component.
            embedder: Text and image embedding provider.
        """
        self.settings = settings
        self.settings.ensure_dirs()
        self.db = db
        self.extractor = extractor
        self.embedder = embedder

    def store_upload(self, filename: str, data: bytes) -> str:
        """Persist an uploaded file into the configured upload directory.

        Args:
            filename: User-provided filename from the upload request.
            data: Raw file bytes.

        Returns:
            String path to the saved upload on disk.
        """
        safe_name = Path(filename).name
        out_path = self.settings.upload_root / safe_name
        with open(out_path, "wb") as f:
            f.write(data)
        return str(out_path)

    def ingest_pdf(
        self,
        session_id: str,
        document_name: str,
        pdf_path: str | Path,
        always_include_page_assets: bool | None = None,
        enable_segmentation_fallback: bool | None = None,
    ) -> IngestResponse:
        """Ingest a PDF into document, asset, and embedding records.

        The workflow stores document metadata, extracts visual assets from each
        page, optionally includes full-page renders, writes deduplicated image
        files, records per-session asset links, and computes any embeddings not
        already present in the cache for the active embedding model.

        Args:
            session_id: Active Doc Lens session identifier.
            document_name: Human-readable name for the uploaded PDF.
            pdf_path: Path to the PDF file on disk.
            always_include_page_assets: Optional override for storing a full-page
                rendered asset for every page.
            enable_segmentation_fallback: Optional override for enabling page
                segmentation when direct embedded-image extraction yields no
                assets on a page.

        Returns:
            IngestResponse summarizing document identity plus new and reused
            asset and embedding counts.
        """
        always_include_page_assets = (
            self.settings.always_include_page_assets
            if always_include_page_assets is None
            else always_include_page_assets
        )
        enable_segmentation_fallback = (
            self.settings.enable_segmentation_fallback
            if enable_segmentation_fallback is None
            else enable_segmentation_fallback
        )

        pdf_path = Path(pdf_path)
        document_hash = sha256_file(pdf_path)
        document_id = make_document_id(document_hash)

        doc = self.extractor.open_document(pdf_path)
        page_count = len(doc)

        self.db.upsert_document(
            session_id=session_id,
            document_id=document_id,
            document_name=document_name,
            document_hash=document_hash,
            source_path=str(pdf_path),
            page_count=page_count,
        )

        num_new_assets = 0
        num_reused_assets = 0
        observed_asset_hashes: list[str] = []
        pending_assets: dict[str, tuple[str, str, int, int, str]] = {}
        pending_session_assets: list[
            tuple[
                str,
                str,
                str,
                int,
                str,
                str,
                str,
                int,
                float | None,
                float | None,
                float | None,
                float | None,
            ]
        ] = []

        for page_index in range(page_count):
            page = doc.load_page(page_index)
            page_number = page_index + 1

            rendered = None

            def get_rendered_page() -> Image.Image:
                nonlocal rendered
                if rendered is None:
                    # Render lazily: only when page assets or fallback need it.
                    rendered = self.extractor.render_page(page)
                return rendered

            if always_include_page_assets:
                page_image = get_rendered_page()
                page_asset_hash = image_hash(page_image)
                page_asset_path = save_image_content_addressed(
                    page_image,
                    self.settings.asset_root,
                    "pages",
                    page_asset_hash,
                )
                observed_asset_hashes.append(page_asset_hash)
                pending_assets[page_asset_hash] = (
                    page_asset_hash,
                    "page",
                    page_image.width,
                    page_image.height,
                    page_asset_path,
                )

                pending_session_assets.append(
                    (
                        make_session_asset_row_id(
                            session_id=session_id,
                            document_id=document_id,
                            page_number=page_number,
                            asset_hash=page_asset_hash,
                            extraction_method="full_page_render",
                            ordinal=0,
                        ),
                        session_id,
                        document_id,
                        page_number,
                        page_asset_hash,
                        "page",
                        "full_page_render",
                        0,
                        None,
                        None,
                        None,
                        None,
                    )
                )

            extracted_assets = self.extractor.extract_embedded_images(page)
            if not extracted_assets and enable_segmentation_fallback:
                page_image = get_rendered_page()
                extracted_assets = self.extractor.segment_photos_from_page(page_image)

            for asset in extracted_assets:
                asset_hash = image_hash(asset.image)
                asset_path = save_image_content_addressed(
                    asset.image,
                    self.settings.asset_root,
                    "photos",
                    asset_hash,
                )
                observed_asset_hashes.append(asset_hash)
                pending_assets[asset_hash] = (
                    asset_hash,
                    asset.asset_type,
                    asset.image.width,
                    asset.image.height,
                    asset_path,
                )

                x0, y0, x1, y1 = (
                    asset.bbox_norm
                    if asset.bbox_norm is not None
                    else (None, None, None, None)
                )
                pending_session_assets.append(
                    (
                        make_session_asset_row_id(
                            session_id=session_id,
                            document_id=document_id,
                            page_number=page_number,
                            asset_hash=asset_hash,
                            extraction_method=asset.extraction_method,
                            ordinal=asset.ordinal,
                        ),
                        session_id,
                        document_id,
                        page_number,
                        asset_hash,
                        asset.asset_type,
                        asset.extraction_method,
                        asset.ordinal,
                        x0,
                        y0,
                        x1,
                        y1,
                    )
                )

        doc.close()

        unique_asset_hashes = list(pending_assets.keys())
        existing_asset_hashes = self.db.get_existing_asset_hashes(unique_asset_hashes)
        known_present = set(existing_asset_hashes)
        for asset_hash in observed_asset_hashes:
            if asset_hash in known_present:
                num_reused_assets += 1
            else:
                num_new_assets += 1
                known_present.add(asset_hash)

        self.db.bulk_insert_assets(list(pending_assets.values()))
        self.db.bulk_insert_session_assets(pending_session_assets)

        missing_asset_hashes = self.db.get_missing_embedding_asset_hashes(
            session_id, self.embedder.model_key
        )
        asset_paths = self.db.get_asset_paths(missing_asset_hashes)

        num_new_embeddings = 0
        batch_size = 64
        pending_embedding_rows: list[tuple[str, str, object]] = []
        for start in range(0, len(missing_asset_hashes), batch_size):
            batch_hashes = missing_asset_hashes[start : start + batch_size]
            batch_images: list[Image.Image] = []
            for asset_hash in batch_hashes:
                with Image.open(asset_paths[asset_hash]) as image:
                    # Copy out of context manager so files are closed promptly.
                    batch_images.append(image.copy())

            batch_vectors = self.embedder.embed_images(batch_images)
            for asset_hash, vec in zip(batch_hashes, batch_vectors, strict=True):
                pending_embedding_rows.append((asset_hash, self.embedder.model_key, vec))
                num_new_embeddings += 1

        self.db.bulk_insert_embeddings(pending_embedding_rows)

        total_session_assets = self.db.get_session_summary(session_id)["asset_count"]
        num_reused_embeddings = max(0, int(total_session_assets) - num_new_embeddings)

        return IngestResponse(
            session_id=session_id,
            document_id=document_id,
            document_hash=document_hash,
            page_count=page_count,
            num_new_assets=num_new_assets,
            num_reused_assets=num_reused_assets,
            num_new_embeddings=num_new_embeddings,
            num_reused_embeddings=num_reused_embeddings,
        )

    def ingest_image(
        self,
        session_id: str,
        document_name: str,
        image_path: str | Path,
    ) -> IngestResponse:
        """Ingest a standalone image file (jpg, png) as a single-asset document.

        The image is treated as a one-page document whose sole asset is the
        image itself, stored with ``asset_type="photo"`` and
        ``extraction_method="standalone_image"``.

        Args:
            session_id: Active session identifier.
            document_name: Human-readable name (typically the file name).
            image_path: Path to the image file on disk.

        Returns:
            IngestResponse with page_count=1 and asset/embedding counts.

        Example:
            >>> resp = service.ingest_image("sess1", "photo.jpg", "/uploads/photo.jpg")
            >>> resp.page_count
            1
        """
        image_path = Path(image_path)
        file_hash = sha256_file(image_path)
        document_id = make_document_id(file_hash)

        self.db.upsert_document(
            session_id=session_id,
            document_id=document_id,
            document_name=document_name,
            document_hash=file_hash,
            source_path=str(image_path),
            page_count=1,
        )

        with Image.open(image_path) as img:
            img_copy = img.copy()

        asset_hash_val = image_hash(img_copy)
        asset_path = save_image_content_addressed(
            img_copy,
            self.settings.asset_root,
            "photos",
            asset_hash_val,
        )

        # Check whether the asset already exists in the global store
        existing = self.db.get_existing_asset_hashes([asset_hash_val])
        if asset_hash_val in existing:
            num_new_assets, num_reused_assets = 0, 1
        else:
            num_new_assets, num_reused_assets = 1, 0

        self.db.bulk_insert_assets([
            (asset_hash_val, "photo", img_copy.width, img_copy.height, asset_path)
        ])

        row_id = make_session_asset_row_id(
            session_id=session_id,
            document_id=document_id,
            page_number=1,
            asset_hash=asset_hash_val,
            extraction_method="standalone_image",
            ordinal=0,
        )
        self.db.bulk_insert_session_assets([
            (row_id, session_id, document_id, 1, asset_hash_val,
             "photo", "standalone_image", 0, None, None, None, None)
        ])

        # Embed if not already cached
        missing = self.db.get_missing_embedding_asset_hashes(
            session_id, self.embedder.model_key
        )
        num_new_embeddings = 0
        if asset_hash_val in missing:
            vec = self.embedder.embed_image(img_copy)
            self.db.bulk_insert_embeddings([
                (asset_hash_val, self.embedder.model_key, vec)
            ])
            num_new_embeddings = 1

        num_reused_embeddings = max(0, 1 - num_new_embeddings)

        return IngestResponse(
            session_id=session_id,
            document_id=document_id,
            document_hash=file_hash,
            page_count=1,
            num_new_assets=num_new_assets,
            num_reused_assets=num_reused_assets,
            num_new_embeddings=num_new_embeddings,
            num_reused_embeddings=num_reused_embeddings,
        )

    def query(
        self,
        session_id: str,
        query_text: str,
        top_k: int,
        asset_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> QueryResponse:
        """Run semantic search over the assets stored for one session.

        Args:
            session_id: Active Doc Lens session identifier.
            query_text: Natural-language search text to embed.
            top_k: Maximum number of results to return.
            asset_types: Optional asset-type filter such as `["photo"]`.
            document_ids: Optional document-id filter.

        Returns:
            QueryResponse containing ranked hits enriched with document and
            bounding-box metadata for UI rendering.
        """
        query_vector = self.embedder.embed_text(query_text)

        rows = self.db.query_session(
            session_id=session_id,
            model_key=self.embedder.model_key,
            query_vector=query_vector,
            top_k=top_k,
            asset_types=asset_types,
            document_ids=document_ids,
        )

        hits: list[QueryHit] = []
        for row in rows:
            bbox_norm = None
            if all(row[k] is not None for k in ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]):
                bbox_norm = BBoxNorm(
                    x0=float(row["bbox_x0"]),
                    y0=float(row["bbox_y0"]),
                    x1=float(row["bbox_x1"]),
                    y1=float(row["bbox_y1"]),
                )

            hits.append(
                QueryHit(
                    rank=row["rank"],
                    score=row["score"],
                    session_id=row["session_id"],
                    document_id=row["document_id"],
                    document_name=row["document_name"],
                    page_number=row["page_number"],
                    asset_hash=row["asset_hash"],
                    asset_type=row["asset_type"],
                    extraction_method=row["extraction_method"],
                    image_path=row["image_path"],
                    bbox_norm=bbox_norm,
                )
            )

        return QueryResponse(
            session_id=session_id,
            query=query_text,
            model_key=self.embedder.model_key,
            top_k=top_k,
            hits=hits,
        )

    def list_document_assets(
        self,
        session_id: str,
        document_id: str,
    ) -> list[QueryHit]:
        """List all extracted assets for one document in a session.

        Args:
            session_id: Active Doc Lens session identifier.
            document_id: Target document identifier to browse.

        Returns:
            list[QueryHit]: QueryHit-shaped entries so the frontend can reuse the
            existing result card renderer.

        Example:
            >>> hits = service.list_document_assets("session-123", "doc-456")
            >>> len(hits) >= 0
            True
        """
        rows = self.db.list_document_assets(
            session_id=session_id,
            document_id=document_id,
        )
        hits: list[QueryHit] = []
        for row in rows:
            bbox_norm = None
            if all(row[k] is not None for k in ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]):
                bbox_norm = BBoxNorm(
                    x0=float(row["bbox_x0"]),
                    y0=float(row["bbox_y0"]),
                    x1=float(row["bbox_x1"]),
                    y1=float(row["bbox_y1"]),
                )
            # We keep QueryHit shape for UI reuse; score is synthetic in browse mode.
            hits.append(
                QueryHit(
                    rank=row["rank"],
                    score=1.0,
                    session_id=row["session_id"],
                    document_id=row["document_id"],
                    document_name=row["document_name"],
                    page_number=row["page_number"],
                    asset_hash=row["asset_hash"],
                    asset_type=row["asset_type"],
                    extraction_method=row["extraction_method"],
                    image_path=row["image_path"],
                    bbox_norm=bbox_norm,
                )
            )
        return hits

    def get_session_summary(self, session_id: str) -> SessionSummary:
        """Return aggregate counts for one Doc Lens session.

        Args:
            session_id: Active Doc Lens session identifier.

        Returns:
            SessionSummary with document, asset, and embedding counts.
        """
        return SessionSummary(**self.db.get_session_summary(session_id))

    def clear_session(self, session_id: str) -> None:
        """Remove session-scoped document and asset-link records.

        Args:
            session_id: Active Doc Lens session identifier.
        """
        self.db.clear_session(session_id)


# if __name__ == "__main__":
#     settings = Settings()

#     from .embedder import FastEmbedCLIPEmbedder
#     test_pdf_path = Path(__file__).parent.parent / "uploads" / "Appraisal_2021 10 21.pdf"

#     db = DuckDBStore(settings.duckdb_path, embedding_dim=settings.embedding_dim)
#     extractor = PDFExtractor(
#         render_dpi=settings.render_dpi,
#         min_area_ratio=settings.min_area_ratio,
#         max_area_ratio=settings.max_area_ratio,
#         crop_padding_px=settings.crop_padding_px,
#     )
#     embedder = FastEmbedCLIPEmbedder(model_key="clip-vit-b32")
#     service = DocLensService(settings=settings, db=db, extractor=extractor, embedder=embedder)

#     ingest_response = service.ingest_pdf(
#         session_id="test",
#         document_name="Appraisal_2021 10 21.pdf",
#         pdf_path=test_pdf_path,
#     )
#     print(ingest_response)

#     test_query = "water damage"

#     query_response = service.query(
#         session_id="test",
#         query_text=test_query,
#         top_k=10,
#     )
#     print(query_response.model_dump_json(indent=4))

#     session_summary = service.get_session_summary(session_id="test")
#     print(session_summary.model_dump_json(indent=4))

#     service.clear_session(session_id="test")
