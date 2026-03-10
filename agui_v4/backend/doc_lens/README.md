# Doc Lens

Doc Lens is the backend image-ingestion and semantic retrieval layer used to turn uploaded PDFs and standalone images into searchable visual assets.

## TL;RD

- PyMuPDF + OpenCV for getting visual units out of documents,
- content hashes for dedupe and cache reuse,
- FastEmbed CLIP on ONNX Runtime for multimodal embeddings,
- DuckDB for local metadata, embedding cache, and vector search,
- and session-scoped querying on top of globally reusable assets.

At a high level it:

1. Stores uploaded files in a local workspace cache.
2. Extracts image-like assets from PDFs or ingests standalone images directly.
3. Deduplicates assets by content hash and stores them once on disk.
4. Generates CLIP-compatible text and image embeddings with FastEmbed on ONNX Runtime.
5. Uses DuckDB to track documents, session-to-asset links, and cached embeddings.
6. Runs natural-language image search per session using cosine similarity in SQL.

## Main Pieces

- `service.py`: orchestration layer for ingest, embed, browse, and query flows.
- `extractor.py`: PDF rendering plus visual asset extraction.
- `embedder.py`: embedding abstraction and FastEmbed CLIP implementation.
- `db.py`: DuckDB schema plus all read/write/query SQL.
- `settings.py`: cache paths, model selection, extraction thresholds, and feature flags.
- `utils.py`: hashing, content-addressed image writes, and deterministic row IDs.

## End-to-End Flow

### PDF ingest

For each uploaded PDF, `DocLensService.ingest_pdf(...)`:

1. Hashes the source PDF and derives a stable `document_id`.
2. Upserts document metadata into DuckDB.
3. Walks each page.
4. Optionally renders and stores a full-page image asset.
5. Extracts photo-like assets from the page.
6. Saves extracted assets under content-addressed paths like `assets/photos/<asset_hash>.png`.
7. Inserts session-level asset references with page number, extraction method, ordinal, and optional normalized bounding box.
8. Finds any assets missing embeddings for the active model key.
9. Batches image embedding generation and caches those vectors in DuckDB.

### Query

For a search request, `DocLensService.query(...)`:

1. Embeds the query text with the text CLIP tower.
2. Filters the current session's cached image embeddings in DuckDB.
3. Scores assets with `array_cosine_similarity(...)`.
4. Picks the best representative source row per asset hash.
5. Returns ranked hits with document metadata, image path, extraction method, and bounding box info.

## Image Extraction Methods

Doc Lens uses multiple extraction paths because PDFs are inconsistent in how they encode visuals.

### 1. Embedded image extraction

Implemented in `PDFExtractor.extract_embedded_images(...)`.

- Uses PyMuPDF `page.get_image_info(xrefs=True)` and `extract_image(...)`.
- Preserves original embedded raster bytes instead of re-cropping a rendered page.
- Produces the highest-fidelity result when the PDF contains real embedded images.
- Filters candidates by page-area ratio so tiny icons and near-full-page artifacts can be skipped.
- Records normalized bounding boxes from the PDF placement rectangle.
- Tagged as `extraction_method="pdf_embedded_image"`.

### 2. Page segmentation fallback

Implemented in `PDFExtractor.segment_photos_from_page(...)`.

- Used when embedded image extraction finds nothing and fallback is enabled.
- Starts from a rendered page image rather than the PDF object model.
- Runs grayscale conversion, Gaussian blur, adaptive thresholding, morphological closing, contour detection, aspect-ratio filtering, and overlap deduplication.
- Useful for flattened scans, baked-in visuals, or pages where image xrefs are not available or not useful.
- Tagged as `extraction_method="page_segmentation"`.

### 3. Full-page render assets

Used when `always_include_page_assets` is enabled.

- Renders the full page to an RGB image.
- Stores it as an asset of type `page`.
- Gives the system a reliable fallback visual representation even if no photo crop is extracted.
- Tagged as `extraction_method="full_page_render"`.

### How the methods work together

Per page, the pipeline behaves like this:

- Full-page render can be included independently as a page-level asset.
- Embedded-image extraction is tried first for photo assets.
- Page segmentation only runs when no embedded assets were found and fallback is enabled.

That ordering is intentional:

- prefer PDF-native bytes when available,
- avoid unnecessary page rendering work,
- still recover visually useful crops from harder PDFs,
- and keep a page-level asset available for recall-oriented search.

## Efficiency Plays

### DuckDB as the local retrieval engine

DuckDB keeps the whole pipeline simple and fast for local/session-scoped workloads:

- one embedded database file,
- typed fixed-size embedding arrays like `FLOAT[512]`,
- vector scoring directly in SQL,
- window functions and CTEs for ranking/deduping result rows,
- and no extra service to run.

### FastEmbed CLIP on ONNX Runtime

`FastEmbedCLIPEmbedder` uses:

- `Qdrant/clip-ViT-B-32-text` for query text,
- `Qdrant/clip-ViT-B-32-vision` for images,
- FastEmbed as the wrapper,
- ONNX Runtime under the hood for local inference.

Why this is a good fit here:

- text and image land in the same embedding space,
- startup is lightweight after model download,
- inference is local and cacheable,
- and the 512-dim output maps cleanly into the DuckDB schema.

### Cache strategy

Doc Lens has layered caching:

- Uploaded source files are staged under `uploads/`.
- Extracted assets are content-addressed and only written once per hash.
- `image_assets` is global across sessions, so identical visuals are reused.
- `session_assets` stores session-specific references to those shared assets.
- `embeddings_cache` is keyed by `(asset_hash, model_key)`, so embedding work is reused across sessions and across repeated ingests.
- FastEmbed model files are cached separately under `fastembed_models/`.

This means repeated ingestion usually pays only for new pages/assets, not for previously seen visuals or already-embedded images.

### Other practical optimizations

- Page rendering is lazy inside `ingest_pdf(...)`; pages are only rasterized when needed.
- Image embedding is batched in chunks of 64.
- Asset dedupe happens before DB inserts via content hashing and again at insert time via `INSERT OR IGNORE`.
- Session asset row IDs are deterministic, which makes repeated ingestion idempotent without extra lookups.
- `DuckDBStore` reconnects automatically after known fatal/invalidation errors.

## Storage Model

The core schema is:

- `documents`: one row per `(session_id, document_id)`.
- `image_assets`: deduplicated asset metadata keyed by `asset_hash`.
- `session_assets`: page-level links from a session/document/page to an asset hash.
- `embeddings_cache`: one embedding per `(asset_hash, model_key)`.

This split is what allows Doc Lens to reuse assets and embeddings globally while still answering queries in a session-specific context.

## SQL We Use

The SQL is intentionally straightforward and mostly falls into four buckets.

### 1. Schema and indexes

`db.py` creates:

- `documents`,
- `image_assets`,
- `session_assets`,
- `embeddings_cache`,
- plus indexes on `session_id`, `(session_id, document_id)`, composite session asset identity, and `model_key`.

Embeddings are stored as fixed-length arrays:

```sql
CREATE TABLE IF NOT EXISTS embeddings_cache (
    asset_hash TEXT NOT NULL,
    model_key TEXT NOT NULL,
    embedding FLOAT[512] NOT NULL,
    PRIMARY KEY (asset_hash, model_key)
)
```

### 2. Idempotent writes

The ingest pipeline depends heavily on:

- `INSERT OR REPLACE` for document metadata,
- `INSERT OR IGNORE` for assets, session links, and embeddings.

That keeps retries and repeated ingests safe.

### 3. Missing-embedding detection

To avoid recomputing vectors, Doc Lens asks DuckDB which session assets still need embeddings for the active model:

```sql
SELECT DISTINCT sa.asset_hash
FROM session_assets sa
LEFT JOIN embeddings_cache ec
  ON sa.asset_hash = ec.asset_hash
 AND ec.model_key = ?
WHERE sa.session_id = ?
  AND ec.asset_hash IS NULL
ORDER BY sa.asset_hash
```

### 4. Search and result shaping

Search is done in SQL with CTEs:

- `scored_assets`: cosine similarity per asset hash,
- `ranked`: stable rank assignment,
- `best_source`: picks the preferred display row for each asset hash.

The core scoring expression is:

```sql
MAX(array_cosine_similarity(ec.embedding, ?::FLOAT[512])) AS score
```

This lets Doc Lens:

- search only within one session,
- filter by asset type or document,
- rank by semantic similarity,
- and collapse duplicate references to the same underlying asset before returning results.

## Config Knobs

Environment variables use the `DOC_LENS_` prefix. The most relevant ones are:

- `DOC_LENS_CACHE_ROOT`
- `DOC_LENS_DUCKDB_PATH`
- `DOC_LENS_RENDER_DPI`
- `DOC_LENS_MIN_AREA_RATIO`
- `DOC_LENS_MAX_AREA_RATIO`
- `DOC_LENS_CROP_PADDING_PX`
- `DOC_LENS_TEXT_MODEL_NAME`
- `DOC_LENS_IMAGE_MODEL_NAME`
- `DOC_LENS_EMBEDDING_DIM`
- `DOC_LENS_ALWAYS_INCLUDE_PAGE_ASSETS`
- `DOC_LENS_ENABLE_SEGMENTATION_FALLBACK`

## Backend Integration

The backend creates a shared singleton service in `services/doc_lens_factory.py` and exposes it through the Doc Lens API router.

Current API capabilities:

- create a session and ingest files,
- query a session,
- list document assets,
- read session summary,
- clear a session.

