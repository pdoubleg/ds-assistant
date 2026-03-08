"""DuckDB-backed persistence for Doc Lens documents, assets, and embeddings.

This module centralizes the local relational storage used by Doc Lens during
document ingestion and retrieval. The database tracks uploaded documents,
deduplicated image assets, per-session asset references, and cached embedding
vectors so later search requests can reuse previously computed results.

The `DuckDBStore` class intentionally exposes a small set of idempotent helper
methods around parameterized SQL. Callers can upsert document metadata, insert
new assets and embeddings, inspect what already exists, and execute retrieval
queries for UI display or vector similarity search.

SQL Overview:
    Schema SQL:
        - `documents` stores one row per `(session_id, document_id)` with the
          document name, content hash, source path, and page count.
        - `image_assets` stores deduplicated rendered or extracted image assets
          keyed by `asset_hash`.
        - `embeddings_cache` stores one embedding vector per
          `(asset_hash, model_key)` pair using DuckDB fixed-size float arrays.
        - `session_assets` links a session and document page to an asset hash,
          preserving extraction metadata, display ordering, and optional
          normalized bounding box coordinates.
        - Secondary indexes speed up the most common lookup patterns by
          `session_id`, `(session_id, document_id)`, and `model_key`.

    Mutation SQL:
        - `INSERT OR REPLACE` is used for `documents` so refreshed metadata can
          overwrite stale rows for the same session/document pair.
        - `INSERT OR IGNORE` is used for assets, session links, and embeddings
          so repeated ingestion passes remain idempotent.
        - Bulk insert helpers reuse the same SQL shapes through
          `executemany(...)` to minimize Python-side loop overhead.

    Read SQL:
        - Existence checks use `SELECT 1 ... LIMIT 1` for efficient probes.
        - Set-based lookups such as `get_existing_asset_hashes(...)` and
          `get_asset_paths(...)` build placeholder lists for `IN (...)` clauses
          while still binding values as parameters.
        - `get_missing_embedding_asset_hashes(...)` uses a `LEFT JOIN` from
          `session_assets` to `embeddings_cache` to find assets that do not yet
          have an embedding for the requested model.
        - `get_session_summary(...)` uses scalar subqueries to return counts for
          documents, distinct assets, and cached embeddings in one round trip.
        - `list_document_assets(...)` uses a `best_source` CTE plus
          `ROW_NUMBER()` to collapse duplicate asset references down to a single
          preferred row, prioritizing photos and stable page/ordinal ordering.
        - `query_session(...)` uses three CTEs:
            1. `scored_assets` computes cosine similarity between the query
               vector and each cached embedding with
               `array_cosine_similarity(...)`.
            2. `ranked` assigns a stable rank ordered by score descending.
            3. `best_source` joins ranked assets back to document and image
               metadata, again choosing one preferred representative row per
               asset hash for the final response.
"""

from pathlib import Path
from threading import RLock
from typing import Any

import duckdb
import numpy as np


class DuckDBStore:
    """Thread-safe DuckDB access layer for Doc Lens persistence.

    The store wraps one shared DuckDB connection and serializes cursor usage
    with an `RLock` because ingestion and retrieval may happen concurrently in
    async or threaded service paths.
    """

    def __init__(self, db_path: str | Path, embedding_dim: int):
        """Initialize the store and ensure the schema exists.

        Args:
            db_path: File path for the DuckDB database.
            embedding_dim: Expected embedding vector length for cached rows.
        """
        self.db_path = str(db_path)
        self.embedding_dim = embedding_dim
        self.conn = duckdb.connect(self.db_path)
        # Guard DB operations when ingest/query run concurrently in async/threaded
        # service paths.
        self._lock = RLock()
        self._init_schema()

    def _execute(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyConnection:
        """Execute a SQL statement with a thread-local cursor under lock."""
        with self._lock:
            cursor = self.conn.cursor()
            return cursor.execute(sql, params or [])

    def _executemany(self, sql: str, params: list[list[Any]]) -> None:
        """Execute many rows for one statement under lock."""
        if not params:
            return
        with self._lock:
            cursor = self.conn.cursor()
            cursor.executemany(sql, params)

    def _init_schema(self) -> None:
        """Create tables and indexes required by Doc Lens if missing."""
        dim = self.embedding_dim

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                session_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                document_hash TEXT NOT NULL,
                source_path TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                PRIMARY KEY (session_id, document_id)
            )
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS image_assets (
                asset_hash TEXT PRIMARY KEY,
                asset_type TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                image_path TEXT NOT NULL
            )
            """
        )

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                asset_hash TEXT NOT NULL,
                model_key TEXT NOT NULL,
                embedding FLOAT[{dim}] NOT NULL,
                PRIMARY KEY (asset_hash, model_key)
            )
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS session_assets (
                row_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                asset_hash TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                extraction_method TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                bbox_x0 DOUBLE,
                bbox_y0 DOUBLE,
                bbox_x1 DOUBLE,
                bbox_y1 DOUBLE
            )
            """
        )

        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id)"
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_session_assets_session ON session_assets(session_id)"
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_session_assets_doc ON session_assets(session_id, document_id)"
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_session_assets_composite ON session_assets(session_id, document_id, page_number, asset_hash, extraction_method, ordinal)"
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model_key ON embeddings_cache(model_key)"
        )

    def upsert_document(
        self,
        session_id: str,
        document_id: str,
        document_name: str,
        document_hash: str,
        source_path: str,
        page_count: int,
    ) -> None:
        """Insert or replace document metadata for a session.

        Args:
            session_id: Active Doc Lens session identifier.
            document_id: Stable identifier for the uploaded document.
            document_name: Human-readable document filename or label.
            document_hash: Content hash used to detect duplicate documents.
            source_path: Local filesystem path for the source document.
            page_count: Total number of pages available for the document.
        """
        self._execute(
            """
            INSERT OR REPLACE INTO documents
            (session_id, document_id, document_name, document_hash, source_path, page_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [session_id, document_id, document_name, document_hash, source_path, page_count],
        )

    def asset_exists(self, asset_hash: str) -> bool:
        """Return whether an image asset already exists in the cache.

        Args:
            asset_hash: Content hash that uniquely identifies an asset image.

        Returns:
            True if the asset is already present, else False.
        """
        row = self._execute(
            "SELECT 1 FROM image_assets WHERE asset_hash = ? LIMIT 1",
            [asset_hash],
        ).fetchone()
        return row is not None

    def insert_asset(
        self,
        asset_hash: str,
        asset_type: str,
        width: int,
        height: int,
        image_path: str,
    ) -> None:
        """Insert one deduplicated image asset if it is new.

        Args:
            asset_hash: Content hash that uniquely identifies an asset image.
            asset_type: Logical asset category such as `photo` or `page`.
            width: Asset width in pixels.
            height: Asset height in pixels.
            image_path: Local path to the rendered or extracted image file.
        """
        self._execute(
            """
            INSERT OR IGNORE INTO image_assets
            (asset_hash, asset_type, width, height, image_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [asset_hash, asset_type, width, height, image_path],
        )

    def bulk_insert_assets(self, rows: list[tuple[str, str, int, int, str]]) -> None:
        """Bulk insert image assets with `INSERT OR IGNORE` semantics."""
        self._executemany(
            """
            INSERT OR IGNORE INTO image_assets
            (asset_hash, asset_type, width, height, image_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [list(row) for row in rows],
        )

    def get_existing_asset_hashes(self, asset_hashes: list[str]) -> set[str]:
        """Return the subset of `asset_hashes` already present in image_assets."""
        if not asset_hashes:
            return set()
        placeholders = ",".join(["?"] * len(asset_hashes))
        rows = self._execute(
            f"""
            SELECT asset_hash
            FROM image_assets
            WHERE asset_hash IN ({placeholders})
            """,
            asset_hashes,
        ).fetchall()
        return {row[0] for row in rows}

    def session_asset_exists(
        self,
        session_id: str,
        document_id: str,
        page_number: int,
        asset_hash: str,
        extraction_method: str,
        ordinal: int,
    ) -> bool:
        """Return whether a session-to-asset link already exists.

        Args:
            session_id: Active Doc Lens session identifier.
            document_id: Document identifier from the `documents` table.
            page_number: One-based page number containing the asset.
            asset_hash: Content hash for the linked asset.
            extraction_method: Name of the extraction pipeline that produced it.
            ordinal: Stable per-page ordering value for repeated assets.

        Returns:
            True if the link row is already stored, else False.
        """
        row = self._execute(
            """
            SELECT 1
            FROM session_assets
            WHERE session_id = ?
              AND document_id = ?
              AND page_number = ?
              AND asset_hash = ?
              AND extraction_method = ?
              AND ordinal = ?
            LIMIT 1
            """,
            [session_id, document_id, page_number, asset_hash, extraction_method, ordinal],
        ).fetchone()
        return row is not None

    def insert_session_asset(
        self,
        row_id: str,
        session_id: str,
        document_id: str,
        page_number: int,
        asset_hash: str,
        asset_type: str,
        extraction_method: str,
        ordinal: int,
        bbox_norm: tuple[float, float, float, float] | None,
    ) -> None:
        """Insert one session asset link with an optional normalized bounding box.

        Args:
            row_id: Unique primary key for the session asset row.
            session_id: Active Doc Lens session identifier.
            document_id: Document identifier from the `documents` table.
            page_number: One-based page number containing the asset.
            asset_hash: Content hash for the linked asset.
            asset_type: Logical asset category such as `photo` or `page`.
            extraction_method: Name of the extraction pipeline that produced it.
            ordinal: Stable per-page ordering value for repeated assets.
            bbox_norm: Optional normalized `(x0, y0, x1, y1)` coordinates.
        """
        x0, y0, x1, y1 = bbox_norm if bbox_norm is not None else (None, None, None, None)
        self._execute(
            """
            INSERT OR IGNORE INTO session_assets
            (row_id, session_id, document_id, page_number, asset_hash, asset_type, extraction_method, ordinal,
             bbox_x0, bbox_y0, bbox_x1, bbox_y1)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row_id,
                session_id,
                document_id,
                page_number,
                asset_hash,
                asset_type,
                extraction_method,
                ordinal,
                x0,
                y0,
                x1,
                y1,
            ],
        )

    def bulk_insert_session_assets(
        self,
        rows: list[
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
        ],
    ) -> None:
        """Bulk insert session asset links with normalized bounding boxes."""
        self._executemany(
            """
            INSERT OR IGNORE INTO session_assets
            (row_id, session_id, document_id, page_number, asset_hash, asset_type, extraction_method, ordinal,
             bbox_x0, bbox_y0, bbox_x1, bbox_y1)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [list(row) for row in rows],
        )

    def embedding_exists(self, asset_hash: str, model_key: str) -> bool:
        """Return whether an embedding is cached for an asset/model pair.

        Args:
            asset_hash: Content hash for the asset image.
            model_key: Embedding model identifier used for the cached vector.

        Returns:
            True if the embedding row exists, else False.
        """
        row = self._execute(
            """
            SELECT 1
            FROM embeddings_cache
            WHERE asset_hash = ? AND model_key = ?
            LIMIT 1
            """,
            [asset_hash, model_key],
        ).fetchone()
        return row is not None

    def insert_embedding(self, asset_hash: str, model_key: str, embedding: np.ndarray) -> None:
        """Insert one embedding vector if the asset/model pair is new.

        Args:
            asset_hash: Content hash for the asset image.
            model_key: Embedding model identifier used for the cached vector.
            embedding: Dense embedding vector for the asset.

        Raises:
            ValueError: If `embedding` does not match `self.embedding_dim`.
        """
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (self.embedding_dim,):
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {vec.shape}")
        self._execute(
            """
            INSERT OR IGNORE INTO embeddings_cache
            (asset_hash, model_key, embedding)
            VALUES (?, ?, ?)
            """,
            [asset_hash, model_key, vec],
        )

    def get_missing_embedding_asset_hashes(self, session_id: str, model_key: str) -> list[str]:
        """List session asset hashes missing embeddings for one model.

        Args:
            session_id: Active Doc Lens session identifier.
            model_key: Embedding model identifier to check.

        Returns:
            Asset hashes referenced by the session that do not yet have a cached
            embedding for `model_key`.
        """
        rows = self._execute(
            """
            SELECT DISTINCT sa.asset_hash
            FROM session_assets sa
            LEFT JOIN embeddings_cache ec
              ON sa.asset_hash = ec.asset_hash
             AND ec.model_key = ?
            WHERE sa.session_id = ?
              AND ec.asset_hash IS NULL
            ORDER BY sa.asset_hash
            """,
            [model_key, session_id],
        ).fetchall()
        return [row[0] for row in rows]

    def bulk_insert_embeddings(
        self, rows: list[tuple[str, str, np.ndarray]]
    ) -> None:
        """Bulk insert embedding vectors."""
        normalized_rows: list[list[Any]] = []
        for asset_hash, model_key, embedding in rows:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.shape != (self.embedding_dim,):
                raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {vec.shape}")
            normalized_rows.append([asset_hash, model_key, vec])

        self._executemany(
            """
            INSERT OR IGNORE INTO embeddings_cache
            (asset_hash, model_key, embedding)
            VALUES (?, ?, ?)
            """,
            normalized_rows,
        )

    def get_asset_paths(self, asset_hashes: list[str]) -> dict[str, str]:
        """Map asset hashes to local image paths.

        Args:
            asset_hashes: Asset hashes to resolve.

        Returns:
            Dictionary keyed by asset hash with filesystem image paths as values.
            Missing hashes are omitted from the result.
        """
        if not asset_hashes:
            return {}
        placeholders = ",".join(["?"] * len(asset_hashes))
        rows = self._execute(
            f"""
            SELECT asset_hash, image_path
            FROM image_assets
            WHERE asset_hash IN ({placeholders})
            """,
            asset_hashes,
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_session_summary(self, session_id: str) -> dict[str, int | str]:
        """Return high-level counts for one Doc Lens session.

        Args:
            session_id: Active Doc Lens session identifier.

        Returns:
            Dictionary containing the session id plus document, asset, and
            embedding counts.
        """
        row = self._execute(
            """
            SELECT
                (SELECT COUNT(*) FROM documents WHERE session_id = ?) AS document_count,
                (
                    SELECT COUNT(DISTINCT asset_hash)
                    FROM session_assets
                    WHERE session_id = ?
                ) AS asset_count,
                (
                    SELECT COUNT(DISTINCT ec.asset_hash)
                    FROM embeddings_cache ec
                    JOIN session_assets sa
                      ON sa.asset_hash = ec.asset_hash
                    WHERE sa.session_id = ?
                ) AS embedding_count
            """,
            [session_id, session_id, session_id],
        ).fetchone()

        return {
            "session_id": session_id,
            "document_count": int(row[0]),
            "asset_count": int(row[1]),
            "embedding_count": int(row[2]),
        }

    def clear_session(self, session_id: str) -> None:
        """Delete document and session-asset rows for one session.

        Args:
            session_id: Active Doc Lens session identifier.

        Note:
            This intentionally leaves shared `image_assets` and
            `embeddings_cache` rows intact because those tables are deduplicated
            across sessions.
        """
        self._execute("DELETE FROM session_assets WHERE session_id = ?", [session_id])
        self._execute("DELETE FROM documents WHERE session_id = ?", [session_id])

    def list_document_assets(
        self,
        session_id: str,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """Return all unique assets for one document within a session.

        Args:
            session_id: Active Doc Lens session identifier.
            document_id: Document identifier from the `documents` table.

        Returns:
            List[Dict[str, Any]]: One row per unique asset hash, with stable sort
            ordering that prioritizes photos before page renders.
        """
        sql = """
        WITH best_source AS (
            SELECT
                sa.session_id,
                sa.document_id,
                d.document_name,
                sa.page_number,
                sa.asset_hash,
                sa.asset_type,
                sa.extraction_method,
                ia.image_path,
                sa.bbox_x0,
                sa.bbox_y0,
                sa.bbox_x1,
                sa.bbox_y1,
                ROW_NUMBER() OVER (
                    PARTITION BY sa.asset_hash
                    ORDER BY
                        CASE sa.asset_type WHEN 'photo' THEN 0 ELSE 1 END,
                        sa.page_number ASC,
                        sa.ordinal ASC
                ) AS source_rank
            FROM session_assets sa
            JOIN image_assets ia
              ON ia.asset_hash = sa.asset_hash
            JOIN documents d
              ON d.session_id = sa.session_id
             AND d.document_id = sa.document_id
            WHERE sa.session_id = ?
              AND sa.document_id = ?
        )
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY
                    CASE asset_type WHEN 'photo' THEN 0 ELSE 1 END,
                    page_number ASC,
                    asset_hash ASC
            ) AS rank,
            session_id,
            document_id,
            document_name,
            page_number,
            asset_hash,
            asset_type,
            extraction_method,
            image_path,
            bbox_x0,
            bbox_y0,
            bbox_x1,
            bbox_y1
        FROM best_source
        WHERE source_rank = 1
        ORDER BY rank
        """
        rows = self._execute(sql, [session_id, document_id]).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "rank": int(row[0]),
                    "session_id": row[1],
                    "document_id": row[2],
                    "document_name": row[3],
                    "page_number": int(row[4]),
                    "asset_hash": row[5],
                    "asset_type": row[6],
                    "extraction_method": row[7],
                    "image_path": row[8],
                    "bbox_x0": row[9],
                    "bbox_y0": row[10],
                    "bbox_x1": row[11],
                    "bbox_y1": row[12],
                }
            )
        return results

    def query_session(
        self,
        session_id: str,
        model_key: str,
        query_vector: np.ndarray,
        top_k: int,
        asset_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run vector similarity search over one session's cached assets.

        Args:
            session_id: Active Doc Lens session identifier.
            model_key: Embedding model identifier to query against.
            query_vector: Dense embedding vector for the search query.
            top_k: Maximum number of ranked results to return.
            asset_types: Optional asset-type filter such as `["photo"]`.
            document_ids: Optional document-id filter.

        Returns:
            List[Dict[str, Any]]: Ranked result rows enriched with document,
            image path, extraction metadata, and optional bounding box fields.

        Raises:
            ValueError: If `query_vector` does not match `self.embedding_dim`.
        """
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.shape != (self.embedding_dim,):
            raise ValueError(f"Expected query dim {self.embedding_dim}, got {query_vector.shape}")

        filters = ["sa.session_id = ?", "ec.model_key = ?"]
        params: list[Any] = [session_id, model_key]

        if asset_types:
            placeholders = ",".join(["?"] * len(asset_types))
            filters.append(f"sa.asset_type IN ({placeholders})")
            params.extend(asset_types)

        if document_ids:
            placeholders = ",".join(["?"] * len(document_ids))
            filters.append(f"sa.document_id IN ({placeholders})")
            params.extend(document_ids)

        where_clause = " AND ".join(filters)

        sql = f"""
        WITH scored_assets AS (
            SELECT
                sa.asset_hash,
                MAX(array_cosine_similarity(ec.embedding, ?::FLOAT[{self.embedding_dim}])) AS score
            FROM session_assets sa
            JOIN embeddings_cache ec
              ON sa.asset_hash = ec.asset_hash
            WHERE {where_clause}
            GROUP BY sa.asset_hash
        ),
        ranked AS (
            SELECT
                asset_hash,
                score,
                ROW_NUMBER() OVER (ORDER BY score DESC, asset_hash) AS rank
            FROM scored_assets
        ),
        best_source AS (
            SELECT
                r.rank,
                r.score,
                sa.session_id,
                sa.document_id,
                d.document_name,
                sa.page_number,
                sa.asset_hash,
                sa.asset_type,
                sa.extraction_method,
                ia.image_path,
                sa.bbox_x0,
                sa.bbox_y0,
                sa.bbox_x1,
                sa.bbox_y1,
                ROW_NUMBER() OVER (
                    PARTITION BY sa.asset_hash
                    ORDER BY
                        CASE sa.asset_type WHEN 'photo' THEN 0 ELSE 1 END,
                        sa.page_number ASC,
                        sa.ordinal ASC
                ) AS source_rank
            FROM ranked r
            JOIN session_assets sa
              ON sa.asset_hash = r.asset_hash
            JOIN image_assets ia
              ON ia.asset_hash = sa.asset_hash
            JOIN documents d
              ON d.session_id = sa.session_id
             AND d.document_id = sa.document_id
            WHERE sa.session_id = ?
        )
        SELECT
            rank,
            score,
            session_id,
            document_id,
            document_name,
            page_number,
            asset_hash,
            asset_type,
            extraction_method,
            image_path,
            bbox_x0,
            bbox_y0,
            bbox_x1,
            bbox_y1
        FROM best_source
        WHERE source_rank = 1
        ORDER BY rank
        LIMIT ?
        """

        full_params = [query_vector, *params, session_id, top_k]
        rows = self._execute(sql, full_params).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "rank": int(row[0]),
                    "score": float(row[1]),
                    "session_id": row[2],
                    "document_id": row[3],
                    "document_name": row[4],
                    "page_number": int(row[5]),
                    "asset_hash": row[6],
                    "asset_type": row[7],
                    "extraction_method": row[8],
                    "image_path": row[9],
                    "bbox_x0": row[10],
                    "bbox_y0": row[11],
                    "bbox_x1": row[12],
                    "bbox_y1": row[13],
                }
            )
        return results
