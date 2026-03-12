"use client";

/**
 * useDocLens — manages Doc Lens session lifecycle, NDJSON ingest streaming,
 * and query execution against the backend.
 *
 * Designed for repeated open/close usage: the session persists so users can
 * re-enter without re-ingesting the same files.
 */

import { useState, useCallback, useRef } from "react";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

// ── Types ──────────────────────────────────────────────────────────────

export type DocLensStatus =
  | "idle"
  | "initializing"
  | "ready"
  | "querying"
  | "error";

export interface IngestFileProgress {
  file_name: string;
  mime_type: string;
  file_index: number;
  total_files: number;
  status: "pending" | "ingesting" | "complete" | "error";
  page_count?: number;
  num_new_assets?: number;
  num_reused_assets?: number;
  num_new_embeddings?: number;
  num_reused_embeddings?: number;
  document_id?: string;
  error?: string;
}

export interface DocLensSessionSummary {
  session_id: string;
  document_count: number;
  asset_count: number;
  embedding_count: number;
}

export interface QueryHit {
  rank: number;
  score: number;
  session_id: string;
  document_id: string;
  document_name: string;
  page_number: number;
  asset_hash: string;
  asset_type: "page" | "photo";
  extraction_method: string;
  image_path: string;
  bbox_norm: { x0: number; y0: number; x1: number; y1: number } | null;
  page_text: string | null;
  text_snippet: string | null;
}

export interface QueryResponse {
  session_id: string;
  query: string;
  search_mode: "image" | "text";
  model_key: string;
  top_k: number;
  hits: QueryHit[];
}

export interface DocumentAssetsResponse {
  session_id: string;
  document_id: string;
  hits: QueryHit[];
}

export interface DocLensQueryParams {
  search_mode: "image" | "text";
  top_k: number;
  asset_types: ("page" | "photo")[] | null;
}

interface DocLensFile {
  file_name: string;
  mime_type: string;
}

// ── Hook ───────────────────────────────────────────────────────────────

export function useDocLens() {
  const [status, setStatus] = useState<DocLensStatus>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [ingestProgress, setIngestProgress] = useState<IngestFileProgress[]>(
    []
  );
  const [sessionSummary, setSessionSummary] =
    useState<DocLensSessionSummary | null>(null);
  const [queryResults, setQueryResults] = useState<QueryHit[]>([]);
  const [lastQuery, setLastQuery] = useState<string>("");
  const [queryParams, setQueryParams] = useState<DocLensQueryParams>({
    search_mode: "image",
    top_k: 10,
    asset_types: null,
  });
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  // ── Start session (NDJSON streaming ingest) ──────────────────────────

  const startSession = useCallback(async (files: DocLensFile[]) => {
    if (files.length === 0) return;

    setStatus("initializing");
    setErrorMessage(null);
    setQueryResults([]);
    setLastQuery("");
    setSessionSummary(null);

    // Build initial progress entries
    const initial: IngestFileProgress[] = files.map((f, i) => ({
      file_name: f.file_name,
      mime_type: f.mime_type,
      file_index: i + 1,
      total_files: files.length,
      status: "pending" as const,
    }));
    setIngestProgress(initial);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const resp = await fetch(`${BACKEND_URL}/doc-lens/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files }),
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        setStatus("error");
        setErrorMessage(`Session init failed: HTTP ${resp.status}`);
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Keep the last potentially incomplete line in the buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          try {
            const msg = JSON.parse(trimmed);

            if (msg.type === "session_created") {
              setSessionId(msg.session_id);
            } else if (msg.type === "ingest_start") {
              setIngestProgress((prev) =>
                prev.map((p) =>
                  p.file_name === msg.file_name
                    ? { ...p, status: "ingesting" as const }
                    : p
                )
              );
            } else if (msg.type === "ingest_complete") {
              setIngestProgress((prev) =>
                prev.map((p) =>
                  p.file_name === msg.file_name
                    ? {
                        ...p,
                        status: "complete" as const,
                        page_count: msg.page_count,
                        num_new_assets: msg.num_new_assets,
                        num_reused_assets: msg.num_reused_assets,
                        num_new_embeddings: msg.num_new_embeddings,
                        num_reused_embeddings: msg.num_reused_embeddings,
                        document_id: msg.document_id,
                      }
                    : p
                )
              );
            } else if (msg.type === "ingest_error") {
              setIngestProgress((prev) =>
                prev.map((p) =>
                  p.file_name === msg.file_name
                    ? { ...p, status: "error" as const, error: msg.error }
                    : p
                )
              );
            } else if (msg.type === "session_ready") {
              setSessionSummary({
                session_id: msg.session_id,
                document_count: msg.document_count,
                asset_count: msg.asset_count,
                embedding_count: msg.embedding_count,
              });
              setStatus("ready");
            } else if (msg.type === "session_error") {
              setStatus("error");
              setErrorMessage(msg.error);
            }
          } catch {
            // skip malformed lines
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setStatus("error");
        setErrorMessage(`Session init failed: ${(err as Error).message}`);
      }
    } finally {
      abortRef.current = null;
    }
  }, []);

  // ── Run query ────────────────────────────────────────────────────────

  const runQuery = useCallback(
    async (queryText: string) => {
      if (!sessionId || !queryText.trim()) return;

      setStatus("querying");
      setErrorMessage(null);
      setLastQuery(queryText);

      try {
        const resp = await fetch(`${BACKEND_URL}/doc-lens/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            query: queryText,
            search_mode: queryParams.search_mode,
            top_k: queryParams.top_k,
            asset_types:
              queryParams.search_mode === "image"
                ? queryParams.asset_types
                : null,
          }),
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          setStatus("ready");
          setErrorMessage(err.error || `Query failed: HTTP ${resp.status}`);
          return;
        }

        const data: QueryResponse = await resp.json();
        setQueryResults(data.hits);
        setStatus("ready");
      } catch (err) {
        setStatus("ready");
        setErrorMessage(`Query failed: ${(err as Error).message}`);
      }
    },
    [sessionId, queryParams]
  );

  // ── Browse all assets from one document ───────────────────────────────
  const fetchDocumentAssets = useCallback(
    async (documentId: string): Promise<QueryHit[]> => {
      if (!sessionId || !documentId) return [];

      setStatus("querying");
      setErrorMessage(null);
      setLastQuery("");

      try {
        const resp = await fetch(`${BACKEND_URL}/doc-lens/document-assets`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            document_id: documentId,
          }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          setStatus("ready");
          setErrorMessage(
            err.error || `Document assets failed: HTTP ${resp.status}`
          );
          return [];
        }

        const data: DocumentAssetsResponse = await resp.json();
        setStatus("ready");
        return data.hits;
      } catch (err) {
        setStatus("ready");
        setErrorMessage(`Document assets failed: ${(err as Error).message}`);
        return [];
      }
    },
    [sessionId]
  );

  // ── Cancel / clear ───────────────────────────────────────────────────

  const cancelSession = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
  }, []);

  const clearSession = useCallback(async () => {
    if (sessionId) {
      try {
        await fetch(`${BACKEND_URL}/doc-lens/session/${sessionId}`, {
          method: "DELETE",
        });
      } catch {
        // best-effort cleanup
      }
    }
    setSessionId(null);
    setStatus("idle");
    setIngestProgress([]);
    setSessionSummary(null);
    setQueryResults([]);
    setLastQuery("");
    setErrorMessage(null);
  }, [sessionId]);

  const clearResults = useCallback(() => {
    setQueryResults([]);
    setLastQuery("");
    setErrorMessage(null);
    setStatus((prev) => (prev === "querying" ? "ready" : prev));
  }, []);

  return {
    status,
    sessionId,
    ingestProgress,
    sessionSummary,
    queryResults,
    lastQuery,
    queryParams,
    errorMessage,
    setQueryParams,
    startSession,
    runQuery,
    fetchDocumentAssets,
    clearResults,
    cancelSession,
    clearSession,
  };
}
