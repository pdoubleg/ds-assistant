"use client";

/**
 * DocLensContext — lifts session lifecycle state above the overlay so it
 * survives open/close cycles without re-mounting.
 *
 * The provider owns:
 *   - useDocLens()      — backend session, ingest streaming, query execution
 *   - useFlaggedHits()  — localStorage-backed saved-image list
 *   - docLensOpen       — overlay visibility flag
 *   - docLensEligibleDocs — the current set of ingest-eligible documents
 *
 * Session init strategy:
 *   - The session is NOT started on provider mount (lazy — no cost until first open).
 *   - On first open, startSession is called with the current eligible docs.
 *   - On subsequent opens the existing session is reused; startSession is
 *     skipped unless the set of eligible file names has actually changed.
 *   - If the visible doc set changes while the overlay is closed, that change
 *     is only applied the next time the user re-opens Doc Lens.
 *
 * Example usage:
 *   ```tsx
 *   <DocLensProvider docs={docLensEligibleDocs}>
 *     <DocLensOverlay />
 *   </DocLensProvider>
 *   ```
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { useDocLens, type DocLensStatus } from "@/hooks/use-doc-lens";
import { useFlaggedHits } from "@/hooks/use-flagged-hits";
import type { UploadedDoc } from "@/hooks/use-uploaded-docs";
import type {
  IngestFileProgress,
  QueryHit,
  DocLensSessionSummary,
  DocLensQueryParams,
} from "@/hooks/use-doc-lens";
import type { FlaggedHit } from "@/hooks/use-flagged-hits";

// ── Context shape ───────────────────────────────────────────────────────────

export interface DocLensContextValue {
  // ── Overlay visibility ────────────────────────────────────────────────────
  /** Whether the overlay is currently visible. */
  open: boolean;
  /** Open the overlay (triggers a session init if not yet started). */
  openLens: () => void;
  /** Close the overlay without destroying session state. */
  closeLens: () => void;

  // ── Documents ─────────────────────────────────────────────────────────────
  /** Ingest-eligible docs (PDFs + images) passed down from DocumentsPane. */
  docs: UploadedDoc[];

  // ── Session (useDocLens passthrough) ─────────────────────────────────────
  status: DocLensStatus;
  sessionId: string | null;
  ingestProgress: IngestFileProgress[];
  sessionSummary: DocLensSessionSummary | null;
  queryResults: QueryHit[];
  lastQuery: string;
  queryParams: DocLensQueryParams;
  errorMessage: string | null;
  setQueryParams: ReturnType<typeof useDocLens>["setQueryParams"];
  runQuery: ReturnType<typeof useDocLens>["runQuery"];
  fetchDocumentAssets: ReturnType<typeof useDocLens>["fetchDocumentAssets"];
  clearResults: ReturnType<typeof useDocLens>["clearResults"];
  cancelSession: ReturnType<typeof useDocLens>["cancelSession"];
  clearSession: ReturnType<typeof useDocLens>["clearSession"];

  // ── Retry helper (restarts ingest with current docs) ─────────────────────
  retrySession: () => void;

  // ── Flagged hits (useFlaggedHits passthrough) ─────────────────────────────
  flaggedHits: FlaggedHit[];
  flagCount: number;
  isFlagged: ReturnType<typeof useFlaggedHits>["isFlagged"];
  toggleFlag: ReturnType<typeof useFlaggedHits>["toggleFlag"];
  removeFlag: ReturnType<typeof useFlaggedHits>["removeFlag"];
  clearAllFlagged: ReturnType<typeof useFlaggedHits>["clearAll"];
  getImageUrl: ReturnType<typeof useFlaggedHits>["getImageUrl"];
  downloadImage: ReturnType<typeof useFlaggedHits>["downloadImage"];
  exportAllImages: ReturnType<typeof useFlaggedHits>["exportAllImages"];
}

const DocLensContext = createContext<DocLensContextValue | null>(null);

// ── Provider ────────────────────────────────────────────────────────────────

export interface DocLensProviderProps {
  /** Ingest-eligible documents derived from DocumentsPane. */
  docs: UploadedDoc[];
  children: React.ReactNode;
  /**
   * Optional ref that will be populated with the `openLens` callback.
   * Allows a parent component that sits *outside* the provider tree to
   * imperatively trigger the overlay open (e.g. a toolbar button in
   * DocumentsPane whose handler is defined before the JSX return).
   *
   * @example
   * ```tsx
   * const openLensRef = useRef<(() => void) | null>(null);
   * // … later in JSX:
   * <DocLensProvider docs={docs} openLensRef={openLensRef}>
   * // … and in a callback:
   * openLensRef.current?.();
   * ```
   */
  openLensRef?: React.MutableRefObject<(() => void) | null>;
}

/**
 * DocLensProvider — wraps the documents pane to keep Doc Lens session
 * state alive across overlay open/close cycles.
 */
export function DocLensProvider({ docs, children, openLensRef }: DocLensProviderProps) {
  const dl = useDocLens();
  const flagged = useFlaggedHits();

  const [open, setOpen] = useState(false);

  // ── Track whether we have ever opened (lazy init guard) ───────────────────
  const hasOpenedRef = useRef(false);

  // ── Track the last file set sent to the backend ───────────────────────────
  // Stored as a serialized sorted JSON string for cheap equality comparison.
  const lastIngestedKeyRef = useRef<string>("");

  /** Compute a stable sort key for the current eligible doc list. */
  const currentDocKey = useMemo(
    () =>
      JSON.stringify(
        docs
          .map((d) => ({ n: d.file_name, m: d.mime_type }))
          .sort((a, b) => a.n.localeCompare(b.n))
      ),
    [docs]
  );

  /**
   * Start or restart a Doc Lens backend session for a specific doc snapshot.
   *
   * Args:
   *   docList: Visible ingest-eligible documents to index.
   *   docKey: Stable key representing the snapshot being indexed.
   */
  const initiateSession = useCallback(
    async (docList: UploadedDoc[], docKey: string) => {
      // Clear any existing backend session before starting fresh.
      if (dl.sessionId) {
        await dl.clearSession();
      }
      lastIngestedKeyRef.current = docKey;
      const files = docList.map((d) => ({
        file_name: d.file_name,
        mime_type: d.mime_type,
      }));
      dl.startSession(files);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dl.sessionId, dl.clearSession, dl.startSession]
  );

  // ── Effect: lazy init / deferred refresh on open ──────────────────────────
  // While the overlay is closed, docs may change freely without re-ingesting.
  // We only reconcile that new doc set when the user returns to Doc Lens.
  useEffect(() => {
    if (!open) return;
    if (docs.length === 0) return;

    if (!hasOpenedRef.current) {
      // Very first open — start fresh.
      hasOpenedRef.current = true;
      lastIngestedKeyRef.current = currentDocKey;
      const files = docs.map((d) => ({
        file_name: d.file_name,
        mime_type: d.mime_type,
      }));
      dl.startSession(files);
      return;
    }

    if (currentDocKey !== lastIngestedKeyRef.current) {
      // The visible docs changed since the last session snapshot. Refresh now
      // that the user has navigated back into Doc Lens.
      void initiateSession(docs, currentDocKey);
    }

    // Otherwise reuse the existing session exactly as-is so the user can pick
    // up where they left off without losing results or browse state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, docs, currentDocKey, initiateSession, dl.startSession]);

  // ── Public actions ────────────────────────────────────────────────────────

  const openLens = useCallback(() => setOpen(true), []);
  const closeLens = useCallback(() => setOpen(false), []);

  // Populate the optional imperative ref so parent components outside the
  // provider tree can trigger the overlay open (e.g. a toolbar button).
  useEffect(() => {
    if (openLensRef) {
      openLensRef.current = openLens;
    }
  }, [openLens, openLensRef]);

  /** Manually retry after an error (clears session and re-ingests). */
  const retrySession = useCallback(() => {
    void initiateSession(docs, currentDocKey);
  }, [initiateSession, docs, currentDocKey]);

  // ── Assemble context value ─────────────────────────────────────────────────

  const value = useMemo<DocLensContextValue>(
    () => ({
      // Visibility
      open,
      openLens,
      closeLens,

      // Documents
      docs,

      // Session
      status: dl.status,
      sessionId: dl.sessionId,
      ingestProgress: dl.ingestProgress,
      sessionSummary: dl.sessionSummary,
      queryResults: dl.queryResults,
      lastQuery: dl.lastQuery,
      queryParams: dl.queryParams,
      errorMessage: dl.errorMessage,
      setQueryParams: dl.setQueryParams,
      runQuery: dl.runQuery,
      fetchDocumentAssets: dl.fetchDocumentAssets,
      clearResults: dl.clearResults,
      cancelSession: dl.cancelSession,
      clearSession: dl.clearSession,
      retrySession,

      // Flagged hits
      flaggedHits: flagged.flaggedHits,
      flagCount: flagged.flagCount,
      isFlagged: flagged.isFlagged,
      toggleFlag: flagged.toggleFlag,
      removeFlag: flagged.removeFlag,
      clearAllFlagged: flagged.clearAll,
      getImageUrl: flagged.getImageUrl,
      downloadImage: flagged.downloadImage,
      exportAllImages: flagged.exportAllImages,
    }),
    // Spread individual primitives so the memoization is granular.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      open, openLens, closeLens, docs,
      dl.status, dl.sessionId, dl.ingestProgress, dl.sessionSummary,
      dl.queryResults, dl.lastQuery, dl.queryParams, dl.errorMessage,
      dl.setQueryParams, dl.runQuery, dl.fetchDocumentAssets, dl.clearResults,
      dl.cancelSession, dl.clearSession, retrySession,
      flagged.flaggedHits, flagged.flagCount, flagged.isFlagged,
      flagged.toggleFlag, flagged.removeFlag, flagged.clearAll,
      flagged.getImageUrl, flagged.downloadImage, flagged.exportAllImages,
    ]
  );

  return (
    <DocLensContext.Provider value={value}>
      {children}
    </DocLensContext.Provider>
  );
}

// ── Consumer hook ───────────────────────────────────────────────────────────

/**
 * useDocLensContext — consume the nearest DocLensProvider.
 *
 * Throws if called outside of a <DocLensProvider> tree.
 *
 * @example
 * ```tsx
 * const { open, status, runQuery } = useDocLensContext();
 * ```
 */
export function useDocLensContext(): DocLensContextValue {
  const ctx = useContext(DocLensContext);
  if (!ctx) {
    throw new Error(
      "useDocLensContext must be used within a <DocLensProvider>. " +
        "Wrap the component tree with <DocLensProvider docs={...}>."
    );
  }
  return ctx;
}
