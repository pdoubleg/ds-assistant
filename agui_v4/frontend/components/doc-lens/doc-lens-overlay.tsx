"use client";

/**
 * DocLensOverlay — full-screen focus component for the Doc Lens
 * text-to-image retrieval feature.
 *
 * Layout:
 *   - Left sidebar: session stats + document cards (reused from the
 *     documents pane) + flagged-hits panel.
 *   - Main area: initialization progress **or** query results grid.
 *   - Bottom bar: chat-style query input with parameter controls.
 *
 * Session lifecycle is owned by the parent <DocLensProvider> so this
 * component stays in the DOM between open/close cycles and never loses
 * session state on close.  Only the CSS visibility changes — no re-mount.
 */

import React, { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Send,
  ScanSearch,
  Loader2,
  BookmarkCheck,
  FileText,
  ImageIcon,
  Database,
  AlertCircle,
  CheckCircle2,
  Clock,
  Settings2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  DocumentCard,
  type DocumentCardUiState,
  type DocumentTagData,
  type DocumentSummaryData,
  type DocSearchData,
  type BulkExpandedCommand,
  type CardVariant,
} from "@/components/a2ui/documents";
import { DocumentViewerSheet } from "@/components/document-viewer-sheet";
import { QueryHitCard } from "./query-hit-card";
import { FlaggedHitsInline } from "./flagged-hits-panel";
import { useDocLensContext } from "./doc-lens-context";
import type { IngestFileProgress, QueryHit } from "@/hooks/use-doc-lens";
import type { UploadedDoc } from "@/hooks/use-uploaded-docs";

// ── Props ──────────────────────────────────────────────────────────────

interface DocLensOverlayProps {
  /** Card enrichment data mirrored from the main documents pane. */
  summaries?: Map<string, DocumentSummaryData>;
  searchScores?: Map<string, DocSearchData>;
  agentTags?: Map<string, DocumentTagData[]>;
  getTextContent?: (fileName: string) => string;
  cardVariant?: CardVariant;

  /** Per-card UI expansion state. */
  cardUiByFileName?: Map<string, DocumentCardUiState>;
  bulkExpandedCommand?: BulkExpandedCommand;
  onCardUiStateChange?: (fileName: string, nextState: DocumentCardUiState) => void;

  /** Action handlers forwarded from the documents pane. */
  onToggleHidden?: (fileName: string) => void;
  chatDocNames?: Set<string>;
  onToggleChatContext?: (fileName: string) => void;
  activeTagFilters?: Set<string>;
  onTagClick?: (tag: string) => void;
}

// ── Helpers ────────────────────────────────────────────────────────────

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

function getImageUrl(imagePath: string): string {
  // Normalize Windows backslashes so the marker search works cross-platform.
  const normalized = imagePath.replace(/\\/g, "/");
  const marker = "assets/";
  const idx = normalized.indexOf(marker);
  const relative = idx >= 0 ? normalized.slice(idx + marker.length) : normalized;
  return `${BACKEND_URL}/doc-lens-assets/${relative}`;
}

// ── Status icon for ingest progress rows ───────────────────────────────

function IngestStatusIcon({ status }: { status: IngestFileProgress["status"] }) {
  switch (status) {
    case "pending":
      return <Clock className="h-3.5 w-3.5 text-muted-foreground" />;
    case "ingesting":
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />;
    case "complete":
      return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />;
    case "error":
      return <AlertCircle className="h-3.5 w-3.5 text-destructive" />;
  }
}

// ── Component ──────────────────────────────────────────────────────────

export function DocLensOverlay({
  summaries,
  searchScores,
  agentTags,
  getTextContent,
  cardVariant = "narrow",
  cardUiByFileName,
  bulkExpandedCommand,
  onCardUiStateChange,
  onToggleHidden,
  chatDocNames,
  onToggleChatContext,
  activeTagFilters,
  onTagClick,
}: DocLensOverlayProps) {
  // All session + visibility state comes from the persistent provider.
  const ctx = useDocLensContext();
  const { open, closeLens, docs } = ctx;

  // ── Local UI state (survives open/close because the component stays in DOM) ──
  const [queryText, setQueryText] = useState("");
  const [showFlagged, setShowFlagged] = useState(false);
  const [showParams, setShowParams] = useState(false);
  const [resultsMode, setResultsMode] = useState<"query" | "document">("query");
  const [documentBrowseHits, setDocumentBrowseHits] = useState<QueryHit[]>([]);
  const [browseDocumentId, setBrowseDocumentId] = useState("none");
  const [loadingDocumentAssets, setLoadingDocumentAssets] = useState(false);
  const [postFilters, setPostFilters] = useState<{
    documentName: string;
    assetType: "all" | "page" | "photo";
    extractionMethod: string;
  }>({
    documentName: "all",
    assetType: "all",
    extractionMethod: "all",
  });
  const [previewDoc, setPreviewDoc] = useState<
    (UploadedDoc & { _id: string; _initialPage?: number }) | null
  >(null);
  const [previewQuery, setPreviewQuery] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Stable ref for onCardUiStateChange to avoid infinite re-render loops.
  // DocumentCard has a useEffect whose deps include onUiStateChange identity;
  // an unstable closure would cause setState → re-render → new closure → effect → setState...
  const cardUiChangeRef = useRef(onCardUiStateChange);
  cardUiChangeRef.current = onCardUiStateChange;

  // Cache of per-file stable callbacks so DocumentCard sees a referentially
  // equal onUiStateChange across renders.
  const perFileUiCallbacks = useRef(
    new Map<string, (next: DocumentCardUiState) => void>()
  );
  const getStableUiCallback = useCallback(
    (fileName: string): ((next: DocumentCardUiState) => void) => {
      let cb = perFileUiCallbacks.current.get(fileName);
      if (!cb) {
        cb = (next: DocumentCardUiState) => {
          cardUiChangeRef.current?.(fileName, next);
        };
        perFileUiCallbacks.current.set(fileName, cb);
      }
      return cb;
    },
    []
  );

  // Focus query input when session becomes ready.
  useEffect(() => {
    if (ctx.status === "ready" && open) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [ctx.status, open]);

  // ESC to close.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeLens();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, closeLens]);

  const handleSendQuery = useCallback(() => {
    if (queryText.trim() && ctx.status === "ready") {
      // Query mode and document-browse mode are mutually exclusive in one viewport.
      setResultsMode("query");
      setBrowseDocumentId("none");
      setDocumentBrowseHits([]);
      setPostFilters({
        documentName: "all",
        assetType: "all",
        extractionMethod: "all",
      });
      ctx.runQuery(queryText.trim());
    }
  }, [queryText, ctx]);

  const handleClearResults = useCallback(() => {
    ctx.clearResults();
    setResultsMode("query");
    setBrowseDocumentId("none");
    setDocumentBrowseHits([]);
    setPostFilters({
      documentName: "all",
      assetType: "all",
      extractionMethod: "all",
    });
  }, [ctx]);

  const handleSearchModeChange = useCallback(
    (searchMode: "image" | "text") => {
      if (ctx.queryParams.search_mode === searchMode) return;
      ctx.setQueryParams((prev) => ({
        ...prev,
        search_mode: searchMode,
        asset_types: searchMode === "image" ? prev.asset_types : null,
      }));
      handleClearResults();
    },
    [ctx, handleClearResults]
  );

  const handleBrowseDocument = useCallback(
    async (documentId: string) => {
      if (documentId === "none") {
        setBrowseDocumentId("none");
        setResultsMode("query");
        setDocumentBrowseHits([]);
        return;
      }
      setLoadingDocumentAssets(true);
      try {
        const hits = await ctx.fetchDocumentAssets(documentId);
        setBrowseDocumentId(documentId);
        setResultsMode("document");
        setDocumentBrowseHits(hits);
        setPostFilters({
          documentName: "all",
          assetType: "all",
          extractionMethod: "all",
        });
      } finally {
        setLoadingDocumentAssets(false);
      }
    },
    [ctx]
  );

  const handlePreviewDoc = useCallback(
    (fileName: string, page: number, query?: string) => {
      const doc = docs.find((d) => d.file_name === fileName);
      if (doc) {
        setPreviewQuery(query ?? "");
        setPreviewDoc({
          ...doc,
          _id: doc.file_name,
          _initialPage: page,
        });
      }
    },
    [docs]
  );

  // ── Derived values ────────────────────────────────────────────────────

  const isInitializing = ctx.status === "initializing";
  const isQuerying = ctx.status === "querying";
  const isReady = ctx.status === "ready";
  const isTextMode = ctx.queryParams.search_mode === "text";
  const completedFiles = ctx.ingestProgress.filter(
    (p) => p.status === "complete"
  );
  const totalPages = completedFiles.reduce(
    (s, f) => s + (f.page_count ?? 0),
    0
  );
  const activeHits = resultsMode === "document" ? documentBrowseHits : ctx.queryResults;
  const postFilterOptions = useMemo(() => {
    const documentNames = Array.from(
      new Set(activeHits.map((hit) => hit.document_name))
    ).sort((a, b) => a.localeCompare(b));
    const extractionMethods = Array.from(
      new Set(activeHits.map((hit) => hit.extraction_method))
    ).sort((a, b) => a.localeCompare(b));
    return { documentNames, extractionMethods };
  }, [activeHits]);
  const filteredHits = useMemo(
    () =>
      activeHits.filter((hit) => {
        if (
          postFilters.documentName !== "all" &&
          hit.document_name !== postFilters.documentName
        ) {
          return false;
        }
        if (
          !isTextMode &&
          postFilters.assetType !== "all" &&
          hit.asset_type !== postFilters.assetType
        ) {
          return false;
        }
        if (
          !isTextMode &&
          postFilters.extractionMethod !== "all" &&
          hit.extraction_method !== postFilters.extractionMethod
        ) {
          return false;
        }
        return true;
      }),
    [activeHits, isTextMode, postFilters]
  );
  const completedDocTargets = ctx.ingestProgress.filter(
    (p): p is IngestFileProgress & { document_id: string } =>
      p.status === "complete" && Boolean(p.document_id)
  );
  const isResultsSpaceClear =
    ctx.queryResults.length === 0 &&
    documentBrowseHits.length === 0 &&
    !isQuerying &&
    !loadingDocumentAssets;

  // ── Render ─────────────────────────────────────────────────────────────
  // The overlay is always present in the DOM; visibility is toggled via the
  // `hidden` class so session state is never destroyed on close.

  return (
    <div className={cn(!open && "hidden")}>
      {/* Full-screen overlay with enter/exit animation driven by `open` */}
      <AnimatePresence>
        {open && (
          <motion.div
            key="doc-lens-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-100 flex flex-col bg-background/95 backdrop-blur-md"
          >
            {/* ── Header ──────────────────────────────────────────────── */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b bg-background/80">
              <div className="flex items-center gap-2.5">
                <ScanSearch className="h-5 w-5 text-primary" />
                <h2 className="text-sm font-semibold">Doc Lens</h2>
                {ctx.sessionSummary && (
                  <div className="flex items-center gap-2 ml-2">
                    <Badge variant="secondary" className="text-[10px] gap-1">
                      <FileText className="h-3 w-3" />
                      {ctx.sessionSummary.document_count} docs
                    </Badge>
                    <Badge variant="secondary" className="text-[10px] gap-1">
                      <ImageIcon className="h-3 w-3" />
                      {ctx.sessionSummary.asset_count} images
                    </Badge>
                    <Badge variant="secondary" className="text-[10px] gap-1">
                      <Database className="h-3 w-3" />
                      {ctx.sessionSummary.embedding_count} embeddings
                    </Badge>
                  </div>
                )}
                {isInitializing && (
                  <Badge
                    variant="outline"
                    className="text-[10px] gap-1 animate-pulse"
                  >
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Initializing...
                  </Badge>
                )}
              </div>

              <div className="flex items-center gap-1.5">
                {/* Flagged hits toggle */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant={showFlagged ? "default" : "outline"}
                      size="sm"
                      className="h-7 text-[11px] gap-1"
                      onClick={() => setShowFlagged((p) => !p)}
                    >
                      <BookmarkCheck className="h-3.5 w-3.5" />
                      Saved Images
                      {ctx.flagCount > 0 && (
                        <Badge
                          variant="secondary"
                          className="text-[9px] px-1 py-0 h-4 ml-0.5"
                        >
                          {ctx.flagCount}
                        </Badge>
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="text-xs">
                    Toggle saved images panel
                  </TooltipContent>
                </Tooltip>

                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={closeLens}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* ── Body ────────────────────────────────────────────────── */}
            <div className="flex flex-1 min-h-0">
              {/* ── Left sidebar — matches the Documents pane width (25vw) ── */}
              <div className="w-[25vw] min-w-[320px] max-w-[480px] shrink-0 border-r flex flex-col bg-muted/30">
                {/* Session stats summary */}
                {ctx.sessionSummary && (
                  <div className="px-3 py-2 border-b">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                      Session
                    </p>
                    <div className="grid grid-cols-2 gap-1 text-[11px]">
                      <span className="text-muted-foreground">Documents</span>
                      <span className="font-medium text-right">
                        {ctx.sessionSummary.document_count}
                      </span>
                      <span className="text-muted-foreground">Pages</span>
                      <span className="font-medium text-right">{totalPages}</span>
                      <span className="text-muted-foreground">Images</span>
                      <span className="font-medium text-right">
                        {ctx.sessionSummary.asset_count}
                      </span>
                      <span className="text-muted-foreground">Embeddings</span>
                      <span className="font-medium text-right">
                        {ctx.sessionSummary.embedding_count}
                      </span>
                    </div>
                  </div>
                )}

                {/* Document cards */}
                <div className="flex-1 min-h-0">
                  <ScrollArea className="h-full">
                    <div className="p-2 space-y-1">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground px-1 mb-1">
                        Documents ({docs.length})
                      </p>
                      {docs.map((doc) => {
                        const progress = ctx.ingestProgress.find(
                          (p) => p.file_name === doc.file_name
                        );
                        return (
                          <div key={doc.file_name} className="relative">
                            <DocumentCard
                              file_name={doc.file_name}
                              mime_type={doc.mime_type}
                              content_id={doc.content_id}
                              claim_number={doc.claim_number}
                              content_url={doc.content_url}
                              domain={
                                doc.domain === "claim" || doc.domain === "policy"
                                  ? doc.domain
                                  : undefined
                              }
                              document_type={doc.document_type}
                              document_sub_type={doc.document_sub_type}
                              document_description={doc.document_description}
                              create_date={doc.create_date}
                              source_system={doc.source_system}
                              company_name={doc.company_name}
                              summaryData={summaries?.get(doc.file_name)}
                              searchData={searchScores?.get(doc.file_name)}
                              tags={agentTags?.get(doc.file_name)}
                              variant={cardVariant}
                              isHidden={false}
                              onToggleHidden={
                                onToggleHidden
                                  ? () => onToggleHidden(doc.file_name)
                                  : undefined
                              }
                              isInChatContext={chatDocNames?.has(doc.file_name)}
                              onToggleChatContext={
                                onToggleChatContext
                                  ? () => onToggleChatContext(doc.file_name)
                                  : undefined
                              }
                              onTagClick={onTagClick}
                              activeTagFilters={activeTagFilters}
                              onPreview={() =>
                                handlePreviewDoc(doc.file_name, 1)
                              }
                              initialUiState={cardUiByFileName?.get(doc.file_name)}
                              bulkExpandedCommand={bulkExpandedCommand}
                              onUiStateChange={
                                onCardUiStateChange
                                  ? getStableUiCallback(doc.file_name)
                                  : undefined
                              }
                            />
                            {/* Ingest status icon (top-right corner) */}
                            {progress && (
                              <div className="absolute top-1 right-1">
                                <IngestStatusIcon status={progress.status} />
                              </div>
                            )}
                            {/* Image count pill (bottom-left, Doc Lens only) */}
                            {progress?.status === "complete" && (
                              <div className="px-3 pb-1.5 -mt-1">
                                <Badge
                                  variant="outline"
                                  className="text-[9px] px-1.5 py-0 h-4 gap-0.5 border-amber-500/40 text-amber-600 dark:text-amber-400 bg-amber-500/10"
                                >
                                  <ImageIcon className="h-2.5 w-2.5" />
                                  {(progress.num_new_assets ?? 0) +
                                    (progress.num_reused_assets ?? 0)}{" "}
                                  images
                                </Badge>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </ScrollArea>
                </div>
              </div>

              {/* ── Main content area ─────────────────────────────────── */}
              <div className="flex-1 flex flex-col min-w-0">
                {/* Scrollable results / progress area */}
                <ScrollArea className="flex-1">
                  <div className="p-4">
                    {/* Initialization progress */}
                    {isInitializing && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2 mb-4">
                          <Loader2 className="h-4 w-4 animate-spin text-primary" />
                          <span className="text-sm font-medium">
                            Initializing Doc Lens session...
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {completedFiles.length} / {ctx.ingestProgress.length}{" "}
                            files
                          </span>
                        </div>

                        {/* Overall progress bar */}
                        <div className="h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full bg-primary transition-all duration-500"
                            style={{
                              width: `${ctx.ingestProgress.length > 0 ? (completedFiles.length / ctx.ingestProgress.length) * 100 : 0}%`,
                            }}
                          />
                        </div>

                        {/* Per-file progress */}
                        <div className="space-y-1.5">
                          {ctx.ingestProgress.map((fp) => (
                            <div
                              key={fp.file_name}
                              className="flex items-center gap-2 text-xs px-2 py-1.5 rounded bg-muted/40"
                            >
                              <IngestStatusIcon status={fp.status} />
                              <span className="flex-1 truncate font-medium">
                                {fp.file_name}
                              </span>
                              {fp.status === "complete" && (
                                <span className="text-muted-foreground tabular-nums">
                                  {fp.page_count} pg, {(fp.num_new_assets ?? 0) + (fp.num_reused_assets ?? 0)} img
                                </span>
                              )}
                              {fp.status === "error" && (
                                <span className="text-destructive truncate max-w-[200px]">
                                  {fp.error}
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Error state */}
                    {ctx.status === "error" && (
                      <div className="flex flex-col items-center justify-center py-16 gap-3">
                        <AlertCircle className="h-8 w-8 text-destructive" />
                        <p className="text-sm text-destructive">
                          {ctx.errorMessage || "An error occurred."}
                        </p>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={ctx.retrySession}
                        >
                          Retry
                        </Button>
                      </div>
                    )}

                    {/* Ready state — empty or query results */}
                    {(isReady || isQuerying) && (
                      <>
                        <div className="mb-3 flex flex-wrap items-center gap-2">
                          {resultsMode === "query" && ctx.queryResults.length > 0 && (
                            <>
                              <span className="text-xs text-muted-foreground">
                                {filteredHits.length} / {ctx.queryResults.length} results for
                              </span>
                              <Badge
                                variant="outline"
                                className="text-[10px] h-auto py-1 max-w-full whitespace-normal wrap-break-word leading-snug"
                              >
                                &ldquo;{ctx.lastQuery}&rdquo;
                              </Badge>
                            </>
                          )}
                          {resultsMode === "document" && (
                            <span className="text-xs text-muted-foreground">
                              Showing {filteredHits.length} / {documentBrowseHits.length} images
                              from selected document
                            </span>
                          )}
                        </div>

                        <div className="mb-3 flex flex-wrap items-end gap-2 rounded-lg border bg-muted/20 px-2.5 py-2">
                          <div className="flex items-center gap-1.5">
                            <label className="text-[10px] text-muted-foreground font-medium">
                              Document
                            </label>
                            <Select
                              value={postFilters.documentName}
                              onValueChange={(v) =>
                                setPostFilters((prev) => ({ ...prev, documentName: v }))
                              }
                              disabled={activeHits.length === 0}
                            >
                              <SelectTrigger className="h-7 w-[180px] text-[11px]">
                                <SelectValue placeholder="All docs" />
                              </SelectTrigger>
                              <SelectContent className="z-150">
                                <SelectItem value="all" className="text-[11px]">
                                  All docs
                                </SelectItem>
                                {postFilterOptions.documentNames.map((name) => (
                                  <SelectItem key={name} value={name} className="text-[11px]">
                                    {name}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>

                          {!isTextMode && (
                            <div className="flex items-center gap-1.5">
                              <label className="text-[10px] text-muted-foreground font-medium">
                                Asset type
                              </label>
                              <Select
                                value={postFilters.assetType}
                                onValueChange={(v) =>
                                  setPostFilters((prev) => ({
                                    ...prev,
                                    assetType: v as "all" | "page" | "photo",
                                  }))
                                }
                                disabled={activeHits.length === 0}
                              >
                                <SelectTrigger className="h-7 w-[110px] text-[11px]">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent className="z-150">
                                  <SelectItem value="all" className="text-[11px]">
                                    All
                                  </SelectItem>
                                  <SelectItem value="photo" className="text-[11px]">
                                    Photo
                                  </SelectItem>
                                  <SelectItem value="page" className="text-[11px]">
                                    Page
                                  </SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          )}

                          {!isTextMode && (
                            <div className="flex items-center gap-1.5">
                              <label className="text-[10px] text-muted-foreground font-medium">
                                Extraction
                              </label>
                              <Select
                                value={postFilters.extractionMethod}
                                onValueChange={(v) =>
                                  setPostFilters((prev) => ({ ...prev, extractionMethod: v }))
                                }
                                disabled={activeHits.length === 0}
                              >
                                <SelectTrigger className="h-7 w-[170px] text-[11px]">
                                  <SelectValue placeholder="All methods" />
                                </SelectTrigger>
                                <SelectContent className="z-150">
                                  <SelectItem value="all" className="text-[11px]">
                                    All methods
                                  </SelectItem>
                                  {postFilterOptions.extractionMethods.map((method) => (
                                    <SelectItem key={method} value={method} className="text-[11px]">
                                      {method}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          )}

                          <div className="flex items-center gap-1.5 ml-auto">
                            {!isTextMode && (
                              <>
                                <label className="text-[10px] text-muted-foreground font-medium">
                                  View all images
                                </label>
                                <Select
                                  value={browseDocumentId}
                                  onValueChange={handleBrowseDocument}
                                  disabled={
                                    !isReady ||
                                    !isResultsSpaceClear ||
                                    completedDocTargets.length === 0
                                  }
                                >
                                  <SelectTrigger className="h-7 w-[220px] text-[11px]">
                                    <SelectValue
                                      placeholder={
                                        isResultsSpaceClear
                                          ? "Select document"
                                          : "Clear results to enable"
                                      }
                                    />
                                  </SelectTrigger>
                                  <SelectContent className="z-150">
                                    <SelectItem value="none" className="text-[11px]">
                                      None
                                    </SelectItem>
                                    {completedDocTargets.map((docTarget) => (
                                      <SelectItem
                                        key={docTarget.document_id}
                                        value={docTarget.document_id}
                                        className="text-[11px]"
                                      >
                                        {docTarget.file_name}
                                      </SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </>
                            )}
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 text-[11px]"
                              disabled={activeHits.length === 0}
                              onClick={handleClearResults}
                            >
                              Clear
                            </Button>
                          </div>
                        </div>

                        {(isQuerying || loadingDocumentAssets) && (
                          <div className="flex items-center gap-2 mb-4">
                            <Loader2 className="h-4 w-4 animate-spin text-primary" />
                            <span className="text-sm">
                              {loadingDocumentAssets
                                ? "Loading document images..."
                                : isTextMode
                                  ? "Searching text..."
                                  : "Searching images..."}
                            </span>
                          </div>
                        )}

                        {/* Results grid (masonry / bento-like so tall pages are not cropped) */}
                        {filteredHits.length > 0 && (
                          <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 2xl:columns-5 gap-3 space-y-3">
                            {filteredHits.map((hit) => (
                              <div
                                key={`${hit.asset_hash}-${hit.rank}`}
                                className="break-inside-avoid"
                              >
                                <QueryHitCard
                                  hit={hit}
                                  query={ctx.lastQuery}
                                  imageUrl={getImageUrl(hit.image_path)}
                                  isFlagged={ctx.isFlagged(hit.asset_hash)}
                                  onToggleFlag={() =>
                                    ctx.toggleFlag(hit, ctx.lastQuery)
                                  }
                                  onPreviewDoc={handlePreviewDoc}
                                  onDownload={() =>
                                    ctx.downloadImage(
                                      hit.image_path,
                                      `${hit.document_name}_p${hit.page_number}_${hit.asset_hash.slice(0, 8)}.png`
                                    )
                                  }
                                />
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Empty ready state */}
                        {activeHits.length === 0 && !isQuerying && !loadingDocumentAssets && (
                          <div className="flex flex-col items-center justify-center py-20 gap-3 text-muted-foreground">
                            <ScanSearch className="h-10 w-10 opacity-30" />
                            <p className="text-sm">
                              Ready to search. Enter a query below.
                            </p>
                            <p className="text-xs opacity-60">
                              {isTextMode
                                ? "Search extracted document text and jump directly to matching pages."
                                : "Describe the image you&rsquo;re looking for using natural language."}
                            </p>
                          </div>
                        )}
                        {activeHits.length > 0 &&
                          filteredHits.length === 0 &&
                          !isQuerying &&
                          !loadingDocumentAssets && (
                            <div className="flex flex-col items-center justify-center py-20 gap-3 text-muted-foreground">
                              <ScanSearch className="h-10 w-10 opacity-30" />
                              <p className="text-sm">
                                No {isTextMode ? "text hits" : "images"} match the current post-query filters.
                              </p>
                              <p className="text-xs opacity-60">
                                Adjust filters or click Clear to reset this results view.
                              </p>
                            </div>
                          )}
                      </>
                    )}
                  </div>
                </ScrollArea>

                {/* ── Query input bar ─────────────────────────────────── */}
                <div className="border-t bg-background/80 px-4 py-2.5">
                  {/* Parameter controls (collapsible) */}
                  <AnimatePresence>
                    {showParams && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden mb-2"
                      >
                        <div className="flex items-center gap-3 pb-2">
                          <div className="flex items-center gap-1.5">
                            <label className="text-[10px] text-muted-foreground font-medium">
                              Top K
                            </label>
                            <Select
                              value={String(ctx.queryParams.top_k)}
                              onValueChange={(v) =>
                                ctx.setQueryParams((p) => ({
                                  ...p,
                                  top_k: parseInt(v),
                                }))
                              }
                            >
                              <SelectTrigger className="h-6 w-16 text-[11px]">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="z-150">
                                {[5, 10, 20, 30, 50].map((k) => (
                                  <SelectItem
                                    key={k}
                                    value={String(k)}
                                    className="text-[11px]"
                                  >
                                    {k}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>

                          {!isTextMode && (
                            <div className="flex items-center gap-1.5">
                              <label className="text-[10px] text-muted-foreground font-medium">
                                Asset Types
                              </label>
                              <Select
                                value={
                                  ctx.queryParams.asset_types === null
                                    ? "all"
                                    : ctx.queryParams.asset_types.join(",")
                                }
                                onValueChange={(v) =>
                                  ctx.setQueryParams((p) => ({
                                    ...p,
                                    asset_types:
                                      v === "all"
                                        ? null
                                        : (v.split(",") as ("page" | "photo")[]),
                                  }))
                                }
                              >
                                <SelectTrigger className="h-6 w-24 text-[11px]">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent className="z-150">
                                  <SelectItem value="all" className="text-[11px]">
                                    All
                                  </SelectItem>
                                  <SelectItem value="photo" className="text-[11px]">
                                    Photos only
                                  </SelectItem>
                                  <SelectItem value="page" className="text-[11px]">
                                    Pages only
                                  </SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Input row */}
                  <div className="flex items-center gap-2">
                    <div className="flex items-center rounded-md border p-0.5 shrink-0">
                      <Button
                        variant={isTextMode ? "ghost" : "default"}
                        size="sm"
                        className="h-8 px-2.5 text-[11px]"
                        onClick={() => handleSearchModeChange("image")}
                      >
                        Image
                      </Button>
                      <Button
                        variant={isTextMode ? "default" : "ghost"}
                        size="sm"
                        className="h-8 px-2.5 text-[11px]"
                        onClick={() => handleSearchModeChange("text")}
                      >
                        Text
                      </Button>
                    </div>

                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 shrink-0"
                          onClick={() => setShowParams((p) => !p)}
                        >
                          <Settings2 className="h-3.5 w-3.5" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent side="top" className="text-xs">
                        Query parameters
                      </TooltipContent>
                    </Tooltip>

                    <Input
                      ref={inputRef}
                      value={queryText}
                      onChange={(e) => setQueryText(e.target.value)}
                      placeholder={
                        isTextMode
                          ? 'Search document text, e.g. "repeated seepage" or "mold exclusion"'
                          : 'Describe an image, e.g. "source of water damage" or "roofing damage close-up"'
                      }
                      className="h-8 text-xs flex-1"
                      disabled={!isReady}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSendQuery();
                        }
                      }}
                    />

                    <Button
                      variant="default"
                      size="icon"
                      className="h-8 w-8 shrink-0"
                      onClick={handleSendQuery}
                      disabled={
                        !isReady || !queryText.trim() || isQuerying
                      }
                    >
                      {isQuerying ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Send className="h-3.5 w-3.5" />
                      )}
                    </Button>
                  </div>

                  {ctx.errorMessage && isReady && (
                    <p className="text-[10px] text-destructive mt-1 px-1">
                      {ctx.errorMessage}
                    </p>
                  )}
                </div>
              </div>

              {/* ── Flagged hits sidebar (right) ──────────────────────── */}
              <AnimatePresence>
                {showFlagged && (
                  <motion.div
                    initial={{ width: 0, opacity: 0 }}
                    animate={{ width: 280, opacity: 1 }}
                    exit={{ width: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="border-l bg-muted/30 overflow-hidden shrink-0"
                  >
                    <div className="w-[280px] h-full flex flex-col">
                      <div className="px-3 py-2 border-b flex items-center justify-between">
                        <span className="text-xs font-semibold">Saved Images</span>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => setShowFlagged(false)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                      <ScrollArea className="flex-1">
                        <div className="p-2">
                          <FlaggedHitsInline
                            flaggedHits={ctx.flaggedHits}
                            getImageUrl={getImageUrl}
                            onRemove={ctx.removeFlag}
                            onClearAll={ctx.clearAllFlagged}
                            onDownloadImage={ctx.downloadImage}
                            onPreviewDoc={handlePreviewDoc}
                            isFlagged={ctx.isFlagged}
                            onToggleFlag={ctx.toggleFlag}
                          />
                        </div>
                      </ScrollArea>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Document preview sheet — z-[150] to render above the DocLens overlay (z-100) */}
      <DocumentViewerSheet
        doc={previewDoc}
        open={!!previewDoc}
        onOpenChange={(isOpen) => {
          if (!isOpen) {
            setPreviewDoc(null);
            setPreviewQuery("");
          }
        }}
        initialPage={previewDoc?._initialPage}
        textContent={previewDoc ? getTextContent?.(previewDoc.file_name) : undefined}
        highlightQuery={previewQuery}
        summaryData={previewDoc ? summaries?.get(previewDoc.file_name) : undefined}
        contentClassName="z-[150]"
        overlayClassName="z-[150]"
      />
    </div>
  );
}
