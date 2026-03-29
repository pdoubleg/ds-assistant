"use client";

/**
 * DocumentsPane — center pane for document triage, filtering, and context.
 *
 * Layout (top to bottom):
 *   1. Header with doc count + visible/hidden badges
 *   2. Toolbar: search, filter dropdowns (responsive), sort, agent actions
 *   3. Active filter chips (when filters applied)
 *   4. Document list ScrollArea with compact DocumentCards
 *   5. "Hidden" collapsible dock at bottom
 *   6. DocumentViewerSheet overlay (controlled by previewDoc state)
 *
 * Context model:
 *   - Doc-agent context: all visible docs (not manually hidden, passing
 *     current filters) are included. Hiding a doc or filtering it out
 *     removes it from the doc agent's scope for tagging/summarizing.
 *   - Chat-agent context: independent. Each card has a shimmer button to
 *     add/remove the doc from the chat agent's shared state.
 *
 * Features:
 *   - Loads example docs from GET /example-docs on mount
 *   - Responsive card variant (narrow/medium/wide) via ResizeObserver
 *   - Multi-axis filtering: search, mime, doc type, subtype, domain, tags
 *   - Clicking a tag pill toggles that tag filter
 *   - Batched agent-assisted auto-tagging with NDJSON progress
 *   - Summarize with streaming results (merges, never clears)
 *   - Search & Sort agent for scoring/ranking/selecting documents
 *   - "Hidden" dock shows manually hidden docs with unhide action
 *   - Document viewer sheet for PDF preview and extracted text
 */

import React, {
  useState,
  useMemo,
  useCallback,
  useRef,
  useEffect,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuditAgent, useClaimSessionState } from "@/hooks/use-audit-agent";
import { useUploadedDocs } from "@/hooks/use-uploaded-docs";
import { useChatDocs } from "@/hooks/use-chat-docs";
import {
  type BulkExpandedCommand,
  deriveFileExt,
  type DocumentCardUiState,
  type DocumentTagData,
  type DocumentSummaryData,
  type DocSearchData,
  type CardVariant,
} from "@/components/a2ui/documents";
import { DocumentViewerSheet } from "@/components/document-viewer-sheet";
import {
  getDefaultTagIconName,
  isTagIconName,
  ALL_TAGS,
} from "@/lib/tag-registry";
import {
  DocLensOverlay,
  DocLensProvider,
} from "@/components/doc-lens";
import { DocumentsPaneHeader } from "@/components/documents-pane/documents-pane-header";
import { DocumentsPaneToolbar } from "@/components/documents-pane/documents-pane-toolbar";
import { DocumentsPaneFilterPanel } from "@/components/documents-pane/documents-pane-filter-panel";
import { DocumentsPaneAutoTagPanel } from "@/components/documents-pane/documents-pane-auto-tag-panel";
import { DocumentsPaneSummarizeBar } from "@/components/documents-pane/documents-pane-summarize-bar";
import { DocumentsPaneSearchBar } from "@/components/documents-pane/documents-pane-search-bar";
import { DocumentsPaneFilterChips } from "@/components/documents-pane/documents-pane-filter-chips";
import { DocumentsGrid } from "@/components/documents-pane/documents-grid";
import { HiddenDocumentsDock } from "@/components/documents-pane/hidden-documents-dock";
import type {
  AutoTagMode,
  DocWithId,
  FilterChip,
  Filters,
  HiddenSortKey,
  SortKey,
  TagFilterMode,
} from "@/components/documents-pane/types";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";
const HIDDEN_DOC_NAMES_STORAGE_KEY = "agui_v3.hiddenDocNames.v1";
const HIDDEN_DOC_TIMESTAMPS_STORAGE_KEY = "agui_v3.hiddenDocTimestamps.v1";
const DOC_SUMMARIES_STORAGE_KEY = "agui_v3.docSummaries.v2";
const DOC_SEARCH_STORAGE_KEY = "agui_v3.docSearch.v1";
const DOC_SEARCH_HIDDEN_STORAGE_KEY = "agui_v4.docSearchHidden.v1";
const DOC_SEARCH_HIDDEN_TIMESTAMPS_STORAGE_KEY = "agui_v4.docSearchHiddenTimestamps.v1";
const DOC_TAGS_STORAGE_KEY = "agui_v3.docTags.v2";
const DOC_CARD_UI_STORAGE_KEY = "agui_v3.docCardUiState.v2";
const HIDDEN_DOCK_UI_STORAGE_KEY = "agui_v4.hiddenDockUi.v1";

const EMPTY_FILTERS: Filters = {
  search: "",
  mimeTypes: new Set(),
  docTypes: new Set(),
  subTypes: new Set(),
  domains: new Set(),
  tags: new Set(),
};

const MAX_CUSTOM_TAGS = 20;
const DEFAULT_TAG_SET = new Set(ALL_TAGS);

function normalizeTagLabel(label: string): string {
  return label.replace(/\s+/g, " ").trim();
}

function parseStoredTagData(value: unknown): DocumentTagData[] {
  if (!Array.isArray(value)) return [];

  return value.flatMap((item) => {
    if (typeof item === "string") {
      const label = normalizeTagLabel(item);
      return label
        ? [{ label, icon: getDefaultTagIconName(label) }]
        : [];
    }

    if (
      item &&
      typeof item === "object" &&
      typeof (item as { label?: unknown }).label === "string"
    ) {
      const label = normalizeTagLabel((item as { label: string }).label);
      if (!label) return [];
      const rawIcon =
        typeof (item as { icon?: unknown }).icon === "string"
          ? ((item as { icon?: string }).icon ?? null)
          : null;
      return [{ label, icon: isTagIconName(rawIcon) ? rawIcon : null }];
    }

    return [];
  });
}

// ── Utility: check if any filter is active ─────────────────────────────

function hasActiveFilters(f: Filters): boolean {
  return (
    f.search.length > 0 ||
    f.mimeTypes.size > 0 ||
    f.docTypes.size > 0 ||
    f.subTypes.size > 0 ||
    f.domains.size > 0 ||
    f.tags.size > 0
  );
}

/**
 * Checks whether a document's tags satisfy the active tag filters.
 *
 * Args:
 *   docTags: Tags assigned to the document.
 *   selectedTags: Active tag filters selected by the user.
 *   mode: Whether to match any selected tag (`or`) or all selected tags (`and`).
 *
 * Returns:
 *   `true` when the document should remain visible for the current tag filter.
 */
function matchesTagFilters(
  docTags: DocumentTagData[],
  selectedTags: Set<string>,
  mode: TagFilterMode
): boolean {
  if (selectedTags.size === 0) return true;

  if (mode === "or") {
    return docTags.some((tag) => selectedTags.has(tag.label));
  }

  const docTagLabels = new Set(docTags.map((tag) => tag.label));
  return [...selectedTags].every((tag) => docTagLabels.has(tag));
}

// ═══════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════

export function DocumentsPane() {
  const { state } = useAuditAgent();
  const { claimSession, isHydrated: isClaimSessionHydrated } =
    useClaimSessionState();
  const { uploadedDocs, addUploadedDoc } = useUploadedDocs();
  const { chatDocNames, toggleChatDoc } = useChatDocs();

  // ── Responsive variant via ResizeObserver ────────────────────────────

  const containerRef = useRef<HTMLDivElement>(null);
  const [paneWidth, setPaneWidth] = useState(400);
  const [viewportWidth, setViewportWidth] = useState(1280);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setPaneWidth(entry.contentRect.width);
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const syncViewportWidth = () => {
      setViewportWidth(window.innerWidth || 1280);
    };
    syncViewportWidth();
    window.addEventListener("resize", syncViewportWidth);
    return () => window.removeEventListener("resize", syncViewportWidth);
  }, []);

  const cardVariant: CardVariant =
    paneWidth > 900 ? "wide" : paneWidth > 550 ? "medium" : "narrow";
  const isNarrow = cardVariant === "narrow";
  const isCompactHiddenTable = paneWidth < 760;

  // Ratio of pane width to viewport distinguishes which sibling panes are open:
  //   ~0.25 = all three, ~0.33 = docs+output, ~0.50 = chat+docs, ~1.0 = docs-only
  const paneRatio = paneWidth / Math.max(viewportWidth, 1);

  // Grid columns: 1 when cramped, 2 for chat+docs, 4 for docs-only.
  // Minimum paneWidth guards prevent multi-column on very small viewports.
  const gridCols =
    paneWidth > 1100 && paneRatio > 0.70 ? 4 :
    paneWidth > 700 && paneRatio > 0.42 ? 2 :
    1;

  // Filter icon button (+ collapsible panel) for every scenario except
  // docs-only, where there is ample room for inline filter dropdowns.
  const useNarrowToolbar = paneRatio < 0.70;

  // Icon-only agent buttons only for the smallest views (all-three and
  // docs+output). Chat+docs and docs-only get labels alongside icons.
  const iconOnlyButtons = paneRatio < 0.42;

  // Show tags only when documents pane is effectively full-width
  // (e.g., chat and output are collapsed).
  const showHiddenTagsColumn =
    cardVariant === "wide" && paneRatio >= 0.58;

  // ── Fetch example docs only for hydrated local mode ────────────────

  const exampleDocsFetched = useRef(false);

  useEffect(() => {
    if (!isClaimSessionHydrated) return;
    if (claimSession.claimNumber) return;
    if (exampleDocsFetched.current) return;
    exampleDocsFetched.current = true;

    (async () => {
      try {
        const resp = await fetch(`${BACKEND_URL}/example-docs`);
        if (!resp.ok) return;
        const data = await resp.json();
        const docs = data.documents as Array<Record<string, string | number>>;
        for (const d of docs) {
          addUploadedDoc({
            file_name: d.file_name as string,
            claim_number: "",
            content_id: (d.content_id as string) || crypto.randomUUID(),
            mime_type: (d.mime_type as string) || "application/octet-stream",
            content_url: (d.content_url as string) || (d.path as string) || "",
            domain: "claim",
            document_type: "Example",
            document_description: `${d.file_size || ""}, ${d.page_count ?? 0} pages`,
            create_date: new Date().toISOString(),
            source_system: "EXAMPLE",
            content: (d.content as string) || "",
            token_count: (d.token_count as number) || undefined,
          });
        }
      } catch (err) {
        console.error("[ExampleDocs] Failed to load:", err);
      }
    })();
  }, [addUploadedDoc, claimSession.claimNumber, isClaimSessionHydrated]);

  // ── Hidden state (docs explicitly hidden from doc-agent context) ────

  const [hiddenFileNames, setHiddenFileNames] = useState<Set<string>>(new Set());
  const [hiddenAtByFileName, setHiddenAtByFileName] = useState<Map<string, string>>(
    new Map()
  );
  const [searchHiddenFileNames, setSearchHiddenFileNames] = useState<Set<string>>(
    new Set()
  );
  const [searchHiddenAtByFileName, setSearchHiddenAtByFileName] = useState<
    Map<string, string>
  >(new Map());
  const [hiddenStateHydrated, setHiddenStateHydrated] = useState(false);
  const [dockExpanded, setDockExpanded] = useState(false);
  const [dockMinimized, setDockMinimized] = useState(false);
  const [hiddenDockUiHydrated, setHiddenDockUiHydrated] = useState(false);
  const [hiddenSortKey, setHiddenSortKey] = useState<HiddenSortKey>("create_date");
  const [hiddenSortDir, setHiddenSortDir] = useState<"asc" | "desc">("desc");

  // Rehydrate hidden-doc state after pane remounts or route navigation.
  useEffect(() => {
    try {
      const rawNames = window.localStorage.getItem(HIDDEN_DOC_NAMES_STORAGE_KEY);
      if (rawNames) {
        const parsed = JSON.parse(rawNames) as unknown;
        if (Array.isArray(parsed)) {
          const names = parsed.filter((v): v is string => typeof v === "string");
          setHiddenFileNames(new Set(names));
        }
      }

      const rawTimes = window.localStorage.getItem(HIDDEN_DOC_TIMESTAMPS_STORAGE_KEY);
      if (rawTimes) {
        const parsed = JSON.parse(rawTimes) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const entries = Object.entries(parsed as Record<string, unknown>).filter(
            ([k, v]) => typeof k === "string" && typeof v === "string"
          ) as [string, string][];
          setHiddenAtByFileName(new Map(entries));
        }
      }
    } catch (error) {
      console.warn("[DocumentsPane] Failed to restore hidden-doc state:", error);
    } finally {
      setHiddenStateHydrated(true);
    }
  }, []);

  // Persist hidden-doc state whenever it changes.
  useEffect(() => {
    if (!hiddenStateHydrated) return;
    try {
      window.localStorage.setItem(
        HIDDEN_DOC_NAMES_STORAGE_KEY,
        JSON.stringify([...hiddenFileNames])
      );
      window.localStorage.setItem(
        HIDDEN_DOC_TIMESTAMPS_STORAGE_KEY,
        JSON.stringify(Object.fromEntries(hiddenAtByFileName))
      );
    } catch (error) {
      console.warn("[DocumentsPane] Failed to persist hidden-doc state:", error);
    }
  }, [hiddenAtByFileName, hiddenFileNames, hiddenStateHydrated]);

  // Rehydrate hidden-dock UI state so collapse/minimize choice persists.
  useEffect(() => {
    try {
      const rawDockUi = window.localStorage.getItem(HIDDEN_DOCK_UI_STORAGE_KEY);
      if (rawDockUi) {
        const parsed = JSON.parse(rawDockUi) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const expanded = (parsed as Record<string, unknown>).dockExpanded;
          const minimized = (parsed as Record<string, unknown>).dockMinimized;
          if (typeof expanded === "boolean") setDockExpanded(expanded);
          if (typeof minimized === "boolean") setDockMinimized(minimized);
        }
      }
    } catch (error) {
      console.warn("[DocumentsPane] Failed to restore hidden-dock UI state:", error);
    } finally {
      setHiddenDockUiHydrated(true);
    }
  }, []);

  // Persist hidden-dock UI state whenever the user toggles section visibility.
  useEffect(() => {
    if (!hiddenDockUiHydrated) return;
    try {
      window.localStorage.setItem(
        HIDDEN_DOCK_UI_STORAGE_KEY,
        JSON.stringify({
          dockExpanded,
          dockMinimized,
        })
      );
    } catch (error) {
      console.warn("[DocumentsPane] Failed to persist hidden-dock UI state:", error);
    }
  }, [dockExpanded, dockMinimized, hiddenDockUiHydrated]);

  // ── Filter state ────────────────────────────────────────────────────

  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [showFilters, setShowFilters] = useState(false);
  const [tagFilterMode, setTagFilterMode] = useState<TagFilterMode>("and");

  useEffect(() => {
    // Reset to the default multi-tag mode when no tag filters are active.
    if (filters.tags.size === 0 && tagFilterMode !== "and") {
      setTagFilterMode("and");
    }
  }, [filters.tags, tagFilterMode]);

  // ── Sort state ──────────────────────────────────────────────────────

  const [sortKey, setSortKey] = useState<SortKey>("default");

  // ── Summarization state ─────────────────────────────────────────────

  const [summaries, setSummaries] = useState<
    Map<string, DocumentSummaryData>
  >(new Map());
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summarizeProgress, setSummarizeProgress] = useState({
    done: 0,
    total: 0,
  });
  const [showSummarizeInput, setShowSummarizeInput] = useState(false);
  const [summarizeInstructions, setSummarizeInstructions] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  // ── Search & Sort state ────────────────────────────────────────────

  const [searchScores, setSearchScores] = useState<
    Map<string, DocSearchData>
  >(new Map());
  const [isSearching, setIsSearching] = useState(false);
  const [showSearchInput, setShowSearchInput] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const searchAbortRef = useRef<AbortController | null>(null);

  // ── Tagging state ───────────────────────────────────────────────────

  const [agentTags, setAgentTags] = useState<Map<string, DocumentTagData[]>>(
    new Map()
  );
  const [tagMode, setTagMode] = useState<AutoTagMode>("default");
  const [customTagCatalog, setCustomTagCatalog] = useState<string[]>(ALL_TAGS);
  const [selectedCustomTags, setSelectedCustomTags] = useState<string[]>(ALL_TAGS);
  const [customTagDraft, setCustomTagDraft] = useState("");
  const [customTagError, setCustomTagError] = useState<string | null>(null);

  // Derive filter options from the tags actually assigned to documents
  const allTagOptions = useMemo(() => {
    const seen = new Set<string>();
    for (const tags of agentTags.values()) {
      for (const tag of tags) seen.add(tag.label);
    }
    const defaultLabels = ALL_TAGS.filter((tag) => seen.has(tag));
    const customLabels = [...seen]
      .filter((tag) => !DEFAULT_TAG_SET.has(tag))
      .sort((a, b) => a.localeCompare(b));
    return [...defaultLabels, ...customLabels];
  }, [agentTags]);
  const selectedCustomTagSet = useMemo(
    () => new Set(selectedCustomTags),
    [selectedCustomTags]
  );
  const [cardUiByFileName, setCardUiByFileName] = useState<
    Map<string, DocumentCardUiState>
  >(new Map());
  const [bulkExpandedCommand, setBulkExpandedCommand] =
    useState<BulkExpandedCommand | null>(null);
  const [docEnrichmentHydrated, setDocEnrichmentHydrated] = useState(false);
  const [isTagging, setIsTagging] = useState(false);
  const [taggingProgress, setTaggingProgress] = useState<{
    batch: number;
    totalBatches: number;
  } | null>(null);
  const [showTagConfirm, setShowTagConfirm] = useState(false);
  const tagAbortRef = useRef<AbortController | null>(null);

  // ── Doc Lens state ──────────────────────────────────────────────────
  // openLensRef is populated by DocLensProvider and lets this component
  // trigger the overlay open without being inside the provider tree.
  const openLensRef = useRef<(() => void) | null>(null);
  const [docLensWarning, setDocLensWarning] = useState<string | null>(null);

  // Rehydrate summaries, search scores, search-hidden docs, tags, and card UI state.
  useEffect(() => {
    try {
      const rawSummaries = window.localStorage.getItem(DOC_SUMMARIES_STORAGE_KEY);
      if (rawSummaries) {
        const parsed = JSON.parse(rawSummaries) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const entries = Object.entries(parsed as Record<string, unknown>).filter(
            ([k, v]) =>
              typeof k === "string" &&
              !!v &&
              typeof v === "object" &&
              typeof (v as Record<string, unknown>).title === "string" &&
              typeof (v as Record<string, unknown>).summary === "string"
          ) as [string, DocumentSummaryData][];
          setSummaries(new Map(entries));
        }
      }

      const rawSearch = window.localStorage.getItem(DOC_SEARCH_STORAGE_KEY);
      if (rawSearch) {
        const parsed = JSON.parse(rawSearch) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const entries = Object.entries(parsed as Record<string, unknown>).filter(
            ([k, v]) =>
              typeof k === "string" &&
              !!v &&
              typeof v === "object" &&
              typeof (v as Record<string, unknown>).score === "number"
          ) as [string, DocSearchData][];
          setSearchScores(new Map(entries));
        }
      }

      const rawSearchHidden = window.localStorage.getItem(
        DOC_SEARCH_HIDDEN_STORAGE_KEY
      );
      if (rawSearchHidden) {
        const parsed = JSON.parse(rawSearchHidden) as unknown;
        if (Array.isArray(parsed)) {
          const names = parsed.filter((v): v is string => typeof v === "string");
          setSearchHiddenFileNames(new Set(names));
        }
      }

      const rawSearchHiddenTimes = window.localStorage.getItem(
        DOC_SEARCH_HIDDEN_TIMESTAMPS_STORAGE_KEY
      );
      if (rawSearchHiddenTimes) {
        const parsed = JSON.parse(rawSearchHiddenTimes) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const entries = Object.entries(parsed as Record<string, unknown>).filter(
            ([k, v]) => typeof k === "string" && typeof v === "string"
          ) as [string, string][];
          setSearchHiddenAtByFileName(new Map(entries));
        }
      }

      const rawTags = window.localStorage.getItem(DOC_TAGS_STORAGE_KEY);
      if (rawTags) {
        const parsed = JSON.parse(rawTags) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const entries = Object.entries(parsed as Record<string, unknown>)
            .filter(([k, v]) => typeof k === "string" && Array.isArray(v))
            .map(([k, v]) => [k, parseStoredTagData(v)]) as [string, DocumentTagData[]][];
          setAgentTags(new Map(entries));
        }
      }

      const rawCardUi = window.localStorage.getItem(DOC_CARD_UI_STORAGE_KEY);
      if (rawCardUi) {
        const parsed = JSON.parse(rawCardUi) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const entries = Object.entries(parsed as Record<string, unknown>)
            .filter(
              ([k, v]) =>
                typeof k === "string" &&
                !!v &&
                typeof v === "object" &&
                typeof (v as Record<string, unknown>).expanded === "boolean"
            )
            .map(([k, v]) => [k, v as DocumentCardUiState]) as [
            string,
            DocumentCardUiState,
          ][];
          setCardUiByFileName(new Map(entries));
        }
      }
    } catch (error) {
      console.warn("[DocumentsPane] Failed to restore summary/tag state:", error);
    } finally {
      setDocEnrichmentHydrated(true);
    }
  }, []);

  // Persist generated enrichments, including search-driven hidden docs.
  useEffect(() => {
    if (!docEnrichmentHydrated) return;
    try {
      window.localStorage.setItem(
        DOC_SUMMARIES_STORAGE_KEY,
        JSON.stringify(Object.fromEntries(summaries))
      );
      window.localStorage.setItem(
        DOC_SEARCH_STORAGE_KEY,
        JSON.stringify(Object.fromEntries(searchScores))
      );
      window.localStorage.setItem(
        DOC_SEARCH_HIDDEN_STORAGE_KEY,
        JSON.stringify([...searchHiddenFileNames])
      );
      window.localStorage.setItem(
        DOC_SEARCH_HIDDEN_TIMESTAMPS_STORAGE_KEY,
        JSON.stringify(Object.fromEntries(searchHiddenAtByFileName))
      );
      window.localStorage.setItem(
        DOC_TAGS_STORAGE_KEY,
        JSON.stringify(Object.fromEntries(agentTags))
      );
      window.localStorage.setItem(
        DOC_CARD_UI_STORAGE_KEY,
        JSON.stringify(Object.fromEntries(cardUiByFileName))
      );
    } catch (error) {
      console.warn("[DocumentsPane] Failed to persist summary/tag state:", error);
    }
  }, [
    agentTags,
    cardUiByFileName,
    docEnrichmentHydrated,
    searchHiddenAtByFileName,
    searchHiddenFileNames,
    searchScores,
    summaries,
  ]);

  // ── Viewer state ────────────────────────────────────────────────────

  const [previewDoc, setPreviewDoc] = useState<DocWithId | null>(null);

  // ── Build unified document list ─────────────────────────────────────

  const allDocs = useMemo(() => {
    const seenNames = new Set<string>();
    const docs: DocWithId[] = [];

    uploadedDocs.forEach((d, i) => {
      if (seenNames.has(d.file_name)) return;
      docs.push({ ...d, _id: `upload-${i}` });
      seenNames.add(d.file_name);
    });

    (state.documents || []).forEach((d, i) => {
      const fileName =
        (d.file_name as string) || (d.content_url as string) || "Untitled";
      if (seenNames.has(fileName)) return;
      docs.push({
        file_name: fileName,
        claim_number: (d.claim_number as string) || "",
        content_id: (d.content_id as string) || "",
        mime_type: (d.mime_type as string) || "application/octet-stream",
        content_url: (d.content_url as string) || "",
        domain: (d.domain as "claim" | "policy") || "claim",
        document_type: (d.document_type as string) || undefined,
        document_sub_type: (d.document_sub_type as string) || undefined,
        document_description:
          (d.document_description as string) || undefined,
        create_date: (d.create_date as string) || "",
        source_system: (d.source_system as string) || undefined,
        company_name: (d.company_name as string) || undefined,
        _id: `agent-${i}`,
      });
    });

    return docs;
  }, [state.documents, uploadedDocs]);

  const allHiddenFileNames = useMemo(
    () => new Set([...hiddenFileNames, ...searchHiddenFileNames]),
    [hiddenFileNames, searchHiddenFileNames]
  );

  const hiddenAtLookup = useMemo(
    () => new Map([...hiddenAtByFileName, ...searchHiddenAtByFileName]),
    [hiddenAtByFileName, searchHiddenAtByFileName]
  );

  // Count only hidden docs that currently exist in the live document list.
  const hiddenCount = useMemo(
    () => allDocs.filter((d) => allHiddenFileNames.has(d.file_name)).length,
    [allDocs, allHiddenFileNames]
  );

  // ── Derive filter option sets from all docs ─────────────────────────

  const filterOptions = useMemo(() => {
    const mimes = new Set<string>();
    const types = new Set<string>();
    const subs = new Set<string>();
    const doms = new Set<string>();

    for (const doc of allDocs) {
      const ext = deriveFileExt(doc.mime_type, doc.file_name).toUpperCase();
      mimes.add(ext);
      if (doc.document_type) types.add(doc.document_type);
      if (doc.document_sub_type) subs.add(doc.document_sub_type);
      doms.add(doc.domain);
    }

    return {
      mimeTypes: [...mimes].sort(),
      docTypes: [...types].sort(),
      subTypes: [...subs].sort(),
      domains: [...doms].sort(),
    };
  }, [allDocs]);

  // ── Split docs into visible vs hidden lists ─────────────────────────

  const hiddenDocs = useMemo(
    () => allDocs.filter((d) => allHiddenFileNames.has(d.file_name)),
    [allDocs, allHiddenFileNames]
  );

  const nonHiddenDocs = useMemo(
    () => allDocs.filter((d) => !allHiddenFileNames.has(d.file_name)),
    [allDocs, allHiddenFileNames]
  );

  // ── Apply filters to non-hidden docs ───────────────────────────────

  const filteredDocs = useMemo(() => {
    let docs = nonHiddenDocs;

    if (filters.search) {
      const q = filters.search.toLowerCase();
      docs = docs.filter(
        (d) =>
          d.file_name.toLowerCase().includes(q) ||
          d.document_description?.toLowerCase().includes(q) ||
          d.document_type?.toLowerCase().includes(q) ||
          d.document_sub_type?.toLowerCase().includes(q)
      );
    }

    if (filters.mimeTypes.size > 0) {
      docs = docs.filter((d) =>
        filters.mimeTypes.has(
          deriveFileExt(d.mime_type, d.file_name).toUpperCase()
        )
      );
    }

    if (filters.docTypes.size > 0) {
      docs = docs.filter(
        (d) => d.document_type && filters.docTypes.has(d.document_type)
      );
    }

    if (filters.subTypes.size > 0) {
      docs = docs.filter(
        (d) =>
          d.document_sub_type && filters.subTypes.has(d.document_sub_type)
      );
    }

    if (filters.domains.size > 0) {
      docs = docs.filter((d) => filters.domains.has(d.domain));
    }

    if (filters.tags.size > 0) {
      docs = docs.filter((d) => {
        const docTags = agentTags.get(d.file_name) || [];
        return matchesTagFilters(docTags, filters.tags, tagFilterMode);
      });
    }

    return docs;
  }, [nonHiddenDocs, filters, agentTags, tagFilterMode]);

  // Sum of token counts across all in-scope (filtered) documents.
  const filteredTokenCount = useMemo(
    () => filteredDocs.reduce((sum, d) => sum + (d.token_count ?? 0), 0),
    [filteredDocs]
  );

  // ── Doc Lens eligible docs (must be after filteredDocs) ──────────────

  const DOC_LENS_ELIGIBLE_MIMES = useMemo(
    () => new Set(["application/pdf", "image/jpeg", "image/png"]),
    []
  );

  const docLensEligibleDocs = useMemo(
    () => filteredDocs.filter((d) => DOC_LENS_ELIGIBLE_MIMES.has(d.mime_type)),
    [filteredDocs, DOC_LENS_ELIGIBLE_MIMES]
  );

  const handleDocLensClick = useCallback(() => {
    if (docLensEligibleDocs.length === 0) {
      setDocLensWarning(
        "No PDF or image files are visible. Doc Lens requires at least one PDF or image file."
      );
      setTimeout(() => setDocLensWarning(null), 4000);
      return;
    }
    setDocLensWarning(null);
    // Delegate to the provider via the imperative ref.
    openLensRef.current?.();
  }, [docLensEligibleDocs]);

  // Docs that are out of doc-agent context (manually hidden + filtered out)
  const filterHiddenCount = nonHiddenDocs.length - filteredDocs.length;
  const showHiddenDock = hiddenCount > 0 || filterHiddenCount > 0;

  /**
   * Build hidden-doc type stats and latest hidden timestamp for the collapsed
   * dock state. Stats are based on all currently hidden docs, regardless of
   * whether they were manually hidden or search-excluded.
   */
  const hiddenStats = useMemo(() => {
    const extensionCounts = new Map<string, number>();
    let latestHiddenAt: string | null = null;

    for (const doc of hiddenDocs) {
      const ext = deriveFileExt(doc.mime_type, doc.file_name).toUpperCase();
      extensionCounts.set(ext, (extensionCounts.get(ext) || 0) + 1);

      const hiddenAt = hiddenAtLookup.get(doc.file_name);
      if (
        hiddenAt &&
        (!latestHiddenAt ||
          new Date(hiddenAt).getTime() > new Date(latestHiddenAt).getTime())
      ) {
        latestHiddenAt = hiddenAt;
      }
    }

    const topExtensions = [...extensionCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2);

    const hiddenPercent =
      allDocs.length > 0 ? Math.round((hiddenCount / allDocs.length) * 100) : 0;

    return {
      topExtensions,
      latestHiddenAt,
      hiddenPercent,
    };
  }, [allDocs.length, hiddenAtLookup, hiddenCount, hiddenDocs]);

  // Sort hidden docs for table rendering.
  const sortedHiddenDocs = useMemo(() => {
    const docs = [...hiddenDocs];

    const comparableValue = (doc: DocWithId, key: HiddenSortKey): string | number => {
      switch (key) {
        case "file_name":
          return doc.file_name.toLowerCase();
        case "ext":
          return deriveFileExt(doc.mime_type, doc.file_name).toLowerCase();
        case "domain":
          return (doc.domain || "").toLowerCase();
        case "document_type":
          return (doc.document_type || "").toLowerCase();
        case "document_sub_type":
          return (doc.document_sub_type || "").toLowerCase();
        case "create_date":
          return doc.create_date ? new Date(doc.create_date).getTime() : 0;
        case "source_system":
          return (doc.source_system || "").toLowerCase();
        default:
          return "";
      }
    };

    docs.sort((a, b) => {
      const aVal = comparableValue(a, hiddenSortKey);
      const bVal = comparableValue(b, hiddenSortKey);

      if (typeof aVal === "number" && typeof bVal === "number") {
        const delta = hiddenSortDir === "asc" ? aVal - bVal : bVal - aVal;
        if (delta !== 0) return delta;
      } else {
        const aStr = String(aVal);
        const bStr = String(bVal);
        const delta =
          hiddenSortDir === "asc"
            ? aStr.localeCompare(bStr, undefined, { sensitivity: "base" })
            : bStr.localeCompare(aStr, undefined, { sensitivity: "base" });
        if (delta !== 0) return delta;
      }

      // Deterministic tie-break so sorting always produces a stable, visible order.
      return a.file_name.localeCompare(b.file_name, undefined, {
        sensitivity: "base",
      });
    });

    return docs;
  }, [hiddenDocs, hiddenSortDir, hiddenSortKey]);

  // ── Sort filtered docs ──────────────────────────────────────────────

  const sortedDocs = useMemo(() => {
    return [...filteredDocs].sort((a, b) => {
      switch (sortKey) {
        case "score": {
          const aScore = searchScores.get(a.file_name)?.score ?? -1;
          const bScore = searchScores.get(b.file_name)?.score ?? -1;
          return bScore - aScore;
        }
        case "date": {
          const aDate = a.create_date
            ? new Date(a.create_date).getTime()
            : 0;
          const bDate = b.create_date
            ? new Date(b.create_date).getTime()
            : 0;
          return bDate - aDate;
        }
        case "title":
          return a.file_name.localeCompare(b.file_name);
        case "default":
        default:
          return 0;
      }
    });
  }, [filteredDocs, sortKey, searchScores]);

  // ── Hidden helpers ──────────────────────────────────────────────────

  const toggleHidden = useCallback((fileName: string) => {
    setHiddenFileNames((prev) => {
      const next = new Set(prev);
      const isCurrentlyHidden = next.has(fileName);

      if (isCurrentlyHidden) next.delete(fileName);
      else next.add(fileName);

      setHiddenAtByFileName((prevTimes) => {
        const nextTimes = new Map(prevTimes);
        if (isCurrentlyHidden) {
          nextTimes.delete(fileName);
        } else if (!nextTimes.has(fileName)) {
          nextTimes.set(fileName, new Date().toISOString());
        }
        return nextTimes;
      });

      return next;
    });
  }, []);

  const hideAll = useCallback(() => {
    const hiddenAt = new Date().toISOString();

    setHiddenFileNames((prev) => {
      const next = new Set(prev);
      for (const doc of allDocs) next.add(doc.file_name);
      return next;
    });

    setHiddenAtByFileName((prev) => {
      const next = new Map(prev);
      for (const doc of allDocs) {
        if (!next.has(doc.file_name)) next.set(doc.file_name, hiddenAt);
      }
      return next;
    });
  }, [allDocs]);

  const setExpandedForNonHidden = useCallback(
    (expanded: boolean) => {
      const version = Date.now();
      setBulkExpandedCommand({
        version,
        expanded,
        tagsOpen: expanded,
        summaryOpen: expanded,
      });
      setCardUiByFileName((prev) => {
        const next = new Map(prev);
        for (const doc of nonHiddenDocs) {
          next.set(doc.file_name, {
            expanded,
            tagsOpen: expanded,
            summaryOpen: expanded,
          });
        }
        return next;
      });
    },
    [nonHiddenDocs]
  );

  const collapseAllNonHidden = useCallback(() => {
    setExpandedForNonHidden(false);
  }, [setExpandedForNonHidden]);

  const expandAllNonHidden = useCallback(() => {
    setExpandedForNonHidden(true);
  }, [setExpandedForNonHidden]);

  const unhide = useCallback((fileName: string) => {
    setHiddenFileNames((prev) => {
      const next = new Set(prev);
      next.delete(fileName);
      return next;
    });

    setHiddenAtByFileName((prev) => {
      const next = new Map(prev);
      next.delete(fileName);
      return next;
    });

    setSearchHiddenFileNames((prev) => {
      const next = new Set(prev);
      next.delete(fileName);
      return next;
    });

    setSearchHiddenAtByFileName((prev) => {
      const next = new Map(prev);
      next.delete(fileName);
      return next;
    });
  }, []);

  const unhideAll = useCallback(() => {
    setHiddenFileNames(new Set());
    setHiddenAtByFileName(new Map());
    setSearchHiddenFileNames(new Set());
    setSearchHiddenAtByFileName(new Map());
  }, []);

  const refreshDocs = useCallback(() => {
    // Stop any in-flight generation jobs before clearing derived data.
    abortRef.current?.abort();
    tagAbortRef.current?.abort();
    searchAbortRef.current?.abort();

    setIsSummarizing(false);
    setShowSummarizeInput(false);
    setSummarizeProgress({ done: 0, total: 0 });
    setIsTagging(false);
    setTaggingProgress(null);
    setIsSearching(false);
    setShowSearchInput(false);
    setSearchQuery("");

    // Clear generated enrichment state (summaries + search + tags), but keep docs/context.
    setSummaries(new Map());
    setSearchScores(new Map());
    setSearchHiddenFileNames(new Set());
    setSearchHiddenAtByFileName(new Map());
    setAgentTags(new Map());
    setSortKey("default");

    // Preserve card expansion while closing generated sub-sections.
    setCardUiByFileName((prev) => {
      const next = new Map(prev);
      for (const [fileName, ui] of prev.entries()) {
        next.set(fileName, {
          ...ui,
          tagsOpen: false,
          summaryOpen: false,
        });
      }
      return next;
    });

    // Remove tag/search filters so all docs are visible after refresh.
    setFilters((prev) => ({
      ...prev,
      search: "",
      tags: new Set(),
    }));

    // Keep persistent state in sync immediately.
    try {
      window.localStorage.removeItem(DOC_SUMMARIES_STORAGE_KEY);
      window.localStorage.removeItem(DOC_SEARCH_STORAGE_KEY);
      window.localStorage.removeItem(DOC_SEARCH_HIDDEN_STORAGE_KEY);
      window.localStorage.removeItem(DOC_SEARCH_HIDDEN_TIMESTAMPS_STORAGE_KEY);
      window.localStorage.removeItem(DOC_TAGS_STORAGE_KEY);
    } catch (error) {
      console.warn("[DocumentsPane] Failed to clear generated doc state:", error);
    }
  }, []);

  const updateCardUiState = useCallback(
    (fileName: string, nextState: DocumentCardUiState) => {
      setCardUiByFileName((prev) => {
        const prevState = prev.get(fileName);
        if (
          prevState &&
          prevState.expanded === nextState.expanded &&
          prevState.tagsOpen === nextState.tagsOpen &&
          prevState.summaryOpen === nextState.summaryOpen
        ) {
          return prev;
        }
        const next = new Map(prev);
        next.set(fileName, nextState);
        return next;
      });
    },
    []
  );

  // ── Filter helpers ──────────────────────────────────────────────────

  const toggleFilter = useCallback(
    (key: keyof Omit<Filters, "search">, value: string) => {
      setFilters((prev) => {
        const set = new Set(prev[key]);
        if (set.has(value)) set.delete(value);
        else set.add(value);
        return { ...prev, [key]: set };
      });
    },
    []
  );

  const removeFilterChip = useCallback(
    (key: keyof Filters, value?: string) => {
      setFilters((prev) => {
        if (key === "search") return { ...prev, search: "" };
        const set = new Set(prev[key] as Set<string>);
        if (value) set.delete(value);
        return { ...prev, [key]: set };
      });
    },
    []
  );

  const clearAllFilters = useCallback(() => setFilters(EMPTY_FILTERS), []);

  const handleHiddenSort = useCallback(
    (key: HiddenSortKey) => {
      if (hiddenSortKey === key) {
        setHiddenSortDir((prevDir) => (prevDir === "asc" ? "desc" : "asc"));
        return;
      }
      setHiddenSortKey(key);
      setHiddenSortDir("asc");
    },
    [hiddenSortKey]
  );

  // ── Build content lookup from AG-UI state ───────────────────────────

  const contentByName = useMemo(() => {
    const map = new Map<string, Record<string, unknown>>();
    for (const d of state.documents || []) {
      const name =
        (d.file_name as string) || (d.content_url as string) || "";
      if (name) map.set(name, d);
    }
    return map;
  }, [state.documents]);

  // Content stored in uploadedDocs (includes example docs and user uploads)
  const uploadedContentByName = useMemo(() => {
    const map = new Map<string, string>();
    for (const d of uploadedDocs) {
      if (d.content) map.set(d.file_name, d.content);
    }
    return map;
  }, [uploadedDocs]);

  /**
   * Look up text content for a document. Priority:
   *  1. uploadedDocs.content (persists independently of state.documents)
   *  2. state.documents content / text
   *  3. document_description fallback (metadata-only docs)
   */
  const getTextContent = useCallback(
    (fileName: string): string => {
      const fromUploaded = uploadedContentByName.get(fileName);
      if (fromUploaded) return fromUploaded;

      const stateDoc = contentByName.get(fileName);
      const fromState =
        (stateDoc?.content as string) || (stateDoc?.text as string) || "";
      if (fromState) return fromState;

      // Last resort: use description so metadata-only docs aren't skipped
      const doc = allDocs.find((d) => d.file_name === fileName);
      return doc?.document_description || "";
    },
    [uploadedContentByName, contentByName, allDocs]
  );

  // ── Build doc payloads for visible (non-hidden, filtered-in) docs ───

  const buildVisiblePayloads = useCallback(() => {
    return filteredDocs.map((doc) => ({
      file_name: doc.file_name,
      content_id: doc.content_id || doc.file_name,
      content: getTextContent(doc.file_name),
      mime_type: doc.mime_type,
      content_url: doc.content_url || "",
      document_type: doc.document_type || "",
      document_description: doc.document_description || "",
    }));
  }, [filteredDocs, getTextContent]);

  // ── Summarize via NDJSON stream ─────────────────────────────────────

  const runSummarize = useCallback(async (overrideInstructions?: string) => {
    const payloads = buildVisiblePayloads();
    if (payloads.length === 0) return;
    const trimmedAdditionalInstructions =
      overrideInstructions?.trim() ?? summarizeInstructions.trim();

    if (typeof overrideInstructions === "string") {
      setSummarizeInstructions(overrideInstructions);
    }

    setIsSummarizing(true);
    setSummarizeProgress({ done: 0, total: payloads.length });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const resp = await fetch(`${BACKEND_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          documents: payloads,
          ...(trimmedAdditionalInstructions
            ? { additional_instructions: trimmedAdditionalInstructions }
            : {}),
        }),
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        console.error("[Summarize] Bad response", resp.status);
        setIsSummarizing(false);
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
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const obj = JSON.parse(trimmed) as Record<string, unknown>;
            if (obj.error) {
              setSummarizeProgress((prev) => ({
                ...prev,
                done: prev.done + 1,
              }));
              continue;
            }
            const summary = obj as unknown as DocumentSummaryData & {
              file_name: string;
            };
            setSummaries((prev) => {
              const next = new Map(prev);
              next.set(summary.file_name, {
                title: summary.title,
                summary: summary.summary,
                label: summary.label,
              });
              return next;
            });
            setSummarizeProgress((prev) => ({
              ...prev,
              done: prev.done + 1,
            }));
          } catch {
            /* skip unparsable */
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        console.error("[Summarize] Stream error", err);
      }
    } finally {
      setIsSummarizing(false);
      setShowSummarizeInput(false);
      abortRef.current = null;
    }
  }, [buildVisiblePayloads, summarizeInstructions]);

  const cancelSummarize = useCallback(() => {
    abortRef.current?.abort();
    setIsSummarizing(false);
    setShowSummarizeInput(false);
  }, []);

  // ── Search & Sort via JSON response ─────────────────────────────────

  const runSearchSort = useCallback(async (overrideQuery?: string) => {
    const effectiveQuery = overrideQuery?.trim() ?? searchQuery.trim();
    if (!effectiveQuery) return;
    const docs = filteredDocs;
    if (docs.length === 0) return;

    if (typeof overrideQuery === "string") {
      setSearchQuery(overrideQuery);
    }

    setIsSearching(true);
    setSortKey("score");

    const controller = new AbortController();
    searchAbortRef.current = controller;

    // Build payloads with full metadata for the agent
    const payloads = docs.map((doc) => ({
      file_name: doc.file_name,
      content_id: doc.content_id || doc.file_name,
      content: getTextContent(doc.file_name),
      mime_type: doc.mime_type,
      content_url: doc.content_url || "",
      claim_number: doc.claim_number || "",
      domain: doc.domain || "claim",
      document_type: doc.document_type || "",
      document_sub_type: doc.document_sub_type || "",
      document_description: doc.document_description || "",
      create_date: doc.create_date || "",
      source_system: doc.source_system || "",
      company_name: doc.company_name || "",
    }));

    try {
      const resp = await fetch(`${BACKEND_URL}/search-sort`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: effectiveQuery, documents: payloads }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        console.error("[SearchSort] Bad response", resp.status);
        setIsSearching(false);
        return;
      }

      const data = (await resp.json()) as {
        scores: Array<{ content_id: string; score: number; label: string }>;
        content_id_to_file_name: Record<string, string>;
      };

      if (data.scores) {
        const currentFileNames = new Set(docs.map((doc) => doc.file_name));
        const scoredFileNames = new Set<string>();
        const hiddenFromSearch = new Set<string>();
        const scopedContentIdToFileName = new Map(
          docs.map((doc) => [doc.content_id || doc.file_name, doc.file_name])
        );

        // Replace search scores for the current in-scope set so omitted docs do
        // not keep stale data from a previous query.
        setSearchScores((prev) => {
          const next = new Map(prev);
          for (const fileName of currentFileNames) {
            next.delete(fileName);
          }
          for (const s of data.scores) {
            const fileName =
              scopedContentIdToFileName.get(s.content_id) ||
              data.content_id_to_file_name[s.content_id] ||
              s.content_id;
            next.set(fileName, { score: s.score, label: s.label });
            scoredFileNames.add(fileName);
            if (s.score === 0) hiddenFromSearch.add(fileName);
          }
          return next;
        });

        // Omission is the primary exclusion signal. For backward compatibility,
        // an explicit zero score is still treated as hidden.
        for (const fileName of currentFileNames) {
          if (!scoredFileNames.has(fileName)) {
            hiddenFromSearch.add(fileName);
          }
        }

        const hiddenAt = new Date().toISOString();
        setSearchHiddenFileNames((prev) => {
          const next = new Set(prev);
          for (const fileName of currentFileNames) {
            next.delete(fileName);
          }
          for (const fileName of hiddenFromSearch) {
            next.add(fileName);
          }
          return next;
        });
        setSearchHiddenAtByFileName((prev) => {
          const next = new Map(prev);
          for (const fileName of currentFileNames) {
            next.delete(fileName);
          }
          for (const fileName of hiddenFromSearch) {
            next.set(fileName, hiddenAt);
          }
          return next;
        });
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        console.error("[SearchSort] Error", err);
      }
    } finally {
      setIsSearching(false);
      setShowSearchInput(false);
      searchAbortRef.current = null;
    }
  }, [searchQuery, filteredDocs, getTextContent]);

  const cancelSearchSort = useCallback(() => {
    searchAbortRef.current?.abort();
    setIsSearching(false);
    setShowSearchInput(false);
  }, []);

  // ── Auto-tag via NDJSON stream ──────────────────────────────────────

  const addCustomTag = useCallback(() => {
    const label = normalizeTagLabel(customTagDraft);
    if (!label) {
      setCustomTagError("Enter a tag name before adding it.");
      return;
    }

    const isDuplicate = customTagCatalog.some(
      (existing) => existing.toLowerCase() === label.toLowerCase()
    );
    if (isDuplicate) {
      setCustomTagError(`Duplicate tags are not allowed: "${label}" already exists.`);
      return;
    }

    if (customTagCatalog.length >= MAX_CUSTOM_TAGS) {
      setCustomTagError(`You can customize up to ${MAX_CUSTOM_TAGS} tags in total.`);
      return;
    }

    setCustomTagCatalog((prev) => [...prev, label]);
    setSelectedCustomTags((prev) => [...prev, label]);
    setCustomTagDraft("");
    setCustomTagError(null);
  }, [customTagCatalog, customTagDraft]);

  const toggleCustomTagSelection = useCallback((label: string) => {
    setSelectedCustomTags((prev) =>
      prev.includes(label)
        ? prev.filter((tag) => tag !== label)
        : [...prev, label]
    );
  }, []);

  const removeCustomTag = useCallback((label: string) => {
    setCustomTagCatalog((prev) => prev.filter((tag) => tag !== label));
    setSelectedCustomTags((prev) => prev.filter((tag) => tag !== label));
    setCustomTagError(null);
  }, []);

  const selectAllCustomTags = useCallback(() => {
    setSelectedCustomTags(customTagCatalog);
  }, [customTagCatalog]);

  const unselectAllCustomTags = useCallback(() => {
    setSelectedCustomTags([]);
  }, []);

  const removeAllCustomTags = useCallback(() => {
    setCustomTagCatalog([]);
    setSelectedCustomTags([]);
    setCustomTagError(null);
  }, []);

  const restoreDefaultCustomTags = useCallback(() => {
    setCustomTagCatalog(ALL_TAGS);
    setSelectedCustomTags(ALL_TAGS);
    setCustomTagError(null);
  }, []);

  const runAutoTag = useCallback(async () => {
    // Tag visible docs (not hidden) so the user can use tags to filter
    const payloads = buildVisiblePayloads();
    const activeCustomTags = customTagCatalog.filter((tag) =>
      selectedCustomTagSet.has(tag)
    );

    if (payloads.length === 0) return;
    if (tagMode === "custom" && activeCustomTags.length === 0) return;

    setShowTagConfirm(false);
    setIsTagging(true);
    setTaggingProgress(null);

    const controller = new AbortController();
    tagAbortRef.current = controller;

    try {
      const resp = await fetch(`${BACKEND_URL}/document-tags`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          documents: payloads,
          tag_mode: tagMode,
          selected_tags: tagMode === "custom" ? activeCustomTags : [],
        }),
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        console.error("[AutoTag] Bad response", resp.status);
        setIsTagging(false);
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
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const obj = JSON.parse(trimmed) as Record<string, unknown>;

            if (obj.batch) {
              setTaggingProgress({
                batch: obj.batch as number,
                totalBatches: obj.total_batches as number,
              });
              if (!obj.error) {
                const results = obj.results as Array<{
                  file_name: string;
                  tags: DocumentTagData[];
                }>;
                setAgentTags((prev) => {
                  const next = new Map(prev);
                  for (const r of results) next.set(r.file_name, r.tags);
                  return next;
                });
              }
            }
          } catch {
            /* skip */
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        console.error("[AutoTag] Stream error", err);
      }
    } finally {
      setIsTagging(false);
      setTaggingProgress(null);
      tagAbortRef.current = null;
    }
  }, [
    buildVisiblePayloads,
    customTagCatalog,
    selectedCustomTagSet,
    tagMode,
  ]);

  const cancelAutoTag = useCallback(() => {
    tagAbortRef.current?.abort();
    setIsTagging(false);
    setTaggingProgress(null);
    setShowTagConfirm(false);
  }, []);

  // ── Collect active filter chips for display ─────────────────────────

  const filterChips = useMemo(() => {
    const chips: FilterChip[] = [];
    if (filters.search)
      chips.push({ key: "search", value: "", label: `"${filters.search}"` });
    for (const v of filters.mimeTypes)
      chips.push({ key: "mimeTypes", value: v, label: v });
    for (const v of filters.docTypes)
      chips.push({ key: "docTypes", value: v, label: v });
    for (const v of filters.subTypes)
      chips.push({ key: "subTypes", value: v, label: v });
    for (const v of filters.domains)
      chips.push({ key: "domains", value: v, label: v });
    for (const v of filters.tags)
      chips.push({ key: "tags", value: v, label: v });
    return chips;
  }, [filters]);

  const customSelectionCount = selectedCustomTags.length;
  const hasCustomSelection = customSelectionCount > 0;
  const customSelectionNotice =
    tagMode === "custom" && !hasCustomSelection
      ? "No tags selected. Auto-tagging is disabled until you select at least one tag."
      : null;

  // ═══════════════════════════════════════════════════════════════════
  // Render
  // ═══════════════════════════════════════════════════════════════════

  return (
    <DocLensProvider docs={docLensEligibleDocs} openLensRef={openLensRef}>
      <div
        ref={containerRef}
        className="relative flex flex-col h-full overflow-hidden"
      >
        <DocumentsPaneHeader
          allDocsCount={allDocs.length}
          filteredDocsCount={filteredDocs.length}
          hiddenCount={hiddenCount}
          nonHiddenDocsCount={nonHiddenDocs.length}
          bulkExpanded={bulkExpandedCommand?.expanded ?? null}
          onUnhideAll={unhideAll}
          onHideAll={hideAll}
          onExpandAll={expandAllNonHidden}
          onCollapseAll={collapseAllNonHidden}
        />

        <DocumentsPaneToolbar
          summariesSize={summaries.size}
          searchScoresSize={searchScores.size}
          agentTagsSize={agentTags.size}
          allTagOptionsLength={allTagOptions.length}
          isSummarizing={isSummarizing}
          isSearching={isSearching}
          isTagging={isTagging}
          filteredDocsCount={filteredDocs.length}
          filteredTokenCount={filteredTokenCount}
          useNarrowToolbar={useNarrowToolbar}
          showFilters={showFilters}
          filters={filters}
          filterOptions={filterOptions}
          allTagOptions={allTagOptions}
          sortKey={sortKey}
          iconOnlyButtons={iconOnlyButtons}
          showTagConfirm={showTagConfirm}
          showSearchInput={showSearchInput}
          showSummarizeInput={showSummarizeInput}
          taggingProgress={taggingProgress}
          summarizeProgress={summarizeProgress}
          onRefreshDocs={refreshDocs}
          onToggleShowFilters={() => setShowFilters((prev) => !prev)}
          onToggleFilter={toggleFilter}
          onSortKeyChange={setSortKey}
          onToggleTagConfirm={() => {
            setShowTagConfirm((prev) => !prev);
            setShowSummarizeInput(false);
            setShowSearchInput(false);
          }}
          onToggleSearchInput={() => {
            setShowSearchInput((prev) => !prev);
            setShowTagConfirm(false);
            setShowSummarizeInput(false);
          }}
          onToggleSummarizeInput={() => {
            setShowSummarizeInput((prev) => !prev);
            setShowTagConfirm(false);
            setShowSearchInput(false);
          }}
          onCancelAutoTag={cancelAutoTag}
          onCancelSearchSort={cancelSearchSort}
          onCancelSummarize={cancelSummarize}
          onHandleDocLensClick={handleDocLensClick}
        />

        <DocumentsPaneFilterPanel
          useNarrowToolbar={useNarrowToolbar}
          showFilters={showFilters}
          filterOptions={filterOptions}
          allTagOptions={allTagOptions}
          filters={filters}
          onToggleFilter={toggleFilter}
        />

        <DocumentsPaneAutoTagPanel
          showTagConfirm={showTagConfirm}
          isTagging={isTagging}
          filteredDocsCount={filteredDocs.length}
          tagMode={tagMode}
          hasCustomSelection={hasCustomSelection}
          selectedCustomTags={selectedCustomTags}
          customTagCatalog={customTagCatalog}
          customTagDraft={customTagDraft}
          customTagError={customTagError}
          customSelectionNotice={customSelectionNotice}
          selectedCustomTagSet={selectedCustomTagSet}
          maxCustomTags={MAX_CUSTOM_TAGS}
          onRunAutoTag={runAutoTag}
          onClose={() => setShowTagConfirm(false)}
          onSetTagMode={(mode) => {
            setTagMode(mode);
            setCustomTagError(null);
          }}
          onSelectAllCustomTags={selectAllCustomTags}
          onUnselectAllCustomTags={unselectAllCustomTags}
          onRestoreDefaultCustomTags={restoreDefaultCustomTags}
          onRemoveAllCustomTags={removeAllCustomTags}
          onCustomTagDraftChange={(value) => {
            setCustomTagDraft(value);
            if (customTagError) setCustomTagError(null);
          }}
          onAddCustomTag={addCustomTag}
          onToggleCustomTagSelection={toggleCustomTagSelection}
          onRemoveCustomTag={removeCustomTag}
        />

        <DocumentsPaneSummarizeBar
          showSummarizeInput={showSummarizeInput}
          isSummarizing={isSummarizing}
          filteredDocsCount={filteredDocs.length}
          summarizeInstructions={summarizeInstructions}
          onSummarizeInstructionsChange={setSummarizeInstructions}
          onRunSummarize={runSummarize}
          onClose={() => setShowSummarizeInput(false)}
        />

        <DocumentsPaneSearchBar
          showSearchInput={showSearchInput}
          isSearching={isSearching}
          filteredDocsCount={filteredDocs.length}
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          onRunSearchSort={runSearchSort}
          onClose={() => setShowSearchInput(false)}
        />

        <DocumentsPaneFilterChips
          filterChips={filterChips}
          filters={filters}
          tagFilterMode={tagFilterMode}
          onRemoveFilterChip={removeFilterChip}
          onClearAllFilters={clearAllFilters}
          onToggleTagFilterMode={() =>
            setTagFilterMode((prev) => (prev === "or" ? "and" : "or"))
          }
        />

        <DocumentsGrid
          sortedDocs={sortedDocs}
          summaries={summaries}
          searchScores={searchScores}
          agentTags={agentTags}
          sortKey={sortKey}
          cardVariant={cardVariant}
          chatDocNames={chatDocNames}
          filters={filters}
          gridCols={gridCols}
          showHiddenDock={showHiddenDock}
          dockMinimized={dockMinimized}
          dockExpanded={dockExpanded}
          bulkExpandedCommand={bulkExpandedCommand}
          cardUiByFileName={cardUiByFileName}
          hasActiveFilters={hasActiveFilters}
          onToggleHidden={toggleHidden}
          onToggleChatDoc={toggleChatDoc}
          onToggleTagFilter={(tag) => toggleFilter("tags", tag)}
          onPreviewDoc={setPreviewDoc}
          onCardUiStateChange={updateCardUiState}
        />

        <HiddenDocumentsDock
          showHiddenDock={showHiddenDock}
          hiddenCount={hiddenCount}
          filterHiddenCount={filterHiddenCount}
          dockExpanded={dockExpanded}
          dockMinimized={dockMinimized}
          hiddenStats={hiddenStats}
          hiddenDocs={hiddenDocs}
          sortedHiddenDocs={sortedHiddenDocs}
          hiddenSortKey={hiddenSortKey}
          hiddenSortDir={hiddenSortDir}
          isCompactHiddenTable={isCompactHiddenTable}
          showHiddenTagsColumn={showHiddenTagsColumn}
          agentTags={agentTags}
          onToggleDockExpanded={() => {
            if (dockMinimized) {
              setDockMinimized(false);
              return;
            }
            setDockExpanded((prev) => !prev);
          }}
          onToggleDockMinimized={() => setDockMinimized((prev) => !prev)}
          onUnhideAll={unhideAll}
          onHandleHiddenSort={handleHiddenSort}
          onPreviewDoc={setPreviewDoc}
          onUnhide={unhide}
        />

      {/* ── Document Viewer Sheet ───────────────────────────────── */}
      <DocumentViewerSheet
        doc={previewDoc}
        open={!!previewDoc}
        onOpenChange={(open) => {
          if (!open) setPreviewDoc(null);
        }}
        textContent={
          previewDoc ? getTextContent(previewDoc.file_name) : undefined
        }
        summaryData={
          previewDoc ? summaries.get(previewDoc.file_name) : undefined
        }
      />

      {/* ── Doc Lens warning toast ────────────────────────────────── */}
      <AnimatePresence>
        {docLensWarning && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-4 left-1/2 -translate-x-1/2 z-60 bg-destructive text-destructive-foreground px-4 py-2 rounded-lg text-xs shadow-lg max-w-md text-center"
          >
            {docLensWarning}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Doc Lens Overlay ──────────────────────────────────────── */}
      {/* Always in the DOM — DocLensProvider controls visibility so session
          state is never destroyed when the user closes the overlay. */}
      <DocLensOverlay
        summaries={summaries}
        searchScores={searchScores}
        agentTags={agentTags}
        getTextContent={getTextContent}
        cardVariant={cardVariant}
        cardUiByFileName={cardUiByFileName}
        bulkExpandedCommand={bulkExpandedCommand ?? undefined}
        onCardUiStateChange={updateCardUiState}
        onToggleHidden={toggleHidden}
        chatDocNames={chatDocNames}
        onToggleChatContext={toggleChatDoc}
        activeTagFilters={filters.tags}
        onTagClick={(tag) => toggleFilter("tags", tag)}
      />

      </div>
    </DocLensProvider>
  );
}
