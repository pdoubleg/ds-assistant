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
import { useAuditAgent } from "@/hooks/use-audit-agent";
import { useUploadedDocs, type UploadedDoc } from "@/hooks/use-uploaded-docs";
import { useChatDocs } from "@/hooks/use-chat-docs";
import {
  DocumentCard,
  type BulkExpandedCommand,
  deriveFileExt,
  type DocumentCardUiState,
  type DocumentSummaryData,
  type DocSearchData,
  type CardVariant,
} from "@/components/a2ui/documents";
import { DocumentViewerSheet } from "@/components/document-viewer-sheet";
import { getTagConfig, ALL_TAGS } from "@/lib/tag-registry";
import {
  FileUp,
  Sparkles,
  Search,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Send,
  Loader2,
  X,
  RotateCcw,
  Tag,
  ChevronDown,
  ChevronUp,
  Filter,
  EyeOff,
  Eye,
  Undo2,
  ScanSearch,
  Info,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";
import { ShineBorder } from "@/components/ui/shine-border";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import {
  DocLensOverlay,
  DocLensProvider,
} from "@/components/doc-lens";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";
const HIDDEN_DOC_NAMES_STORAGE_KEY = "agui_v3.hiddenDocNames.v1";
const HIDDEN_DOC_TIMESTAMPS_STORAGE_KEY = "agui_v3.hiddenDocTimestamps.v1";
const DOC_SUMMARIES_STORAGE_KEY = "agui_v3.docSummaries.v2";
const DOC_SEARCH_STORAGE_KEY = "agui_v3.docSearch.v1";
const DOC_TAGS_STORAGE_KEY = "agui_v3.docTags.v2";
const DOC_CARD_UI_STORAGE_KEY = "agui_v3.docCardUiState.v2";
const HIDDEN_DOCK_UI_STORAGE_KEY = "agui_v3.hiddenDockUi.v1";

// ── Types ──────────────────────────────────────────────────────────────

type DocWithId = UploadedDoc & { _id: string };
type SortKey = "default" | "score" | "date" | "title";
type HiddenSortKey =
  | "file_name"
  | "ext"
  | "domain"
  | "document_type"
  | "document_sub_type"
  | "create_date"
  | "source_system";

interface Filters {
  search: string;
  mimeTypes: Set<string>;
  docTypes: Set<string>;
  subTypes: Set<string>;
  domains: Set<string>;
  tags: Set<string>;
}

const EMPTY_FILTERS: Filters = {
  search: "",
  mimeTypes: new Set(),
  docTypes: new Set(),
  subTypes: new Set(),
  domains: new Set(),
  tags: new Set(),
};

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

function activeFilterCount(f: Filters): number {
  return (
    (f.search.length > 0 ? 1 : 0) +
    f.mimeTypes.size +
    f.docTypes.size +
    f.subTypes.size +
    f.domains.size +
    f.tags.size
  );
}

// ── Multi-select filter dropdown ───────────────────────────────────────

function FilterDropdown({
  label,
  options,
  selected,
  onToggle,
}: {
  label: string;
  options: string[];
  selected: Set<string>;
  onToggle: (value: string) => void;
}) {
  if (options.length === 0) return null;

  return (
    <Select
      value={selected.size === 1 ? [...selected][0] : "__multi__"}
      onValueChange={(v) => {
        if (v !== "__multi__") onToggle(v);
      }}
    >
      <SelectTrigger className="h-7 w-auto min-w-[80px] max-w-[120px] text-[11px]">
        <SelectValue>
          {selected.size > 0 ? `${label} (${selected.size})` : label}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {options.map((opt) => (
          <SelectItem
            key={opt}
            value={opt}
            className={cn(
              "text-xs",
              selected.has(opt) && "font-bold text-primary"
            )}
          >
            {selected.has(opt) ? `✓ ${opt}` : opt}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════

export function DocumentsPane() {
  const { state } = useAuditAgent();
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
  // Show tags only when documents pane is effectively full-width
  // (e.g., chat and output are collapsed).
  const showHiddenTagsColumn =
    cardVariant === "wide" && paneWidth / Math.max(viewportWidth, 1) >= 0.58;

  // ── Fetch example docs from backend on mount ────────────────────────

  const exampleDocsFetched = useRef(false);

  useEffect(() => {
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
            content_id: crypto.randomUUID(),
            mime_type: (d.mime_type as string) || "application/octet-stream",
            content_url: (d.path as string) || "",
            domain: "claim",
            document_type: "Example",
            document_description: `${d.file_size || ""}, ${d.page_count ?? 0} pages`,
            create_date: new Date().toISOString(),
            source_system: "EXAMPLE",
            content: (d.content as string) || "",
          });
        }
      } catch (err) {
        console.error("[ExampleDocs] Failed to load:", err);
      }
    })();
  }, [addUploadedDoc]);

  // ── Hidden state (docs explicitly hidden from doc-agent context) ────

  const [hiddenFileNames, setHiddenFileNames] = useState<Set<string>>(new Set());
  const [hiddenAtByFileName, setHiddenAtByFileName] = useState<Map<string, string>>(
    new Map()
  );
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

  const [agentTags, setAgentTags] = useState<Map<string, string[]>>(
    new Map()
  );

  // Derive filter options from the tags actually assigned to documents
  const allTagOptions = useMemo(() => {
    const seen = new Set<string>();
    for (const tags of agentTags.values()) {
      for (const t of tags) seen.add(t);
    }
    // Preserve the canonical order from ALL_TAGS
    return ALL_TAGS.filter((t) => seen.has(t));
  }, [agentTags]);
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
  const tagAbortRef = useRef<AbortController | null>(null);

  // ── Doc Lens state ──────────────────────────────────────────────────
  // openLensRef is populated by DocLensProvider and lets this component
  // trigger the overlay open without being inside the provider tree.
  const openLensRef = useRef<(() => void) | null>(null);
  const [docLensWarning, setDocLensWarning] = useState<string | null>(null);

  // Rehydrate summaries, search scores, tags, and card UI expansion state.
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

      const rawTags = window.localStorage.getItem(DOC_TAGS_STORAGE_KEY);
      if (rawTags) {
        const parsed = JSON.parse(rawTags) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          const entries = Object.entries(parsed as Record<string, unknown>)
            .filter(([k, v]) => typeof k === "string" && Array.isArray(v))
            .map(([k, v]) => [
              k,
              (v as unknown[]).filter((item): item is string => typeof item === "string"),
            ]) as [string, string[]][];
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

  // Persist summaries, search scores, tags, and card UI expansion state.
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
  }, [agentTags, cardUiByFileName, docEnrichmentHydrated, searchScores, summaries]);

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

  // Count only hidden docs that currently exist in the live document list.
  const hiddenCount = useMemo(
    () => allDocs.filter((d) => hiddenFileNames.has(d.file_name)).length,
    [allDocs, hiddenFileNames]
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

  const manuallyHiddenDocs = useMemo(
    () => allDocs.filter((d) => hiddenFileNames.has(d.file_name)),
    [allDocs, hiddenFileNames]
  );

  const nonHiddenDocs = useMemo(
    () => allDocs.filter((d) => !hiddenFileNames.has(d.file_name)),
    [allDocs, hiddenFileNames]
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
        return docTags.some((t) => filters.tags.has(t));
      });
    }

    return docs;
  }, [nonHiddenDocs, filters, agentTags]);

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
   * dock state. Stats are based on manually hidden docs.
   */
  const hiddenStats = useMemo(() => {
    const extensionCounts = new Map<string, number>();
    let latestHiddenAt: string | null = null;

    for (const doc of manuallyHiddenDocs) {
      const ext = deriveFileExt(doc.mime_type, doc.file_name).toUpperCase();
      extensionCounts.set(ext, (extensionCounts.get(ext) || 0) + 1);

      const hiddenAt = hiddenAtByFileName.get(doc.file_name);
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
  }, [allDocs.length, hiddenAtByFileName, hiddenCount, manuallyHiddenDocs]);

  // Sort manually-hidden docs for table rendering.
  const sortedManuallyHiddenDocs = useMemo(() => {
    const docs = [...manuallyHiddenDocs];

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
  }, [hiddenSortDir, hiddenSortKey, manuallyHiddenDocs]);

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

  const hideFiltered = useCallback(() => {
    const hiddenAt = new Date().toISOString();

    setHiddenFileNames((prev) => {
      const next = new Set(prev);
      for (const doc of filteredDocs) next.add(doc.file_name);
      return next;
    });

    setHiddenAtByFileName((prev) => {
      const next = new Map(prev);
      for (const doc of filteredDocs) {
        if (!next.has(doc.file_name)) next.set(doc.file_name, hiddenAt);
      }
      return next;
    });
  }, [filteredDocs]);

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
  }, []);

  const unhideAll = useCallback(() => {
    setHiddenFileNames(new Set());
    setHiddenAtByFileName(new Map());
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
      content: getTextContent(doc.file_name),
      mime_type: doc.mime_type,
      document_type: doc.document_type || "",
    }));
  }, [filteredDocs, getTextContent]);

  // ── Summarize via NDJSON stream ─────────────────────────────────────

  const runSummarize = useCallback(async () => {
    const payloads = buildVisiblePayloads();
    if (payloads.length === 0) return;

    setIsSummarizing(true);
    setSummarizeProgress({ done: 0, total: payloads.length });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const resp = await fetch(`${BACKEND_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: payloads }),
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
  }, [buildVisiblePayloads]);

  const cancelSummarize = useCallback(() => {
    abortRef.current?.abort();
    setIsSummarizing(false);
    setShowSummarizeInput(false);
  }, []);

  // ── Search & Sort via JSON response ─────────────────────────────────

  const runSearchSort = useCallback(async () => {
    if (!searchQuery.trim()) return;
    const docs = filteredDocs;
    if (docs.length === 0) return;

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
        body: JSON.stringify({ query: searchQuery, documents: payloads }),
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
        const zeroScoreFileNames: string[] = [];

        setSearchScores((prev) => {
          const next = new Map(prev);
          for (const s of data.scores) {
            const fileName =
              data.content_id_to_file_name[s.content_id] || s.content_id;
            next.set(fileName, { score: s.score, label: s.label });
            if (s.score === 0) zeroScoreFileNames.push(fileName);
          }
          return next;
        });

        // Auto-hide docs that scored 0 (excluded by the agent)
        if (zeroScoreFileNames.length > 0) {
          const hiddenAt = new Date().toISOString();
          setHiddenFileNames((prev) => {
            const next = new Set(prev);
            for (const name of zeroScoreFileNames) next.add(name);
            return next;
          });
          setHiddenAtByFileName((prev) => {
            const next = new Map(prev);
            for (const name of zeroScoreFileNames) {
              if (!next.has(name)) next.set(name, hiddenAt);
            }
            return next;
          });
        }
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

  const runAutoTag = useCallback(async () => {
    // Tag visible docs (not hidden) so the user can use tags to filter
    const payloads = buildVisiblePayloads();

    if (payloads.length === 0) return;

    setIsTagging(true);
    setTaggingProgress(null);

    const controller = new AbortController();
    tagAbortRef.current = controller;

    try {
      const resp = await fetch(`${BACKEND_URL}/document-tags`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents: payloads }),
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
                  tags: string[];
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
  }, [buildVisiblePayloads]);

  const cancelAutoTag = useCallback(() => {
    tagAbortRef.current?.abort();
    setIsTagging(false);
    setTaggingProgress(null);
  }, []);

  // ── Collect active filter chips for display ─────────────────────────

  const filterChips = useMemo(() => {
    const chips: Array<{ key: keyof Filters; value: string; label: string }> =
      [];
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

  // ═══════════════════════════════════════════════════════════════════
  // Render
  // ═══════════════════════════════════════════════════════════════════

  return (
    <DocLensProvider docs={docLensEligibleDocs} openLensRef={openLensRef}>
    <div ref={containerRef} className="relative flex flex-col h-full overflow-hidden">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-4 py-3 pr-10 border-b border-border/50">
        <FileUp className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-foreground">Documents</h2>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="text-[10px] text-muted-foreground"></span>
          <div className="flex items-center rounded-md border border-border/60 overflow-hidden">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={hideAll}
              disabled={allDocs.length === 0 || hiddenCount >= allDocs.length}
              className="h-6 rounded-none px-2 text-[10px] text-muted-foreground hover:text-foreground"
            >
              Hide
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={collapseAllNonHidden}
              disabled={nonHiddenDocs.length === 0}
              className="h-6 rounded-none border-l border-border/60 px-2 text-[10px] text-muted-foreground hover:text-foreground"
            >
              Collapse
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={expandAllNonHidden}
              disabled={nonHiddenDocs.length === 0}
              className="h-6 rounded-none border-l border-border/60 px-2 text-[10px] text-muted-foreground hover:text-foreground"
            >
              Expand
            </Button>
          </div>
          <Badge
            variant="secondary"
            className="text-[10px]"
          >
            {allDocs.length} docs
          </Badge>
          <Badge className="text-[10px] bg-primary/15 text-primary border-primary/30">
            {filteredDocs.length} visible
          </Badge>
          {hiddenCount > 0 && (
            <Badge variant="outline" className="text-[10px] gap-0.5">
              <EyeOff className="h-2.5 w-2.5" />
              {hiddenCount}
            </Badge>
          )}
        </div>
      </div>

      {/* ── Toolbar ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-border/30 min-w-0">
        <NativeTooltip content="Clear generated AI content" side="bottom" animation="blur" shine>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-7 text-[11px] gap-1"
            onClick={refreshDocs}
            disabled={
              (summaries.size === 0 && searchScores.size === 0 && agentTags.size === 0 && allTagOptions.length === 0) ||
              isSummarizing ||
              isSearching ||
              isTagging
            }
          >
            <RotateCcw className="h-3 w-3" />
            Refresh docs
          </Button>
        </NativeTooltip>

        {/* Filter toggle (narrow) / inline dropdowns (medium/wide) + sort — shrinks to yield space to agent group */}
        <div className="flex items-center gap-1.5 min-w-0 shrink overflow-hidden">
        {/* Filter toggle (narrow) / inline dropdowns (medium/wide) */}
        {isNarrow ? (
          <NativeTooltip content="Filters" side="bottom" animation="blur">
            <Button
              variant={showFilters ? "secondary" : "ghost"}
              size="icon"
              className="h-7 w-7 relative"
              onClick={() => setShowFilters((p) => !p)}
            >
              <Filter className="h-3.5 w-3.5" />
              {activeFilterCount(filters) > 0 && (
                <span className="absolute -top-0.5 -right-0.5 h-3.5 w-3.5 rounded-full bg-primary text-[8px] text-primary-foreground flex items-center justify-center font-bold">
                  {activeFilterCount(filters)}
                </span>
              )}
            </Button>
          </NativeTooltip>
        ) : (
          <>
            <FilterDropdown
              label="Type"
              options={filterOptions.mimeTypes}
              selected={filters.mimeTypes}
              onToggle={(v) => toggleFilter("mimeTypes", v)}
            />
            <FilterDropdown
              label="Doc Type"
              options={filterOptions.docTypes}
              selected={filters.docTypes}
              onToggle={(v) => toggleFilter("docTypes", v)}
            />
            {filterOptions.subTypes.length > 0 && (
              <FilterDropdown
                label="Sub Type"
                options={filterOptions.subTypes}
                selected={filters.subTypes}
                onToggle={(v) => toggleFilter("subTypes", v)}
              />
            )}
            <FilterDropdown
              label="Domain"
              options={filterOptions.domains}
              selected={filters.domains}
              onToggle={(v) => toggleFilter("domains", v)}
            />
            {allTagOptions.length > 0 && (
              <FilterDropdown
                label="Tags"
                options={allTagOptions}
                selected={filters.tags}
                onToggle={(v) => toggleFilter("tags", v)}
              />
            )}
          </>
        )}

        {/* Sort */}
        <div className="flex items-center gap-1 shrink-0">
          <ArrowUpDown className="h-3 w-3 text-muted-foreground" />
          <Select
            value={sortKey}
            onValueChange={(v) => setSortKey(v as SortKey)}
          >
            <SelectTrigger className="h-7 w-[90px] text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">Default</SelectItem>
              <SelectItem value="score">Score</SelectItem>
              <SelectItem value="date">Date</SelectItem>
              <SelectItem value="title">Title</SelectItem>
            </SelectContent>
          </Select>
        </div>
        </div>{/* end filter+sort shrink group */}

        {/* Agent actions — stacked: scope label centered above the four AI buttons */}
        <div className="flex flex-col items-end gap-1.5 ml-auto rounded-md border border-border/40 px-2 pt-1.5 pb-1.5">
          {/* Context hint: shown whenever there are docs to act on */}
          {filteredDocs.length > 0 && (
            <NativeTooltip
              content={`All ${filteredDocs.length} visible doc${filteredDocs.length !== 1 ? "s" : ""} are sent to the model when you run any AI action. Hide or filter docs to narrow the scope.`}
              side="top"
              align="end"
            >
              <span className="flex items-center gap-1 text-[11px] text-muted-foreground/60 cursor-default select-none w-full justify-center">
                <Info className="h-3 w-3 shrink-0" />
                <span>{filteredDocs.length} doc{filteredDocs.length !== 1 ? "s" : ""} in scope</span>
              </span>
            </NativeTooltip>
          )}
          <div className="flex items-center gap-1">
          {/* Auto-tag */}
          <NativeTooltip
            content={isTagging ? "Cancel auto-tagging" : "Auto-tag"}
            side="bottom"
            animation="blur"
            shine
          >
            <Button
              variant="outline"
              size="sm"
              className="relative h-7 gap-1 overflow-hidden text-[11px]"
              onClick={isTagging ? cancelAutoTag : runAutoTag}
              disabled={filteredDocs.length === 0 && !isTagging}
            >
              {!isTagging ? (
                <ShineBorder
                  borderWidth={1}
                  duration={16}
                  shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
                />
              ) : null}
              <span className="relative z-10 inline-flex items-center gap-1">
                {isTagging ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {taggingProgress
                      ? `${taggingProgress.batch}/${taggingProgress.totalBatches}`
                      : "..."}
                  </>
                ) : (
                  <>
                    <Tag className="h-3 w-3" />
                    {isNarrow ? "" : "Auto-tag"}
                  </>
                )}
              </span>
            </Button>
          </NativeTooltip>

          {/* Summarize */}
          <NativeTooltip
            content={isSummarizing ? "Cancel summarization" : "Summarize"}
            side="bottom"
            animation="blur"
            shine
          >
            <Button
              variant={showSummarizeInput ? "secondary" : "outline"}
              size="sm"
              className="relative h-7 gap-1 overflow-hidden text-[11px]"
              onClick={() => {
                if (isSummarizing) cancelSummarize();
                else {
                  setShowSummarizeInput((p) => !p);
                  setShowSearchInput(false);
                }
              }}
              disabled={filteredDocs.length === 0 && !isSummarizing}
            >
              {!isSummarizing ? (
                <ShineBorder
                  borderWidth={1}
                  duration={16}
                  shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
                />
              ) : null}
              <span className="relative z-10 inline-flex items-center gap-1">
                {isSummarizing ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {summarizeProgress.done}/{summarizeProgress.total}
                  </>
                ) : (
                  <>
                    <Sparkles className="h-3 w-3" />
                    {isNarrow ? "" : "Summarize"}
                  </>
                )}
              </span>
            </Button>
          </NativeTooltip>

          {/* Search & Sort */}
          <NativeTooltip
            content={isSearching ? "Cancel search" : "Search & Sort"}
            side="bottom"
            animation="blur"
            shine
          >
            <Button
              variant={showSearchInput ? "secondary" : "outline"}
              size="sm"
              className="relative h-7 gap-1 overflow-hidden text-[11px]"
              onClick={() => {
                if (isSearching) cancelSearchSort();
                else {
                  setShowSearchInput((p) => !p);
                  setShowSummarizeInput(false);
                }
              }}
              disabled={filteredDocs.length === 0 && !isSearching}
            >
              {!isSearching ? (
                <ShineBorder
                  borderWidth={1}
                  duration={16}
                  shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
                />
              ) : null}
              <span className="relative z-10 inline-flex items-center gap-1">
                {isSearching ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Scoring...
                  </>
                ) : (
                  <>
                    <Search className="h-3 w-3" />
                    {isNarrow ? "" : "Search & Sort"}
                  </>
                )}
              </span>
            </Button>
          </NativeTooltip>

          {/* Doc Lens */}
          <NativeTooltip content="Document image search (Doc Lens)" side="bottom" animation="blur" shine>
            <Button
              variant="outline"
              size="sm"
              className="relative h-7 gap-1 overflow-hidden text-[11px]"
              onClick={handleDocLensClick}
              disabled={filteredDocs.length === 0}
            >
              <ShineBorder
                borderWidth={1}
                duration={16}
                shineColor={["#A07CFE", "#FE8FB5", "#FFBE7B"]}
              />
              <span className="relative z-10 inline-flex items-center gap-1">
                <ScanSearch className="h-3 w-3" />
                {isNarrow ? "" : "Doc Lens"}
              </span>
            </Button>
          </NativeTooltip>

          </div>{/* end buttons row */}
        </div>{/* end agent actions */}
      </div>

      {/* ── Narrow-mode filter panel (collapsible) ──────────────── */}
      <AnimatePresence>
        {isNarrow && showFilters && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-border/30"
          >
            <div className="flex items-center gap-1.5 px-3 py-2 flex-wrap">
              <FilterDropdown
                label="Type"
                options={filterOptions.mimeTypes}
                selected={filters.mimeTypes}
                onToggle={(v) => toggleFilter("mimeTypes", v)}
              />
              <FilterDropdown
                label="Doc Type"
                options={filterOptions.docTypes}
                selected={filters.docTypes}
                onToggle={(v) => toggleFilter("docTypes", v)}
              />
              <FilterDropdown
                label="Domain"
                options={filterOptions.domains}
                selected={filters.domains}
                onToggle={(v) => toggleFilter("domains", v)}
              />
              {allTagOptions.length > 0 && (
                <FilterDropdown
                  label="Tags"
                  options={allTagOptions}
                  selected={filters.tags}
                  onToggle={(v) => toggleFilter("tags", v)}
                />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Summarize confirmation bar ──────────────────────────── */}
      <AnimatePresence>
        {showSummarizeInput && !isSummarizing && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-border/30"
          >
            <div className="flex items-center gap-1.5 px-3 py-2">
              <span className="text-xs text-muted-foreground flex-1">
                Summarize {filteredDocs.length} visible document{filteredDocs.length !== 1 ? "s" : ""}?
              </span>
              <NativeTooltip content="Run summarization" side="bottom" animation="blur">
                <Button
                  variant="default"
                  size="sm"
                  className="h-7 text-[11px] gap-1 shrink-0"
                  onClick={runSummarize}
                  disabled={filteredDocs.length === 0}
                >
                  <Send className="h-3 w-3" />
                  Run
                </Button>
              </NativeTooltip>
              <NativeTooltip content="Cancel" side="bottom" animation="blur">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 shrink-0"
                  onClick={() => setShowSummarizeInput(false)}
                >
                  <X className="h-3 w-3" />
                </Button>
              </NativeTooltip>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Search & Sort query input ────────────────────────────── */}
      <AnimatePresence>
        {showSearchInput && !isSearching && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-border/30"
          >
            <div className="flex items-center gap-1.5 px-3 py-2">
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="e.g. &quot;find all estimates&quot; or &quot;rank by relevance to roofing&quot;"
                className="h-7 text-xs flex-1"
                onKeyDown={(e) => {
                  if (e.key === "Enter") runSearchSort();
                }}
                autoFocus
              />
              <NativeTooltip content="Run search & sort" side="bottom" animation="blur">
                <Button
                  variant="default"
                  size="icon"
                  className="h-7 w-7 shrink-0"
                  onClick={runSearchSort}
                  disabled={filteredDocs.length === 0 || !searchQuery.trim()}
                >
                  <Send className="h-3 w-3" />
                </Button>
              </NativeTooltip>
              <NativeTooltip content="Cancel" side="bottom" animation="blur">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 shrink-0"
                  onClick={() => setShowSearchInput(false)}
                >
                  <X className="h-3 w-3" />
                </Button>
              </NativeTooltip>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Active filter chips ──────────────────────────────────── */}
      {filterChips.length > 0 && (
        <div className="flex items-center gap-1 px-3 py-1.5 border-b border-border/20 flex-wrap">
          {filterChips.map((chip, i) => {
            const isTagChip = chip.key === "tags";
            const tagCfg = isTagChip ? getTagConfig(chip.value) : null;
            const TagIcon = tagCfg?.icon;
            return (
              <Badge
                key={`${chip.key}-${chip.value}-${i}`}
                variant="secondary"
                className={cn(
                  "text-[10px] px-1.5 py-0 gap-1 cursor-pointer hover:bg-destructive/20 inline-flex items-center",
                  isTagChip && tagCfg && `${tagCfg.bg} ${tagCfg.text} ${tagCfg.border}`,
                )}
                onClick={() => removeFilterChip(chip.key, chip.value)}
              >
                {TagIcon && <TagIcon className="h-2.5 w-2.5 shrink-0" />}
                {chip.label}
                <X className="h-2.5 w-2.5" />
              </Badge>
            );
          })}
          <button
            onClick={clearAllFilters}
            className="text-[10px] text-muted-foreground hover:text-foreground ml-1"
          >
            Clear all
          </button>
          {/* Hide all filtered docs */}
          {filteredDocs.length > 0 && (
            <NativeTooltip content="Hide all visible docs from doc agent context" side="bottom" animation="blur">
              <button
                onClick={hideFiltered}
                className="ml-auto flex items-center gap-0.5 text-[10px] text-muted-foreground hover:text-foreground"
              >
                <EyeOff className="h-3 w-3" />
                Hide {filteredDocs.length}
              </button>
            </NativeTooltip>
          )}
        </div>
      )}

      {/* ── Search/Triage document list ─────────────────────────── */}
      <ScrollArea className="flex-1 relative z-0">
        <div
          className={cn(
            "px-2 py-2 space-y-1.5",
            showHiddenDock &&
              (dockMinimized
                ? "pb-[52px]"
                : dockExpanded
                  ? "pb-[360px]"
                  : "pb-[88px]")
          )}
        >
          <AnimatePresence>
            {sortedDocs.map((doc) => (
              <motion.div
                key={doc._id}
                layout
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.2 }}
              >
                <DocumentCard
                  file_name={doc.file_name}
                  mime_type={doc.mime_type}
                  content_id={doc.content_id}
                  claim_number={doc.claim_number}
                  content_url={doc.content_url}
                  domain={doc.domain}
                  document_type={doc.document_type}
                  document_sub_type={doc.document_sub_type}
                  document_description={doc.document_description}
                  create_date={doc.create_date}
                  source_system={doc.source_system}
                  company_name={doc.company_name}
                  summaryData={summaries.get(doc.file_name)}
                  searchData={searchScores.get(doc.file_name)}
                  tags={agentTags.get(doc.file_name)}
                  variant={cardVariant}
                  isHidden={false}
                  onToggleHidden={() => toggleHidden(doc.file_name)}
                  isInChatContext={chatDocNames.has(doc.file_name)}
                  onToggleChatContext={() => toggleChatDoc(doc.file_name)}
                  onTagClick={(tag) => toggleFilter("tags", tag)}
                  activeTagFilters={filters.tags}
                  onPreview={() => setPreviewDoc(doc)}
                  initialUiState={cardUiByFileName.get(doc.file_name)}
                  bulkExpandedCommand={bulkExpandedCommand ?? undefined}
                  onUiStateChange={(nextState) =>
                    updateCardUiState(doc.file_name, nextState)
                  }
                />
              </motion.div>
            ))}
          </AnimatePresence>

          {sortedDocs.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground/60 gap-2">
              <FileUp className="h-8 w-8" />
              <p className="text-sm">
                {hasActiveFilters(filters)
                  ? "No documents match the current filters."
                  : "No documents in triage. Upload or load documents to begin."}
              </p>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* ── "Hidden" Dock ────────────────────────────────────────── */}
      {showHiddenDock && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 px-2 pb-2">
          <div className="pointer-events-auto rounded-lg border border-border/60 bg-background/95 shadow-lg backdrop-blur-sm">
            {/* Dock header */}
            <div className="flex items-center gap-2 px-2 pt-2 pb-1.5">
              <button
                className="group flex flex-1 items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted/45 transition-colors"
                onClick={() => {
                  if (dockMinimized) {
                    setDockMinimized(false);
                    return;
                  }
                  setDockExpanded((prev) => !prev);
                }}
              >
                <EyeOff className="h-4 w-4 text-muted-foreground" />
                <div className="min-w-0 text-left">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-semibold text-foreground">
                      Hidden Documents
                    </span>
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                      {hiddenCount}
                    </Badge>
                    {filterHiddenCount > 0 && (
                      <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                        {filterHiddenCount} filtered
                      </Badge>
                    )}
                  </div>
                  {!dockMinimized && (
                    <p className="text-[10px] text-muted-foreground">
                      {dockExpanded
                        ? "Close table view"
                        : "Open table view for hidden documents"}
                    </p>
                  )}
                </div>
                {dockMinimized ? (
                  <ChevronUp className="ml-auto h-4 w-4 text-muted-foreground" />
                ) : dockExpanded ? (
                  <ChevronDown className="ml-auto h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronUp className="ml-auto h-4 w-4 text-muted-foreground" />
                )}
              </button>

              {!dockMinimized && hiddenCount > 1 && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 text-[11px]"
                  onClick={unhideAll}
                >
                  Unhide all
                </Button>
              )}

              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() => setDockMinimized((prev) => !prev)}
                title={dockMinimized ? "Expand hidden dock" : "Minimize hidden dock"}
              >
                {dockMinimized ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>
            </div>

            {/* Collapsed helper stats */}
            {!dockMinimized && !dockExpanded && (
              <div className="border-t border-border/35 px-3 py-2">
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                  <span>
                    Hidden: <strong className="text-foreground">{hiddenCount}</strong>
                  </span>
                  <span>
                    Hidden %:{" "}
                    <strong className="text-foreground">{hiddenStats.hiddenPercent}%</strong>
                  </span>
                  <span>
                    Filtered out:{" "}
                    <strong className="text-foreground">{filterHiddenCount}</strong>
                  </span>
                  <span className="truncate">
                    Top types:{" "}
                    <strong className="text-foreground">
                      {hiddenStats.topExtensions.length > 0
                        ? hiddenStats.topExtensions
                            .map(([ext, count]) => `${ext} (${count})`)
                            .join(", ")
                        : "N/A"}
                    </strong>
                  </span>
                  <span className="col-span-2">
                    Latest hidden:{" "}
                    <strong className="text-foreground">
                      {hiddenStats.latestHiddenAt
                        ? new Date(hiddenStats.latestHiddenAt).toLocaleString()
                        : "N/A"}
                    </strong>
                  </span>
                </div>
              </div>
            )}

            {/* Expanded content */}
            <AnimatePresence>
              {!dockMinimized && dockExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden border-t border-border/45"
                >
                  <div className="px-3 py-2">
                    {manuallyHiddenDocs.length === 0 ? (
                      <div className="rounded-md border border-dashed border-border/50 bg-muted/25 px-3 py-3">
                        <div className="flex items-center gap-1.5 flex-wrap text-[11px]">
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            Hidden: 0
                          </Badge>
                          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                            Filtered out: {filterHiddenCount}
                          </Badge>
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            Hidden %: {hiddenStats.hiddenPercent}%
                          </Badge>
                        </div>
                      </div>
                    ) : (
                      <div className="max-h-[min(62vh,560px)] overflow-auto rounded-md border border-border/50">
                        <Table className="min-w-full text-sm">
                          <TableHeader>
                            <TableRow className="sticky top-0 z-10 bg-secondary/70 backdrop-blur-sm">
                              <TableHead
                                className="min-w-[220px] cursor-pointer select-none"
                                onClick={() => handleHiddenSort("file_name")}
                              >
                                <div className="flex items-center gap-1">
                                  Document
                                  {hiddenSortKey === "file_name" ? (
                                    hiddenSortDir === "asc" ? (
                                      <ArrowUp className="h-3 w-3" />
                                    ) : (
                                      <ArrowDown className="h-3 w-3" />
                                    )
                                  ) : (
                                    <ArrowUpDown className="h-3 w-3 opacity-30" />
                                  )}
                                </div>
                              </TableHead>
                              <TableHead
                                className="w-[80px] cursor-pointer select-none"
                                onClick={() => handleHiddenSort("ext")}
                              >
                                <div className="flex items-center gap-1">
                                  Type
                                  {hiddenSortKey === "ext" ? (
                                    hiddenSortDir === "asc" ? (
                                      <ArrowUp className="h-3 w-3" />
                                    ) : (
                                      <ArrowDown className="h-3 w-3" />
                                    )
                                  ) : (
                                    <ArrowUpDown className="h-3 w-3 opacity-30" />
                                  )}
                                </div>
                              </TableHead>
                              {!isCompactHiddenTable && (
                                <>
                                  <TableHead
                                    className="w-[90px] cursor-pointer select-none"
                                    onClick={() => handleHiddenSort("domain")}
                                  >
                                    <div className="flex items-center gap-1">
                                      Domain
                                      {hiddenSortKey === "domain" ? (
                                        hiddenSortDir === "asc" ? (
                                          <ArrowUp className="h-3 w-3" />
                                        ) : (
                                          <ArrowDown className="h-3 w-3" />
                                        )
                                      ) : (
                                        <ArrowUpDown className="h-3 w-3 opacity-30" />
                                      )}
                                    </div>
                                  </TableHead>
                                  <TableHead
                                    className="min-w-[120px] cursor-pointer select-none"
                                    onClick={() => handleHiddenSort("document_type")}
                                  >
                                    <div className="flex items-center gap-1">
                                      Doc Type
                                      {hiddenSortKey === "document_type" ? (
                                        hiddenSortDir === "asc" ? (
                                          <ArrowUp className="h-3 w-3" />
                                        ) : (
                                          <ArrowDown className="h-3 w-3" />
                                        )
                                      ) : (
                                        <ArrowUpDown className="h-3 w-3 opacity-30" />
                                      )}
                                    </div>
                                  </TableHead>
                                  <TableHead
                                    className="min-w-[120px] cursor-pointer select-none"
                                    onClick={() => handleHiddenSort("document_sub_type")}
                                  >
                                    <div className="flex items-center gap-1">
                                      Subtype
                                      {hiddenSortKey === "document_sub_type" ? (
                                        hiddenSortDir === "asc" ? (
                                          <ArrowUp className="h-3 w-3" />
                                        ) : (
                                          <ArrowDown className="h-3 w-3" />
                                        )
                                      ) : (
                                        <ArrowUpDown className="h-3 w-3 opacity-30" />
                                      )}
                                    </div>
                                  </TableHead>
                                  <TableHead
                                    className="w-[115px] cursor-pointer select-none"
                                    onClick={() => handleHiddenSort("create_date")}
                                  >
                                    <div className="flex items-center gap-1">
                                      Created
                                      {hiddenSortKey === "create_date" ? (
                                        hiddenSortDir === "asc" ? (
                                          <ArrowUp className="h-3 w-3" />
                                        ) : (
                                          <ArrowDown className="h-3 w-3" />
                                        )
                                      ) : (
                                        <ArrowUpDown className="h-3 w-3 opacity-30" />
                                      )}
                                    </div>
                                  </TableHead>
                                  <TableHead
                                    className="min-w-[100px] cursor-pointer select-none"
                                    onClick={() => handleHiddenSort("source_system")}
                                  >
                                    <div className="flex items-center gap-1">
                                      Source
                                      {hiddenSortKey === "source_system" ? (
                                        hiddenSortDir === "asc" ? (
                                          <ArrowUp className="h-3 w-3" />
                                        ) : (
                                          <ArrowDown className="h-3 w-3" />
                                        )
                                      ) : (
                                        <ArrowUpDown className="h-3 w-3 opacity-30" />
                                      )}
                                    </div>
                                  </TableHead>
                                  {showHiddenTagsColumn && (
                                    <TableHead className="min-w-[150px]">
                                      Tags
                                    </TableHead>
                                  )}
                                </>
                              )}
                              {isCompactHiddenTable && (
                                <TableHead
                                  className="w-[160px] cursor-pointer select-none"
                                  onClick={() => handleHiddenSort("create_date")}
                                >
                                  <div className="flex items-center gap-1">
                                    Created
                                    {hiddenSortKey === "create_date" ? (
                                      hiddenSortDir === "asc" ? (
                                        <ArrowUp className="h-3 w-3" />
                                      ) : (
                                        <ArrowDown className="h-3 w-3" />
                                      )
                                    ) : (
                                      <ArrowUpDown className="h-3 w-3 opacity-30" />
                                    )}
                                  </div>
                                </TableHead>
                              )}
                              <TableHead className="sticky right-0 z-20 w-[88px] bg-secondary/70">
                                Actions
                              </TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {sortedManuallyHiddenDocs.map((doc) => {
                              const ext = deriveFileExt(
                                doc.mime_type,
                                doc.file_name
                              ).toUpperCase();
                              const createDate = doc.create_date
                                ? new Date(doc.create_date).toLocaleDateString()
                                : "—";
                              const docTags = agentTags.get(doc.file_name) || [];
                              const visibleTags = docTags.slice(0, 2);
                              const overflowTagCount =
                                docTags.length > visibleTags.length
                                  ? docTags.length - visibleTags.length
                                  : 0;

                              return (
                                <TableRow key={doc._id} className="odd:bg-muted/15">
                                  <TableCell className="max-w-[350px]">
                                    <div className="truncate font-medium text-foreground">
                                      {doc.file_name}
                                    </div>
                                    <div className="text-[10px] text-muted-foreground truncate">
                                      {doc.document_description || "No description"}
                                    </div>
                                  </TableCell>
                                  <TableCell>
                                    <Badge
                                      variant="outline"
                                      className="font-mono text-[10px] px-1.5 py-0 bg-primary/10 text-primary border-primary/30"
                                    >
                                      {ext}
                                    </Badge>
                                  </TableCell>

                                  {!isCompactHiddenTable && (
                                    <>
                                      <TableCell>
                                        <Badge
                                          variant="outline"
                                          className={cn(
                                            "text-[10px] px-1.5 py-0",
                                            doc.domain === "policy"
                                              ? "text-violet-700 dark:text-violet-400 border-violet-500/30 bg-violet-500/10"
                                              : "text-blue-700 dark:text-blue-400 border-blue-500/30 bg-blue-500/10"
                                          )}
                                        >
                                          {doc.domain || "claim"}
                                        </Badge>
                                      </TableCell>
                                      <TableCell className="max-w-[160px] truncate">
                                        {doc.document_type ? (
                                          <Badge
                                            variant="secondary"
                                            className="text-[10px] px-1.5 py-0"
                                          >
                                            {doc.document_type}
                                          </Badge>
                                        ) : (
                                          "—"
                                        )}
                                      </TableCell>
                                      <TableCell className="max-w-[160px] truncate">
                                        {doc.document_sub_type ? (
                                          <Badge
                                            variant="secondary"
                                            className="text-[10px] px-1.5 py-0"
                                          >
                                            {doc.document_sub_type}
                                          </Badge>
                                        ) : (
                                          "—"
                                        )}
                                      </TableCell>
                                      <TableCell className="text-xs text-muted-foreground">
                                        {createDate}
                                      </TableCell>
                                      <TableCell className="max-w-[120px] truncate">
                                        {doc.source_system || "—"}
                                      </TableCell>
                                      {showHiddenTagsColumn && (
                                        <TableCell className="max-w-[220px]">
                                          {visibleTags.length > 0 ? (
                                            <div className="flex items-center gap-1 flex-wrap">
                                              {visibleTags.map((tag) => {
                                                const cfg = getTagConfig(tag);
                                                const Icon = cfg.icon;
                                                return (
                                                  <Badge
                                                    key={`${doc._id}-${tag}`}
                                                    variant="outline"
                                                    className={`text-[10px] px-1.5 py-0 inline-flex items-center gap-0.5 ${cfg.bg} ${cfg.text} ${cfg.border}`}
                                                  >
                                                    <Icon className="h-2.5 w-2.5 shrink-0" />
                                                    {tag}
                                                  </Badge>
                                                );
                                              })}
                                              {overflowTagCount > 0 && (
                                                <Badge
                                                  variant="secondary"
                                                  className="text-[10px] px-1.5 py-0"
                                                >
                                                  +{overflowTagCount}
                                                </Badge>
                                              )}
                                            </div>
                                          ) : (
                                            <span className="text-muted-foreground">—</span>
                                          )}
                                        </TableCell>
                                      )}
                                    </>
                                  )}

                                  {isCompactHiddenTable && (
                                    <TableCell className="text-xs text-muted-foreground">
                                      {createDate}
                                    </TableCell>
                                  )}

                                  <TableCell className="sticky right-0 z-10 w-[88px] bg-background/95">
                                    <div className="flex items-center justify-end gap-1">
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="sm"
                                        className="h-7 w-7 p-0"
                                        onClick={() => setPreviewDoc(doc)}
                                        title="View document"
                                      >
                                        <Eye className="h-4 w-4" />
                                      </Button>
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="sm"
                                        className="h-7 w-7 p-0"
                                        onClick={() => unhide(doc.file_name)}
                                        title="Unhide document"
                                      >
                                        <Undo2 className="h-3.5 w-3.5" />
                                      </Button>
                                    </div>
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}

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
