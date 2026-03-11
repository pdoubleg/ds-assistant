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
  type DocumentTagData,
  type DocumentSummaryData,
  type DocSearchData,
  type CardVariant,
} from "@/components/a2ui/documents";
import { DocumentViewerSheet } from "@/components/document-viewer-sheet";
import {
  CUSTOM_FALLBACK_TAG_LABEL,
  getTagConfig,
  getDefaultTagIconName,
  isTagIconName,
  ALL_TAGS,
} from "@/lib/tag-registry";
import {
  AlertTriangle,
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
  Plus,
  Square,
  CheckSquare,
  Trash2,
  ChevronsDownUp,
  ChevronsUpDown,
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
const HIDDEN_DOCK_UI_STORAGE_KEY = "agui_v4.hiddenDockUi.v1";

// ── Types ──────────────────────────────────────────────────────────────

type DocWithId = UploadedDoc & { _id: string };
type SortKey = "default" | "score" | "date" | "title";
type AutoTagMode = "default" | "custom";
type TagFilterMode = "or" | "and";
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
  const [tagFilterMode, setTagFilterMode] = useState<TagFilterMode>("or");

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
        return matchesTagFilters(docTags, filters.tags, tagFilterMode);
      });
    }

    return docs;
  }, [nonHiddenDocs, filters, agentTags, tagFilterMode]);

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
    const trimmedAdditionalInstructions = summarizeInstructions.trim();

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
    <div ref={containerRef} className="relative flex flex-col h-full overflow-hidden">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="flex items-center gap-2.5 px-4 pr-10 border-b border-border/50 h-12">
        <FileUp className="h-[18px] w-[18px] text-primary shrink-0" />
        <h2 className="text-[15px] font-semibold tracking-tight text-foreground shrink-0">Documents</h2>
        <div className="ml-auto flex items-center gap-2">
          {/* Icon-only action group: Show All · Hide All · Expand · Collapse */}
          <div className="flex items-center rounded-lg border border-border/60 bg-secondary/30 overflow-hidden">
            <NativeTooltip content="Show All" side="bottom">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={unhideAll}
                disabled={hiddenCount === 0}
                className={cn(
                  "h-7 w-7 rounded-none",
                  hiddenCount === 0
                    ? "text-muted-foreground/40"
                    : hiddenCount >= allDocs.length
                      ? "bg-primary/10 text-primary hover:bg-primary/15"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                )}
              >
                <Eye className="h-3.5 w-3.5" />
              </Button>
            </NativeTooltip>
            <NativeTooltip content="Hide All" side="bottom">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={hideAll}
                disabled={allDocs.length === 0 || hiddenCount >= allDocs.length}
                className={cn(
                  "h-7 w-7 rounded-none border-l border-border/60",
                  hiddenCount >= allDocs.length
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                )}
              >
                <EyeOff className="h-3.5 w-3.5" />
              </Button>
            </NativeTooltip>
            <NativeTooltip content="Expand All" side="bottom">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={expandAllNonHidden}
                disabled={nonHiddenDocs.length === 0}
                className={cn(
                  "h-7 w-7 rounded-none border-l border-border/60",
                  bulkExpandedCommand?.expanded === true
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                )}
              >
                <ChevronsUpDown className="h-3.5 w-3.5" />
              </Button>
            </NativeTooltip>
            <NativeTooltip content="Collapse All" side="bottom">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={collapseAllNonHidden}
                disabled={nonHiddenDocs.length === 0}
                className={cn(
                  "h-7 w-7 rounded-none border-l border-border/60",
                  bulkExpandedCommand?.expanded === false
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                )}
              >
                <ChevronsDownUp className="h-3.5 w-3.5" />
              </Button>
            </NativeTooltip>
          </div>
          <Badge variant="secondary" className="text-[11px]">
            {allDocs.length} docs
          </Badge>
          <Badge className="text-[11px] bg-primary/15 text-primary border-primary/30">
            {filteredDocs.length} visible
          </Badge>
          {hiddenCount > 0 && (
            <Badge variant="outline" className="text-[11px] gap-1">
              <EyeOff className="h-3 w-3" />
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
        {/* Filter toggle (narrow/medium) / inline dropdowns (docs-only) */}
        {useNarrowToolbar ? (
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
            content={
              isTagging
                ? "Cancel auto-tagging"
                : filteredDocs.length === 0
                  ? "Requires at least one document"
                  : "Auto-tag"
            }
            side="bottom"
            animation="blur"
            shine
          >
            <span className="inline-flex">
              <Button
                variant={showTagConfirm ? "secondary" : "outline"}
                size="sm"
                className="relative h-7 gap-1 overflow-hidden text-[11px]"
                onClick={() => {
                  if (isTagging) cancelAutoTag();
                  else {
                    setShowTagConfirm((p) => !p);
                    setShowSummarizeInput(false);
                    setShowSearchInput(false);
                  }
                }}
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
                      {iconOnlyButtons ? "" : "Auto-tag"}
                    </>
                  )}
                </span>
              </Button>
            </span>
          </NativeTooltip>

          {/* Search & Sort */}
          <NativeTooltip
            content={
              isSearching
                ? "Cancel search"
                : filteredDocs.length < 2
                  ? "Requires at least two documents"
                  : "Search & Sort"
            }
            side="bottom"
            animation="blur"
            shine
          >
            <span className="inline-flex">
              <Button
                variant={showSearchInput ? "secondary" : "outline"}
                size="sm"
                className="relative h-7 gap-1 overflow-hidden text-[11px]"
                onClick={() => {
                  if (isSearching) cancelSearchSort();
                  else {
                    setShowSearchInput((p) => !p);
                    setShowTagConfirm(false);
                    setShowSummarizeInput(false);
                  }
                }}
                disabled={filteredDocs.length < 2 && !isSearching}
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
                      {iconOnlyButtons ? "" : "Search & Sort"}
                    </>
                  )}
                </span>
              </Button>
            </span>
          </NativeTooltip>

          {/* Summarize */}
          <NativeTooltip
            content={
              isSummarizing
                ? "Cancel summarization"
                : filteredDocs.length === 0
                  ? "Requires at least one document"
                  : "Summarize"
            }
            side="bottom"
            animation="blur"
            shine
          >
            <span className="inline-flex">
              <Button
                variant={showSummarizeInput ? "secondary" : "outline"}
                size="sm"
                className="relative h-7 gap-1 overflow-hidden text-[11px]"
                onClick={() => {
                  if (isSummarizing) cancelSummarize();
                  else {
                    setShowSummarizeInput((p) => !p);
                    setShowTagConfirm(false);
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
                      {iconOnlyButtons ? "" : "Summarize"}
                    </>
                  )}
                </span>
              </Button>
            </span>
          </NativeTooltip>

          {/* Doc Lens */}
          <NativeTooltip
            content={
              filteredDocs.length === 0
                ? "Requires at least one document"
                : "Document image search (Doc Lens)"
            }
            side="bottom"
            animation="blur"
            shine
          >
            <span className="inline-flex">
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
                  {iconOnlyButtons ? "" : "Doc Lens"}
                </span>
              </Button>
            </span>
          </NativeTooltip>

          </div>{/* end buttons row */}
        </div>{/* end agent actions */}
      </div>

      {/* ── Collapsible filter panel (shown when toolbar is narrow) ── */}
      <AnimatePresence>
        {useNarrowToolbar && showFilters && (
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

      {/* ── Auto-tag confirmation bar ───────────────────────────── */}
      <AnimatePresence>
        {showTagConfirm && !isTagging && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-border/30"
          >
            <div className="px-3 py-3 space-y-3">
              <div className="flex items-start gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-muted-foreground">
                    Auto-tag {filteredDocs.length} visible document
                    {filteredDocs.length !== 1 ? "s" : ""}?
                  </p>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <NativeTooltip content="Run auto-tagging" side="bottom" animation="blur">
                    <Button
                      variant="default"
                      size="sm"
                      className="h-7 text-[11px] gap-1"
                      onClick={runAutoTag}
                      disabled={
                        filteredDocs.length === 0 ||
                        (tagMode === "custom" && !hasCustomSelection)
                      }
                    >
                      <Send className="h-3 w-3" />
                      Run
                    </Button>
                  </NativeTooltip>
                  <NativeTooltip content="Cancel" side="bottom" animation="blur">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => setShowTagConfirm(false)}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </NativeTooltip>
                </div>
              </div>

              <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/20 px-3 py-2">
                <p className="text-[11px] font-medium text-foreground">Tag Mode</p>
                <div className="flex items-center rounded-md border border-border/60 overflow-hidden shrink-0">
                  <NativeTooltip
                    content="Use the default tag set."
                    side="bottom"
                    animation="blur"
                  >
                    <span className="inline-flex">
                      <Button
                        type="button"
                        variant={tagMode === "default" ? "secondary" : "ghost"}
                        size="sm"
                        className="h-7 rounded-none px-3 text-[11px]"
                        onClick={() => {
                          setTagMode("default");
                          setCustomTagError(null);
                        }}
                      >
                        Default
                      </Button>
                    </span>
                  </NativeTooltip>
                  <NativeTooltip
                    content="Customize which tags the model can use."
                    side="bottom"
                    animation="blur"
                  >
                    <span className="inline-flex">
                      <Button
                        type="button"
                        variant={tagMode === "custom" ? "secondary" : "ghost"}
                        size="sm"
                        className="h-7 rounded-none border-l border-border/60 px-3 text-[11px]"
                        onClick={() => {
                          setTagMode("custom");
                          setCustomTagError(null);
                        }}
                      >
                        Custom
                      </Button>
                    </span>
                  </NativeTooltip>
                </div>
              </div>

              {tagMode === "custom" && (
                <div className="rounded-md border border-border/50 bg-muted/10">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/40 px-3 py-2">
                    <div>
                      <p className="text-[11px] font-medium text-foreground">
                        Custom Tag Set
                      </p>
                      <p className="text-[10px] text-muted-foreground">
                        {selectedCustomTags.length} selected of {customTagCatalog.length} configured
                        tags ({MAX_CUSTOM_TAGS} max).
                      </p>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-7 text-[11px]"
                        onClick={selectAllCustomTags}
                        disabled={customTagCatalog.length === 0}
                      >
                        <CheckSquare className="mr-1 h-3 w-3" />
                        Select all
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-7 text-[11px]"
                        onClick={unselectAllCustomTags}
                        disabled={customTagCatalog.length === 0}
                      >
                        <Square className="mr-1 h-3 w-3" />
                        Unselect all
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-7 text-[11px]"
                        onClick={restoreDefaultCustomTags}
                        disabled={
                          customTagCatalog.length === ALL_TAGS.length &&
                          customTagCatalog.every((tag, index) => tag === ALL_TAGS[index]) &&
                          selectedCustomTags.length === ALL_TAGS.length &&
                          selectedCustomTags.every((tag, index) => tag === ALL_TAGS[index])
                        }
                      >
                        Restore Default
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-7 text-[11px]"
                        onClick={removeAllCustomTags}
                        disabled={customTagCatalog.length === 0}
                      >
                        <Trash2 className="mr-1 h-3 w-3" />
                        Remove all
                      </Button>
                    </div>
                  </div>

                  <div className="flex flex-col gap-2 border-b border-border/40 px-3 py-3 sm:flex-row">
                    <Input
                      value={customTagDraft}
                      onChange={(e) => {
                        setCustomTagDraft(e.target.value);
                        if (customTagError) setCustomTagError(null);
                      }}
                      placeholder="Add a tag"
                      className="h-8 text-xs flex-1"
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          addCustomTag();
                        }
                      }}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-8 text-[11px] shrink-0"
                      onClick={addCustomTag}
                      disabled={customTagCatalog.length >= MAX_CUSTOM_TAGS}
                    >
                      <Plus className="mr-1 h-3 w-3" />
                      Add tag
                    </Button>
                  </div>

                  {(customSelectionNotice || customTagError) && (
                    <div className="space-y-1 px-3 pt-3">
                      {customSelectionNotice && (
                        <div className="flex items-center gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-2 text-[11px] text-amber-800 dark:text-amber-200">
                          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                          {customSelectionNotice}
                        </div>
                      )}
                      {customTagError && (
                        <div className="flex items-center gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 px-2.5 py-2 text-[11px] text-destructive">
                          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                          {customTagError}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="px-3 pt-3">
                    <div className="flex items-center gap-1.5 rounded-md border border-sky-500/30 bg-sky-500/10 px-2.5 py-2 text-[11px] text-sky-800 dark:text-sky-200">
                      <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                      Docs with no matching custom tags will get `{CUSTOM_FALLBACK_TAG_LABEL}`.
                    </div>
                  </div>

                  <div className="px-3 py-3">
                    <div className="rounded-md border border-border/40 overflow-hidden">
                      <div className="max-h-72 overflow-y-auto">
                        <Table>
                          <TableHeader className="sticky top-0 z-10 bg-background">
                            <TableRow>
                              <TableHead className="w-[60%]">Tag</TableHead>
                              <TableHead>Selected</TableHead>
                              <TableHead className="w-[64px] text-right">Remove</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {customTagCatalog.map((tagLabel) => {
                              const cfg = getTagConfig(tagLabel);
                              const Icon = cfg.icon;
                              const isSelected = selectedCustomTagSet.has(tagLabel);
                              return (
                                <TableRow key={tagLabel}>
                                  <TableCell>
                                    <div className="flex items-center gap-2">
                                      <Badge
                                        variant="outline"
                                        className={cn(
                                          "text-[10px] px-2 py-0.5 inline-flex items-center gap-1",
                                          cfg.bg,
                                          cfg.text,
                                          cfg.border
                                        )}
                                      >
                                        <Icon className="h-3 w-3 shrink-0" />
                                        {tagLabel}
                                      </Badge>
                                    </div>
                                  </TableCell>
                                  <TableCell>
                                    <Button
                                      type="button"
                                      variant={isSelected ? "secondary" : "outline"}
                                      size="sm"
                                      className="h-7 text-[11px]"
                                      onClick={() => toggleCustomTagSelection(tagLabel)}
                                    >
                                      {isSelected ? (
                                        <CheckSquare className="mr-1 h-3 w-3" />
                                      ) : (
                                        <Square className="mr-1 h-3 w-3" />
                                      )}
                                      {isSelected ? "Selected" : "Unselected"}
                                    </Button>
                                  </TableCell>
                                  <TableCell className="text-right">
                                    <Button
                                      type="button"
                                      variant="ghost"
                                      size="icon"
                                      className="h-7 w-7"
                                      onClick={() => removeCustomTag(tagLabel)}
                                    >
                                      <Trash2 className="h-3.5 w-3.5" />
                                    </Button>
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Summarize instruction input ─────────────────────────── */}
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
              <Input
                value={summarizeInstructions}
                onChange={(e) => setSummarizeInstructions(e.target.value)}
                placeholder={`Optional: tell the summarizer what to focus on for ${filteredDocs.length} visible document${filteredDocs.length !== 1 ? "s" : ""}`}
                className="h-7 text-xs flex-1"
                onKeyDown={(e) => {
                  if (e.key === "Enter") runSummarize();
                }}
                autoFocus
              />
              <NativeTooltip content="Run summarization" side="bottom" animation="blur">
                <Button
                  variant="default"
                  size="icon"
                  className="h-7 w-7 shrink-0"
                  onClick={runSummarize}
                  disabled={filteredDocs.length === 0}
                >
                  <Send className="h-3 w-3" />
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
          {filters.tags.size > 0 && (
            <button
              type="button"
              role="switch"
              aria-checked={tagFilterMode === "and"}
              aria-label={`Tag filter mode: ${tagFilterMode.toUpperCase()}`}
              onClick={() =>
                setTagFilterMode((prev) => (prev === "or" ? "and" : "or"))
              }
              className={cn(
                "ml-auto inline-flex items-center gap-1 rounded-full border border-border/70 bg-muted/70 p-1 shadow-sm transition-all hover:bg-muted hover:shadow-md",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              )}
            >
              <span
                className={cn(
                  "rounded-full px-3 py-1 text-[10px] font-semibold tracking-wide transition-all",
                  tagFilterMode === "or"
                    ? "bg-foreground text-background shadow-sm"
                    : "text-muted-foreground bg-transparent opacity-70"
                )}
              >
                OR
              </span>
              <span
                className={cn(
                  "rounded-full px-3 py-1 text-[10px] font-semibold tracking-wide transition-all",
                  tagFilterMode === "and"
                    ? "bg-secondary-foreground text-secondary shadow-sm"
                    : "text-muted-foreground bg-transparent opacity-70"
                )}
              >
                AND
              </span>
            </button>
          )}
        </div>
      )}

      {/* ── Search/Triage document list ─────────────────────────── */}
      <ScrollArea className="flex-1 relative z-0">
        <div
          className={cn(
            "px-2 py-2 grid gap-1.5",
            gridCols === 4
              ? "grid-cols-4"
              : gridCols === 2
                ? "grid-cols-2"
                : "grid-cols-1",
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
            <div className="col-span-full flex flex-col items-center justify-center py-12 text-muted-foreground/60 gap-2">
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
                                                const cfg = getTagConfig(tag.label, tag.icon);
                                                const Icon = cfg.icon;
                                                return (
                                                  <Badge
                                                    key={`${doc._id}-${tag.label}-${tag.icon ?? "general"}`}
                                                    variant="outline"
                                                    className={`text-[10px] px-1.5 py-0 inline-flex items-center gap-0.5 ${cfg.bg} ${cfg.text} ${cfg.border}`}
                                                  >
                                                    <Icon className="h-2.5 w-2.5 shrink-0" />
                                                    {tag.label}
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
