"use client";

/**
 * DocumentCard Component
 *
 * Three-tier responsive card for claim/policy documents. The `variant` prop
 * controls how much detail is shown by default:
 *   - narrow  (~25% screen): compact single row, expander reveals everything
 *   - medium  (~40-50%):     two-line card with type/date inline
 *   - wide    (~90%+):       description visible inline, expander for AI summary
 *
 * Action buttons: eye (preview), context toggle, and collapsible AI summary.
 */

import React, { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { getTagConfig, type TagIconName } from "@/lib/tag-registry";
import { Badge } from "@/components/ui/badge";
import { GfmMarkdown } from "@/components/a2ui/general/gfm-markdown";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import {
  FileText,
  FileSpreadsheet,
  FileImage,
  File,
  Calendar,
  ChevronDown,
  ChevronRight,
  Sparkles,
  Search,
  Eye,
  MessageSquarePlus,
  MessageSquareX,
  Shield,
  ShieldOff,
  Tag,
} from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────

/** Shape of the AI-generated summary attached to a document. */
export interface DocumentSummaryData {
  title: string;
  summary: string;
  label: string | null;
}

/** Shape of the search/sort agent score attached to a document. */
export interface DocSearchData {
  score: number;
  label: string | null;
}

export type CardVariant = "narrow" | "medium" | "wide";
export interface DocumentCardUiState {
  expanded: boolean;
  tagsOpen: boolean;
  summaryOpen: boolean;
}
export interface BulkExpandedCommand {
  version: number;
  expanded: boolean;
  tagsOpen: boolean;
  summaryOpen: boolean;
}

export interface DocumentTagData {
  label: string;
  icon?: TagIconName | null;
}

export interface DocumentCardProps {
  file_name: string;
  mime_type: string;
  content_id?: string;
  claim_number?: string;
  content_url?: string;
  domain?: "claim" | "policy";
  document_type?: string;
  document_sub_type?: string;
  document_description?: string;
  create_date?: string;
  source_system?: string;
  company_name?: string;
  /** AI-generated summary. When present the card can show a summary section. */
  summaryData?: DocumentSummaryData;
  /** Search/sort agent score. When present the card can show a score section. */
  searchData?: DocSearchData;
  /** Agent-generated tags shown as small pills. */
  tags?: DocumentTagData[];

  /** Responsive layout tier set by the parent pane. */
  variant?: CardVariant;

  /** Whether this doc is hidden from the doc-agent context. */
  isHidden?: boolean;
  /** Toggle hidden state (doc-agent context). */
  onToggleHidden?: () => void;

  /** Whether this doc is in the chat-agent's context. */
  isInChatContext?: boolean;
  /** Toggle chat-agent context membership. */
  onToggleChatContext?: () => void;

  /** Clicking a tag pill toggles a tag filter in the parent pane. */
  onTagClick?: (tag: string) => void;
  /** Set of currently active tag filters (highlighted pills). */
  activeTagFilters?: Set<string>;

  /** Open the document viewer sheet. */
  onPreview?: () => void;
  /**
   * Restored UI state for this card (used across hide/unhide and remounts).
   * If omitted, the card uses its default collapsed state.
   */
  initialUiState?: Partial<DocumentCardUiState>;
  /**
   * Optional bulk command from parent to expand/collapse many cards at once.
   * A command is applied only when `version` changes.
   */
  bulkExpandedCommand?: BulkExpandedCommand;
  /** Callback fired whenever the card's local UI state changes. */
  onUiStateChange?: (nextState: DocumentCardUiState) => void;

  // Kept for backward compat during transition; prefer isHidden/onToggleHidden.
  /** @deprecated Use isHidden / onToggleHidden instead. */
  isInContext?: boolean;
  /** @deprecated Use onToggleHidden instead. */
  onAddToContext?: () => void;
}

// ── Helpers ────────────────────────────────────────────────────────────

export function deriveFileExt(mime_type: string, file_name: string): string {
  const mimeMap: Record<string, string> = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
      "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
      "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/tiff": "tiff",
  };
  if (mimeMap[mime_type]) return mimeMap[mime_type];
  const dotIdx = file_name.lastIndexOf(".");
  if (dotIdx >= 0) return file_name.slice(dotIdx + 1).toLowerCase();
  return "file";
}

const FILE_ICONS: Record<string, React.ReactNode> = {
  pdf: <FileText className="h-4 w-4 text-red-500 dark:text-red-400" />,
  docx: <FileText className="h-4 w-4 text-blue-500 dark:text-blue-400" />,
  xlsx: (
    <FileSpreadsheet className="h-4 w-4 text-emerald-500 dark:text-emerald-400" />
  ),
  jpg: <FileImage className="h-4 w-4 text-amber-500 dark:text-amber-400" />,
  png: <FileImage className="h-4 w-4 text-amber-500 dark:text-amber-400" />,
  tiff: <FileImage className="h-4 w-4 text-amber-500 dark:text-amber-400" />,
};

const FILE_TYPE_BADGE: Record<string, string> = {
  pdf: "bg-red-500/20 text-red-700 dark:text-red-400 border-red-500/30 font-bold",
  docx: "bg-blue-500/20 text-blue-700 dark:text-blue-400 border-blue-500/30 font-bold",
  xlsx: "bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 font-bold",
  jpg: "bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/30 font-bold",
  png: "bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/30 font-bold",
  tiff: "bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/30 font-bold",
};

const DOMAIN_BADGE: Record<string, string> = {
  claim: "bg-blue-500/20 text-blue-700 dark:text-blue-400 border-blue-500/30",
  policy:
    "bg-violet-500/20 text-violet-700 dark:text-violet-400 border-violet-500/30",
};

function scoreColorClass(score: number): string {
  if (score >= 0.8)
    return "bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 border-emerald-500/40";
  if (score >= 0.5)
    return "bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/40";
  return "bg-red-500/20 text-red-700 dark:text-red-400 border-red-500/40";
}

// ── Component ──────────────────────────────────────────────────────────

export function DocumentCard({
  file_name,
  mime_type,
  domain = "claim",
  document_type,
  document_sub_type,
  document_description,
  create_date,
  source_system,
  company_name,
  summaryData,
  searchData,
  tags,
  variant = "narrow",
  isHidden = false,
  onToggleHidden,
  isInChatContext = false,
  onToggleChatContext,
  onTagClick,
  activeTagFilters,
  onPreview,
  initialUiState,
  bulkExpandedCommand,
  onUiStateChange,
  isInContext,
  onAddToContext,
}: DocumentCardProps): React.ReactElement {
  const [expanded, setExpanded] = useState(initialUiState?.expanded ?? false);
  const [tagsOpen, setTagsOpen] = useState(initialUiState?.tagsOpen ?? false);
  const [summaryOpen, setSummaryOpen] = useState(initialUiState?.summaryOpen ?? false);
  const previousSummary = useRef<DocumentSummaryData | undefined>(summaryData);
  const previousTagsCount = useRef<number>(tags?.length ?? 0);
  const lastBulkVersion = useRef<number | null>(null);
  const didApplyHydratedUiState = useRef<boolean>(!!initialUiState);
  const ext = deriveFileExt(mime_type, file_name);
  const icon = FILE_ICONS[ext] || (
    <File className="h-4 w-4 text-muted-foreground" />
  );
  const typeBadgeClass =
    FILE_TYPE_BADGE[ext] || "bg-muted text-muted-foreground border-border";
  const domainBadgeClass = DOMAIN_BADGE[domain] || DOMAIN_BADGE.claim;

  const formattedDate = create_date
    ? new Date(create_date).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : null;

  const hasSummary = !!summaryData;

  // Auto-open summary only when it newly appears in this mounted card.
  useEffect(() => {
    if (!previousSummary.current && summaryData) {
      setSummaryOpen(true);
    }
    previousSummary.current = summaryData;
  }, [summaryData]);

  // Auto-open tags row only when tags transition from empty -> non-empty.
  useEffect(() => {
    const nextCount = tags?.length ?? 0;
    if (previousTagsCount.current === 0 && nextCount > 0) {
      setTagsOpen(true);
    }
    previousTagsCount.current = nextCount;
  }, [tags]);

  useEffect(() => {
    onUiStateChange?.({
      expanded,
      tagsOpen,
      summaryOpen,
    });
  }, [expanded, onUiStateChange, summaryOpen, tagsOpen]);

  // Apply restored UI state once if it arrives after initial mount.
  // This happens when parent state hydrates from localStorage asynchronously.
  useEffect(() => {
    if (didApplyHydratedUiState.current) return;
    if (!initialUiState) return;
    setExpanded(initialUiState.expanded ?? false);
    setTagsOpen(initialUiState.tagsOpen ?? false);
    setSummaryOpen(initialUiState.summaryOpen ?? false);
    didApplyHydratedUiState.current = true;
  }, [initialUiState]);

  // Apply parent-driven expand/collapse commands once per command version.
  useEffect(() => {
    if (!bulkExpandedCommand) return;
    if (lastBulkVersion.current === bulkExpandedCommand.version) return;
    lastBulkVersion.current = bulkExpandedCommand.version;
    setExpanded(bulkExpandedCommand.expanded);
    setTagsOpen(bulkExpandedCommand.tagsOpen);
    setSummaryOpen(bulkExpandedCommand.summaryOpen);
  }, [bulkExpandedCommand]);

  const showDescInline = variant === "wide";

  // Metadata that goes into the expandable section in narrow/medium variants
  const hasExpandableMetadata =
    (!showDescInline && document_description) ||
    source_system ||
    company_name;

  return (
    <Collapsible open={expanded} onOpenChange={setExpanded}>
      <div
        className={cn(
          "relative rounded-lg border overflow-hidden transition-all duration-200",
          "hover:shadow-md group",
          isHidden || isInContext
            ? "opacity-60 bg-muted/30 border-border/30 before:absolute before:left-0 before:top-0 before:h-full before:w-1 before:bg-muted-foreground/30 before:content-['']"
            : isInChatContext
              ? "border-2 agent-doc-border bg-card before:absolute before:left-0 before:top-0 before:h-full before:w-1 before:bg-amber-500 before:content-['']"
              : "bg-card border-border/50 hover:border-border before:absolute before:left-0 before:top-0 before:h-full before:w-1 before:bg-transparent hover:before:bg-primary/40 before:content-['']"
        )}
      >
        {/* ── Search/sort score ribbon ── */}
        {searchData != null && (
          <div className="flex items-center gap-1.5 px-3 py-1 pl-6 bg-blue-500/5 border-b border-blue-500/15">
            <Search className="h-3 w-3 text-blue-500 dark:text-blue-400 shrink-0" />
            <Badge
              variant="outline"
              className={cn(
                "text-[9px] px-1 py-0 font-bold tabular-nums shrink-0",
                scoreColorClass(searchData.score)
              )}
            >
              {Math.round(searchData.score * 100)}%
            </Badge>
            <span className="text-[9px] text-muted-foreground/70 truncate">
              {searchData.label || "Relevance Score"}
            </span>
          </div>
        )}

        {/* ── Primary row: icon + name + badges + actions ── */}
        <div className="flex items-center gap-2 px-3 py-2 min-h-[40px]">
          <span className="shrink-0">{icon}</span>

          <div className="flex-1 min-w-0">
            {/* Filename + inline badges */}
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="text-sm font-medium truncate leading-tight">
                {file_name}
              </span>
              <Badge
                variant="outline"
                className={cn(
                  "text-[9px] px-1 py-0 font-mono shrink-0",
                  typeBadgeClass
                )}
              >
                {ext.toUpperCase()}
              </Badge>
              <Badge
                variant="outline"
                className={cn(
                  "text-[9px] px-1 py-0 shrink-0",
                  domainBadgeClass
                )}
              >
                {domain}
              </Badge>
            </div>

            {/* Second line: type, subtype, date (all variants) */}
            {(document_type || formattedDate) && (
              <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                {document_type && (
                  <span className="text-[11px] text-muted-foreground">
                    {document_type}
                  </span>
                )}
                {document_sub_type && (
                  <span className="text-[11px] text-muted-foreground/70">
                    / {document_sub_type}
                  </span>
                )}
                {formattedDate && (
                  <span className="flex items-center gap-0.5 text-xs font-medium text-muted-foreground">
                    <Calendar className="h-3 w-3" />
                    {formattedDate}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-0.5 shrink-0">
            {onPreview && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onPreview();
                    }}
                    className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
                  >
                    <Eye className="h-4 w-4" />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="text-xs">
                  Preview document
                </TooltipContent>
              </Tooltip>
            )}

            {/* Chat context toggle */}
            {onToggleChatContext && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleChatContext();
                    }}
                    className={cn(
                      "p-1.5 rounded-md transition-colors",
                      isInChatContext
                        ? "text-amber-600 dark:text-amber-400 bg-amber-500/10 hover:bg-amber-500/20"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                    )}
                  >
                    {isInChatContext ? (
                      <MessageSquareX className="h-4 w-4" />
                    ) : (
                      <MessageSquarePlus className="h-4 w-4" />
                    )}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="text-xs">
                  {isInChatContext
                    ? "Remove from chat context"
                    : "Add to chat context"}
                </TooltipContent>
              </Tooltip>
            )}

            {/* Hide / unhide toggle (doc-agent context) */}
            {(onToggleHidden || onAddToContext) && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      (onToggleHidden || onAddToContext)?.();
                    }}
                    className={cn(
                      "p-1.5 rounded-md transition-colors",
                      isHidden || isInContext
                        ? "text-rose-600 dark:text-rose-400 bg-rose-500/10 hover:bg-rose-500/20"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                    )}
                  >
                    {isHidden || isInContext ? (
                      <ShieldOff className="h-4 w-4" />
                    ) : (
                      <Shield className="h-4 w-4" />
                    )}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="text-xs">
                  {isHidden || isInContext
                    ? "Include in doc agent"
                    : "Hide from doc agent"}
                </TooltipContent>
              </Tooltip>
            )}

            {/* Chevron toggle for expandable content */}
            <CollapsibleTrigger asChild>
              <button
                className="p-1 rounded-md text-muted-foreground/50 hover:text-foreground hover:bg-secondary/60 transition-colors"
                onClick={(e) => e.stopPropagation()}
              >
                <ChevronDown
                  className={cn(
                    "h-3.5 w-3.5 transition-transform duration-200",
                    expanded && "rotate-180"
                  )}
                />
              </button>
            </CollapsibleTrigger>
          </div>
        </div>

        {/* ── Wide variant: inline description + tags ── */}
        {showDescInline && document_description && (
          <div className="px-3 pb-2 -mt-1">
            <p className="text-xs text-muted-foreground line-clamp-2 pl-6">
              {document_description}
            </p>
          </div>
        )}

        {/* ── Collapsible tags row (all variants) ── */}
        {tags && tags.length > 0 && (
          <div className="border-t border-border/20">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setTagsOpen((p) => !p);
              }}
              className="flex items-center gap-1 px-3 py-1 pl-6 w-full text-left hover:bg-secondary/30 transition-colors"
            >
              <Tag className="h-3 w-3 text-indigo-500 dark:text-indigo-400" />
              <span className="text-[10px] font-medium text-muted-foreground">
                Tags
              </span>
              <Badge
                variant="outline"
                className="text-[9px] px-1 py-0 ml-0.5 bg-indigo-500/10 text-indigo-600 dark:text-indigo-300 border-indigo-500/20"
              >
                {tags.length}
              </Badge>
              {tagsOpen ? (
                <ChevronDown className="h-2.5 w-2.5 text-muted-foreground/50 ml-auto" />
              ) : (
                <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/50 ml-auto" />
              )}
            </button>
            {tagsOpen && (
              <div className="px-3 pb-1.5 pt-0.5 flex items-center gap-1.5 flex-wrap pl-6">
                {tags.map((tag) => {
                  const isActive = activeTagFilters?.has(tag.label);
                  const cfg = getTagConfig(tag.label, tag.icon);
                  const Icon = cfg.icon;
                  return (
                    <Badge
                      key={`${tag.label}-${tag.icon ?? "general"}`}
                      variant="outline"
                      className={cn(
                        "text-[11px] px-2 py-0.5 font-medium transition-all inline-flex items-center gap-1",
                        isActive
                          ? `${cfg.activeBg} ${cfg.activeText} ${cfg.activeBorder}`
                          : `${cfg.bg} ${cfg.text} ${cfg.border}`,
                        onTagClick &&
                          "cursor-pointer hover:brightness-110 active:scale-95"
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        onTagClick?.(tag.label);
                      }}
                    >
                      <Icon className="h-3 w-3 shrink-0" />
                      {tag.label}
                    </Badge>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ── Collapsible AI summary section (independent of main expander) ── */}
        {hasSummary && (
          <div className="border-t border-border/20">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setSummaryOpen((p) => !p);
              }}
              className="flex items-center gap-1 px-3 py-1 pl-6 w-full text-left hover:bg-secondary/30 transition-colors"
            >
              <Sparkles className="h-3 w-3 text-amber-500 dark:text-amber-400" />
              <span className="text-[10px] font-medium text-muted-foreground">
                AI Summary
              </span>
              <span className="text-[9px] text-muted-foreground/60 ml-1 truncate max-w-[200px]">
                {summaryData.label || "Document Summary"}
              </span>
              {summaryOpen ? (
                <ChevronDown className="h-2.5 w-2.5 text-muted-foreground/50 ml-auto" />
              ) : (
                <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/50 ml-auto" />
              )}
            </button>
            {summaryOpen && (
              <div className="px-3 pb-2.5 pt-1 space-y-1 bg-muted/20 pl-6">
                <p className="text-xs font-semibold text-foreground leading-tight">
                  {summaryData.title}
                </p>
                <GfmMarkdown
                  content={summaryData.summary}
                  compact
                  className="doc-summary-markdown text-xs text-muted-foreground leading-relaxed"
                />
              </div>
            )}
          </div>
        )}

        {/* ── Expandable section (metadata details for narrow/medium) ── */}
        <CollapsibleContent>
          {hasExpandableMetadata && !showDescInline && (
            <div className="px-3 pb-2 pt-1 border-t border-border/30 space-y-1.5">
              {document_description && (
                <p className="text-xs text-muted-foreground line-clamp-3 pl-6">
                  {document_description}
                </p>
              )}
              <div className="flex items-center gap-1.5 flex-wrap pl-6">
                {source_system && (
                  <Badge
                    variant="secondary"
                    className="text-[9px] px-1 py-0"
                  >
                    {source_system}
                  </Badge>
                )}
                {company_name && (
                  <Badge
                    variant="secondary"
                    className="text-[9px] px-1 py-0"
                  >
                    {company_name}
                  </Badge>
                )}
              </div>
            </div>
          )}
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

export default DocumentCard;
