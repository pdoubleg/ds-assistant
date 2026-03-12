"use client";

/**
 * QueryHitCard — renders a single Doc Lens query result.
 *
 * Shows the extracted image prominently with metadata (rank, score,
 * document name, page number, asset type, extraction method).
 * Supports click-to-expand lightbox, flag/dock toggle, and a link
 * to open the document preview pane at the target page.
 */

import React, { useState } from "react";
import {
  Bookmark,
  BookmarkCheck,
  Eye,
  Download,
  ImageIcon,
  Hash,
  X,
  FileText,
  ScanSearch,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { QueryHit } from "@/hooks/use-doc-lens";

// ── Props ──────────────────────────────────────────────────────────────

interface QueryHitCardProps {
  hit: QueryHit;
  query: string;
  imageUrl: string;
  isFlagged: boolean;
  onToggleFlag: () => void;
  onPreviewDoc: (fileName: string, page: number, query?: string) => void;
  onDownload?: () => void;
  /** When true, renders a more compact layout for the flagged panel. */
  compact?: boolean;
}

// ── Helpers ────────────────────────────────────────────────────────────

function formatScore(score: number): string {
  return (score * 100).toFixed(1);
}

function extractionLabel(method: string): string {
  const labels: Record<string, string> = {
    full_page_render: "Page Render",
    pdf_embedded_image: "Embedded Image",
    page_segmentation: "Segmented",
    standalone_image: "Standalone",
    text_page: "Text Match",
  };
  return labels[method] || method;
}

function assetTypeBadgeClass(assetType: string): string {
  const normalized = assetType.toLowerCase();
  if (normalized === "photo") {
    return "border-emerald-500/40 bg-emerald-500/15 text-emerald-700 dark:text-emerald-300";
  }
  if (normalized === "page") {
    return "border-sky-500/40 bg-sky-500/15 text-sky-700 dark:text-sky-300";
  }
  return "border-border/70 bg-muted/50 text-muted-foreground";
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getHighlightTerms(query: string): string[] {
  const normalizedQuery = query.trim();
  const wordTokens = normalizedQuery
    .split(/\W+/)
    .map((token) => token.trim())
    .filter(Boolean);

  return Array.from(new Set([normalizedQuery, ...wordTokens].filter(Boolean))).sort(
    (left, right) => right.length - left.length
  );
}

function renderHighlightedText(text: string, query: string): React.ReactNode {
  const tokens = getHighlightTerms(query);

  if (tokens.length === 0) {
    return text;
  }

  const pattern = new RegExp(`(${tokens.map(escapeRegExp).join("|")})`, "gi");
  const parts = text.split(pattern);

  return parts.map((part, index) => {
    const isMatch = tokens.some((token) => token.toLowerCase() === part.toLowerCase());
    if (!isMatch) {
      return <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>;
    }

    return (
      <mark
        key={`${part}-${index}`}
        className="rounded bg-amber-400/40 px-0.5 text-foreground"
      >
        {part}
      </mark>
    );
  });
}

// ── Component ──────────────────────────────────────────────────────────

export function QueryHitCard({
  hit,
  query,
  imageUrl,
  isFlagged,
  onToggleFlag,
  onPreviewDoc,
  onDownload,
  compact = false,
}: QueryHitCardProps) {
  const [expanded, setExpanded] = useState(false);
  const isTextHit = hit.extraction_method === "text_page";

  return (
    <>
      {/* Lightbox overlay — z-[200] to sit above the DocLens overlay (z-100) */}
      {expanded && (
        <div
          className="fixed inset-0 z-200 flex items-center justify-center bg-black/70 backdrop-blur-sm cursor-zoom-out"
          onClick={() => setExpanded(false)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]">
            <img
              src={imageUrl}
              alt={`${hit.document_name} p${hit.page_number}`}
              className="max-w-full max-h-[85vh] rounded-lg shadow-2xl object-contain"
            />
            <Button
              variant="secondary"
              size="icon"
              className="absolute top-2 right-2 h-8 w-8 rounded-full"
              onClick={(e) => {
                e.stopPropagation();
                setExpanded(false);
              }}
            >
              <X className="h-4 w-4" />
            </Button>
            <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-xs px-3 py-2 rounded-b-lg">
              <span className="font-medium">{hit.document_name}</span>
              {" — Page "}
              {hit.page_number} — Rank #{hit.rank} — Score{" "}
              {formatScore(hit.score)}%
            </div>
          </div>
        </div>
      )}

      {/* Card */}
      <div
        className={cn(
          "group relative rounded-xl border border-border/60 bg-card/95 text-card-foreground overflow-hidden transition-all duration-200 hover:shadow-xl hover:-translate-y-0.5 hover:border-primary/30",
          compact ? "flex flex-row gap-2.5 p-2.5" : "flex flex-col"
        )}
      >
        {/* Image area */}
        <div
          className={cn(
            "relative cursor-zoom-in bg-muted overflow-hidden",
            compact ? "w-24 h-28 shrink-0 rounded" : "w-full"
          )}
          onClick={() => setExpanded(true)}
        >
          <img
            src={imageUrl}
            alt={`${hit.document_name} p${hit.page_number}`}
            className={cn(
              "block transition-[filter] duration-300 group-hover:brightness-[1.03]",
              compact
                ? "w-full h-full object-cover"
                : "w-full h-auto object-contain bg-muted/40"
            )}
            loading="lazy"
          />
          {!compact && (
            <>
              <div className="pointer-events-none absolute inset-0 bg-linear-to-t from-black/65 via-black/15 to-transparent" />
              <div className="absolute top-2 left-2 right-2 flex items-start justify-between gap-2">
                <Badge className="bg-black/55 text-white border-white/15 text-[10px] font-semibold px-1.5 py-0 h-5">
                  #{hit.rank}
                </Badge>
                <div className="flex items-center gap-1 mr-auto">
                  <Badge className="bg-black/55 text-white border-white/15 text-[10px] font-semibold px-1.5 py-0 h-5 tabular-nums">
                    {formatScore(hit.score)}%
                  </Badge>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Metadata + actions */}
        <div
          className={cn(
            "flex flex-col",
            compact ? "flex-1 min-w-0 gap-1.5 py-0.5" : "gap-2 px-3 py-2.5"
          )}
        >
          {/* ── Document name ─────────────────────────────────────── */}
          <p
            className={cn(
              "font-semibold leading-tight text-foreground min-w-0",
              compact ? "text-[11px] line-clamp-2 wrap-break-word" : "text-[14px] wrap-break-word"
            )}
            title={hit.document_name}
          >
            {hit.document_name}
          </p>

          {/* ── Full card: query line ─────────────────────────────── */}
          {!compact && (
            <p
              className="text-[11px] text-muted-foreground whitespace-normal wrap-break-word leading-snug"
              title={query}
            >
              Query: &ldquo;{query}&rdquo;
            </p>
          )}

          {isTextHit && hit.text_snippet && (
            <p
              className={cn(
                "text-muted-foreground leading-snug whitespace-normal",
                compact ? "text-[10px] line-clamp-3" : "text-[11px] line-clamp-4"
              )}
              title={hit.text_snippet}
            >
              {renderHighlightedText(hit.text_snippet, query)}
            </p>
          )}

          {/* ── Compact card: query snippet ───────────────────────── */}
          {compact && query && (
            <p
              className="text-[10px] text-muted-foreground leading-snug line-clamp-1 flex items-center gap-1"
              title={query}
            >
              <ScanSearch className="h-2.5 w-2.5 shrink-0 opacity-60" />
              <span className="truncate italic">&ldquo;{query}&rdquo;</span>
            </p>
          )}

          {/* ── Location + score pill row ─────────────────────────── */}
          <div className="flex flex-wrap items-center gap-1">
            {/* Page number */}
            <Badge
              variant="outline"
              className="text-[10px] font-semibold px-1.5 py-0 h-[18px] border-violet-500/40 bg-violet-500/15 text-violet-700 dark:text-violet-300 gap-0.5"
            >
              <FileText className="h-2.5 w-2.5" />
              p.{hit.page_number}
            </Badge>

            {/* Asset type */}
            <Badge
              variant="outline"
              className={cn(
                "text-[10px] px-1.5 py-0 h-[18px] gap-0.5",
                assetTypeBadgeClass(hit.asset_type)
              )}
            >
              <ImageIcon className="h-2.5 w-2.5" />
              {hit.asset_type}
            </Badge>

            {/* Extraction method — full card only to keep compact tight */}
            {!compact && (
              <Badge
                variant="outline"
                className="text-[10px] px-1.5 py-0 h-[18px] gap-0.5 text-muted-foreground"
              >
                <Hash className="h-2.5 w-2.5" />
                {extractionLabel(hit.extraction_method)}
              </Badge>
            )}
          </div>

          {/* ── Score + rank row (compact) / score in image overlay (full) ── */}
          {compact && (
            <div className="flex items-center gap-1">
              <Badge
                variant="secondary"
                className="text-[10px] font-bold px-1.5 py-0 h-[18px] tabular-nums"
              >
                #{hit.rank}
              </Badge>
              <Badge
                variant="outline"
                className="text-[10px] font-semibold px-1.5 py-0 h-[18px] tabular-nums text-muted-foreground"
              >
                {formatScore(hit.score)}%
              </Badge>
            </div>
          )}

          {/* ── Action buttons ────────────────────────────────────── */}
          <div
            className={cn(
              "flex items-center gap-1 mt-auto",
              compact ? "pt-1 border-t border-border/40" : "pt-1.5 border-t border-border/50"
            )}
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={isFlagged ? "default" : "ghost"}
                  size="icon"
                  className="h-7 w-7"
                  onClick={onToggleFlag}
                >
                  {isFlagged ? (
                    <BookmarkCheck className="h-3.5 w-3.5" />
                  ) : (
                    <Bookmark className="h-3.5 w-3.5" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="text-xs">
                {isFlagged ? "Remove from saved" : "Save to flagged list"}
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={() =>
                    onPreviewDoc(
                      hit.document_name,
                      hit.page_number,
                      isTextHit ? query : undefined
                    )
                  }
                >
                  <Eye className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="text-xs">
                Preview document at page {hit.page_number}
              </TooltipContent>
            </Tooltip>

            {onDownload && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={onDownload}
                  >
                    <Download className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="text-xs">
                  Download image
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
