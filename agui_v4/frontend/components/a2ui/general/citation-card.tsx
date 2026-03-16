"use client";

/**
 * CitationCard Component
 *
 * Compact, neutral citation card linking the user to a specific document
 * page.  Designed to sit four-wide in the output grid.  Structured as:
 *   - Thin header: compass icon + title (no color on the title)
 *   - Body: description text
 *   - Doc name row
 *   - Wide clickable page pill (doubles as preview link)
 */

import React, { useCallback, useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ChevronDown, Compass, FileText } from "lucide-react";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CitationCardProps {
  /** Stable backend document identifier. */
  content_id: string;
  /** 1-based page number within the document. */
  page_number: number;
  /** Short citation headline. */
  title: string;
  /** Brief description of what is cited at this location. */
  description: string;
  /** Human-readable document file name. */
  file_name: string;
  /** Optional URL for preview/download access. */
  content_url?: string;
  /** Callback fired when the user clicks the page pill. */
  onPreviewDoc?: (contentId: string, page: number) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CitationCard({
  content_id,
  page_number,
  title,
  description,
  file_name,
  onPreviewDoc,
}: CitationCardProps): React.ReactElement {
  const handlePreview = useCallback(() => {
    onPreviewDoc?.(content_id, page_number);
  }, [onPreviewDoc, content_id, page_number]);

  const isInteractive = !!onPreviewDoc;
  const [expanded, setExpanded] = useState(false);

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex flex-col rounded-xl border border-border/70 bg-card shadow-xs transition-all duration-200 hover:shadow-md hover:border-border">
        {/* ── Header ──────────────────────────────────────────── */}
        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          className="flex items-center gap-2 px-3.5 py-2 border-b border-border/40 text-left w-full"
        >
          <Compass
            className="h-3.5 w-3.5 shrink-0 text-muted-foreground/70"
            strokeWidth={1.75}
          />
          <h3
            className="text-[13px] font-medium leading-snug text-foreground truncate flex-1 min-w-0"
            title={title}
          >
            {title}
          </h3>
          <ChevronDown
            className={[
              "h-3 w-3 shrink-0 text-muted-foreground/50 transition-transform duration-200",
              expanded ? "rotate-180" : "",
            ].join(" ")}
          />
        </button>

        {/* ── Description (collapsible, default closed) ───────── */}
        <div
          className={[
            "grid transition-[grid-template-rows] duration-200 ease-out",
            expanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
          ].join(" ")}
        >
          <div className="overflow-hidden">
            <div className="px-3.5 py-2">
              <p className="text-xs leading-relaxed text-muted-foreground">
                {description}
              </p>
            </div>
          </div>
        </div>

        {/* ── Doc name ────────────────────────────────────────── */}
        <div className="px-3.5 pb-1.5">
          <span
            className="text-[10px] text-muted-foreground/70 truncate block"
            title={file_name}
          >
            {file_name}
          </span>
        </div>

        {/* ── Page pill ───────────────────────────────────────── */}
        <div className="px-3.5 pb-3">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={isInteractive ? handlePreview : undefined}
                disabled={!isInteractive}
                className={[
                  "flex w-full items-center justify-center gap-1.5 rounded-full border py-1.5",
                  "text-[11px] font-semibold tabular-nums leading-none",
                  "transition-all duration-150",
                  isInteractive
                    ? "border-border/70 bg-secondary/50 text-foreground hover:bg-secondary hover:border-border hover:shadow-sm cursor-pointer"
                    : "border-border/40 bg-muted/30 text-muted-foreground cursor-default",
                ].join(" ")}
              >
                <FileText className="h-3 w-3" />
                Page {page_number}
              </button>
            </TooltipTrigger>
            {isInteractive && (
              <TooltipContent side="bottom" className="text-xs">
                Open document at page {page_number}
              </TooltipContent>
            )}
          </Tooltip>
        </div>
      </div>
    </TooltipProvider>
  );
}

export default CitationCard;
