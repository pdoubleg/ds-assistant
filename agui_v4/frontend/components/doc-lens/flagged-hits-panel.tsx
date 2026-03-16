"use client";

/**
 * FlaggedHitsPanel — displays saved / flagged Doc Lens query hits.
 *
 * Used both inside the DocLensOverlay (as a collapsible sidebar section)
 * and as a standalone panel accessible from the Output pane.
 */

import React from "react";
import {
  BookmarkCheck,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { QueryHitCard } from "./query-hit-card";
import type { FlaggedHit } from "@/hooks/use-flagged-hits";

// ── Inline panel (used inside DocLensOverlay and OutputPane) ────────────

interface FlaggedHitsInlineProps {
  flaggedHits: FlaggedHit[];
  getImageUrl: (imagePath: string) => string;
  onRemove: (assetHash: string) => void;
  onClearAll: () => void;
  onDownloadImage: (imagePath: string, fileName?: string) => void;
  onPreviewDoc: (fileName: string, page: number, query?: string) => void;
  isFlagged: (assetHash: string) => boolean;
  onToggleFlag: (hit: FlaggedHit["hit"], query: string) => void;
}

export function FlaggedHitsInline({
  flaggedHits,
  getImageUrl,
  onRemove,
  onClearAll,
  onDownloadImage,
  onPreviewDoc,
  isFlagged,
  onToggleFlag,
}: FlaggedHitsInlineProps) {
  if (flaggedHits.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-muted-foreground text-xs gap-2">
        <BookmarkCheck className="h-6 w-6 opacity-40" />
        <p>No saved images yet.</p>
        <p className="text-[10px]">
          Click the bookmark icon on result cards to save them here.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Header actions */}
      <div className="flex items-center justify-between px-1">
        <span className="text-xs font-medium text-muted-foreground">
          {flaggedHits.length} saved image{flaggedHits.length !== 1 ? "s" : ""}
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-[10px] gap-1 text-destructive hover:text-destructive"
              onClick={onClearAll}
            >
              <Trash2 className="h-3 w-3" />
              Clear All
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs">
            Remove all saved images
          </TooltipContent>
        </Tooltip>
      </div>

      {/* Cards */}
      <div className="flex flex-col gap-1.5">
        {flaggedHits.map((fh) => (
          <QueryHitCard
            key={fh.hit.asset_hash}
            hit={fh.hit}
            query={fh.query}
            imageUrl={getImageUrl(fh.hit.image_url || fh.hit.image_path)}
            isFlagged={isFlagged(fh.hit.asset_hash)}
            onToggleFlag={() => onToggleFlag(fh.hit, fh.query)}
            onPreviewDoc={onPreviewDoc}
            onDownload={() =>
              onDownloadImage(
                fh.hit.image_url || fh.hit.image_path,
                `${fh.hit.document_name}_p${fh.hit.page_number}_${fh.hit.asset_hash.slice(0, 8)}.png`
              )
            }
            compact
          />
        ))}
      </div>
    </div>
  );
}
