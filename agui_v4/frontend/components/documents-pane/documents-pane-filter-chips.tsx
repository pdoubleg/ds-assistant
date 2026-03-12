"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { getTagConfig } from "@/lib/tag-registry";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";
import type { FilterChip, Filters, TagFilterMode } from "./types";

export interface DocumentsPaneFilterChipsProps {
  filterChips: FilterChip[];
  filters: Filters;
  tagFilterMode: TagFilterMode;
  onRemoveFilterChip: (key: keyof Filters, value?: string) => void;
  onClearAllFilters: () => void;
  onToggleTagFilterMode: () => void;
}

/**
 * Active-filter chip row shown beneath the toolbar.
 */
export function DocumentsPaneFilterChips({
  filterChips,
  filters,
  tagFilterMode,
  onRemoveFilterChip,
  onClearAllFilters,
  onToggleTagFilterMode,
}: DocumentsPaneFilterChipsProps) {
  if (filterChips.length === 0) return null;

  return (
    <div className="flex items-center gap-1 px-3 py-1.5 border-b border-border/20 flex-wrap">
      {filterChips.map((chip, index) => {
        const isTagChip = chip.key === "tags";
        const tagCfg = isTagChip ? getTagConfig(chip.value) : null;
        const TagIcon = tagCfg?.icon;
        return (
          <Badge
            key={`${chip.key}-${chip.value}-${index}`}
            variant="secondary"
            className={cn(
              "text-[10px] px-1.5 py-0 gap-1 cursor-pointer hover:bg-destructive/20 inline-flex items-center",
              isTagChip && tagCfg && `${tagCfg.bg} ${tagCfg.text} ${tagCfg.border}`
            )}
            onClick={() => onRemoveFilterChip(chip.key, chip.value)}
          >
            {TagIcon && <TagIcon className="h-2.5 w-2.5 shrink-0" />}
            {chip.label}
            <X className="h-2.5 w-2.5" />
          </Badge>
        );
      })}
      <button
        onClick={onClearAllFilters}
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
          onClick={onToggleTagFilterMode}
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
  );
}
