"use client";

import React from "react";
import {
  ArrowUpDown,
  Filter,
  Info,
  Loader2,
  RotateCcw,
  ScanSearch,
  Search,
  Sparkles,
  Tag,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";
import { ShineBorder } from "@/components/ui/shine-border";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FilterDropdown } from "./filter-dropdown";
import type { Filters, FilterOptions, SortKey } from "./types";

export interface DocumentsPaneToolbarProps {
  summariesSize: number;
  searchScoresSize: number;
  agentTagsSize: number;
  allTagOptionsLength: number;
  isSummarizing: boolean;
  isSearching: boolean;
  isTagging: boolean;
  filteredDocsCount: number;
  useNarrowToolbar: boolean;
  showFilters: boolean;
  filters: Filters;
  filterOptions: FilterOptions;
  allTagOptions: string[];
  sortKey: SortKey;
  iconOnlyButtons: boolean;
  showTagConfirm: boolean;
  showSearchInput: boolean;
  showSummarizeInput: boolean;
  taggingProgress: {
    batch: number;
    totalBatches: number;
  } | null;
  summarizeProgress: {
    done: number;
    total: number;
  };
  onRefreshDocs: () => void;
  onToggleShowFilters: () => void;
  onToggleFilter: (key: keyof Omit<Filters, "search">, value: string) => void;
  onSortKeyChange: (key: SortKey) => void;
  onToggleTagConfirm: () => void;
  onToggleSearchInput: () => void;
  onToggleSummarizeInput: () => void;
  onCancelAutoTag: () => void;
  onCancelSearchSort: () => void;
  onCancelSummarize: () => void;
  onHandleDocLensClick: () => void;
}

/**
 * Main toolbar with filters, sorting, and AI actions.
 */
export function DocumentsPaneToolbar({
  summariesSize,
  searchScoresSize,
  agentTagsSize,
  allTagOptionsLength,
  isSummarizing,
  isSearching,
  isTagging,
  filteredDocsCount,
  useNarrowToolbar,
  showFilters,
  filters,
  filterOptions,
  allTagOptions,
  sortKey,
  iconOnlyButtons,
  showTagConfirm,
  showSearchInput,
  showSummarizeInput,
  taggingProgress,
  summarizeProgress,
  onRefreshDocs,
  onToggleShowFilters,
  onToggleFilter,
  onSortKeyChange,
  onToggleTagConfirm,
  onToggleSearchInput,
  onToggleSummarizeInput,
  onCancelAutoTag,
  onCancelSearchSort,
  onCancelSummarize,
  onHandleDocLensClick,
}: DocumentsPaneToolbarProps) {
  return (
    <div className="flex items-center gap-1.5 px-3 py-2 border-b border-border/30 min-w-0">
      <NativeTooltip
        content="Clear generated AI content"
        side="bottom"
        animation="blur"
        shine
      >
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-7 text-[11px] gap-1"
          onClick={onRefreshDocs}
          disabled={
            (summariesSize === 0 &&
              searchScoresSize === 0 &&
              agentTagsSize === 0 &&
              allTagOptionsLength === 0) ||
            isSummarizing ||
            isSearching ||
            isTagging
          }
        >
          <RotateCcw className="h-3 w-3" />
          Refresh docs
        </Button>
      </NativeTooltip>

      <div className="flex items-center gap-1.5 min-w-0 shrink overflow-hidden">
        {useNarrowToolbar ? (
          <NativeTooltip content="Filters" side="bottom" animation="blur">
            <Button
              variant={showFilters ? "secondary" : "ghost"}
              size="icon"
              className="h-7 w-7 relative"
              onClick={onToggleShowFilters}
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
              onToggle={(value) => onToggleFilter("mimeTypes", value)}
            />
            <FilterDropdown
              label="Doc Type"
              options={filterOptions.docTypes}
              selected={filters.docTypes}
              onToggle={(value) => onToggleFilter("docTypes", value)}
            />
            {filterOptions.subTypes.length > 0 && (
              <FilterDropdown
                label="Sub Type"
                options={filterOptions.subTypes}
                selected={filters.subTypes}
                onToggle={(value) => onToggleFilter("subTypes", value)}
              />
            )}
            <FilterDropdown
              label="Domain"
              options={filterOptions.domains}
              selected={filters.domains}
              onToggle={(value) => onToggleFilter("domains", value)}
            />
            {allTagOptions.length > 0 && (
              <FilterDropdown
                label="Tags"
                options={allTagOptions}
                selected={filters.tags}
                onToggle={(value) => onToggleFilter("tags", value)}
              />
            )}
          </>
        )}

        <div className="flex items-center gap-1 shrink-0">
          <ArrowUpDown className="h-3 w-3 text-muted-foreground" />
          <Select
            value={sortKey}
            onValueChange={(value) => onSortKeyChange(value as SortKey)}
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
      </div>

      <div className="flex flex-col items-end gap-1.5 ml-auto rounded-md border border-border/40 px-2 pt-1.5 pb-1.5">
        {filteredDocsCount > 0 && (
          <NativeTooltip
            content={`All ${filteredDocsCount} visible doc${filteredDocsCount !== 1 ? "s" : ""} are sent to the model when you run any AI action. Hide or filter docs to narrow the scope.`}
            side="top"
            align="end"
          >
            <span className="flex items-center gap-1 text-[11px] text-muted-foreground/60 cursor-default select-none w-full justify-center">
              <Info className="h-3 w-3 shrink-0" />
              <span>
                {filteredDocsCount} doc{filteredDocsCount !== 1 ? "s" : ""} in
                scope
              </span>
            </span>
          </NativeTooltip>
        )}

        <div className="flex items-center gap-1">
          <NativeTooltip
            content={
              isTagging
                ? "Cancel auto-tagging"
                : filteredDocsCount === 0
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
                onClick={isTagging ? onCancelAutoTag : onToggleTagConfirm}
                disabled={filteredDocsCount === 0 && !isTagging}
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

          <NativeTooltip
            content={
              isSearching
                ? "Cancel search"
                : filteredDocsCount < 2
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
                onClick={isSearching ? onCancelSearchSort : onToggleSearchInput}
                disabled={filteredDocsCount < 2 && !isSearching}
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

          <NativeTooltip
            content={
              isSummarizing
                ? "Cancel summarization"
                : filteredDocsCount === 0
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
                onClick={
                  isSummarizing ? onCancelSummarize : onToggleSummarizeInput
                }
                disabled={filteredDocsCount === 0 && !isSummarizing}
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

          <NativeTooltip
            content={
              filteredDocsCount === 0
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
                onClick={onHandleDocLensClick}
                disabled={filteredDocsCount === 0}
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
        </div>
      </div>
    </div>
  );
}

function activeFilterCount(filters: Filters): number {
  return (
    (filters.search.length > 0 ? 1 : 0) +
    filters.mimeTypes.size +
    filters.docTypes.size +
    filters.subTypes.size +
    filters.domains.size +
    filters.tags.size
  );
}
