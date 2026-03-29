"use client";

import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Eye,
  EyeOff,
  Undo2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  deriveFileExt,
  getFileTypeBadgeClass,
  type DocumentTagData,
} from "@/components/a2ui/documents";
import { getTagConfig } from "@/lib/tag-registry";
import { cn } from "@/lib/utils";
import type { DocWithId, HiddenSortKey, HiddenStats } from "./types";

export interface HiddenDocumentsDockProps {
  showHiddenDock: boolean;
  hiddenCount: number;
  filterHiddenCount: number;
  dockExpanded: boolean;
  dockMinimized: boolean;
  hiddenStats: HiddenStats;
  hiddenDocs: DocWithId[];
  sortedHiddenDocs: DocWithId[];
  hiddenSortKey: HiddenSortKey;
  hiddenSortDir: "asc" | "desc";
  isCompactHiddenTable: boolean;
  showHiddenTagsColumn: boolean;
  agentTags: Map<string, DocumentTagData[]>;
  onToggleDockExpanded: () => void;
  onToggleDockMinimized: () => void;
  onUnhideAll: () => void;
  onHandleHiddenSort: (key: HiddenSortKey) => void;
  onPreviewDoc: (doc: DocWithId) => void;
  onUnhide: (fileName: string) => void;
}

/**
 * Bottom dock for hidden documents and filtered-out stats.
 */
export function HiddenDocumentsDock({
  showHiddenDock,
  hiddenCount,
  filterHiddenCount,
  dockExpanded,
  dockMinimized,
  hiddenStats,
  hiddenDocs,
  sortedHiddenDocs,
  hiddenSortKey,
  hiddenSortDir,
  isCompactHiddenTable,
  showHiddenTagsColumn,
  agentTags,
  onToggleDockExpanded,
  onToggleDockMinimized,
  onUnhideAll,
  onHandleHiddenSort,
  onPreviewDoc,
  onUnhide,
}: HiddenDocumentsDockProps) {
  if (!showHiddenDock) return null;

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 px-2 pb-2">
      <div className="pointer-events-auto rounded-lg border border-border/60 bg-background/95 shadow-lg backdrop-blur-sm">
        <div className="flex items-center gap-2 px-2 pt-2 pb-1.5">
          <button
            className="group flex flex-1 items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted/45 transition-colors"
            onClick={onToggleDockExpanded}
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
                  <Badge
                    variant="secondary"
                    className="text-[10px] px-1.5 py-0"
                  >
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
              onClick={onUnhideAll}
            >
              Unhide all
            </Button>
          )}

          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            onClick={onToggleDockMinimized}
            title={dockMinimized ? "Expand hidden dock" : "Minimize hidden dock"}
          >
            {dockMinimized ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </Button>
        </div>

        {!dockMinimized && !dockExpanded && (
          <div className="border-t border-border/35 px-3 py-2">
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
              <span>
                Hidden: <strong className="text-foreground">{hiddenCount}</strong>
              </span>
              <span>
                Hidden %:{" "}
                <strong className="text-foreground">
                  {hiddenStats.hiddenPercent}%
                </strong>
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
                {hiddenDocs.length === 0 ? (
                  <div className="rounded-md border border-dashed border-border/50 bg-muted/25 px-3 py-3">
                    <div className="flex items-center gap-1.5 flex-wrap text-[11px]">
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        Hidden: 0
                      </Badge>
                      <Badge
                        variant="secondary"
                        className="text-[10px] px-1.5 py-0"
                      >
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
                            onClick={() => onHandleHiddenSort("file_name")}
                          >
                            <div className="flex items-center gap-1">
                              Document
                              <SortIndicator
                                active={hiddenSortKey === "file_name"}
                                direction={hiddenSortDir}
                              />
                            </div>
                          </TableHead>
                          <TableHead
                            className="w-[80px] cursor-pointer select-none"
                            onClick={() => onHandleHiddenSort("ext")}
                          >
                            <div className="flex items-center gap-1">
                              Type
                              <SortIndicator
                                active={hiddenSortKey === "ext"}
                                direction={hiddenSortDir}
                              />
                            </div>
                          </TableHead>
                          {!isCompactHiddenTable && (
                            <>
                              <TableHead
                                className="w-[90px] cursor-pointer select-none"
                                onClick={() => onHandleHiddenSort("domain")}
                              >
                                <div className="flex items-center gap-1">
                                  Domain
                                  <SortIndicator
                                    active={hiddenSortKey === "domain"}
                                    direction={hiddenSortDir}
                                  />
                                </div>
                              </TableHead>
                              <TableHead
                                className="min-w-[120px] cursor-pointer select-none"
                                onClick={() => onHandleHiddenSort("document_type")}
                              >
                                <div className="flex items-center gap-1">
                                  Doc Type
                                  <SortIndicator
                                    active={hiddenSortKey === "document_type"}
                                    direction={hiddenSortDir}
                                  />
                                </div>
                              </TableHead>
                              <TableHead
                                className="min-w-[120px] cursor-pointer select-none"
                                onClick={() =>
                                  onHandleHiddenSort("document_sub_type")
                                }
                              >
                                <div className="flex items-center gap-1">
                                  Subtype
                                  <SortIndicator
                                    active={hiddenSortKey === "document_sub_type"}
                                    direction={hiddenSortDir}
                                  />
                                </div>
                              </TableHead>
                              <TableHead
                                className="w-[115px] cursor-pointer select-none"
                                onClick={() => onHandleHiddenSort("create_date")}
                              >
                                <div className="flex items-center gap-1">
                                  Created
                                  <SortIndicator
                                    active={hiddenSortKey === "create_date"}
                                    direction={hiddenSortDir}
                                  />
                                </div>
                              </TableHead>
                              <TableHead
                                className="min-w-[100px] cursor-pointer select-none"
                                onClick={() =>
                                  onHandleHiddenSort("source_system")
                                }
                              >
                                <div className="flex items-center gap-1">
                                  Source
                                  <SortIndicator
                                    active={hiddenSortKey === "source_system"}
                                    direction={hiddenSortDir}
                                  />
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
                              onClick={() => onHandleHiddenSort("create_date")}
                            >
                              <div className="flex items-center gap-1">
                                Created
                                <SortIndicator
                                  active={hiddenSortKey === "create_date"}
                                  direction={hiddenSortDir}
                                />
                              </div>
                            </TableHead>
                          )}
                          <TableHead className="sticky right-0 z-20 w-[88px] bg-secondary/70">
                            Actions
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortedHiddenDocs.map((doc) => (
                          <HiddenDocRow
                            key={doc._id}
                            doc={doc}
                            isCompactHiddenTable={isCompactHiddenTable}
                            showHiddenTagsColumn={showHiddenTagsColumn}
                            tags={agentTags.get(doc.file_name) || []}
                            onPreviewDoc={onPreviewDoc}
                            onUnhide={onUnhide}
                          />
                        ))}
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
  );
}

interface SortIndicatorProps {
  active: boolean;
  direction: "asc" | "desc";
}

function SortIndicator({ active, direction }: SortIndicatorProps) {
  if (active) {
    return direction === "asc" ? (
      <ArrowUp className="h-3 w-3" />
    ) : (
      <ArrowDown className="h-3 w-3" />
    );
  }

  return <ArrowUpDown className="h-3 w-3 opacity-30" />;
}

interface HiddenDocRowProps {
  doc: DocWithId;
  isCompactHiddenTable: boolean;
  showHiddenTagsColumn: boolean;
  tags: DocumentTagData[];
  onPreviewDoc: (doc: DocWithId) => void;
  onUnhide: (fileName: string) => void;
}

function HiddenDocRow({
  doc,
  isCompactHiddenTable,
  showHiddenTagsColumn,
  tags,
  onPreviewDoc,
  onUnhide,
}: HiddenDocRowProps) {
  const normalizedExt = deriveFileExt(doc.mime_type, doc.file_name);
  const ext = normalizedExt.toUpperCase();
  const createDate = doc.create_date
    ? new Date(doc.create_date).toLocaleDateString()
    : "—";
  const visibleTags = tags.slice(0, 2);
  const overflowTagCount =
    tags.length > visibleTags.length ? tags.length - visibleTags.length : 0;

  return (
    <TableRow className="odd:bg-muted/15">
      <TableCell className="max-w-[350px]">
        <div className="truncate font-medium text-foreground">{doc.file_name}</div>
        <div className="text-[10px] text-muted-foreground truncate">
          {doc.document_description || "No description"}
        </div>
      </TableCell>
      <TableCell>
        <Badge
          variant="outline"
          className={cn(
            "font-mono text-[10px] px-1.5 py-0",
            getFileTypeBadgeClass(normalizedExt)
          )}
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
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                {doc.document_type}
              </Badge>
            ) : (
              "—"
            )}
          </TableCell>
          <TableCell className="max-w-[160px] truncate">
            {doc.document_sub_type ? (
              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
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
            onClick={() => onPreviewDoc(doc)}
            title="View document"
          >
            <Eye className="h-4 w-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            onClick={() => onUnhide(doc.file_name)}
            title="Unhide document"
          >
            <Undo2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
}
