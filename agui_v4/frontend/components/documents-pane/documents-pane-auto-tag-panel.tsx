"use client";

import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  CheckSquare,
  Plus,
  Send,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ALL_TAGS,
  CUSTOM_FALLBACK_TAG_LABEL,
  getTagConfig,
} from "@/lib/tag-registry";
import { cn } from "@/lib/utils";
import type { AutoTagMode } from "./types";

export interface DocumentsPaneAutoTagPanelProps {
  showTagConfirm: boolean;
  isTagging: boolean;
  filteredDocsCount: number;
  tagMode: AutoTagMode;
  hasCustomSelection: boolean;
  selectedCustomTags: string[];
  customTagCatalog: string[];
  customTagDraft: string;
  customTagError: string | null;
  customSelectionNotice: string | null;
  selectedCustomTagSet: Set<string>;
  maxCustomTags: number;
  onRunAutoTag: () => void;
  onClose: () => void;
  onSetTagMode: (mode: AutoTagMode) => void;
  onSelectAllCustomTags: () => void;
  onUnselectAllCustomTags: () => void;
  onRestoreDefaultCustomTags: () => void;
  onRemoveAllCustomTags: () => void;
  onCustomTagDraftChange: (value: string) => void;
  onAddCustomTag: () => void;
  onToggleCustomTagSelection: (label: string) => void;
  onRemoveCustomTag: (label: string) => void;
}

/**
 * Confirmation and customization panel for auto-tagging.
 */
export function DocumentsPaneAutoTagPanel({
  showTagConfirm,
  isTagging,
  filteredDocsCount,
  tagMode,
  hasCustomSelection,
  selectedCustomTags,
  customTagCatalog,
  customTagDraft,
  customTagError,
  customSelectionNotice,
  selectedCustomTagSet,
  maxCustomTags,
  onRunAutoTag,
  onClose,
  onSetTagMode,
  onSelectAllCustomTags,
  onUnselectAllCustomTags,
  onRestoreDefaultCustomTags,
  onRemoveAllCustomTags,
  onCustomTagDraftChange,
  onAddCustomTag,
  onToggleCustomTagSelection,
  onRemoveCustomTag,
}: DocumentsPaneAutoTagPanelProps) {
  return (
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
                  Auto-tag {filteredDocsCount} visible document
                  {filteredDocsCount !== 1 ? "s" : ""}?
                </p>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <NativeTooltip
                  content="Run auto-tagging"
                  side="bottom"
                  animation="blur"
                >
                  <Button
                    variant="default"
                    size="sm"
                    className="h-7 text-[11px] gap-1"
                    onClick={onRunAutoTag}
                    disabled={
                      filteredDocsCount === 0 ||
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
                    onClick={onClose}
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
                      onClick={() => onSetTagMode("default")}
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
                      onClick={() => onSetTagMode("custom")}
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
                      {selectedCustomTags.length} selected of{" "}
                      {customTagCatalog.length} configured tags ({maxCustomTags}{" "}
                      max).
                    </p>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-7 text-[11px]"
                      onClick={onSelectAllCustomTags}
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
                      onClick={onUnselectAllCustomTags}
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
                      onClick={onRestoreDefaultCustomTags}
                      disabled={
                        customTagCatalog.length === ALL_TAGS.length &&
                        customTagCatalog.every(
                          (tag, index) => tag === ALL_TAGS[index]
                        ) &&
                        selectedCustomTags.length === ALL_TAGS.length &&
                        selectedCustomTags.every(
                          (tag, index) => tag === ALL_TAGS[index]
                        )
                      }
                    >
                      Restore Default
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-7 text-[11px]"
                      onClick={onRemoveAllCustomTags}
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
                    onChange={(event) =>
                      onCustomTagDraftChange(event.target.value)
                    }
                    placeholder="Add a tag"
                    className="h-8 text-xs flex-1"
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        onAddCustomTag();
                      }
                    }}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8 text-[11px] shrink-0"
                    onClick={onAddCustomTag}
                    disabled={customTagCatalog.length >= maxCustomTags}
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
                    Docs with no matching custom tags will get `
                    {CUSTOM_FALLBACK_TAG_LABEL}`.
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
                            <TableHead className="w-[64px] text-right">
                              Remove
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {customTagCatalog.map((tagLabel) => {
                            const cfg = getTagConfig(tagLabel);
                            const Icon = cfg.icon;
                            const isSelected =
                              selectedCustomTagSet.has(tagLabel);
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
                                    variant={
                                      isSelected ? "secondary" : "outline"
                                    }
                                    size="sm"
                                    className="h-7 text-[11px]"
                                    onClick={() =>
                                      onToggleCustomTagSelection(tagLabel)
                                    }
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
                                    onClick={() => onRemoveCustomTag(tagLabel)}
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
  );
}
