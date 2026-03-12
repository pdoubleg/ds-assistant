"use client";

import React from "react";
import {
  ChevronsDownUp,
  ChevronsUpDown,
  Eye,
  EyeOff,
  FileUp,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";
import { cn } from "@/lib/utils";

export interface DocumentsPaneHeaderProps {
  allDocsCount: number;
  filteredDocsCount: number;
  hiddenCount: number;
  nonHiddenDocsCount: number;
  bulkExpanded: boolean | null;
  onUnhideAll: () => void;
  onHideAll: () => void;
  onExpandAll: () => void;
  onCollapseAll: () => void;
}

/**
 * Header row for the documents pane.
 */
export function DocumentsPaneHeader({
  allDocsCount,
  filteredDocsCount,
  hiddenCount,
  nonHiddenDocsCount,
  bulkExpanded,
  onUnhideAll,
  onHideAll,
  onExpandAll,
  onCollapseAll,
}: DocumentsPaneHeaderProps) {
  return (
    <div className="flex items-center gap-2.5 px-4 pr-10 border-b border-border/50 h-12">
      <FileUp className="h-[18px] w-[18px] text-primary shrink-0" />
      <h2 className="text-[15px] font-semibold tracking-tight text-foreground shrink-0">
        Documents
      </h2>
      <div className="ml-auto flex items-center gap-2">
        <div className="flex items-center rounded-lg border border-border/60 bg-secondary/30 overflow-hidden">
          <NativeTooltip content="Show All" side="bottom">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onUnhideAll}
              disabled={hiddenCount === 0}
              className={cn(
                "h-7 w-7 rounded-none",
                hiddenCount === 0
                  ? "text-muted-foreground/40"
                  : hiddenCount >= allDocsCount
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
              onClick={onHideAll}
              disabled={allDocsCount === 0 || hiddenCount >= allDocsCount}
              className={cn(
                "h-7 w-7 rounded-none border-l border-border/60",
                hiddenCount >= allDocsCount
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
              onClick={onExpandAll}
              disabled={nonHiddenDocsCount === 0}
              className={cn(
                "h-7 w-7 rounded-none border-l border-border/60",
                bulkExpanded === true
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
              onClick={onCollapseAll}
              disabled={nonHiddenDocsCount === 0}
              className={cn(
                "h-7 w-7 rounded-none border-l border-border/60",
                bulkExpanded === false
                  ? "bg-secondary text-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
              )}
            >
              <ChevronsDownUp className="h-3.5 w-3.5" />
            </Button>
          </NativeTooltip>
        </div>
        <Badge variant="secondary" className="text-[11px]">
          {allDocsCount} docs
        </Badge>
        <Badge className="text-[11px] bg-primary/15 text-primary border-primary/30">
          {filteredDocsCount} visible
        </Badge>
        {hiddenCount > 0 && (
          <Badge variant="outline" className="text-[11px] gap-1">
            <EyeOff className="h-3 w-3" />
            {hiddenCount}
          </Badge>
        )}
      </div>
    </div>
  );
}
