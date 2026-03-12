"use client";

import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";

export interface DocumentsPaneSearchBarProps {
  showSearchInput: boolean;
  isSearching: boolean;
  filteredDocsCount: number;
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  onRunSearchSort: () => void;
  onClose: () => void;
}

/**
 * Input row for the search-and-sort query.
 */
export function DocumentsPaneSearchBar({
  showSearchInput,
  isSearching,
  filteredDocsCount,
  searchQuery,
  onSearchQueryChange,
  onRunSearchSort,
  onClose,
}: DocumentsPaneSearchBarProps) {
  return (
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
              onChange={(event) => onSearchQueryChange(event.target.value)}
              placeholder='e.g. "find all estimates" or "rank by relevance to roofing"'
              className="h-7 text-xs flex-1"
              onKeyDown={(event) => {
                if (event.key === "Enter") onRunSearchSort();
              }}
              autoFocus
            />
            <NativeTooltip
              content="Run search & sort"
              side="bottom"
              animation="blur"
            >
              <Button
                variant="default"
                size="icon"
                className="h-7 w-7 shrink-0"
                onClick={onRunSearchSort}
                disabled={filteredDocsCount === 0 || !searchQuery.trim()}
              >
                <Send className="h-3 w-3" />
              </Button>
            </NativeTooltip>
            <NativeTooltip content="Cancel" side="bottom" animation="blur">
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 shrink-0"
                onClick={onClose}
              >
                <X className="h-3 w-3" />
              </Button>
            </NativeTooltip>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
