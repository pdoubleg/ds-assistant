"use client";

import React from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Lightbulb, Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";

const SEARCH_SORT_SUGGESTIONS: ReadonlyArray<{
  title: string;
  message: string;
}> = [
  {
    title: "Find roofing docs",
    message: "please find all documents related to roofing, sorted by importance.",
  },
  {
    title: "Get all customer contacts",
    message: "get all customer contacts sorted by importance.",
  },
];

export interface DocumentsPaneSearchBarProps {
  showSearchInput: boolean;
  isSearching: boolean;
  filteredDocsCount: number;
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  onRunSearchSort: (overrideQuery?: string) => void | Promise<void>;
  onClose: () => void;
}

/**
 * Input row for the search-and-sort query with an inline suggestions popover.
 *
 * Suggestions are hidden behind a small lightbulb icon to keep the UI
 * uncluttered. Clicking the icon reveals a floating pill list; selecting
 * a suggestion (or clicking outside) closes the popover automatically.
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
  const [showSuggestions, setShowSuggestions] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const popoverRef = React.useRef<HTMLDivElement>(null);
  const [popoverPos, setPopoverPos] = React.useState<{ top: number; left: number } | null>(null);

  // Position the portal popover relative to the trigger button
  React.useEffect(() => {
    if (!showSuggestions || !triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    setPopoverPos({ top: rect.bottom + 4, left: rect.left });
  }, [showSuggestions]);

  // Close popover on outside click
  React.useEffect(() => {
    if (!showSuggestions) return;
    const handleClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        triggerRef.current?.contains(target) ||
        popoverRef.current?.contains(target)
      ) return;
      setShowSuggestions(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showSuggestions]);

  return (
    <>
      <AnimatePresence>
        {showSearchInput && !isSearching && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-border/30"
          >
            <div className="px-3 py-2">
              <div className="flex items-center gap-1.5">
                {/* Suggestions icon — popover portals to body to escape overflow clip */}
                <NativeTooltip
                  content="Suggestions"
                  side="bottom"
                  animation="blur"
                >
                  <Button
                    ref={triggerRef}
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 text-muted-foreground hover:text-foreground"
                    onClick={() => setShowSuggestions((prev) => !prev)}
                  >
                    <Lightbulb className="h-3.5 w-3.5" />
                  </Button>
                </NativeTooltip>

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
                    onClick={() => {
                      void onRunSearchSort();
                    }}
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
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Portaled popover — rendered on document.body to escape overflow-hidden ancestors */}
      {showSuggestions &&
        popoverPos &&
        createPortal(
          <div
            ref={popoverRef}
            style={{ position: "fixed", top: popoverPos.top, left: popoverPos.left }}
            className="z-9999 flex flex-wrap gap-1.5 rounded-lg border border-border/40 bg-popover p-2 shadow-lg"
          >
            {SEARCH_SORT_SUGGESTIONS.map((suggestion) => (
              <Button
                key={suggestion.title}
                type="button"
                variant="outline"
                size="sm"
                disabled={filteredDocsCount === 0}
                onClick={() => {
                  setShowSuggestions(false);
                  void onRunSearchSort(suggestion.message);
                }}
                className="h-auto whitespace-nowrap rounded-full border-primary/30 bg-primary/5 px-3 py-1 text-xs text-foreground hover:border-primary/50 hover:bg-primary/10"
              >
                {suggestion.title}
              </Button>
            ))}
          </div>,
          document.body
        )}
    </>
  );
}
