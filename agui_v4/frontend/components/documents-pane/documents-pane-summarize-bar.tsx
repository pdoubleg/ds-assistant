"use client";

import React from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Lightbulb, Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";

const SUMMARIZE_SUGGESTIONS: ReadonlyArray<{
  title: string;
  message: string;
}> = [
  {
    title: "Focus on dollar amounts",
    message: "focus on dollar amounts found in the documents. Include markdown table summarizing the amounts.",
  },
  {
    title: "Be extra concise",
    message: "be ultra-concise; make every word count.",
  },
];

export interface DocumentsPaneSummarizeBarProps {
  showSummarizeInput: boolean;
  isSummarizing: boolean;
  filteredDocsCount: number;
  summarizeInstructions: string;
  onSummarizeInstructionsChange: (value: string) => void;
  onRunSummarize: (overrideInstructions?: string) => void | Promise<void>;
  onClose: () => void;
}

/**
 * Input row for summarize instructions with an inline suggestions popover.
 *
 * Suggestions are hidden behind a small lightbulb icon to keep the UI
 * uncluttered. Clicking the icon reveals a floating pill list; selecting
 * a suggestion (or clicking outside) closes the popover automatically.
 */
export function DocumentsPaneSummarizeBar({
  showSummarizeInput,
  isSummarizing,
  filteredDocsCount,
  summarizeInstructions,
  onSummarizeInstructionsChange,
  onRunSummarize,
  onClose,
}: DocumentsPaneSummarizeBarProps) {
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
        {showSummarizeInput && !isSummarizing && (
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
                  value={summarizeInstructions}
                  onChange={(event) =>
                    onSummarizeInstructionsChange(event.target.value)
                  }
                  placeholder={`Optional: tell the summarizer what to focus on for ${filteredDocsCount} visible document${filteredDocsCount !== 1 ? "s" : ""}`}
                  className="h-7 text-xs flex-1"
                  onKeyDown={(event) => {
                    if (event.key === "Enter") onRunSummarize();
                  }}
                  autoFocus
                />
                <NativeTooltip
                  content="Run summarization"
                  side="bottom"
                  animation="blur"
                >
                  <Button
                    variant="default"
                    size="icon"
                    className="h-7 w-7 shrink-0"
                    onClick={() => {
                      void onRunSummarize();
                    }}
                    disabled={filteredDocsCount === 0}
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
            className="z-[9999] flex flex-wrap gap-1.5 rounded-lg border border-border/40 bg-popover p-2 shadow-lg"
          >
            {SUMMARIZE_SUGGESTIONS.map((suggestion) => (
              <Button
                key={suggestion.title}
                type="button"
                variant="outline"
                size="sm"
                disabled={filteredDocsCount === 0}
                onClick={() => {
                  setShowSuggestions(false);
                  void onRunSummarize(suggestion.message);
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
