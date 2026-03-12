"use client";

import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";

export interface DocumentsPaneSummarizeBarProps {
  showSummarizeInput: boolean;
  isSummarizing: boolean;
  filteredDocsCount: number;
  summarizeInstructions: string;
  onSummarizeInstructionsChange: (value: string) => void;
  onRunSummarize: () => void;
  onClose: () => void;
}

/**
 * Input row for summarize instructions.
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
  return (
    <AnimatePresence>
      {showSummarizeInput && !isSummarizing && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="overflow-hidden border-b border-border/30"
        >
          <div className="flex items-center gap-1.5 px-3 py-2">
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
                onClick={onRunSummarize}
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
        </motion.div>
      )}
    </AnimatePresence>
  );
}
