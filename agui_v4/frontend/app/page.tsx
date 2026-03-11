"use client";

/**
 * Q-Bot App — Three-Pane Collapsible Layout
 *
 * Layout (left to right, default proportions):
 *   1. Chat UI      (flex 1)  — user input with file upload
 *   2. Documents     (flex 1)  — uploaded / retrieved document cards
 *   3. Audit Output  (flex 2)  — generated forms, charts, tables
 *
 * Each pane can be collapsed to a narrow sidebar strip and re-expanded.
 */

import React, { useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquareText,
  FolderOpen,
  LayoutDashboard,
  PanelLeftClose,
} from "lucide-react";
import { AppHeader } from "@/components/app-header";
import { ChatPane } from "@/components/chat-pane";
import { DocumentsPane } from "@/components/documents-pane";
import { OutputPane } from "@/components/output-pane";
import {
  CollapsedStrip,
  type PaneId,
  type PaneMeta,
} from "@/components/collapsed-strip";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";

const PANE_META: Record<PaneId, PaneMeta> = {
  chat: {
    icon: <MessageSquareText className="h-4 w-4" />,
    label: "Chat",
  },
  documents: {
    icon: <FolderOpen className="h-4 w-4" />,
    label: "Documents",
  },
  output: {
    icon: <LayoutDashboard className="h-4 w-4" />,
    label: "Output",
  },
};

const PANE_CONTENT: Record<PaneId, React.FC> = {
  chat: ChatPane,
  documents: DocumentsPane,
  output: OutputPane,
};

const PANE_FLEX: Record<PaneId, number> = {
  chat: 1,
  documents: 1,
  output: 2,
};

export default function HomePage() {
  const [expanded, setExpanded] = useState<Record<PaneId, boolean>>({
    chat: true,
    documents: true,
    output: true,
  });

  const expandedCount = useMemo(
    () => Object.values(expanded).filter(Boolean).length,
    [expanded]
  );

  const togglePane = useCallback(
    (id: PaneId) => {
      setExpanded((prev) => {
        const isOpen = prev[id];
        // Don't allow collapsing the last open pane
        if (isOpen && expandedCount <= 1) return prev;
        return { ...prev, [id]: !isOpen };
      });
    },
    [expandedCount]
  );

  const paneOrder: PaneId[] = ["chat", "documents", "output"];

  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
        <AppHeader />

        {/* Three-pane layout */}
        <div className="flex-1 flex min-h-0">
          <AnimatePresence initial={false}>
            {paneOrder.map((id, idx) => {
              const isOpen = expanded[id];
              const PaneComponent = PANE_CONTENT[id];
              const isLast = idx === paneOrder.length - 1;
              const canCollapse = expandedCount > 1 || !isOpen;

              if (!isOpen) {
                return (
                  <CollapsedStrip
                    key={`strip-${id}`}
                    meta={PANE_META[id]}
                    onExpand={() => togglePane(id)}
                  />
                );
              }

              return (
                <motion.div
                  key={`pane-${id}`}
                  layout
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className={`min-w-0 min-h-0 flex flex-col ${
                    !isLast ? "border-r border-border" : ""
                  } ${id === "output" ? "bg-background" : "bg-card/80"}`}
                  style={{ flex: PANE_FLEX[id] }}
                >
                  <div className="relative">
                    {canCollapse && (
                      <NativeTooltip
                        content={`Collapse ${PANE_META[id].label}`}
                        side="bottom"
                      >
                        <button
                          onClick={() => togglePane(id)}
                          className="absolute top-2.5 right-2 z-10 p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary hover:ring-1 hover:ring-border transition-colors"
                        >
                          <PanelLeftClose className="h-4.5 w-4.5" />
                        </button>
                      </NativeTooltip>
                    )}
                  </div>
                  <PaneComponent />
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
    </div>
  );
}
