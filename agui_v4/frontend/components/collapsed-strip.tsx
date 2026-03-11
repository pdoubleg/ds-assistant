"use client";

/**
 * CollapsedStrip — narrow sidebar indicator for a collapsed pane.
 * Shows the pane's icon and label vertically, with an expand button.
 */

import React from "react";
import { motion } from "framer-motion";
import { PanelLeftOpen } from "lucide-react";
import { NativeTooltip } from "@/components/ui/native-tooltip-shadcnui";

export type PaneId = "chat" | "documents" | "output";

export interface PaneMeta {
  icon: React.ReactNode;
  label: string;
}

export function CollapsedStrip({
  meta,
  onExpand,
}: {
  meta: PaneMeta;
  onExpand: () => void;
}) {
  return (
    <NativeTooltip content={`Expand ${meta.label}`} side="right">
      <motion.button
        initial={{ width: 0, opacity: 0 }}
        animate={{ width: 40, opacity: 1 }}
        exit={{ width: 0, opacity: 0 }}
        transition={{ duration: 0.2 }}
        onClick={onExpand}
        className="shrink-0 flex flex-col items-center justify-center gap-2 border-r border-border bg-card/70 hover:bg-secondary transition-colors cursor-pointer overflow-hidden"
      >
        <span className="text-primary">{meta.icon}</span>
        <span className="text-[10px] font-medium text-muted-foreground [writing-mode:vertical-lr] rotate-180 select-none tracking-wide">
          {meta.label}
        </span>
        <PanelLeftOpen className="h-4.5 w-4.5 text-muted-foreground/60" />
      </motion.button>
    </NativeTooltip>
  );
}
