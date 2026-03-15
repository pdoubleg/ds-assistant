"use client";

/**
 * FindingCard Component
 *
 * Polished alert-style card for agent observations about a claim.
 * Displays a severity-colored accent, icon-mapped category badge,
 * title, markdown-rendered content, and a copy button.
 *
 * Categories and severities are typed string literals that map to
 * distinct icons and color palettes for quick visual scanning.
 */

import React, { useCallback, useMemo, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ShieldCheck,
  Scale,
  Hammer,
  Clock,
  FileSearch,
  ShieldAlert,
  DollarSign,
  Fingerprint,
  HeartPulse,
  ArrowLeftRight,
  CircleDot,
  Lightbulb,
  Info,
  StickyNote,
  TriangleAlert,
  OctagonAlert,
  Siren,
  Copy,
  Check,
  Truck,
  Gavel,
  Headset,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type FindingCategory =
  | "coverage"
  | "liability"
  | "damages"
  | "time_sensitive"
  | "documentation"
  | "compliance"
  | "financial"
  | "fraud"
  | "medical"
  | "subrogation"
  | "vendor"
  | "litigation"
  | "customer_service"
  | "general";

type FindingSeverity =
  | "tip"
  | "info"
  | "note"
  | "warning"
  | "critical"
  | "urgent";

export interface FindingCardProps {
  title: string;
  content: string;
  severity?: FindingSeverity;
  category?: FindingCategory | null;
}

// ---------------------------------------------------------------------------
// Category icon / color config
// ---------------------------------------------------------------------------

const CATEGORY_CONFIG: Record<
  FindingCategory,
  { label: string; icon: React.ReactNode; badgeBg: string }
> = {
  coverage: {
    label: "Coverage",
    icon: <ShieldCheck className="h-3 w-3" />,
    badgeBg:
      "bg-[#28A3AF]/8 text-[#06748C] ring-[#28A3AF]/25 dark:bg-[#28A3AF]/15 dark:text-[#78E1E1] dark:ring-[#28A3AF]/30",
  },
  liability: {
    label: "Liability",
    icon: <Scale className="h-3 w-3" />,
    badgeBg:
      "bg-purple-50 text-purple-700 ring-purple-200/50 dark:bg-purple-950/40 dark:text-purple-300 dark:ring-purple-700/40",
  },
  damages: {
    label: "Damages",
    icon: <Hammer className="h-3 w-3" />,
    badgeBg:
      "bg-orange-50 text-orange-700 ring-orange-200/50 dark:bg-orange-950/40 dark:text-orange-300 dark:ring-orange-700/40",
  },
  time_sensitive: {
    label: "Time Sensitive",
    icon: <Clock className="h-3 w-3" />,
    badgeBg:
      "bg-[#06748C]/8 text-[#06748C] ring-[#06748C]/20 dark:bg-[#06748C]/15 dark:text-[#99E5EA] dark:ring-[#28A3AF]/30",
  },
  documentation: {
    label: "Documentation",
    icon: <FileSearch className="h-3 w-3" />,
    badgeBg:
      "bg-violet-50 text-violet-700 ring-violet-200/50 dark:bg-violet-950/40 dark:text-violet-300 dark:ring-violet-700/40",
  },
  compliance: {
    label: "Compliance",
    icon: <ShieldAlert className="h-3 w-3" />,
    badgeBg:
      "bg-[#1A1446]/6 text-[#29254F] ring-[#1A1446]/15 dark:bg-indigo-950/40 dark:text-indigo-300 dark:ring-indigo-700/40",
  },
  financial: {
    label: "Financial",
    icon: <DollarSign className="h-3 w-3" />,
    badgeBg:
      "bg-emerald-50 text-emerald-700 ring-emerald-200/50 dark:bg-emerald-950/40 dark:text-emerald-300 dark:ring-emerald-700/40",
  },
  fraud: {
    label: "Fraud",
    icon: <Fingerprint className="h-3 w-3" />,
    badgeBg:
      "bg-red-50 text-red-700 ring-red-200/50 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-700/40",
  },
  medical: {
    label: "Medical",
    icon: <HeartPulse className="h-3 w-3" />,
    badgeBg:
      "bg-rose-50 text-rose-700 ring-rose-200/50 dark:bg-rose-950/40 dark:text-rose-300 dark:ring-rose-700/40",
  },
  subrogation: {
    label: "Subrogation",
    icon: <ArrowLeftRight className="h-3 w-3" />,
    badgeBg:
      "bg-[#78E1E1]/8 text-[#06748C] ring-[#78E1E1]/25 dark:bg-[#78E1E1]/12 dark:text-[#AEEDED] dark:ring-[#78E1E1]/25",
  },
  vendor: {
    label: "Vendor",
    icon: <Truck className="h-3 w-3" />,
    badgeBg:
      "bg-[#FFD000]/10 text-amber-700 ring-[#FFD000]/25 dark:bg-[#FFD000]/12 dark:text-[#FFE280] dark:ring-[#FFD000]/25",
  },
  litigation: {
    label: "Litigation",
    icon: <Gavel className="h-3 w-3" />,
    badgeBg:
      "bg-[#1A1446]/6 text-[#29254F] ring-[#1A1446]/15 dark:bg-indigo-950/40 dark:text-indigo-300 dark:ring-indigo-700/40",
  },
  customer_service: {
    label: "Customer Service",
    icon: <Headset className="h-3 w-3" />,
    badgeBg:
      "bg-[#28A3AF]/8 text-[#06748C] ring-[#28A3AF]/25 dark:bg-[#28A3AF]/15 dark:text-[#78E1E1] dark:ring-[#28A3AF]/30",
  },
  general: {
    label: "General",
    icon: <CircleDot className="h-3 w-3" />,
    badgeBg:
      "bg-[#343741]/5 text-[#565656] ring-[#343741]/12 dark:bg-[#343741]/20 dark:text-[#C0BFC0] dark:ring-[#565656]/25",
  },
};

// ---------------------------------------------------------------------------
// Severity config
// ---------------------------------------------------------------------------

const SEVERITY_CONFIG: Record<
  FindingSeverity,
  {
    icon: React.ReactNode;
    label: string;
    border: string;
    bg: string;
    titleColor: string;
    accentColor: string;
    iconBg: string;
    hoverGlow: string;
  }
> = {
  tip: {
    icon: <Lightbulb className="h-4 w-4" strokeWidth={1.75} />,
    label: "Tip",
    border: "border-l-[#78E1E1] dark:border-l-[#99E5EA]",
    bg: "bg-gradient-to-br from-[#78E1E1]/5 via-transparent to-[#AEEDED]/5",
    titleColor: "text-[#06748C] dark:text-[#99E5EA]",
    accentColor: "text-[#28A3AF] dark:text-[#78E1E1]",
    iconBg: "bg-[#78E1E1]/12 dark:bg-[#78E1E1]/15",
    hoverGlow:
      "hover:ring-2 hover:ring-[#78E1E1]/30 dark:hover:ring-[#78E1E1]/25 hover:shadow-[0_4px_20px_-4px_rgba(120,225,225,0.18)]",
  },
  info: {
    icon: <Info className="h-4 w-4" strokeWidth={1.75} />,
    label: "Info",
    border: "border-l-[#06748C] dark:border-l-[#28A3AF]",
    bg: "bg-gradient-to-br from-[#06748C]/5 via-transparent to-[#28A3AF]/5",
    titleColor: "text-[#06748C] dark:text-[#78E1E1]",
    accentColor: "text-[#06748C] dark:text-[#28A3AF]",
    iconBg: "bg-[#06748C]/10 dark:bg-[#28A3AF]/15",
    hoverGlow:
      "hover:ring-2 hover:ring-[#28A3AF]/30 dark:hover:ring-[#28A3AF]/25 hover:shadow-[0_4px_20px_-4px_rgba(6,116,140,0.15)]",
  },
  note: {
    icon: <StickyNote className="h-4 w-4" strokeWidth={1.75} />,
    label: "Note",
    border: "border-l-[#29254F] dark:border-l-indigo-400",
    bg: "bg-gradient-to-br from-[#1A1446]/4 via-transparent to-indigo-400/5",
    titleColor: "text-[#1A1446] dark:text-indigo-300",
    accentColor: "text-[#29254F] dark:text-indigo-400",
    iconBg: "bg-[#1A1446]/8 dark:bg-indigo-500/15",
    hoverGlow:
      "hover:ring-2 hover:ring-[#29254F]/20 dark:hover:ring-indigo-400/25 hover:shadow-[0_4px_20px_-4px_rgba(41,37,79,0.12)]",
  },
  warning: {
    icon: <TriangleAlert className="h-4 w-4" strokeWidth={1.75} />,
    label: "Warning",
    border: "border-l-[#FFD000] dark:border-l-[#FFDB50]",
    bg: "bg-gradient-to-br from-[#FFD000]/6 via-transparent to-[#FFDB50]/5",
    titleColor: "text-amber-800 dark:text-[#FFD000]",
    accentColor: "text-[#FFD000] dark:text-[#FFDB50]",
    iconBg: "bg-[#FFD000]/12 dark:bg-[#FFD000]/18",
    hoverGlow:
      "hover:ring-2 hover:ring-[#FFD000]/30 dark:hover:ring-[#FFD000]/25 hover:shadow-[0_4px_20px_-4px_rgba(255,208,0,0.18)]",
  },
  critical: {
    icon: <OctagonAlert className="h-4 w-4" strokeWidth={1.75} />,
    label: "Critical",
    border: "border-l-red-500 dark:border-l-red-400",
    bg: "bg-gradient-to-br from-red-500/5 via-transparent to-rose-500/5",
    titleColor: "text-red-700 dark:text-red-300",
    accentColor: "text-red-500 dark:text-red-400",
    iconBg: "bg-red-500/10 dark:bg-red-400/12",
    hoverGlow:
      "hover:ring-2 hover:ring-red-400/30 dark:hover:ring-red-500/25 hover:shadow-[0_4px_20px_-4px_rgba(239,68,68,0.15)]",
  },
  urgent: {
    icon: <Siren className="h-4 w-4" strokeWidth={1.75} />,
    label: "Urgent",
    border: "border-l-rose-600 dark:border-l-rose-400",
    bg: "bg-gradient-to-br from-rose-600/6 via-transparent to-red-500/6",
    titleColor: "text-rose-700 dark:text-rose-300",
    accentColor: "text-rose-600 dark:text-rose-400",
    iconBg: "bg-rose-500/10 dark:bg-rose-400/12",
    hoverGlow:
      "hover:ring-2 hover:ring-rose-400/30 dark:hover:ring-rose-500/25 hover:shadow-[0_4px_20px_-4px_rgba(225,29,72,0.15)]",
  },
};

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------

/**
 * Decode HTML entities that LLMs sometimes emit in generated content.
 * Handles named entities (&amp; &lt; &gt; &quot; &apos; &nbsp;),
 * decimal numeric entities (&#38;), and hex entities (&#x26;).
 */
function decodeHtmlEntities(text: string): string {
  return text
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) =>
      String.fromCodePoint(parseInt(hex, 16)),
    )
    .replace(/&#(\d+);/g, (_, dec) =>
      String.fromCodePoint(parseInt(dec, 10)),
    )
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, "\u00A0");
}

/** Minimal inline markdown renderer for bold and line breaks. */
function renderMarkdown(text: string): React.ReactNode {
  const decoded = decodeHtmlEntities(text);
  const parts = decoded.split(/(\*\*[^*]+\*\*)/g);
  const nodes: React.ReactNode[] = [];

  parts.forEach((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      nodes.push(
        <strong key={`b-${i}`} className="font-semibold text-foreground">
          {part.slice(2, -2)}
        </strong>,
      );
      return;
    }
    const lines = part.split("\n");
    lines.forEach((line, j) => {
      if (j > 0) nodes.push(<br key={`br-${i}-${j}`} />);
      if (line) nodes.push(<React.Fragment key={`t-${i}-${j}`}>{line}</React.Fragment>);
    });
  });

  return nodes;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function FindingCard({
  title,
  content,
  severity = "info",
  category,
}: FindingCardProps): React.ReactElement {
  const config = SEVERITY_CONFIG[severity] ?? SEVERITY_CONFIG.info;
  const catCfg = category
    ? (CATEGORY_CONFIG[category] ?? CATEGORY_CONFIG.general)
    : null;

  const [copied, setCopied] = useState(false);

  const contentAsText = useMemo(() => {
    const severityLabel = config.label;
    const categoryLabel = catCfg?.label ?? "";
    const header = `[${severityLabel}]${categoryLabel ? ` (${categoryLabel})` : ""} ${title}`;
    return `${header}\n${content}`;
  }, [config.label, catCfg, title, content]);

  const handleCopyFull = useCallback(async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(contentAsText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard access may be denied in some contexts */
    }
  }, [contentAsText]);

  return (
    <TooltipProvider delayDuration={300}>
      <Card
        className={`overflow-hidden border-l-4 border-border/60 shadow-sm ring-0 ring-transparent transition-all duration-200 ease-out hover:-translate-y-0.5 hover:shadow-lg ${config.border} ${config.bg} ${config.hoverGlow}`}
      >
        {/* ── Header ──────────────────────────────────────────────────────── */}
        <CardHeader className="border-b border-border/40 px-5 py-3">
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2.5">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className={`flex h-7 w-7 shrink-0 cursor-default items-center justify-center rounded-lg transition-colors duration-150 hover:brightness-110 ${config.iconBg}`}
                  >
                    <span className={config.accentColor}>{config.icon}</span>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="top">
                  <span>{config.label}</span>
                </TooltipContent>
              </Tooltip>
              <div className="flex min-w-0 flex-col">
                <h3
                  className={`text-sm font-semibold leading-snug tracking-tight ${config.titleColor}`}
                >
                  {title}
                </h3>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              {catCfg && (
                <span
                  className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset transition-opacity duration-150 hover:opacity-80 ${catCfg.badgeBg}`}
                >
                  {catCfg.icon}
                  {catCfg.label}
                </span>
              )}
              <button
                type="button"
                onClick={handleCopyFull}
                aria-label={copied ? "Copied" : "Copy finding"}
                className="rounded-md p-1.5 text-muted-foreground transition-colors duration-150 hover:bg-accent hover:text-foreground"
              >
                {copied ? (
                  <Check className="h-3.5 w-3.5 text-emerald-500" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </button>
            </div>
          </div>
        </CardHeader>

        {/* ── Body ────────────────────────────────────────────────────────── */}
        <CardContent className="px-5 py-4">
          <p className="text-sm leading-relaxed text-foreground/80">
            {renderMarkdown(content)}
          </p>
        </CardContent>
      </Card>
    </TooltipProvider>
  );
}

export default FindingCard;
