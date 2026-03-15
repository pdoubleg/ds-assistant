"use client";

/**
 * ClaimTimeline Component
 *
 * Polished vertical timeline showing chronological events in a claim's
 * lifecycle. Events are color-coded by category, display status indicators,
 * prominently feature dates, and show elapsed-time gaps between events.
 */

import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Clock,
  CheckCircle,
  AlertTriangle,
  Search,
  FileText,
  DollarSign,
  Mail,
  CircleDot,
  Copy,
  Check,
  ShieldCheck,
  Handshake,
  XCircle,
  FilePlus,
  RotateCcw,
  Send,
  Inbox,
  Flag,
  Gavel,
  Scale,
  CirclePlay,
  Lock,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TimelineEvent {
  date: string;
  title: string;
  description: string;
  category:
    | "fnol"
    | "inspection"
    | "estimate"
    | "payment"
    | "correspondence"
    | "coverage_update"
    | "settlement"
    | "denial"
    | "supplement"
    | "reopen"
    | "info_request"
    | "info_receipt"
    | "complaint"
    | "demand"
    | "attorney"
    | "other";
  status: "completed" | "pending" | "flagged" | "closed";
}

export interface ClaimTimelineProps {
  title: string;
  events: TimelineEvent[];
}

// ---------------------------------------------------------------------------
// Category icon / color config
// ---------------------------------------------------------------------------

const CATEGORY_CONFIG: Record<
  TimelineEvent["category"],
  { label: string; icon: React.ReactNode; badgeBg: string; isOrigin?: boolean }
> = {
  fnol: {
    label: "FNOL",
    icon: <CirclePlay className="h-3 w-3" />,
    badgeBg:
      "bg-primary/10 text-primary font-semibold ring-primary/30 dark:bg-primary/20 dark:text-primary dark:ring-primary/40",
    isOrigin: true,
  },
  inspection: {
    label: "Inspection",
    icon: <Search className="h-3 w-3" />,
    badgeBg:
      "bg-blue-50 text-blue-700 ring-blue-200/60 dark:bg-blue-950/40 dark:text-blue-300 dark:ring-blue-800/40",
  },
  estimate: {
    label: "Estimate",
    icon: <FileText className="h-3 w-3" />,
    badgeBg:
      "bg-violet-50 text-violet-700 ring-violet-200/60 dark:bg-violet-950/40 dark:text-violet-300 dark:ring-violet-800/40",
  },
  payment: {
    label: "Payment",
    icon: <DollarSign className="h-3 w-3" />,
    badgeBg:
      "bg-emerald-50 text-emerald-700 ring-emerald-200/60 dark:bg-emerald-950/40 dark:text-emerald-300 dark:ring-emerald-800/40",
  },
  correspondence: {
    label: "Correspondence",
    icon: <Mail className="h-3 w-3" />,
    badgeBg:
      "bg-amber-50 text-amber-700 ring-amber-200/60 dark:bg-amber-950/40 dark:text-amber-300 dark:ring-amber-800/40",
  },
  coverage_update: {
    label: "Coverage Update",
    icon: <ShieldCheck className="h-3 w-3" />,
    badgeBg:
      "bg-teal-50 text-teal-700 ring-teal-200/60 dark:bg-teal-950/40 dark:text-teal-300 dark:ring-teal-800/40",
  },
  settlement: {
    label: "Settlement",
    icon: <Handshake className="h-3 w-3" />,
    badgeBg:
      "bg-lime-50 text-lime-700 ring-lime-200/60 dark:bg-lime-950/40 dark:text-lime-300 dark:ring-lime-800/40",
  },
  denial: {
    label: "Denial",
    icon: <XCircle className="h-3 w-3" />,
    badgeBg:
      "bg-red-50 text-red-700 ring-red-200/60 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/40",
  },
  supplement: {
    label: "Supplement",
    icon: <FilePlus className="h-3 w-3" />,
    badgeBg:
      "bg-indigo-50 text-indigo-700 ring-indigo-200/60 dark:bg-indigo-950/40 dark:text-indigo-300 dark:ring-indigo-800/40",
  },
  reopen: {
    label: "Re-open",
    icon: <RotateCcw className="h-3 w-3" />,
    badgeBg:
      "bg-orange-50 text-orange-700 ring-orange-200/60 dark:bg-orange-950/40 dark:text-orange-300 dark:ring-orange-800/40",
  },
  info_request: {
    label: "Info Request",
    icon: <Send className="h-3 w-3" />,
    badgeBg:
      "bg-sky-50 text-sky-700 ring-sky-200/60 dark:bg-sky-950/40 dark:text-sky-300 dark:ring-sky-800/40",
  },
  info_receipt: {
    label: "Info Receipt",
    icon: <Inbox className="h-3 w-3" />,
    badgeBg:
      "bg-cyan-50 text-cyan-700 ring-cyan-200/60 dark:bg-cyan-950/40 dark:text-cyan-300 dark:ring-cyan-800/40",
  },
  complaint: {
    label: "Complaint",
    icon: <Flag className="h-3 w-3" />,
    badgeBg:
      "bg-rose-50 text-rose-700 ring-rose-200/60 dark:bg-rose-950/40 dark:text-rose-300 dark:ring-rose-800/40",
  },
  demand: {
    label: "Demand",
    icon: <Gavel className="h-3 w-3" />,
    badgeBg:
      "bg-purple-50 text-purple-700 ring-purple-200/60 dark:bg-purple-950/40 dark:text-purple-300 dark:ring-purple-800/40",
  },
  attorney: {
    label: "Attorney",
    icon: <Scale className="h-3 w-3" />,
    badgeBg:
      "bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-200/60 dark:bg-fuchsia-950/40 dark:text-fuchsia-300 dark:ring-fuchsia-800/40",
  },
  other: {
    label: "Other",
    icon: <CircleDot className="h-3 w-3" />,
    badgeBg:
      "bg-gray-100 text-gray-600 ring-gray-200/60 dark:bg-gray-800/50 dark:text-gray-400 dark:ring-gray-700/40",
  },
};

// ---------------------------------------------------------------------------
// Status config
// ---------------------------------------------------------------------------

const STATUS_CONFIG: Record<
  TimelineEvent["status"],
  { icon: React.ReactNode; ring: string }
> = {
  completed: {
    icon: <CheckCircle className="h-3.5 w-3.5 text-emerald-500/80 dark:text-emerald-400/80" strokeWidth={1.5} />,
    ring: "ring-emerald-500/80 dark:ring-emerald-400/80",
  },
  pending: {
    icon: <Clock className="h-3.5 w-3.5 text-amber-500/80 dark:text-amber-400/80" strokeWidth={1.5} />,
    ring: "ring-amber-500/80 dark:ring-amber-400/80",
  },
  flagged: {
    icon: <AlertTriangle className="h-3.5 w-3.5 text-orange-500/80 dark:text-orange-400/80" strokeWidth={1.5} />,
    ring: "ring-orange-500/80 dark:ring-orange-400/80",
  },
  closed: {
    icon: <Lock className="h-3.5 w-3.5 text-rose-800/80 dark:text-rose-300/80" strokeWidth={1.5} />,
    ring: "ring-rose-800/80 dark:ring-rose-400/80",
  },
};

// ---------------------------------------------------------------------------
// Date / gap helpers
// ---------------------------------------------------------------------------

interface ParsedDate {
  monthDay: string;
  year: string;
  timestamp: number;
}

const MONTH_ABBR = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/**
 * Parse a date string into display parts and a comparable timestamp.
 * Handles ISO (YYYY-MM-DD) explicitly to avoid timezone drift.
 */
function parseEventDate(raw: string): ParsedDate | null {
  const isoMatch = raw.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (isoMatch) {
    const [, y, m, d] = isoMatch;
    const mi = parseInt(m, 10) - 1;
    return {
      monthDay: `${MONTH_ABBR[mi]} ${parseInt(d, 10)}`,
      year: y,
      timestamp: new Date(parseInt(y), mi, parseInt(d, 10)).getTime(),
    };
  }
  const fallback = new Date(raw);
  if (isNaN(fallback.getTime())) return null;
  return {
    monthDay: fallback.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    }),
    year: fallback.getFullYear().toString(),
    timestamp: fallback.getTime(),
  };
}

/** Human-friendly elapsed-time label. */
function formatGap(days: number): string {
  if (days <= 0) return "";
  if (days === 1) return "1 day";
  if (days < 7) return `${days} days`;
  if (days < 14) return "~1 wk";
  if (days < 30) return `~${Math.round(days / 7)} wks`;
  if (days < 60) return "~1 mo";
  if (days < 365) return `~${Math.round(days / 30)} mos`;
  if (days < 730) return "~1 yr";
  return `~${Math.round(days / 365)} yrs`;
}

/** Padding class that scales with gap size to subtly convey magnitude. */
function gapSizeClass(days: number): string {
  if (days <= 7) return "py-1";
  if (days <= 30) return "py-1.5";
  if (days <= 90) return "py-2";
  return "py-2.5";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ClaimTimeline({
  title,
  events,
}: ClaimTimelineProps): React.ReactElement {
  const [copyStatus, setCopyStatus] = useState<"idle" | "success" | "error">(
    "idle"
  );

  // Pre-parse dates and compute gaps between consecutive events.
  const enriched = useMemo(() => {
    return events.map((evt, idx) => {
      const parsed = parseEventDate(evt.date);
      let gapDays: number | null = null;
      if (idx < events.length - 1) {
        const next = parseEventDate(events[idx + 1].date);
        if (parsed && next) {
          gapDays = Math.abs(
            Math.round((next.timestamp - parsed.timestamp) / 86_400_000)
          );
        }
      }
      return { ...evt, parsed, gapDays };
    });
  }, [events]);

  const timelineAsText = useMemo(() => {
    const cleanedTitle = title.trim() || "Claim Timeline";
    const eventLines = events.map((event) => {
      const categoryLabel =
        CATEGORY_CONFIG[event.category]?.label ?? CATEGORY_CONFIG.other.label;
      return `- ${event.date.trim()} | ${event.title.trim()} (${categoryLabel}, ${event.status})${
        event.description.trim() ? `: ${event.description.trim()}` : ""
      }`;
    });
    return [cleanedTitle, ...eventLines].join("\n");
  }, [events, title]);

  const handleCopyTimeline = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(timelineAsText);
      setCopyStatus("success");
      window.setTimeout(() => setCopyStatus("idle"), 1800);
    } catch {
      const textArea = document.createElement("textarea");
      textArea.value = timelineAsText;
      textArea.setAttribute("readonly", "");
      textArea.style.position = "fixed";
      textArea.style.left = "-9999px";
      document.body.appendChild(textArea);
      textArea.select();
      const didCopy = document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopyStatus(didCopy ? "success" : "error");
      window.setTimeout(() => setCopyStatus("idle"), 1800);
    }
  };

  return (
    <Card className="overflow-hidden border-border/60 shadow-sm">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <CardHeader className="border-b bg-muted/30 px-5 py-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
              <Clock className="h-4 w-4 text-primary" />
            </div>
            <h3 className="text-sm font-semibold tracking-tight text-foreground">
              {title}
            </h3>
          </div>

          <div className="flex items-center gap-2">
            <Badge
              variant="secondary"
              className="rounded-md px-2 py-0.5 text-[11px] font-medium"
            >
              {events.length} event{events.length !== 1 ? "s" : ""}
            </Badge>
            <button
              type="button"
              onClick={handleCopyTimeline}
              disabled={!events.length}
              aria-label={
                copyStatus === "success" ? "Copied" : "Copy timeline events"
              }
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
            >
              {copyStatus === "success" ? (
                <Check className="h-3.5 w-3.5 text-emerald-500" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
        </div>
        {copyStatus === "error" && (
          <span className="mt-1 text-xs text-destructive">
            Copy failed. Please try again.
          </span>
        )}
      </CardHeader>

      {/* ── Timeline body ───────────────────────────────────────────────── */}
      <CardContent className="px-5 py-4">
        <div className="flex flex-col">
          {enriched.map((evt, idx) => {
            const isFirst = idx === 0;
            const isLast = idx === enriched.length - 1;
            const catCfg =
              CATEGORY_CONFIG[evt.category] ?? CATEGORY_CONFIG.other;
            const statusCfg =
              STATUS_CONFIG[evt.status] ?? STATUS_CONFIG.pending;

            return (
              <React.Fragment key={idx}>
                {/* ── Event row ─────────────────────────────────────────── */}
                <div className="flex items-stretch">
                  {/* Date column */}
                  <div className="flex w-18 shrink-0 flex-col items-end justify-start pr-4 pt-2.5">
                    {evt.parsed ? (
                      <>
                        <span className="text-[13px] font-semibold leading-tight text-foreground">
                          {evt.parsed.monthDay}
                        </span>
                        <span className="text-[11px] leading-tight text-muted-foreground">
                          {evt.parsed.year}
                        </span>
                      </>
                    ) : (
                      <span className="text-[11px] leading-tight text-muted-foreground">
                        {evt.date}
                      </span>
                    )}
                  </div>

                  {/* Spine column — dot + line segments */}
                  <div className="relative flex w-9 shrink-0 flex-col items-center">
                    {/* Top line segment (hidden for first event) */}
                    {!isFirst && (
                      <div className="w-px flex-1 bg-border" />
                    )}

                    {/* Timeline node — origin events get a distinct primary dot */}
                    {catCfg.isOrigin ? (
                      <div className="z-10 flex h-8 w-8 items-center justify-center rounded-full bg-primary shadow-md ring-[3px] ring-primary/30">
                        <CirclePlay className="h-4 w-4 text-primary-foreground" strokeWidth={1.5} />
                      </div>
                    ) : (
                      <div
                        className={`z-10 flex h-7 w-7 items-center justify-center rounded-full bg-muted shadow-sm ring-[3px] ${statusCfg.ring}`}
                      >
                        {statusCfg.icon}
                      </div>
                    )}

                    {/* Bottom line segment (hidden for last event when no gap follows) */}
                    {(!isLast || (evt.gapDays != null && evt.gapDays > 0)) && (
                      <div className="w-px flex-1 bg-border" />
                    )}
                  </div>

                  {/* Event card */}
                  <div className="flex-1 pb-1 pl-3 pt-0.5">
                    <div className="rounded-lg border border-border/70 bg-card p-3 shadow-xs transition-colors hover:border-border">
                      <p className="text-sm font-medium leading-snug text-foreground">
                        {evt.title}
                      </p>
                      {evt.description && (
                        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                          {evt.description}
                        </p>
                      )}
                      <div className="mt-2">
                        <span
                          className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${catCfg.badgeBg}`}
                        >
                          {catCfg.icon}
                          {catCfg.label}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ── Gap indicator between consecutive events ──────────── */}
                {evt.gapDays != null && evt.gapDays > 0 && (
                  <div className="flex items-stretch">
                    {/* Date spacer */}
                    <div className="w-18 shrink-0" />

                    {/* Spine spacer with gap pill */}
                    <div
                      className={`flex w-9 shrink-0 flex-col items-center ${gapSizeClass(evt.gapDays)}`}
                    >
                      <div className="w-px flex-1 bg-border/50" />
                      <span className="my-0.5 whitespace-nowrap rounded-full bg-muted/80 px-2 py-px text-[10px] font-medium tabular-nums text-muted-foreground">
                        {formatGap(evt.gapDays)}
                      </span>
                      <div className="w-px flex-1 bg-border/50" />
                    </div>

                    {/* Content spacer */}
                    <div className="flex-1" />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

export default ClaimTimeline;
