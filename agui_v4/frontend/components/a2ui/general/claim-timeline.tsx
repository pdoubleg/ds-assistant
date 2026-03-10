"use client";

/**
 * ClaimTimeline Component
 *
 * Vertical timeline showing chronological events in a claim's lifecycle.
 * Events are color-coded by category and display status indicators.
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
} from "lucide-react";

interface TimelineEvent {
  date: string;
  title: string;
  description: string;
  category:
    | "inspection"
    | "estimate"
    | "payment"
    | "correspondence"
    | "other";
  status: "completed" | "pending" | "flagged";
}

export interface ClaimTimelineProps {
  title: string;
  events: TimelineEvent[];
}

const CATEGORY_CONFIG: Record<
  TimelineEvent["category"],
  { label: string; icon: React.ReactNode; color: string; badgeBg: string }
> = {
  inspection: {
    label: "Inspection",
    icon: <Search className="h-3.5 w-3.5" />,
    color: "text-blue-700 dark:text-blue-400",
    badgeBg:
      "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",
  },
  estimate: {
    label: "Estimate",
    icon: <FileText className="h-3.5 w-3.5" />,
    color: "text-violet-700 dark:text-violet-400",
    badgeBg:
      "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-300",
  },
  payment: {
    label: "Payment",
    icon: <DollarSign className="h-3.5 w-3.5" />,
    color: "text-emerald-700 dark:text-emerald-400",
    badgeBg:
      "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  },
  correspondence: {
    label: "Correspondence",
    icon: <Mail className="h-3.5 w-3.5" />,
    color: "text-amber-700 dark:text-amber-400",
    badgeBg:
      "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  },
  other: {
    label: "Other",
    icon: <CircleDot className="h-3.5 w-3.5" />,
    color: "text-gray-700 dark:text-gray-400",
    badgeBg:
      "bg-gray-100 text-gray-800 dark:bg-gray-800/60 dark:text-gray-300",
  },
};

const STATUS_CONFIG: Record<
  TimelineEvent["status"],
  { icon: React.ReactNode; ringColor: string }
> = {
  completed: {
    icon: <CheckCircle className="h-4 w-4 text-emerald-500" />,
    ringColor: "ring-emerald-500/30",
  },
  pending: {
    icon: <Clock className="h-4 w-4 text-amber-500" />,
    ringColor: "ring-amber-500/30",
  },
  flagged: {
    icon: <AlertTriangle className="h-4 w-4 text-red-500" />,
    ringColor: "ring-red-500/30",
  },
};

export function ClaimTimeline({
  title,
  events,
}: ClaimTimelineProps): React.ReactElement {
  const [copyStatus, setCopyStatus] = useState<"idle" | "success" | "error">(
    "idle"
  );

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

  /**
   * Copies rendered timeline events to clipboard with fallback support.
   */
  const handleCopyTimeline = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(timelineAsText);
      setCopyStatus("success");
      window.setTimeout(() => setCopyStatus("idle"), 1800);
    } catch {
      // Fallback for browsers without clipboard API support.
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
    <Card className="border-primary/20 bg-linear-to-br from-primary/5 to-primary/10">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-primary" />
            <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs font-normal">
              {events.length} event{events.length !== 1 ? "s" : ""}
            </Badge>
            <button
              type="button"
              onClick={handleCopyTimeline}
              disabled={!events.length}
              aria-label={
                copyStatus === "success" ? "Copied" : "Copy timeline events"
              }
              className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
            >
              {copyStatus === "success" ? (
                <Check className="h-4 w-4 text-emerald-500" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>
        {copyStatus === "error" && (
          <span className="text-xs text-destructive">
            Copy failed. Please try again.
          </span>
        )}
      </CardHeader>
      <CardContent>
        <div className="relative ml-3">
          <div className="absolute left-0 top-0 bottom-0 w-px bg-border" />

          <div className="space-y-4">
            {events.map((event, idx) => {
              const catCfg =
                CATEGORY_CONFIG[event.category] ?? CATEGORY_CONFIG.other;
              const statusCfg =
                STATUS_CONFIG[event.status] ?? STATUS_CONFIG.pending;

              return (
                <div key={idx} className="relative pl-7">
                  <div
                    className={`absolute left-0 top-1 -translate-x-1/2 flex items-center justify-center rounded-full bg-background ring-2 ${statusCfg.ringColor} p-0.5`}
                  >
                    {statusCfg.icon}
                  </div>

                  <div className="rounded-lg border bg-card p-3 shadow-xs">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-foreground">
                          {event.title}
                        </p>
                        <p className="mt-0.5 text-xs text-muted-foreground leading-relaxed">
                          {event.description}
                        </p>
                      </div>
                      <span className="shrink-0 text-xs text-muted-foreground whitespace-nowrap">
                        {event.date}
                      </span>
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <span
                        className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${catCfg.badgeBg}`}
                      >
                        {catCfg.icon}
                        {catCfg.label}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default ClaimTimeline;
