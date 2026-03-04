"use client";

/**
 * SummaryCard Component
 *
 * Responsive grid of key-value metric tiles for claim-at-a-glance display.
 * Each tile shows a label, prominent value, and optional trend/icon.
 */

import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  DollarSign,
  Calendar,
  User,
  Shield,
  FileText,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  Home,
  CloudRain,
  Flame,
  Wind,
  Wrench,
  TreePine,
  Copy,
  Check,
} from "lucide-react";

interface SummaryMetric {
  label: string;
  value: string;
  icon?: string | null;
  trend?: "up" | "down" | "stable" | null;
}

export interface SummaryCardProps {
  title: string;
  metrics: SummaryMetric[];
}

const ICON_MAP: Record<string, React.ReactNode> = {
  dollar: <DollarSign className="h-4 w-4" />,
  calendar: <Calendar className="h-4 w-4" />,
  user: <User className="h-4 w-4" />,
  shield: <Shield className="h-4 w-4" />,
  file: <FileText className="h-4 w-4" />,
  alert: <AlertCircle className="h-4 w-4" />,
  home: <Home className="h-4 w-4" />,
  weather: <CloudRain className="h-4 w-4" />,
  fire: <Flame className="h-4 w-4" />,
  wind: <Wind className="h-4 w-4" />,
  repair: <Wrench className="h-4 w-4" />,
  tree: <TreePine className="h-4 w-4" />,
};

const TREND_CONFIG: Record<string, { icon: React.ReactNode; color: string }> = {
  up: {
    icon: <TrendingUp className="h-3.5 w-3.5" />,
    color: "text-emerald-700 dark:text-emerald-400",
  },
  down: {
    icon: <TrendingDown className="h-3.5 w-3.5" />,
    color: "text-red-700 dark:text-red-400",
  },
  stable: {
    icon: <Minus className="h-3.5 w-3.5" />,
    color: "text-muted-foreground",
  },
};

export function SummaryCard({
  title,
  metrics,
}: SummaryCardProps): React.ReactElement {
  const [copyStatus, setCopyStatus] = useState<"idle" | "success" | "error">(
    "idle"
  );

  const metricsAsText = useMemo(() => {
    const cleanedTitle = title.trim() || "Summary Metrics";
    const metricLines = metrics.map(
      (metric) => `- ${metric.label.trim()}: ${metric.value.trim()}`
    );

    return [cleanedTitle, ...metricLines].join("\n");
  }, [metrics, title]);

  /**
   * Copies rendered metrics to clipboard with graceful fallback support.
   */
  const handleCopyMetrics = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(metricsAsText);
      setCopyStatus("success");
      window.setTimeout(() => setCopyStatus("idle"), 1800);
    } catch {
      // Fallback for browsers that block or do not expose navigator.clipboard.
      const textArea = document.createElement("textarea");
      textArea.value = metricsAsText;
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
            <BarChart3 className="h-5 w-5 text-primary" />
            <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          </div>
          <button
            type="button"
            onClick={handleCopyMetrics}
            disabled={!metrics.length}
            aria-label={copyStatus === "success" ? "Copied" : "Copy summary metrics"}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
          >
            {copyStatus === "success" ? (
              <Check className="h-4 w-4 text-emerald-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </button>
        </div>
        {copyStatus === "error" && (
          <span className="text-xs text-destructive">
            Copy failed. Please try again.
          </span>
        )}
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {metrics.map((metric, idx) => {
            const iconNode = metric.icon ? ICON_MAP[metric.icon] : null;
            const trendCfg = metric.trend
              ? TREND_CONFIG[metric.trend]
              : null;

            return (
              <div key={idx} className="rounded-lg border bg-card p-3 shadow-xs">
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  {iconNode && (
                    <span className="text-muted-foreground/70">
                      {iconNode}
                    </span>
                  )}
                  <span className="text-xs font-medium truncate">
                    {metric.label}
                  </span>
                </div>
                <div className="mt-1 flex items-baseline gap-1.5">
                  <span className="text-lg font-semibold text-foreground truncate">
                    {metric.value}
                  </span>
                  {trendCfg && (
                    <span className={trendCfg.color}>{trendCfg.icon}</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

export default SummaryCard;
