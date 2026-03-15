"use client";

/**
 * DashboardMetrics — grid of high-level metric tiles for the audit dashboard.
 *
 * Mirrors the visual style of the existing SummaryCard component but is
 * purpose-built for dashboard-level aggregates computed from all saved forms.
 */

import React, { useMemo } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  FileText,
  Shield,
  AlertCircle,
  Home,
  Wind,
  BarChart3,
  AlertTriangle,
  ListChecks,
} from "lucide-react";
import type { SavedForm } from "@/lib/dashboard-types";

interface MetricTile {
  label: string;
  value: string;
  icon: React.ReactNode;
  color: string;
}

export interface DashboardMetricsProps {
  forms: SavedForm[];
}

function normalizeSubAnswer(answer: unknown): boolean {
  if (typeof answer === "boolean") {
    return answer;
  }

  if (answer === "Yes") {
    return true;
  }

  if (answer === "No" || answer === "Insufficient information") {
    return false;
  }

  return true;
}

export function DashboardMetrics({ forms }: DashboardMetricsProps) {
  const metrics: MetricTile[] = useMemo(() => {
    const total = forms.length;
    if (total === 0) {
      return [
        {
          label: "Total Forms",
          value: "0",
          icon: <FileText className="h-4 w-4" />,
          color: "text-sky-600 dark:text-sky-400",
        },
      ];
    }

    const meetsCount = forms.filter(
      (f) => f.overall_outcome === "Meets"
    ).length;
    const doesNotMeetCount = total - meetsCount;
    const interiorCount = forms.filter(
      (f) => f.peril?.peril === "Interior"
    ).length;
    const exteriorCount = total - interiorCount;

    const totalQuestions = forms.reduce(
      (sum, f) => sum + (f.questions?.length ?? 0),
      0
    );
    const avgQuestions =
      total > 0 ? (totalQuestions / total).toFixed(1) : "0";

    // Count drivers across all forms
    const totalDrivers = forms.reduce((sum, f) => {
      return (
        sum +
        (f.questions ?? []).reduce((qSum, q) => {
          return (
            qSum +
            (q.sub_questions ?? []).filter(
              (sq) => normalizeSubAnswer(sq.answer)
            ).length
          );
        }, 0)
      );
    }, 0);

    const pct = (n: number) =>
      total > 0 ? `${Math.round((n / total) * 100)}%` : "0%";

    return [
      {
        label: "Total Forms",
        value: String(total),
        icon: <FileText className="h-4 w-4" />,
        color: "text-sky-600 dark:text-sky-400",
      },
      {
        label: "Meets Expectations",
        value: `${meetsCount} (${pct(meetsCount)})`,
        icon: <Shield className="h-4 w-4" />,
        color: "text-emerald-600 dark:text-emerald-400",
      },
      {
        label: "Does Not Meet",
        value: `${doesNotMeetCount} (${pct(doesNotMeetCount)})`,
        icon: <AlertCircle className="h-4 w-4" />,
        color: "text-red-600 dark:text-red-400",
      },
      {
        label: "Interior Perils",
        value: String(interiorCount),
        icon: <Home className="h-4 w-4" />,
        color: "text-orange-600 dark:text-orange-400",
      },
      {
        label: "Exterior Perils",
        value: String(exteriorCount),
        icon: <Wind className="h-4 w-4" />,
        color: "text-blue-600 dark:text-blue-400",
      },
      {
        label: "Avg Questions / Form",
        value: avgQuestions,
        icon: <ListChecks className="h-4 w-4" />,
        color: "text-violet-600 dark:text-violet-400",
      },
      {
        label: "Total Drivers",
        value: String(totalDrivers),
        icon: <AlertTriangle className="h-4 w-4" />,
        color: "text-rose-600 dark:text-rose-400",
      },
    ];
  }, [forms]);

  return (
    <Card className="border-primary/20 bg-linear-to-br from-primary/5 to-primary/10">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">
            Audit Overview
          </h3>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7">
          {metrics.map((metric, idx) => (
            <div
              key={idx}
              className="rounded-lg border bg-card p-3 shadow-xs"
            >
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <span className={metric.color}>{metric.icon}</span>
                <span className="text-xs font-medium truncate">
                  {metric.label}
                </span>
              </div>
              <div className="mt-1">
                <span className="text-lg font-semibold text-foreground truncate">
                  {metric.value}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default DashboardMetrics;
