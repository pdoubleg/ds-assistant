"use client";

/**
 * QuestionsAggregationTable — cross-form question analytics.
 *
 * Aggregates all TFR questions across every saved form, showing per-question
 * answer distributions (count + %).  Each question row is collapsible to
 * reveal its sub-questions with driver counts.
 */

import React, { useMemo, useState, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ChevronDown,
  ChevronRight,
  Download,
  Copy,
  Check,
  ListChecks,
} from "lucide-react";
import type {
  SavedForm,
  AggregatedQuestion,
  AggregatedSubQuestion,
} from "@/lib/dashboard-types";

// ── Aggregation logic ────────────────────────────────────────────

function normalizeSubAnswer(answer: unknown): boolean {
  return answer === true;
}

function aggregateQuestions(forms: SavedForm[]): AggregatedQuestion[] {
  // Map: questionId -> aggregated data
  const qMap = new Map<
    string,
    {
      id: string;
      text: string;
      yes: number;
      no: number;
      insufficient: number;
      total: number;
      subMap: Map<
        string,
        { id: string; text: string; driverCount: number; totalAppearances: number }
      >;
    }
  >();

  for (const form of forms) {
    for (const q of form.questions ?? []) {
      let entry = qMap.get(q.id);
      if (!entry) {
        entry = {
          id: q.id,
          text: q.text,
          yes: 0,
          no: 0,
          insufficient: 0,
          total: 0,
          subMap: new Map(),
        };
        qMap.set(q.id, entry);
      }

      entry.total++;
      if (q.answer === "Yes") entry.yes++;
      else if (q.answer === "No") entry.no++;
      else entry.insufficient++;

      // Aggregate sub-questions
      for (const sq of q.sub_questions ?? []) {
        let sqEntry = entry.subMap.get(sq.id);
        if (!sqEntry) {
          sqEntry = {
            id: sq.id,
            text: sq.text,
            driverCount: 0,
            totalAppearances: 0,
          };
          entry.subMap.set(sq.id, sqEntry);
        }
        sqEntry.totalAppearances++;
        if (normalizeSubAnswer(sq.answer)) {
          sqEntry.driverCount++;
        }
      }
    }
  }

  // Convert to sorted array
  const pct = (n: number, total: number) =>
    total > 0 ? Math.round((n / total) * 100) : 0;

  return Array.from(qMap.values())
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))
    .map((entry) => ({
      id: entry.id,
      text: entry.text,
      yesCount: entry.yes,
      noCount: entry.no,
      insufficientCount: entry.insufficient,
      totalCount: entry.total,
      yesPercent: pct(entry.yes, entry.total),
      noPercent: pct(entry.no, entry.total),
      insufficientPercent: pct(entry.insufficient, entry.total),
      subQuestions: Array.from(entry.subMap.values())
        .sort((a, b) =>
          a.id.localeCompare(b.id, undefined, { numeric: true })
        )
        .map(
          (sq): AggregatedSubQuestion => ({
            ...sq,
            driverPercent: pct(sq.driverCount, sq.totalAppearances),
          })
        ),
    }));
}

// ── Props ────────────────────────────────────────────────────────

export interface QuestionsAggregationTableProps {
  forms: SavedForm[];
}

export function QuestionsAggregationTable({
  forms,
}: QuestionsAggregationTableProps) {
  const aggregated = useMemo(() => aggregateQuestions(forms), [forms]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [copyStatus, setCopyStatus] = useState<"idle" | "success">("idle");

  const toggleExpand = useCallback((qId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(qId)) next.delete(qId);
      else next.add(qId);
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    setExpanded(new Set(aggregated.map((q) => q.id)));
  }, [aggregated]);

  const collapseAll = useCallback(() => {
    setExpanded(new Set());
  }, []);

  // ── Export helpers ──────────────────────────────────────────────

  const buildCsv = useCallback(() => {
    const headers = [
      "ID",
      "Question",
      "Yes",
      "No",
      "Insuff.",
      "Total",
      "% Yes",
      "% No",
      "% Insuff.",
    ];
    const subHeaders = [
      "Sub ID",
      "Sub-Question",
      "Drivers",
      "Appearances",
      "% Driver",
    ];

    const escape = (v: string) => {
      const s = v.replace(/"/g, '""');
      return /[",\n]/.test(s) ? `"${s}"` : s;
    };

    const lines: string[] = [
      [...headers, ...subHeaders].map(escape).join(","),
    ];

    for (const q of aggregated) {
      const qRow = [
        q.id,
        q.text,
        String(q.yesCount),
        String(q.noCount),
        String(q.insufficientCount),
        String(q.totalCount),
        `${q.yesPercent}%`,
        `${q.noPercent}%`,
        `${q.insufficientPercent}%`,
      ];

      if (q.subQuestions.length === 0) {
        lines.push([...qRow, "", "", "", "", ""].map(escape).join(","));
      } else {
        for (const sq of q.subQuestions) {
          lines.push(
            [
              ...qRow,
              sq.id,
              sq.text,
              String(sq.driverCount),
              String(sq.totalAppearances),
              `${sq.driverPercent}%`,
            ]
              .map(escape)
              .join(",")
          );
        }
      }
    }

    return lines.join("\n");
  }, [aggregated]);

  const buildTsv = useCallback(() => {
    return buildCsv().replace(/,/g, "\t").replace(/"/g, "");
  }, [buildCsv]);

  const triggerDownload = (
    content: string,
    fileName: string,
    mimeType: string
  ) => {
    const blob = new Blob([content], { type: `${mimeType};charset=utf-8;` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(buildTsv());
      setCopyStatus("success");
      setTimeout(() => setCopyStatus("idle"), 1800);
    } catch {
      /* clipboard may be blocked */
    }
  }, [buildTsv]);

  // ── Render ─────────────────────────────────────────────────────

  return (
    <Card className="border-primary/20 overflow-hidden">
      <CardContent className="p-0">
        {/* Toolbar */}
        <div className="flex items-center justify-between gap-3 border-b border-border/40 bg-secondary/40 px-4 py-3">
          <div className="flex items-center gap-2">
            <ListChecks className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold text-foreground">
              Question Analytics
            </h3>
            <span className="text-xs text-muted-foreground">
              ({aggregated.length} questions across {forms.length} forms)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={expandAll}
            >
              Expand All
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={collapseAll}
            >
              Collapse All
            </Button>
            <div className="w-px h-5 bg-border/60" />
            <Button
              variant="outline"
              size="sm"
              className="h-8 gap-1.5"
              onClick={() =>
                triggerDownload(
                  buildCsv(),
                  "question_analytics.csv",
                  "text/csv"
                )
              }
            >
              <Download className="h-3.5 w-3.5" />
              CSV
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 gap-1.5"
              onClick={() =>
                triggerDownload(
                  buildTsv(),
                  "question_analytics.xls",
                  "application/vnd.ms-excel"
                )
              }
            >
              <Download className="h-3.5 w-3.5" />
              Excel
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 gap-1.5"
              onClick={handleCopy}
            >
              {copyStatus === "success" ? (
                <>
                  <Check className="h-3.5 w-3.5" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  Copy
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Table */}
        <Table>
          <TableHeader>
            <TableRow className="bg-secondary/60">
              <TableHead className="w-8" />
              <TableHead className="w-16">ID</TableHead>
              <TableHead className="min-w-[250px]">Question</TableHead>
              <TableHead className="text-center">Yes</TableHead>
              <TableHead className="text-center">No</TableHead>
              <TableHead className="text-center">Insuff.</TableHead>
              <TableHead className="text-center">Total</TableHead>
              <TableHead className="text-center">% Yes</TableHead>
              <TableHead className="text-center">% No</TableHead>
              <TableHead className="text-center">% Insuff.</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {aggregated.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={10}
                  className="h-24 text-center text-muted-foreground"
                >
                  No questions to display.
                </TableCell>
              </TableRow>
            ) : (
              aggregated.map((q) => {
                const isExpanded = expanded.has(q.id);
                const hasSubs = q.subQuestions.length > 0;

                return (
                  <React.Fragment key={q.id}>
                    {/* Question row */}
                    <TableRow
                      className={`${hasSubs ? "cursor-pointer" : ""} ${isExpanded ? "bg-secondary/30" : ""}`}
                      onClick={() => hasSubs && toggleExpand(q.id)}
                    >
                      <TableCell className="w-8 text-center">
                        {hasSubs &&
                          (isExpanded ? (
                            <ChevronDown className="h-4 w-4 text-muted-foreground mx-auto" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-muted-foreground mx-auto" />
                          ))}
                      </TableCell>
                      <TableCell className="font-mono text-xs font-semibold text-primary">
                        {q.id}
                      </TableCell>
                      <TableCell className="text-sm max-w-[400px]">
                        <span className="line-clamp-2">{q.text}</span>
                      </TableCell>
                      <TableCell className="text-center tabular-nums">
                        <Badge className="bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 text-xs">
                          {q.yesCount}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-center tabular-nums">
                        <Badge className="bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30 text-xs">
                          {q.noCount}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-center tabular-nums">
                        <Badge className="bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30 text-xs">
                          {q.insufficientCount}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-center tabular-nums font-medium">
                        {q.totalCount}
                      </TableCell>
                      <TableCell className="text-center tabular-nums text-emerald-700 dark:text-emerald-400">
                        {q.yesPercent}%
                      </TableCell>
                      <TableCell className="text-center tabular-nums text-red-700 dark:text-red-400">
                        {q.noPercent}%
                      </TableCell>
                      <TableCell className="text-center tabular-nums text-amber-700 dark:text-amber-400">
                        {q.insufficientPercent}%
                      </TableCell>
                    </TableRow>

                    {/* Expanded sub-question rows */}
                    {isExpanded &&
                      q.subQuestions.map((sq) => (
                        <TableRow
                          key={sq.id}
                          className="bg-secondary/15 border-l-2 border-l-primary/20"
                        >
                          <TableCell />
                          <TableCell className="font-mono text-xs text-muted-foreground pl-6">
                            {sq.id}
                          </TableCell>
                          <TableCell
                            className="text-sm text-muted-foreground max-w-[400px]"
                            colSpan={2}
                          >
                            <span className="line-clamp-2">{sq.text}</span>
                          </TableCell>
                          <TableCell className="text-center" colSpan={2}>
                            <Badge className="bg-rose-500/15 text-rose-700 dark:text-rose-400 border-rose-500/30 text-xs">
                              {sq.driverCount} drivers
                            </Badge>
                          </TableCell>
                          <TableCell className="text-center tabular-nums text-sm">
                            {sq.totalAppearances}
                          </TableCell>
                          <TableCell
                            className="text-center tabular-nums text-rose-700 dark:text-rose-400"
                            colSpan={3}
                          >
                            {sq.driverPercent}% flagged
                          </TableCell>
                        </TableRow>
                      ))}
                  </React.Fragment>
                );
              })
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

export default QuestionsAggregationTable;
