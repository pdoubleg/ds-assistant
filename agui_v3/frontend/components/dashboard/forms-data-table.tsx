"use client";

/**
 * FormsDataTable — sortable, filterable table of saved audit forms.
 *
 * Each row represents one saved form with key fields from TFRAnalysisResult.
 * Supports column sorting, search, peril/outcome filters, row selection
 * to open the form viewer, and CSV/Excel/clipboard export.
 */

import React, { useMemo, useState, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search,
  Download,
  Copy,
  Check,
  Eye,
} from "lucide-react";
import type { SavedForm, FormRowStats } from "@/lib/dashboard-types";

// ── Helpers ──────────────────────────────────────────────────────

/** Derive per-form stats from a full SavedForm record. */
function computeRowStats(form: SavedForm): FormRowStats {
  const questions = form.questions ?? [];
  return {
    id: form.id,
    title: form.title || "Untitled",
    peril: form.peril?.peril ?? "Interior",
    overall_outcome: form.overall_outcome,
    questionCount: questions.length,
    yesCount: questions.filter((q) => q.answer === "Yes").length,
    noCount: questions.filter((q) => q.answer === "No").length,
    insufficientCount: questions.filter(
      (q) => q.answer === "Insufficient information"
    ).length,
    driverCount: questions.reduce(
      (sum, q) =>
        sum +
        (q.sub_questions ?? []).filter((s) => (s.answer || "No") === "No")
          .length,
      0
    ),
    created_at: form.created_at,
    updated_at: form.updated_at,
  };
}

function formatDate(iso: string): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

// ── Sort key type ────────────────────────────────────────────────

type SortKey = keyof FormRowStats;
type SortDir = "asc" | "desc";

// ── Column definitions ──────────────────────────────────────────

interface ColumnDef {
  key: SortKey;
  label: string;
  className?: string;
}

const COLUMNS: ColumnDef[] = [
  { key: "title", label: "Title", className: "min-w-[180px]" },
  { key: "peril", label: "Peril" },
  { key: "overall_outcome", label: "Outcome" },
  { key: "questionCount", label: "Questions" },
  { key: "yesCount", label: "Yes" },
  { key: "noCount", label: "No" },
  { key: "insufficientCount", label: "Insuff." },
  { key: "driverCount", label: "Drivers" },
  { key: "created_at", label: "Created" },
  { key: "updated_at", label: "Updated" },
];

// ── Props ────────────────────────────────────────────────────────

export interface FormsDataTableProps {
  forms: SavedForm[];
  onSelectForm: (form: SavedForm) => void;
  selectedFormId?: string | null;
}

export function FormsDataTable({
  forms,
  onSelectForm,
  selectedFormId,
}: FormsDataTableProps) {
  const [search, setSearch] = useState("");
  const [perilFilter, setPerilFilter] = useState<string>("all");
  const [outcomeFilter, setOutcomeFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("updated_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [copyStatus, setCopyStatus] = useState<"idle" | "success">("idle");

  // Pre-compute row stats for every form
  const allRows = useMemo(() => forms.map(computeRowStats), [forms]);

  // Apply filters
  const filteredRows = useMemo(() => {
    let rows = allRows;

    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter((r) => r.title.toLowerCase().includes(q));
    }
    if (perilFilter !== "all") {
      rows = rows.filter((r) => r.peril === perilFilter);
    }
    if (outcomeFilter !== "all") {
      rows = rows.filter((r) => r.overall_outcome === outcomeFilter);
    }

    return rows;
  }, [allRows, search, perilFilter, outcomeFilter]);

  // Apply sort
  const sortedRows = useMemo(() => {
    const sorted = [...filteredRows];
    sorted.sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];

      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDir === "asc" ? aVal - bVal : bVal - aVal;
      }
      const aStr = String(aVal ?? "").toLowerCase();
      const bStr = String(bVal ?? "").toLowerCase();
      if (aStr < bStr) return sortDir === "asc" ? -1 : 1;
      if (aStr > bStr) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return sorted;
  }, [filteredRows, sortKey, sortDir]);

  const handleSort = useCallback(
    (key: SortKey) => {
      if (sortKey === key) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setSortKey(key);
        setSortDir("asc");
      }
    },
    [sortKey]
  );

  // ── Export helpers ──────────────────────────────────────────────

  const buildCsv = useCallback(() => {
    const headers = COLUMNS.map((c) => c.label);
    const csvRows = sortedRows.map((r) =>
      COLUMNS.map((c) => {
        const val = r[c.key];
        if (c.key === "created_at" || c.key === "updated_at")
          return formatDate(String(val));
        return String(val ?? "");
      })
    );

    const escape = (v: string) => {
      const s = v.replace(/"/g, '""');
      return /[",\n]/.test(s) ? `"${s}"` : s;
    };

    return [headers.map(escape).join(",")]
      .concat(csvRows.map((row) => row.map(escape).join(",")))
      .join("\n");
  }, [sortedRows]);

  const buildTsv = useCallback(() => {
    const headers = COLUMNS.map((c) => c.label);
    const tsvRows = sortedRows.map((r) =>
      COLUMNS.map((c) => {
        const val = r[c.key];
        if (c.key === "created_at" || c.key === "updated_at")
          return formatDate(String(val));
        return String(val ?? "");
      })
    );

    return [headers.join("\t")]
      .concat(tsvRows.map((row) => row.join("\t")))
      .join("\n");
  }, [sortedRows]);

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
        <div className="flex flex-wrap items-center gap-3 border-b border-border/40 bg-secondary/40 px-4 py-3">
          <div className="relative flex-1 min-w-[200px] max-w-sm">
            <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search forms..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9 h-8"
            />
          </div>

          <Select value={perilFilter} onValueChange={setPerilFilter}>
            <SelectTrigger size="sm" className="w-[130px]">
              <SelectValue placeholder="Peril" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Perils</SelectItem>
              <SelectItem value="Interior">Interior</SelectItem>
              <SelectItem value="Exterior">Exterior</SelectItem>
            </SelectContent>
          </Select>

          <Select value={outcomeFilter} onValueChange={setOutcomeFilter}>
            <SelectTrigger size="sm" className="w-[180px]">
              <SelectValue placeholder="Outcome" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Outcomes</SelectItem>
              <SelectItem value="Meets">Meets</SelectItem>
              <SelectItem value="Does Not Meet Expectations">
                Does Not Meet
              </SelectItem>
            </SelectContent>
          </Select>

          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 gap-1.5"
              onClick={() =>
                triggerDownload(buildCsv(), "audit_forms.csv", "text/csv")
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
                  "audit_forms.xls",
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
              <TableHead className="w-10" />
              {COLUMNS.map((col) => (
                <TableHead
                  key={col.key}
                  className={`cursor-pointer select-none ${col.className ?? ""}`}
                  onClick={() => handleSort(col.key)}
                >
                  <div className="flex items-center gap-1">
                    {col.label}
                    {sortKey === col.key ? (
                      sortDir === "asc" ? (
                        <ArrowUp className="h-3 w-3" />
                      ) : (
                        <ArrowDown className="h-3 w-3" />
                      )
                    ) : (
                      <ArrowUpDown className="h-3 w-3 opacity-30" />
                    )}
                  </div>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedRows.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={COLUMNS.length + 1}
                  className="h-24 text-center text-muted-foreground"
                >
                  No forms found.
                </TableCell>
              </TableRow>
            ) : (
              sortedRows.map((row) => {
                const isSelected = selectedFormId === row.id;
                const srcForm = forms.find((f) => f.id === row.id);

                return (
                  <TableRow
                    key={row.id}
                    data-state={isSelected ? "selected" : undefined}
                    className="cursor-pointer"
                    onClick={() => srcForm && onSelectForm(srcForm)}
                  >
                    {/* View icon */}
                    <TableCell className="w-10 text-center">
                      <Eye className="h-4 w-4 text-muted-foreground/60 mx-auto" />
                    </TableCell>

                    {/* Title */}
                    <TableCell className="font-medium max-w-[220px] truncate">
                      {row.title}
                    </TableCell>

                    {/* Peril pill */}
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={
                          row.peril === "Exterior"
                            ? "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/30"
                            : "bg-orange-500/15 text-orange-700 dark:text-orange-400 border-orange-500/30"
                        }
                      >
                        {row.peril}
                      </Badge>
                    </TableCell>

                    {/* Outcome pill */}
                    <TableCell>
                      <Badge
                        className={
                          row.overall_outcome === "Meets"
                            ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30"
                            : "bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30"
                        }
                      >
                        {row.overall_outcome === "Meets"
                          ? "Meets"
                          : "Does Not Meet"}
                      </Badge>
                    </TableCell>

                    {/* Numeric columns */}
                    <TableCell className="text-center tabular-nums">
                      {row.questionCount}
                    </TableCell>
                    <TableCell className="text-center tabular-nums text-emerald-700 dark:text-emerald-400">
                      {row.yesCount}
                    </TableCell>
                    <TableCell className="text-center tabular-nums text-red-700 dark:text-red-400">
                      {row.noCount}
                    </TableCell>
                    <TableCell className="text-center tabular-nums text-amber-700 dark:text-amber-400">
                      {row.insufficientCount}
                    </TableCell>
                    <TableCell className="text-center tabular-nums text-rose-700 dark:text-rose-400">
                      {row.driverCount}
                    </TableCell>

                    {/* Dates */}
                    <TableCell className="text-xs text-muted-foreground">
                      {formatDate(row.created_at)}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {formatDate(row.updated_at)}
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>

        {/* Footer with count */}
        <div className="border-t border-border/40 bg-secondary/30 px-4 py-2 text-xs text-muted-foreground">
          {sortedRows.length} of {allRows.length} form
          {allRows.length !== 1 ? "s" : ""}
        </div>
      </CardContent>
    </Card>
  );
}

export default FormsDataTable;
