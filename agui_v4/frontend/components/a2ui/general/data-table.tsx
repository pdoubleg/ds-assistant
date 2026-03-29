"use client";

/**
 * DataTable Component
 *
 * Displays structured tabular data with optional sorting.
 * Supports headers, rows, and captions with the corporate audit theme.
 */

import React, { useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  Check,
  Copy,
  Download,
} from "lucide-react";

export interface DataTableProps {
  headers: string[];
  rows: (string | number)[][];
  caption?: string;
  sortable?: boolean;
  copyable?: boolean;
  downloadable?: boolean;
}

export function DataTable({
  headers,
  rows,
  caption,
  sortable = true,
  copyable = true,
  downloadable = true,
}: DataTableProps): React.ReactElement {
  const [sortColumn, setSortColumn] = useState<number | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [copyStatus, setCopyStatus] = useState<"idle" | "success" | "error">(
    "idle"
  );

  const handleSort = (columnIndex: number) => {
    if (!sortable) return;

    const newDirection =
      sortColumn === columnIndex && sortDirection === "asc" ? "desc" : "asc";

    setSortColumn(columnIndex);
    setSortDirection(newDirection);
  };

  const displayRows = useMemo(() => {
    if (!sortable || sortColumn === null) {
      return rows;
    }

    return [...rows].sort((a, b) => {
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];

      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDirection === "asc" ? aVal - bVal : bVal - aVal;
      }

      const aStr = String(aVal ?? "").toLowerCase();
      const bStr = String(bVal ?? "").toLowerCase();

      if (aStr < bStr) return sortDirection === "asc" ? -1 : 1;
      if (aStr > bStr) return sortDirection === "asc" ? 1 : -1;
      return 0;
    });
  }, [rows, sortColumn, sortDirection, sortable]);

  /**
   * Builds a stable, filesystem-safe filename seed from table metadata.
   */
  const baseFileName = useMemo(() => {
    const sourceLabel = caption || headers[0] || "table";
    const cleaned = sourceLabel
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
    return cleaned || "table";
  }, [caption, headers]);

  const tableAsTsv = useMemo(() => {
    const serialize = (value: string | number): string =>
      // Keep output clean for pasted spreadsheets/text by removing line breaks.
      String(value).replace(/\r?\n/g, " ").replace(/\t/g, " ").trim();

    const allRows = [headers, ...displayRows];
    return allRows
      .map((row) => row.map((cell) => serialize(cell)).join("\t"))
      .join("\n");
  }, [displayRows, headers]);

  const tableAsCsv = useMemo(() => {
    const serialize = (value: string | number): string => {
      const normalized = String(value).replace(/\r?\n/g, " ").trim();
      const escaped = normalized.replace(/"/g, '""');
      return /[",]/.test(escaped) ? `"${escaped}"` : escaped;
    };

    const allRows = [headers, ...displayRows];
    return allRows
      .map((row) => row.map((cell) => serialize(cell)).join(","))
      .join("\n");
  }, [displayRows, headers]);

  const triggerDownload = (
    content: string,
    fileName: string,
    mimeType: string
  ): void => {
    const blob = new Blob([content], {
      type: `${mimeType};charset=utf-8;`,
    });
    const objectUrl = URL.createObjectURL(blob);
    const downloadLink = document.createElement("a");
    downloadLink.href = objectUrl;
    downloadLink.download = fileName;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
    URL.revokeObjectURL(objectUrl);
  };

  const handleDownloadCsv = (): void => {
    triggerDownload(tableAsCsv, `${baseFileName}.csv`, "text/csv");
  };

  const handleDownloadExcel = (): void => {
    // TSV + .xls is broadly compatible for local Excel opens.
    triggerDownload(
      tableAsTsv,
      `${baseFileName}.xls`,
      "application/vnd.ms-excel"
    );
  };

  const handleCopyTable = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(tableAsTsv);
      setCopyStatus("success");
      window.setTimeout(() => setCopyStatus("idle"), 1800);
    } catch {
      // Fallback for environments where the modern clipboard API is unavailable.
      const textArea = document.createElement("textarea");
      textArea.value = tableAsTsv;
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
    <Card className="border-primary/20 overflow-hidden">
      <CardContent className="p-0">
        {(caption || copyable || downloadable) && (
          <div className="flex items-center justify-between gap-3 border-b border-border/40 bg-secondary/40 px-3 py-2">
            {caption ? (
              <h3 className="text-base font-extrabold underline text-foreground">
                {caption}
              </h3>
            ) : (
              <div />
            )}
            <div className="flex items-center gap-2">
              {downloadable && (
                <>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    className="h-8 gap-2"
                    onClick={handleDownloadCsv}
                  >
                    <Download className="h-3.5 w-3.5" />
                    CSV
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    className="h-8 gap-2"
                    onClick={handleDownloadExcel}
                  >
                    <Download className="h-3.5 w-3.5" />
                    Excel
                  </Button>
                </>
              )}
              {copyable && (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-8 gap-2"
                  onClick={handleCopyTable}
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
              )}
              {copyStatus === "error" && (
                <span className="ml-3 text-xs text-destructive">
                  Copy failed. Please try again.
                </span>
              )}
            </div>
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="border-b border-border/60 bg-secondary/70">
              <tr>
                {headers?.map((header, idx) => (
                  <th
                    key={idx}
                    className={`px-4 py-3 text-left text-base font-extrabold text-foreground ${
                      sortable ? "cursor-pointer select-none hover:bg-secondary" : ""
                    }`}
                    onClick={() => handleSort(idx)}
                  >
                    <div className="flex items-center gap-2">
                      {header}
                      {sortable ? (
                        sortColumn === idx ? (
                          sortDirection === "asc" ? (
                            <ArrowUp className="h-3 w-3 text-foreground/80" />
                          ) : (
                            <ArrowDown className="h-3 w-3 text-foreground/80" />
                          )
                        ) : (
                          <ArrowUpDown className="h-3 w-3 text-foreground/40" />
                        )
                      ) : null}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayRows?.map((row, rowIdx) => (
                <tr
                  key={rowIdx}
                  className={`border-b border-border/30 last:border-0 hover:bg-accent/5 transition-colors ${
                    rowIdx % 2 === 0 ? "bg-secondary/20" : "bg-transparent"
                  }`}
                >
                  {row.map((cell, cellIdx) => (
                    <td
                      key={cellIdx}
                      className="px-4 py-3 text-sm text-foreground/90"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

export default DataTable;
