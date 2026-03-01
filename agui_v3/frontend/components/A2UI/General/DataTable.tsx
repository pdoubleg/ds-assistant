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
import { Check, Copy } from "lucide-react";

export interface DataTableProps {
  headers: string[];
  rows: (string | number)[][];
  caption?: string;
  sortable?: boolean;
  copyable?: boolean;
}

export function DataTable({
  headers,
  rows,
  caption,
  sortable = false,
  copyable = true,
}: DataTableProps): React.ReactElement {
  const [sortColumn, setSortColumn] = useState<number | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [sortedRows, setSortedRows] = useState(rows);
  const [copyStatus, setCopyStatus] = useState<"idle" | "success" | "error">(
    "idle"
  );

  const handleSort = (columnIndex: number) => {
    if (!sortable) return;

    const newDirection =
      sortColumn === columnIndex && sortDirection === "asc" ? "desc" : "asc";

    const sorted = [...rows].sort((a, b) => {
      const aVal = a[columnIndex];
      const bVal = b[columnIndex];

      if (typeof aVal === "number" && typeof bVal === "number") {
        return newDirection === "asc" ? aVal - bVal : bVal - aVal;
      }

      const aStr = String(aVal).toLowerCase();
      const bStr = String(bVal).toLowerCase();

      if (aStr < bStr) return newDirection === "asc" ? -1 : 1;
      if (aStr > bStr) return newDirection === "asc" ? 1 : -1;
      return 0;
    });

    setSortColumn(columnIndex);
    setSortDirection(newDirection);
    setSortedRows(sorted);
  };

  const displayRows = sortable ? sortedRows : rows;
  const tableAsTsv = useMemo(() => {
    const serialize = (value: string | number): string => {
      // Keep output clean for pasted spreadsheets/text by removing line breaks.
      return String(value).replace(/\r?\n/g, " ").trim();
    };

    const allRows = [headers, ...displayRows];
    return allRows
      .map((row) => row.map((cell) => serialize(cell)).join("\t"))
      .join("\n");
  }, [displayRows, headers]);

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
        {(caption || copyable) && (
          <div className="flex items-center justify-between gap-3 border-b border-border/40 bg-secondary/40 px-3 py-2">
            {caption ? (
              <h3 className="text-base font-extrabold underline text-foreground">
                {caption}
              </h3>
            ) : (
              <div />
            )}
            <div className="flex items-center">
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
                      Copy table
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
                      sortable ? "cursor-pointer hover:bg-secondary" : ""
                    }`}
                    onClick={() => handleSort(idx)}
                  >
                    <div className="flex items-center gap-2">
                      {header}
                      {sortable && sortColumn === idx && (
                        <span className="text-xs text-foreground/80">
                          {sortDirection === "asc" ? "\u2191" : "\u2193"}
                        </span>
                      )}
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
