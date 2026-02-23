/**
 * DataTable Component
 *
 * Displays structured tabular data with optional sorting.
 * Supports headers, rows, and captions with the corporate audit theme.
 */

import React, { useState } from 'react';
import { Card, CardContent } from "@/components/ui/card";

export interface DataTableProps {
  /** Column headers */
  headers: string[];
  /** Table rows (array of arrays) */
  rows: (string | number)[][];
  /** Optional table caption */
  caption?: string;
  /** Enable column sorting */
  sortable?: boolean;
}

export function DataTable({
  headers,
  rows,
  caption,
  sortable = false,
}: DataTableProps): React.ReactElement {
  const [sortColumn, setSortColumn] = useState<number | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [sortedRows, setSortedRows] = useState(rows);

  const handleSort = (columnIndex: number) => {
    if (!sortable) return;

    const newDirection =
      sortColumn === columnIndex && sortDirection === 'asc' ? 'desc' : 'asc';

    const sorted = [...rows].sort((a, b) => {
      const aVal = a[columnIndex];
      const bVal = b[columnIndex];

      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return newDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }

      const aStr = String(aVal).toLowerCase();
      const bStr = String(bVal).toLowerCase();

      if (aStr < bStr) return newDirection === 'asc' ? -1 : 1;
      if (aStr > bStr) return newDirection === 'asc' ? 1 : -1;
      return 0;
    });

    setSortColumn(columnIndex);
    setSortDirection(newDirection);
    setSortedRows(sorted);
  };

  const displayRows = sortable ? sortedRows : rows;

  return (
    <Card className="border-primary/20 overflow-hidden">
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full">
            {caption && (
              <caption className="p-4 text-sm text-primary/70">{caption}</caption>
            )}
            <thead className="border-b border-primary/20 bg-linear-to-r from-primary/10 to-transparent">
              <tr>
                {headers?.map((header, idx) => (
                  <th
                    key={idx}
                    className={`px-4 py-3 text-left text-sm font-semibold text-foreground ${
                      sortable ? 'cursor-pointer hover:bg-primary/10' : ''
                    }`}
                    onClick={() => handleSort(idx)}
                  >
                    <div className="flex items-center gap-2">
                      {header}
                      {sortable && sortColumn === idx && (
                        <span className="text-xs text-primary">
                          {sortDirection === 'asc' ? '\u2191' : '\u2193'}
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
                    rowIdx % 2 === 0 ? 'bg-secondary/20' : 'bg-transparent'
                  }`}
                >
                  {row.map((cell, cellIdx) => (
                    <td key={cellIdx} className="px-4 py-3 text-sm text-foreground/90">
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
