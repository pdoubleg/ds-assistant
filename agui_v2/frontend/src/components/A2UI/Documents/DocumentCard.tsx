/**
 * DocumentCard Component
 *
 * Displays an uploaded document with metadata (file type, size, page count)
 * and a selection checkbox. Uses the shadcn Card's grid-based header layout
 * with CardAction for the checkbox and CardDescription for metadata pills.
 */

import React from 'react';
import { cn } from "@/lib/utils";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { FileText, FileSpreadsheet, File, Calendar } from "lucide-react";

export interface DocumentCardProps {
  /** Document filename or title */
  title: string;
  /** File extension without dot (pdf, docx, xlsx) */
  file_type?: string;
  /** Human-readable file size (e.g., '2.4 MB') */
  file_size?: string;
  /** Number of pages */
  page_count?: number | null;
  /** ISO date string of upload */
  upload_date?: string;
  /** Brief document summary */
  summary?: string;
  /** Whether the document is selected for analysis */
  selected?: boolean;
  /** Category tags */
  tags?: string[];
  /** Callback when selection changes */
  onSelectionChange?: (selected: boolean) => void;
}

const FILE_ICONS: Record<string, React.ReactNode> = {
  pdf: <FileText className="h-5 w-5 text-red-500 dark:text-red-400" />,
  docx: <FileText className="h-5 w-5 text-blue-500 dark:text-blue-400" />,
  xlsx: <FileSpreadsheet className="h-5 w-5 text-emerald-500 dark:text-emerald-400" />,
};

const FILE_TYPE_BADGE: Record<string, string> = {
  pdf: "bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30 font-bold",
  docx: "bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/30 font-bold",
  xlsx: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 font-bold",
};

export function DocumentCard({
  title,
  file_type = "pdf",
  file_size,
  page_count,
  upload_date,
  summary,
  selected = false,
  tags = [],
  onSelectionChange,
}: DocumentCardProps): React.ReactElement {
  const icon = FILE_ICONS[file_type] || <File className="h-5 w-5 text-muted-foreground" />;
  const typeBadgeClass = FILE_TYPE_BADGE[file_type] || "bg-muted text-muted-foreground border-border";

  const formattedDate = upload_date
    ? new Date(upload_date).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : null;

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all duration-200 border-l-[5px]",
        selected
          ? "ring-2 ring-accent bg-accent/15 border-l-accent shadow-lg shadow-accent/15 scale-[1.01]"
          : "border-l-transparent hover:border-l-border/60 opacity-75 hover:opacity-100"
      )}
    >
      <CardHeader>
        <CardTitle className="flex items-center gap-2.5 text-sm">
          <span className="shrink-0">{icon}</span>
          <span className="truncate">{title}</span>
        </CardTitle>

        <CardDescription className="flex items-center gap-2">
          <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0 font-mono", typeBadgeClass)}>
            {file_type.toUpperCase()}
          </Badge>
          {file_size && <span className="text-xs">{file_size}</span>}
          {page_count != null && (
            <span className="text-xs">{page_count} pages</span>
          )}
        </CardDescription>

        <CardAction>
          <Checkbox
            checked={selected}
            onCheckedChange={(checked) => onSelectionChange?.(checked === true)}
          />
        </CardAction>
      </CardHeader>

      {(summary || tags.length > 0 || formattedDate) && (
        <CardContent>
          {summary && (
            <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
              {summary}
            </p>
          )}
          <div className="flex items-center justify-between gap-2">
            {tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-[10px] px-1.5 py-0">
                    {tag}
                  </Badge>
                ))}
              </div>
            )}
            {formattedDate && (
              <span className="flex items-center gap-1 text-[10px] text-muted-foreground shrink-0">
                <Calendar className="h-3 w-3" />
                {formattedDate}
              </span>
            )}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

export default DocumentCard;
