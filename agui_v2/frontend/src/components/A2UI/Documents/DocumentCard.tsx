/**
 * DocumentCard Component
 *
 * Displays a claim/policy document with metadata from the Document schema:
 * file_name, mime_type, document_type, domain, description, create_date,
 * source_system, and a selection checkbox. Does NOT display the `text` field.
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
import { FileText, FileSpreadsheet, FileImage, File, Calendar } from "lucide-react";

export interface DocumentCardProps {
  /** Document filename (computed from content_url in the backend model) */
  file_name: string;
  /** MIME type (e.g. 'application/pdf') */
  mime_type: string;
  /** Unique content identifier */
  content_id?: string;
  /** Associated claim number */
  claim_number?: string;
  /** URL where the document can be accessed */
  content_url?: string;
  /** 'claim' or 'policy' */
  domain?: "claim" | "policy";
  /** High-level type classification (e.g. 'Estimate', 'Photo', 'Policy') */
  document_type?: string;
  /** Finer-grained type classification */
  document_sub_type?: string;
  /** Human-readable description */
  document_description?: string;
  /** ISO datetime string of creation */
  create_date?: string;
  /** Originating system name */
  source_system?: string;
  /** Associated company name */
  company_name?: string;
  /** Whether the document is selected for analysis */
  selected?: boolean;
  /** Callback when selection changes */
  onSelectionChange?: (selected: boolean) => void;
}

/** Derive a simple extension string from mime_type or file_name for icon/badge rendering. */
function deriveFileExt(mime_type: string, file_name: string): string {
  const mimeMap: Record<string, string> = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/tiff": "tiff",
  };
  if (mimeMap[mime_type]) return mimeMap[mime_type];
  // Fall back to extension from file name
  const dotIdx = file_name.lastIndexOf(".");
  if (dotIdx >= 0) return file_name.slice(dotIdx + 1).toLowerCase();
  return "file";
}

const FILE_ICONS: Record<string, React.ReactNode> = {
  pdf: <FileText className="h-5 w-5 text-red-500 dark:text-red-400" />,
  docx: <FileText className="h-5 w-5 text-blue-500 dark:text-blue-400" />,
  xlsx: <FileSpreadsheet className="h-5 w-5 text-emerald-500 dark:text-emerald-400" />,
  jpg: <FileImage className="h-5 w-5 text-amber-500 dark:text-amber-400" />,
  png: <FileImage className="h-5 w-5 text-amber-500 dark:text-amber-400" />,
  tiff: <FileImage className="h-5 w-5 text-amber-500 dark:text-amber-400" />,
};

const FILE_TYPE_BADGE: Record<string, string> = {
  pdf: "bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30 font-bold",
  docx: "bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/30 font-bold",
  xlsx: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 font-bold",
  jpg: "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 font-bold",
  png: "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 font-bold",
  tiff: "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 font-bold",
};

const DOMAIN_BADGE: Record<string, string> = {
  claim: "bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/30",
  policy: "bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30",
};

export function DocumentCard({
  file_name,
  mime_type,
  domain = "claim",
  document_type,
  document_sub_type,
  document_description,
  create_date,
  source_system,
  company_name,
  selected = false,
  onSelectionChange,
}: DocumentCardProps): React.ReactElement {
  const ext = deriveFileExt(mime_type, file_name);
  const icon = FILE_ICONS[ext] || <File className="h-5 w-5 text-muted-foreground" />;
  const typeBadgeClass = FILE_TYPE_BADGE[ext] || "bg-muted text-muted-foreground border-border";
  const domainBadgeClass = DOMAIN_BADGE[domain] || DOMAIN_BADGE.claim;

  const formattedDate = create_date
    ? new Date(create_date).toLocaleDateString("en-US", {
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
          <span className="truncate">{file_name}</span>
        </CardTitle>

        <CardDescription className="flex items-center gap-2 flex-wrap">
          <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0 font-mono", typeBadgeClass)}>
            {ext.toUpperCase()}
          </Badge>
          <Badge variant="outline" className={cn("text-[10px] px-1.5 py-0", domainBadgeClass)}>
            {domain}
          </Badge>
          {document_type && (
            <span className="text-xs text-muted-foreground">{document_type}</span>
          )}
          {document_sub_type && (
            <span className="text-xs text-muted-foreground/70">/ {document_sub_type}</span>
          )}
        </CardDescription>

        <CardAction>
          <Checkbox
            checked={selected}
            onCheckedChange={(checked) => onSelectionChange?.(checked === true)}
          />
        </CardAction>
      </CardHeader>

      {(document_description || formattedDate || source_system || company_name) && (
        <CardContent>
          {document_description && (
            <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
              {document_description}
            </p>
          )}
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2 flex-wrap">
              {source_system && (
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                  {source_system}
                </Badge>
              )}
              {company_name && (
                <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                  {company_name}
                </Badge>
              )}
            </div>
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
