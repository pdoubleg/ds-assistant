"use client";

/**
 * DocumentViewerSheet — read-only slide-in panel for previewing document content.
 *
 * Three tabs:
 *   - Preview:  Raw PDF rendered in an iframe (PDFs only).
 *   - Text:     Extracted text content in a monospace scroll area with copy.
 *   - Summary:  AI-generated summary with label (when available).
 *
 * Modeled after FormViewerSheet; uses Sheet (Radix) from the right.
 */

import React, { useState, useCallback } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { GfmMarkdown } from "@/components/a2ui/general/gfm-markdown";
import {
  FileText,
  Copy,
  ClipboardCheck,
  Calendar,
  Sparkles,
  Eye,
  FileCode,
  Expand,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { UploadedDoc } from "@/hooks/use-uploaded-docs";
import {
  deriveFileExt,
  getFileTypeBadgeClass,
  getFileTypeIcon,
  type DocumentSummaryData,
} from "@/components/a2ui/documents";

// ── Helpers ────────────────────────────────────────────────────────────

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

function resolveDocumentUrl(contentUrl: string): string {
  if (!contentUrl) {
    return "";
  }
  if (contentUrl.startsWith("http://") || contentUrl.startsWith("https://")) {
    return contentUrl;
  }
  return `${BACKEND_URL}${contentUrl}`;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getHighlightTerms(query: string): string[] {
  const normalizedQuery = query.trim();
  const wordTokens = normalizedQuery
    .split(/\W+/)
    .map((token) => token.trim())
    .filter(Boolean);

  return Array.from(new Set([normalizedQuery, ...wordTokens].filter(Boolean))).sort(
    (left, right) => right.length - left.length
  );
}

function renderHighlightedPreformattedText(
  text: string,
  query: string
): React.ReactNode {
  const tokens = getHighlightTerms(query);
  if (tokens.length === 0) {
    return text;
  }

  const pattern = new RegExp(`(${tokens.map(escapeRegExp).join("|")})`, "gi");
  const parts = text.split(pattern);

  return parts.map((part, index) => {
    const isMatch = tokens.some((token) => token.toLowerCase() === part.toLowerCase());
    if (!isMatch) {
      return <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>;
    }

    return (
      <mark
        key={`${part}-${index}`}
        className="rounded bg-amber-400/40 px-0.5 text-foreground"
      >
        {part}
      </mark>
    );
  });
}


// ── Copy button ────────────────────────────────────────────────────────

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be blocked */
    }
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className="shrink-0 p-1.5 rounded-md text-muted-foreground/50 hover:text-foreground hover:bg-secondary/60 transition-colors"
      title="Copy to clipboard"
      type="button"
    >
      {copied ? (
        <ClipboardCheck className="h-3.5 w-3.5 text-emerald-500" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </button>
  );
}

// ── Main component ─────────────────────────────────────────────────────

export interface DocumentViewerSheetProps {
  doc: (UploadedDoc & { _id: string }) | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Extracted text content from agent state. */
  textContent?: string;
  /** AI-generated summary data. */
  summaryData?: DocumentSummaryData;
  /** Optional text query to highlight inside the extracted text view. */
  highlightQuery?: string;
  /** Optional page number to auto-scroll the PDF preview to. */
  initialPage?: number;
  /** Extra className applied to SheetContent (e.g. z-index overrides). */
  contentClassName?: string;
  /** Extra className applied to the Sheet's backdrop overlay. */
  overlayClassName?: string;
}

export function DocumentViewerSheet({
  doc,
  open,
  onOpenChange,
  textContent,
  summaryData,
  highlightQuery,
  initialPage,
  contentClassName,
  overlayClassName,
}: DocumentViewerSheetProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen) setIsFullscreen(false);
      onOpenChange(nextOpen);
    },
    [onOpenChange]
  );

  if (!doc) return null;

  const ext = deriveFileExt(doc.mime_type, doc.file_name);
  const icon = getFileTypeIcon(ext, "lg");
  const isPdf = ext === "pdf";
  const isImage = ["jpg", "png", "gif", "bmp", "tiff", "webp"].includes(ext);

  const formattedDate = doc.create_date
    ? new Date(doc.create_date).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      })
    : null;

  // Build the raw PDF URL for the iframe, optionally targeting a page.
  const resolvedDocumentUrl = resolveDocumentUrl(doc.content_url);
  const pdfUrl = isPdf ? resolvedDocumentUrl : null;
  const pageFragment =
    initialPage && initialPage > 1 ? `page=${initialPage}` : "";
  const fullscreenFragment = isFullscreen ? "zoom=100" : "";
  const fragments = [pageFragment, fullscreenFragment]
    .filter(Boolean)
    .join("&");
  const previewPdfUrl =
    pdfUrl && fragments ? `${pdfUrl}#${fragments}` : pdfUrl;

  // Pick the default tab based on what content is available
  const hasPreview = isPdf || isImage;
  const defaultTab = hasPreview ? "preview" : textContent ? "text" : "summary";
  const typeBadgeClass = getFileTypeBadgeClass(ext);

  const contentHeight = isFullscreen
    ? "h-[calc(100vh-52px)]"
    : "h-[calc(100vh-280px)]";

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent
        side="right"
        showCloseButton={!isFullscreen}
        overlayClassName={overlayClassName}
        className={cn(
          "p-0 transition-all duration-200",
          isFullscreen
            ? "w-screen sm:max-w-none"
            : "w-full sm:max-w-2xl lg:max-w-3xl",
          contentClassName
        )}
      >
        <Tabs defaultValue={defaultTab} className="flex-1 flex flex-col min-h-0">
          {isFullscreen ? (
            /* ── Fullscreen: compact header with tabs inline ──────── */
            <div className="flex items-center gap-3 px-4 h-[52px] border-b border-border/40 bg-secondary/20 shrink-0">
              {icon}
              <SheetTitle className="text-sm font-medium truncate max-w-[260px]">
                {doc.file_name}
              </SheetTitle>
              <SheetDescription className="sr-only">
                Document viewer for {doc.file_name}
              </SheetDescription>

              <TabsList className="h-8 ml-2">
                {hasPreview && (
                  <TabsTrigger value="preview" className="text-xs gap-1.5">
                    <Eye className="h-3 w-3" />
                    Preview
                  </TabsTrigger>
                )}
                <TabsTrigger value="text" className="text-xs gap-1.5">
                  <FileCode className="h-3 w-3" />
                  Extracted Text
                </TabsTrigger>
                {summaryData && (
                  <TabsTrigger value="summary" className="text-xs gap-1.5">
                    <Sparkles className="h-3 w-3" />
                    Summary
                  </TabsTrigger>
                )}
              </TabsList>

              <div className="ml-auto flex items-center gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs gap-1.5"
                  onClick={() => setIsFullscreen(false)}
                >
                  <X className="h-3.5 w-3.5" />
                  Close Full Screen
                </Button>
              </div>
            </div>
          ) : (
            /* ── Normal mode: rich header + separate tab bar ─────── */
            <>
              <SheetHeader className="px-6 pt-6 pb-4 border-b border-border/40 bg-secondary/20">
                <div className="flex items-center gap-3 flex-wrap">
                  {icon}
                  <SheetTitle className="text-lg truncate flex-1">
                    {doc.file_name}
                  </SheetTitle>
                </div>

                <SheetDescription className="sr-only">
                  Document viewer for {doc.file_name}
                </SheetDescription>

                {/* Metadata badges */}
                <div className="flex items-center gap-2 flex-wrap mt-1">
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-[10px] px-1.5 py-0 font-mono font-bold",
                      typeBadgeClass
                    )}
                  >
                    {ext.toUpperCase()}
                  </Badge>
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-[10px] px-1.5 py-0",
                      doc.domain === "policy"
                        ? "bg-violet-500/20 text-violet-700 dark:text-violet-400 border-violet-500/30"
                        : "bg-blue-500/20 text-blue-700 dark:text-blue-400 border-blue-500/30"
                    )}
                  >
                    {doc.domain}
                  </Badge>
                  {doc.document_type && (
                    <span className="text-xs text-muted-foreground">
                      {doc.document_type}
                      {doc.document_sub_type && ` / ${doc.document_sub_type}`}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 mt-1 flex-wrap">
                  {doc.source_system && (
                    <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                      {doc.source_system}
                    </Badge>
                  )}
                  {doc.company_name && (
                    <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                      {doc.company_name}
                    </Badge>
                  )}
                  {formattedDate && (
                    <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
                      <Calendar className="h-3.5 w-3.5" />
                      {formattedDate}
                    </span>
                  )}
                </div>

                {doc.document_description && (
                  <p className="text-xs text-muted-foreground/80 mt-1.5 leading-relaxed">
                    {doc.document_description}
                  </p>
                )}
              </SheetHeader>

              {/* Tab bar + Expand button */}
              <div className="flex items-center gap-2 px-6 pt-3 border-b border-border/30">
                <TabsList className="h-8">
                  {hasPreview && (
                    <TabsTrigger value="preview" className="text-xs gap-1.5">
                      <Eye className="h-3 w-3" />
                      Preview
                    </TabsTrigger>
                  )}
                  <TabsTrigger value="text" className="text-xs gap-1.5">
                    <FileCode className="h-3 w-3" />
                    Extracted Text
                  </TabsTrigger>
                  {summaryData && (
                    <TabsTrigger value="summary" className="text-xs gap-1.5">
                      <Sparkles className="h-3 w-3" />
                      Summary
                    </TabsTrigger>
                  )}
                </TabsList>
                <Button
                  variant="ghost"
                  size="sm"
                  className="ml-auto h-7 text-xs gap-1.5 text-muted-foreground"
                  onClick={() => setIsFullscreen(true)}
                >
                  <Expand className="h-3.5 w-3.5" />
                  Expand
                </Button>
              </div>
            </>
          )}

          {/* ── Tab content ──────────────────────────────────────── */}

          {/* Preview tab — PDF iframe or image */}
          {hasPreview && (
            <TabsContent value="preview" className="flex-1 min-h-0">
              {isPdf && pdfUrl ? (
                <iframe
                  key={`pdf-preview-${isFullscreen ? "fullscreen" : "normal"}`}
                  src={previewPdfUrl || undefined}
                  className={cn("w-full border-0", contentHeight)}
                  title={`PDF preview: ${doc.file_name}`}
                />
              ) : isImage && resolvedDocumentUrl ? (
                <div
                  className={cn(
                    "flex items-center justify-center bg-muted/20 px-4 py-4",
                    contentHeight
                  )}
                >
                  <img
                    src={resolvedDocumentUrl}
                    alt={doc.file_name}
                    className="max-h-full max-w-full rounded-md border border-border/40 shadow-sm object-contain"
                  />
                </div>
              ) : (
                <div className="flex items-center justify-center h-40 text-sm text-muted-foreground">
                  Preview not available.
                </div>
              )}
            </TabsContent>
          )}

          {/* Extracted text tab */}
          <TabsContent value="text" className="flex-1 min-h-0">
            {textContent ? (
              <div className="relative">
                <div className="absolute top-2 right-4 z-10">
                  <CopyBtn text={textContent} />
                </div>
                <ScrollArea className={contentHeight}>
                  <pre className="px-6 py-4 text-xs font-mono text-foreground/85 whitespace-pre-wrap leading-relaxed">
                    {renderHighlightedPreformattedText(textContent, highlightQuery || "")}
                  </pre>
                </ScrollArea>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-40 gap-2 text-sm text-muted-foreground">
                <FileText className="h-6 w-6 text-muted-foreground/40" />
                No extracted text available.
              </div>
            )}
          </TabsContent>

          {/* Summary tab */}
          {summaryData && (
            <TabsContent value="summary" className="flex-1 min-h-0">
              <ScrollArea className={contentHeight}>
                <div className="px-6 py-4 space-y-3">
                  <div className="flex items-center gap-1.5">
                    <Sparkles className="h-4 w-4 text-primary" />
                    <span className="text-primary text-sm font-semibold">
                      AI Summary
                    </span>
                    <span className="text-xs text-muted-foreground/70 ml-1">
                      {summaryData.label || "Document Summary"}
                    </span>
                  </div>

                  <h3 className="text-sm font-semibold text-foreground">
                    {summaryData.title}
                  </h3>

                  <GfmMarkdown
                    content={summaryData.summary}
                    className="doc-summary-markdown text-sm text-muted-foreground leading-relaxed"
                  />
                </div>
              </ScrollArea>
            </TabsContent>
          )}
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}

export default DocumentViewerSheet;
