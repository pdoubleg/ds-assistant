"use client";

import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { FileUp } from "lucide-react";
import {
  DocumentCard,
  type BulkExpandedCommand,
  type CardVariant,
  type DocumentCardUiState,
  type DocumentSummaryData,
  type DocumentTagData,
  type DocSearchData,
} from "@/components/a2ui/documents";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { DocWithId, Filters } from "./types";

export interface DocumentsGridProps {
  sortedDocs: DocWithId[];
  summaries: Map<string, DocumentSummaryData>;
  searchScores: Map<string, DocSearchData>;
  agentTags: Map<string, DocumentTagData[]>;
  cardVariant: CardVariant;
  chatDocNames: Set<string>;
  filters: Filters;
  gridCols: number;
  showHiddenDock: boolean;
  dockMinimized: boolean;
  dockExpanded: boolean;
  bulkExpandedCommand: BulkExpandedCommand | null;
  cardUiByFileName: Map<string, DocumentCardUiState>;
  hasActiveFilters: (filters: Filters) => boolean;
  onToggleHidden: (fileName: string) => void;
  onToggleChatDoc: (fileName: string) => void;
  onToggleTagFilter: (tag: string) => void;
  onPreviewDoc: (doc: DocWithId) => void;
  onCardUiStateChange: (fileName: string, nextState: DocumentCardUiState) => void;
}

/**
 * Scrollable document grid used for the visible triage list.
 */
export function DocumentsGrid({
  sortedDocs,
  summaries,
  searchScores,
  agentTags,
  cardVariant,
  chatDocNames,
  filters,
  gridCols,
  showHiddenDock,
  dockMinimized,
  dockExpanded,
  bulkExpandedCommand,
  cardUiByFileName,
  hasActiveFilters,
  onToggleHidden,
  onToggleChatDoc,
  onToggleTagFilter,
  onPreviewDoc,
  onCardUiStateChange,
}: DocumentsGridProps) {
  return (
    <ScrollArea className="flex-1 relative z-0">
      <div
        className={cn(
          "px-2 py-2 grid gap-1.5",
          gridCols === 4
            ? "grid-cols-4"
            : gridCols === 2
              ? "grid-cols-2"
              : "grid-cols-1",
          showHiddenDock &&
            (dockMinimized
              ? "pb-[52px]"
              : dockExpanded
                ? "pb-[360px]"
                : "pb-[88px]")
        )}
      >
        <AnimatePresence>
          {sortedDocs.map((doc) => (
            <motion.div
              key={doc._id}
              layout
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.2 }}
            >
              <DocumentCard
                file_name={doc.file_name}
                mime_type={doc.mime_type}
                content_id={doc.content_id}
                claim_number={doc.claim_number}
                content_url={doc.content_url}
                domain={doc.domain}
                document_type={doc.document_type}
                document_sub_type={doc.document_sub_type}
                document_description={doc.document_description}
                create_date={doc.create_date}
                source_system={doc.source_system}
                company_name={doc.company_name}
                summaryData={summaries.get(doc.file_name)}
                searchData={searchScores.get(doc.file_name)}
                tags={agentTags.get(doc.file_name)}
                variant={cardVariant}
                isHidden={false}
                onToggleHidden={() => onToggleHidden(doc.file_name)}
                isInChatContext={chatDocNames.has(doc.file_name)}
                onToggleChatContext={() => onToggleChatDoc(doc.file_name)}
                onTagClick={onToggleTagFilter}
                activeTagFilters={filters.tags}
                onPreview={() => onPreviewDoc(doc)}
                initialUiState={cardUiByFileName.get(doc.file_name)}
                bulkExpandedCommand={bulkExpandedCommand ?? undefined}
                onUiStateChange={(nextState) =>
                  onCardUiStateChange(doc.file_name, nextState)
                }
              />
            </motion.div>
          ))}
        </AnimatePresence>

        {sortedDocs.length === 0 && (
          <div className="col-span-full flex flex-col items-center justify-center py-12 text-muted-foreground/60 gap-2">
            <FileUp className="h-8 w-8" />
            <p className="text-sm">
              {hasActiveFilters(filters)
                ? "No documents match the current filters."
                : "No documents in triage. Upload or load documents to begin."}
            </p>
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
