"use client";

import type {
  BulkExpandedCommand,
  DocumentCardUiState,
  DocumentSummaryData,
  DocumentTagData,
  DocSearchData,
  CardVariant,
} from "@/components/a2ui/documents";
import type { UploadedDoc } from "@/hooks/use-uploaded-docs";

/**
 * Unified document shape used throughout the documents pane.
 */
export type DocWithId = UploadedDoc & { _id: string };

export type SortKey = "default" | "score" | "date" | "title";
export type AutoTagMode = "default" | "custom";
export type TagFilterMode = "or" | "and";
export type HiddenSortKey =
  | "file_name"
  | "ext"
  | "domain"
  | "document_type"
  | "document_sub_type"
  | "create_date"
  | "source_system";

/**
 * Active filter state for the documents pane.
 */
export interface Filters {
  search: string;
  mimeTypes: Set<string>;
  docTypes: Set<string>;
  subTypes: Set<string>;
  domains: Set<string>;
  tags: Set<string>;
}

/**
 * Available option lists shown in the filter UI.
 */
export interface FilterOptions {
  mimeTypes: string[];
  docTypes: string[];
  subTypes: string[];
  domains: string[];
}

/**
 * Badge metadata used in the active filters row.
 */
export interface FilterChip {
  key: keyof Filters;
  value: string;
  label: string;
}

/**
 * Summary stats shown in the hidden-documents dock.
 */
export interface HiddenStats {
  topExtensions: Array<[string, number]>;
  latestHiddenAt: string | null;
  hiddenPercent: number;
}

/**
 * Shared enrichment maps passed to extracted child components.
 */
export interface DocumentEnrichmentMaps {
  summaries: Map<string, DocumentSummaryData>;
  searchScores: Map<string, DocSearchData>;
  agentTags: Map<string, DocumentTagData[]>;
  cardUiByFileName: Map<string, DocumentCardUiState>;
  bulkExpandedCommand: BulkExpandedCommand | null;
  cardVariant: CardVariant;
}
