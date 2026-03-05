"use client";

/**
 * DocumentsPane — center pane showing uploaded and agent-provided documents.
 *
 * Merges dummy documents, locally uploaded docs, and agent state documents
 * into a single unified list. Supports:
 *  - Select / Deselect All helpers
 *  - Sortable list (by rank, date, title, selected-first)
 *  - Summarize button with optional ranking instructions input
 *  - Async SSE-driven summarization that streams per-document results
 */

import React, { useState, useMemo, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuditAgent } from "@/hooks/useAuditAgent";
import { useUploadedDocs, type UploadedDoc } from "@/hooks/use-uploaded-docs";
import {
  DocumentCard,
  type DocumentSummaryData,
} from "@/components/A2UI/Documents";
import {
  FileUp,
  Sparkles,
  CheckSquare,
  Square,
  ArrowUpDown,
  Send,
  Loader2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

type SortKey = "selected" | "rank" | "date" | "title";

const DUMMY_DOCUMENTS: UploadedDoc[] = [
  {
    file_name: "Corporate_Security_Policy_v3.2.pdf",
    claim_number: "CLM-2026-00100",
    content_id: "doc-001",
    mime_type: "application/pdf",
    content_url: "/docs/Corporate_Security_Policy_v3.2.pdf",
    domain: "claim",
    document_type: "Policy",
    document_sub_type: "Information Security",
    document_description:
      "Comprehensive information security policy covering access control, data protection, incident response, and vendor management.",
    create_date: "2026-02-15T10:30:00Z",
    source_system: "UPLOAD",
    company_name: "Acme Corp",
  },
  {
    file_name: "SOC_2_Type_II_Report_2025.pdf",
    claim_number: "CLM-2026-00100",
    content_id: "doc-002",
    mime_type: "application/pdf",
    content_url: "/docs/SOC_2_Type_II_Report_2025.pdf",
    domain: "claim",
    document_type: "Report",
    document_sub_type: "SOC 2 Audit",
    document_description:
      "Annual SOC 2 Type II examination report covering security, availability, and confidentiality trust service criteria.",
    create_date: "2026-01-20T14:15:00Z",
    source_system: "UPLOAD",
    company_name: "Acme Corp",
  },
  {
    file_name: "IT_Risk_Assessment_Template.xlsx",
    claim_number: "CLM-2026-00100",
    content_id: "doc-003",
    mime_type:
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    content_url: "/docs/IT_Risk_Assessment_Template.xlsx",
    domain: "claim",
    document_type: "Template",
    document_sub_type: "Risk Assessment",
    document_description:
      "Spreadsheet template for conducting IT risk assessments with impact and likelihood scoring matrices.",
    create_date: "2026-02-10T09:00:00Z",
    source_system: "UPLOAD",
  },
  {
    file_name: "Access_Control_Procedures.docx",
    claim_number: "CLM-2026-00100",
    content_id: "doc-004",
    mime_type:
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    content_url: "/docs/Access_Control_Procedures.docx",
    domain: "policy",
    document_type: "Procedure",
    document_sub_type: "Access Control",
    document_description:
      "Detailed procedures for user provisioning, access reviews, privileged account management, and deprovisioning.",
    create_date: "2026-02-18T16:45:00Z",
    source_system: "UPLOAD",
  },
];

export function DocumentsPane() {
  const { state } = useAuditAgent();
  const { uploadedDocs } = useUploadedDocs();

  // Selection state — default two dummy docs selected.
  const [selectedIds, setSelectedIds] = useState<Set<string>>(
    new Set(["dummy-0", "dummy-1"])
  );

  // Sort state.
  const [sortKey, setSortKey] = useState<SortKey>("selected");

  // Summarization state.
  const [summaries, setSummaries] = useState<Map<string, DocumentSummaryData>>(
    new Map()
  );
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summarizeProgress, setSummarizeProgress] = useState({ done: 0, total: 0 });
  const [showSummarizeInput, setShowSummarizeInput] = useState(false);
  const [rankingInstructions, setRankingInstructions] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  // ── Build unified document list ──────────────────────────────────────

  const allDocs = useMemo(() => {
    const seenNames = new Set<string>();
    const docs: Array<UploadedDoc & { _id: string }> = [];

    DUMMY_DOCUMENTS.forEach((d, i) => {
      const id = `dummy-${i}`;
      docs.push({ ...d, _id: id });
      seenNames.add(d.file_name);
    });

    uploadedDocs.forEach((d, i) => {
      if (seenNames.has(d.file_name)) return;
      docs.push({ ...d, _id: `upload-${i}` });
      seenNames.add(d.file_name);
    });

    (state.documents || []).forEach((d, i) => {
      const fileName =
        (d.file_name as string) || (d.content_url as string) || "Untitled";
      if (seenNames.has(fileName)) return;
      docs.push({
        file_name: fileName,
        claim_number: (d.claim_number as string) || "",
        content_id: (d.content_id as string) || "",
        mime_type: (d.mime_type as string) || "application/octet-stream",
        content_url: (d.content_url as string) || "",
        domain: (d.domain as "claim" | "policy") || "claim",
        document_type: (d.document_type as string) || undefined,
        document_sub_type: (d.document_sub_type as string) || undefined,
        document_description:
          (d.document_description as string) || undefined,
        create_date: (d.create_date as string) || "",
        source_system: (d.source_system as string) || undefined,
        company_name: (d.company_name as string) || undefined,
        _id: `agent-${i}`,
      });
    });

    return docs;
  }, [state.documents, uploadedDocs]);

  // ── Sorting ──────────────────────────────────────────────────────────

  const sortedDocs = useMemo(() => {
    return [...allDocs].sort((a, b) => {
      switch (sortKey) {
        case "rank": {
          const aRank = summaries.get(a.file_name)?.rank ?? -1;
          const bRank = summaries.get(b.file_name)?.rank ?? -1;
          return bRank - aRank; // highest first
        }
        case "date": {
          const aDate = a.create_date ? new Date(a.create_date).getTime() : 0;
          const bDate = b.create_date ? new Date(b.create_date).getTime() : 0;
          return bDate - aDate; // newest first
        }
        case "title":
          return a.file_name.localeCompare(b.file_name);
        case "selected":
        default: {
          const aSel = selectedIds.has(a._id) ? 0 : 1;
          const bSel = selectedIds.has(b._id) ? 0 : 1;
          return aSel - bSel;
        }
      }
    });
  }, [allDocs, selectedIds, sortKey, summaries]);

  // ── Selection helpers ────────────────────────────────────────────────

  const toggleSelection = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedIds(new Set(allDocs.map((d) => d._id)));
  }, [allDocs]);

  const deselectAll = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  const allSelected = selectedIds.size === allDocs.length && allDocs.length > 0;

  // ── Summarize via NDJSON stream ─────────────────────────────────────

  /** Build the document content lookup from AG-UI state.documents. */
  const contentByName = useMemo(() => {
    const map = new Map<string, Record<string, unknown>>();
    for (const d of state.documents || []) {
      const name =
        (d.file_name as string) || (d.content_url as string) || "";
      if (name) map.set(name, d);
    }
    return map;
  }, [state.documents]);

  const runSummarize = useCallback(async () => {
    // Build payloads for selected docs that have content.
    const payloads: Array<{
      file_name: string;
      content: string;
      mime_type: string;
      document_type: string;
    }> = [];

    for (const doc of allDocs) {
      if (!selectedIds.has(doc._id)) continue;
      const stateDoc = contentByName.get(doc.file_name);
      const content = (stateDoc?.content as string) || (stateDoc?.text as string) || "";
      payloads.push({
        file_name: doc.file_name,
        content,
        mime_type: doc.mime_type,
        document_type: doc.document_type || "",
      });
    }

    if (payloads.length === 0) return;

    setIsSummarizing(true);
    setSummarizeProgress({ done: 0, total: payloads.length });
    setSummaries(new Map());
    setSortKey("rank");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const resp = await fetch(`${BACKEND_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          documents: payloads,
          ranking_instructions: rankingInstructions,
        }),
        signal: controller.signal,
      });

      if (!resp.ok || !resp.body) {
        console.error("[Summarize] Bad response", resp.status);
        setIsSummarizing(false);
        return;
      }

      // Read NDJSON: one JSON object per line.
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Keep the last partial line in the buffer.
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          try {
            const obj = JSON.parse(trimmed) as Record<string, unknown>;
            if (obj.error) {
              // Skipped / errored doc — still count progress.
              setSummarizeProgress((prev) => ({ ...prev, done: prev.done + 1 }));
              continue;
            }
            const summary = obj as unknown as DocumentSummaryData & { file_name: string };
            setSummaries((prev) => {
              const next = new Map(prev);
              next.set(summary.file_name, {
                title: summary.title,
                summary: summary.summary,
                rank: summary.rank,
                rank_type: summary.rank_type,
              });
              return next;
            });
            setSummarizeProgress((prev) => ({ ...prev, done: prev.done + 1 }));
          } catch {
            // Ignore unparsable lines.
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        console.error("[Summarize] Stream error", err);
      }
    } finally {
      setIsSummarizing(false);
      setShowSummarizeInput(false);
      abortRef.current = null;
    }
  }, [allDocs, selectedIds, rankingInstructions, contentByName]);

  const cancelSummarize = useCallback(() => {
    abortRef.current?.abort();
    setIsSummarizing(false);
    setShowSummarizeInput(false);
  }, []);

  // ── Render ───────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 pr-10 border-b border-border/50">
        <FileUp className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-foreground">Documents</h2>

        <Badge className="ml-auto text-[10px] bg-primary/15 text-primary border-primary/30">
          {selectedIds.size} selected
        </Badge>
      </div>

      {/* Toolbar: selection helpers, sort, summarize */}
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-border/30 flex-wrap">
        {/* Select / Deselect All */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={allSelected ? deselectAll : selectAll}
            >
              {allSelected ? (
                <Square className="h-3.5 w-3.5" />
              ) : (
                <CheckSquare className="h-3.5 w-3.5" />
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs">
            {allSelected ? "Deselect All" : "Select All"}
          </TooltipContent>
        </Tooltip>

        {/* Sort dropdown */}
        <div className="flex items-center gap-1">
          <ArrowUpDown className="h-3 w-3 text-muted-foreground" />
          <Select value={sortKey} onValueChange={(v) => setSortKey(v as SortKey)}>
            <SelectTrigger className="h-7 w-[100px] text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="selected">Selected</SelectItem>
              <SelectItem value="rank">Rank</SelectItem>
              <SelectItem value="date">Date</SelectItem>
              <SelectItem value="title">Title</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Summarize button */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={showSummarizeInput ? "secondary" : "outline"}
              size="sm"
              className="ml-auto h-7 text-[11px] gap-1"
              onClick={() => {
                if (isSummarizing) {
                  cancelSummarize();
                } else {
                  setShowSummarizeInput((prev) => !prev);
                }
              }}
              disabled={selectedIds.size === 0 && !isSummarizing}
            >
              {isSummarizing ? (
                <>
                  <Loader2 className="h-3 w-3 animate-spin" />
                  {summarizeProgress.done}/{summarizeProgress.total}
                </>
              ) : (
                <>
                  <Sparkles className="h-3 w-3" />
                  Summarize
                </>
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="text-xs">
            {isSummarizing
              ? "Cancel summarization"
              : "Summarize & rank selected documents"}
          </TooltipContent>
        </Tooltip>
      </div>

      {/* Ranking instructions input (revealed on Summarize click) */}
      <AnimatePresence>
        {showSummarizeInput && !isSummarizing && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-border/30"
          >
            <div className="flex items-center gap-1.5 px-3 py-2">
              <Input
                value={rankingInstructions}
                onChange={(e) => setRankingInstructions(e.target.value)}
                placeholder="Optional ranking instructions..."
                className="h-7 text-xs flex-1"
                onKeyDown={(e) => {
                  if (e.key === "Enter") runSummarize();
                }}
              />
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="default"
                    size="icon"
                    className="h-7 w-7 shrink-0"
                    onClick={runSummarize}
                    disabled={selectedIds.size === 0}
                  >
                    <Send className="h-3 w-3" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="text-xs">
                  Run summarization
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0"
                    onClick={() => setShowSummarizeInput(false)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="text-xs">
                  Cancel
                </TooltipContent>
              </Tooltip>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Document list */}
      <ScrollArea className="flex-1">
        <div className="px-3 py-3 space-y-3">
          <AnimatePresence>
            {sortedDocs.map((doc) => (
              <motion.div
                key={doc._id}
                layout
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.3 }}
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
                  selected={selectedIds.has(doc._id)}
                  onSelectionChange={() => toggleSelection(doc._id)}
                  summaryData={summaries.get(doc.file_name)}
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </ScrollArea>
    </div>
  );
}
