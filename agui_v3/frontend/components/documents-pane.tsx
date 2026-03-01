"use client";

/**
 * DocumentsPane — the center pane showing uploaded and agent-provided documents.
 *
 * Merges dummy documents, locally uploaded docs, and agent state documents
 * into a single unified list. Selected cards float to the top.
 */

import React, { useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuditAgent } from "@/hooks/useAuditAgent";
import { useUploadedDocs, type UploadedDoc } from "@/hooks/use-uploaded-docs";
import { DocumentCard } from "@/components/A2UI/Documents";
import { FileUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

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
  const [selectedIds, setSelectedIds] = useState<Set<string>>(
    new Set(["dummy-0", "dummy-1"])
  );

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

  const sortedDocs = useMemo(() => {
    return [...allDocs].sort((a, b) => {
      const aSelected = selectedIds.has(a._id) ? 0 : 1;
      const bSelected = selectedIds.has(b._id) ? 0 : 1;
      return aSelected - bSelected;
    });
  }, [allDocs, selectedIds]);

  const toggleSelection = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-4 py-3 pr-10 border-b border-border/50">
        <FileUp className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-foreground">Documents</h2>
        <Badge className="ml-auto text-[10px] bg-primary/15 text-primary border-primary/30">
          {selectedIds.size} selected
        </Badge>
      </div>

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
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </ScrollArea>
    </div>
  );
}
