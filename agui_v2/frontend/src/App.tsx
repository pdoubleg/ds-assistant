/**
 * Q-Bot App — Three-Pane Collapsible Layout
 *
 * Layout (left to right, default proportions):
 *   1. Chat UI      (flex 1)  — user input with file upload
 *   2. Documents     (flex 1)  — uploaded / retrieved document cards
 *   3. Audit Output  (flex 2)  — generated forms, charts, tables
 *
 * Each pane can be collapsed to a narrow sidebar strip and re-expanded.
 */

import React, {
  useState,
  useCallback,
  useRef,
  useMemo,
  useEffect,
  createContext,
  useContext,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuditAgent } from "@/hooks/useAuditAgent";
import type { AuditFormPayload, SavedFormSummary } from "@/hooks/useAuditAgent";
import { A2UIRenderer } from "@/components/A2UIRenderer";
import { DocumentCard } from "@/components/A2UI/Documents";
import { getGridSpan } from "@/lib/layout-engine";
import { useTheme } from "@/lib/theme-context";
import {
  FileUp,
  Send,
  Loader2,
  FileText,
  ChevronRight,
  BarChart3,
  ClipboardCheck,
  Paperclip,
  Sun,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  MessageSquareText,
  FolderOpen,
  LayoutDashboard,
  Check,
  FolderArchive,
  ChevronDown as ChevronDownIcon,
  RotateCcw,
  Shield,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import qbotLogo from "../assets/q-bot.PNG";

// ── MIME type helper ─────────────────────────────────────────────
const EXT_TO_MIME: Record<string, string> = {
  pdf: "application/pdf",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  tiff: "image/tiff",
};

function mimeFromExt(ext: string): string {
  return EXT_TO_MIME[ext.toLowerCase()] || "application/octet-stream";
}

// ── Dummy documents conforming to Document schema ────────────────
const DUMMY_DOCUMENTS = [
  {
    file_name: "Corporate_Security_Policy_v3.2.pdf",
    claim_number: "CLM-2026-00100",
    content_id: "doc-001",
    mime_type: "application/pdf",
    content_url: "/docs/Corporate_Security_Policy_v3.2.pdf",
    domain: "claim" as const,
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
    domain: "claim" as const,
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
    mime_type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    content_url: "/docs/IT_Risk_Assessment_Template.xlsx",
    domain: "claim" as const,
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
    mime_type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    content_url: "/docs/Access_Control_Procedures.docx",
    domain: "policy" as const,
    document_type: "Procedure",
    document_sub_type: "Access Control",
    document_description:
      "Detailed procedures for user provisioning, access reviews, privileged account management, and deprovisioning.",
    create_date: "2026-02-18T16:45:00Z",
    source_system: "UPLOAD",
  },
];

// ── Pane metadata (icon + label) ─────────────────────────────────
type PaneId = "chat" | "documents" | "output";

interface PaneMeta {
  icon: React.ReactNode;
  label: string;
}

const PANE_META: Record<PaneId, PaneMeta> = {
  chat: { icon: <MessageSquareText className="h-4 w-4" />, label: "Chat" },
  documents: { icon: <FolderOpen className="h-4 w-4" />, label: "Documents" },
  output: { icon: <LayoutDashboard className="h-4 w-4" />, label: "Output" },
};

// ── Shared uploaded-docs context (Document schema) ───────────────

interface UploadedDoc {
  file_name: string;
  claim_number: string;
  content_id: string;
  mime_type: string;
  content_url: string;
  domain: "claim" | "policy";
  document_type?: string;
  document_sub_type?: string;
  document_description?: string;
  create_date: string;
  source_system?: string;
  company_name?: string;
}

interface UploadedDocsContextValue {
  uploadedDocs: UploadedDoc[];
  addUploadedDoc: (doc: UploadedDoc) => void;
  /** Replace an existing entry by file_name, or append if not found. */
  updateUploadedDoc: (fileName: string, doc: UploadedDoc) => void;
}

const UploadedDocsContext = createContext<UploadedDocsContextValue>({
  uploadedDocs: [],
  addUploadedDoc: () => {},
  updateUploadedDoc: () => {},
});

function useUploadedDocs() {
  return useContext(UploadedDocsContext);
}

// ── Collapsed sidebar strip ──────────────────────────────────────

function CollapsedStrip({
  paneId,
  onExpand,
}: {
  paneId: PaneId;
  onExpand: () => void;
}) {
  const meta = PANE_META[paneId];
  return (
    <motion.button
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 40, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.2 }}
      onClick={onExpand}
      className="shrink-0 flex flex-col items-center justify-center gap-2 border-r border-border/50 bg-card/60 hover:bg-accent/10 transition-colors cursor-pointer overflow-hidden"
      title={`Expand ${meta.label}`}
    >
      <span className="text-accent">{meta.icon}</span>
      <span className="text-[10px] font-medium text-muted-foreground [writing-mode:vertical-lr] rotate-180 select-none tracking-wide">
        {meta.label}
      </span>
      <PanelLeftOpen className="h-3.5 w-3.5 text-muted-foreground/60" />
    </motion.button>
  );
}

// ── Chat Pane ────────────────────────────────────────────────────

function ChatPane() {
  const { runAudit, addDocument, isGenerating, state, lastAssistantMessage, toolActivity } =
    useAuditAgent();
  const { addUploadedDoc, updateUploadedDoc } = useUploadedDocs();
  const [message, setMessage] = useState("");
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [chatMessages, setChatMessages] = useState<
    Array<{ id: string; role: "user" | "assistant"; content: string }>
  >([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Welcome to Q-Bot, your AI-powered TFR audit assistant. Upload documents and I'll generate a custom audit questionnaire with insights. You can upload .pdf, .docx, or .xlsx files.",
    },
  ]);

  const lastShownRef = useRef<string | null>(null);

  useEffect(() => {
    if (!lastAssistantMessage) return;
    if (isGenerating) return;
    if (lastAssistantMessage === lastShownRef.current) return;

    lastShownRef.current = lastAssistantMessage;
    setChatMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content: lastAssistantMessage,
      },
    ]);
  }, [lastAssistantMessage, isGenerating]);

  // Upload a single file: show card immediately, then call backend
  const uploadFile = useCallback(
    async (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase() || "pdf";
      const friendlySize =
        file.size > 1024 * 1024
          ? `${(file.size / 1024 / 1024).toFixed(1)} MB`
          : `${(file.size / 1024).toFixed(1)} KB`;
      const nowIso = new Date().toISOString();
      const dummyContentId = crypto.randomUUID();

      // Show the card immediately with placeholder metadata
      addUploadedDoc({
        file_name: file.name,
        claim_number: "CLM-UPLOAD-000",
        content_id: dummyContentId,
        mime_type: mimeFromExt(ext),
        content_url: file.name,
        domain: "claim",
        document_type: "Upload",
        document_description: `Uploading ${friendlySize}...`,
        create_date: nowIso,
        source_system: "UPLOAD",
      });

      const backendUrl =
        import.meta.env.VITE_BACKEND_URL || "http://localhost:8001";
      try {
        const formData = new FormData();
        formData.append("file", file);
        const resp = await fetch(`${backendUrl}/upload`, {
          method: "POST",
          body: formData,
        });

        if (!resp.ok) {
          console.error(`Upload failed for ${file.name}:`, await resp.text());
          updateUploadedDoc(file.name, {
            file_name: file.name,
            claim_number: "CLM-UPLOAD-000",
            content_id: dummyContentId,
            mime_type: mimeFromExt(ext),
            content_url: file.name,
            domain: "claim",
            document_type: "Upload",
            document_description: "Upload failed",
            create_date: nowIso,
            source_system: "UPLOAD",
          });
          return;
        }

        const data = await resp.json();

        // Replace the placeholder card with real metadata
        updateUploadedDoc(file.name, {
          file_name: data.filename || file.name,
          claim_number: "CLM-UPLOAD-000",
          content_id: dummyContentId,
          mime_type: mimeFromExt(data.file_type || ext),
          content_url: data.path || file.name,
          domain: "claim",
          document_type: "Upload",
          document_description: `${data.file_size || friendlySize}, ${data.page_count ?? 0} pages`,
          create_date: nowIso,
          source_system: "UPLOAD",
        });

        // Also push into agent state for analysis
        addDocument({
          file_name: data.filename || file.name,
          claim_number: "CLM-UPLOAD-000",
          content_id: dummyContentId,
          mime_type: mimeFromExt(data.file_type || ext),
          content_url: data.path || file.name,
          domain: "claim",
          document_type: "Upload",
          document_description: "",
          create_date: nowIso,
          source_system: "UPLOAD",
          content: data.content ?? "",
        });
      } catch (err) {
        console.error(`Upload error for ${file.name}:`, err);
        updateUploadedDoc(file.name, {
          file_name: file.name,
          claim_number: "CLM-UPLOAD-000",
          content_id: dummyContentId,
          mime_type: mimeFromExt(ext),
          content_url: file.name,
          domain: "claim",
          document_type: "Upload",
          document_description: "Upload failed",
          create_date: nowIso,
          source_system: "UPLOAD",
        });
        addDocument({
          file_name: file.name,
          claim_number: "CLM-UPLOAD-000",
          content_id: dummyContentId,
          mime_type: mimeFromExt(ext),
          content_url: file.name,
          domain: "claim",
          document_type: "Upload",
          document_description: "",
          create_date: nowIso,
          source_system: "UPLOAD",
          content: "",
        });
      }
    },
    [addDocument, addUploadedDoc, updateUploadedDoc]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      const validFiles = files.filter((f) => {
        const ext = f.name.split(".").pop()?.toLowerCase();
        return ["pdf", "docx", "xlsx"].includes(ext || "");
      });
      setAttachedFiles((prev) => [...prev, ...validFiles]);
      validFiles.forEach((f) => uploadFile(f));
    },
    [uploadFile]
  );

  const removeFile = useCallback((index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleSend = useCallback(async () => {
    if (!message.trim() && attachedFiles.length === 0) return;

    const userContent =
      message.trim() ||
      "Please analyze the uploaded documents and generate a TFR audit questionnaire.";

    const fileNames = attachedFiles.map((f) => f.name);

    setChatMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "user",
        content:
          userContent +
          (fileNames.length > 0
            ? `\n\n[Attached ${fileNames.length} file(s): ${fileNames.join(", ")}]`
            : ""),
      },
    ]);

    setMessage("");
    setAttachedFiles([]);

    try {
      await runAudit(userContent);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            "I encountered an error during analysis. Please try again.",
        },
      ]);
    }
  }, [message, attachedFiles, runAudit]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className="flex flex-col h-full">
      {/* Chat header */}
      <div className="flex items-center gap-2 px-4 py-3 pr-10 border-b border-border/50">
        <Send className="h-4 w-4 text-accent" />
        <h2 className="text-sm font-semibold text-foreground">Chat</h2>
        {isGenerating && (
          <Badge variant="outline" className="ml-auto text-[10px] gap-1">
            <Loader2 className="h-3 w-3 animate-spin" />
            {state.current_step || "Working..."}
          </Badge>
        )}
      </div>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {chatMessages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-primary/15 text-foreground border border-primary/25"
                  : "bg-secondary/60 text-foreground/90 border border-border/30"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
          </div>
        ))}

        {isGenerating && (
          <div className="flex justify-start">
            <div className="bg-secondary/60 rounded-xl px-4 py-3 border border-border/30 min-w-[220px]">
              {toolActivity.length > 0 && (
                <div className="space-y-1.5">
                  {toolActivity.map((tc) => (
                    <div key={tc.id} className="flex items-center gap-2 text-sm">
                      {tc.status === "running" ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-accent shrink-0" />
                      ) : (
                        <Check className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                      )}
                      <span
                        className={
                          tc.status === "complete"
                            ? "text-muted-foreground"
                            : "text-foreground/80"
                        }
                      >
                        {tc.displayName}
                      </span>
                    </div>
                  ))}
                  {toolActivity.every((tc) => tc.status === "complete") && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground pt-0.5">
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-accent shrink-0" />
                      <span>Composing response...</span>
                    </div>
                  )}
                </div>
              )}

              {toolActivity.length === 0 && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin text-accent" />
                  <span>{state.current_step || "Working..."}</span>
                </div>
              )}

              {state.progress > 0 && (
                <div className="mt-2 h-1.5 bg-secondary rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-linear-to-r from-accent to-primary rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${state.progress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              )}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Attached files preview */}
      {attachedFiles.length > 0 && (
        <div className="px-4 py-2 border-t border-border/30 bg-secondary/20">
          <div className="flex flex-wrap gap-2">
            {attachedFiles.map((file, idx) => (
              <div
                key={idx}
                className="flex items-center gap-1.5 bg-secondary/50 rounded-lg px-2.5 py-1 text-xs"
              >
                <FileText className="h-3 w-3 text-accent" />
                <span className="text-foreground/80 max-w-[120px] truncate">
                  {file.name}
                </span>
                <button
                  onClick={() => removeFile(idx)}
                  className="text-muted-foreground hover:text-destructive ml-1"
                >
                  &times;
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-border/50 px-4 py-3">
        <div className="flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.xlsx"
            onChange={handleFileSelect}
            className="hidden"
          />
          <Button
            variant="ghost"
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            className="shrink-0 h-9 w-9 text-muted-foreground hover:text-accent"
            title="Attach documents (.pdf, .docx, .xlsx)"
          >
            <Paperclip className="h-4 w-4" />
          </Button>

          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your documents or request a TFR audit..."
            rows={1}
            className="flex-1 resize-none bg-secondary/50 border border-border/50 rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 min-h-[36px] max-h-[120px]"
            style={{
              height: "auto",
              overflowY:
                message.split("\n").length > 3 ? "auto" : "hidden",
            }}
          />

          <Button
            variant="default"
            size="icon"
            onClick={handleSend}
            disabled={
              isGenerating || (!message.trim() && attachedFiles.length === 0)
            }
            className="shrink-0 h-9 w-9"
          >
            {isGenerating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ── Documents Pane ───────────────────────────────────────────────

function DocumentsPane() {
  const { state } = useAuditAgent();
  const { uploadedDocs } = useUploadedDocs();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(
    new Set(["dummy-0", "dummy-1"])
  );

  // Build a unified list: dummies, then locally-uploaded, then agent state (deduped by file_name)
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
      const fileName = (d.file_name as string) || (d.content_url as string) || "Untitled";
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
        document_description: (d.document_description as string) || undefined,
        create_date: (d.create_date as string) || "",
        source_system: (d.source_system as string) || undefined,
        company_name: (d.company_name as string) || undefined,
        _id: `agent-${i}`,
      });
    });

    return docs;
  }, [state.documents, uploadedDocs]);

  // Selected cards float to the top
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
      {/* Documents header */}
      <div className="flex items-center gap-2 px-4 py-3 pr-10 border-b border-border/50">
        <FileUp className="h-4 w-4 text-accent" />
        <h2 className="text-sm font-semibold text-foreground">Documents</h2>
        <Badge className="ml-auto text-[10px] bg-accent/20 text-accent border-accent/40">
          {selectedIds.size} selected
        </Badge>
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
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
    </div>
  );
}

// ── Output Pane ──────────────────────────────────────────────────

/**
 * Collapsible panel that lists saved forms and lets users restore one.
 * Renders inline between the output header and content -- not a floating overlay.
 */
function SavedFormsPanel({
  open,
  onToggle,
  onRestore,
  onDeleteForm,
  listSavedForms,
  activeFormId,
}: {
  open: boolean;
  onToggle: () => void;
  onRestore: (formId: string) => void;
  onDeleteForm: (formId: string) => Promise<void>;
  listSavedForms: () => Promise<SavedFormSummary[]>;
  activeFormId: string | null;
}) {
  const [forms, setForms] = useState<SavedFormSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Fetch the list every time the panel opens
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    listSavedForms()
      .then((result) => {
        if (!cancelled) setForms(result);
      })
      .catch((err) => console.error("Failed to load saved forms:", err))
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [open, listSavedForms]);

  const handleSelect = useCallback(
    async (formId: string) => {
      setRestoringId(formId);
      try {
        await onRestore(formId);
      } finally {
        setRestoringId(null);
      }
    },
    [onRestore],
  );

  const handleDeleteItem = useCallback(
    async (formId: string) => {
      setDeletingId(formId);
      try {
        await onDeleteForm(formId);
        setForms((prev) => prev.filter((f) => f.id !== formId));
      } catch (err) {
        console.error("Delete failed:", err);
      } finally {
        setDeletingId(null);
      }
    },
    [onDeleteForm],
  );

  if (!open) return null;

  return (
    <div className="border-b border-border/50 bg-background">
      {/* Panel header */}
      <div className="flex items-center gap-2 px-5 py-2.5 bg-secondary/30 border-b border-border/30">
        <FolderArchive className="h-4 w-4 text-primary" />
        <span className="text-sm font-semibold text-foreground">Saved Forms</span>
        <span className="text-xs text-muted-foreground">
          {loading ? "loading..." : `${forms.length} form${forms.length !== 1 ? "s" : ""}`}
        </span>
        <button
          type="button"
          onClick={onToggle}
          className="ml-auto p-1 rounded text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          title="Close saved forms"
        >
          <ChevronDownIcon className="h-4 w-4 rotate-180" />
        </button>
      </div>

      {/* Form list */}
      <div className="max-h-64 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center gap-2 px-5 py-6 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading saved forms...
          </div>
        )}

        {!loading && forms.length === 0 && (
          <div className="px-5 py-6 text-center">
            <FolderArchive className="h-8 w-8 text-muted-foreground/30 mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">No saved forms yet.</p>
            <p className="text-xs text-muted-foreground/70 mt-1">
              Generate an audit form and click Submit to save it.
            </p>
          </div>
        )}

        {!loading &&
          forms.map((form) => {
            const isActive = form.id === activeFormId;
            const isRestoring = form.id === restoringId;

            return (
              <div
                key={form.id}
                className={`flex items-center gap-4 px-5 py-3 border-b border-border/20 last:border-b-0 transition-colors ${
                  isActive
                    ? "bg-primary/8 border-l-2 border-l-primary"
                    : "hover:bg-secondary/40"
                }`}
              >
                {/* Form info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-foreground truncate">
                      {form.title}
                    </span>
                    {isActive && (
                      <Badge className="text-[9px] bg-primary/20 text-primary border-primary/30 shrink-0">
                        Active
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge
                      variant="outline"
                      className={`text-[10px] ${
                        form.peril === "Exterior"
                          ? "text-blue-600 dark:text-blue-400 border-blue-500/30"
                          : "text-orange-600 dark:text-orange-400 border-orange-500/30"
                      }`}
                    >
                      <Shield className="h-2.5 w-2.5 mr-0.5" />
                      {form.peril}
                    </Badge>
                    <Badge
                      variant="outline"
                      className={`text-[10px] ${
                        form.overall_outcome === "Meets"
                          ? "text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                          : "text-red-600 dark:text-red-400 border-red-500/30"
                      }`}
                    >
                      {form.overall_outcome}
                    </Badge>
                    <span className="text-[10px] text-muted-foreground">
                      {form.question_count} questions
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      {new Date(form.updated_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1.5 shrink-0">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteItem(form.id)}
                    disabled={form.id === deletingId || isRestoring}
                    className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                    title="Delete form"
                  >
                    {form.id === deletingId ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Trash2 className="h-3.5 w-3.5" />
                    )}
                  </Button>
                  <Button
                    variant={isActive ? "outline" : "default"}
                    size="sm"
                    onClick={() => handleSelect(form.id)}
                    disabled={isRestoring}
                    className="h-8 text-xs gap-1.5"
                  >
                    {isRestoring ? (
                      <>
                        <Loader2 className="h-3 w-3 animate-spin" />
                        Loading...
                      </>
                    ) : (
                      <>
                        <RotateCcw className="h-3 w-3" />
                        {isActive ? "Reload" : "Restore"}
                      </>
                    )}
                  </Button>
                </div>
              </div>
            );
          })}
      </div>
    </div>
  );
}

function OutputPane() {
  const {
    componentsByZone,
    isGenerating,
    isComplete,
    state,
    setState,
    isSaving,
    saveForm,
    listSavedForms,
    restoreForm,
    deleteForm,
  } = useAuditAgent();
  const outputComponents = componentsByZone.output || [];
  const [showSavedForms, setShowSavedForms] = useState(false);

  const handleFormSubmit = useCallback(
    async (formPayload: AuditFormPayload, title?: string) => {
      await saveForm(formPayload, title);
    },
    [saveForm],
  );

  /** Remove the AuditQuestionForm component from the output. */
  const handleFormClose = useCallback(() => {
    const components = state.components || [];
    const filtered = components.filter(
      (c) => c.type !== "a2ui.AuditQuestionForm",
    );
    setState({ ...state, components: filtered });
  }, [state, setState]);

  /** Delete the currently active saved form from disk, then close it. */
  const handleFormDelete = useCallback(async () => {
    if (!state.current_form_id) return;
    await deleteForm(state.current_form_id);
    // Also remove the component from the UI
    const components = state.components || [];
    const filtered = components.filter(
      (c) => c.type !== "a2ui.AuditQuestionForm",
    );
    setState({ ...state, components: filtered, current_form_id: null });
  }, [state, setState, deleteForm]);

  /** Delete a form by ID from the saved-forms panel. */
  const handlePanelDelete = useCallback(
    async (formId: string) => {
      await deleteForm(formId);
      // If we just deleted the active form, clear it from state
      if (state.current_form_id === formId) {
        const components = state.components || [];
        const filtered = components.filter(
          (c) => c.type !== "a2ui.AuditQuestionForm",
        );
        setState({ ...state, components: filtered, current_form_id: null });
      }
    },
    [state, setState, deleteForm],
  );

  const handleRestore = useCallback(
    async (formId: string) => {
      try {
        await restoreForm(formId);
      } catch (err) {
        console.error("Restore failed:", err);
      }
    },
    [restoreForm],
  );

  const toggleSavedForms = useCallback(() => {
    setShowSavedForms((prev) => !prev);
  }, []);

  // Shared header bar used in both empty and populated states
  const headerBar = (
    <div className="flex items-center gap-2 px-4 py-2 pr-10 border-b border-border/50 shrink-0">
      <ClipboardCheck className="h-4 w-4 text-accent" />
      <h2 className="text-sm font-semibold text-foreground">
        Audit Output
      </h2>
      {isComplete && (
        <Badge variant="success" className="text-[10px]">
          Complete
        </Badge>
      )}
      {isGenerating && (
        <Badge variant="outline" className="text-[10px] gap-1">
          <Loader2 className="h-3 w-3 animate-spin" />
          Generating
        </Badge>
      )}
      {outputComponents.length > 0 && (
        <span className="text-[10px] text-muted-foreground">
          {outputComponents.length} component
          {outputComponents.length !== 1 ? "s" : ""}
        </span>
      )}

      <div className="ml-auto flex items-center gap-1.5">
        {state.current_form_id && (
          <Badge variant="outline" className="text-[10px] text-muted-foreground border-border/40">
            Form: {state.current_form_id.slice(0, 8)}...
          </Badge>
        )}
        <Button
          variant={showSavedForms ? "secondary" : "outline"}
          size="sm"
          onClick={toggleSavedForms}
          className="h-7 gap-1.5 text-xs"
        >
          <FolderArchive className="h-3.5 w-3.5" />
          Saved Forms
          <ChevronDownIcon
            className={`h-3 w-3 transition-transform duration-200 ${showSavedForms ? "rotate-180" : ""}`}
          />
        </Button>
      </div>
    </div>
  );

  if (outputComponents.length === 0 && !isGenerating) {
    return (
      <div className="flex flex-col h-full">
        {headerBar}

        <SavedFormsPanel
          open={showSavedForms}
          onToggle={toggleSavedForms}
          onRestore={handleRestore}
          onDeleteForm={handlePanelDelete}
          listSavedForms={listSavedForms}
          activeFormId={state.current_form_id}
        />

        <div className="flex flex-col items-center justify-center flex-1 text-center px-8">
          <div className="relative mb-6">
            <div className="absolute inset-0 bg-accent/20 rounded-full blur-2xl" />
            <ClipboardCheck className="h-16 w-16 text-accent/50 relative" />
          </div>
          <h3 className="text-lg font-semibold text-foreground/80 mb-2">
            No Audit Generated Yet
          </h3>
          <p className="text-sm text-muted-foreground max-w-md">
            Upload documents and ask Q-Bot to generate a TFR audit questionnaire.
            The forms, charts, and insights will appear here.
          </p>
          <div className="flex items-center gap-6 mt-8 text-muted-foreground/50">
            <div className="flex flex-col items-center gap-1">
              <BarChart3 className="h-8 w-8" />
              <span className="text-[10px]">Charts</span>
            </div>
            <ChevronRight className="h-4 w-4" />
            <div className="flex flex-col items-center gap-1">
              <FileText className="h-8 w-8" />
              <span className="text-[10px]">Tables</span>
            </div>
            <ChevronRight className="h-4 w-4" />
            <div className="flex flex-col items-center gap-1">
              <ClipboardCheck className="h-8 w-8" />
              <span className="text-[10px]">TFR Form</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {headerBar}

      <SavedFormsPanel
        open={showSavedForms}
        onToggle={toggleSavedForms}
        onRestore={handleRestore}
        onDeleteForm={handlePanelDelete}
        listSavedForms={listSavedForms}
        activeFormId={state.current_form_id}
      />

      {/* Output content */}
      <div className="flex-1 overflow-y-auto px-5 py-5">
        <div className="grid grid-cols-12 gap-5 auto-rows-min">
          {outputComponents.map((component, index) => (
            <motion.div
              key={component.id || `output-${index}`}
              className={getGridSpan(component)}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
            >
              <A2UIRenderer
                component={component}
                showErrors={true}
                extraProps={
                  component.type === "a2ui.AuditQuestionForm"
                    ? {
                        onSubmit: handleFormSubmit,
                        onClose: handleFormClose,
                        onDelete: handleFormDelete,
                        isSaving,
                        currentFormId: state.current_form_id,
                      }
                    : undefined
                }
              />
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Pane content map ─────────────────────────────────────────────

const PANE_CONTENT: Record<PaneId, React.FC> = {
  chat: ChatPane,
  documents: DocumentsPane,
  output: OutputPane,
};

const PANE_FLEX: Record<PaneId, number> = {
  chat: 1,
  documents: 1,
  output: 2,
};

// ── Main App ─────────────────────────────────────────────────────

function App() {
  const { theme, toggleTheme } = useTheme();

  const [uploadedDocs, setUploadedDocs] = useState<UploadedDoc[]>([]);
  const addUploadedDoc = useCallback((doc: UploadedDoc) => {
    setUploadedDocs((prev) => [...prev, doc]);
  }, []);
  const updateUploadedDoc = useCallback((fileName: string, doc: UploadedDoc) => {
    setUploadedDocs((prev) => {
      const idx = prev.findIndex((d) => d.file_name === fileName);
      if (idx === -1) return [...prev, doc];
      const next = [...prev];
      next[idx] = doc;
      return next;
    });
  }, []);

  const [expanded, setExpanded] = useState<Record<PaneId, boolean>>({
    chat: true,
    documents: true,
    output: true,
  });

  const expandedCount = useMemo(
    () => Object.values(expanded).filter(Boolean).length,
    [expanded]
  );

  const togglePane = useCallback(
    (id: PaneId) => {
      setExpanded((prev) => {
        const isOpen = prev[id];
        if (isOpen && expandedCount <= 1) return prev;
        return { ...prev, [id]: !isOpen };
      });
    },
    [expandedCount]
  );

  const paneOrder: PaneId[] = ["chat", "documents", "output"];

  return (
    <UploadedDocsContext.Provider value={{ uploadedDocs, addUploadedDoc, updateUploadedDoc }}>
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
      {/* ── Header ──────────────────────────────────────────── */}
      <header className="shrink-0 border-b border-primary/20 bg-card/95 backdrop-blur-sm">
        <div className="px-5 py-4 flex items-center gap-5">
          <div className="flex items-center gap-3 shrink-0">
            <div className="relative">
              <div className="absolute -inset-1 bg-primary/15 rounded-2xl blur-lg" />
              <img
                src={qbotLogo}
                alt="Q-Bot"
                className="relative h-14 w-14 rounded-xl shadow-lg shadow-primary/20 ring-1 ring-primary/20"
              />
            </div>
            <h1 className="text-2xl font-extrabold bg-linear-to-r from-primary via-accent to-primary bg-clip-text text-transparent tracking-tight">
              Q-Bot
            </h1>
          </div>

          <p className="text-3xl font-bold tracking-wide text-foreground/70 select-none" style={{ fontFamily: "'Roboto', sans-serif" }}>
            AI-Powered Quality Audit Assistant
          </p>

          <div className="flex-1" />

          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="h-9 w-9 shrink-0 text-muted-foreground hover:text-primary"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? (
              <Sun className="h-[18px] w-[18px]" />
            ) : (
              <Moon className="h-[18px] w-[18px]" />
            )}
          </Button>
        </div>
        <div className="h-px bg-linear-to-r from-transparent via-primary/40 to-transparent" />
      </header>

      {/* ── Three-pane layout ───────────────────────────────── */}
      <div className="flex-1 flex min-h-0">
        <AnimatePresence initial={false}>
          {paneOrder.map((id, idx) => {
            const isOpen = expanded[id];
            const PaneComponent = PANE_CONTENT[id];
            const isLast = idx === paneOrder.length - 1;
            const canCollapse = expandedCount > 1 || !isOpen;

            if (!isOpen) {
              return (
                <CollapsedStrip
                  key={`strip-${id}`}
                  paneId={id}
                  onExpand={() => togglePane(id)}
                />
              );
            }

            return (
              <motion.div
                key={`pane-${id}`}
                layout
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
                className={`min-w-0 min-h-0 flex flex-col ${
                  !isLast ? "border-r border-border/50" : ""
                } ${id === "output" ? "bg-background" : "bg-card/40"}`}
                style={{ flex: PANE_FLEX[id] }}
              >
                <div className="relative">
                  {canCollapse && (
                    <button
                      onClick={() => togglePane(id)}
                      className="absolute top-2.5 right-2 z-10 p-1 rounded-md text-muted-foreground/60 hover:text-foreground hover:bg-secondary/60 transition-colors"
                      title={`Collapse ${PANE_META[id].label}`}
                    >
                      <PanelLeftClose className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
                <PaneComponent />
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
    </UploadedDocsContext.Provider>
  );
}

export default App;
