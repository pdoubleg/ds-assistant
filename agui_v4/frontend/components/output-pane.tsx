"use client";

/**
 * OutputPane — right pane showing generated audit forms, charts, and tables.
 *
 * Includes the SavedFormsPanel for listing, restoring, and deleting
 * previously-persisted audit forms.
 */

import React, { useState, useCallback, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuditAgent } from "@/hooks/use-audit-agent";
import type { AuditFormPayload, SavedFormSummary } from "@/hooks/use-audit-agent";
import { A2UIRenderer } from "@/components/a2ui-renderer";
import { getGridSpan } from "@/lib/layout-engine";
import {
  Loader2,
  ClipboardCheck,
  FolderArchive,
  BookmarkCheck,
  ChevronDown as ChevronDownIcon,
  RotateCcw,
  Shield,
  Trash2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useFlaggedHits } from "@/hooks/use-flagged-hits";
import { FlaggedHitsInline } from "@/components/doc-lens/flagged-hits-panel";
import { DocumentViewerSheet } from "@/components/document-viewer-sheet";
import { useUploadedDocs, type UploadedDoc } from "@/hooks/use-uploaded-docs";
import type { DocumentSummaryData } from "@/components/a2ui/documents";

const DOC_SUMMARIES_STORAGE_KEY = "agui_v3.docSummaries.v2";

// ── SavedFormsPanel ──────────────────────────────────────────────

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
    return () => {
      cancelled = true;
    };
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
    [onRestore]
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
    [onDeleteForm]
  );

  if (!open) return null;

  return (
    <div className="border-b border-border/50 bg-background">
      <div className="flex items-center gap-2 px-5 py-2.5 bg-secondary/30 border-b border-border/30">
        <FolderArchive className="h-4 w-4 text-primary" />
        <span className="text-sm font-semibold text-foreground">
          Saved Forms
        </span>
        <span className="text-xs text-muted-foreground">
          {loading
            ? "loading..."
            : `${forms.length} form${forms.length !== 1 ? "s" : ""}`}
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
            <p className="text-sm text-muted-foreground">
              No saved forms yet.
            </p>
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
                          ? "text-blue-700 dark:text-blue-400 border-blue-500/30"
                          : "text-orange-700 dark:text-orange-400 border-orange-500/30"
                      }`}
                    >
                      <Shield className="h-2.5 w-2.5 mr-0.5" />
                      {form.peril}
                    </Badge>
                    <Badge
                      variant="outline"
                      className={`text-[10px] ${
                        form.overall_outcome === "Meets"
                          ? "text-emerald-700 dark:text-emerald-400 border-emerald-500/30"
                          : "text-red-700 dark:text-red-400 border-red-500/30"
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

// ── SavedImagesPanel ─────────────────────────────────────────────

function SavedImagesPanel({
  open,
  onToggle,
}: {
  open: boolean;
  onToggle: () => void;
}) {
  const { state } = useAuditAgent();
  const { uploadedDocs } = useUploadedDocs();
  const flaggedHits = useFlaggedHits();
  const [previewDoc, setPreviewDoc] = useState<
    (UploadedDoc & { _id: string; _initialPage?: number }) | null
  >(null);
  const [previewQuery, setPreviewQuery] = useState("");
  const [summaries, setSummaries] = useState<Map<string, DocumentSummaryData>>(
    new Map()
  );

  useEffect(() => {
    try {
      const rawSummaries = window.localStorage.getItem(DOC_SUMMARIES_STORAGE_KEY);
      if (!rawSummaries) {
        return;
      }

      const parsed = JSON.parse(rawSummaries) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return;
      }

      const entries = Object.entries(parsed as Record<string, unknown>).filter(
        ([key, value]) =>
          typeof key === "string" &&
          !!value &&
          typeof value === "object" &&
          typeof (value as Record<string, unknown>).title === "string" &&
          typeof (value as Record<string, unknown>).summary === "string"
      ) as [string, DocumentSummaryData][];

      setSummaries(new Map(entries));
    } catch (error) {
      console.warn("[SavedImagesPanel] Failed to restore summaries:", error);
    }
  }, []);

  const previewDocsByFileName = useMemo(() => {
    const docs = new Map<string, UploadedDoc>();

    // Prefer uploaded-doc metadata because it usually includes the resolved URL.
    for (const doc of uploadedDocs) {
      docs.set(doc.file_name, doc);
    }

    // Fill any remaining gaps from shared agent state.
    for (const doc of state.documents || []) {
      const fileName =
        (doc.file_name as string) || (doc.content_url as string) || "Untitled";
      if (docs.has(fileName)) {
        continue;
      }

      docs.set(fileName, {
        file_name: fileName,
        claim_number: (doc.claim_number as string) || "",
        content_id: (doc.content_id as string) || "",
        mime_type: (doc.mime_type as string) || "application/octet-stream",
        content_url:
          (doc.content_url as string) || (doc.path as string) || "",
        domain: (doc.domain as "claim" | "policy") || "claim",
        document_type: (doc.document_type as string) || undefined,
        document_sub_type: (doc.document_sub_type as string) || undefined,
        document_description:
          (doc.document_description as string) || undefined,
        create_date: (doc.create_date as string) || "",
        source_system: (doc.source_system as string) || undefined,
        company_name: (doc.company_name as string) || undefined,
        content: (doc.content as string) || undefined,
        token_count:
          typeof doc.token_count === "number" ? doc.token_count : undefined,
      });
    }

    return docs;
  }, [state.documents, uploadedDocs]);

  const uploadedContentByName = useMemo(() => {
    const contentMap = new Map<string, string>();
    for (const doc of uploadedDocs) {
      if (doc.content) {
        contentMap.set(doc.file_name, doc.content);
      }
    }
    return contentMap;
  }, [uploadedDocs]);

  const contentByName = useMemo(() => {
    const contentMap = new Map<string, Record<string, unknown>>();
    for (const doc of state.documents || []) {
      const fileName =
        (doc.file_name as string) || (doc.content_url as string) || "";
      if (fileName) {
        contentMap.set(fileName, doc);
      }
    }
    return contentMap;
  }, [state.documents]);

  const getTextContent = useCallback(
    (fileName: string): string => {
      const fromUploaded = uploadedContentByName.get(fileName);
      if (fromUploaded) {
        return fromUploaded;
      }

      const stateDoc = contentByName.get(fileName);
      const fromState =
        (stateDoc?.content as string) || (stateDoc?.text as string) || "";
      if (fromState) {
        return fromState;
      }

      return previewDocsByFileName.get(fileName)?.document_description || "";
    },
    [uploadedContentByName, contentByName, previewDocsByFileName]
  );

  const handlePreviewDoc = useCallback(
    (fileName: string, page: number, query?: string) => {
      const doc = previewDocsByFileName.get(fileName);
      setPreviewQuery(query ?? "");

      if (!doc) {
        setPreviewDoc(null);
        return;
      }

      setPreviewDoc({
        ...doc,
        _id: doc.content_id || doc.file_name,
        _initialPage: page,
      });
    },
    [previewDocsByFileName]
  );

  if (!open) return null;

  return (
    <>
      <div className="border-b border-border/50 bg-background">
        <div className="flex items-center gap-2 px-5 py-2.5 bg-secondary/30 border-b border-border/30">
          <BookmarkCheck className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">
            Saved Images
          </span>
          <span className="text-xs text-muted-foreground">
            {flaggedHits.flagCount} image
            {flaggedHits.flagCount !== 1 ? "s" : ""}
          </span>
          <button
            type="button"
            onClick={onToggle}
            className="ml-auto p-1 rounded text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
            title="Close saved images"
          >
            <ChevronDownIcon className="h-4 w-4 rotate-180" />
          </button>
        </div>

        <div className="max-h-96 overflow-y-auto px-3 py-2">
          <FlaggedHitsInline
            flaggedHits={flaggedHits.flaggedHits}
            getImageUrl={flaggedHits.getImageUrl}
            onRemove={flaggedHits.removeFlag}
            onClearAll={flaggedHits.clearAll}
            onDownloadImage={flaggedHits.downloadImage}
            onPreviewDoc={handlePreviewDoc}
            isFlagged={flaggedHits.isFlagged}
            onToggleFlag={flaggedHits.toggleFlag}
          />
        </div>
      </div>

      {previewDoc && (
        <DocumentViewerSheet
          doc={previewDoc as any}
          open={!!previewDoc}
          onOpenChange={(isOpen) => {
            if (!isOpen) {
              setPreviewDoc(null);
              setPreviewQuery("");
            }
          }}
          initialPage={previewDoc._initialPage}
          highlightQuery={previewQuery}
          textContent={getTextContent(previewDoc.file_name)}
          summaryData={summaries.get(previewDoc.file_name)}
        />
      )}
    </>
  );
}

/** Component types excluded from output delete mode. */
const DELETE_MODE_PROTECTED_TYPES = new Set(["a2ui.AuditQuestionForm"]);

// ── OutputPane ────────────────────────────────────────────────────

export function OutputPane() {
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
  const { uploadedDocs } = useUploadedDocs();
  const outputComponents = componentsByZone.output || [];
  const [showSavedForms, setShowSavedForms] = useState(false);
  const [showSavedImages, setShowSavedImages] = useState(false);
  const [deleteMode, setDeleteMode] = useState(false);
  const flaggedHits = useFlaggedHits();
  const [summaries, setSummaries] = useState<Map<string, DocumentSummaryData>>(
    new Map()
  );

  useEffect(() => {
    try {
      const rawSummaries = window.localStorage.getItem(DOC_SUMMARIES_STORAGE_KEY);
      if (!rawSummaries) {
        return;
      }

      const parsed = JSON.parse(rawSummaries) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return;
      }

      const entries = Object.entries(parsed as Record<string, unknown>).filter(
        ([key, value]) =>
          typeof key === "string" &&
          !!value &&
          typeof value === "object" &&
          typeof (value as Record<string, unknown>).title === "string" &&
          typeof (value as Record<string, unknown>).summary === "string"
      ) as [string, DocumentSummaryData][];

      setSummaries(new Map(entries));
    } catch (error) {
      console.warn("[OutputPane] Failed to restore summaries:", error);
    }
  }, []);

  const previewDocsByContentId = useMemo(() => {
    const docs = new Map<string, UploadedDoc>();

    for (const doc of uploadedDocs) {
      const key = doc.content_id || doc.file_name;
      docs.set(key, doc);
    }

    for (const doc of state.documents || []) {
      const fileName =
        (doc.file_name as string) || (doc.content_url as string) || "Untitled";
      const key = (doc.content_id as string) || fileName;
      if (docs.has(key)) {
        continue;
      }

      docs.set(key, {
        file_name: fileName,
        claim_number: (doc.claim_number as string) || "",
        content_id: (doc.content_id as string) || "",
        mime_type: (doc.mime_type as string) || "application/octet-stream",
        content_url:
          (doc.content_url as string) || (doc.path as string) || "",
        domain: (doc.domain as "claim" | "policy") || "claim",
        document_type: (doc.document_type as string) || undefined,
        document_sub_type: (doc.document_sub_type as string) || undefined,
        document_description:
          (doc.document_description as string) || undefined,
        create_date: (doc.create_date as string) || "",
        source_system: (doc.source_system as string) || undefined,
        company_name: (doc.company_name as string) || undefined,
        content: (doc.content as string) || undefined,
        token_count:
          typeof doc.token_count === "number" ? doc.token_count : undefined,
      });
    }

    return docs;
  }, [state.documents, uploadedDocs]);

  const uploadedContentByName = useMemo(() => {
    const contentMap = new Map<string, string>();
    for (const doc of uploadedDocs) {
      if (doc.content) {
        contentMap.set(doc.file_name, doc.content);
      }
    }
    return contentMap;
  }, [uploadedDocs]);

  const contentByName = useMemo(() => {
    const contentMap = new Map<string, Record<string, unknown>>();
    for (const doc of state.documents || []) {
      const fileName =
        (doc.file_name as string) || (doc.content_url as string) || "";
      if (fileName) {
        contentMap.set(fileName, doc);
      }
    }
    return contentMap;
  }, [state.documents]);

  const getTextContent = useCallback(
    (fileName: string): string => {
      const fromUploaded = uploadedContentByName.get(fileName);
      if (fromUploaded) {
        return fromUploaded;
      }

      const stateDoc = contentByName.get(fileName);
      const fromState =
        (stateDoc?.content as string) || (stateDoc?.text as string) || "";
      if (fromState) {
        return fromState;
      }

      for (const doc of previewDocsByContentId.values()) {
        if (doc.file_name === fileName) {
          return doc.document_description || "";
        }
      }

      return "";
    },
    [uploadedContentByName, contentByName, previewDocsByContentId]
  );

  // IDs of output components eligible for removal (excludes protected types)
  const removableIds = useMemo(
    () =>
      new Set(
        outputComponents
          .filter((c) => !DELETE_MODE_PROTECTED_TYPES.has(c.type))
          .map((c) => c.id)
      ),
    [outputComponents]
  );

  // Auto-exit delete mode when nothing removable remains or generation starts
  useEffect(() => {
    if (deleteMode && (removableIds.size === 0 || isGenerating)) {
      setDeleteMode(false);
    }
  }, [deleteMode, removableIds, isGenerating]);

  // Citation card → document preview state
  const [citationPreviewDoc, setCitationPreviewDoc] = useState<
    (Record<string, unknown> & { _initialPage?: number }) | null
  >(null);

  const handleCitationPreview = useCallback(
    (contentId: string, page: number) => {
      const doc = previewDocsByContentId.get(contentId);
      if (doc) {
        setCitationPreviewDoc({ ...doc, _initialPage: page });
      } else {
        setCitationPreviewDoc({
          file_name: contentId,
          _initialPage: page,
        });
      }
    },
    [previewDocsByContentId]
  );

  const handleFormSubmit = useCallback(
    async (formPayload: AuditFormPayload, title?: string) => {
      await saveForm(formPayload, title);
    },
    [saveForm]
  );

  const handleFormClose = useCallback(() => {
    const components = state.components || [];
    const filtered = components.filter(
      (c) => c.type !== "a2ui.AuditQuestionForm"
    );
    setState({ ...state, components: filtered });
  }, [state, setState]);

  const handleFormDelete = useCallback(async () => {
    if (!state.current_form_id) return;
    await deleteForm(state.current_form_id);
    const components = state.components || [];
    const filtered = components.filter(
      (c) => c.type !== "a2ui.AuditQuestionForm"
    );
    setState({ ...state, components: filtered, current_form_id: null });
  }, [state, setState, deleteForm]);

  const handlePanelDelete = useCallback(
    async (formId: string) => {
      await deleteForm(formId);
      if (state.current_form_id === formId) {
        const components = state.components || [];
        const filtered = components.filter(
          (c) => c.type !== "a2ui.AuditQuestionForm"
        );
        setState({ ...state, components: filtered, current_form_id: null });
      }
    },
    [state, setState, deleteForm]
  );

  const handleRestore = useCallback(
    async (formId: string) => {
      try {
        await restoreForm(formId);
      } catch (err) {
        console.error("Restore failed:", err);
      }
    },
    [restoreForm]
  );

  const toggleSavedForms = useCallback(() => {
    setShowSavedForms((prev) => !prev);
  }, []);

  const toggleSavedImages = useCallback(() => {
    setShowSavedImages((prev) => !prev);
  }, []);

  /** Remove a single output component by ID from shared agent state. */
  const handleDeleteComponent = useCallback(
    (componentId: string) => {
      const components = state.components || [];
      setState({
        ...state,
        components: components.filter((c) => c.id !== componentId),
      });
    },
    [state, setState]
  );

  /** Remove all removable output components and exit delete mode. */
  const handleDeleteAllRemovable = useCallback(() => {
    const components = state.components || [];
    setState({
      ...state,
      components: components.filter((c) => !removableIds.has(c.id)),
    });
    setDeleteMode(false);
  }, [state, setState, removableIds]);

  const headerBar = (
    <div className="flex items-center gap-2.5 px-4 pr-10 border-b border-border/50 shrink-0 h-12">
      <ClipboardCheck className="h-[18px] w-[18px] text-primary" />
      <h2 className="text-[15px] font-semibold tracking-tight text-foreground">Output</h2>
      {isComplete && (
        <Badge variant="success" className="text-[11px]">
          Complete
        </Badge>
      )}
      {isGenerating && (
        <Badge variant="outline" className="text-[11px] gap-1">
          <Loader2 className="h-3 w-3 animate-spin" />
          Generating
        </Badge>
      )}
      {outputComponents.length > 0 && (
        <span className="text-[11px] text-muted-foreground">
          {outputComponents.length} component
          {outputComponents.length !== 1 ? "s" : ""}
        </span>
      )}

      <div className="ml-auto flex items-center gap-2">
        {deleteMode ? (
          <>
            <Badge
              variant="outline"
              className="text-[11px] text-destructive border-destructive/30 bg-destructive/5"
            >
              Delete Mode
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setDeleteMode(false)}
              className="h-8 gap-1.5 text-[11px] hover:bg-secondary hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={handleDeleteAllRemovable}
              className="h-8 gap-1.5 text-[11px]"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete All ({removableIds.size})
            </Button>
          </>
        ) : (
          <>
            {state.current_form_id && (
              <Badge
                variant="outline"
                className="text-[11px] text-muted-foreground border-border/40"
              >
                Form: {state.current_form_id.slice(0, 8)}...
              </Badge>
            )}
            {removableIds.size > 0 && !isGenerating && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setDeleteMode(true)}
                className="h-8 gap-1.5 text-[11px] text-muted-foreground hover:text-destructive hover:border-destructive/40"
                title="Enter delete mode to remove output components"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            )}
            <Button
              variant={showSavedForms ? "secondary" : "outline"}
              size="sm"
              onClick={toggleSavedForms}
              className="h-8 gap-1.5 text-[11px]"
            >
              <FolderArchive className="h-3.5 w-3.5" />
              Saved Forms
              <ChevronDownIcon
                className={`h-3.5 w-3.5 transition-transform duration-200 ${showSavedForms ? "rotate-180" : ""}`}
              />
            </Button>
            <Button
              variant={showSavedImages ? "secondary" : "outline"}
              size="sm"
              onClick={toggleSavedImages}
              className="h-8 gap-1.5 text-[11px]"
            >
              <BookmarkCheck className="h-3.5 w-3.5" />
              Saved Images
              {flaggedHits.flagCount > 0 && (
                <Badge
                  variant="secondary"
                  className="text-[9px] px-1 py-0 h-4 ml-0.5"
                >
                  {flaggedHits.flagCount}
                </Badge>
              )}
              <ChevronDownIcon
                className={`h-3.5 w-3.5 transition-transform duration-200 ${showSavedImages ? "rotate-180" : ""}`}
              />
            </Button>
          </>
        )}
      </div>
    </div>
  );

  // Empty state
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

        <SavedImagesPanel
          open={showSavedImages}
          onToggle={toggleSavedImages}
        />

        <div className="relative flex flex-col items-center justify-center flex-1 overflow-hidden select-none">
          {/* ── Dot-grid texture ── */}
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.08] dark:opacity-[0.06]"
            style={{
              backgroundImage:
                "radial-gradient(circle, currentColor 0.75px, transparent 0.75px)",
              backgroundSize: "20px 20px",
            }}
          />

          {/* ── Soft radial gradient wash ── */}
          <motion.div
            className="pointer-events-none absolute w-[520px] h-[520px] rounded-full"
            style={{
              top: "50%",
              left: "50%",
              x: "-50%",
              y: "-55%",
              background:
                "radial-gradient(circle, var(--primary) 0%, transparent 70%)",
            }}
            animate={{ opacity: [0.03, 0.06, 0.03], scale: [1, 1.08, 1] }}
            transition={{
              duration: 10,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />

          {/* ── Floating accent ring (decorative) ── */}
          <motion.div
            className="pointer-events-none absolute h-56 w-56 rounded-full border border-border/20"
            style={{ top: "50%", left: "50%", x: "-50%", y: "-55%" }}
            animate={{ scale: [1, 1.12, 1], opacity: [0.5, 0.2, 0.5] }}
            transition={{
              duration: 12,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />

          {/* ── Center icon assembly ── */}
          <motion.div
            className="relative z-10 mb-5"
            animate={{ y: [0, -5, 0] }}
            transition={{
              duration: 6,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          >
            {/* Ambient glow behind the icon */}
            <div className="absolute -inset-6 rounded-full bg-primary/[0.07] blur-2xl" />

            {/* Outer decorative ring */}
            <div className="absolute -inset-3 rounded-2xl border border-border/15 dark:border-border/25" />

            {/* Glass icon card */}
            <div className="relative h-[68px] w-[68px] rounded-2xl bg-secondary/40 dark:bg-secondary/30 border border-border/30 flex items-center justify-center backdrop-blur-sm shadow-sm">
              <ClipboardCheck className="h-8 w-8 text-foreground/30" />
            </div>
          </motion.div>

          {/* ── Copy ── */}
          <motion.div
            className="relative z-10 text-center"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.15 }}
          >
            <h3 className="text-base font-medium tracking-tight text-foreground/60">
              No generated content yet
            </h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground/50 max-w-[320px]">
              Q-Bot's output will appear here as it's created, and will persist throughout the session. 
              Rendered components can be saved, exported, or copied to the clipboard.
            </p>
          </motion.div>
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

      <SavedImagesPanel
        open={showSavedImages}
        onToggle={toggleSavedImages}
      />

      <ScrollArea className="flex-1">
        <div className="px-5 py-5">
          <div className="grid grid-cols-12 gap-5 auto-rows-min">
            {outputComponents.map((component, index) => {
              const isRemovable = removableIds.has(component.id);

              return (
                <motion.div
                  key={component.id || `output-${index}`}
                  className={`${getGridSpan(component)} relative`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                >
                  {deleteMode && isRemovable && (
                    <motion.div
                      className="absolute inset-0 rounded-xl ring-2 ring-destructive/25 pointer-events-none z-5"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    />
                  )}

                  <AnimatePresence>
                    {deleteMode && isRemovable && (
                      <motion.button
                        type="button"
                        initial={{ opacity: 0, scale: 0.6 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.6 }}
                        transition={{
                          type: "spring",
                          stiffness: 400,
                          damping: 22,
                        }}
                        onClick={() => handleDeleteComponent(component.id)}
                        className="absolute -top-2.5 -right-2.5 z-10 h-7 w-7 rounded-full bg-destructive text-destructive-foreground shadow-lg flex items-center justify-center hover:bg-destructive/90 transition-colors cursor-pointer"
                        title="Remove component"
                      >
                        <X className="h-3.5 w-3.5" />
                      </motion.button>
                    )}
                  </AnimatePresence>

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
                        : component.type === "a2ui.CitationCard"
                          ? { onPreviewDoc: handleCitationPreview }
                          : undefined
                    }
                  />
                </motion.div>
              );
            })}
          </div>
        </div>
      </ScrollArea>

      {citationPreviewDoc && (
        <DocumentViewerSheet
          doc={citationPreviewDoc as any}
          open={!!citationPreviewDoc}
          onOpenChange={(isOpen) => {
            if (!isOpen) setCitationPreviewDoc(null);
          }}
          initialPage={citationPreviewDoc._initialPage}
          textContent={
            typeof citationPreviewDoc.file_name === "string"
              ? getTextContent(citationPreviewDoc.file_name)
              : undefined
          }
          summaryData={
            typeof citationPreviewDoc.file_name === "string"
              ? summaries.get(citationPreviewDoc.file_name)
              : undefined
          }
        />
      )}
    </div>
  );
}
