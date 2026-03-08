"use client";

/**
 * OutputPane — right pane showing generated audit forms, charts, and tables.
 *
 * Includes the SavedFormsPanel for listing, restoring, and deleting
 * previously-persisted audit forms.
 */

import React, { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { useAuditAgent } from "@/hooks/use-audit-agent";
import type { AuditFormPayload, SavedFormSummary } from "@/hooks/use-audit-agent";
import { A2UIRenderer } from "@/components/a2ui-renderer";
import { getGridSpan } from "@/lib/layout-engine";
import {
  Loader2,
  FileText,
  ChevronRight,
  BarChart3,
  ClipboardCheck,
  FolderArchive,
  BookmarkCheck,
  ChevronDown as ChevronDownIcon,
  RotateCcw,
  Shield,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useFlaggedHits } from "@/hooks/use-flagged-hits";
import { FlaggedHitsInline } from "@/components/doc-lens/flagged-hits-panel";
import { DocumentViewerSheet } from "@/components/document-viewer-sheet";

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
  const flaggedHits = useFlaggedHits();
  const [previewDoc, setPreviewDoc] = useState<{
    file_name: string;
    _initialPage?: number;
  } | null>(null);

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
            onPreviewDoc={(fileName, page) => {
              setPreviewDoc({ file_name: fileName, _initialPage: page });
            }}
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
            if (!isOpen) setPreviewDoc(null);
          }}
          initialPage={previewDoc._initialPage}
        />
      )}
    </>
  );
}

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
  const outputComponents = componentsByZone.output || [];
  const [showSavedForms, setShowSavedForms] = useState(false);
  const [showSavedImages, setShowSavedImages] = useState(false);
  const flaggedHits = useFlaggedHits();

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

  const headerBar = (
    <div className="flex items-center gap-2 px-4 py-2 pr-10 border-b border-border/50 shrink-0">
      <ClipboardCheck className="h-4 w-4 text-primary" />
      <h2 className="text-sm font-semibold text-foreground">Output</h2>
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
          <Badge
            variant="outline"
            className="text-[10px] text-muted-foreground border-border/40"
          >
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
        <Button
          variant={showSavedImages ? "secondary" : "outline"}
          size="sm"
          onClick={toggleSavedImages}
          className="h-7 gap-1.5 text-xs"
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
            className={`h-3 w-3 transition-transform duration-200 ${showSavedImages ? "rotate-180" : ""}`}
          />
        </Button>
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

        <div className="flex flex-col items-center justify-center flex-1 text-center px-8">
          <div className="relative mb-8">
            <div className="absolute inset-0 bg-primary/20 rounded-full blur-3xl scale-150" />
            <div className="relative h-20 w-20 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
              <ClipboardCheck className="h-10 w-10 text-primary/80" />
            </div>
          </div>
          <h3 className="text-xl font-semibold text-foreground/80 mb-2">
            No Generated Content (yet)
          </h3>
          <p className="text-sm text-muted-foreground max-w-md leading-relaxed">
            Explore documents, analyze claim files, and jump start your audit.
          </p>
          <div className="flex items-center gap-8 mt-10 text-muted-foreground/40">
            <div className="flex flex-col items-center gap-2">
              <div className="h-12 w-12 rounded-xl bg-secondary/60 border border-border/30 flex items-center justify-center">
                <BarChart3 className="h-6 w-6" />
              </div>
              <span className="text-[11px] font-medium">Analysis</span>
            </div>
            <ChevronRight className="h-4 w-4 mt-[-16px]" />
            <div className="flex flex-col items-center gap-2">
              <div className="h-12 w-12 rounded-xl bg-secondary/60 border border-border/30 flex items-center justify-center">
                <FileText className="h-6 w-6" />
              </div>
              <span className="text-[11px] font-medium">Documents</span>
            </div>
            <ChevronRight className="h-4 w-4 mt-[-16px]" />
            <div className="flex flex-col items-center gap-2">
              <div className="h-12 w-12 rounded-xl bg-secondary/60 border border-border/30 flex items-center justify-center">
                <ClipboardCheck className="h-6 w-6" />
              </div>
              <span className="text-[11px] font-medium">Questionnaires</span>
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

      <SavedImagesPanel
        open={showSavedImages}
        onToggle={toggleSavedImages}
      />

      <ScrollArea className="flex-1">
        <div className="px-5 py-5">
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
      </ScrollArea>
    </div>
  );
}
