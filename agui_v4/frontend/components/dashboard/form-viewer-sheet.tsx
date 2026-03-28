"use client";

/**
 * FormViewerSheet — read-only slide-in panel for viewing a single audit form.
 *
 * Renders peril, outcome, all questions with answer pills, sub-questions with
 * reasoning / citations, and read-only help text. Everything is non-editable.
 */

import React, { useState, useCallback } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  ChevronDown,
  ChevronRight,
  Copy,
  ClipboardCheck,
  Shield,
  CheckCircle2,
  AlertTriangle,
  FileWarning,
} from "lucide-react";
import type { SavedForm, TFRQuestion, SubQuestion } from "@/lib/dashboard-types";

function normalizeSubAnswer(answer: unknown): boolean {
  return answer === true;
}

// ── Copy button ──────────────────────────────────────────────────

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
      className="shrink-0 p-1 rounded text-muted-foreground/50 hover:text-foreground transition-colors"
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

// ── Answer pill (read-only) ──────────────────────────────────────

function AnswerBadge({ answer }: { answer: string }) {
  if (answer === "Yes") {
    return (
      <Badge className="bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 text-xs">
        Yes
      </Badge>
    );
  }
  if (answer === "No") {
    return (
      <Badge className="bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30 text-xs">
        No
      </Badge>
    );
  }
  return (
    <Badge className="bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30 text-xs">
      Insufficient
    </Badge>
  );
}

function SubQuestionBadge({ answer }: { answer: unknown }) {
  const isApplicable = normalizeSubAnswer(answer);

  if (isApplicable) {
    return (
      <Badge className="bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30 text-xs">
        Applicable
      </Badge>
    );
  }

  return (
    <Badge className="bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 text-xs">
      Not Applicable
    </Badge>
  );
}

// ── Sub-question viewer ──────────────────────────────────────────

function SubQuestionViewer({ sub }: { sub: SubQuestion }) {
  return (
    <div className="pl-8 pr-4 py-3 bg-secondary/15 border-l-2 border-l-red-500/30">
      <div className="flex items-start gap-3">
        <span className="shrink-0 text-xs font-mono font-semibold text-primary mt-0.5 min-w-[52px]">
          {sub.id}
        </span>
        <div className="flex-1 min-w-0 space-y-2">
          <p className="text-sm text-foreground/90 leading-relaxed">
            {sub.text}
          </p>
          {sub.help_text && (
            <p className="text-sm italic text-muted-foreground">
              {sub.help_text}
            </p>
          )}

          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground uppercase tracking-wider">
              Applicability:
            </span>
            <SubQuestionBadge answer={sub.answer} />
          </div>

          {sub.reasoning && (
            <div>
              <div className="flex items-center gap-1">
                <span className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wider">
                  Reasoning
                </span>
                <CopyBtn text={sub.reasoning} />
              </div>
              <p className="text-sm text-foreground/80 bg-background/60 border border-border/40 rounded-md px-3 py-2 mt-0.5 leading-relaxed">
                {sub.reasoning}
              </p>
            </div>
          )}

          {sub.citations && (
            <div>
              <div className="flex items-center gap-1">
                <span className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wider">
                  Citations
                </span>
                <CopyBtn text={sub.citations} />
              </div>
              <p className="text-sm text-foreground/80 bg-background/60 border border-border/40 rounded-md px-3 py-2 mt-0.5 leading-relaxed">
                {sub.citations}
              </p>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

// ── Question viewer ──────────────────────────────────────────────

function QuestionViewer({ question }: { question: TFRQuestion }) {
  const hasSubs =
    !!question.sub_questions && question.sub_questions.length > 0;
  const isNo = question.answer === "No";
  const [expanded, setExpanded] = useState(isNo && hasSubs);

  return (
    <div className="border border-border/40 rounded-xl overflow-hidden bg-secondary/5">
      {/* Question header */}
      <div
        className={`flex items-start gap-3 p-4 ${hasSubs ? "cursor-pointer" : ""}`}
        onClick={() => hasSubs && setExpanded((p) => !p)}
      >
        <div className="shrink-0 mt-0.5 text-muted-foreground w-4">
          {hasSubs &&
            (expanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            ))}
        </div>
        <span className="shrink-0 text-xs font-mono font-semibold text-primary mt-0.5 min-w-[52px]">
          {question.id}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-foreground leading-relaxed">
            {question.text}
          </p>
          {question.help_text && (
            <p className="mt-1 text-sm italic text-muted-foreground">
              {question.help_text}
            </p>
          )}
        </div>
        <div className="shrink-0 ml-2">
          <AnswerBadge answer={question.answer} />
        </div>
      </div>

      {/* Missing info */}
      {question.answer === "Insufficient information" &&
        question.missing_info && (
          <div className="px-4 pb-3 ml-[76px]">
            <div className="flex items-center gap-1 text-amber-700 dark:text-amber-400">
              <FileWarning className="h-3 w-3" />
              <span className="text-xs font-medium uppercase tracking-wider">
                Missing Information
              </span>
            </div>
            <p className="text-sm text-foreground/80 bg-amber-500/5 border border-amber-500/20 rounded-md px-3 py-2 mt-1">
              {question.missing_info}
            </p>
          </div>
        )}

      {/* Driver count */}
      {isNo && hasSubs && (
        <div className="px-4 py-2 border-t flex items-center gap-2 text-sm bg-red-500/5 border-red-500/15 text-red-700 dark:text-red-400">
          <AlertTriangle className="h-3.5 w-3.5" />
          {question.sub_questions!.filter((s) => normalizeSubAnswer(s.answer))
            .length}{" "}
          driver(s) identified
        </div>
      )}

      {/* Sub-questions */}
      {expanded && hasSubs && (
        <div className="border-t border-border/25 divide-y divide-border/15">
          {question.sub_questions!.map((sq) => (
            <SubQuestionViewer key={sq.id} sub={sq} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Text section viewer ──────────────────────────────────────────

function TextSection({
  label,
  text,
}: {
  label: string;
  text: string | null | undefined;
}) {
  if (!text) return null;

  return (
    <div className="border border-border/40 rounded-xl p-4 bg-secondary/5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-foreground/80 uppercase tracking-wider">
          {label}
        </span>
        <CopyBtn text={text} />
      </div>
      <p className="text-sm text-foreground/85 leading-relaxed mt-2 whitespace-pre-wrap">
        {text}
      </p>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────

export interface FormViewerSheetProps {
  form: SavedForm | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function FormViewerSheet({
  form,
  open,
  onOpenChange,
}: FormViewerSheetProps) {
  if (!form) return null;

  const questions = form.questions ?? [];
  const yesCount = questions.filter((q) => q.answer === "Yes").length;
  const noCount = questions.filter((q) => q.answer === "No").length;
  const insufficientCount = questions.filter(
    (q) => q.answer === "Insufficient information"
  ).length;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full sm:max-w-2xl lg:max-w-3xl p-0"
      >
        <SheetHeader className="px-6 pt-6 pb-4 border-b border-border/40 bg-secondary/20">
          <div className="flex items-center gap-3 flex-wrap">
            <Shield className="h-5 w-5 text-primary" />
            <SheetTitle className="text-lg">
              {form.title || "Untitled Form"}
            </SheetTitle>
          </div>

          <SheetDescription className="sr-only">
            Read-only view of audit form {form.title}
          </SheetDescription>

          {/* Peril + outcome badges */}
          <div className="flex items-center gap-2 flex-wrap mt-1">
            <Badge
              variant="outline"
              className={
                form.peril?.peril === "Exterior"
                  ? "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/30"
                  : "bg-orange-500/15 text-orange-700 dark:text-orange-400 border-orange-500/30"
              }
            >
              {form.peril?.peril ?? "Unknown"} Peril
            </Badge>

            <Badge
              className={
                form.overall_outcome === "Meets"
                  ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30"
                  : "bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30"
              }
            >
              {form.overall_outcome === "Meets" ? (
                <CheckCircle2 className="h-3 w-3 mr-1" />
              ) : (
                <AlertTriangle className="h-3 w-3 mr-1" />
              )}
              {form.overall_outcome}
            </Badge>
          </div>

          {/* Counts */}
          <div className="flex items-center gap-1.5 mt-2">
            <span className="text-xs text-muted-foreground">
              {questions.length} questions
            </span>
            {yesCount > 0 && (
              <Badge className="bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 text-[10px]">
                {yesCount} Yes
              </Badge>
            )}
            {noCount > 0 && (
              <Badge className="bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30 text-[10px]">
                {noCount} No
              </Badge>
            )}
            {insufficientCount > 0 && (
              <Badge className="bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30 text-[10px]">
                {insufficientCount} Insuff.
              </Badge>
            )}
          </div>
        </SheetHeader>

        <ScrollArea className="flex-1 h-[calc(100vh-220px)]">
          <div className="px-6 py-5 space-y-4">
            {/* Questions */}
            {questions.map((q) => (
              <QuestionViewer key={q.id} question={q} />
            ))}

            {questions.length > 0 && <Separator className="my-4" />}

            {/* Supplementary sections */}
            <TextSection
              label="Outcome Justification"
              text={form.outcome_justification}
            />

            {/* Peril notes */}
            {form.peril?.notes && (
              <TextSection label="Peril Notes" text={form.peril.notes} />
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

export default FormViewerSheet;
