"use client";

/**
 * AuditQuestionForm Component (TFR Schema)
 *
 * Displays a TFR (Technical File Review) analysis with:
 *   - Peril determination (Interior / Exterior)
 *   - Questions with pre-populated answers (Yes / No / Insufficient information)
 *   - Sub-questions with editable reasoning and citations
 *   - Missing info field for "Insufficient information" answers
 *   - Overall outcome with editable justification
 *   - Optional read-only help text for questions and sub-questions
 *
 * The LLM pre-populates the form; reviewers can refine answers, reasoning,
 * citations, and the overall outcome justification.
 */

import React, {
  useState,
  useCallback,
  useMemo,
  useEffect,
  useRef,
} from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Copy,
  ClipboardCheck,
  ClipboardList,
  AlertTriangle,
  CheckCircle2,
  FileWarning,
  Save,
  Loader2,
  CheckCheck,
  X,
  XCircle,
  Pencil,
  Trash2,
  Ban,
} from "lucide-react";
import type { AuditFormPayload } from "@/hooks/use-audit-agent";

// ── Types ────────────────────────────────────────────────────────

type AnswerValue = "Yes" | "No" | "Insufficient information";
type SubAnswerValue = boolean;

interface SubQuestion {
  id: string;
  text: string;
  reasoning: string;
  citations: string;
  answer?: SubAnswerValue;
  help_text?: string | null;
}

interface TFRQuestion {
  id: string;
  text: string;
  answer: AnswerValue;
  sub_questions?: SubQuestion[] | null;
  missing_info?: string | null;
  help_text?: string | null;
}

interface PerilDetermination {
  peril: "Interior" | "Exterior";
  notes?: string | null;
}

export interface AuditQuestionFormProps {
  peril: PerilDetermination;
  questions: TFRQuestion[];
  overall_outcome: string;
  outcome_justification: string;
  onSubmit?: (formData: AuditFormPayload, title?: string) => Promise<void>;
  onCancel?: () => void;
  onClose?: () => void;
  onDelete?: () => Promise<void>;
  isSaving?: boolean;
  currentFormId?: string | null;
}

/**
 * Normalize sub-question answers into the internal boolean form.
 *
 * Args:
 *   answer: Persisted answer value.
 *
 * Returns:
 *   ``true`` when the sub-question is selected, otherwise ``false``.
 */
function normalizeSubAnswer(answer: unknown): SubAnswerValue {
  return answer === true;
}

// ── Copy-to-clipboard button ─────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard API may fail in insecure contexts */
    }
  }, [text]);

  return (
    <button
      onClick={handleCopy}
      className="shrink-0 p-1 rounded text-muted-foreground/50 hover:text-accent transition-colors"
      title="Copy to clipboard"
      type="button"
    >
      {copied ? (
        <ClipboardCheck className="h-3.5 w-3.5 text-success" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </button>
  );
}

/** Auto-resizing textarea that grows to fit content. */
function AutoResizeTextarea(
  props: React.TextareaHTMLAttributes<HTMLTextAreaElement>
): React.ReactElement {
  const { value, onChange, className, ...rest } = props;
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const adjustHeight = useCallback(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${element.scrollHeight}px`;
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange?.(event);
      adjustHeight();
    },
    [onChange, adjustHeight]
  );

  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={handleChange}
      className={`${className || ""} resize-none overflow-hidden`}
      {...rest}
    />
  );
}

// ── Answer Pills ─────────────────────────────────────────────────

const ANSWER_OPTIONS: {
  value: AnswerValue;
  label: string;
  icon: React.ReactNode;
  iconSm: React.ReactNode;
  active: string;
  idle: string;
}[] = [
  {
    value: "Yes",
    label: "Yes",
    icon: <CheckCircle2 className="h-3.5 w-3.5" />,
    iconSm: <CheckCircle2 className="h-3 w-3" />,
    active:
      "bg-emerald-600 text-white shadow-sm ring-2 ring-emerald-600/30 ring-offset-1 ring-offset-background",
    idle: "text-emerald-700 dark:text-emerald-400 border-emerald-500/30 dark:border-emerald-500/25 hover:bg-emerald-500/15 dark:hover:bg-emerald-500/20",
  },
  {
    value: "No",
    label: "No",
    icon: <XCircle className="h-3.5 w-3.5" />,
    iconSm: <XCircle className="h-3 w-3" />,
    active:
      "bg-red-600 text-white shadow-sm ring-2 ring-red-600/30 ring-offset-1 ring-offset-background",
    idle: "text-red-700 dark:text-red-400 border-red-500/30 dark:border-red-500/25 hover:bg-red-500/15 dark:hover:bg-red-500/20",
  },
  {
    value: "Insufficient information",
    label: "Insufficient",
    icon: <AlertTriangle className="h-3.5 w-3.5" />,
    iconSm: <AlertTriangle className="h-3 w-3" />,
    active:
      "bg-amber-600 text-white shadow-sm ring-2 ring-amber-600/30 ring-offset-1 ring-offset-background",
    idle: "text-amber-700 dark:text-amber-400 border-amber-500/30 dark:border-amber-500/25 hover:bg-amber-500/15 dark:hover:bg-amber-500/20",
  },
];

function AnswerPills({
  value,
  onChange,
  size = "default",
  disableYes = false,
  yesDisabledTitle,
}: {
  value: AnswerValue | null;
  onChange: (v: AnswerValue) => void;
  size?: "default" | "sm";
  disableYes?: boolean;
  yesDisabledTitle?: string;
}) {
  const pillClass =
    size === "sm"
      ? "px-2.5 py-1 text-xs gap-1"
      : "px-3.5 py-1.5 text-sm gap-1.5";

  return (
    <div className="flex items-center gap-1.5">
      {ANSWER_OPTIONS.map((opt) => {
        const isActive = value === opt.value;
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            disabled={disableYes && opt.value === "Yes"}
            title={
              disableYes && opt.value === "Yes" ? yesDisabledTitle : undefined
            }
            type="button"
            className={`${pillClass} inline-flex items-center rounded-lg font-semibold border transition-all duration-150 hover:scale-[1.03] active:scale-[0.97] disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 ${
              isActive
                ? `${opt.active} border-transparent`
                : `${opt.idle} bg-transparent`
            }`}
          >
            {size === "sm" ? opt.iconSm : opt.icon}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function SubQuestionApplicabilitySelector({
  value,
  onChange,
}: {
  value: SubAnswerValue;
  onChange: (value: SubAnswerValue) => void;
}) {
  const nextValue = !value;

  // Applies = flagged/checked (red), N/A = neutral/muted
  const buttonClass = value
    ? "border-red-500/40 bg-red-500/10 text-red-700 shadow-[0_0_6px_rgba(239,68,68,0.15)] dark:text-red-300 hover:bg-red-500/15"
    : "border-border/60 bg-muted/40 text-muted-foreground hover:border-primary/25 hover:bg-muted/60";

  return (
    <button
      type="button"
      onClick={() => onChange(nextValue)}
      title={value ? "Marked as applicable — click to dismiss" : "Not applicable — click to flag"}
      className={`mt-2 inline-flex items-center justify-center rounded-full border p-1.5
        transition-all duration-200 ease-in-out active:scale-90 ${buttonClass}`}
    >
      <span className="relative flex h-4 w-4 items-center justify-center">
        {/* Cross-fade between the two icons */}
        <CheckCheck
          className={`absolute inset-0 h-4 w-4 transition-all duration-200
            ${value ? "scale-100 opacity-100" : "scale-75 opacity-0"}`}
        />
        <Ban
          className={`absolute inset-0 h-4 w-4 transition-all duration-200
            ${value ? "scale-75 opacity-0" : "scale-100 opacity-100"}`}
        />
      </span>
    </button>
  );
}

// ── Sub-question Row ─────────────────────────────────────────────

function SubQuestionRow({
  sub,
  expanded,
  onToggleExpanded,
  onAnswerChange,
  onReasoningChange,
  onCitationsChange,
}: {
  sub: SubQuestion;
  expanded: boolean;
  onToggleExpanded: () => void;
  onAnswerChange: (answer: SubAnswerValue) => void;
  onReasoningChange: (reasoning: string) => void;
  onCitationsChange: (citations: string) => void;
}) {
  const subAnswer = normalizeSubAnswer(sub.answer);
  const borderColor =
    subAnswer
      ? "border-l-red-500/60 dark:border-l-red-500/40"
      : "border-l-emerald-500/60 dark:border-l-emerald-500/40";

  return (
    <div className={`border-l-[3px] ${borderColor} transition-colors`}>
      <button
        type="button"
        onClick={onToggleExpanded}
        className="flex w-full items-start gap-4 px-5 py-4 text-left"
      >
        {expanded ? (
          <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        )}
        <span className="shrink-0 text-[10px] font-mono font-bold text-primary bg-primary/8 dark:bg-primary/12 border border-primary/10 dark:border-primary/8 rounded px-1.5 py-0.5 mt-0.5">
          {sub.id}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-base text-foreground/90 leading-relaxed">
            {sub.text}
          </p>
          {sub.help_text && (
            <p className="mt-1 text-sm italic text-muted-foreground">
              {sub.help_text}
            </p>
          )}
        </div>
      </button>

      {expanded && (
        <div className="px-10 pb-4 pr-5">
          <div className="mt-1">
            <label className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wider">
            </label>
            <SubQuestionApplicabilitySelector
              value={subAnswer}
              onChange={onAnswerChange}
            />
          </div>

          <div className="mt-3">
            <label className="text-xs font-medium text-muted-foreground/70 uppercase tracking-wider">
              Reasoning
            </label>
            <div className="flex items-start gap-1 mt-1">
              <AutoResizeTextarea
                value={sub.reasoning}
                onChange={(e) => onReasoningChange(e.target.value)}
                placeholder="Explain the reasoning..."
                rows={2}
                className="flex-1 text-sm bg-card dark:bg-background/60 border border-border/80 dark:border-border/25 rounded-lg px-3 py-2 text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/60 min-h-[52px] leading-relaxed shadow-2xs"
              />
              <CopyButton text={sub.reasoning} />
            </div>
          </div>

          <div className="mt-2">
            <label className="text-xs font-medium text-muted-foreground/70 uppercase tracking-wider">
              Citations
            </label>
            <div className="flex items-start gap-1 mt-1">
              <AutoResizeTextarea
                value={sub.citations}
                onChange={(e) => onCitationsChange(e.target.value)}
                placeholder="Reference specific evidence..."
                rows={1}
                className="flex-1 text-sm bg-card dark:bg-background/60 border border-border/80 dark:border-border/25 rounded-lg px-3 py-2 text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/60 min-h-[44px] leading-relaxed shadow-2xs"
              />
              <CopyButton text={sub.citations} />
            </div>
          </div>

        </div>
      )}
    </div>
  );
}

// ── Question Row ─────────────────────────────────────────────────

function QuestionRow({
  question,
  onAnswerChange,
  onMissingInfoChange,
  onSubAnswerChange,
  onSubReasoningChange,
  onSubCitationsChange,
}: {
  question: TFRQuestion;
  onAnswerChange: (answer: AnswerValue) => void;
  onMissingInfoChange: (info: string) => void;
  onSubAnswerChange: (subId: string, answer: SubAnswerValue) => void;
  onSubReasoningChange: (subId: string, reasoning: string) => void;
  onSubCitationsChange: (subId: string, citations: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [expandedSubQuestionIds, setExpandedSubQuestionIds] = useState<Set<string>>(
    () => new Set()
  );

  const hasSubs =
    !!question.sub_questions && question.sub_questions.length > 0;
  const isNo = question.answer === "No";
  const isInsufficient = question.answer === "Insufficient information";
  const allSubsNotApplicable = useMemo(() => {
    if (!hasSubs) return true;
    return question.sub_questions!.every(
      (sub) => !normalizeSubAnswer(sub.answer)
    );
  }, [hasSubs, question.sub_questions]);
  const applicableDriverCount = useMemo(() => {
    if (!hasSubs) return 0;
    return question.sub_questions!.filter(
      (sub) => normalizeSubAnswer(sub.answer)
    ).length;
  }, [hasSubs, question.sub_questions]);

  useEffect(() => {
    if (isNo && hasSubs) {
      setExpanded(true);
    }
  }, [isNo, hasSubs]);

  useEffect(() => {
    if (!hasSubs) {
      setExpandedSubQuestionIds(new Set());
      return;
    }

    setExpandedSubQuestionIds((prev) => {
      const validIds = new Set(question.sub_questions!.map((sub) => sub.id));
      const next = new Set(
        Array.from(prev).filter((subId) => validIds.has(subId))
      );
      return next.size === prev.size ? prev : next;
    });
  }, [hasSubs, question.sub_questions]);

  const subsToShow = useMemo(() => {
    if (!hasSubs || !expanded) return [];
    return question.sub_questions!;
  }, [expanded, hasSubs, question.sub_questions]);

  const showSubSection = subsToShow.length > 0;

  const subCount = hasSubs ? question.sub_questions!.length : 0;
  const allSubQuestionsExpanded =
    subCount > 0 && expandedSubQuestionIds.size === subCount;

  const answerAccent =
    question.answer === "Yes"
      ? "border-l-emerald-500/70 dark:border-l-emerald-500/50"
      : question.answer === "No"
        ? "border-l-red-500/70 dark:border-l-red-500/50"
        : question.answer === "Insufficient information"
          ? "border-l-amber-500/70 dark:border-l-amber-500/50"
          : "border-l-transparent";

  return (
    <div className={`border border-border/60 dark:border-border/25 border-l-[3px] ${answerAccent} rounded-xl overflow-hidden bg-card dark:bg-card/90 shadow-xs hover:shadow-sm transition-all`}>
      {/* Main question row */}
      <div className="flex items-start gap-4 p-5">
        <span className="shrink-0 text-[11px] font-mono font-bold text-primary bg-primary/8 dark:bg-primary/12 border border-primary/12 dark:border-primary/8 rounded-md px-2 py-0.5 mt-0.5">
          {question.id}
        </span>

        <div className="flex-1 min-w-0">
          <p className="text-base text-foreground leading-relaxed font-normal">
            {question.text}
          </p>
          {question.help_text && (
            <p className="mt-1 text-sm italic text-muted-foreground">
              {question.help_text}
            </p>
          )}

          {isInsufficient && (
            <div className="mt-3">
              <label className="text-xs font-medium text-amber-700 dark:text-amber-400 uppercase tracking-wider flex items-center gap-1">
                <FileWarning className="h-3 w-3" />
                Missing Information
              </label>
              <div className="flex items-start gap-1 mt-0.5">
                <AutoResizeTextarea
                  value={question.missing_info || ""}
                  onChange={(e) => onMissingInfoChange(e.target.value)}
                  placeholder="What information is needed to make a determination?"
                  rows={2}
                  className="flex-1 text-sm bg-amber-500/5 dark:bg-amber-500/8 border border-amber-500/40 dark:border-amber-500/25 rounded-lg px-3 py-2 text-foreground/95 placeholder:text-muted-foreground/55 focus:outline-none focus:ring-1 focus:ring-amber-500/50 min-h-[52px] leading-relaxed shadow-2xs"
                />
                <CopyButton text={question.missing_info || ""} />
              </div>
            </div>
          )}

          {/* Sub-question toggle pill */}
          {hasSubs && (
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setExpanded((prev) => !prev)}
                className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold
                  bg-secondary/60 dark:bg-secondary/25 hover:bg-secondary/90 dark:hover:bg-secondary/40
                  border border-border dark:border-border/40 transition-all shadow-sm
                  active:scale-[0.97]"
              >
                {expanded ? (
                  <ChevronDown className="h-3.5 w-3.5 text-foreground/60" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5 text-foreground/60" />
                )}
                <span className="text-foreground/80 dark:text-foreground/70">
                  {subCount} sub-question{subCount !== 1 ? "s" : ""}
                </span>
              </button>
              {expanded && (
                <button
                  type="button"
                  onClick={() =>
                    setExpandedSubQuestionIds(
                      allSubQuestionsExpanded
                        ? new Set()
                        : new Set(question.sub_questions!.map((sub) => sub.id))
                    )
                  }
                  className="inline-flex items-center rounded-md border border-border dark:border-border/40
                    px-2.5 py-1.5 text-xs font-semibold text-foreground/70
                    hover:bg-secondary/60 dark:hover:bg-secondary/25 transition-all shadow-sm
                    active:scale-[0.97]"
                >
                  {allSubQuestionsExpanded ? "Hide all" : "Show all"}
                </button>
              )}
            </div>
          )}
        </div>

        <div className="shrink-0 ml-3">
          <AnswerPills
            value={question.answer}
            onChange={onAnswerChange}
            disableYes={hasSubs && !allSubsNotApplicable}
            yesDisabledTitle="Mark all sub-questions as not applicable before setting this question to Yes."
          />
        </div>
      </div>

      {/* Driver validation banner for "No" answers */}
      {isNo && hasSubs && (
        <div className="px-5 py-2 border-t border-red-500/25 dark:border-red-500/15 flex items-center gap-2 text-sm bg-red-500/10 dark:bg-red-500/12 text-red-700 dark:text-red-400">
          <AlertTriangle className="h-3.5 w-3.5" />
          {applicableDriverCount} driver
          {applicableDriverCount !== 1 ? "s" : ""} identified
        </div>
      )}

      {/* Sub-questions (collapsible) */}
      {showSubSection && (
        <div className="border-t border-border/50 dark:border-border/20 bg-secondary/40 dark:bg-secondary/12 shadow-[inset_0_2px_6px_-2px_oklch(0_0_0/0.06)] dark:shadow-[inset_0_2px_6px_-2px_oklch(0_0_0/0.15)] divide-y divide-border/40 dark:divide-border/15">
          {subsToShow.map((sub) => (
            <SubQuestionRow
              key={sub.id}
              sub={sub}
              expanded={expandedSubQuestionIds.has(sub.id)}
              onToggleExpanded={() =>
                setExpandedSubQuestionIds((prev) => {
                  const next = new Set(prev);
                  if (next.has(sub.id)) {
                    next.delete(sub.id);
                  } else {
                    next.add(sub.id);
                  }
                  return next;
                })
              }
              onAnswerChange={(a) => onSubAnswerChange(sub.id, a)}
              onReasoningChange={(r) => onSubReasoningChange(sub.id, r)}
              onCitationsChange={(c) => onSubCitationsChange(sub.id, c)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Form ────────────────────────────────────────────────────

export function AuditQuestionForm({
  peril: initialPeril,
  questions: initialQuestions,
  overall_outcome: initialOutcome,
  outcome_justification: initialJustification,
  onSubmit,
  onCancel,
  onClose,
  onDelete,
  isSaving = false,
  currentFormId,
}: AuditQuestionFormProps): React.ReactElement {
  const [peril, setPeril] = useState(initialPeril);
  const [isFullyCollapsed, setIsFullyCollapsed] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<"idle" | "success">("idle");
  const [formTitle, setFormTitle] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);
  const normalizedInitialQuestions = useMemo<TFRQuestion[]>(
    () =>
      initialQuestions.map((question) => ({
        ...question,
        sub_questions: question.sub_questions?.map((sub) => ({
          ...sub,
          answer: normalizeSubAnswer(sub.answer),
          help_text: sub.help_text ?? "",
        })),
      })),
    [initialQuestions]
  );
  const [questions, setQuestions] = useState<TFRQuestion[]>(
    normalizedInitialQuestions
  );
  const [overallOutcome, setOverallOutcome] = useState(initialOutcome);
  const [outcomeJustification, setOutcomeJustification] =
    useState(initialJustification);

  // ── Question-level handlers ──────────────────────────────────

  const handleAnswerChange = useCallback(
    (qId: string, answer: AnswerValue) => {
      setQuestions((prev) =>
        prev.map((q) => {
          if (q.id !== qId) return q;
          if (
            answer === "Yes" &&
            q.sub_questions &&
            q.sub_questions.length > 0
          ) {
            const allSubsNotApplicable = q.sub_questions.every(
              (s) => !normalizeSubAnswer(s.answer)
            );
            if (!allSubsNotApplicable) return q;
          }
          return { ...q, answer };
        })
      );
    },
    []
  );

  const handleMissingInfoChange = useCallback(
    (qId: string, missing_info: string) => {
      setQuestions((prev) =>
        prev.map((q) => (q.id === qId ? { ...q, missing_info } : q))
      );
    },
    []
  );

  // ── Sub-question handlers ────────────────────────────────────

  const handleSubAnswerChange = useCallback(
    (qId: string, subId: string, answer: SubAnswerValue) => {
      setQuestions((prev) =>
        prev.map((q) =>
          q.id === qId
            ? {
                ...q,
                sub_questions: q.sub_questions?.map((s) =>
                  s.id === subId ? { ...s, answer } : s
                ),
              }
            : q
        )
      );
    },
    []
  );

  const handleSubReasoningChange = useCallback(
    (qId: string, subId: string, reasoning: string) => {
      setQuestions((prev) =>
        prev.map((q) =>
          q.id === qId
            ? {
                ...q,
                sub_questions: q.sub_questions?.map((s) =>
                  s.id === subId ? { ...s, reasoning } : s
                ),
              }
            : q
        )
      );
    },
    []
  );

  const handleSubCitationsChange = useCallback(
    (qId: string, subId: string, citations: string) => {
      setQuestions((prev) =>
        prev.map((q) =>
          q.id === qId
            ? {
                ...q,
                sub_questions: q.sub_questions?.map((s) =>
                  s.id === subId ? { ...s, citations } : s
                ),
              }
            : q
        )
      );
    },
    []
  );

  // ── Form submission ─────────────────────────────────────────

  const collectFormState = useCallback(
    (): AuditFormPayload => ({
      peril: peril as unknown as Record<string, unknown>,
      questions: questions as unknown as Array<Record<string, unknown>>,
      overall_outcome: overallOutcome,
      outcome_justification: outcomeJustification,
    }),
    [peril, questions, overallOutcome, outcomeJustification]
  );

  const handleSubmit = useCallback(async () => {
    if (!onSubmit) return;
    try {
      await onSubmit(collectFormState(), formTitle.trim() || undefined);
      setSubmitStatus("success");
      setTimeout(() => setSubmitStatus("idle"), 2500);
    } catch (err) {
      console.error("Form submit failed:", err);
    }
  }, [onSubmit, collectFormState, formTitle]);

  const handleCancel = useCallback(() => {
    setPeril(initialPeril);
    setQuestions(normalizedInitialQuestions);
    setOverallOutcome(initialOutcome);
    setOutcomeJustification(initialJustification);
    setFormTitle("");
    setSubmitStatus("idle");
    onCancel?.();
  }, [
    initialPeril,
    normalizedInitialQuestions,
    initialOutcome,
    initialJustification,
    onCancel,
  ]);

  const handleDelete = useCallback(async () => {
    if (!onDelete) return;
    setIsDeleting(true);
    try {
      await onDelete();
    } catch (err) {
      console.error("Form delete failed:", err);
    } finally {
      setIsDeleting(false);
    }
  }, [onDelete]);

  // ── Dirty tracking ─────────────────────────────────────────
  const isDirty = useMemo(() => {
    if (
      peril.peril !== initialPeril.peril ||
      peril.notes !== initialPeril.notes
    )
      return true;
    if (overallOutcome !== initialOutcome) return true;
    if (outcomeJustification !== initialJustification) return true;
    if (
      JSON.stringify(questions) !==
      JSON.stringify(normalizedInitialQuestions)
    )
      return true;
    return false;
  }, [
    peril,
    initialPeril,
    overallOutcome,
    initialOutcome,
    outcomeJustification,
    initialJustification,
    questions,
    normalizedInitialQuestions,
  ]);

  // ── Summary counts ───────────────────────────────────────────

  const yesCount = questions.filter((q) => q.answer === "Yes").length;
  const noCount = questions.filter((q) => q.answer === "No").length;
  const insufficientCount = questions.filter(
    (q) => q.answer === "Insufficient information"
  ).length;
  const driverCount = questions.reduce(
    (count, question) =>
      count +
      (question.sub_questions?.filter((sub) => normalizeSubAnswer(sub.answer))
        .length || 0),
    0
  );

  return (
    <Card className="border-border/60 dark:border-border/20 bg-linear-to-br from-card to-secondary/15 dark:from-card dark:to-secondary/8 shadow-sm">
      <CardHeader className="pb-4">
        {/* Peril + outcome header */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <ClipboardList className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-semibold text-foreground">
                TFR Questionnaire
              </h2>
            </div>
            <button
              type="button"
              onClick={() =>
                setPeril((prev) => ({
                  ...prev,
                  peril:
                    prev.peril === "Interior" ? "Exterior" : "Interior",
                }))
              }
              title="Click to toggle peril"
            >
              <Badge
                variant="outline"
                className={`text-sm font-semibold cursor-pointer transition-colors ${
                  peril.peril === "Exterior"
                    ? "bg-blue-500/20 text-blue-700 dark:text-blue-400 border-blue-500/30"
                    : "bg-orange-500/20 text-orange-700 dark:text-orange-400 border-orange-500/30"
                }`}
              >
                {peril.peril}
              </Badge>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() =>
                setOverallOutcome((prev) =>
                  prev === "Meets" ? "Does Not Meet" : "Meets"
                )
              }
              title="Click to toggle outcome"
            >
              <Badge
                className={`text-sm font-semibold cursor-pointer transition-colors ${
                  overallOutcome === "Meets"
                    ? "bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 border-emerald-500/30"
                    : "bg-red-500/20 text-red-700 dark:text-red-400 border-red-500/30"
                }`}
              >
                {overallOutcome === "Meets" ? (
                  <CheckCircle2 className="h-3.5 w-3.5 mr-1" />
                ) : (
                  <AlertTriangle className="h-3.5 w-3.5 mr-1" />
                )}
                {overallOutcome}
              </Badge>
            </button>

            <button
              type="button"
              onClick={() => setIsFullyCollapsed((prev) => !prev)}
              title={isFullyCollapsed ? "Expand form" : "Collapse to compact view"}
              className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
            >
              {isFullyCollapsed ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronUp className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>

        {/* Counts bar */}
        <div className="flex items-center gap-3 mt-3">
          <p className="text-sm text-muted-foreground">
            {questions.length} questions
          </p>
          <div className="flex items-center gap-1.5">
            {yesCount > 0 && (
              <Badge className="bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 border-emerald-500/30 text-xs">
                {yesCount} Yes
              </Badge>
            )}
            {noCount > 0 && (
              <Badge className="bg-red-500/20 text-red-700 dark:text-red-400 border-red-500/30 text-xs">
                {noCount} No
              </Badge>
            )}
            {insufficientCount > 0 && (
              <Badge className="bg-amber-500/20 text-amber-700 dark:text-amber-400 border-amber-500/30 text-xs">
                {insufficientCount} Insufficient
              </Badge>
            )}
            {driverCount > 0 && (
              <Badge className="bg-rose-500/25 text-rose-700 dark:text-rose-400 border-rose-500/30 text-xs">
                {driverCount} Drivers
              </Badge>
            )}
          </div>
        </div>

        {peril.notes && (
          <p className="text-sm text-muted-foreground mt-2 italic">
            Peril notes: {peril.notes}
          </p>
        )}
      </CardHeader>

      {!isFullyCollapsed && (
      <CardContent className="space-y-4">
        {questions.map((q) => (
          <QuestionRow
            key={q.id}
            question={q}
            onAnswerChange={(a) => handleAnswerChange(q.id, a)}
            onMissingInfoChange={(info) =>
              handleMissingInfoChange(q.id, info)
            }
            onSubAnswerChange={(subId, a) =>
              handleSubAnswerChange(q.id, subId, a)
            }
            onSubReasoningChange={(subId, r) =>
              handleSubReasoningChange(q.id, subId, r)
            }
            onSubCitationsChange={(subId, c) =>
              handleSubCitationsChange(q.id, subId, c)
            }
          />
        ))}

        <Separator className="my-5" />

        {/* Outcome justification (editable) */}
        <div className="border border-border/60 dark:border-border/25 rounded-xl p-5 bg-card dark:bg-card/80 shadow-2xs">
          <label className="text-sm font-semibold text-foreground/85 uppercase tracking-wider">
            Outcome Justification
          </label>
          <div className="flex items-start gap-1 mt-1.5">
            <AutoResizeTextarea
              value={outcomeJustification}
              onChange={(e) => setOutcomeJustification(e.target.value)}
              placeholder="Justification for the overall outcome..."
              rows={3}
              className="flex-1 text-sm bg-secondary/20 dark:bg-background/50 border border-border/70 dark:border-border/20 rounded-lg px-3 py-2.5 text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/60 min-h-[80px] leading-relaxed"
            />
            <CopyButton text={outcomeJustification} />
          </div>
        </div>

        {/* Action bar */}
        {onSubmit &&
          (() => {
            const neverSaved = !currentFormId;
            const needsSave = neverSaved || isDirty;
            const busy = isSaving || isDeleting;

            return (
              <div className="rounded-xl p-5 mt-2 border border-border/60 dark:border-border/25 bg-card dark:bg-card/80 shadow-2xs">
                {/* Title input row */}
                <div className="mb-3">
                  <Input
                    type="text"
                    value={formTitle}
                    onChange={(e) => setFormTitle(e.target.value)}
                    placeholder="Form title (optional)"
                  />
                </div>

                {/* Buttons row */}
                <div className="flex items-center gap-2">
                  {onClose && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={onClose}
                      disabled={busy}
                    >
                      <XCircle className="h-3.5 w-3.5" />
                      Close
                    </Button>
                  )}

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCancel}
                    disabled={busy || !isDirty}
                  >
                    <X className="h-3.5 w-3.5" />
                    Cancel
                  </Button>

                  {onDelete && currentFormId && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleDelete}
                      disabled={busy}
                      className="text-muted-foreground hover:text-destructive hover:bg-destructive/5"
                    >
                      {isDeleting ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="h-3.5 w-3.5" />
                      )}
                      Delete
                    </Button>
                  )}

                  <div className="flex-1" />

                  {/* Status indicator */}
                  {!busy &&
                    submitStatus !== "success" &&
                    (needsSave ? (
                      <span className="inline-flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-400">
                        <Pencil className="h-3 w-3" />
                        {neverSaved ? "Not yet saved" : "Unsaved changes"}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
                        <CheckCheck className="h-3 w-3" />
                        Saved
                      </span>
                    ))}

                  {/* Submit */}
                  <Button
                    onClick={handleSubmit}
                    disabled={busy}
                    variant={
                      submitStatus === "success"
                        ? "outline"
                        : needsSave
                          ? "default"
                          : "secondary"
                    }
                    size="sm"
                    className={
                      submitStatus === "success"
                        ? "border-emerald-500/40 bg-emerald-500/20 text-emerald-700 dark:text-emerald-400"
                        : ""
                    }
                  >
                    {isSaving ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Saving...
                      </>
                    ) : submitStatus === "success" ? (
                      <>
                        <CheckCheck className="h-4 w-4" />
                        Saved
                      </>
                    ) : needsSave ? (
                      <>
                        <Save className="h-4 w-4" />
                        Submit Form
                      </>
                    ) : (
                      <>
                        <CheckCheck className="h-4 w-4" />
                        Up to Date
                      </>
                    )}
                  </Button>
                </div>
              </div>
            );
          })()}
      </CardContent>
      )}
    </Card>
  );
}

export default AuditQuestionForm;
