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
 *   - Optional additional analysis and follow-up sections
 *
 * The LLM pre-populates all fields; the reviewer can edit everything.
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
} from "lucide-react";
import type { AuditFormPayload } from "@/hooks/use-audit-agent";

// ── Types ────────────────────────────────────────────────────────

type AnswerValue = "Yes" | "No" | "Insufficient information";

interface SubQuestion {
  id: string;
  text: string;
  reasoning: string;
  citations: string;
  answer?: "Yes" | "No";
  comments?: string | null;
}

interface TFRQuestion {
  id: string;
  text: string;
  answer: AnswerValue;
  sub_questions?: SubQuestion[] | null;
  missing_info?: string | null;
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
  additional_analysis?: string | null;
  follow_ups?: string | null;
  onSubmit?: (formData: AuditFormPayload, title?: string) => Promise<void>;
  onCancel?: () => void;
  onClose?: () => void;
  onDelete?: () => Promise<void>;
  isSaving?: boolean;
  currentFormId?: string | null;
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

// ── Sub-question Row ─────────────────────────────────────────────

function SubQuestionRow({
  sub,
  onAnswerChange,
  onReasoningChange,
  onCitationsChange,
  onCommentsChange,
}: {
  sub: SubQuestion;
  onAnswerChange: (answer: "Yes" | "No") => void;
  onReasoningChange: (reasoning: string) => void;
  onCitationsChange: (citations: string) => void;
  onCommentsChange: (comments: string) => void;
}) {
  const subAnswer = sub.answer || "No";
  const borderColor =
    subAnswer === "Yes"
      ? "border-l-emerald-500/60 dark:border-l-emerald-500/40"
      : "border-l-red-500/60 dark:border-l-red-500/40";

  return (
    <div className={`flex items-start gap-4 py-4 pl-10 pr-5 border-l-[3px] ${borderColor} transition-colors`}>
      <span className="shrink-0 text-[10px] font-mono font-bold text-primary bg-primary/8 dark:bg-primary/12 border border-primary/10 dark:border-primary/8 rounded px-1.5 py-0.5 mt-0.5">
        {sub.id}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-base text-foreground/90 leading-relaxed mb-2.5">
          {sub.text}
        </p>

        {/* Sub-question answer */}
        <div className="mt-1">
          <label className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wider">
            Sub-Question Answer
          </label>
          <div className="flex items-center gap-1.5 mt-1">
            <button
              type="button"
              onClick={() => onAnswerChange("No")}
              className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm font-semibold border transition-all duration-150 hover:scale-[1.03] active:scale-[0.97] ${
                subAnswer === "No"
                  ? "bg-red-600 text-white border-transparent shadow-sm ring-2 ring-red-600/30 ring-offset-1 ring-offset-background"
                  : "border-red-500/30 dark:border-red-500/25 text-red-700 dark:text-red-400 bg-transparent hover:bg-red-500/15 dark:hover:bg-red-500/20"
              }`}
            >
              <XCircle className="h-3 w-3" />
              No
            </button>
            <button
              type="button"
              onClick={() => onAnswerChange("Yes")}
              className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm font-semibold border transition-all duration-150 hover:scale-[1.03] active:scale-[0.97] ${
                subAnswer === "Yes"
                  ? "bg-emerald-600 text-white border-transparent shadow-sm ring-2 ring-emerald-600/30 ring-offset-1 ring-offset-background"
                  : "border-emerald-500/30 dark:border-emerald-500/25 text-emerald-700 dark:text-emerald-400 bg-transparent hover:bg-emerald-500/15 dark:hover:bg-emerald-500/20"
              }`}
            >
              <CheckCircle2 className="h-3 w-3" />
              Yes
            </button>
          </div>
        </div>

        {/* Reasoning (editable) */}
        <div className="mt-2">
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

        {/* Citations (editable) */}
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

        {/* Final comments (editable) */}
        <div className="mt-2">
          <label className="text-xs font-medium text-muted-foreground/70 uppercase tracking-wider">
            Final Comments
          </label>
          <div className="flex items-start gap-1 mt-1">
            <AutoResizeTextarea
              value={sub.comments || ""}
              onChange={(e) => onCommentsChange(e.target.value)}
              placeholder="Optional final comments for this sub-question..."
              rows={2}
              className="flex-1 text-sm bg-card dark:bg-background/60 border border-border/80 dark:border-border/25 rounded-lg px-3 py-2 text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/60 min-h-[52px] leading-relaxed shadow-2xs"
            />
            <CopyButton text={sub.comments || ""} />
          </div>
        </div>
      </div>
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
  onSubCommentsChange,
}: {
  question: TFRQuestion;
  onAnswerChange: (answer: AnswerValue) => void;
  onMissingInfoChange: (info: string) => void;
  onSubAnswerChange: (subId: string, answer: "Yes" | "No") => void;
  onSubReasoningChange: (subId: string, reasoning: string) => void;
  onSubCitationsChange: (subId: string, citations: string) => void;
  onSubCommentsChange: (subId: string, comments: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const hasSubs =
    !!question.sub_questions && question.sub_questions.length > 0;
  const isNo = question.answer === "No";
  const isInsufficient = question.answer === "Insufficient information";
  const allSubsYes = useMemo(() => {
    if (!hasSubs) return true;
    return question.sub_questions!.every(
      (sub) => (sub.answer || "No") === "Yes"
    );
  }, [hasSubs, question.sub_questions]);
  const noDriverCount = useMemo(() => {
    if (!hasSubs) return 0;
    return question.sub_questions!.filter(
      (sub) => (sub.answer || "No") === "No"
    ).length;
  }, [hasSubs, question.sub_questions]);

  useEffect(() => {
    if (isNo && hasSubs) {
      setExpanded(true);
    }
  }, [isNo, hasSubs]);

  const subsToShow = useMemo(() => {
    if (!hasSubs || !expanded) return [];
    return question.sub_questions!;
  }, [expanded, hasSubs, question.sub_questions]);

  const showSubSection = subsToShow.length > 0;

  const subCount = hasSubs ? question.sub_questions!.length : 0;

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
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="mt-3 inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium bg-secondary/50 dark:bg-secondary/20 hover:bg-secondary/80 dark:hover:bg-secondary/35 border border-border/50 dark:border-border/20 transition-all shadow-2xs"
            >
              {expanded ? (
                <ChevronDown className="h-3 w-3 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-3 w-3 text-muted-foreground" />
              )}
              <span className="text-foreground/70 dark:text-foreground/60">
                {subCount} sub-question{subCount !== 1 ? "s" : ""}
              </span>
              {noDriverCount > 0 && (
                <span className="ml-0.5 inline-flex items-center rounded-full bg-red-500/15 dark:bg-red-500/25 border border-red-500/20 dark:border-red-500/15 px-1.5 py-px text-[10px] font-semibold text-red-700 dark:text-red-400">
                  {noDriverCount} driver{noDriverCount !== 1 ? "s" : ""}
                </span>
              )}
            </button>
          )}
        </div>

        <div className="shrink-0 ml-3">
          <AnswerPills
            value={question.answer}
            onChange={onAnswerChange}
            disableYes={hasSubs && !allSubsYes}
            yesDisabledTitle="Set all sub-questions to Yes before setting this question to Yes."
          />
        </div>
      </div>

      {/* Driver validation banner for "No" answers */}
      {isNo && hasSubs && (
        <div className="px-5 py-2 border-t border-red-500/25 dark:border-red-500/15 flex items-center gap-2 text-sm bg-red-500/10 dark:bg-red-500/12 text-red-700 dark:text-red-400">
          <AlertTriangle className="h-3.5 w-3.5" />
          {noDriverCount} driver{noDriverCount !== 1 ? "s" : ""} identified
        </div>
      )}

      {/* Sub-questions (collapsible) */}
      {showSubSection && (
        <div className="border-t border-border/50 dark:border-border/20 bg-secondary/40 dark:bg-secondary/12 shadow-[inset_0_2px_6px_-2px_oklch(0_0_0/0.06)] dark:shadow-[inset_0_2px_6px_-2px_oklch(0_0_0/0.15)] divide-y divide-border/40 dark:divide-border/15">
          {subsToShow.map((sub) => (
            <SubQuestionRow
              key={sub.id}
              sub={sub}
              onAnswerChange={(a) => onSubAnswerChange(sub.id, a)}
              onReasoningChange={(r) => onSubReasoningChange(sub.id, r)}
              onCitationsChange={(c) => onSubCitationsChange(sub.id, c)}
              onCommentsChange={(comments) =>
                onSubCommentsChange(sub.id, comments)
              }
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
  additional_analysis: initialAnalysis,
  follow_ups: initialFollowUps,
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
          answer: sub.answer || "No",
          comments: sub.comments || "",
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
  const [additionalAnalysis, setAdditionalAnalysis] = useState(
    initialAnalysis || ""
  );
  const [followUps, setFollowUps] = useState(initialFollowUps || "");

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
            const allSubsYes = q.sub_questions.every(
              (s) => (s.answer || "No") === "Yes"
            );
            if (!allSubsYes) return q;
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
    (qId: string, subId: string, answer: "Yes" | "No") => {
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

  const handleSubCommentsChange = useCallback(
    (qId: string, subId: string, comments: string) => {
      setQuestions((prev) =>
        prev.map((q) =>
          q.id === qId
            ? {
                ...q,
                sub_questions: q.sub_questions?.map((s) =>
                  s.id === subId ? { ...s, comments } : s
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
      additional_analysis: additionalAnalysis || null,
      follow_ups: followUps || null,
    }),
    [peril, questions, overallOutcome, outcomeJustification, additionalAnalysis, followUps]
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
    setAdditionalAnalysis(initialAnalysis || "");
    setFollowUps(initialFollowUps || "");
    setFormTitle("");
    setSubmitStatus("idle");
    onCancel?.();
  }, [
    initialPeril,
    normalizedInitialQuestions,
    initialOutcome,
    initialJustification,
    initialAnalysis,
    initialFollowUps,
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
    if ((additionalAnalysis || "") !== (initialAnalysis || "")) return true;
    if ((followUps || "") !== (initialFollowUps || "")) return true;
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
    additionalAnalysis,
    initialAnalysis,
    followUps,
    initialFollowUps,
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
      (question.sub_questions?.filter((sub) => (sub.answer || "No") === "No")
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
                  prev === "Meets" ? "Does Not Meet Expectations" : "Meets"
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
            onSubCommentsChange={(subId, comments) =>
              handleSubCommentsChange(q.id, subId, comments)
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

        {/* Additional analysis (editable) */}
        <div className="border border-border/60 dark:border-border/25 rounded-xl p-5 bg-card dark:bg-card/80 shadow-2xs">
          <label className="text-sm font-semibold text-foreground/85 uppercase tracking-wider">
            Additional Analysis
          </label>
          <p className="text-xs text-muted-foreground mb-2">
            {peril.peril === "Exterior"
              ? "Wind/Hail Analysis"
              : "Flooring & Cabinetry Analysis"}
          </p>
          <div className="flex items-start gap-1">
            <AutoResizeTextarea
              value={additionalAnalysis}
              onChange={(e) => setAdditionalAnalysis(e.target.value)}
              placeholder="Optional additional analysis..."
              rows={2}
              className="flex-1 text-sm bg-secondary/20 dark:bg-background/50 border border-border/70 dark:border-border/20 rounded-lg px-3 py-2.5 text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/60 min-h-[52px] leading-relaxed"
            />
            <CopyButton text={additionalAnalysis} />
          </div>
        </div>

        {/* Follow-ups (editable) */}
        <div className="border border-border/60 dark:border-border/25 rounded-xl p-5 bg-card dark:bg-card/80 shadow-2xs">
          <label className="text-sm font-semibold text-foreground/85 uppercase tracking-wider">
            Recommended Follow-Ups
          </label>
          <div className="flex items-start gap-1 mt-1.5">
            <AutoResizeTextarea
              value={followUps}
              onChange={(e) => setFollowUps(e.target.value)}
              placeholder="Optional follow-up actions..."
              rows={2}
              className="flex-1 text-sm bg-secondary/20 dark:bg-background/50 border border-border/70 dark:border-border/20 rounded-lg px-3 py-2.5 text-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/60 min-h-[52px] leading-relaxed"
            />
            <CopyButton text={followUps} />
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
