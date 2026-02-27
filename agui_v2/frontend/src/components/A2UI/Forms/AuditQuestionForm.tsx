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

import React, { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ChevronDown,
  ChevronRight,
  Copy,
  ClipboardCheck,
  AlertTriangle,
  CheckCircle2,
  Shield,
  FileWarning,
} from "lucide-react";

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

/**
 * Auto-resizing textarea that grows to fit content.
 *
 * Keeps a minimum height via utility classes while expanding for longer text.
 */
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
  active: string;
  idle: string;
}[] = [
  {
    value: "Yes",
    label: "Yes",
    active: "bg-emerald-500 text-white",
    idle: "text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/15",
  },
  {
    value: "No",
    label: "No",
    active: "bg-red-500 text-white",
    idle: "text-red-600 dark:text-red-400 hover:bg-red-500/15",
  },
  {
    value: "Insufficient information",
    label: "Insufficient",
    active: "bg-amber-500 text-white",
    idle: "text-amber-600 dark:text-amber-400 hover:bg-amber-500/15",
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
    size === "sm" ? "px-2.5 py-1 text-xs" : "px-4 py-2 text-sm";

  return (
    <div className="flex items-center gap-1">
      {ANSWER_OPTIONS.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          disabled={disableYes && opt.value === "Yes"}
          title={disableYes && opt.value === "Yes" ? yesDisabledTitle : undefined}
          type="button"
          className={`${pillClass} rounded-md font-semibold border transition-all duration-150 disabled:opacity-45 disabled:cursor-not-allowed ${
            value === opt.value
              ? `${opt.active} border-transparent`
              : `${opt.idle} border-border/40 bg-transparent`
          }`}
        >
          {opt.label}
        </button>
      ))}
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

  return (
    <div className="flex items-start gap-4 py-4 pl-11 pr-5 bg-red-500/5 dark:bg-red-500/8">
      <span className="shrink-0 text-xs font-mono text-accent/80 mt-0.5 min-w-[64px]">
        {sub.id}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-base text-foreground/90 leading-relaxed mb-2.5">
          {sub.text}
        </p>

        {/* Sub-question answer (default to No) */}
        <div className="mt-1">
          <label className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wider">
            Sub-Question Answer
          </label>
          <div className="flex items-center gap-2 mt-1">
            <button
              type="button"
              onClick={() => onAnswerChange("No")}
              className={`px-3.5 py-1.5 rounded-md text-sm font-semibold border transition-colors ${
                subAnswer === "No"
                  ? "bg-red-500 text-white border-transparent"
                  : "bg-transparent border-border/40 text-red-600 dark:text-red-400 hover:bg-red-500/15"
              }`}
            >
              No
            </button>
            <button
              type="button"
              onClick={() => onAnswerChange("Yes")}
              className={`px-3.5 py-1.5 rounded-md text-sm font-semibold border transition-colors ${
                subAnswer === "Yes"
                  ? "bg-emerald-500 text-white border-transparent"
                  : "bg-transparent border-border/40 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/15"
              }`}
            >
              Yes
            </button>
          </div>
        </div>

        {/* Reasoning (editable) */}
        <div className="mt-1">
          <label className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wider">
            Reasoning
          </label>
          <div className="flex items-start gap-1 mt-0.5">
            <AutoResizeTextarea
              value={sub.reasoning}
              onChange={(e) => onReasoningChange(e.target.value)}
              placeholder="Explain the reasoning..."
              rows={2}
              className="flex-1 text-base bg-secondary/40 border border-border/40 rounded-md px-3.5 py-2.5 text-foreground/95 placeholder:text-muted-foreground/55 focus:outline-none focus:ring-1 focus:ring-accent/50 min-h-[56px] leading-relaxed"
            />
            <CopyButton text={sub.reasoning} />
          </div>
        </div>

        {/* Citations (editable) */}
        <div className="mt-2">
          <label className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wider">
            Citations
          </label>
          <div className="flex items-start gap-1 mt-0.5">
            <AutoResizeTextarea
              value={sub.citations}
              onChange={(e) => onCitationsChange(e.target.value)}
              placeholder="Reference specific evidence..."
              rows={1}
              className="flex-1 text-base bg-secondary/40 border border-border/40 rounded-md px-3.5 py-2.5 text-foreground/95 placeholder:text-muted-foreground/55 focus:outline-none focus:ring-1 focus:ring-accent/50 min-h-[48px] leading-relaxed"
            />
            <CopyButton text={sub.citations} />
          </div>
        </div>

        {/* Final comments (editable) */}
        <div className="mt-2">
          <label className="text-xs font-medium text-muted-foreground/80 uppercase tracking-wider">
            Final Comments
          </label>
          <div className="flex items-start gap-1 mt-0.5">
            <AutoResizeTextarea
              value={sub.comments || ""}
              onChange={(e) => onCommentsChange(e.target.value)}
              placeholder="Optional final comments for this sub-question..."
              rows={2}
              className="flex-1 text-base bg-secondary/40 border border-border/40 rounded-md px-3.5 py-2.5 text-foreground/95 placeholder:text-muted-foreground/55 focus:outline-none focus:ring-1 focus:ring-accent/50 min-h-[56px] leading-relaxed"
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

  const hasSubs = !!question.sub_questions && question.sub_questions.length > 0;
  const isNo = question.answer === "No";
  const isInsufficient = question.answer === "Insufficient information";
  const allSubsYes = useMemo(() => {
    if (!hasSubs) return true;
    return question.sub_questions!.every((sub) => (sub.answer || "No") === "Yes");
  }, [hasSubs, question.sub_questions]);
  const noDriverCount = useMemo(() => {
    if (!hasSubs) return 0;
    return question.sub_questions!.filter((sub) => (sub.answer || "No") === "No").length;
  }, [hasSubs, question.sub_questions]);

  useEffect(() => {
    // Re-open existing sub-questions when parent is marked "No".
    if (isNo && hasSubs) {
      setExpanded(true);
    }
  }, [isNo, hasSubs]);

  // When collapsed, still show sub-questions for "No" answers
  const subsToShow = useMemo(() => {
    if (!hasSubs) return [];
    if (expanded) return question.sub_questions!;
    // Collapsed: show subs only if parent is No
    return isNo ? question.sub_questions! : [];
  }, [expanded, hasSubs, question.sub_questions, isNo]);

  const showSubSection = subsToShow.length > 0;

  return (
    <div className="border border-border/40 rounded-xl overflow-hidden bg-secondary/10">
      {/* Main question row */}
      <div className="flex items-start gap-4 p-5">
        {/* Expand toggle */}
        <button
          onClick={() => hasSubs && setExpanded((prev) => !prev)}
          className="shrink-0 mt-0.5 text-muted-foreground"
          type="button"
        >
          {hasSubs ? (
            expanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )
          ) : (
            <div className="w-4" />
          )}
        </button>

        {/* Question ID */}
        <span className="shrink-0 text-sm font-mono text-primary mt-0.5 min-w-[64px]">
          {question.id}
        </span>

        {/* Question text + missing info */}
        <div className="flex-1 min-w-0">
          <p className="text-lg text-foreground leading-relaxed font-normal">
            {question.text}
          </p>

          {/* Missing info field (shown when answer is Insufficient information) */}
          {isInsufficient && (
            <div className="mt-3">
              <label className="text-xs font-medium text-amber-600 dark:text-amber-400 uppercase tracking-wider flex items-center gap-1">
                <FileWarning className="h-3 w-3" />
                Missing Information
              </label>
              <div className="flex items-start gap-1 mt-0.5">
                <AutoResizeTextarea
                  value={question.missing_info || ""}
                  onChange={(e) => onMissingInfoChange(e.target.value)}
                  placeholder="What information is needed to make a determination?"
                  rows={2}
                  className="flex-1 text-base bg-amber-500/5 border border-amber-500/30 rounded-md px-3.5 py-2.5 text-foreground/95 placeholder:text-muted-foreground/55 focus:outline-none focus:ring-1 focus:ring-amber-500/50 min-h-[56px] leading-relaxed"
                />
                <CopyButton text={question.missing_info || ""} />
              </div>
            </div>
          )}
        </div>

        {/* Answer pills */}
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
        <div className="px-5 py-2.5 border-t flex items-center gap-2 text-sm bg-red-500/8 border-red-500/15 text-red-600 dark:text-red-400">
          <AlertTriangle className="h-3.5 w-3.5" />
          {noDriverCount} driver{noDriverCount !== 1 ? "s" : ""} identified
        </div>
      )}

      {/* Sub-questions (reasoning + citations) */}
      {showSubSection && (
        <div className="border-t border-border/25 divide-y divide-border/15">
          {!expanded && hasSubs && (
            <div className="px-5 py-2 text-xs text-muted-foreground/80 bg-secondary/10 flex items-center justify-between">
              <span>
                {subsToShow.length} sub-question{subsToShow.length > 1 ? "s" : ""}
              </span>
              <button
                type="button"
                onClick={() => setExpanded(true)}
                className="text-accent hover:underline"
              >
                Show all
              </button>
            </div>
          )}
          {subsToShow.map((sub) => (
            <SubQuestionRow
              key={sub.id}
              sub={sub}
              onAnswerChange={(a) => onSubAnswerChange(sub.id, a)}
              onReasoningChange={(r) => onSubReasoningChange(sub.id, r)}
              onCitationsChange={(c) => onSubCitationsChange(sub.id, c)}
              onCommentsChange={(comments) => onSubCommentsChange(sub.id, comments)}
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
}: AuditQuestionFormProps): React.ReactElement {
  const [peril, setPeril] = useState(initialPeril);
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
  const [questions, setQuestions] = useState<TFRQuestion[]>(normalizedInitialQuestions);
  const [overallOutcome, setOverallOutcome] = useState(initialOutcome);
  const [outcomeJustification, setOutcomeJustification] = useState(initialJustification);
  const [additionalAnalysis, setAdditionalAnalysis] = useState(initialAnalysis || "");
  const [followUps, setFollowUps] = useState(initialFollowUps || "");

  // ── Question-level handlers ──────────────────────────────────

  const handleAnswerChange = useCallback(
    (qId: string, answer: AnswerValue) => {
      setQuestions((prev) =>
        prev.map((q) => {
          if (q.id !== qId) return q;
          if (answer === "Yes" && q.sub_questions && q.sub_questions.length > 0) {
            const allSubsYes = q.sub_questions.every((s) => (s.answer || "No") === "Yes");
            if (!allSubsYes) {
              return q;
            }
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

  // ── Summary counts ───────────────────────────────────────────

  const yesCount = questions.filter((q) => q.answer === "Yes").length;
  const noCount = questions.filter((q) => q.answer === "No").length;
  const insufficientCount = questions.filter((q) => q.answer === "Insufficient information").length;
  const driverCount = questions.reduce(
    (count, question) =>
      count +
      (question.sub_questions?.filter((sub) => (sub.answer || "No") === "No").length || 0),
    0
  );

  return (
    <Card className="border-primary/20 bg-linear-to-br from-card to-secondary/20">
      <CardHeader className="pb-5">
        {/* Peril + outcome header */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-semibold text-foreground">
                TFR Questionnaire
              </h2>
            </div>
            <button
              type="button"
              onClick={() =>
                setPeril((prev) => ({
                  ...prev,
                  peril: prev.peril === "Interior" ? "Exterior" : "Interior",
                }))
              }
              title="Click to toggle peril"
            >
              <Badge
                variant="outline"
                className={`text-sm font-semibold cursor-pointer ${
                  peril.peril === "Exterior"
                    ? "bg-blue-500/15 text-blue-600 dark:text-blue-400 border-blue-500/30"
                    : "bg-orange-500/15 text-orange-600 dark:text-orange-400 border-orange-500/30"
                }`}
              >
                {peril.peril}
              </Badge>
            </button>
          </div>

          {/* Outcome badge (click to toggle) */}
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
              className={`text-sm font-semibold cursor-pointer ${
                overallOutcome === "Meets"
                  ? "bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                  : "bg-red-500/20 text-red-600 dark:text-red-400 border-red-500/30"
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
        </div>

        {/* Counts bar */}
        <div className="flex items-center gap-3 mt-3">
          <p className="text-sm text-muted-foreground">
            {questions.length} questions
          </p>
          <div className="flex items-center gap-1.5">
            {yesCount > 0 && (
              <Badge className="bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-xs">
                {yesCount} Yes
              </Badge>
            )}
            {noCount > 0 && (
              <Badge className="bg-red-500/20 text-red-600 dark:text-red-400 border-red-500/30 text-xs">
                {noCount} No
              </Badge>
            )}
            {insufficientCount > 0 && (
              <Badge className="bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30 text-xs">
                {insufficientCount} Insufficient
              </Badge>
            )}
            {driverCount > 0 && (
              <Badge className="bg-rose-500/20 text-rose-600 dark:text-rose-400 border-rose-500/30 text-xs">
                {driverCount} Drivers
              </Badge>
            )}
          </div>
        </div>

        {/* Peril notes */}
        {peril.notes && (
          <p className="text-sm text-muted-foreground mt-2 italic">
            Peril notes: {peril.notes}
          </p>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Questions */}
        {questions.map((q) => (
          <QuestionRow
            key={q.id}
            question={q}
            onAnswerChange={(a) => handleAnswerChange(q.id, a)}
            onMissingInfoChange={(info) => handleMissingInfoChange(q.id, info)}
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

        {/* Outcome justification (editable) */}
        <div className="border border-border/40 rounded-xl p-5 bg-secondary/10 mt-5">
          <label className="text-sm font-semibold text-foreground/85 uppercase tracking-wider">
            Outcome Justification
          </label>
          <div className="flex items-start gap-1 mt-1.5">
            <AutoResizeTextarea
              value={outcomeJustification}
              onChange={(e) => setOutcomeJustification(e.target.value)}
              placeholder="Justification for the overall outcome..."
              rows={3}
              className="flex-1 text-base bg-secondary/40 border border-border/40 rounded-md px-3.5 py-2.5 text-foreground/95 placeholder:text-muted-foreground/55 focus:outline-none focus:ring-1 focus:ring-accent/50 min-h-[88px] leading-relaxed"
            />
            <CopyButton text={outcomeJustification} />
          </div>
        </div>

        {/* Additional analysis (editable, optional) */}
        <div className="border border-border/40 rounded-xl p-5 bg-secondary/10">
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
              className="flex-1 text-base bg-secondary/40 border border-border/40 rounded-md px-3.5 py-2.5 text-foreground/95 placeholder:text-muted-foreground/55 focus:outline-none focus:ring-1 focus:ring-accent/50 min-h-[56px] leading-relaxed"
            />
            <CopyButton text={additionalAnalysis} />
          </div>
        </div>

        {/* Follow-ups (editable, optional) */}
        <div className="border border-border/40 rounded-xl p-5 bg-secondary/10">
          <label className="text-sm font-semibold text-foreground/85 uppercase tracking-wider">
            Recommended Follow-Ups
          </label>
          <div className="flex items-start gap-1 mt-1.5">
            <AutoResizeTextarea
              value={followUps}
              onChange={(e) => setFollowUps(e.target.value)}
              placeholder="Optional follow-up actions..."
              rows={2}
              className="flex-1 text-base bg-secondary/40 border border-border/40 rounded-md px-3.5 py-2.5 text-foreground/95 placeholder:text-muted-foreground/55 focus:outline-none focus:ring-1 focus:ring-accent/50 min-h-[56px] leading-relaxed"
            />
            <CopyButton text={followUps} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default AuditQuestionForm;
