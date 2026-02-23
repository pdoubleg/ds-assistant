/**
 * AuditQuestionForm Component
 *
 * Displays audit questions rated Yes / No / NA with optional sub-questions.
 * Sub-questions act as "drivers" — when the parent is rated No, at least
 * one sub-question must also be rated No to identify the specific deficiency.
 *
 * Schema per question:
 *   { id, question, rating: Yes|No|NA|null, comments, sub_questions? }
 * Schema per sub-question:
 *   { id, question, rating: Yes|No|NA|null, comments }
 */

import React, { useState, useCallback, useMemo } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ChevronDown,
  ChevronRight,
  Copy,
  ClipboardCheck,
  AlertTriangle,
  CheckCircle2,
} from "lucide-react";

// ── Types ────────────────────────────────────────────────────────

type RatingValue = "Yes" | "No" | "NA";

interface SubQuestion {
  id: string;
  question: string;
  rating: RatingValue | null;
  comments: string;
}

interface AuditQuestion {
  id: string;
  question: string;
  rating: RatingValue | null;
  comments: string;
  sub_questions?: SubQuestion[];
}

export interface AuditQuestionFormProps {
  /** List of audit questions */
  questions: AuditQuestion[];
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

// ── Rating Pills ─────────────────────────────────────────────────

const RATING_OPTIONS: {
  value: RatingValue;
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
    value: "NA",
    label: "N/A",
    active: "bg-slate-500 text-white",
    idle: "text-slate-600 dark:text-slate-400 hover:bg-slate-500/15",
  },
];

function RatingPills({
  value,
  onChange,
  size = "default",
}: {
  value: RatingValue | null;
  onChange: (v: RatingValue) => void;
  /** "sm" renders smaller pills for sub-question rows */
  size?: "default" | "sm";
}) {
  const pillClass =
    size === "sm" ? "px-2 py-1 text-[10px]" : "px-3 py-1.5 text-xs";

  return (
    <div className="flex items-center gap-1">
      {RATING_OPTIONS.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          type="button"
          className={`${pillClass} rounded-md font-semibold border transition-all duration-150 ${
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
  parentIsNo,
  onRatingChange,
  onCommentChange,
}: {
  sub: SubQuestion;
  /** Whether the parent question is rated No (affects highlight) */
  parentIsNo: boolean;
  onRatingChange: (rating: RatingValue) => void;
  onCommentChange: (comments: string) => void;
}) {
  // Highlight this row if it's been flagged as a driver (rated No)
  const isDriver = sub.rating === "No";

  return (
    <div
      className={`flex items-start gap-3 py-3 pl-10 pr-4 transition-colors ${
        isDriver
          ? "bg-red-500/5 dark:bg-red-500/8"
          : parentIsNo && !sub.rating
            ? "bg-amber-500/3 dark:bg-amber-500/5"
            : ""
      }`}
    >
      <span className="shrink-0 text-[11px] font-mono text-accent/70 mt-0.5 min-w-[56px]">
        {sub.id}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm text-foreground/85 leading-relaxed">
            {sub.question}
          </p>
          <div className="shrink-0">
            <RatingPills
              value={sub.rating}
              onChange={onRatingChange}
              size="sm"
            />
          </div>
        </div>
        {/* Always-visible comment field */}
        <div className="mt-2 flex items-start gap-1">
          <textarea
            value={sub.comments}
            onChange={(e) => onCommentChange(e.target.value)}
            placeholder="Add comments..."
            rows={2}
            className="flex-1 text-sm bg-secondary/40 border border-border/40 rounded-md px-3 py-2 text-foreground/90 placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-accent/50 resize-y min-h-[40px] leading-relaxed"
          />
          <CopyButton text={sub.comments} />
        </div>
      </div>
    </div>
  );
}

// ── Question Row ─────────────────────────────────────────────────

function QuestionRow({
  question,
  onRatingChange,
  onCommentChange,
  onSubRatingChange,
  onSubCommentChange,
}: {
  question: AuditQuestion;
  onRatingChange: (rating: RatingValue) => void;
  onCommentChange: (comments: string) => void;
  onSubRatingChange: (subId: string, rating: RatingValue) => void;
  onSubCommentChange: (subId: string, comments: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const hasSubs =
    question.sub_questions && question.sub_questions.length > 0;
  const isNo = question.rating === "No";

  // Driver validation: when parent is No, at least one sub must be No
  const driverSubs = useMemo(() => {
    if (!hasSubs) return [];
    return question.sub_questions!.filter((s) => s.rating === "No");
  }, [question.sub_questions, hasSubs]);

  const driverCount = driverSubs.length;
  const needsDriver = isNo && hasSubs;
  const hasDriver = driverCount > 0;

  // When collapsed, still show driver sub-questions (rated No)
  const subsToShow = useMemo(() => {
    if (!hasSubs) return [];
    if (expanded) return question.sub_questions!;
    // Collapsed: only show drivers
    return driverSubs;
  }, [expanded, hasSubs, question.sub_questions, driverSubs]);

  const showSubSection = subsToShow.length > 0;
  const isPartialView = !expanded && driverSubs.length > 0 && hasSubs;

  return (
    <div className="border border-border/40 rounded-lg overflow-hidden bg-secondary/10">
      {/* Main question row */}
      <div className="flex items-start gap-3 p-4">
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
        <span className="shrink-0 text-xs font-mono text-primary mt-0.5 min-w-[56px]">
          {question.id}
        </span>

        {/* Question text + comment */}
        <div className="flex-1 min-w-0">
          <p className="text-base text-foreground leading-relaxed font-normal">
            {question.question}
          </p>

          {/* Always-visible comment field with copy button */}
          <div className="mt-3 flex items-start gap-1">
            <textarea
              value={question.comments}
              onChange={(e) => onCommentChange(e.target.value)}
              placeholder="Add comments..."
              rows={2}
              className="flex-1 text-sm bg-secondary/40 border border-border/40 rounded-md px-3 py-2 text-foreground/90 placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-accent/50 resize-y min-h-[40px] leading-relaxed"
            />
            <CopyButton text={question.comments} />
          </div>
        </div>

        {/* Rating pills */}
        <div className="shrink-0 ml-2">
          <RatingPills value={question.rating} onChange={onRatingChange} />
        </div>
      </div>

      {/* Driver validation banner */}
      {needsDriver && (
        <div
          className={`px-4 py-2 border-t flex items-center gap-2 text-xs ${
            hasDriver
              ? "bg-emerald-500/8 border-emerald-500/15 text-emerald-600 dark:text-emerald-400"
              : "bg-red-500/8 border-red-500/15 text-red-500/80 dark:text-red-400/80"
          }`}
        >
          {hasDriver ? (
            <>
              <CheckCircle2 className="h-3.5 w-3.5" />
              {driverCount} driver{driverCount > 1 ? "s" : ""} identified
            </>
          ) : (
            <>
              <AlertTriangle className="h-3.5 w-3.5" />
              Rating is No &mdash; mark at least one sub-question as No to
              identify the driver
            </>
          )}
        </div>
      )}

      {/* Sub-questions (full list when expanded, drivers-only when collapsed) */}
      {showSubSection && (
        <div className="border-t border-border/25 divide-y divide-border/15">
          {isPartialView && (
            <div className="px-4 py-1.5 text-[10px] text-muted-foreground/70 bg-secondary/10 flex items-center justify-between">
              <span>
                Showing {driverCount} driver{driverCount > 1 ? "s" : ""} only
              </span>
              <button
                type="button"
                onClick={() => setExpanded(true)}
                className="text-accent hover:underline"
              >
                Show all {question.sub_questions!.length} sub-questions
              </button>
            </div>
          )}
          {subsToShow.map((sub) => (
            <SubQuestionRow
              key={sub.id}
              sub={sub}
              parentIsNo={isNo}
              onRatingChange={(r) => onSubRatingChange(sub.id, r)}
              onCommentChange={(c) => onSubCommentChange(sub.id, c)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Form ────────────────────────────────────────────────────

export function AuditQuestionForm({
  questions: initialQuestions,
}: AuditQuestionFormProps): React.ReactElement {
  const [questions, setQuestions] =
    useState<AuditQuestion[]>(initialQuestions);

  const handleRatingChange = useCallback(
    (qId: string, rating: RatingValue) => {
      setQuestions((prev) =>
        prev.map((q) => (q.id === qId ? { ...q, rating } : q))
      );
    },
    []
  );

  const handleCommentChange = useCallback(
    (qId: string, comments: string) => {
      setQuestions((prev) =>
        prev.map((q) => (q.id === qId ? { ...q, comments } : q))
      );
    },
    []
  );

  const handleSubRatingChange = useCallback(
    (qId: string, subId: string, rating: RatingValue) => {
      setQuestions((prev) =>
        prev.map((q) =>
          q.id === qId
            ? {
                ...q,
                sub_questions: q.sub_questions?.map((s) =>
                  s.id === subId ? { ...s, rating } : s
                ),
              }
            : q
        )
      );
    },
    []
  );

  const handleSubCommentChange = useCallback(
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

  // Summary counts
  const yesCount = questions.filter((q) => q.rating === "Yes").length;
  const noCount = questions.filter((q) => q.rating === "No").length;
  const naCount = questions.filter((q) => q.rating === "NA").length;
  const pending = questions.filter((q) => q.rating === null).length;

  // Validation: count how many "No" questions are missing drivers
  const missingDrivers = useMemo(() => {
    return questions.filter(
      (q) =>
        q.rating === "No" &&
        q.sub_questions &&
        q.sub_questions.length > 0 &&
        !q.sub_questions.some((s) => s.rating === "No")
    ).length;
  }, [questions]);

  return (
    <Card className="border-primary/20 bg-linear-to-br from-card to-secondary/20">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-foreground">
              Audit Questionnaire
            </h2>
            <p className="text-xs text-muted-foreground mt-1">
              {questions.length} questions &middot; {pending} pending
              {missingDrivers > 0 && (
                <span className="text-red-500 dark:text-red-400 ml-2">
                  &middot; {missingDrivers} missing driver
                  {missingDrivers > 1 ? "s" : ""}
                </span>
              )}
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            {yesCount > 0 && (
              <Badge className="bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-[10px]">
                {yesCount} Yes
              </Badge>
            )}
            {noCount > 0 && (
              <Badge className="bg-red-500/20 text-red-600 dark:text-red-400 border-red-500/30 text-[10px]">
                {noCount} No
              </Badge>
            )}
            {naCount > 0 && (
              <Badge className="bg-slate-500/20 text-slate-600 dark:text-slate-400 border-slate-500/30 text-[10px]">
                {naCount} N/A
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {questions.map((q) => (
          <QuestionRow
            key={q.id}
            question={q}
            onRatingChange={(r) => handleRatingChange(q.id, r)}
            onCommentChange={(c) => handleCommentChange(q.id, c)}
            onSubRatingChange={(subId, r) =>
              handleSubRatingChange(q.id, subId, r)
            }
            onSubCommentChange={(subId, c) =>
              handleSubCommentChange(q.id, subId, c)
            }
          />
        ))}
      </CardContent>
    </Card>
  );
}

export default AuditQuestionForm;
