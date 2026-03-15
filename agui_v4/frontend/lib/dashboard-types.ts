/**
 * Shared TypeScript types for the dashboard feature.
 *
 * These mirror the backend TFRAnalysisResult model and the JSON shape
 * returned by GET /forms/all.
 */

// ── Primitive answer types ───────────────────────────────────────

export type AnswerValue = "Yes" | "No" | "Insufficient information";
export type SubAnswerValue = boolean;
export type PerilType = "Interior" | "Exterior";
export type OutcomeValue = "Meets" | "Does Not Meet Expectations";

// ── Sub-question ─────────────────────────────────────────────────

export interface SubQuestion {
  id: string;
  text: string;
  reasoning: string;
  citations: string;
  answer?: SubAnswerValue;
  comments?: string | null;
}

// ── TFR Question ─────────────────────────────────────────────────

export interface TFRQuestion {
  id: string;
  text: string;
  answer: AnswerValue;
  sub_questions?: SubQuestion[] | null;
  missing_info?: string | null;
}

// ── Peril determination ──────────────────────────────────────────

export interface PerilDetermination {
  peril: PerilType;
  notes?: string | null;
}

// ── Full saved form record (from GET /forms/all) ─────────────────

export interface SavedForm {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  peril: PerilDetermination;
  questions: TFRQuestion[];
  overall_outcome: OutcomeValue;
  outcome_justification: string;
  additional_analysis?: string | null;
  follow_ups?: string | null;
}

// ── Computed per-form stats (used in FormsDataTable rows) ────────

export interface FormRowStats {
  id: string;
  title: string;
  peril: PerilType;
  overall_outcome: OutcomeValue;
  questionCount: number;
  yesCount: number;
  noCount: number;
  insufficientCount: number;
  driverCount: number;
  created_at: string;
  updated_at: string;
}

// ── Aggregated question stats (used in QuestionsAggregationTable) ──

export interface AggregatedSubQuestion {
  id: string;
  text: string;
  driverCount: number;
  totalAppearances: number;
  driverPercent: number;
}

export interface AggregatedQuestion {
  id: string;
  text: string;
  yesCount: number;
  noCount: number;
  insufficientCount: number;
  totalCount: number;
  yesPercent: number;
  noPercent: number;
  insufficientPercent: number;
  subQuestions: AggregatedSubQuestion[];
}
