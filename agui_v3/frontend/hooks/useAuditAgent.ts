"use client";

/**
 * Custom hook for interacting with the Audit Assistant agent.
 *
 * Uses CopilotKit v2's useAgent hook for AG-UI protocol communication
 * with the Pydantic AI backend. Subscribes to AG-UI tool call lifecycle
 * events to provide real-time activity tracking for the UI.
 *
 * Example usage:
 *   const { state, runAudit, isGenerating } = useAuditAgent();
 */

import { useAgent } from "@copilotkit/react-core/v2";
import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import type { A2UIComponent, SemanticZone } from "@/lib/a2ui-catalog";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

/**
 * Audit state synchronized with backend agent.
 * Must match AuditState in agent/agent.py.
 */
export interface AuditState {
  documents: Array<Record<string, unknown>>;
  components: A2UIComponent[];
  audit_questions: Array<Record<string, unknown>>;
  analysis_result: Record<string, unknown>;
  /** Canonical editable form payload, synced with backend via AG-UI. */
  audit_form_result: Record<string, unknown>;
  /** ID of the currently active persisted form, or null if unsaved. */
  current_form_id: string | null;
  status: "idle" | "analyzing" | "generating" | "complete" | "error";
  progress: number;
  current_step: string;
  activity_log: Array<{
    id: string;
    message: string;
    timestamp: string;
    status: "in_progress" | "completed" | "error";
  }>;
  error_message: string | null;
}

/**
 * Represents a tracked tool call during an agent run.
 * Populated from TOOL_CALL_START / TOOL_CALL_END AG-UI events.
 */
export interface ToolCallActivity {
  /** Tool call ID from the AG-UI event. */
  id: string;
  /** Raw tool function name (e.g. "get_documents"). */
  name: string;
  /** Human-friendly label for display. */
  displayName: string;
  /** Whether the tool is still executing or has completed. */
  status: "running" | "complete";
  /** Epoch ms when the tool call started. */
  startedAt: number;
}

/**
 * Represents a single backend step shown in the chat timeline.
 *
 * Built primarily from AuditState.activity_log so steps persist after
 * generation completes and reflect backend-authored messages.
 */
export interface StepActivity {
  /** Stable ID for React rendering. */
  id: string;
  /** Human-readable step text. */
  message: string;
  /** Step lifecycle state for icon/color mapping. */
  status: "in_progress" | "completed" | "error";
  /** ISO timestamp if known. */
  timestamp: string;
}

/** Initial state for the audit agent. */
const initialState: AuditState = {
  documents: [],
  components: [],
  audit_questions: [],
  analysis_result: {},
  audit_form_result: {},
  current_form_id: null,
  status: "idle",
  progress: 0,
  current_step: "",
  activity_log: [],
  error_message: null,
};

/** Shape of a saved form summary returned by GET /forms. */
export interface SavedFormSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  peril: string;
  overall_outcome: string;
  question_count: number;
}

/**
 * Payload shape for persisting an audit form.
 * Matches the backend's expected fields on POST /forms and PUT /state/audit-form.
 */
export interface AuditFormPayload {
  peril: Record<string, unknown>;
  questions: Array<Record<string, unknown>>;
  overall_outcome: string;
  outcome_justification: string;
  additional_analysis?: string | null;
  follow_ups?: string | null;
}

/**
 * Build an A2UIComponent spec for AuditQuestionForm from a form payload.
 * Mirrors the backend's generate_audit_question_form() so we can render
 * restored forms without an active agent run.
 */
function buildAuditFormComponent(payload: AuditFormPayload): A2UIComponent {
  return {
    id: `audit-form-${Date.now()}`,
    type: "a2ui.AuditQuestionForm",
    props: {
      peril: payload.peril,
      questions: payload.questions,
      overall_outcome: payload.overall_outcome,
      outcome_justification: payload.outcome_justification,
      additional_analysis: payload.additional_analysis ?? null,
      follow_ups: payload.follow_ups ?? null,
    },
    layout: { width: "full" },
    zone: "output",
  };
}

/**
 * Replace the first AuditQuestionForm in the components array, or append
 * one if none exists. Returns a new array (does not mutate).
 */
function upsertAuditFormComponent(
  components: A2UIComponent[],
  newComponent: A2UIComponent
): A2UIComponent[] {
  const idx = components.findIndex(
    (c) => c.type === "a2ui.AuditQuestionForm"
  );
  if (idx >= 0) {
    const updated = [...components];
    updated[idx] = newComponent;
    return updated;
  }
  return [...components, newComponent];
}

/**
 * Build a tool activity label from shared AG-UI state.
 *
 * Prefers current_step because it is the most specific user-facing
 * description of what the backend is currently doing.
 */
function getToolDisplayName(
  toolName: unknown,
  state?: AuditState
): string {
  const safeToolName =
    typeof toolName === "string" ? toolName : "tool_call";

  if (state?.current_step?.trim()) {
    return state.current_step.trim();
  }

  if (state?.status && state.status !== "idle") {
    const normalizedStatus = state.status
      .replaceAll("_", " ")
      .replace(/\b\w/g, (match) => match.toUpperCase());
    return `${normalizedStatus}...`;
  }

  return safeToolName.replaceAll("_", " ");
}

/**
 * Build a stable human-readable run-step label from shared state.
 *
 * This avoids UI flicker when current_step is briefly empty between
 * AG-UI snapshots by falling back to status/progress semantics.
 */
function getLiveStepText(state?: AuditState): string {
  if (!state) {
    return "";
  }

  const currentStep = state.current_step?.trim();
  if (currentStep) {
    return currentStep;
  }

  const statusBaseLabel: Record<AuditState["status"], string> = {
    idle: "",
    analyzing: "Analyzing claim...",
    generating: "Generating audit form...",
    complete: "Complete",
    error: "Error",
  };

  const baseLabel = statusBaseLabel[state.status];
  if (!baseLabel) {
    return "";
  }

  if (
    (state.status === "analyzing" || state.status === "generating") &&
    state.progress > 0
  ) {
    return `${baseLabel} (${state.progress}%)`;
  }

  return baseLabel;
}

/** Separate components by zone for the three-pane layout. */
function groupByZone(
  components: A2UIComponent[]
): Record<SemanticZone, A2UIComponent[]> {
  const groups: Record<SemanticZone, A2UIComponent[]> = {
    documents: [],
    output: [],
  };

  for (const comp of components) {
    const zone = (comp.zone as SemanticZone) || "output";
    if (groups[zone]) {
      groups[zone].push(comp);
    } else {
      groups.output.push(comp);
    }
  }

  return groups;
}

/**
 * Hook for audit agent interaction using CopilotKit v2's useAgent.
 *
 * Provides:
 * - state: Current audit state (synced with backend via AG-UI)
 * - componentsByZone: Components grouped by semantic zone
 * - toolActivity: Real-time tool call tracking from AG-UI events
 * - runAudit: Trigger audit analysis and form generation
 * - addDocument: Add a document to the state
 * - isGenerating: Whether generation is in progress
 * - isComplete: Whether generation finished successfully
 * - hasError: Whether an error occurred
 */
export function useAuditAgent() {
  const { agent } = useAgent({
    agentId: "audit_agent",
  });

  const state = (agent.state as AuditState) || initialState;
  const stateRef = useRef<AuditState>(state);
  const lastLiveStepRef = useRef<string>("");

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Tool call activity tracking from AG-UI events
  const [toolActivity, setToolActivity] = useState<ToolCallActivity[]>([]);

  useEffect(() => {
    const subscription = agent.subscribe({
      onToolCallStartEvent: ({ event }) => {
        const nextDisplayName = getToolDisplayName(
          event.toolCallName,
          stateRef.current
        );
        setToolActivity((prev) => [
          ...prev,
          {
            id: event.toolCallId,
            name: event.toolCallName,
            displayName: nextDisplayName,
            status: "running",
            startedAt: Date.now(),
          },
        ]);
      },
      onToolCallEndEvent: ({ event }) => {
        const nextDisplayName = getToolDisplayName(
          event.toolCallName,
          stateRef.current
        );
        setToolActivity((prev) =>
          prev.map((tc) =>
            tc.id === event.toolCallId
              ? { ...tc, status: "complete", displayName: nextDisplayName }
              : tc
          )
        );
      },
    });

    return () => subscription.unsubscribe();
  }, [agent]);

  // Keep labels in sync with live shared-state updates while tools are running.
  useEffect(() => {
    const liveLabel = getToolDisplayName("tool_call", state);
    setToolActivity((prev) => {
      let changed = false;
      const next = prev.map((tc) => {
        if (tc.status !== "running" || tc.displayName === liveLabel) {
          return tc;
        }
        changed = true;
        return { ...tc, displayName: liveLabel };
      });
      return changed ? next : prev;
    });
  }, [state.current_step, state.status]);

  const stepActivity = useMemo<StepActivity[]>(() => {
    const currentlyGenerating =
      agent.isRunning ||
      state.status === "analyzing" ||
      state.status === "generating";

    // Primary source: backend-owned activity log from shared state.
    const fromStateLog: StepActivity[] = (state.activity_log || []).map(
      (entry) => ({
        id: entry.id,
        message: entry.message,
        // Prevent stale spinner after run completion when backend left
        // a start-step as in_progress without an explicit completion row.
        status:
          !currentlyGenerating && entry.status === "in_progress"
            ? "completed"
            : entry.status,
        timestamp: entry.timestamp,
      })
    );

    // Secondary source: active tool calls while waiting for next state snapshot.
    const fromRunningTools: StepActivity[] = toolActivity
      .filter((tc) => tc.status === "running")
      .map((tc) => ({
        id: `tool-${tc.id}`,
        message: tc.displayName,
        status: "in_progress" as const,
        timestamp: new Date(tc.startedAt).toISOString(),
      }));

    // If backend set current_step but hasn't logged it yet, keep it visible.
    const liveStepText = getLiveStepText(state);
    const hasCurrentStepAlready =
      !liveStepText ||
      [...fromStateLog, ...fromRunningTools].some(
        (entry) => entry.message === liveStepText
      );
    const currentStepEntry: StepActivity[] = hasCurrentStepAlready
      ? []
      : [
          {
            id: "current-step-live",
            message: liveStepText,
            status: currentlyGenerating ? "in_progress" : "completed",
            timestamp: new Date().toISOString(),
          },
        ];

    return [...fromStateLog, ...fromRunningTools, ...currentStepEntry];
  }, [
    state.activity_log,
    state.current_step,
    state.status,
    toolActivity,
    agent.isRunning,
  ]);

  const isGenerating =
    agent.isRunning ||
    state.status === "analyzing" ||
    state.status === "generating";

  useEffect(() => {
    if (!isGenerating) {
      lastLiveStepRef.current = "";
      return;
    }
    const liveText = getLiveStepText(state);
    if (liveText) {
      lastLiveStepRef.current = liveText;
    }
  }, [isGenerating, state.current_step, state.status, state.progress]);

  const currentRunStepLabel = useMemo(() => {
    if (isGenerating) {
      return getLiveStepText(state) || lastLiveStepRef.current || "Working...";
    }
    return state.current_step?.trim() || "";
  }, [isGenerating, state.current_step, state.status, state.progress]);

  const componentsByZone = useMemo(
    () => groupByZone(state.components || []),
    [state.components]
  );

  const addDocument = useCallback(
    (doc: Record<string, unknown>) => {
      const currentDocs = (agent.state as AuditState)?.documents || [];
      agent.setState({
        ...((agent.state as AuditState) || initialState),
        documents: [...currentDocs, doc],
      });
    },
    [agent]
  );

  const runAudit = useCallback(
    async (userMessage: string) => {
      setToolActivity([]);

      agent.setState({
        ...((agent.state as AuditState) || initialState),
        activity_log: [],
        error_message: null,
        status: "analyzing",
        progress: 0,
        current_step: "Working...",
      });

      agent.addMessage({
        id: crypto.randomUUID(),
        role: "user",
        content: userMessage,
      });

      await agent.runAgent();

      // Reset status if still stuck on analyzing/generating after run completes
      const finalState = agent.state as AuditState;
      if (
        finalState &&
        (finalState.status === "analyzing" ||
          finalState.status === "generating")
      ) {
        agent.setState({
          ...finalState,
          status: "idle",
          progress: 0,
          current_step: "",
        });
      }
    },
    [agent]
  );

  const lastAssistantMessage = useMemo(() => {
    const msgs = agent.messages || [];
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "assistant" && msgs[i].content) {
        return msgs[i].content as string;
      }
    }
    return null;
  }, [agent.messages]);

  const stop = useCallback(() => {
    agent.abortRun();
  }, [agent]);

  // ── Form persistence ─────────────────────────────────────────

  const [isSaving, setIsSaving] = useState(false);

  /**
   * Sync the form payload into AG-UI shared state, then persist to disk
   * via POST /forms. Reuses current_form_id when updating an existing form.
   */
  const saveForm = useCallback(
    async (
      formPayload: AuditFormPayload,
      title?: string
    ): Promise<{ form_id: string; title: string }> => {
      setIsSaving(true);
      try {
        const currentState = (agent.state as AuditState) || initialState;

        agent.setState({
          ...currentState,
          audit_form_result:
            formPayload as unknown as Record<string, unknown>,
        });

        const formId = currentState.current_form_id || crypto.randomUUID();

        const resp = await fetch(`${BACKEND_URL}/forms`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audit_form_result: formPayload,
            id: formId,
            ...(title ? { title } : {}),
          }),
        });

        if (!resp.ok) {
          throw new Error(`Save failed: ${resp.status} ${await resp.text()}`);
        }

        const data = await resp.json();

        agent.setState({
          ...((agent.state as AuditState) || initialState),
          current_form_id: data.form_id,
        });

        return { form_id: data.form_id, title: data.title };
      } finally {
        setIsSaving(false);
      }
    },
    [agent]
  );

  /** Fetch the list of all saved forms from the backend. */
  const listSavedForms = useCallback(async (): Promise<SavedFormSummary[]> => {
    const resp = await fetch(`${BACKEND_URL}/forms`);
    if (!resp.ok) {
      throw new Error(`List forms failed: ${resp.status}`);
    }
    const data = await resp.json();
    return data.forms as SavedFormSummary[];
  }, []);

  /**
   * Restore a previously saved form by ID. Loads it on the backend,
   * then builds the AuditQuestionForm component on the frontend.
   */
  const restoreForm = useCallback(
    async (formId: string): Promise<void> => {
      const resp = await fetch(`${BACKEND_URL}/forms/${formId}/restore`, {
        method: "POST",
      });

      if (!resp.ok) {
        throw new Error(
          `Restore failed: ${resp.status} ${await resp.text()}`
        );
      }

      const data = await resp.json();
      const restoredPayload = data.audit_form_result as AuditFormPayload;

      const formComponent = buildAuditFormComponent(restoredPayload);
      const currentState = (agent.state as AuditState) || initialState;
      const updatedComponents = upsertAuditFormComponent(
        currentState.components || [],
        formComponent
      );

      agent.setState({
        ...currentState,
        audit_form_result:
          restoredPayload as unknown as Record<string, unknown>,
        current_form_id: formId,
        components: updatedComponents,
        status: "complete",
        current_step: `Restored saved form ${formId}`,
      });
    },
    [agent]
  );

  /**
   * Delete a saved form from disk. If it was the active form, clears
   * current_form_id from AG-UI state.
   */
  const deleteForm = useCallback(
    async (formId: string): Promise<void> => {
      const resp = await fetch(`${BACKEND_URL}/forms/${formId}`, {
        method: "DELETE",
      });

      if (!resp.ok) {
        throw new Error(`Delete failed: ${resp.status} ${await resp.text()}`);
      }

      const currentState = (agent.state as AuditState) || initialState;
      if (currentState.current_form_id === formId) {
        agent.setState({
          ...currentState,
          current_form_id: null,
        });
      }
    },
    [agent]
  );

  const isComplete = state.status === "complete";
  const hasError = state.status === "error";

  return {
    state,
    setState: agent.setState.bind(agent),
    componentsByZone,
    toolActivity,
    stepActivity,
    addDocument,
    runAudit,
    stop,
    isGenerating,
    currentRunStepLabel,
    isComplete,
    hasError,
    lastAssistantMessage,
    agent,
    isSaving,
    saveForm,
    listSavedForms,
    restoreForm,
    deleteForm,
  };
}
