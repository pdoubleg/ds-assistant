/**
 * Custom hook for interacting with the Audit Assistant agent.
 *
 * Uses CopilotKit v2's useAgent hook for AG-UI protocol communication
 * with the Pydantic AI backend. Subscribes to AG-UI tool call lifecycle
 * events to provide real-time activity tracking for the UI.
 */

import { useAgent } from "@copilotkit/react-core/v2";
import { useState, useMemo, useCallback, useEffect } from "react";
import type { A2UIComponent, SemanticZone } from "@/lib/a2ui-catalog";

/**
 * Audit state synchronized with backend agent.
 * Must match AuditState in agent/agent.py.
 */
export interface AuditState {
  documents: Array<Record<string, unknown>>;
  components: A2UIComponent[];
  audit_questions: Array<Record<string, unknown>>;
  document_review: Record<string, unknown>;
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
 * Initial state for the audit agent.
 */
const initialState: AuditState = {
  documents: [],
  components: [],
  audit_questions: [],
  document_review: {},
  status: "idle",
  progress: 0,
  current_step: "",
  activity_log: [],
  error_message: null,
};

/** Human-friendly display names for backend tool functions. */
const TOOL_DISPLAY_NAMES: Record<string, string> = {
  get_documents: "Retrieving documents",
  analyze_documents: "Analyzing documents",
  generate_audit_form: "Generating audit form",
};

/**
 * Separate components by zone for the three-pane layout.
 */
function groupByZone(components: A2UIComponent[]): Record<SemanticZone, A2UIComponent[]> {
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

  // Get state from agent, with fallback to initial state
  const state = (agent.state as AuditState) || initialState;

  // ── Tool call activity tracking ────────────────────────────────
  // Populated by subscribing to TOOL_CALL_START / TOOL_CALL_END
  // AG-UI events that Pydantic AI emits automatically for every tool.
  const [toolActivity, setToolActivity] = useState<ToolCallActivity[]>([]);

  useEffect(() => {
    const subscription = agent.subscribe({
      onToolCallStartEvent: ({ event }) => {
        setToolActivity((prev) => [
          ...prev,
          {
            id: event.toolCallId,
            name: event.toolCallName,
            displayName:
              TOOL_DISPLAY_NAMES[event.toolCallName] || event.toolCallName,
            status: "running",
            startedAt: Date.now(),
          },
        ]);
      },
      onToolCallEndEvent: ({ event }) => {
        setToolActivity((prev) =>
          prev.map((tc) =>
            tc.id === event.toolCallId ? { ...tc, status: "complete" } : tc
          )
        );
      },
    });

    return () => subscription.unsubscribe();
  }, [agent]);

  // Group components by zone for the three-pane layout
  const componentsByZone = useMemo(
    () => groupByZone(state.components || []),
    [state.components]
  );

  // Add a document to the shared state
  const addDocument = useCallback((doc: Record<string, unknown>) => {
    const currentDocs = (agent.state as AuditState)?.documents || [];
    agent.setState({
      ...(agent.state as AuditState || initialState),
      documents: [...currentDocs, doc],
    });
  }, [agent]);

  // Send a user message and run the agent. Preserves existing components
  // unless the agent itself clears them during the run.
  const runAudit = useCallback(async (userMessage: string) => {
    // Clear tool activity from any previous run
    setToolActivity([]);

    agent.setState({
      ...(agent.state as AuditState || initialState),
      status: "analyzing",
      progress: 0,
      current_step: "",
    });

    agent.addMessage({
      id: crypto.randomUUID(),
      role: "user",
      content: userMessage,
    });

    await agent.runAgent();

    // If the run completed without a tool emitting a StateSnapshotEvent
    // (e.g. only get_documents was called, or no tools at all), the status
    // will still be stuck on "analyzing"/"generating". Reset it so the UI
    // unblocks and the assistant response can be displayed.
    const finalState = agent.state as AuditState;
    if (
      finalState &&
      (finalState.status === "analyzing" || finalState.status === "generating")
    ) {
      agent.setState({
        ...finalState,
        status: "idle",
        progress: 0,
        current_step: "",
      });
    }
  }, [agent]);

  // The latest assistant message text from the agent's message history
  const lastAssistantMessage = useMemo(() => {
    const msgs = agent.messages || [];
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "assistant" && msgs[i].content) {
        return msgs[i].content as string;
      }
    }
    return null;
  }, [agent.messages]);

  // Stop any running generation
  const stop = useCallback(() => {
    agent.abortRun();
  }, [agent]);

  // Derived state flags
  const isGenerating = agent.isRunning || state.status === "analyzing" || state.status === "generating";
  const isComplete = state.status === "complete";
  const hasError = state.status === "error";

  return {
    state,
    setState: agent.setState.bind(agent),
    componentsByZone,
    toolActivity,
    addDocument,
    runAudit,
    stop,
    isGenerating,
    isComplete,
    hasError,
    lastAssistantMessage,
    agent,
  };
}
