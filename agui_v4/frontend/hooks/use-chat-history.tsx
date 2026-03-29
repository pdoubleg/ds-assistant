"use client";

/**
 * Shared chat-history context for the chat pane.
 *
 * Keeps the in-memory transcript alive while the app navigates between routes
 * under the same client session. Because this state is not written to browser
 * storage, a full page refresh naturally starts a new conversation.
 *
 * Example usage:
 *   const {
 *     chatMessages,
 *     setChatMessages,
 *     collapsedToolBubbleIds,
 *     setCollapsedToolBubbleIds,
 *   } = useChatHistory();
 */

import {
  createContext,
  useContext,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import type { StepActivity } from "@/hooks/use-audit-agent";

export type UserAssistantChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type ToolStatusChatMessage = {
  id: string;
  role: "tool_status";
  steps: StepActivity[];
  isLive: boolean;
};

export type ChatMessage = UserAssistantChatMessage | ToolStatusChatMessage;

export const DEFAULT_WELCOME_MESSAGE: UserAssistantChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Welcome to Q-Bot, your AI-powered TFR audit assistant. Upload documents and I'll generate a custom audit questionnaire with insights. You can upload plain text, HTML, PDF, Word, Excel, Outlook email, RFC822 email, and common image files.",
};

export interface ChatHistoryContextValue {
  /** In-memory transcript shown in the chat pane. */
  chatMessages: ChatMessage[];
  /** React state setter for transcript updates. */
  setChatMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  /** IDs of completed tool bubbles currently rendered as collapsed. */
  collapsedToolBubbleIds: Set<string>;
  /** React state setter for collapsed tool-bubble IDs. */
  setCollapsedToolBubbleIds: Dispatch<SetStateAction<Set<string>>>;
}

const ChatHistoryContext = createContext<ChatHistoryContextValue>({
  chatMessages: [DEFAULT_WELCOME_MESSAGE],
  setChatMessages: () => {},
  collapsedToolBubbleIds: new Set<string>(),
  setCollapsedToolBubbleIds: () => {},
});

/**
 * Access shared in-memory chat history for the current app session.
 *
 * Returns:
 *   The chat transcript state and collapsed tool-bubble state.
 */
export function useChatHistory(): ChatHistoryContextValue {
  return useContext(ChatHistoryContext);
}

/**
 * Provide session-scoped chat history to the app.
 *
 * Args:
 *   children: Descendant UI that needs access to the shared transcript.
 *
 * Returns:
 *   The provider-wrapped subtree.
 */
export function ChatHistoryProvider({
  children,
}: {
  children: ReactNode;
}): React.ReactElement {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    DEFAULT_WELCOME_MESSAGE,
  ]);
  const [collapsedToolBubbleIds, setCollapsedToolBubbleIds] = useState<Set<string>>(
    new Set()
  );

  const contextValue = useMemo(
    () => ({
      chatMessages,
      setChatMessages,
      collapsedToolBubbleIds,
      setCollapsedToolBubbleIds,
    }),
    [chatMessages, collapsedToolBubbleIds]
  );

  return (
    <ChatHistoryContext.Provider value={contextValue}>
      {children}
    </ChatHistoryContext.Provider>
  );
}
