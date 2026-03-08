"use client";

/**
 * Chat document context — tracks which documents the user has explicitly
 * added to the chat agent's context. Independent from the doc-agent's
 * "hidden / visible" paradigm in DocumentsPane.
 *
 * Example usage:
 *   const { chatDocNames, toggleChatDoc } = useChatDocs();
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
} from "react";
import type React from "react";

export interface ChatDocsContextValue {
  /** Set of file_name strings currently in the chat agent context. */
  chatDocNames: Set<string>;
  /** Add or remove a document by file_name. */
  toggleChatDoc: (fileName: string) => void;
  /** Remove a single document from chat context. */
  removeChatDoc: (fileName: string) => void;
  /** Add a single document to chat context (no-op if already present). */
  addChatDoc: (fileName: string) => void;
}

const ChatDocsContext = createContext<ChatDocsContextValue>({
  chatDocNames: new Set(),
  toggleChatDoc: () => {},
  removeChatDoc: () => {},
  addChatDoc: () => {},
});

/** Local storage key used to persist chat-context document names. */
const CHAT_DOCS_STORAGE_KEY = "agui_v3.chatDocNames.v1";

export function useChatDocs() {
  return useContext(ChatDocsContext);
}

/**
 * Provider that wraps the pane layout so ChatPane and DocumentsPane can
 * share the set of documents selected for the chat agent.
 */
export function ChatDocsProvider({ children }: { children: React.ReactNode }) {
  const [chatDocNames, setChatDocNames] = useState<Set<string>>(new Set());
  const [hasHydratedFromStorage, setHasHydratedFromStorage] = useState(false);

  // Rehydrate chat-context doc names after navigation / remount.
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(CHAT_DOCS_STORAGE_KEY);
      if (!raw) {
        setHasHydratedFromStorage(true);
        return;
      }

      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        const names = parsed.filter((v): v is string => typeof v === "string");
        setChatDocNames(new Set(names));
      }
    } catch (error) {
      console.warn("[ChatDocs] Failed to restore local state:", error);
    } finally {
      setHasHydratedFromStorage(true);
    }
  }, []);

  // Persist whenever the chat-context selection changes.
  useEffect(() => {
    if (!hasHydratedFromStorage) return;
    try {
      const payload = JSON.stringify([...chatDocNames]);
      window.localStorage.setItem(CHAT_DOCS_STORAGE_KEY, payload);
    } catch (error) {
      console.warn("[ChatDocs] Failed to persist local state:", error);
    }
  }, [chatDocNames, hasHydratedFromStorage]);

  const toggleChatDoc = useCallback((fileName: string) => {
    setChatDocNames((prev) => {
      const next = new Set(prev);
      if (next.has(fileName)) next.delete(fileName);
      else next.add(fileName);
      return next;
    });
  }, []);

  const removeChatDoc = useCallback((fileName: string) => {
    setChatDocNames((prev) => {
      if (!prev.has(fileName)) return prev;
      const next = new Set(prev);
      next.delete(fileName);
      return next;
    });
  }, []);

  const addChatDoc = useCallback((fileName: string) => {
    setChatDocNames((prev) => {
      if (prev.has(fileName)) return prev;
      const next = new Set(prev);
      next.add(fileName);
      return next;
    });
  }, []);

  return (
    <ChatDocsContext.Provider
      value={{ chatDocNames, toggleChatDoc, removeChatDoc, addChatDoc }}
    >
      {children}
    </ChatDocsContext.Provider>
  );
}
