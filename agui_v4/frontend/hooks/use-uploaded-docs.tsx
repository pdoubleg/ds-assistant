"use client";

/**
 * Uploaded documents context — shared state for documents uploaded
 * via the file input, available to both the Chat and Documents panes.
 *
 * Example usage:
 *   const { uploadedDocs, addUploadedDoc } = useUploadedDocs();
 */

import { createContext, useContext, useState, useCallback } from "react";
import type React from "react";

export interface UploadedDoc {
  file_name: string;
  claim_number: string;
  content_id: string;
  mime_type: string;
  content_url: string;
  domain: "claim" | "policy";
  document_type?: string;
  document_sub_type?: string;
  document_description?: string;
  create_date: string;
  source_system?: string;
  company_name?: string;
  /** Extracted text content, stored alongside metadata so it persists
   *  independently of the AG-UI ``state.documents`` array. */
  content?: string;
}

export interface UploadedDocsContextValue {
  uploadedDocs: UploadedDoc[];
  addUploadedDoc: (doc: UploadedDoc) => void;
  /** Replace an existing entry by file_name, or append if not found. */
  updateUploadedDoc: (fileName: string, doc: UploadedDoc) => void;
}

export const UploadedDocsContext = createContext<UploadedDocsContextValue>({
  uploadedDocs: [],
  addUploadedDoc: () => {},
  updateUploadedDoc: () => {},
});

export function useUploadedDocs() {
  return useContext(UploadedDocsContext);
}

/**
 * Provider component that wraps children with uploaded document state.
 * Place this around the pane layout so Chat and Documents can share state.
 */
export function UploadedDocsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [uploadedDocs, setUploadedDocs] = useState<UploadedDoc[]>([]);

  const addUploadedDoc = useCallback((doc: UploadedDoc) => {
    setUploadedDocs((prev) => [...prev, doc]);
  }, []);

  const updateUploadedDoc = useCallback(
    (fileName: string, doc: UploadedDoc) => {
      setUploadedDocs((prev) => {
        const idx = prev.findIndex((d) => d.file_name === fileName);
        if (idx === -1) return [...prev, doc];
        const next = [...prev];
        next[idx] = doc;
        return next;
      });
    },
    []
  );

  return (
    <UploadedDocsContext.Provider
      value={{ uploadedDocs, addUploadedDoc, updateUploadedDoc }}
    >
      {children}
    </UploadedDocsContext.Provider>
  );
}
