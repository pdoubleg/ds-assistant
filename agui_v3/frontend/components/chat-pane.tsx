"use client";

/**
 * ChatPane — the left pane of the three-pane layout.
 *
 * Provides:
 *   - Chat message history with user / assistant bubbles
 *   - File attachment and upload flow
 *   - Real-time tool-call activity indicator during generation
 *   - Progress bar for long-running agent tasks
 */

import React, { useState, useCallback, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { useAuditAgent, type StepActivity } from "@/hooks/useAuditAgent";
import { useUploadedDocs } from "@/hooks/use-uploaded-docs";
import {
  Send,
  Loader2,
  FileText,
  Paperclip,
  Check,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  User,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";
const TOOL_MESSAGE_MAX_CHARS = 110;

const EXT_TO_MIME: Record<string, string> = {
  pdf: "application/pdf",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  tiff: "image/tiff",
};

function mimeFromExt(ext: string): string {
  return EXT_TO_MIME[ext.toLowerCase()] || "application/octet-stream";
}

/**
 * Truncate long tool-step messages for compact chat rendering.
 */
function truncateToolMessage(message: string, maxChars: number): string {
  if (message.length <= maxChars) {
    return message;
  }
  return `${message.slice(0, maxChars - 1).trimEnd()}…`;
}

function getToolBubbleSummary(steps: StepActivity[]): string {
  const runningStepCount = steps.filter(
    (step) => step.status === "in_progress"
  ).length;
  const completedStepCount = steps.filter(
    (step) => step.status === "completed"
  ).length;
  const errorStepCount = steps.filter((step) => step.status === "error").length;

  return [
    `${steps.length} step${steps.length === 1 ? "" : "s"}`,
    runningStepCount > 0
      ? `${runningStepCount} running`
      : `${completedStepCount} complete`,
    errorStepCount > 0 ? `${errorStepCount} error` : null,
  ]
    .filter(Boolean)
    .join(" • ");
}

function areStepListsEqual(
  previous: StepActivity[],
  next: StepActivity[]
): boolean {
  if (previous.length !== next.length) {
    return false;
  }

  for (let i = 0; i < previous.length; i++) {
    if (
      previous[i].id !== next[i].id ||
      previous[i].message !== next[i].message ||
      previous[i].status !== next[i].status ||
      previous[i].timestamp !== next[i].timestamp
    ) {
      return false;
    }
  }

  return true;
}

type UserAssistantChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type ToolStatusChatMessage = {
  id: string;
  role: "tool_status";
  steps: StepActivity[];
  isLive: boolean;
};

type ChatMessage = UserAssistantChatMessage | ToolStatusChatMessage;

function QBotAvatarIcon({
  className,
}: {
  className?: string;
}): React.ReactElement {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 512 512"
      role="img"
      aria-label="Q-Bot monochrome outlined logo"
      className={className}
    >
      <g
        fill="currentColor"
        stroke="#FFFFFF"
        strokeWidth="14"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M270 72c6-24 20-40 36-50 2 22-3 41-16 58l22 10c16-20 36-31 58-34-9 20-25 37-48 50l8 22c-30-4-52-20-66-48l6-8z" />
        <path d="M246 78c-16-22-37-35-61-39 12 22 31 38 56 49l5-10z" />
        <path d="M118 180c10-58 48-100 106-116 54-15 120-2 165 34 42 34 63 85 56 138-8 57-48 101-105 117-57 16-123 1-169-37-42-35-61-83-53-136z" />
        <path d="M294 222c45-22 105-18 142 9 18 13 16 37-4 45l-104 38c-15 5-31 0-40-12l-31-39c-12-15-3-31 37-41z" />
        <path d="M261 320c-4-28 6-54 24-73 16-16 39-25 60-24 13 1 19 16 11 26-17 21-22 43-17 66 6 28-7 57-34 71-20 10-40 6-44-16z" />
        <ellipse cx="329" cy="385" rx="84" ry="56" />
        <path d="M255 363c10-27 39-43 74-43 37 0 70 18 77 46-21-8-29 8-46 1-16-6-19 11-36 5-17-6-19 12-37 7-15-4-20 7-32 2z" />
        <path d="M233 346c-26 10-47 31-55 56 28-5 55-20 71-40 10-12 6-20-16-16z" />
        <path d="M394 352c37 8 65 31 76 59-31-1-58-15-77-40-10-13-10-20 1-19z" />
        <path d="M305 426c5 0 9 4 8 9l-5 47c-1 5-5 8-10 8-5-1-8-5-8-10l5-46c1-5 5-8 10-8z" />
        <path d="M360 429c5-1 9 3 10 8l4 47c1 5-3 10-8 10-5 1-9-3-10-8l-4-47c-1-5 3-9 8-10z" />
        <path d="M282 478c7-8 21-9 30-2 6 4 7 12 2 17-8 9-20 13-34 11-9-1-12-10-6-18z" />
        <path d="M341 481c8-8 22-9 31-2 6 5 7 12 2 18-8 9-21 13-35 11-9-2-12-11-6-19z" />
        <circle cx="178" cy="408" r="28" />
        <path d="M110 396l24-12 24 12-24 12-24-12z" />
        <path d="M110 396v24l24 12v-24l-24-12z" />
        <path d="M158 396v24l-24 12v-24l24-12z" />
      </g>
      <g
        fill="#E5E7EB"
        stroke="#E5E7EB"
        strokeWidth="12"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M182 228c0-31 25-56 56-56s56 25 56 56-25 56-56 56-56-25-56-56z" />
        <path d="M278 236c0-34 28-62 62-62s62 28 62 62-28 62-62 62-62-28-62-62z" />
      </g>
      <g
        fill="#111111"
        stroke="#FFFFFF"
        strokeWidth="12"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="242" cy="232" r="20" />
        <circle cx="340" cy="240" r="22" />
      </g>
      <g
        fill="none"
        stroke="#FFFFFF"
        strokeWidth="12"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M166 408l9 9 16-18" />
      </g>
    </svg>
  );
}

export function ChatPane() {
  const {
    runAudit,
    addDocument,
    isGenerating,
    currentRunStepLabel,
    state,
    lastAssistantMessage,
    stepActivity,
  } = useAuditAgent();
  const { addUploadedDoc, updateUploadedDoc } = useUploadedDocs();
  const [message, setMessage] = useState("");
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [collapsedToolBubbleIds, setCollapsedToolBubbleIds] = useState<
    Set<string>
  >(new Set());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Welcome to Q-Bot, your AI-powered TFR audit assistant. Upload documents and I'll generate a custom audit questionnaire with insights. You can upload .pdf, .docx, or .xlsx files.",
    },
  ]);

  const lastShownRef = useRef<string | null>(null);

  useEffect(() => {
    if (!lastAssistantMessage) return;
    if (isGenerating) return;
    if (lastAssistantMessage === lastShownRef.current) return;

    lastShownRef.current = lastAssistantMessage;
    setChatMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content: lastAssistantMessage,
      },
    ]);
  }, [lastAssistantMessage, isGenerating]);

  // Auto-scroll when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, isGenerating]);

  const uploadFile = useCallback(
    async (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase() || "pdf";
      const friendlySize =
        file.size > 1024 * 1024
          ? `${(file.size / 1024 / 1024).toFixed(1)} MB`
          : `${(file.size / 1024).toFixed(1)} KB`;
      const nowIso = new Date().toISOString();
      const dummyContentId = crypto.randomUUID();

      addUploadedDoc({
        file_name: file.name,
        claim_number: "CLM-UPLOAD-000",
        content_id: dummyContentId,
        mime_type: mimeFromExt(ext),
        content_url: file.name,
        domain: "claim",
        document_type: "Upload",
        document_description: `Uploading ${friendlySize}...`,
        create_date: nowIso,
        source_system: "UPLOAD",
      });

      try {
        const formData = new FormData();
        formData.append("file", file);
        const resp = await fetch(`${BACKEND_URL}/upload`, {
          method: "POST",
          body: formData,
        });

        if (!resp.ok) {
          console.error(`Upload failed for ${file.name}:`, await resp.text());
          updateUploadedDoc(file.name, {
            file_name: file.name,
            claim_number: "CLM-UPLOAD-000",
            content_id: dummyContentId,
            mime_type: mimeFromExt(ext),
            content_url: file.name,
            domain: "claim",
            document_type: "Upload",
            document_description: "Upload failed",
            create_date: nowIso,
            source_system: "UPLOAD",
          });
          return;
        }

        const data = await resp.json();

        updateUploadedDoc(file.name, {
          file_name: data.filename || file.name,
          claim_number: "CLM-UPLOAD-000",
          content_id: dummyContentId,
          mime_type: mimeFromExt(data.file_type || ext),
          content_url: data.path || file.name,
          domain: "claim",
          document_type: "Upload",
          document_description: `${data.file_size || friendlySize}, ${data.page_count ?? 0} pages`,
          create_date: nowIso,
          source_system: "UPLOAD",
        });

        addDocument({
          file_name: data.filename || file.name,
          claim_number: "CLM-UPLOAD-000",
          content_id: dummyContentId,
          mime_type: mimeFromExt(data.file_type || ext),
          content_url: data.path || file.name,
          domain: "claim",
          document_type: "Upload",
          document_description: "",
          create_date: nowIso,
          source_system: "UPLOAD",
          content: data.content ?? "",
        });
      } catch (err) {
        console.error(`Upload error for ${file.name}:`, err);
        updateUploadedDoc(file.name, {
          file_name: file.name,
          claim_number: "CLM-UPLOAD-000",
          content_id: dummyContentId,
          mime_type: mimeFromExt(ext),
          content_url: file.name,
          domain: "claim",
          document_type: "Upload",
          document_description: "Upload failed",
          create_date: nowIso,
          source_system: "UPLOAD",
        });
        addDocument({
          file_name: file.name,
          claim_number: "CLM-UPLOAD-000",
          content_id: dummyContentId,
          mime_type: mimeFromExt(ext),
          content_url: file.name,
          domain: "claim",
          document_type: "Upload",
          document_description: "",
          create_date: nowIso,
          source_system: "UPLOAD",
          content: "",
        });
      }
    },
    [addDocument, addUploadedDoc, updateUploadedDoc]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      const validFiles = files.filter((f) => {
        const ext = f.name.split(".").pop()?.toLowerCase();
        return ["pdf", "docx", "xlsx"].includes(ext || "");
      });
      setAttachedFiles((prev) => [...prev, ...validFiles]);
      validFiles.forEach((f) => uploadFile(f));
    },
    [uploadFile]
  );

  const removeFile = useCallback((index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleSend = useCallback(async () => {
    if (!message.trim() && attachedFiles.length === 0) return;

    const userContent =
      message.trim() ||
      "Please analyze the uploaded documents and generate a TFR audit questionnaire.";

    const fileNames = attachedFiles.map((f) => f.name);
    const toolStatusMessageId = crypto.randomUUID();

    setChatMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "user",
        content:
          userContent +
          (fileNames.length > 0
            ? `\n\n[Attached ${fileNames.length} file(s): ${fileNames.join(", ")}]`
            : ""),
      },
      {
        id: toolStatusMessageId,
        role: "tool_status",
        steps: [],
        isLive: true,
      },
    ]);
    setCollapsedToolBubbleIds((prev) => {
      const next = new Set(prev);
      next.delete(toolStatusMessageId);
      return next;
    });

    setMessage("");
    setAttachedFiles([]);

    try {
      await runAudit(userContent);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            "I encountered an error during analysis. Please try again.",
        },
      ]);
    }
  }, [message, attachedFiles, runAudit]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const renderChatMessage = useCallback(
    (msg: UserAssistantChatMessage) => (
      <div
        key={msg.id}
        className={`flex gap-2.5 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
      >
        {msg.role === "assistant" && (
          <div className="shrink-0 mt-0.5 h-8 w-8 rounded-full bg-primary/15 flex items-center justify-center">
            <QBotAvatarIcon className="h-5 w-5 text-primary" />
          </div>
        )}
        <div
          className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
            msg.role === "user"
              ? "bg-primary/15 text-foreground border border-primary/25"
              : "bg-secondary/60 text-foreground/90 border border-border/30"
          }`}
        >
          <p className="whitespace-pre-wrap">{msg.content}</p>
        </div>
        {msg.role === "user" && (
          <div className="shrink-0 mt-0.5 h-8 w-8 rounded-full bg-primary/15 flex items-center justify-center">
            <User className="h-5 w-5 text-primary" />
          </div>
        )}
      </div>
    ),
    []
  );

  // Keep each run's latest tool-status message synced with live state.
  useEffect(() => {
    setChatMessages((prev) => {
      const liveIndex = [...prev]
        .map((msg, idx) =>
          msg.role === "tool_status" && msg.isLive ? idx : -1
        )
        .filter((idx) => idx >= 0)
        .pop();

      if (liveIndex === undefined) {
        return prev;
      }

      const current = prev[liveIndex];
      if (current.role !== "tool_status") {
        return prev;
      }

      const nextIsLive = isGenerating;
      const stepsChanged = !areStepListsEqual(current.steps, stepActivity);
      const liveChanged = current.isLive !== nextIsLive;

      if (!stepsChanged && !liveChanged) {
        return prev;
      }

      const next = [...prev];
      next[liveIndex] = {
        ...current,
        steps: stepActivity,
        isLive: nextIsLive,
      };

      // Default completed tool bubbles to collapsed once streaming finishes.
      if (current.isLive && !nextIsLive) {
        setCollapsedToolBubbleIds((prevCollapsed) => {
          const collapsedNext = new Set(prevCollapsed);
          collapsedNext.add(current.id);
          return collapsedNext;
        });
      }

      return next;
    });
  }, [stepActivity, isGenerating]);

  return (
    <div className="flex flex-col h-full">
      {/* Chat header */}
      <div className="flex items-center gap-2 px-4 py-3 pr-10 border-b border-border/50">
        <Send className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold text-foreground">Chat</h2>
        {isGenerating && (
          <Badge variant="outline" className="ml-auto text-[10px] gap-1">
            <Loader2 className="h-3 w-3 animate-spin" />
            {currentRunStepLabel}
          </Badge>
        )}
      </div>

      {/* Chat messages */}
      <ScrollArea className="flex-1">
        <div className="px-4 py-4 space-y-4">
          {chatMessages.map((msg) => {
            if (msg.role !== "tool_status") {
              return renderChatMessage(msg);
            }

            const isCollapsed =
              !msg.isLive && collapsedToolBubbleIds.has(msg.id);
            const toolBubbleSummary = getToolBubbleSummary(msg.steps);

            return (
              <div key={msg.id} className="flex gap-2.5 justify-start">
                <div className="shrink-0 mt-0.5 h-8 w-8 rounded-full bg-primary/15 flex items-center justify-center">
                  <QBotAvatarIcon className="h-5 w-5 text-primary" />
                </div>
                <div className="bg-secondary/60 rounded-xl px-4 py-3 border border-border/30 min-w-[220px]">
                  {msg.steps.length > 0 && (
                    <>
                      <button
                        type="button"
                        onClick={() =>
                          setCollapsedToolBubbleIds((prev) => {
                            if (msg.isLive) {
                              return prev;
                            }
                            const next = new Set(prev);
                            if (next.has(msg.id)) {
                              next.delete(msg.id);
                            } else {
                              next.add(msg.id);
                            }
                            return next;
                          })
                        }
                        className="w-full flex items-center justify-between gap-2 text-xs text-muted-foreground hover:text-foreground/80 transition-colors"
                      >
                        <span>{toolBubbleSummary}</span>
                        {isCollapsed ? (
                          <ChevronRight className="h-3.5 w-3.5 shrink-0" />
                        ) : (
                          <ChevronDown className="h-3.5 w-3.5 shrink-0" />
                        )}
                      </button>

                      {!isCollapsed && (
                        <div className="space-y-1.5 mt-2">
                          {msg.steps.map((step) => (
                            <div
                              key={`${msg.id}-${step.id}`}
                              className="flex items-center gap-2 text-sm"
                            >
                              {step.status === "in_progress" ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary shrink-0" />
                              ) : step.status === "error" ? (
                                <AlertCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
                              ) : (
                                <Check className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                              )}
                              <span
                                title={step.message}
                                className={
                                  step.status === "completed"
                                    ? "text-muted-foreground"
                                    : step.status === "error"
                                      ? "text-destructive"
                                      : "text-foreground/80"
                                }
                              >
                                {truncateToolMessage(
                                  step.message,
                                  TOOL_MESSAGE_MAX_CHARS
                                )}
                              </span>
                            </div>
                          ))}
                          {msg.isLive &&
                            msg.steps.every(
                              (step) => step.status !== "in_progress"
                            ) && (
                              <div className="flex items-center gap-2 text-sm text-muted-foreground pt-0.5">
                                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary shrink-0" />
                                <span>Working...</span>
                              </div>
                            )}
                        </div>
                      )}
                    </>
                  )}

                  {msg.isLive && msg.steps.length === 0 && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      <span>{currentRunStepLabel}</span>
                    </div>
                  )}

                  {msg.isLive && state.progress > 0 && (
                    <div className="mt-2 h-1.5 bg-secondary rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-linear-to-r from-accent to-primary rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${state.progress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Attached files preview */}
      {attachedFiles.length > 0 && (
        <div className="px-4 py-2 border-t border-border/30 bg-secondary/20">
          <div className="flex flex-wrap gap-2">
            {attachedFiles.map((file, idx) => (
              <div
                key={idx}
                className="flex items-center gap-1.5 bg-secondary/50 rounded-lg px-2.5 py-1 text-xs"
              >
                <FileText className="h-3 w-3 text-primary" />
                <span className="text-foreground/80 max-w-[120px] truncate">
                  {file.name}
                </span>
                <button
                  onClick={() => removeFile(idx)}
                  className="text-muted-foreground hover:text-destructive ml-1"
                >
                  &times;
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-border/50 px-4 py-3">
        <div className="flex items-end gap-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.xlsx"
            onChange={handleFileSelect}
            className="hidden"
          />
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                className="shrink-0 h-9 w-9 text-muted-foreground hover:text-primary"
              >
                <Paperclip className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Attach documents (.pdf, .docx, .xlsx)</TooltipContent>
          </Tooltip>

          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your documents or request a TFR audit..."
            rows={1}
            className="flex-1 resize-none bg-secondary/50 border border-border/50 rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 focus:border-primary/50 min-h-[36px] max-h-[120px]"
            style={{
              height: "auto",
              overflowY:
                message.split("\n").length > 3 ? "auto" : "hidden",
            }}
          />

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="default"
                size="icon"
                onClick={handleSend}
                disabled={
                  isGenerating ||
                  (!message.trim() && attachedFiles.length === 0)
                }
                className="shrink-0 h-9 w-9"
              >
                {isGenerating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>Send message</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}
