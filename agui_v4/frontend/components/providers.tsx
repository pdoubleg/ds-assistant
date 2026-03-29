"use client";

/**
 * Client-side providers wrapper.
 *
 * Combines next-themes ThemeProvider with CopilotKit + AG-UI agent
 * so the entire app has access to theming and the audit agent.
 */

import React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { CopilotKitProvider } from "@copilotkit/react-core/v2";
import { HttpAgent } from "@ag-ui/client";
import { TooltipProvider } from "@/components/ui/tooltip";
import { UploadedDocsProvider } from "@/hooks/use-uploaded-docs";
import { ChatDocsProvider } from "@/hooks/use-chat-docs";
import { ChatHistoryProvider } from "@/hooks/use-chat-history";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

const auditAgent = new HttpAgent({
  url: BACKEND_URL,
  agentId: "audit_agent",
});

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem
      disableTransitionOnChange
    >
      <CopilotKitProvider
        agents__unsafe_dev_only={{
          // Type cast needed: @ag-ui/client version may differ from CopilotKit's bundled copy
          audit_agent: auditAgent as any,
        }}
      >
        <UploadedDocsProvider>
          <ChatDocsProvider>
            <ChatHistoryProvider>
              <TooltipProvider delayDuration={300}>{children}</TooltipProvider>
            </ChatHistoryProvider>
          </ChatDocsProvider>
        </UploadedDocsProvider>
      </CopilotKitProvider>
    </NextThemesProvider>
  );
}
