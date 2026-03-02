"use client";

/**
 * TextBox Component
 *
 * General-purpose text display for insights, summaries, and analysis results.
 * Supports multiple visual variants: info, warning, success, error.
 * Renders content as GitHub-flavored Markdown and provides a copy-to-clipboard
 * button in the card header.
 *
 * Example usage:
 *   <TextBox
 *     title="Summary"
 *     content="**Bold insight** with a [link](https://example.com)."
 *     variant="success"
 *   />
 */

import React, { useCallback, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Info,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Copy,
  Check,
} from "lucide-react";

export interface TextBoxProps {
  /** Heading displayed next to the variant icon. */
  title: string;
  /** Markdown-compatible content string. */
  content: string;
  /** Visual treatment: info (default), warning, success, or error. */
  variant?: "info" | "warning" | "success" | "error";
}

const VARIANT_CONFIG = {
  info: {
    icon: <Info className="h-5 w-5 text-primary" />,
    border: "border-primary/25",
    bg: "bg-linear-to-br from-primary/5 to-primary/10",
    titleColor: "text-foreground",
  },
  warning: {
    icon: (
      <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
    ),
    border: "border-amber-500/30",
    bg: "bg-linear-to-br from-amber-500/8 to-amber-400/8",
    titleColor: "text-amber-700 dark:text-amber-300",
  },
  success: {
    icon: (
      <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
    ),
    border: "border-emerald-500/30",
    bg: "bg-linear-to-br from-emerald-500/8 to-green-500/8",
    titleColor: "text-emerald-700 dark:text-emerald-300",
  },
  error: {
    icon: <XCircle className="h-5 w-5 text-red-600 dark:text-red-400" />,
    border: "border-red-500/30",
    bg: "bg-linear-to-br from-red-500/8 to-rose-500/8",
    titleColor: "text-red-700 dark:text-red-300",
  },
};

export function TextBox({
  title,
  content,
  variant = "info",
}: TextBoxProps): React.ReactElement {
  const config = VARIANT_CONFIG[variant];
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      // Reset the checkmark back to the copy icon after a short delay
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard access may be denied in some contexts */
    }
  }, [content]);

  return (
    <Card className={`${config.border} ${config.bg}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {config.icon}
            <h3 className={`text-sm font-semibold ${config.titleColor}`}>
              {title}
            </h3>
          </div>
          <button
            type="button"
            onClick={handleCopy}
            aria-label={copied ? "Copied" : "Copy content"}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            {copied ? (
              <Check className="h-4 w-4 text-emerald-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="textbox-markdown text-sm text-foreground/80 leading-relaxed">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content}
          </ReactMarkdown>
        </div>
      </CardContent>
    </Card>
  );
}

export default TextBox;
