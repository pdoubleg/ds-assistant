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

import React, { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import { useTheme } from "next-themes";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneDark,
  oneLight,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Info,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Lightbulb,
  CircleAlert,
  ShieldAlert,
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

/**
 * Converts markdown code children into a plain string for rendering.
 *
 * Args:
 *   children: React children inside a markdown code node.
 *
 * Returns:
 *   A normalized string representation of the code contents.
 */
function toCodeString(children: React.ReactNode): string {
  if (typeof children === "string") {
    return children.replace(/\n$/, "");
  }

  if (Array.isArray(children)) {
    return children
      .map((child) => (typeof child === "string" ? child : ""))
      .join("")
      .replace(/\n$/, "");
  }

  return "";
}

const LIGHT_CODE_THEME = oneLight as Record<string, React.CSSProperties>;
const DARK_CODE_THEME = oneDark as Record<string, React.CSSProperties>;

type CalloutType = "note" | "tip" | "important" | "warning" | "caution";

const CALLOUT_CONFIG: Record<
  CalloutType,
  {
    label: string;
    icon: React.ReactElement;
    className: string;
  }
> = {
  note: {
    label: "Note",
    icon: <Info className="h-4 w-4 text-blue-600 dark:text-blue-300" />,
    className: "border-blue-500/35 bg-blue-500/8",
  },
  tip: {
    label: "Tip",
    icon: <Lightbulb className="h-4 w-4 text-emerald-600 dark:text-emerald-300" />,
    className: "border-emerald-500/35 bg-emerald-500/8",
  },
  important: {
    label: "Important",
    icon: <CircleAlert className="h-4 w-4 text-violet-600 dark:text-violet-300" />,
    className: "border-violet-500/35 bg-violet-500/8",
  },
  warning: {
    label: "Warning",
    icon: <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-300" />,
    className: "border-amber-500/35 bg-amber-500/8",
  },
  caution: {
    label: "Caution",
    icon: <ShieldAlert className="h-4 w-4 text-red-600 dark:text-red-300" />,
    className: "border-red-500/35 bg-red-500/8",
  },
};

/**
 * Converts a React node tree into plain text.
 *
 * Args:
 *   node: React node content.
 *
 * Returns:
 *   A plain text representation of the node content.
 */
function nodeToPlainText(node: React.ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }

  if (Array.isArray(node)) {
    return node.map((part) => nodeToPlainText(part)).join("");
  }

  if (React.isValidElement<{ children?: React.ReactNode }>(node)) {
    return nodeToPlainText(node.props.children);
  }

  return "";
}

/**
 * Detects markdown-style callout markers at the start of a string.
 *
 * Supported markers:
 *   [!NOTE], [!TIP], [!IMPORTANT], [!WARNING], [!CAUTION]
 *
 * Args:
 *   value: Candidate text value.
 *
 * Returns:
 *   Parsed callout type and body, or ``null`` when no marker is present.
 */
function parseCalloutMarker(
  value: string,
): { type: CalloutType; body: string } | null {
  const match = value.trimStart().match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/i);
  if (!match) {
    return null;
  }

  const normalized = match[1].toLowerCase() as CalloutType;
  return {
    type: normalized,
    body: value.trimStart().replace(match[0], ""),
  };
}

/**
 * Extracts callout metadata and preserves inline markdown nodes.
 *
 * Args:
 *   children: React children for a paragraph-like markdown node.
 *
 * Returns:
 *   Parsed callout type and body nodes when a callout marker is found.
 */
function parseCalloutChildren(
  children: React.ReactNode,
): { type: CalloutType; bodyNodes: React.ReactNode[] } | null {
  const nodes = React.Children.toArray(children);
  const firstNode = nodes[0];

  if (typeof firstNode !== "string") {
    return null;
  }

  const match = firstNode.trimStart().match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/i);
  if (!match) {
    return null;
  }

  const type = match[1].toLowerCase() as CalloutType;
  const markerStart = firstNode.search(/\[!/);
  const markerLength = match[0].length;
  const textBeforeMarker = markerStart > 0 ? firstNode.slice(0, markerStart) : "";
  const textAfterMarker = firstNode.slice(markerStart + markerLength);
  const normalizedFirst = `${textBeforeMarker}${textAfterMarker}`;

  const bodyNodes: React.ReactNode[] = [];
  if (normalizedFirst.length > 0) {
    bodyNodes.push(normalizedFirst);
  }
  bodyNodes.push(...nodes.slice(1));

  return { type, bodyNodes };
}

/**
 * Renders Mermaid diagrams from fenced markdown code blocks.
 *
 * Example usage:
 *   ```mermaid
 *   graph LR
 *     A[Start] --> B[Done]
 *   ```
 */
function MermaidBlock({
  code,
  isDarkTheme,
}: {
  code: string;
  isDarkTheme: boolean;
}): React.ReactElement {
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string>("");

  const safeId = useMemo(
    () => `mermaid-${Math.random().toString(36).slice(2)}`,
    [],
  );

  useEffect(() => {
    let isMounted = true;

    async function renderMermaid(): Promise<void> {
      try {
        const mermaid = (await import("mermaid")).default;

        // Keep Mermaid in strict mode so raw HTML is not executed.
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: isDarkTheme ? "dark" : "default",
        });

        const result = await mermaid.render(safeId, code);

        if (isMounted) {
          setSvg(result.svg);
          setError("");
        }
      } catch {
        if (isMounted) {
          setError("Unable to render Mermaid diagram.");
          setSvg("");
        }
      }
    }

    void renderMermaid();

    return () => {
      isMounted = false;
    };
  }, [code, isDarkTheme, safeId]);

  if (error) {
    return (
      <pre>
        <code>{code}</code>
      </pre>
    );
  }

  if (!svg) {
    return <div className="text-xs text-muted-foreground">Rendering diagram...</div>;
  }

  return (
    <div
      className="my-2 overflow-x-auto rounded-md border border-border bg-card p-2"
      // Mermaid returns SVG markup; this is the intended rendering path.
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

export function TextBox({
  title,
  content,
  variant = "info",
}: TextBoxProps): React.ReactElement {
  const config = VARIANT_CONFIG[variant];
  const [copied, setCopied] = useState(false);
  const { resolvedTheme } = useTheme();
  const isDarkTheme = resolvedTheme === "dark";

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

  const markdownComponents = useMemo<Components>(
    () => ({
      code({ className, children }) {
        const codeString = toCodeString(children);
        const languageMatch = /language-(\w+)/.exec(className ?? "");
        const language = languageMatch?.[1]?.toLowerCase();
        const isInlineCode = !language && !codeString.includes("\n");

        if (isInlineCode) {
          return (
            <code className={className}>
              {children}
            </code>
          );
        }

        if (language === "mermaid") {
          return <MermaidBlock code={codeString} isDarkTheme={isDarkTheme} />;
        }

        return (
          <SyntaxHighlighter
            language={language}
            style={isDarkTheme ? DARK_CODE_THEME : LIGHT_CODE_THEME}
            PreTag="pre"
            customStyle={{
              margin: "0.6em 0",
              borderRadius: "8px",
              padding: "0.85em 1em",
              background: "var(--muted)",
            }}
            codeTagProps={{ style: { fontFamily: "var(--font-mono)" } }}
          >
            {codeString}
          </SyntaxHighlighter>
        );
      },
      blockquote({ children }) {
        const nodes = React.Children.toArray(children);
        const firstNode = nodes[0] ?? null;
        const firstParagraphChildren =
          React.isValidElement<{ children?: React.ReactNode }>(firstNode)
            ? firstNode.props.children
            : null;
        const parsedCallout = firstParagraphChildren
          ? parseCalloutChildren(firstParagraphChildren)
          : null;

        if (!parsedCallout) {
          return <blockquote>{children}</blockquote>;
        }

        const { type, bodyNodes } = parsedCallout;
        const config = CALLOUT_CONFIG[type];
        const remainingNodes = nodes.slice(1);

        return (
          <div className={`textbox-callout my-3 rounded-md border p-3 ${config.className}`}>
            <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide">
              {config.icon}
              <span>{config.label}</span>
            </div>
            {bodyNodes.length > 0 ? <p className="mb-2">{bodyNodes}</p> : null}
            {remainingNodes}
          </div>
        );
      },
      a({ href, children, ...props }) {
        const isExternal = Boolean(href?.startsWith("http"));
        return (
          <a
            href={href}
            target={isExternal ? "_blank" : undefined}
            rel={isExternal ? "noopener noreferrer" : undefined}
            {...props}
          >
            {children}
          </a>
        );
      },
      input({ type, checked, ...props }) {
        if (type !== "checkbox") {
          return <input type={type} {...props} />;
        }

        return (
          <input
            type="checkbox"
            defaultChecked={Boolean(checked)}
            className="mr-2 h-4 w-4 rounded border border-border accent-primary"
            onChange={() => {
              // Intentionally local-only state: lets users mark checklist progress in the UI.
            }}
            aria-label="Checklist item"
          />
        );
      },
      p({ children }) {
        const parsedCallout = parseCalloutChildren(children);

        if (!parsedCallout) {
          return <p>{children}</p>;
        }

        const config = CALLOUT_CONFIG[parsedCallout.type];
        return (
          <div className={`textbox-callout my-3 rounded-md border p-3 ${config.className}`}>
            <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide">
              {config.icon}
              <span>{config.label}</span>
            </div>
            <p className="mb-0">{parsedCallout.bodyNodes}</p>
          </div>
        );
      },
      sup({ children, ...props }) {
        return (
          <sup className="text-[0.7rem] font-medium align-super text-primary" {...props}>
            {children}
          </sup>
        );
      },
    }),
    [isDarkTheme],
  );

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
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeRaw, rehypeSanitize, rehypeKatex]}
            components={markdownComponents}
          >
            {content}
          </ReactMarkdown>
        </div>
      </CardContent>
    </Card>
  );
}

export default TextBox;
