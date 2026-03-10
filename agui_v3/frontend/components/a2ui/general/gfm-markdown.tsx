"use client";

/**
 * GfmMarkdown Component
 *
 * Lightweight GitHub-flavored Markdown renderer for compact UI surfaces such as
 * document summaries. It supports common GFM features like tables, lists,
 * checklists, links, blockquotes, and inline code without the heavier TextBox
 * features like math, Mermaid, or raw HTML rendering.
 *
 * Example usage:
 *   <GfmMarkdown content="## Summary\n- First item\n- Second item" compact />
 */

import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

export interface GfmMarkdownProps {
  /** Markdown content rendered with common GFM support. */
  content: string;
  /** Optional wrapper classes for sizing and color. */
  className?: string;
  /** Tightens spacing for compact surfaces such as cards. */
  compact?: boolean;
}

/**
 * Returns whether a link should open in a new tab.
 *
 * Args:
 *   href: Candidate href value from the markdown AST.
 *
 * Returns:
 *   ``true`` when the URL is external, otherwise ``false``.
 */
function isExternalHref(href?: string): boolean {
  return Boolean(href?.startsWith("http://") || href?.startsWith("https://"));
}

/**
 * Normalizes markdown that may contain double-escaped line breaks.
 *
 * Some structured LLM outputs occasionally return literal ``\\n`` sequences
 * instead of actual newline characters. Markdown tables and lists require real
 * line breaks, so this helper converts the common escaped whitespace sequences
 * back into their intended form before rendering.
 *
 * Args:
 *   content: Raw markdown text from the backend.
 *
 * Returns:
 *   Markdown text with escaped line break sequences normalized.
 */
function normalizeMarkdownContent(content: string): string {
  if (!content.includes("\\n") && !content.includes("\\r") && !content.includes("\\t")) {
    return content;
  }

  return content
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t");
}

export function GfmMarkdown({
  content,
  className,
  compact = false,
}: GfmMarkdownProps): React.ReactElement {
  const normalizedContent = useMemo(() => normalizeMarkdownContent(content), [content]);
  const components = useMemo<Components>(
    () => ({
      a({ href, children, ...props }) {
        const isExternal = isExternalHref(href);
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

        // Keep checklist items visually accurate while preventing card-local edits.
        return (
          <input
            type="checkbox"
            checked={Boolean(checked)}
            readOnly
            tabIndex={-1}
            className="mr-2 h-4 w-4 rounded border border-border accent-primary pointer-events-none"
            aria-label="Checklist item"
          />
        );
      },
      table({ children }) {
        return (
          <div className="gfm-markdown-table-wrap">
            <table>{children}</table>
          </div>
        );
      },
    }),
    [],
  );

  return (
    <div
      className={cn("gfm-markdown", compact && "gfm-markdown-compact", className)}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {normalizedContent}
      </ReactMarkdown>
    </div>
  );
}

export default GfmMarkdown;
