/**
 * FindingCard Component
 *
 * Alert-style card for agent observations about a claim. Displays a
 * severity-colored left border, title, markdown-rendered content, and
 * an optional category badge.
 */

import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Info, AlertTriangle, AlertOctagon } from "lucide-react";

export interface FindingCardProps {
  /** Finding headline */
  title: string;
  /** Detailed explanation (supports basic markdown) */
  content: string;
  /** Visual severity level */
  severity?: "info" | "warning" | "critical";
  /** Optional grouping tag */
  category?: string | null;
}

const SEVERITY_CONFIG = {
  info: {
    icon: <Info className="h-4.5 w-4.5 text-blue-500 dark:text-blue-400" />,
    border: "border-l-blue-500 dark:border-l-blue-400",
    bg: "bg-linear-to-br from-blue-500/5 to-blue-400/5",
    titleColor: "text-blue-700 dark:text-blue-300",
    badgeColor: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  },
  warning: {
    icon: <AlertTriangle className="h-4.5 w-4.5 text-amber-500 dark:text-amber-400" />,
    border: "border-l-amber-500 dark:border-l-amber-400",
    bg: "bg-linear-to-br from-amber-500/5 to-amber-400/5",
    titleColor: "text-amber-700 dark:text-amber-300",
    badgeColor: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  },
  critical: {
    icon: <AlertOctagon className="h-4.5 w-4.5 text-red-500 dark:text-red-400" />,
    border: "border-l-red-500 dark:border-l-red-400",
    bg: "bg-linear-to-br from-red-500/5 to-rose-500/5",
    titleColor: "text-red-700 dark:text-red-300",
    badgeColor: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
  },
};

/**
 * Minimal inline markdown renderer for bold and line breaks.
 * Converts **bold** to <strong> and newlines to <br />.
 */
function renderMarkdown(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-foreground">
          {part.slice(2, -2)}
        </strong>
      );
    }
    // Split on newlines for line breaks
    const lines = part.split("\n");
    return lines.map((line, j) => (
      <React.Fragment key={`${i}-${j}`}>
        {j > 0 && <br />}
        {line}
      </React.Fragment>
    ));
  });
}

export function FindingCard({
  title,
  content,
  severity = "info",
  category,
}: FindingCardProps): React.ReactElement {
  const config = SEVERITY_CONFIG[severity];

  return (
    <Card className={`border-l-4 ${config.border} ${config.bg}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          {config.icon}
          <h3 className={`text-sm font-semibold ${config.titleColor}`}>
            {title}
          </h3>
          {category && (
            <Badge
              variant="outline"
              className={`ml-auto text-[11px] font-medium border-0 ${config.badgeColor}`}
            >
              {category}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-foreground/80 leading-relaxed">
          {renderMarkdown(content)}
        </p>
      </CardContent>
    </Card>
  );
}

export default FindingCard;
