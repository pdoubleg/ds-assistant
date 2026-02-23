/**
 * TextBox Component
 *
 * General-purpose text display for insights, summaries, and analysis results.
 * Supports multiple visual variants: info, warning, success, error.
 */

import React from 'react';
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Info, AlertTriangle, CheckCircle, XCircle } from "lucide-react";

export interface TextBoxProps {
  /** Heading for the text box */
  title: string;
  /** Main text content */
  content: string;
  /** Visual variant */
  variant?: "info" | "warning" | "success" | "error";
}

const VARIANT_CONFIG = {
  info: {
    icon: <Info className="h-5 w-5 text-accent" />,
    border: "border-accent/25",
    bg: "bg-linear-to-br from-accent/5 to-accent/10",
    titleColor: "text-accent",
  },
  warning: {
    icon: <AlertTriangle className="h-5 w-5 text-amber-500 dark:text-amber-400" />,
    border: "border-amber-500/25",
    bg: "bg-linear-to-br from-amber-500/5 to-amber-400/5",
    titleColor: "text-amber-600 dark:text-amber-300",
  },
  success: {
    icon: <CheckCircle className="h-5 w-5 text-emerald-500 dark:text-emerald-400" />,
    border: "border-emerald-500/25",
    bg: "bg-linear-to-br from-emerald-500/5 to-green-500/5",
    titleColor: "text-emerald-600 dark:text-emerald-300",
  },
  error: {
    icon: <XCircle className="h-5 w-5 text-red-500 dark:text-red-400" />,
    border: "border-red-500/25",
    bg: "bg-linear-to-br from-red-500/5 to-rose-500/5",
    titleColor: "text-red-600 dark:text-red-300",
  },
};

export function TextBox({
  title,
  content,
  variant = "info",
}: TextBoxProps): React.ReactElement {
  const config = VARIANT_CONFIG[variant];

  return (
    <Card className={`${config.border} ${config.bg}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          {config.icon}
          <h3 className={`text-sm font-semibold ${config.titleColor}`}>{title}</h3>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-foreground/80 leading-relaxed whitespace-pre-wrap">{content}</p>
      </CardContent>
    </Card>
  );
}

export default TextBox;
