"use client";

/**
 * TextBox Component
 *
 * General-purpose text display for insights, summaries, and analysis results.
 * Supports multiple visual variants: info, warning, success, error.
 */

import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Info, AlertTriangle, CheckCircle, XCircle } from "lucide-react";

export interface TextBoxProps {
  title: string;
  content: string;
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

  return (
    <Card className={`${config.border} ${config.bg}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          {config.icon}
          <h3 className={`text-sm font-semibold ${config.titleColor}`}>
            {title}
          </h3>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-foreground/80 leading-relaxed whitespace-pre-wrap">
          {content}
        </p>
      </CardContent>
    </Card>
  );
}

export default TextBox;
