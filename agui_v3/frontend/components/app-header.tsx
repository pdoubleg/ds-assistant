"use client";

/**
 * AppHeader — top navigation bar with branding, subtitle, and theme toggle.
 * Uses next-themes for SSR-safe dark/light mode switching.
 */

import React from "react";
import Image from "next/image";
import { Space_Grotesk } from "next/font/google";
import { useTheme } from "next-themes";
import { Sun, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";

const headerFont = Space_Grotesk({
  subsets: ["latin"],
  weight: ["600", "700"],
});

export function AppHeader() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  // Avoid hydration mismatch — only show the toggle after mount
  React.useEffect(() => setMounted(true), []);

  return (
    <header className="shrink-0 border-b border-sky-300/70 bg-linear-to-r from-sky-50/95 via-blue-50/95 to-sky-100/95 backdrop-blur-md transition-colors dark:border-cyan-900/60 dark:from-slate-950/95 dark:via-slate-900/95 dark:to-cyan-950/95">
      <div className="relative px-5 py-4 flex items-center gap-5">
        <div className="flex items-center gap-4 shrink-0">
          <div className="relative">
            <Image
              src="/q-bot.PNG"
              alt="Q-Bot"
              width={76}
              height={76}
              className="relative rounded-2xl shadow-xl shadow-sky-300/40 ring-2 ring-sky-300/80 dark:shadow-cyan-950/40 dark:ring-cyan-700/70"
              priority
            />
          </div>
          <h1
            className={`${headerFont.className} text-3xl font-bold tracking-tight text-sky-900 drop-shadow-sm dark:text-cyan-200`}
          >
            Q-Bot
          </h1>
        </div>

        <p
          className={`${headerFont.className} pointer-events-none absolute left-1/2 hidden -translate-x-1/2 select-none text-center text-3xl font-semibold tracking-wide text-sky-800/95 md:block lg:text-4xl dark:text-cyan-100/95`}
        >
          AI-Powered Quality Audit Assistant
        </p>
        <div className="ml-auto" />

        {mounted && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="h-9 w-9 shrink-0 text-sky-700 hover:bg-sky-200/40 hover:text-sky-900 dark:text-cyan-300 dark:hover:bg-cyan-900/40 dark:hover:text-cyan-100"
              >
                {theme === "dark" ? (
                  <Sun className="h-[18px] w-[18px]" />
                ) : (
                  <Moon className="h-[18px] w-[18px]" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Switch to {theme === "dark" ? "light" : "dark"} mode
            </TooltipContent>
          </Tooltip>
        )}
      </div>
      <div className="h-px bg-linear-to-r from-transparent via-sky-400/80 to-transparent dark:via-cyan-600/80" />
    </header>
  );
}
