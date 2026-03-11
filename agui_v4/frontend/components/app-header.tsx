"use client";

/**
 * AppHeader — top navigation bar with branding, nav links, and theme toggle.
 * Uses next-themes for SSR-safe dark/light mode switching.
 */

import React from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sora } from "next/font/google";
import { useTheme } from "next-themes";
import {
  CalendarDays,
  Eye,
  Home,
  Info,
  LayoutDashboard,
  Moon,
  Sun,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  ClaimNumberDialog,
  type ClaimSessionData,
} from "@/components/claim-number-dialog";
import {
  useClaimSessionState,
  type ClaimSessionState,
} from "@/hooks/use-audit-agent";

const headerFont = Sora({
  subsets: ["latin"],
  weight: ["500"],
});

const NAV_LINKS = [
  { href: "/", label: "Home", icon: Home },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/about", label: "About", icon: Info },
] as const;

const ACTIVE_SESSION_TILE_CLASSES =
  "shadow-xl shadow-sky-300/35 ring-2 ring-sky-400/80 hover:shadow-sky-400/55 hover:ring-sky-500/80 dark:shadow-cyan-950/40 dark:ring-cyan-600/80 dark:hover:ring-cyan-500/80";

const INACTIVE_SESSION_TILE_CLASSES =
  "shadow-lg shadow-slate-300/30 ring-1 ring-slate-300/80 hover:ring-slate-400/80 dark:shadow-slate-950/40 dark:ring-slate-700/80 dark:hover:ring-slate-600/80";

const HEADER_SHELL_CLASSES =
  "relative overflow-hidden shrink-0 border-b border-stone-200/90 bg-[linear-gradient(180deg,rgba(250,250,249,0.97)_0%,rgba(245,245,244,0.985)_100%)] shadow-[0_10px_30px_-24px_rgba(41,37,36,0.45)] backdrop-blur-xl dark:border-slate-700/75 dark:bg-[linear-gradient(180deg,rgba(51,65,85,0.95)_0%,rgba(30,41,59,0.985)_100%)] dark:shadow-[0_16px_36px_-28px_rgba(0,0,0,0.85)]";

const HEADER_BUTTON_CLASSES =
  "h-10 rounded-xl border border-stone-200/80 bg-stone-50/85 text-stone-700 shadow-sm shadow-stone-300/30 backdrop-blur-md hover:bg-stone-50 hover:text-stone-950 dark:border-slate-700/80 dark:bg-slate-900/55 dark:text-slate-300 dark:shadow-black/20 dark:hover:bg-slate-900/80 dark:hover:text-slate-50";

/**
 * Format an ISO date string for the compact header display.
 *
 * Args:
 *   effectiveDate: Raw ISO date string (`YYYY-MM-DD`).
 *
 * Returns:
 *   A readable date label for the Now Viewing card.
 */
function formatEffectiveDate(effectiveDate: string): string {
  if (!effectiveDate) {
    return "";
  }

  const parsedDate = new Date(`${effectiveDate}T00:00:00`);
  if (Number.isNaN(parsedDate.getTime())) {
    return effectiveDate;
  }

  return new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  }).format(parsedDate);
}

interface NowViewingCardProps {
  claimSession: ClaimSessionState;
}

/**
 * Compact status card that surfaces the currently active claim context.
 *
 * Args:
 *   claimSession: Active claim-session payload from shared AG-UI state.
 *
 * Returns:
 *   A styled header card that highlights whether claim mode is active.
 */
function NowViewingCard({ claimSession }: NowViewingCardProps) {
  const hasActiveClaim = Boolean(claimSession.claimNumber);
  const formattedDate = formatEffectiveDate(claimSession.effectiveDate);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          className={cn(
            "flex h-[76px] min-w-0 items-center rounded-2xl bg-white/95 px-4 py-3 dark:bg-slate-950/95",
            hasActiveClaim
              ? ACTIVE_SESSION_TILE_CLASSES
              : `${INACTIVE_SESSION_TILE_CLASSES} opacity-75`
          )}
        >
          <div className="flex items-start gap-3">
            <div
              className={cn(
                "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border",
                hasActiveClaim
                  ? "border-sky-200 bg-white text-sky-700 dark:border-cyan-800 dark:bg-slate-950 dark:text-cyan-300"
                  : "border-slate-200 bg-white text-slate-400 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-500"
              )}
            >
              <Eye className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500 dark:text-slate-400">
                Now Viewing
              </p>
              {hasActiveClaim ? (
                <div className="mt-1 flex min-w-0 items-center gap-2">
                  <p className="truncate text-sm font-semibold text-slate-900 dark:text-slate-50">
                    {claimSession.claimNumber}
                  </p>
                  {formattedDate ? (
                    <div className="flex min-w-0 items-center gap-1.5 text-xs italic text-slate-500 dark:text-slate-400">
                      <CalendarDays className="h-3.5 w-3.5 shrink-0 not-italic" />
                      <span className="truncate">Data as of {formattedDate}</span>
                    </div>
                  ) : null}
                </div>
              ) : (
                <p className="text-sm font-semibold text-slate-600 dark:text-slate-300">
                  Local Mode
                </p>
              )}
              {!hasActiveClaim ? (
                <p className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400">
                  Local mode active. Use sample docs or uploads.
                </p>
              ) : null}
            </div>
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent>
        Click the Q-Bot icon to enter or update claim details. You can also
        stay in local mode and use sample docs or your own uploads.
      </TooltipContent>
    </Tooltip>
  );
}

export function AppHeader() {
  const { theme, setTheme } = useTheme();
  const pathname = usePathname();
  const [mounted, setMounted] = React.useState(false);
  const { claimSession, setClaimSession } = useClaimSessionState();
  const hasActiveClaim = Boolean(claimSession.claimNumber);

  React.useEffect(() => setMounted(true), []);

  const handleNewSession = React.useCallback(
    async (data: ClaimSessionData) => {
      setClaimSession(data);

      // Force a full reload so every pane re-initializes against the new claim.
      window.setTimeout(() => {
        window.location.reload();
      }, 120);
    },
    [setClaimSession]
  );

  return (
    <header className={HEADER_SHELL_CLASSES}>
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.42),transparent_18%,transparent_82%,rgba(214,211,209,0.24))] dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.045),transparent_20%,transparent_78%,rgba(148,163,184,0.035))]" />
        <div className="absolute inset-0 bg-[repeating-linear-gradient(90deg,rgba(168,162,158,0.06)_0,rgba(168,162,158,0.06)_1px,transparent_1px,transparent_14px)] opacity-40 dark:bg-[repeating-linear-gradient(90deg,rgba(226,232,240,0.03)_0,rgba(226,232,240,0.03)_1px,transparent_1px,transparent_14px)] dark:opacity-35" />
        <div className="absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-white/80 to-transparent dark:via-slate-200/20" />
      </div>
      <div className="relative flex items-center gap-4 px-5 py-4">
        <div className="flex shrink-0 items-center gap-5">
          {/* Logo doubles as the New Audit dialog trigger */}
          <ClaimNumberDialog
            onSubmit={handleNewSession}
            initialData={claimSession}
            trigger={
              <button
                type="button"
                aria-label="Start a new audit session"
                className="qbot-launch-btn relative rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/70 dark:focus-visible:ring-cyan-500/70"
              >
                <Image
                  src="/q-bot.png"
                  alt="Q-Bot"
                  width={76}
                  height={76}
                  className={cn(
                    "relative cursor-pointer rounded-2xl",
                    hasActiveClaim
                      ? ACTIVE_SESSION_TILE_CLASSES
                      : INACTIVE_SESSION_TILE_CLASSES
                  )}
                  priority
                />
                <span
                  aria-hidden="true"
                  className={cn(
                    "absolute -bottom-1 -right-1 h-4 w-4 rounded-full border-2 border-white shadow-sm dark:border-slate-950",
                    hasActiveClaim
                      ? "bg-emerald-500 shadow-emerald-500/50"
                      : "bg-slate-300 dark:bg-slate-700"
                  )}
                />
              </button>
            }
          />
          <div className="min-w-0 w-[360px] max-w-[360px]">
            <NowViewingCard claimSession={claimSession} />
          </div>
        </div>

        <div className="min-w-0 flex-1 flex items-center gap-4">
          <p
            className={`${headerFont.className} hidden min-w-0 flex-1 truncate text-center text-[2rem] font-medium tracking-[0.03em] text-stone-800 xl:block 2xl:text-[2.35rem] dark:text-slate-100`}
          >
            Quality Improvement Workbench
          </p>
        </div>

        {/* Navigation links */}
        <nav className="flex items-center gap-1 shrink-0">
          {NAV_LINKS.map(({ href, label, icon: Icon }) => {
            const isActive =
              href === "/" ? pathname === "/" : pathname.startsWith(href);

            return (
              <Link key={href} href={href}>
                <Button
                  variant="ghost"
                  size="sm"
                  className={cn(
                    `gap-1.5 text-sm font-medium ${HEADER_BUTTON_CLASSES}`,
                    isActive
                      ? "border-stone-300 bg-white text-stone-950 ring-1 ring-stone-300/80 dark:border-slate-600 dark:bg-slate-800/90 dark:text-slate-50 dark:ring-slate-500/60"
                      : ""
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{label}</span>
                </Button>
              </Link>
            );
          })}
        </nav>

        {mounted && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className={`${HEADER_BUTTON_CLASSES} w-10 shrink-0`}
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
    </header>
  );
}
