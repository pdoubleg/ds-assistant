"use client";

/**
 * ClaimNumberDialog — modal for entering a Claim Number and optional metadata
 * before starting an audit session.
 *
 * Usage:
 *   <ClaimNumberDialog onSubmit={(data) => console.log(data)} />
 *
 * The dialog exposes a controlled `open` prop so a parent can also open it
 * programmatically (e.g. on first page load).
 */

import React, { useState, useCallback, useEffect } from "react";
import {
  CalendarDays,
  FilePlus2,
  Hash,
  Loader2,
  RotateCcw,
  Save,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  NativeDialog,
  NativeDialogContent,
  NativeDialogDescription,
  NativeDialogFooter,
  NativeDialogHeader,
  NativeDialogTitle,
  NativeDialogTrigger,
} from "@/components/ui/native-dialog-shadcnui";

// ── Types ─────────────────────────────────────────────────────────────────────

/** Data submitted by the dialog form. */
export interface ClaimSessionData {
  /** Optional claim identifier, e.g. "012345678". */
  claimNumber: string;
  /** Optional ISO date string (YYYY-MM-DD) for the claim effective date. */
  effectiveDate: string;
}

interface ClaimNumberDialogProps {
  /**
   * Called when the user submits valid form data.
   * The dialog closes automatically after this fires.
   */
  onSubmit: (data: ClaimSessionData) => void | Promise<void>;
  /**
   * Controlled open state.  When omitted the dialog manages its own state.
   */
  open?: boolean;
  /** Callback when the open state should change (controlled mode). */
  onOpenChange?: (open: boolean) => void;
  /** Custom trigger element.  Defaults to a "New Audit" button. */
  trigger?: React.ReactNode;
  /** Initial values used to prefill the dialog when reopened. */
  initialData?: Partial<ClaimSessionData>;
}

// ── Component ─────────────────────────────────────────────────────────────────

/**
 * Modal dialog that collects an optional Claim Number and Effective Date
 * before starting an audit session.
 *
 * @example
 * <ClaimNumberDialog
 *   onSubmit={({ claimNumber, effectiveDate }) => startSession(claimNumber, effectiveDate)}
 * />
 */
export function ClaimNumberDialog({
  onSubmit,
  open,
  onOpenChange,
  trigger,
  initialData,
}: ClaimNumberDialogProps) {
  // Local form state ──────────────────────────────────────────────────────────
  const [claimNumber, setClaimNumber] = useState("");
  const [effectiveDate, setEffectiveDate] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Internal open state used when the dialog is uncontrolled ─────────────────
  const [internalOpen, setInternalOpen] = useState(false);
  const isControlled = open !== undefined;
  const isOpen = isControlled ? open : internalOpen;
  const initialClaimNumber = initialData?.claimNumber ?? "";
  const initialEffectiveDate = initialData?.effectiveDate ?? "";

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    // Rehydrate the form from the active session each time the dialog opens.
    setClaimNumber(initialClaimNumber);
    setEffectiveDate(initialEffectiveDate);
  }, [initialClaimNumber, initialEffectiveDate, isOpen]);

  /** Propagate open-state changes to the appropriate handler. */
  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (isControlled) {
        onOpenChange?.(next);
      } else {
        setInternalOpen(next);
      }
    },
    [isControlled, onOpenChange]
  );

  /**
   * Persist a claim-session selection and close the dialog.
   *
   * Args:
   *   nextClaimNumber: Optional claim number to save.
   *   nextEffectiveDate: Optional effective date to save.
   */
  const submitSession = useCallback(
    async (nextClaimNumber: string, nextEffectiveDate: string) => {
      const normalizedClaimNumber = nextClaimNumber.trim();
      const normalizedEffectiveDate = nextEffectiveDate.trim();

      setIsSubmitting(true);

      try {
        await Promise.resolve(
          onSubmit({
            claimNumber: normalizedClaimNumber,
            effectiveDate: normalizedEffectiveDate,
          })
        );

        // Keep the latest submitted values visible if the dialog is reopened.
        setClaimNumber(normalizedClaimNumber);
        setEffectiveDate(normalizedEffectiveDate);
        handleOpenChange(false);
      } finally {
        setIsSubmitting(false);
      }
    },
    [onSubmit, handleOpenChange]
  );

  /** Save the current modal values. */
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      await submitSession(claimNumber, effectiveDate);
    },
    [claimNumber, effectiveDate, submitSession]
  );

  /** Reset the session back to local mode with no claim context. */
  const handleResetToLocalMode = useCallback(async () => {
    await submitSession("", "");
  }, [submitSession]);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <NativeDialog open={isOpen} onOpenChange={handleOpenChange}>
      {/* Trigger — default button shown when no custom trigger is provided */}
      <NativeDialogTrigger asChild>
        {trigger ?? (
          // Solid sky/primary style so it stands apart from the ghost nav buttons
          <Button
            size="sm"
            className="gap-1.5 bg-sky-600 text-white shadow-sm hover:bg-sky-700 dark:bg-cyan-700 dark:hover:bg-cyan-600 dark:text-white"
          >
            <FilePlus2 className="h-4 w-4" />
            <span className="hidden sm:inline">New Audit</span>
          </Button>
        )}
      </NativeDialogTrigger>

      {/* Dialog panel */}
      <NativeDialogContent className="dialog-zinc-surface overflow-hidden p-0 sm:max-w-[520px]">
        <form onSubmit={handleSubmit} noValidate>
          <NativeDialogHeader className="dialog-zinc-section border-b border-zinc-200 px-6 py-5 text-left dark:border-zinc-800">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-zinc-200 bg-zinc-100 text-zinc-700 shadow-sm dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200">
                <Hash className="h-5 w-5" />
              </div>
              <div className="space-y-1">
                <NativeDialogTitle className="text-xl font-semibold text-zinc-950 dark:text-zinc-50">
                  Claim Session Details
                </NativeDialogTitle>
                <NativeDialogDescription className="text-sm leading-6 text-zinc-600 dark:text-zinc-300">
                  Add a claim number to activate claim-aware mode, or leave it
                  blank to keep working in local mode.
                </NativeDialogDescription>
              </div>
            </div>
          </NativeDialogHeader>

          <div className="grid gap-5 px-6 py-6">
            {/* Claim Number — optional ───────────────────────────────── */}
            <div className="grid gap-2">
              <label
                htmlFor="claim-number"
                className="text-sm font-medium text-foreground"
              >
                Claim Number
                <span className="ml-1.5 text-xs font-normal text-muted-foreground">
                  optional
                </span>
              </label>
              <Input
                id="claim-number"
                placeholder="e.g. 012345678"
                value={claimNumber}
                onChange={(e) => setClaimNumber(e.target.value)}
                aria-describedby="claim-help"
                className="h-12 rounded-lg border-zinc-300 bg-white text-base shadow-sm focus-visible:ring-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:focus-visible:ring-zinc-600"
                autoFocus
              />
              <p id="claim-help" className="text-xs text-muted-foreground">
                Leave blank for local mode with sample docs or uploaded files.
              </p>
            </div>

            {/* Effective Date — optional ─────────────────────────────── */}
            <div className="grid gap-2">
              <label
                htmlFor="effective-date"
                className="flex items-center gap-2 text-sm font-medium text-foreground"
              >
                <CalendarDays className="h-4 w-4 text-zinc-500 dark:text-zinc-300" />
                Effective Date
                <span className="text-xs font-normal text-muted-foreground">
                  optional
                </span>
              </label>
              <Input
                id="effective-date"
                type="date"
                value={effectiveDate}
                onChange={(e) => setEffectiveDate(e.target.value)}
                // Suppress the native date picker's browser-default color in
                // dark mode so the field stays consistent with the design system.
                className="h-12 rounded-lg border-zinc-300 bg-white shadow-sm focus-visible:ring-zinc-400 dark:scheme-dark dark:border-zinc-700 dark:bg-zinc-900 dark:focus-visible:ring-zinc-600"
              />
              <p className="text-xs text-muted-foreground">
                Leave blank unless you want to pin guidance to a specific date.
              </p>
            </div>
          </div>

          <NativeDialogFooter className="dialog-zinc-section border-t border-zinc-200 px-6 py-4 dark:border-zinc-800">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              className="h-11 rounded-lg border-zinc-300 bg-white px-5 text-sm font-medium text-zinc-700 shadow-sm hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={handleResetToLocalMode}
              className="h-11 rounded-lg border-zinc-300 bg-white px-5 text-sm font-medium text-zinc-700 shadow-sm hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4" />
              )}
              Reset to Local Mode
            </Button>
            <Button
              type="submit"
              className="h-11 min-w-[168px] rounded-lg border border-zinc-900 bg-zinc-900 px-5 text-sm font-semibold text-white shadow-sm hover:bg-zinc-800 dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4" />
                  Save and Refresh
                </>
              )}
            </Button>
          </NativeDialogFooter>
        </form>
      </NativeDialogContent>
    </NativeDialog>
  );
}
