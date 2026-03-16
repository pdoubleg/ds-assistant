"use client";

/**
 * useFlaggedHits — localStorage-backed store for "docked" / flagged
 * Doc Lens query result images.
 *
 * Persists across sessions so users can accumulate images from multiple
 * Doc Lens queries and export them later.
 *
 * Singleton model: all calls to useFlaggedHits() share the same in-memory
 * state via a module-level store + subscriber registry. Mutations made
 * inside DocLensOverlay are immediately reflected in OutputPane (and any
 * other consumer) without requiring a common React ancestor or prop
 * threading. localStorage is the durable backing store; the in-memory
 * store is the live source-of-truth for the current page session.
 */

import { useState, useCallback, useEffect } from "react";
import type { QueryHit } from "./use-doc-lens";

const STORAGE_KEY = "agui_v3.flaggedHits.v1";
const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

// ── Types ──────────────────────────────────────────────────────────────

export interface FlaggedHit {
  /** The original query hit data. */
  hit: QueryHit;
  /** The query string that produced this hit. */
  query: string;
  /** ISO timestamp when the hit was flagged. */
  flagged_at: string;
}

// ── Module-level singleton store ────────────────────────────────────────
// Lives outside React so it survives re-renders and is shared across all
// useFlaggedHits() consumers on the same page.

type Listener = (hits: FlaggedHit[]) => void;

/** Module-level state — the single source of truth for in-memory hits. */
let _hits: FlaggedHit[] = [];
/** Whether the store has been hydrated from localStorage yet. */
let _hydrated = false;
/** All active useFlaggedHits() subscribers. */
const _listeners = new Set<Listener>();

/**
 * Notify all subscribers of the current state.
 * @param hits - The updated list of flagged hits.
 */
function _notify(hits: FlaggedHit[]): void {
  for (const listener of _listeners) {
    listener(hits);
  }
}

/**
 * Update the singleton store, persist to localStorage, and broadcast
 * the change to all subscribed hook instances.
 *
 * @param updater - A pure function (like a setState updater) that receives
 *   the current hits and returns the new list.
 */
function _update(updater: (prev: FlaggedHit[]) => FlaggedHit[]): void {
  _hits = updater(_hits);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(_hits));
  } catch {
    // localStorage full or unavailable
  }
  _notify(_hits);
}

/**
 * Hydrate the singleton store from localStorage exactly once.
 * Safe to call multiple times — subsequent calls are no-ops.
 */
function _hydrateOnce(): void {
  if (_hydrated) return;
  _hydrated = true;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      _hits = JSON.parse(raw) as FlaggedHit[];
    }
  } catch {
    // corrupt data — start fresh
  }
}

// ── Hook ───────────────────────────────────────────────────────────────

export function useFlaggedHits() {
  // Always start with an empty array so the server render matches the client's
  // initial render — localStorage is only accessible after hydration.
  const [flaggedHits, setFlaggedHits] = useState<FlaggedHit[]>([]);

  // Subscribe to singleton changes and unsubscribe on unmount.
  // Also hydrates the store on first mount (client-only, post-hydration).
  useEffect(() => {
    // Hydrate the singleton from localStorage exactly once, then sync local state.
    _hydrateOnce();
    setFlaggedHits(_hits);

    _listeners.add(setFlaggedHits);
    return () => {
      _listeners.delete(setFlaggedHits);
    };
  }, []);

  // ── Mutators — all go through _update() so every instance is notified ──

  const isFlagged = useCallback(
    (assetHash: string) =>
      flaggedHits.some((f) => f.hit.asset_hash === assetHash),
    [flaggedHits]
  );

  const toggleFlag = useCallback((hit: QueryHit, query: string) => {
    _update((prev) => {
      const exists = prev.some((f) => f.hit.asset_hash === hit.asset_hash);
      if (exists) {
        return prev.filter((f) => f.hit.asset_hash !== hit.asset_hash);
      }
      return [...prev, { hit, query, flagged_at: new Date().toISOString() }];
    });
  }, []);

  const removeFlag = useCallback((assetHash: string) => {
    _update((prev) => prev.filter((f) => f.hit.asset_hash !== assetHash));
  }, []);

  const clearAll = useCallback(() => {
    _update(() => []);
  }, []);

  /**
   * Build the URL for a Doc Lens asset image so the frontend can render it.
   * The image_path from the backend is an absolute path; we resolve the
   * portion after /assets/ into the static mount.
   *
   * @param imagePath - The absolute image path returned by the backend.
   * @returns The frontend-accessible URL for the image.
   *
   * @example
   * ```ts
   * getImageUrl("/data/assets/session123/photo.png")
   * // → "http://localhost:8001/doc-lens-assets/session123/photo.png"
   * ```
   */
  const getImageUrl = useCallback((imagePath: string) => {
    if (imagePath.startsWith("/")) {
      return `${BACKEND_URL}${imagePath}`;
    }
    // Normalize Windows backslashes so the marker search works cross-platform.
    const normalized = imagePath.replace(/\\/g, "/");
    const marker = "assets/";
    const idx = normalized.indexOf(marker);
    const relative =
      idx >= 0 ? normalized.slice(idx + marker.length) : normalized;
    return `${BACKEND_URL}/doc-lens-assets/${relative}`;
  }, []);

  /** Download a single image by triggering a temporary anchor click. */
  const downloadImage = useCallback(
    (imagePath: string, fileName?: string) => {
      const url = getImageUrl(imagePath);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName || imagePath.split("/").pop() || "image.png";
      a.target = "_blank";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    },
    [getImageUrl]
  );

  /** Download all flagged images one-by-one. */
  const exportAllImages = useCallback(() => {
    for (const flagged of flaggedHits) {
      const name = `${flagged.hit.document_name}_p${flagged.hit.page_number}_${flagged.hit.asset_hash.slice(0, 8)}.png`;
      downloadImage(flagged.hit.image_url || flagged.hit.image_path, name);
    }
  }, [flaggedHits, downloadImage]);

  return {
    flaggedHits,
    isFlagged,
    toggleFlag,
    removeFlag,
    clearAll,
    getImageUrl,
    downloadImage,
    exportAllImages,
    flagCount: flaggedHits.length,
  };
}
