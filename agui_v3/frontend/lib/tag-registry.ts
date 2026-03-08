import {
  User,
  HardHat,
  UserCog,
  Store,
  Scale,
  Phone,
  Calculator,
  FilePlus,
  FileWarning,
  Home,
  Package,
  CalendarClock,
  Siren,
  Camera,
  ClipboardList,
  CloudRain,
  Gavel,
  Clock,
  AlertTriangle,
  MessageSquareWarning,
  type LucideIcon,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TagCategory = "source" | "type" | "flag";

export interface TagConfig {
  category: TagCategory;
  /** Tailwind classes for the pill background, text, and border (default state). */
  bg: string;
  text: string;
  border: string;
  /** Tailwind classes applied when the pill is active (filter selected). */
  activeBg: string;
  activeText: string;
  activeBorder: string;
  icon: LucideIcon;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const TAG_REGISTRY: Record<string, TagConfig> = {
  // ── Sources ──────────────────────────────────────────────────────────
  Insured: {
    category: "source",
    bg: "bg-blue-500/15",
    text: "text-blue-700 dark:text-blue-300",
    border: "border-blue-500/25",
    activeBg: "bg-blue-500/30",
    activeText: "text-blue-800 dark:text-blue-200",
    activeBorder: "border-blue-500/60",
    icon: User,
  },
  Contractor: {
    category: "source",
    bg: "bg-slate-500/15",
    text: "text-slate-700 dark:text-slate-300",
    border: "border-slate-500/25",
    activeBg: "bg-slate-500/30",
    activeText: "text-slate-800 dark:text-slate-200",
    activeBorder: "border-slate-500/60",
    icon: HardHat,
  },
  Agent: {
    category: "source",
    bg: "bg-cyan-500/15",
    text: "text-cyan-700 dark:text-cyan-300",
    border: "border-cyan-500/25",
    activeBg: "bg-cyan-500/30",
    activeText: "text-cyan-800 dark:text-cyan-200",
    activeBorder: "border-cyan-500/60",
    icon: UserCog,
  },
  Vendor: {
    category: "source",
    bg: "bg-teal-500/15",
    text: "text-teal-700 dark:text-teal-300",
    border: "border-teal-500/25",
    activeBg: "bg-teal-500/30",
    activeText: "text-teal-800 dark:text-teal-200",
    activeBorder: "border-teal-500/60",
    icon: Store,
  },
  Attorney: {
    category: "source",
    bg: "bg-purple-500/15",
    text: "text-purple-700 dark:text-purple-300",
    border: "border-purple-500/25",
    activeBg: "bg-purple-500/30",
    activeText: "text-purple-800 dark:text-purple-200",
    activeBorder: "border-purple-500/60",
    icon: Scale,
  },

  // ── Types ────────────────────────────────────────────────────────────
  "Contact/Status": {
    category: "type",
    bg: "bg-sky-500/15",
    text: "text-sky-700 dark:text-sky-300",
    border: "border-sky-500/25",
    activeBg: "bg-sky-500/30",
    activeText: "text-sky-800 dark:text-sky-200",
    activeBorder: "border-sky-500/60",
    icon: Phone,
  },
  Estimate: {
    category: "type",
    bg: "bg-emerald-500/15",
    text: "text-emerald-700 dark:text-emerald-300",
    border: "border-emerald-500/25",
    activeBg: "bg-emerald-500/30",
    activeText: "text-emerald-800 dark:text-emerald-200",
    activeBorder: "border-emerald-500/60",
    icon: Calculator,
  },
  Supplement: {
    category: "type",
    bg: "bg-lime-500/15",
    text: "text-lime-700 dark:text-lime-300",
    border: "border-lime-500/25",
    activeBg: "bg-lime-500/30",
    activeText: "text-lime-800 dark:text-lime-200",
    activeBorder: "border-lime-500/60",
    icon: FilePlus,
  },
  Demand: {
    category: "type",
    bg: "bg-orange-500/15",
    text: "text-orange-700 dark:text-orange-300",
    border: "border-orange-500/25",
    activeBg: "bg-orange-500/30",
    activeText: "text-orange-800 dark:text-orange-200",
    activeBorder: "border-orange-500/60",
    icon: FileWarning,
  },
  Dwelling: {
    category: "type",
    bg: "bg-amber-500/15",
    text: "text-amber-700 dark:text-amber-300",
    border: "border-amber-500/25",
    activeBg: "bg-amber-500/30",
    activeText: "text-amber-800 dark:text-amber-200",
    activeBorder: "border-amber-500/60",
    icon: Home,
  },
  Contents: {
    category: "type",
    bg: "bg-indigo-500/15",
    text: "text-indigo-700 dark:text-indigo-300",
    border: "border-indigo-500/25",
    activeBg: "bg-indigo-500/30",
    activeText: "text-indigo-800 dark:text-indigo-200",
    activeBorder: "border-indigo-500/60",
    icon: Package,
  },
  ALE: {
    category: "type",
    bg: "bg-violet-500/15",
    text: "text-violet-700 dark:text-violet-300",
    border: "border-violet-500/25",
    activeBg: "bg-violet-500/30",
    activeText: "text-violet-800 dark:text-violet-200",
    activeBorder: "border-violet-500/60",
    icon: CalendarClock,
  },
  EMS: {
    category: "type",
    bg: "bg-rose-500/15",
    text: "text-rose-700 dark:text-rose-300",
    border: "border-rose-500/25",
    activeBg: "bg-rose-500/30",
    activeText: "text-rose-800 dark:text-rose-200",
    activeBorder: "border-rose-500/60",
    icon: Siren,
  },
  Photos: {
    category: "type",
    bg: "bg-pink-500/15",
    text: "text-pink-700 dark:text-pink-300",
    border: "border-pink-500/25",
    activeBg: "bg-pink-500/30",
    activeText: "text-pink-800 dark:text-pink-200",
    activeBorder: "border-pink-500/60",
    icon: Camera,
  },
  "Damage Report": {
    category: "type",
    bg: "bg-stone-500/15",
    text: "text-stone-700 dark:text-stone-300",
    border: "border-stone-500/25",
    activeBg: "bg-stone-500/30",
    activeText: "text-stone-800 dark:text-stone-200",
    activeBorder: "border-stone-500/60",
    icon: ClipboardList,
  },
  "Weather Report": {
    category: "type",
    bg: "bg-sky-500/15",
    text: "text-sky-700 dark:text-sky-300",
    border: "border-sky-400/25",
    activeBg: "bg-sky-500/30",
    activeText: "text-sky-800 dark:text-sky-200",
    activeBorder: "border-sky-500/60",
    icon: CloudRain,
  },

  // ── Flags ────────────────────────────────────────────────────────────
  "Attorney Demand": {
    category: "flag",
    bg: "bg-red-500/15",
    text: "text-red-700 dark:text-red-300",
    border: "border-red-500/25",
    activeBg: "bg-red-500/30",
    activeText: "text-red-800 dark:text-red-200",
    activeBorder: "border-red-500/60",
    icon: Gavel,
  },
  "Time Sensitive": {
    category: "flag",
    bg: "bg-red-500/15",
    text: "text-red-700 dark:text-red-300",
    border: "border-red-500/25",
    activeBg: "bg-red-500/30",
    activeText: "text-red-800 dark:text-red-200",
    activeBorder: "border-red-500/60",
    icon: Clock,
  },
  "Compliance Issue": {
    category: "flag",
    bg: "bg-yellow-500/15",
    text: "text-yellow-700 dark:text-yellow-300",
    border: "border-yellow-500/25",
    activeBg: "bg-yellow-500/30",
    activeText: "text-yellow-800 dark:text-yellow-200",
    activeBorder: "border-yellow-500/60",
    icon: AlertTriangle,
  },
  "Customer Complaint": {
    category: "flag",
    bg: "bg-orange-500/15",
    text: "text-orange-700 dark:text-orange-300",
    border: "border-orange-500/25",
    activeBg: "bg-orange-500/30",
    activeText: "text-orange-800 dark:text-orange-200",
    activeBorder: "border-orange-500/60",
    icon: MessageSquareWarning,
  },
} as const;

// ---------------------------------------------------------------------------
// Derived helpers
// ---------------------------------------------------------------------------

/** All valid tag labels (stable order matching the registry). */
export const ALL_TAGS = Object.keys(TAG_REGISTRY);

/** Tag labels grouped by category. */
export const TAGS_BY_CATEGORY: Record<TagCategory, string[]> = {
  source: ALL_TAGS.filter((t) => TAG_REGISTRY[t].category === "source"),
  type: ALL_TAGS.filter((t) => TAG_REGISTRY[t].category === "type"),
  flag: ALL_TAGS.filter((t) => TAG_REGISTRY[t].category === "flag"),
};

const FALLBACK_CONFIG: TagConfig = {
  category: "type",
  bg: "bg-gray-500/15",
  text: "text-gray-700 dark:text-gray-300",
  border: "border-gray-500/25",
  activeBg: "bg-gray-500/30",
  activeText: "text-gray-800 dark:text-gray-200",
  activeBorder: "border-gray-500/60",
  icon: Package,
};

/**
 * Look up the config for a tag label. Returns a neutral fallback for
 * unknown tags so the UI never breaks.
 */
export function getTagConfig(tag: string): TagConfig {
  return TAG_REGISTRY[tag] ?? FALLBACK_CONFIG;
}
