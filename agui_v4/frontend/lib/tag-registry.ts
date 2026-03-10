import {
  AlertTriangle,
  Calculator,
  CalendarClock,
  Camera,
  ClipboardList,
  Clock,
  CloudRain,
  FilePlus,
  FileWarning,
  Gavel,
  HardHat,
  Home,
  MessageSquareWarning,
  Package,
  Phone,
  Scale,
  Siren,
  Store,
  Tag as TagGlyph,
  User,
  UserCog,
  type LucideIcon,
} from "lucide-react";

export type TagCategory = "source" | "type" | "flag";

export const CUSTOM_FALLBACK_TAG_LABEL = "No Applicable Tags";

export type TagIconName =
  | "general"
  | "insured"
  | "contractor"
  | "agent"
  | "vendor"
  | "attorney"
  | "contact_status"
  | "estimate"
  | "supplement"
  | "demand"
  | "dwelling"
  | "contents"
  | "ale"
  | "ems"
  | "photos"
  | "damage_report"
  | "weather_report"
  | "attorney_demand"
  | "time_sensitive"
  | "compliance_issue"
  | "customer_complaint";

export interface TagConfig {
  category: TagCategory;
  iconName: TagIconName;
  bg: string;
  text: string;
  border: string;
  activeBg: string;
  activeText: string;
  activeBorder: string;
  icon: LucideIcon;
}

type TagStyleConfig = Omit<TagConfig, "icon">;

const CUSTOM_FALLBACK_CONFIG: TagConfig = {
  category: "flag",
  iconName: "general",
  bg: "bg-amber-500/15",
  text: "text-amber-800 dark:text-amber-200",
  border: "border-amber-500/35",
  activeBg: "bg-amber-500/30",
  activeText: "text-amber-900 dark:text-amber-100",
  activeBorder: "border-amber-500/60",
  icon: TagGlyph,
};

const TAG_ICON_COMPONENTS: Record<TagIconName, LucideIcon> = {
  general: TagGlyph,
  insured: User,
  contractor: HardHat,
  agent: UserCog,
  vendor: Store,
  attorney: Scale,
  contact_status: Phone,
  estimate: Calculator,
  supplement: FilePlus,
  demand: FileWarning,
  dwelling: Home,
  contents: Package,
  ale: CalendarClock,
  ems: Siren,
  photos: Camera,
  damage_report: ClipboardList,
  weather_report: CloudRain,
  attorney_demand: Gavel,
  time_sensitive: Clock,
  compliance_issue: AlertTriangle,
  customer_complaint: MessageSquareWarning,
};

const TAG_ICON_STYLES: Record<TagIconName, TagStyleConfig> = {
  general: {
    category: "type",
    iconName: "general",
    bg: "bg-gray-500/15",
    text: "text-gray-700 dark:text-gray-300",
    border: "border-gray-500/25",
    activeBg: "bg-gray-500/30",
    activeText: "text-gray-800 dark:text-gray-200",
    activeBorder: "border-gray-500/60",
  },
  insured: {
    category: "source",
    iconName: "insured",
    bg: "bg-blue-500/15",
    text: "text-blue-700 dark:text-blue-300",
    border: "border-blue-500/25",
    activeBg: "bg-blue-500/30",
    activeText: "text-blue-800 dark:text-blue-200",
    activeBorder: "border-blue-500/60",
  },
  contractor: {
    category: "source",
    iconName: "contractor",
    bg: "bg-slate-500/15",
    text: "text-slate-700 dark:text-slate-300",
    border: "border-slate-500/25",
    activeBg: "bg-slate-500/30",
    activeText: "text-slate-800 dark:text-slate-200",
    activeBorder: "border-slate-500/60",
  },
  agent: {
    category: "source",
    iconName: "agent",
    bg: "bg-cyan-500/15",
    text: "text-cyan-700 dark:text-cyan-300",
    border: "border-cyan-500/25",
    activeBg: "bg-cyan-500/30",
    activeText: "text-cyan-800 dark:text-cyan-200",
    activeBorder: "border-cyan-500/60",
  },
  vendor: {
    category: "source",
    iconName: "vendor",
    bg: "bg-teal-500/15",
    text: "text-teal-700 dark:text-teal-300",
    border: "border-teal-500/25",
    activeBg: "bg-teal-500/30",
    activeText: "text-teal-800 dark:text-teal-200",
    activeBorder: "border-teal-500/60",
  },
  attorney: {
    category: "source",
    iconName: "attorney",
    bg: "bg-purple-500/15",
    text: "text-purple-700 dark:text-purple-300",
    border: "border-purple-500/25",
    activeBg: "bg-purple-500/30",
    activeText: "text-purple-800 dark:text-purple-200",
    activeBorder: "border-purple-500/60",
  },
  contact_status: {
    category: "type",
    iconName: "contact_status",
    bg: "bg-sky-500/15",
    text: "text-sky-700 dark:text-sky-300",
    border: "border-sky-500/25",
    activeBg: "bg-sky-500/30",
    activeText: "text-sky-800 dark:text-sky-200",
    activeBorder: "border-sky-500/60",
  },
  estimate: {
    category: "type",
    iconName: "estimate",
    bg: "bg-emerald-500/15",
    text: "text-emerald-700 dark:text-emerald-300",
    border: "border-emerald-500/25",
    activeBg: "bg-emerald-500/30",
    activeText: "text-emerald-800 dark:text-emerald-200",
    activeBorder: "border-emerald-500/60",
  },
  supplement: {
    category: "type",
    iconName: "supplement",
    bg: "bg-lime-500/15",
    text: "text-lime-700 dark:text-lime-300",
    border: "border-lime-500/25",
    activeBg: "bg-lime-500/30",
    activeText: "text-lime-800 dark:text-lime-200",
    activeBorder: "border-lime-500/60",
  },
  demand: {
    category: "type",
    iconName: "demand",
    bg: "bg-orange-500/15",
    text: "text-orange-700 dark:text-orange-300",
    border: "border-orange-500/25",
    activeBg: "bg-orange-500/30",
    activeText: "text-orange-800 dark:text-orange-200",
    activeBorder: "border-orange-500/60",
  },
  dwelling: {
    category: "type",
    iconName: "dwelling",
    bg: "bg-amber-500/15",
    text: "text-amber-700 dark:text-amber-300",
    border: "border-amber-500/25",
    activeBg: "bg-amber-500/30",
    activeText: "text-amber-800 dark:text-amber-200",
    activeBorder: "border-amber-500/60",
  },
  contents: {
    category: "type",
    iconName: "contents",
    bg: "bg-indigo-500/15",
    text: "text-indigo-700 dark:text-indigo-300",
    border: "border-indigo-500/25",
    activeBg: "bg-indigo-500/30",
    activeText: "text-indigo-800 dark:text-indigo-200",
    activeBorder: "border-indigo-500/60",
  },
  ale: {
    category: "type",
    iconName: "ale",
    bg: "bg-violet-500/15",
    text: "text-violet-700 dark:text-violet-300",
    border: "border-violet-500/25",
    activeBg: "bg-violet-500/30",
    activeText: "text-violet-800 dark:text-violet-200",
    activeBorder: "border-violet-500/60",
  },
  ems: {
    category: "type",
    iconName: "ems",
    bg: "bg-rose-500/15",
    text: "text-rose-700 dark:text-rose-300",
    border: "border-rose-500/25",
    activeBg: "bg-rose-500/30",
    activeText: "text-rose-800 dark:text-rose-200",
    activeBorder: "border-rose-500/60",
  },
  photos: {
    category: "type",
    iconName: "photos",
    bg: "bg-pink-500/15",
    text: "text-pink-700 dark:text-pink-300",
    border: "border-pink-500/25",
    activeBg: "bg-pink-500/30",
    activeText: "text-pink-800 dark:text-pink-200",
    activeBorder: "border-pink-500/60",
  },
  damage_report: {
    category: "type",
    iconName: "damage_report",
    bg: "bg-stone-500/15",
    text: "text-stone-700 dark:text-stone-300",
    border: "border-stone-500/25",
    activeBg: "bg-stone-500/30",
    activeText: "text-stone-800 dark:text-stone-200",
    activeBorder: "border-stone-500/60",
  },
  weather_report: {
    category: "type",
    iconName: "weather_report",
    bg: "bg-sky-500/15",
    text: "text-sky-700 dark:text-sky-300",
    border: "border-sky-400/25",
    activeBg: "bg-sky-500/30",
    activeText: "text-sky-800 dark:text-sky-200",
    activeBorder: "border-sky-500/60",
  },
  attorney_demand: {
    category: "flag",
    iconName: "attorney_demand",
    bg: "bg-red-500/15",
    text: "text-red-700 dark:text-red-300",
    border: "border-red-500/25",
    activeBg: "bg-red-500/30",
    activeText: "text-red-800 dark:text-red-200",
    activeBorder: "border-red-500/60",
  },
  time_sensitive: {
    category: "flag",
    iconName: "time_sensitive",
    bg: "bg-red-500/15",
    text: "text-red-700 dark:text-red-300",
    border: "border-red-500/25",
    activeBg: "bg-red-500/30",
    activeText: "text-red-800 dark:text-red-200",
    activeBorder: "border-red-500/60",
  },
  compliance_issue: {
    category: "flag",
    iconName: "compliance_issue",
    bg: "bg-yellow-500/15",
    text: "text-yellow-700 dark:text-yellow-300",
    border: "border-yellow-500/25",
    activeBg: "bg-yellow-500/30",
    activeText: "text-yellow-800 dark:text-yellow-200",
    activeBorder: "border-yellow-500/60",
  },
  customer_complaint: {
    category: "flag",
    iconName: "customer_complaint",
    bg: "bg-orange-500/15",
    text: "text-orange-700 dark:text-orange-300",
    border: "border-orange-500/25",
    activeBg: "bg-orange-500/30",
    activeText: "text-orange-800 dark:text-orange-200",
    activeBorder: "border-orange-500/60",
  },
};

export const DEFAULT_TAG_ICON_BY_LABEL: Record<string, TagIconName> = {
  Insured: "insured",
  Contractor: "contractor",
  Agent: "agent",
  Vendor: "vendor",
  Attorney: "attorney",
  "Contact/Status": "contact_status",
  Estimate: "estimate",
  Supplement: "supplement",
  Demand: "demand",
  Dwelling: "dwelling",
  Contents: "contents",
  ALE: "ale",
  EMS: "ems",
  Photos: "photos",
  "Damage Report": "damage_report",
  "Weather Report": "weather_report",
  "Attorney Demand": "attorney_demand",
  "Time Sensitive": "time_sensitive",
  "Compliance Issue": "compliance_issue",
  "Customer Complaint": "customer_complaint",
};

export const ALL_TAGS = Object.keys(DEFAULT_TAG_ICON_BY_LABEL);

export const TAGS_BY_CATEGORY: Record<TagCategory, string[]> = {
  source: ALL_TAGS.filter(
    (tag) => TAG_ICON_STYLES[DEFAULT_TAG_ICON_BY_LABEL[tag]].category === "source"
  ),
  type: ALL_TAGS.filter(
    (tag) => TAG_ICON_STYLES[DEFAULT_TAG_ICON_BY_LABEL[tag]].category === "type"
  ),
  flag: ALL_TAGS.filter(
    (tag) => TAG_ICON_STYLES[DEFAULT_TAG_ICON_BY_LABEL[tag]].category === "flag"
  ),
};

export function isTagIconName(value: string | null | undefined): value is TagIconName {
  if (!value) return false;
  return value in TAG_ICON_STYLES;
}

export function getDefaultTagIconName(tag: string): TagIconName {
  if (tag === CUSTOM_FALLBACK_TAG_LABEL) return "general";
  return DEFAULT_TAG_ICON_BY_LABEL[tag] ?? "general";
}

export function getTagConfig(tag: string, iconName?: string | null): TagConfig {
  if (tag === CUSTOM_FALLBACK_TAG_LABEL) {
    return CUSTOM_FALLBACK_CONFIG;
  }
  const resolvedIcon = isTagIconName(iconName) ? iconName : getDefaultTagIconName(tag);
  const styleConfig = TAG_ICON_STYLES[resolvedIcon] ?? TAG_ICON_STYLES.general;
  return {
    ...styleConfig,
    icon: TAG_ICON_COMPONENTS[resolvedIcon] ?? TAG_ICON_COMPONENTS.general,
  };
}
