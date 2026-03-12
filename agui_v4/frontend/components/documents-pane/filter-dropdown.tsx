"use client";

import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export interface FilterDropdownProps {
  label: string;
  options: string[];
  selected: Set<string>;
  onToggle: (value: string) => void;
}

/**
 * Compact multi-select dropdown used by the documents pane toolbar.
 */
export function FilterDropdown({
  label,
  options,
  selected,
  onToggle,
}: FilterDropdownProps) {
  if (options.length === 0) return null;

  return (
    <Select
      value={selected.size === 1 ? [...selected][0] : "__multi__"}
      onValueChange={(value) => {
        if (value !== "__multi__") onToggle(value);
      }}
    >
      <SelectTrigger className="h-7 w-auto min-w-[80px] max-w-[120px] text-[11px]">
        <SelectValue>
          {selected.size > 0 ? `${label} (${selected.size})` : label}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem
            key={option}
            value={option}
            className={cn(
              "text-xs",
              selected.has(option) && "font-bold text-primary"
            )}
          >
            {selected.has(option) ? `✓ ${option}` : option}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
