"use client";

import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { FilterDropdown } from "./filter-dropdown";
import type { Filters, FilterOptions } from "./types";

export interface DocumentsPaneFilterPanelProps {
  useNarrowToolbar: boolean;
  showFilters: boolean;
  filterOptions: FilterOptions;
  allTagOptions: string[];
  filters: Filters;
  onToggleFilter: (key: keyof Omit<Filters, "search">, value: string) => void;
}

/**
 * Collapsible filter row shown when the toolbar switches into narrow mode.
 */
export function DocumentsPaneFilterPanel({
  useNarrowToolbar,
  showFilters,
  filterOptions,
  allTagOptions,
  filters,
  onToggleFilter,
}: DocumentsPaneFilterPanelProps) {
  return (
    <AnimatePresence>
      {useNarrowToolbar && showFilters && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="overflow-hidden border-b border-border/30"
        >
          <div className="flex items-center gap-1.5 px-3 py-2 flex-wrap">
            <FilterDropdown
              label="Type"
              options={filterOptions.mimeTypes}
              selected={filters.mimeTypes}
              onToggle={(value) => onToggleFilter("mimeTypes", value)}
            />
            <FilterDropdown
              label="Doc Type"
              options={filterOptions.docTypes}
              selected={filters.docTypes}
              onToggle={(value) => onToggleFilter("docTypes", value)}
            />
            <FilterDropdown
              label="Domain"
              options={filterOptions.domains}
              selected={filters.domains}
              onToggle={(value) => onToggleFilter("domains", value)}
            />
            {allTagOptions.length > 0 && (
              <FilterDropdown
                label="Tags"
                options={allTagOptions}
                selected={filters.tags}
                onToggle={(value) => onToggleFilter("tags", value)}
              />
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
