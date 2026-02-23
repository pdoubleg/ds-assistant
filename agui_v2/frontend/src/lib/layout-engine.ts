/**
 * Layout Engine - Component width and grid span calculation
 *
 * Handles conversion of width hints from the backend to CSS grid
 * column span classes for responsive layout in the output pane.
 */

import type { A2UIComponent } from './a2ui-catalog';

/**
 * Maps width hint values to Tailwind CSS grid column span classes.
 * Uses a 12-column grid system with responsive breakpoints.
 */
const WIDTH_TO_SPAN: Record<string, string> = {
  full: 'col-span-12',
  half: 'col-span-12 md:col-span-6',
  third: 'col-span-12 sm:col-span-6 lg:col-span-4',
  quarter: 'col-span-6 sm:col-span-3',
};

/**
 * Default width hints for each component type.
 */
const TYPE_DEFAULT_WIDTHS: Record<string, string> = {
  'a2ui.AuditQuestionForm': 'full',
  'a2ui.DataTable': 'full',
  'a2ui.TextBox': 'full',
  'a2ui.SimpleChart': 'half',
  'a2ui.DocumentCard': 'full',
};

/**
 * Get the CSS grid span class for a component based on its layout hints.
 *
 * Priority:
 * 1. Explicit width from component.layout.width
 * 2. Default width for the component type
 * 3. Full width as fallback
 *
 * @param component - The A2UI component to get grid span for.
 * @returns Tailwind CSS classes for grid column span.
 */
export function getGridSpan(component: A2UIComponent): string {
  const explicitWidth = component.layout?.width;
  const defaultWidth = TYPE_DEFAULT_WIDTHS[component.type];
  const width = explicitWidth || defaultWidth || 'full';
  return WIDTH_TO_SPAN[width] || WIDTH_TO_SPAN.full;
}

/**
 * Get all available width options.
 */
export function getAvailableWidths(): string[] {
  return Object.keys(WIDTH_TO_SPAN);
}

/**
 * Check if a component has an explicit width hint.
 */
export function hasExplicitWidth(component: A2UIComponent): boolean {
  return !!component.layout?.width;
}
