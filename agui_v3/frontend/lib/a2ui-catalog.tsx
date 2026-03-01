"use client";

/**
 * A2UI Catalog - Component Type Registry for Audit Assistant
 *
 * Maps backend A2UI component types (a2ui.*) to React components.
 * The A2UIRenderer uses this catalog to dynamically render components
 * based on backend-generated specs.
 *
 * Components:
 *   - DocumentCard: Document display with metadata and checkbox
 *   - AuditQuestionForm: Audit questions with expandable sub-questions
 *   - TextBox: General-purpose text/insight display
 *   - DataTable: Structured tabular data
 *   - SimpleChart: Simple bar/line/pie charts
 *   - ClaimTimeline: Vertical timeline of claim lifecycle events
 *   - SummaryCard: Grid of key-value metric tiles
 *   - FindingCard: Observation card with severity level
 */

import React from "react";
import { DocumentCard } from "@/components/A2UI/Documents";
import { AuditQuestionForm } from "@/components/A2UI/Forms";
import {
  TextBox,
  DataTable,
  SimpleChart,
  ClaimTimeline,
  SummaryCard,
  FindingCard,
} from "@/components/A2UI/General";

/**
 * Semantic zones for component grouping in the three-pane layout.
 *   - documents: Left-center pane showing uploaded documents
 *   - output: Right pane for generative UI (forms, charts, tables)
 */
export type SemanticZone = "documents" | "output";

/**
 * A2UI Component Specification.
 * Matches the backend A2UIComponent structure.
 */
export interface A2UIComponent {
  id: string;
  type: string;
  props: Record<string, unknown>;
  children?: A2UIComponent[];
  layout?: {
    width?: string;
    height?: string;
    position?: "relative" | "absolute" | "fixed" | "sticky";
    className?: string;
  };
  styling?: {
    variant?: string;
    theme?: string;
    className?: string;
  };
  zone?: SemanticZone;
}

/** Component renderer function signature. */
export type ComponentRenderer = (
  props: Record<string, unknown>,
  children?: React.ReactNode
) => React.ReactElement;

/** A2UI Component Catalog — maps a2ui.* types to React component renderers. */
export const a2uiCatalog: Record<string, ComponentRenderer> = {
  "a2ui.DocumentCard": (props) => <DocumentCard {...(props as any)} />,
  "a2ui.AuditQuestionForm": (props) => (
    <AuditQuestionForm {...(props as any)} />
  ),
  "a2ui.TextBox": (props) => <TextBox {...(props as any)} />,
  "a2ui.DataTable": (props) => <DataTable {...(props as any)} />,
  "a2ui.SimpleChart": (props) => <SimpleChart {...(props as any)} />,
  "a2ui.ClaimTimeline": (props) => <ClaimTimeline {...(props as any)} />,
  "a2ui.SummaryCard": (props) => <SummaryCard {...(props as any)} />,
  "a2ui.FindingCard": (props) => <FindingCard {...(props as any)} />,
};

/** Get component renderer from catalog. */
export function getComponentRenderer(
  type: string
): ComponentRenderer | undefined {
  return a2uiCatalog[type];
}

/** Check if component type is registered. */
export function isComponentRegistered(type: string): boolean {
  return type in a2uiCatalog;
}

/** Get all registered component types. */
export function getRegisteredTypes(): string[] {
  return Object.keys(a2uiCatalog);
}
