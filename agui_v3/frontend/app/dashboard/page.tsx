"use client";

/**
 * Dashboard page — high-level analytics view over all saved audit forms.
 *
 * Fetches full form data from GET /forms/all, then renders:
 *   1. Metric cards (aggregate stats)
 *   2. Forms data table (one row per form, sortable/filterable)
 *   3. Questions aggregation table (collapsible question/sub-question stats)
 *
 * Selecting a row in the forms table opens a read-only FormViewerSheet.
 */

import React, { useEffect, useState, useCallback } from "react";
import { AppHeader } from "@/components/app-header";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DashboardMetrics } from "@/components/dashboard/dashboard-metrics";
import { FormsDataTable } from "@/components/dashboard/forms-data-table";
import { QuestionsAggregationTable } from "@/components/dashboard/questions-aggregation-table";
import { FormViewerSheet } from "@/components/dashboard/form-viewer-sheet";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SavedForm } from "@/lib/dashboard-types";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

export default function DashboardPage() {
  const [forms, setForms] = useState<SavedForm[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selected form for the viewer sheet
  const [selectedForm, setSelectedForm] = useState<SavedForm | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);

  const fetchForms = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/forms/all`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setForms(data.forms ?? []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to fetch forms"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchForms();
  }, [fetchForms]);

  const handleSelectForm = useCallback((form: SavedForm) => {
    setSelectedForm(form);
    setSheetOpen(true);
  }, []);

  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
      <AppHeader />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
          {/* Page heading */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">
                Audit Dashboard
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Overview of all saved TFR audit forms and question analytics.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5"
              onClick={fetchForms}
              disabled={loading}
            >
              <RefreshCw
                className={`h-4 w-4 ${loading ? "animate-spin" : ""}`}
              />
              Refresh
            </Button>
          </div>

          {/* Error state */}
          {error && (
            <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>Failed to load dashboard data: {error}</span>
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto"
                onClick={fetchForms}
              >
                Retry
              </Button>
            </div>
          )}

          {/* Loading skeletons */}
          {loading && (
            <div className="space-y-6">
              <Skeleton className="h-[120px] w-full rounded-xl" />
              <Skeleton className="h-[400px] w-full rounded-xl" />
            </div>
          )}

          {/* Content */}
          {!loading && !error && (
            <>
              {/* Metrics */}
              <DashboardMetrics forms={forms} />

              {/* Tabbed tables */}
              <Tabs defaultValue="forms" className="w-full">
                <TabsList>
                  <TabsTrigger value="forms">
                    Forms ({forms.length})
                  </TabsTrigger>
                  <TabsTrigger value="questions">
                    Question Analytics
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="forms" className="mt-4">
                  <FormsDataTable
                    forms={forms}
                    onSelectForm={handleSelectForm}
                    selectedFormId={selectedForm?.id}
                  />
                </TabsContent>

                <TabsContent value="questions" className="mt-4">
                  <QuestionsAggregationTable forms={forms} />
                </TabsContent>
              </Tabs>
            </>
          )}

          {/* Empty state */}
          {!loading && !error && forms.length === 0 && (
            <div className="text-center py-16 text-muted-foreground">
              <p className="text-lg font-medium">No forms saved yet</p>
              <p className="text-sm mt-1">
                Complete an audit in the main app to see data here.
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Form viewer sheet */}
      <FormViewerSheet
        form={selectedForm}
        open={sheetOpen}
        onOpenChange={setSheetOpen}
      />
    </div>
  );
}
