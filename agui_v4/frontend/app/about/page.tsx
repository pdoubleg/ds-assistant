"use client";

/**
 * About page — simple informational page for Q-Bot.
 */

import React from "react";
import { AppHeader } from "@/components/app-header";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Shield, Bot, FileText } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="h-screen flex flex-col bg-background text-foreground overflow-hidden">
      <AppHeader />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-8">
          <div className="text-center space-y-2">
            <h1 className="text-3xl font-bold text-foreground">
              About Q-Bot
            </h1>
            <p className="text-muted-foreground">
              AI-Powered Quality Audit Assistant
            </p>
          </div>

          <div className="grid gap-6 sm:grid-cols-3">
            <Card className="border-primary/20">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Bot className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold text-sm">AI Analysis</h3>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Upload claim documents and let Q-Bot perform automated
                  TFR (Technical File Review) analysis with structured
                  questionnaires.
                </p>
              </CardContent>
            </Card>

            <Card className="border-primary/20">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Shield className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold text-sm">Quality Auditing</h3>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Review and edit AI-generated audit forms with full
                  question-by-question control, reasoning, citations, and
                  outcome determination.
                </p>
              </CardContent>
            </Card>

            <Card className="border-primary/20">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold text-sm">Dashboard</h3>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Track audit metrics, view cross-form question analytics,
                  and export data — all from the centralized dashboard.
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="text-center text-xs text-muted-foreground pt-8">
            Built with Next.js, AG-UI, CopilotKit, and Pydantic AI
          </div>
        </div>
      </main>
    </div>
  );
}
