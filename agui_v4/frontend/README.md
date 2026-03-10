# AGUI v3 — Frontend

Next.js 15 (App Router) + React 19 frontend for the Audit Assistant.
Communicates with a Pydantic AI backend over the
[AG-UI protocol](https://docs.ag-ui.com) via CopilotKit, and renders
dynamic agent-generated UI through a pattern called **A2UI** (Agent-to-UI).

---

## Tech Stack

| Layer | Technology |
| ----- | ---------- |
| Framework | Next.js 15, React 19 |
| Agent protocol | AG-UI (`@ag-ui/client`), CopilotKit v2 (`@copilotkit/react-core`) |
| UI primitives | Radix UI + Tailwind CSS 4 (shadcn "new-york" style) |
| Animations | Framer Motion |
| Icons | Lucide React |
| Theming | `next-themes` (dark / light / system) |
| Markdown | `react-markdown`, `remark-gfm`, `remark-math`, `rehype-katex` |

---

## Architecture Overview

```
┌──────────────────── Browser ────────────────────┐
│                                                  │
│  Providers (providers.tsx)                        │
│  ┌─ NextThemesProvider                           │
│  │  └─ CopilotKitProvider  ← AG-UI HttpAgent     │
│  │     └─ UploadedDocsProvider                   │
│  │        └─ ChatDocsProvider                    │
│  │           └─ TooltipProvider                  │
│  │              └─ <App />                       │
│  └───────────────────────────────────────────    │
│                                                  │
│  Three-Pane Layout (page.tsx)                    │
│  ┌───────────┬──────────────┬───────────────┐    │
│  │ ChatPane  │ DocumentsPane│  OutputPane   │    │
│  │  (flex 1) │   (flex 1)   │   (flex 2)    │    │
│  └───────────┴──────────────┴───────────────┘    │
│                                                  │
└──────────────────────────────────────────────────┘
         │                          ▲
         │  AG-UI events            │  AG-UI state sync
         ▼                          │
┌──────────────────── Backend ────────────────────┐
│  Pydantic AI agent  (agent/agent.py)             │
│  Emits A2UIComponent specs via AuditState        │
└──────────────────────────────────────────────────┘
```

Each pane is collapsible to a narrow sidebar strip via `CollapsedStrip`;
at least one pane stays open at all times. Framer Motion handles the
layout animations.

---

## AGUI — Shared Agent State

The frontend and backend share a synchronized state object via the AG-UI
protocol. CopilotKit's `useAgent` hook handles the bidirectional sync.

### Provider hierarchy

Defined in `components/providers.tsx`:

```
NextThemesProvider
  └─ CopilotKitProvider          ← wraps an HttpAgent pointed at the backend
       └─ UploadedDocsProvider    ← documents uploaded via file input
            └─ ChatDocsProvider   ← which docs are included in chat context
                 └─ TooltipProvider
```

### AuditState shape

The canonical shared state (must mirror `agent/agent.py`):

```ts
interface AuditState {
  documents: Array<Record<string, unknown>>;
  components: A2UIComponent[];       // A2UI specs rendered by the frontend
  audit_questions: Array<Record<string, unknown>>;
  analysis_result: Record<string, unknown>;
  audit_form_result: Record<string, unknown>;
  current_form_id: string | null;
  status: "idle" | "analyzing" | "generating" | "complete" | "error";
  progress: number;
  current_step: string;
  activity_log: Array<{ id; message; timestamp; status }>;
  error_message: string | null;
}
```

The backend updates state via `agent.setState()`. The frontend reads it
through `agent.state` (from the `useAgent` hook).

### Hooks and their state domains

| Hook | File | State owned | Storage |
| ---- | ---- | ----------- | ------- |
| `useAuditAgent` | `hooks/use-audit-agent.ts` | AG-UI `AuditState`, tool-call activity, step activity, form persistence (save/restore/list/delete) | AG-UI protocol + backend REST |
| `useUploadedDocs` | `hooks/use-uploaded-docs.tsx` | `UploadedDoc[]` — metadata and extracted text for uploaded files | React Context (in-memory) |
| `useChatDocs` | `hooks/use-chat-docs.tsx` | `chatDocNames: Set<string>` — which docs are included in chat context | React Context + `localStorage` |
| `useDocLens` | `hooks/use-doc-lens.ts` | Doc Lens session lifecycle: ingest progress, query results, session ID | Local state (hook-scoped) |
| `useFlaggedHits` | `hooks/use-flagged-hits.ts` | Flagged/docked Doc Lens hits shared across components without a common ancestor | Module-level singleton + `localStorage` |

---

## A2UI — Agent-to-UI Rendering

A2UI is the pattern by which the backend dynamically generates UI. The
backend emits component *specs* (type string + props); the frontend
catalog maps them to real React components.

### Flow

1. Backend produces `A2UIComponent` specs during an agent run.
2. Specs land in `AuditState.components` and sync to the frontend via
   AG-UI.
3. Components are grouped by `SemanticZone` (`"documents"` or `"output"`)
   to target the correct pane.
4. `a2ui-renderer.tsx` looks up each spec in the catalog and renders it.

### Catalog (`lib/a2ui-catalog.tsx`)

| Type string | React component | Location |
| ----------- | --------------- | -------- |
| `a2ui.DocumentCard` | `DocumentCard` | `components/a2ui/documents/` |
| `a2ui.AuditQuestionForm` | `AuditQuestionForm` | `components/a2ui/forms/` |
| `a2ui.TextBox` | `TextBox` | `components/a2ui/general/` |
| `a2ui.DataTable` | `DataTable` | `components/a2ui/general/` |
| `a2ui.SimpleChart` | `SimpleChart` | `components/a2ui/general/` |
| `a2ui.ClaimTimeline` | `ClaimTimeline` | `components/a2ui/general/` |
| `a2ui.SummaryCard` | `SummaryCard` | `components/a2ui/general/` |
| `a2ui.FindingCard` | `FindingCard` | `components/a2ui/general/` |

### Layout engine (`lib/layout-engine.ts`)

`getGridSpan()` maps `A2UIComponent.layout.width` values (`full`, `half`,
`third`, `quarter`) to Tailwind grid `col-span-*` classes for a 12-column
grid.

### Adding a new A2UI component

1. Create the React component in the appropriate `components/a2ui/`
   subfolder (kebab-case filename).
2. Re-export it from the subfolder's `index.ts` barrel.
3. Register it in `lib/a2ui-catalog.tsx` with a `a2ui.*` type key.
4. Emit the matching spec from the backend.

---

## Doc Lens

Doc Lens is a text-to-image retrieval feature for semantic search over
uploaded PDFs and images.

| Module | Purpose |
| ------ | ------- |
| `hooks/use-doc-lens.ts` | Session lifecycle, NDJSON ingest streaming, query execution |
| `hooks/use-flagged-hits.ts` | Module-level singleton + `localStorage` for saved hits shared across overlays and panes |
| `components/doc-lens/doc-lens-context.tsx` | React Context provider wrapping `DocumentsPane`; owns session and overlay visibility |
| `components/doc-lens/doc-lens-overlay.tsx` | Full-screen overlay with session stats, document cards, query input, results grid |
| `components/doc-lens/query-hit-card.tsx` | Single query result card: image, metadata, flag, download |
| `components/doc-lens/flagged-hits-panel.tsx` | Inline panel for flagged hits (used inside overlay and `OutputPane`) |

Sessions start lazily on first overlay open, are reused across
open/close, and reset when the eligible document set changes.

---

## Folder Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout + font loading
│   ├── page.tsx                # Three-pane collapsible layout
│   ├── globals.css             # Tailwind + CSS variables
│   ├── about/page.tsx
│   └── dashboard/page.tsx
│
├── components/
│   ├── providers.tsx           # CopilotKit + theme + context providers
│   ├── app-header.tsx          # Top header bar
│   ├── chat-pane.tsx           # Chat UI with file upload + tool activity
│   ├── documents-pane.tsx      # Document triage, filtering, tagging
│   ├── output-pane.tsx         # A2UI output, saved forms, flagged hits
│   ├── collapsed-strip.tsx     # Collapsed-pane sidebar strip
│   ├── document-viewer-sheet.tsx
│   ├── a2ui-renderer.tsx       # Dynamic renderer for A2UI specs
│   │
│   ├── a2ui/                   # Agent-generated UI components
│   │   ├── documents/          #   DocumentCard
│   │   ├── forms/              #   AuditQuestionForm
│   │   └── general/            #   TextBox, DataTable, SimpleChart, ...
│   │
│   ├── doc-lens/               # Doc Lens overlay + context
│   │   ├── index.ts            #   Barrel exports
│   │   ├── doc-lens-context.tsx
│   │   ├── doc-lens-overlay.tsx
│   │   ├── query-hit-card.tsx
│   │   └── flagged-hits-panel.tsx
│   │
│   ├── dashboard/              # Dashboard route components
│   │   ├── dashboard-metrics.tsx
│   │   ├── form-viewer-sheet.tsx
│   │   ├── forms-data-table.tsx
│   │   └── questions-aggregation-table.tsx
│   │
│   └── ui/                     # shadcn/Radix primitives
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       ├── ...
│       └── tooltip.tsx
│
├── hooks/
│   ├── use-audit-agent.ts      # AG-UI agent state + tool activity
│   ├── use-uploaded-docs.tsx    # Uploaded document context
│   ├── use-chat-docs.tsx       # Chat document selection context
│   ├── use-doc-lens.ts         # Doc Lens session lifecycle
│   └── use-flagged-hits.ts     # Flagged hits singleton
│
├── lib/
│   ├── a2ui-catalog.tsx        # A2UI type → React component registry
│   ├── layout-engine.ts        # Grid span mapping
│   ├── tag-registry.ts         # Document tag definitions
│   ├── dashboard-types.ts      # Dashboard TypeScript types
│   └── utils.ts                # cn() and other utilities
│
└── public/
    └── q-bot.png               # App logo
```

---

## Naming Conventions

| Scope | Convention | Example |
| ----- | ---------- | ------- |
| Files & folders | kebab-case | `use-audit-agent.ts`, `a2ui/general/` |
| React components | PascalCase exports | `export function SummaryCard` |
| Hooks | `use-` prefix, kebab-case file | `use-doc-lens.ts` → `useDocLens()` |
| A2UI type keys | `a2ui.PascalCase` | `"a2ui.FindingCard"` |
| UI primitives | shadcn convention (kebab-case) | `components/ui/button.tsx` |
| Barrel exports | `index.ts` per feature folder | `components/a2ui/general/index.ts` |

---

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Production build
npx next build
```

The frontend expects the backend at `http://localhost:8001` by default.
Override with the `NEXT_PUBLIC_BACKEND_URL` environment variable.
