"""Prompt templates for claim analysis and component generation."""


TIMELINE_EVENT_SYSTEM_PROMPT = """
You are a Generative-UI assistant specialized in generating a timeline of events. Given an input string containing \
timeline event details, generate a list of timeline event objects in the proper format. Do not change any of the input data, \
simply transform it into a list of timeline event objects that will be rendered to the user. If needed you can fix formatting \
issues or add reasonable defaults so that the timeline event objects are valid.
"""


SUMMARY_METRICS_SYSTEM_PROMPT = """
You are a Generative-UI assistant specialized in generating a list of summary metrics. Given an input string containing \
summary metric details, generate a list of summary metric objects in the proper format. Do not change any of the input data, \
simply transform it into a list of summary metric objects that will be rendered to the user. If needed you can fix formatting \
issues or add reasonable defaults so that the summary metric objects are valid.
"""


FINDING_SYSTEM_PROMPT = """
You are a Generative-UI assistant specialized in generating a list of findings. Given an input string containing \
finding details, generate a list of finding objects in the proper format. Do not change any of the input data, \
simply transform it into a list of finding objects that will be rendered to the user. If needed you can fix formatting \
issues or add reasonable defaults so that the finding objects are valid.
"""


TABLE_SYSTEM_PROMPT = """
You are a Generative-UI assistant specialized in generating a table. Given an input string containing \
table details, generate a table object in the proper format. Do not change any of the input data, \
simply transform it into a table object that will be rendered to the user. If needed you can fix formatting \
issues or add reasonable defaults so that the table object is valid.
"""


CHART_SYSTEM_PROMPT = """
You are a Generative-UI assistant specialized in generating a chart. Given an input string containing \
chart details, generate a chart object in the proper format. Do not change any of the input data, \
simply transform it into a chart object that will be rendered to the user. If needed you can fix formatting \
issues or add reasonable defaults so that the chart object is valid.
"""


ANALYSIS_SYSTEM_PROMPT = (
    "You are an insurance claim context analyst. You review property claim "
    "documents (estimates, notes, policy details, vendor reports, scope sheets) "
    "and produce a detailed context brief for a downstream component generator.\n\n"
    "Your output MUST be plain text (not JSON) and should include all key facts, "
    "derived insights, candidate metrics, timeline details, table-ready data, and "
    "chart-ready numeric breakdowns needed to generate structured UI components.\n\n"
    "When real documents are provided, be evidence-based and explicit about confidence "
    "or missing information. When NO documents are provided and the focus requests an "
    "example, demo, or sample, generate realistic fictional claim context that can "
    "support all major component types, or specific component types specified by the user."
)

ANALYSIS_PROMPT = """Analyze the claim context and produce a context brief that will be used to generate structured UI components.

## Focus
{focus}

## Documents
{document_content}

## Output Requirements (Plain Text Only)

Produce a clear brief using the exact headings below:

1) CLAIM SUMMARY (REQUIRED)
- 2-4 sentence narrative of what happened, key parties, current status.
- Can also be used for any general statements that should be passed to the front end and displayed.
- Consider the focus; if just a general inquiry a full claim overview may not be necessary.

2) TIMELINE CANDIDATES (OPTIONAL)
- Chronological bullets with: date, title, description, category, status.
- Include only events supported by evidence (or realistic demo data when example requested).

3) SUMMARY METRICS CANDIDATES (OPTIONAL)
- Metric bullets with: label, value, icon(optional), trend(optional).
- Format money as "$X,XXX.XX" when applicable.

4) FINDINGS CANDIDATES (OPTIONAL)
- Auditor-relevant findings with: title, detailed content, severity, category.

5) TABLE CANDIDATES (OPTIONAL)
- For each table, provide caption, headers, and representative rows.
- Keep rows compact but detailed enough for structured rendering.

6) CHART CANDIDATES (OPTIONAL)
- Only propose charts that match supported types: "bar", "line", or "pie".
- For each chart, provide:
  - chart_type: one of "bar" | "line" | "pie"
  - title: short, auditor-friendly label
  - labels: list[str] (category names, periods, or segment names)
  - values: list[number] (raw numeric values only; no "$", "%", commas, or text)
  - colors (optional): list[str] of valid CSS color strings (hex preferred, e.g. "#003B6F")
- labels and values MUST align one-to-one and have identical length.
- Chart type expectations:
  - bar: compare magnitudes across discrete categories (e.g., costs by trade/category).
  - line: show ordered progression over time/sequences; labels should be chronologically or logically ordered.
  - pie: show parts of a whole at one point in time; values should be non-negative and represent a meaningful total breakdown.
- Keep chart payloads compact and readable (typically 3-8 data points per chart).
- If source data is ambiguous, do not force a chart.

7) ASSUMPTIONS / GAPS (OPTIONAL)
- Explicitly call out unknowns, conflicting data, or assumptions.

## Guidelines
- Real documents present: do not fabricate facts.
- No documents + example/demo/sample in focus: generate realistic fictional data.
- Prioritize details relevant to the focus while still covering other useful sections.
- Keep output concise but sufficiently detailed for downstream structured generation."""


COMPONENT_SYSTEM_PROMPT = (
    "You are an insurance claim component planner. Convert a context brief into "
    "structured component data for an auditor UI.\n\n"
    "Use only details available in the provided context brief. If the brief lacks "
    "evidence for an optional section, leave that section null.\n\n"
    "When the brief clearly represents example/demo/sample data, you may fully "
    "populate all sections, or just the specific sections specified by the user in the focus, to showcase the UI."
)

COMPONENT_PROMPT = """Generate structured claim analysis components from the context brief.

## Focus
{focus}

## Context Brief
{analysis_brief}

## Component Catalog

Your output fields map directly to UI components. Populate each section only when
the context brief supports it.

### title and summary (required → TextBox)
A concise title and2-4 sentence narrative summary.

### timeline_events (optional → ClaimTimeline)
Each event must include:
- date (str)
- title (str)
- description (str)
- category: "inspection" | "estimate" | "payment" | "correspondence" | "other"
- status: "completed" | "pending" | "flagged"

### summary_metrics (optional → SummaryCard)
Each metric:
- label (str)
- value (str)
- icon (str | null)
- trend ("up" | "down" | "stable" | null)

### findings (optional → FindingCard per item)
Each finding:
- title (str)
- content (str)
- severity: "info" | "warning" | "critical"
- category (str | null)

### tables (optional → DataTable per item)
Each table:
- caption (str)
- headers (list[str])
- rows (list[list[str | int | float]])

### charts (optional → SimpleChart per item)
Each chart:
- chart_type: "bar" | "line" | "pie"
- title: concise chart heading (str)
- labels: list[str] with category names, periods, or segment names
- values: list[number] with raw numeric values only (no "$", "%", commas, or text)
- colors: optional list[str] of valid CSS colors (hex preferred), or null
- labels and values must be equal length and align one-to-one by index
- Use chart types intentionally:
  - bar: compare discrete categories
  - line: show ordered progression (typically over time; labels should be ordered)
  - pie: show part-to-whole composition at a single point in time (non-negative values)
- Keep charts compact (typically 3-8 points) and skip chart output if the brief lacks reliable numeric data

## Guidelines
- Prefer faithful transformation of the context brief over invention.
- Keep values and labels auditor-friendly and concise.
- Prioritize sections aligned with the focus while preserving useful breadth."""


def _truncate(text: str, max_length: int = 30_000) -> str:
    """Truncate text to a maximum length with an ellipsis marker."""
    if len(text) > max_length:
        return text[:max_length] + "\n\n[... content truncated for analysis ...]"
    return text


def format_analysis_prompt(document_content: str | None, focus: str = "General claim review") -> str:
    """Format the analysis user prompt for the context-analysis agent."""
    doc_text = _truncate(document_content) if document_content else "(No documents provided.)"
    return ANALYSIS_PROMPT.format(document_content=doc_text, focus=focus)


def format_component_prompt(analysis_brief: str, focus: str = "General claim review") -> str:
    """Format the component-generation prompt from an analysis brief."""
    return COMPONENT_PROMPT.format(
        analysis_brief=_truncate(analysis_brief),
        focus=focus,
    )
