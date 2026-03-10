"""Factory module for shared pydantic-ai workflow agents."""

from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext

from model_config import get_orchestrator_model
from models.analysis import AnalysisResult, ChartSpec, Finding, SummaryMetrics, TableSpec, TimelineEvents
from models.audit import TFRAnalysisResult
from models.documents import Documents
from models.search import DocSearchResult, DocumentSummary
from models.tagging import BatchTagResult
from prompts.analysis import (
    ANALYSIS_SYSTEM_PROMPT,
    CHART_SYSTEM_PROMPT,
    COMPONENT_SYSTEM_PROMPT,
    FINDING_SYSTEM_PROMPT,
    SUMMARY_METRICS_SYSTEM_PROMPT,
    TABLE_SYSTEM_PROMPT,
    TIMELINE_EVENT_SYSTEM_PROMPT,
)
from prompts.audit_form import AUDIT_FORM_SYSTEM_PROMPT
from prompts.documents import DOC_SEARCH_SORT_SYSTEM_PROMPT, DOCUMENT_SUMMARY_SYSTEM_PROMPT
from prompts.tagging import BATCH_TAGGER_SYSTEM_PROMPT


@dataclass
class SearchSortDeps:
    """Runtime dependencies injected into the search/sort agent."""

    documents: Documents


analysis_agent = Agent(
    model=get_orchestrator_model(),
    output_type=str,
    instructions=ANALYSIS_SYSTEM_PROMPT,
)

component_agent = Agent(
    model=get_orchestrator_model(),
    output_type=AnalysisResult,
    instructions=COMPONENT_SYSTEM_PROMPT,
)

audit_question_agent = Agent(
    model=get_orchestrator_model(),
    output_type=TFRAnalysisResult,
    instructions=AUDIT_FORM_SYSTEM_PROMPT,
)

timeline_event_agent = Agent(
    model=get_orchestrator_model(),
    output_type=TimelineEvents,
    instructions=TIMELINE_EVENT_SYSTEM_PROMPT,
)

summary_metrics_agent = Agent(
    model=get_orchestrator_model(),
    output_type=SummaryMetrics,
    instructions=SUMMARY_METRICS_SYSTEM_PROMPT,
)

findings_agent = Agent(
    model=get_orchestrator_model(),
    output_type=list[Finding],
    instructions=FINDING_SYSTEM_PROMPT,
)

tables_agent = Agent(
    model=get_orchestrator_model(),
    output_type=TableSpec,
    instructions=TABLE_SYSTEM_PROMPT,
)

charts_agent = Agent(
    model=get_orchestrator_model(),
    output_type=ChartSpec,
    instructions=CHART_SYSTEM_PROMPT,
)

document_summary_agent = Agent(
    model=get_orchestrator_model(),
    output_type=DocumentSummary,
    instructions=DOCUMENT_SUMMARY_SYSTEM_PROMPT,
)

batch_tagger_agent = Agent(
    model=get_orchestrator_model(),
    output_type=BatchTagResult,
    retries=5,
    instructions=BATCH_TAGGER_SYSTEM_PROMPT,
)

search_sort_agent = Agent(
    model=get_orchestrator_model(),
    output_type=DocSearchResult,
    deps_type=SearchSortDeps,
    retries=4,
    instructions=DOC_SEARCH_SORT_SYSTEM_PROMPT,
)


@search_sort_agent.tool
def as_metadata_string(ctx: RunContext[SearchSortDeps]) -> str:
    """Return metadata for all documents sorted newest-first."""
    print(
        f"[ORCHESTRATOR] Getting metadata for {len(ctx.deps.documents.documents)} documents...",
        flush=True,
    )
    return ctx.deps.documents.as_metadata_string()


@search_sort_agent.tool
def get_doc_by_content_id(ctx: RunContext[SearchSortDeps], content_id: str) -> str:
    """Retrieve the full text content of a document by content ID."""
    print(f"[ORCHESTRATOR] Getting document by content_id: {content_id}...", flush=True)
    try:
        document = ctx.deps.documents.get_doc_by_content_id(content_id)
        if document:
            return document.as_string()
        return f"Document with content_id '{content_id}' not found."
    except ValueError as exc:
        raise ModelRetry(f"Error: {exc}") from exc


def get_batch_tagger_agent() -> Agent:
    """Return the shared batch tagger agent for runtime overrides."""
    return batch_tagger_agent
