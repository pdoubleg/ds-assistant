from textwrap import dedent
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai.models.openai import OpenAIResponsesModel

# load environment variables
from dotenv import load_dotenv
load_dotenv()


# =====
# State
# =====
class LogEntry(BaseModel):
  """Represents a progress log line for the research process.

  Args:
    message: Human-readable progress update or action taken.
    done: Whether the step has completed.

  Example:
    >>> LogEntry(message="Searched initial keywords", done=True)
  """
  message: str
  done: bool = False


class Resource(BaseModel):
  """A web or document resource captured during research.

  Args:
    url: Canonical URL or identifier for the resource.
    title: Short title describing the resource.
    description: One to three sentence summary extracted by the agent.

  Example:
    >>> Resource(url="https://arxiv.org/abs/1234.5678", title="Paper", description="A relevant paper")
  """
  url: str
  title: str
  description: str


class ResearchState(BaseModel):
  """Shared state between the UI and the research agent.

  Attributes:
    model: Backend model family to target (e.g., "openai", "google_genai").
    research_question: The current research question being pursued.
    report: Accumulated draft report text.
    resources: Collected resource metadata for citations and review.
    logs: Structured progress updates rendered in the UI.

  Example:
    >>> ResearchState(model="openai", research_question="Lifespan of penguins")
  """
  model: Optional[str] = Field(default=None, description="Selected model family")
  research_question: str = Field(default="", description="User-provided research question")
  report: str = Field(default="", description="Draft report content")
  resources: List[Resource] = Field(default_factory=list, description="Collected resources")
  logs: List[LogEntry] = Field(default_factory=list, description="Stepwise progress logs")


# =====
# Agent
# =====
agent = Agent(
  model=OpenAIResponsesModel("gpt-4.1-mini"),
  deps_type=StateDeps[ResearchState],
  system_prompt=dedent(
    """
    You are a helpful research assistant.

    Your job is to help the user research their question by:
    - clarifying or updating the research_question
    - collecting and curating resources
    - maintaining a concise progress log in `logs`
    - drafting and revising the `report` using the collected resources

    IMPORTANT:
    - Prefer calling the available tools to update state instead of free-form replies.
    - Use `log_progress` to keep the user informed of steps being taken.
    - When deleting resources, first ask the UI to confirm via the "DeleteResources" frontend action, then call `delete_resources` if confirmed.
    """
  ).strip(),
)


# =====
# Tools
# =====
@agent.tool
def set_research_question(ctx: RunContext[StateDeps[ResearchState]], question: str) -> StateSnapshotEvent:
  """Set or update the current research question and log the change."""
  # Update research question
  ctx.deps.state.research_question = question
  # Append a progress log entry indicating the update
  ctx.deps.state.logs.append(LogEntry(message=f"Updated research question: {question}", done=True))
  return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)


@agent.tool
def add_resources(ctx: RunContext[StateDeps[ResearchState]], resources: List[Resource]) -> StateSnapshotEvent:
  """Append one or more resources to the state.

  The agent should ensure uniqueness by URL when feasible.
  """
  # Build a set of existing URLs for deduplication
  existing_urls = {r.url for r in ctx.deps.state.resources}
  # Add any new resources by URL
  for r in resources:
    if r.url not in existing_urls:
      ctx.deps.state.resources.append(r)
      existing_urls.add(r.url)
  # Log progress for visibility in UI
  ctx.deps.state.logs.append(LogEntry(message=f"Added {len(resources)} resource(s)", done=True))
  return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)


@agent.tool
def delete_resources(ctx: RunContext[StateDeps[ResearchState]], urls: List[str]) -> StateSnapshotEvent:
  """Delete resources by URL and return the updated state snapshot."""
  # Filter out any resource whose URL is listed for deletion
  before_count = len(ctx.deps.state.resources)
  ctx.deps.state.resources = [r for r in ctx.deps.state.resources if r.url not in set(urls)]
  after_count = len(ctx.deps.state.resources)
  # Log what was done
  removed = before_count - after_count
  ctx.deps.state.logs.append(LogEntry(message=f"Deleted {removed} resource(s)", done=True))
  return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)


@agent.tool
def set_report(ctx: RunContext[StateDeps[ResearchState]], report: str) -> StateSnapshotEvent:
  """Replace the entire draft report text with the provided content."""
  ctx.deps.state.report = report
  return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)


@agent.tool
def log_progress(ctx: RunContext[StateDeps[ResearchState]], message: str, done: bool = False) -> StateSnapshotEvent:
  """Append a progress log entry for display in the UI."""
  ctx.deps.state.logs.append(LogEntry(message=message, done=done))
  return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)


@agent.tool
def clear_logs(ctx: RunContext[StateDeps[ResearchState]]) -> StateSnapshotEvent:
  """Clear any existing progress logs at the start of a new research session."""
  ctx.deps.state.logs = []
  return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=ctx.deps.state)
