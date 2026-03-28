"""Pydantic models and types for the Recursive Language Model (RLM) module.

Provides local replacements for dspy REPL types so the RLM module has zero
dspy dependency.  All models are plain Pydantic ``BaseModel`` subclasses or
simple dataclasses.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_ai.usage import RunUsage

# ---------------------------------------------------------------------------
# Generic type variable for the user's output model
# ---------------------------------------------------------------------------
OutputT = TypeVar("OutputT")

# Simple Python types whose __name__ is safe to embed in function signatures.
SIMPLE_TYPES: frozenset[type] = frozenset({str, int, float, bool, list, dict})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class CodeExecutionError(Exception):
    """Raised when code execution in the sandbox fails at runtime."""


# ---------------------------------------------------------------------------
# REPL variable metadata
# ---------------------------------------------------------------------------
_MAX_PREVIEW_CHARS = 500


class REPLVariable(BaseModel):
    """Metadata about a single variable available inside the REPL sandbox.

    Mirrors the role of ``dspy.primitives.repl_types.REPLVariable``.

    Example:
        >>> var = REPLVariable.from_value("query", "What is 2+2?")
        >>> print(var.format())
        `query` (str): What is 2+2?
    """

    name: str = Field(description="Variable name as it appears in the sandbox.")
    type_name: str = Field(description="Python type name, e.g. 'str', 'list'.")
    desc: str = Field(
        default="", description="Human-readable description of the variable."
    )
    preview: str = Field(
        default="", description="Truncated string preview of the value."
    )

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        *,
        desc: str = "",
    ) -> REPLVariable:
        """Create a ``REPLVariable`` from a Python name and its runtime value.

        Args:
            name: The variable name.
            value: The runtime value of the variable.
            desc: Optional human-readable description of the variable.

        Returns:
            A populated ``REPLVariable`` instance.
        """
        # Determine type name
        type_name = type(value).__name__

        # Build a truncated preview string
        str_value = str(value)
        if len(str_value) > _MAX_PREVIEW_CHARS:
            preview = str_value[:_MAX_PREVIEW_CHARS] + "..."
        else:
            preview = str_value

        return cls(name=name, type_name=type_name, desc=desc, preview=preview)

    def format(self) -> str:
        r"""Render a human-readable description for inclusion in LLM prompts.

        Returns:
            Formatted string like ``\`name\` (type): description``.
        """
        parts = [f"`{self.name}` ({self.type_name})"]
        if self.desc:
            parts.append(f": {self.desc}")

        # Append a compact length hint for long values
        if len(self.preview) >= _MAX_PREVIEW_CHARS:
            parts.append(f" [{len(self.preview)}+ chars]")

        return "".join(parts)


# ---------------------------------------------------------------------------
# REPL history entry
# ---------------------------------------------------------------------------
class REPLEntry(BaseModel):
    """A single iteration of the REPL loop: reasoning, code, and output.

    Example:
        >>> entry = REPLEntry(
        ...     reasoning="Check the length",
        ...     code="print(len(context))",
        ...     output="4200",
        ... )
    """

    reasoning: str = Field(
        default="", description="The LLM's chain-of-thought reasoning."
    )
    code: str = Field(default="", description="Python code that was executed.")
    output: str = Field(
        default="", description="Stdout / result from executing the code."
    )

    @staticmethod
    def format_output(output: str, max_chars: int) -> str:
        """Truncate output to *max_chars*, appending an ellipsis marker.

        Args:
            output: The raw output string.
            max_chars: Maximum characters to keep.

        Returns:
            The (possibly truncated) output.
        """
        if len(output) <= max_chars:
            return output
        return output[:max_chars] + "\n... [output truncated]"

    def format(self, max_output_chars: int = 100_000) -> str:
        """Render the entry as a prompt-friendly string.

        Args:
            max_output_chars: Truncation limit for the output section.

        Returns:
            Multi-line formatted string.
        """
        lines: list[str] = []
        if self.reasoning:
            lines.append(f"[Reasoning] {self.reasoning}")
        if self.code:
            lines.append(f"[Code]\n```python\n{self.code}\n```")
        if self.output:
            truncated = self.format_output(self.output, max_output_chars)
            lines.append(f"[Output]\n{truncated}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# REPL history (immutable append-only sequence)
# ---------------------------------------------------------------------------
class REPLHistory(BaseModel):
    """Ordered collection of REPL entries with immutable append semantics.

    Each call to :meth:`append` returns a **new** ``REPLHistory`` instance,
    keeping the original unchanged (important for backtracking / branching).

    Example:
        >>> history = REPLHistory()
        >>> history = history.append(
        ...     reasoning="explore",
        ...     code="print(len(context))",
        ...     output="4200",
        ... )
        >>> print(len(history))
        1
    """

    entries: list[REPLEntry] = Field(default_factory=list)
    max_output_chars: int = Field(default=100_000)

    # -- mutation (returns new instance) ------------------------------------

    def append(
        self,
        *,
        reasoning: str = "",
        code: str = "",
        output: str = "",
    ) -> REPLHistory:
        """Return a new history with an additional entry appended.

        Args:
            reasoning: Chain-of-thought from the LLM.
            code: Code that was executed.
            output: Stdout / result of the execution.

        Returns:
            A new ``REPLHistory`` with the entry added.
        """
        entry = REPLEntry(reasoning=reasoning, code=code, output=output)
        return REPLHistory(
            entries=[*self.entries, entry],
            max_output_chars=self.max_output_chars,
        )

    # -- rendering ----------------------------------------------------------

    def format(self) -> str:
        """Render the full trajectory as a prompt-friendly string.

        Returns:
            Concatenated, numbered entries separated by blank lines.
        """
        if not self.entries:
            return "(no interactions yet)"
        parts: list[str] = []
        for i, entry in enumerate(self.entries, start=1):
            parts.append(f"--- Iteration {i} ---")
            parts.append(entry.format(self.max_output_chars))
        return "\n\n".join(parts)

    # -- sequence protocol --------------------------------------------------

    def __iter__(self):  # type: ignore[override]
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)


# ---------------------------------------------------------------------------
# Final output sentinel
# ---------------------------------------------------------------------------
@dataclass
class FinalOutput:
    """Sentinel returned by ``SUBMIT()`` in the sandbox to signal completion.

    Attributes:
        output: Dictionary mapping output field names to their values.
    """

    output: dict[str, Any]


# ---------------------------------------------------------------------------
# Action response (structured output of the action agent)
# ---------------------------------------------------------------------------
class ActionResponse(BaseModel):
    """Structured output produced by the action-generation agent.

    Example:
        >>> resp = ActionResponse(
        ...     reasoning="I should explore the data first.",
        ...     code="print(len(context))",
        ... )
    """

    reasoning: str = Field(
        description="Think step-by-step: what do you know? What remains? Plan your next action."
    )
    code: str = Field(
        description="Python code to execute. Use markdown code block format: "
        "```python\\n<code>\\n```"
    )


# ---------------------------------------------------------------------------
# RLM result wrapper
# ---------------------------------------------------------------------------
class RLMResult(BaseModel, Generic[OutputT]):
    """Result of a single :meth:`MontyRLM.run` invocation.

    Bundles the structured output with debugging metadata and accumulated
    token usage from **all** LLM sub-calls (action, extract, llm_query).

    Example:
        >>> result = await rlm.run(query="What is 2+2?", context="Math")
        >>> print(result.output)
        >>> print(result.usage.total_tokens)
    """

    output: Any = Field(
        description="Parsed structured output (instance of the user's output_type)."
    )
    trajectory: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialised REPL history entries for debugging.",
    )
    final_reasoning: str = Field(
        default="",
        description="The reasoning from the final iteration or extract step.",
    )
    usage: RunUsage = Field(
        default_factory=RunUsage,
        description="Accumulated token usage across all LLM calls in this run.",
    )

    def render_trajectory(
        self,
        *,
        include_reasoning: bool = True,
        max_output_chars: int = 5_000,
    ) -> str:
        """Render the REPL trajectory as a Markdown string.

        Each iteration is shown as a numbered section with optional
        reasoning, a Python code block, and the execution output.

        Args:
            include_reasoning: Whether to include the LLM's chain-of-thought
                reasoning before each code block.
            max_output_chars: Maximum characters to show per output block.
                Longer outputs are truncated with an ellipsis marker.

        Returns:
            A Markdown-formatted string suitable for display in notebooks,
            ``rich.Markdown``, or any Markdown renderer.

        Example:
            >>> result = await rlm.run(query="What is 2+2?", context="Math")
            >>> print(result.render_trajectory())
        """
        if not self.trajectory:
            return "_No trajectory recorded._"

        parts: list[str] = []
        for i, entry in enumerate(self.trajectory, start=1):
            section: list[str] = [f"### Iteration {i}"]

            # Reasoning (collapsible-friendly italic block)
            reasoning = entry.get("reasoning", "")
            if include_reasoning and reasoning:
                section.append(f"> **Reasoning:** {reasoning}")

            # Code
            code = entry.get("code", "")
            if code:
                section.append(f"```python\n{code}\n```")

            # Output (truncated if needed, wrapped in a fenced block)
            output = entry.get("output", "")
            if output:
                if len(output) > max_output_chars:
                    output = output[:max_output_chars] + "\n... [truncated]"
                section.append(f"**Output:**\n```\n{output}\n```")

            parts.append("\n\n".join(section))

        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def translate_field_type(name: str, field_info: FieldInfo) -> str:
    r"""Format a Pydantic field for display in LLM instruction prompts.

    Produces a string like ``\`answer\` (str): The final answer`` that the
    LLM can read in its system prompt to understand expected output fields.

    Args:
        name: The field name.
        field_info: Pydantic ``FieldInfo`` with annotation and description.

    Returns:
        Human-readable field description string.

    Example:
        >>> from pydantic.fields import FieldInfo
        >>> fi = FieldInfo(annotation=str, description="The final answer")
        >>> translate_field_type("answer", fi)
        '`answer` (str): The final answer'
    """
    annotation = getattr(field_info, "annotation", None) or str
    type_name = getattr(annotation, "__name__", str(annotation))
    desc = ""
    if hasattr(field_info, "description") and field_info.description:
        desc = f": {field_info.description}"
    return f"`{name}` ({type_name}){desc}"


def get_output_fields_info(output_type: type) -> list[dict[str, Any]]:
    """Extract output field metadata for the interpreter's SUBMIT mapping.

    Works with both ``BaseModel`` subclasses (multiple named fields) and
    simple scalar types like ``str`` or ``int`` (single ``output`` field).

    Args:
        output_type: The expected output type -- either a ``BaseModel``
            subclass or a simple type (``str``, ``int``, ``float``, ``bool``).

    Returns:
        List of dicts with ``name`` and optionally ``type`` keys.

    Example:
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     answer: str
        ...     confidence: float
        >>> get_output_fields_info(Answer)
        [{'name': 'answer', 'type': 'str'}, {'name': 'confidence', 'type': 'float'}]
        >>> get_output_fields_info(str)
        [{'name': 'output', 'type': 'str'}]
    """
    # Simple / scalar types get a single synthetic "output" field
    if isinstance(output_type, type) and not issubclass(output_type, BaseModel):
        entry: dict[str, Any] = {"name": "output"}
        if output_type in SIMPLE_TYPES:
            entry["type"] = output_type.__name__
        return [entry]

    # BaseModel -- iterate over declared fields
    fields_info: list[dict[str, Any]] = []
    for name, field_info in output_type.model_fields.items():
        entry = {"name": name}
        annotation = field_info.annotation
        if annotation in SIMPLE_TYPES:
            entry["type"] = annotation.__name__
        fields_info.append(entry)
    return fields_info
