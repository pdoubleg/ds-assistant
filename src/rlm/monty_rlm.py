"""Recursive Language Model (RLM) powered by pydantic-monty.

Implements :class:`MontyRLM`, an agentic loop where an LLM writes Python code
that runs in a restricted ``pydantic-monty`` sandbox.  The LLM examines large
contexts programmatically, calls sub-LLMs for semantic analysis, and builds up
answers iteratively -- all without feeding the full context into a single prompt.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import pydantic
import pydantic_monty
from pydantic import BaseModel
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.models import Model

from .interpreter import MontyCodeInterpreter
from .types import (
    ActionResponse,
    CodeExecutionError,
    FinalOutput,
    REPLHistory,
    REPLVariable,
    RLMResult,
    get_output_fields_info,
    translate_field_type,
)
from .usage import UsageTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type variable for the user's output schema (BaseModel *or* simple type)
# ---------------------------------------------------------------------------
OutputT = TypeVar("OutputT")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


MONTY_EXAMPLES_V1 = """
EXAMPLES — Strategies for large contexts:

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub-LLMs are powerful -- they \
can fit around 250K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to \
feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, just write it directly. For example, \
say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```python
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")
print(answer)
SAVE(answer=answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM \
on that chunk, and track relevant information in a buffer using SAVE():
```python
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
# First iteration - process first section
section = context[0]
buffer = llm_query(f"You are iteratively looking through a book. Gather information to help answer {query}. Here is section 0 of {len(context)}: {section}")
SAVE(buffer=buffer, idx=1)
print(f"After section 0 of {len(context)}, you have tracked: {buffer}")
# Subsequent iterations - continue processing
if idx < len(context) - 1:
    section = context[idx]
    buffer = llm_query(f"You are iteratively looking through a book, and are on section {idx} of {len(context)}. So far you know: {buffer}. Gather information to help answer {query}. Here is the section: {section}")
    SAVE(buffer=buffer, idx=idx + 1)
    print(f"After section {idx} of {len(context)}, you have tracked: {buffer}")
# Final iteration - answer the question
else:
    section = context[-1]
    final_answer = llm_query(f"You are on the last section of the book. So far you know that: {buffer}. Gather from this last section to answer {query}. Here is the section: {section}")
    print(f"Based on reading iteratively through the book, the answer is: {final_answer}")
    SUBMIT(answer=final_answer)
```

As another example, when the context isn't that long (e.g. <100M characters), a simple but viable strategy is, based on the context chunk lengths, \
to combine them and query an LLM over chunks using llm_query_batched for concurrent processing. For example, if the context is a List[str], we \
ask the same query over each chunk:
```python
query = "A man became famous for his book 'The Great Gatsby'. How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 20 chunks
chunk_size = len(context) // 20
chunks = []
for i in range(20):
    if i < 9:
        chunk_str = "\\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)
SAVE(chunks=chunks)

# Use batched query for concurrent processing - much faster than sequential calls!
prompts = [f"Try to answer the following query: {query}. Here are the documents:\\n{chunk}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
answers = llm_query_batched(prompts)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {i}: {answer}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {query}\\n\\nAnswers:\\n" + "\\n".join(answers))
print(f"Final answer: {final_answer}")
SAVE(final_answer=final_answer)
# Don't call SUBMIT() here, we'll call it in the next iteration after reviewing the printed output
```
```python
SUBMIT(answer=final_answer)
```

As a final example, implement the solution as a program: try one approach; inspect the result and branch. If it suffices, use it. If not, break \
into one easier subproblem and delegate that. More branches, one path runs—don't overload the context. Example: prove sqrt 2 irrational.
```python
# First iteration - try direct approach
approach_result = llm_query("Prove sqrt 2 is irrational. Give a 1-2 sentence proof, or reply only: USE_LEMMA or USE_CONTRADICTION.")
SAVE(approach_result=approach_result)
print(f"Initial approach result: {approach_result}")
# Second iteration - branch based on result
if "USE_LEMMA" in approach_result.upper():
    final_answer = llm_query("Prove 'n^2 even => n even' then use it to show sqrt 2 irrational. Two sentences.")
    SAVE(final_answer=final_answer)
    print(f"Final proof: {final_answer}")
else:
    final_answer = approach_result
    SAVE(final_answer=final_answer)
    print(f"Using original proof: {final_answer}")
# In the NEXT iteration, after reviewing the printed output:
```python
SUBMIT(answer=final_answer)
```

IMPORTANT - Completing Your Task:

When you are done with the iterative process, you MUST call SUBMIT() to provide your final answer. However, SUBMIT() immediately ends execution, so \
you won't see any output after calling it.

CRITICAL WORKFLOW:
1. First, create your final answer, SAVE it, and print() it to inspect the result
2. Review the printed output in the next iteration
3. Only then call SUBMIT() with the verified result, optionally modifying the result one last time

Example of CORRECT workflow:
# Iteration N: Create and inspect final answer
final_answer = llm_query(f"Based on all evidence, what is the final answer? Evidence: {accumulated_data}")
SAVE(final_answer=final_answer)
print(f"Final answer: {final_answer}")
# DON'T call SUBMIT yet - need to see the output first!
Then in the NEXT iteration, after reviewing the printed output:
```python
# Iteration N+1: Now submit after confirming it looks correct
# Optionally, modify the result one last time, e.g., final_answer = f"some_additional_info: {final_answer}"
# Then submit
SUBMIT(answer=final_answer)
```

SAVE() usage:
- Always SAVE variables that you want to persist across iterations.
- If you SAVE a variable using the same name, it will overwrite the old value.
- If an error occurs, the variable(s) created by that code may not be saved.

SUBMIT() usage:
- For structured output with multiple fields: SUBMIT(field1=value1, field2=value2, ...)
- For simple scalar output: SUBMIT(output=value)
- Match the output fields required by your task (check the "You are tasked with producing" section at the top)

COMMON MISTAKES TO AVOID:
WRONG: Calling SUBMIT() without first printing and inspecting the result
WRONG: Trying to print after SUBMIT() - execution already ended
CORRECT: print(result) → review output → SUBMIT(field=result) in next iteration

Remember: Variables saved with SAVE() persist across iterations and are automatically available in the next iteration - just reference them directly by name.
"""


MONTY_DEFAULT_EXAMPLES = """
EXAMPLES — Strategies for large contexts:
Sub-LLMs are powerful — they can fit ~250K characters in their context window. Analyze your input data size and structure first, then pick a strategy. \
Often a few well-targeted sub-LLM calls are sufficient. If accuracy is important, favor chunking the document and using llm_query_batched to cover the entire document. \
You can also use the allowed standard-library imports (`sys`, `os`, `typing`, `re`, `datetime`, `json`) when they help with parsing, chunking, or metadata inspection.

Example 1 — Simple chunking with llm_query:
If the context is a long string, chunk it and query each piece:
```python
# Split context into chunks
chunk_size = 10000
chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]
SAVE(chunks=chunks)

# Query each chunk
answers = []
for i, chunk in enumerate(chunks):
    answer = llm_query(f"What is the magic number in chunk {{i+1}}? Here is the chunk:\n{{chunk}}")
    answers.append(answer)
    print(f"Chunk {{i+1}} answer: {{answer}}")
SAVE(answers=answers)

# Combine results with a final query
if len(answers) > 1:
    final_answer = llm_query(f"Based on these answers from different chunks, what is the final magic number?\n{{answers}}")
    print(f"Final answer: {{final_answer}}")
    SAVE(final_answer=final_answer)
    SUBMIT(magic_number=final_answer)
else:
    SAVE(final_answer=answers[0])
    SUBMIT(magic_number=answers[0])
```

Example 2 — Iterative section-by-section reading with SAVE():
When your context is a list (e.g. sections, pages, chapters), process it iteratively and persist progress with SAVE(). In the first iteration, start processing:
```python
section = context[0]
result = llm_query(f"Extract key facts about '{{query}}' from this section:\\n{{section}}")
SAVE(results=[result], next_idx=1)
print(f"Section 0: {{result}}")
```
In subsequent iterations, continue from where you left off:
```python
if next_idx < len(context):
    section = context[next_idx]
    result = llm_query(f"Known so far: {{results}}\\nExtract new facts from this section:\\n{{section}}")
    results.append(result)
    SAVE(results=results, next_idx=next_idx + 1)
    print(f"Section {{next_idx}}: {{result}}")
else:
    # All sections processed, combine and submit
    final_answer = llm_query(f"Based on all these facts, provide a comprehensive answer to '{{query}}':\\n" + "\\n".join(results))
    print(f"Final answer: {{final_answer}}")
    SUBMIT(answer=final_answer)
```

Example 3 — Batched concurrent queries over chunks:
When you can split the context into independent chunks, use llm_query_batched for parallel processing — much faster than sequential calls:
```python
chunk_size = len(context) // 10
chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]
prompts = [f"Answer '{{query}}' based on this excerpt:\\n{{c}}" for c in chunks]
answers = llm_query_batched(prompts)
for i, ans in enumerate(answers):
    print(f"Chunk {{i}}: {{ans}}")
final_answer = llm_query(f"Combine these partial answers into one final answer for '{{query}}':\\n" + "\\n".join(answers))
print(f"Final answer: {{final_answer}}")
SAVE(final_answer=final_answer)
# Don't call SUBMIT() here, we'll call it in the next iteration after reviewing the printed output
```

Example 4 — Regex-guided chunking for large structured text:
If the context is large but has recognizable delimiters or headings, use `re` to find boundaries before sending chunks to `llm_query`:
```python
import re

parts = re.split(r'(?m)^###\\s+', context)
summaries = []
for part in parts[1:]:
    lines = part.split('\\n')
    header = lines[0].strip()
    body = '\\n'.join(lines[1:]).strip()
    summary = llm_query(f"Summarize the '{{header}}' section:\\n{{body}}")
    summaries.append(f"{{header}}: {{summary}}")
    print(f"Summarized: {{header}}")
final_answer = llm_query(f"Based on these summaries, answer '{{query}}':\\n" + "\\n".join(summaries))
SAVE(final_answer=final_answer)
print(f"Answer: {{final_answer}}")
# Don't call SUBMIT() here, we'll call it in the next iteration after reviewing the printed output
```

Example 5 — Building large outputs efficiently (avoiding context bloat):
When accumulating results from many chunks, use SAVE() to build the output without showing everything to the LLM. Only print progress indicators:
```python
# First iteration: start processing
SAVE(full_results=[], next_idx=0)
print("Starting to process chunks...")
```
Subsequent iterations:
```python
if next_idx < len(chunks):
    # Process chunk and accumulate result in saved variable
    result = llm_query(f"Extract information from: {{chunks[next_idx]}}")
    full_results.append(result)
    SAVE(full_results=full_results, next_idx=next_idx + 1)
    # Only print progress, not the full accumulated data
    print(f"Processed chunk {{next_idx + 1}}/{{len(chunks)}}")
else:
    # All chunks done - build final output (can be megabytes)
    final_output = "\\n\\n".join(full_results)
    print(f"Complete! Total length: {{len(final_output)}} chars")
    SUBMIT(output=final_output)  # LLM never sees this in context since SUBMIT ends execution
```

Example 6 — Simple computation (no llm_query needed):
Not every task requires sub-LLM calls. Use Python directly for computation, parsing, and filtering:
```python
# Count occurrences of a term
term = "error"
count = context.lower().count(term.lower())
print(f"Found {{count}} occurrences of '{{term}}'")
SUBMIT(count=count)
```
"""

MONTY_ACTION_INSTRUCTIONS_TEMPLATE = """\
You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a restricted Python sandbox (Monty). Write Python code and it will be executed. You will see the output, then write \
more code based on what you learned. This is an iterative process, with each step building on the previous one.

Available:
- Variables: {inputs} (your input data — re-injected every iteration)
- `llm_query(prompt)` - query a sub-LLM (~100K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SAVE(name=value, ...)` - IMPORTANT: use this to persist variables across iterations (re-injected automatically next iteration)
- `CLEAR(name1, name2, ...)` or `CLEAR()` - remove saved variables (all if no args)
- `SUBMIT({final_output_names})` - submit final output when done
- Builtins plus these imports: `sys`, `os`, `typing`, `re`, `datetime`, `json`. Use them when helpful for parsing, chunking, \
filesystem/context inspection, or formatting. NO class definitions.
- Always use one import per line. Do not write import a, b, c. Monty does not support comma-separated import statements.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step. \
Be mindful that in the case of code interpreter errors, objects created by that code may not be saved. If errors occur you should pivot to a simpler approach.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. Use `SAVE()` to persist intermediate results across iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via input variables and parse/compute in code instead of retyping.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

{examples}

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."""

_EXTRACT_INSTRUCTIONS_TEMPLATE = """\
{task_instructions}Based on the REPL trajectory, extract the final outputs now.

Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."""

# Pattern to match markdown code fences: ```python\n...\n``` or ```\n...\n```
_CODE_FENCE_PATTERN = re.compile(r"^```(?:python|py)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fence wrappers from LLM-generated code.

    Args:
        code: Raw code string, possibly wrapped in triple-backtick fences.

    Returns:
        The inner code without fences.
    """
    code = code.strip()
    match = _CODE_FENCE_PATTERN.match(code)
    if match:
        return match.group(1)
    return code


# ---------------------------------------------------------------------------
# Functions reserved for the sandbox interpreter
# ---------------------------------------------------------------------------

_RESERVED_TOOL_NAMES = frozenset(
    {"llm_query", "llm_query_batched", "SUBMIT", "print", "SAVE", "CLEAR"}
)


@dataclass(frozen=True)
class _ToolInfo:
    """Metadata for a user-provided tool callable."""

    name: str
    func: Callable[..., Any]
    desc: str
    params: list[tuple[str, str]]  # (param_name, type_hint_str)


def _extract_tool_info(func: Callable[..., Any], name: str | None = None) -> _ToolInfo:
    """Inspect a callable to build :class:`_ToolInfo`.

    Args:
        func: The tool function.
        name: Override for the tool name (defaults to ``func.__name__``).

    Returns:
        Populated ``_ToolInfo``.
    """
    tool_name = name or getattr(func, "__name__", "unknown")
    desc = (inspect.getdoc(func) or "No description").replace("\n", "  ")
    params: list[tuple[str, str]] = []
    try:
        sig = inspect.signature(func)
        for pname, param in sig.parameters.items():
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                type_str = "Any"
            else:
                type_str = getattr(ann, "__name__", str(ann))
            params.append((pname, type_str))
    except (ValueError, TypeError):
        pass
    return _ToolInfo(name=tool_name, func=func, desc=desc, params=params)


# ---------------------------------------------------------------------------
# MontyRLM
# ---------------------------------------------------------------------------


class MontyRLM(Generic[OutputT]):
    """Recursive Language Model using pydantic-ai agents and a Monty sandbox.

    The LLM iteratively writes Python code that is executed in a restricted
    ``pydantic-monty`` sandbox.  Sub-LLM calls (``llm_query`` /
    ``llm_query_batched``) are routed through a dedicated pydantic-ai agent
    so their token usage is tracked.

    Args:
        output_type: The expected output type.  Pass a Pydantic ``BaseModel``
            subclass for multi-field structured output, or a simple type
            (``str``, ``int``, ``float``, ``bool``) for single-value output.
            When a simple type is used the sandbox expects
            ``SUBMIT(output=value)``.
        model: The LLM to use for action generation and output extraction.
            Accepts a model name string (e.g. ``"openai:gpt-4o"``) or a
            pydantic-ai ``Model`` instance.
        sub_model: Optional cheaper LLM for ``llm_query`` sub-calls.
            Defaults to *model* when ``None``.
        instructions: Task-specific instructions prepended to the system
            prompt (e.g. ``"Summarise the document"``).
        examples: Examples of how to use the RLM to solve common tasks.
        max_iterations: Maximum REPL interaction iterations.
        max_llm_calls: Maximum ``llm_query`` / ``llm_query_batched``
            invocations per run.
        max_output_chars: Truncation limit for REPL output in history.
        tools: Optional list of extra callable tools available inside the
            sandbox.  Tool names are inferred from function names.
        type_check: Whether Monty should type-check sandbox code.
        limits: Resource limits passed to ``pydantic-monty``.
        verbose: If ``True``, log detailed iteration info.

    Example:
        >>> from pydantic import BaseModel, Field
        >>> from gepadantic.rlm import MontyRLM
        >>>
        >>> class Answer(BaseModel):
        ...     answer: str = Field(description="The final answer")
        ...
        >>> rlm = MontyRLM(
        ...     output_type=Answer,
        ...     model="openai:gpt-4o",
        ...     sub_model="openai:gpt-4o-mini",
        ... )
        >>> result = await rlm.run(
        ...     query="What is the magic number?",
        ...     context="The magic number is 42.",
        ... )
        >>> print(result.output.answer)
        42
        >>> print(result.usage.total_tokens)
    """

    def __init__(
        self,
        output_type: type[OutputT],
        model: str | Model,
        *,
        sub_model: str | Model | None = None,
        instructions: str = "",
        examples: str = MONTY_EXAMPLES_V1,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 100_000,
        tools: list[Callable[..., Any]] | None = None,
        type_check: bool = False,
        limits: pydantic_monty.ResourceLimits | None = None,
        verbose: bool = False,
    ) -> None:
        self._output_type = output_type
        # Whether the output type is a structured BaseModel or a scalar
        self._is_model_output = isinstance(output_type, type) and issubclass(
            output_type, BaseModel
        )
        self._model = model
        self._sub_model = sub_model or model
        self._instructions = instructions
        self._examples = examples
        self._max_iterations = max_iterations
        self._max_llm_calls = max_llm_calls
        self._max_output_chars = max_output_chars
        self._type_check = type_check
        self._limits = limits
        self._verbose = verbose

        # Normalize and validate user-provided tools
        self._user_tools = self._normalize_tools(tools)
        self._validate_tools(self._user_tools)

        # Build pydantic-ai agents (instructions are overridden per-run)
        self._action_agent: Agent[None, ActionResponse] = Agent(
            model,
            output_type=ActionResponse,
            name="rlm_action",
        )
        self._extract_agent: Agent[None, OutputT] = Agent(
            model,
            output_type=output_type,
            name="rlm_extract",
        )
        self._sub_query_agent: Agent[None, str] = Agent(
            self._sub_model,
            output_type=str,
            name="rlm_sub_query",
        )

        # Per-run usage tracker (reset at the start of each run)
        self._tracker = UsageTracker()

    # =====================================================================
    # Tool normalisation and validation
    # =====================================================================

    @staticmethod
    def _normalize_tools(
        tools: list[Callable[..., Any]] | None,
    ) -> dict[str, _ToolInfo]:
        """Convert a list of callables into a name-keyed ``_ToolInfo`` dict.

        Args:
            tools: User-provided list of tool functions.

        Returns:
            Dict mapping tool name to ``_ToolInfo``.

        Raises:
            TypeError: If *tools* is not a list or an element is not callable.
        """
        if not tools:
            return {}
        if isinstance(tools, dict):
            raise TypeError(
                "tools must be a list, not a dict.  "
                "Pass tools=[func1, func2] (names are inferred from __name__)."
            )
        result: dict[str, _ToolInfo] = {}
        for func in tools:
            if not callable(func):
                raise TypeError(
                    f"Tool {func!r} must be callable, got {type(func).__name__}"
                )
            info = _extract_tool_info(func)
            result[info.name] = info
        return result

    @staticmethod
    def _validate_tools(tools: dict[str, _ToolInfo]) -> None:
        """Ensure user tools don't clash with built-in sandbox names.

        Args:
            tools: Dict of user tools.

        Raises:
            ValueError: On invalid or reserved names.
        """
        for name in tools:
            if not name.isidentifier():
                raise ValueError(
                    f"Invalid tool name '{name}': must be a valid Python identifier"
                )
            if name in _RESERVED_TOOL_NAMES:
                raise ValueError(
                    f"Tool name '{name}' conflicts with a built-in sandbox function"
                )

    def _format_tool_docs(self) -> str:
        """Render user-tool signatures for inclusion in the system prompt.

        Returns:
            Formatted string block, or empty string if no tools.
        """
        if not self._user_tools:
            return ""
        lines = [
            "\nAdditional tools available (use these instead of standard library equivalents):"
        ]
        for info in self._user_tools.values():
            params_str = ", ".join(f"{p}: {t}" for p, t in info.params)
            lines.append(f"- `{info.name}({params_str})` - {info.desc}")
        return "\n".join(lines)

    # =====================================================================
    # Prompt construction
    # =====================================================================

    def _build_action_system_prompt(self, input_names: list[str]) -> str:
        """Build the full system prompt for the action agent.

        Args:
            input_names: Names of the input variables for this run.

        Returns:
            Complete system prompt string.
        """
        inputs_str = ", ".join(f"`{n}`" for n in input_names)

        if self._is_model_output:
            # Multi-field BaseModel output
            final_output_names = ", ".join(self._output_type.model_fields.keys())
            output_fields = "\n".join(
                f"- {translate_field_type(name, fi)}"
                for name, fi in self._output_type.model_fields.items()
            )
        else:
            # Simple scalar output (str, int, etc.)
            type_name = getattr(self._output_type, "__name__", str(self._output_type))
            final_output_names = "output"
            output_fields = f"- `output` ({type_name})"

        # Optional task instructions prefix
        task_prefix = f"{self._instructions}\n\n" if self._instructions else ""

        # Format the template
        action_body = MONTY_ACTION_INSTRUCTIONS_TEMPLATE.format(
            inputs=inputs_str,
            final_output_names=final_output_names,
            output_fields=output_fields,
            max_llm_calls=self._max_llm_calls,
            examples=self._examples,
        )

        tool_docs = self._format_tool_docs()
        return task_prefix + action_body + tool_docs

    def _build_extract_system_prompt(self) -> str:
        """Build the system prompt for the extract (fallback) agent.

        Returns:
            System prompt string.
        """
        task_instructions = ""
        if self._instructions:
            task_instructions = (
                "The trajectory was generated with the following objective:\n"
                + self._instructions
                + "\n\n"
            )
        return _EXTRACT_INSTRUCTIONS_TEMPLATE.format(
            task_instructions=task_instructions
        )

    @staticmethod
    def _build_action_user_prompt(
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Compose the per-iteration user prompt for the action agent.

        Args:
            variables: Input variable metadata list.
            history: Current REPL history.
            iteration: Zero-based iteration index.
            max_iterations: Total maximum iterations.

        Returns:
            User prompt string.
        """
        sections: list[str] = []

        # Variables section
        var_lines = [v.format() for v in variables]
        sections.append("## Variables\n" + "\n".join(var_lines))

        # History section
        sections.append("## REPL History\n" + history.format())

        # Iteration marker
        sections.append(f"## Current Iteration\n{iteration + 1}/{max_iterations}")

        return "\n\n".join(sections)

    @staticmethod
    def _build_extract_user_prompt(
        variables: list[REPLVariable],
        history: REPLHistory,
    ) -> str:
        """Compose the user prompt for the extract agent.

        Args:
            variables: Input variable metadata list.
            history: Current REPL history.

        Returns:
            User prompt string.
        """
        var_lines = [v.format() for v in variables]
        return (
            "## Variables\n"
            + "\n".join(var_lines)
            + "\n\n## REPL History\n"
            + history.format()
        )

    # =====================================================================
    # LLM sub-query tools (injected into the interpreter)
    # =====================================================================

    def _make_llm_tools(
        self,
        tracker: UsageTracker,
    ) -> dict[str, Callable[..., Any]]:
        """Create async ``llm_query`` and ``llm_query_batched`` functions.

        Both functions are ``async def`` coroutines that use
        ``await agent.run()`` directly.  When executed inside the Monty
        sandbox, returning a coroutine causes the interpreter to register
        the call as a Monty future via ``resume(future=...)``.  This means
        multiple ``llm_query`` calls issued before their results are consumed
        automatically run concurrently -- no threading required.

        Args:
            tracker: Usage tracker to record token spend.

        Returns:
            Dict with ``llm_query`` and ``llm_query_batched`` async callables.
        """
        state = {"call_count": 0}
        sub_agent = self._sub_query_agent
        max_llm_calls = self._max_llm_calls

        def _check_and_increment(n: int = 1) -> None:
            """Enforce the per-run sub-LLM call limit.

            Safe without a lock because the interpreter steps through
            MontySnapshot calls sequentially -- this is always called
            from the interpreter's stepping loop before the async task
            is created.
            """
            if state["call_count"] + n > max_llm_calls:
                raise RuntimeError(
                    f"LLM call limit exceeded: {state['call_count']} + {n} > "
                    f"{max_llm_calls}.  Use Python code for aggregation "
                    f"instead of making more LLM calls."
                )
            state["call_count"] += n

        async def _query_async(prompt: str) -> str:
            """Execute a single sub-LLM query asynchronously."""
            result = await sub_agent.run(prompt)
            tracker.incr(result.usage())
            return result.output

        async def llm_query(prompt: str) -> str:
            """Query the sub-LLM with a single prompt string.

            This is an async function -- when called from the Monty sandbox,
            the interpreter dispatches it as a Monty future for implicit
            concurrency.

            Args:
                prompt: The prompt to send to the LLM.

            Returns:
                The LLM's text response.
            """
            if not prompt:
                raise ValueError("prompt cannot be empty")
            _check_and_increment(1)
            return await _query_async(prompt)

        async def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query the sub-LLM with multiple prompts concurrently.

            Uses ``asyncio.gather`` to run all prompts in parallel.  This
            is still useful as an explicit batching API (atomic rate-limit
            reservation, single external call from the sandbox), even though
            individual ``llm_query`` calls also get implicit concurrency
            through Monty's futures mechanism.

            Args:
                prompts: List of prompt strings.

            Returns:
                List of response strings in the same order as *prompts*.
            """
            if not prompts:
                return []
            _check_and_increment(len(prompts))

            # Fire all queries concurrently via asyncio.gather
            tasks = [_query_async(p) for p in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            output: list[str] = []
            for r in results:
                if isinstance(r, BaseException):
                    output.append(f"[ERROR] {r}")
                else:
                    output.append(r)
            return output

        return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}

    # =====================================================================
    # Execution helpers
    # =====================================================================

    def _prepare_execution_tools(
        self, tracker: UsageTracker
    ) -> dict[str, Callable[..., Any]]:
        """Merge LLM tools with user-provided tools for the interpreter.

        Args:
            tracker: Usage tracker for this run.

        Returns:
            Combined tools dict.
        """
        execution_tools = self._make_llm_tools(tracker)
        # Add user-provided tool functions
        for name, info in self._user_tools.items():
            execution_tools[name] = info.func
        return execution_tools

    @staticmethod
    def _format_output(output: str) -> str:
        """Provide a fallback message when code produces no visible output.

        Args:
            output: Raw output string (may be empty).

        Returns:
            The output, or a helpful reminder to use ``print()``.
        """
        if not output:
            return "(no output - did you forget to print?)"
        return output

    def _process_final_output(
        self,
        result: FinalOutput,
    ) -> tuple[Any | None, str | None]:
        """Validate and parse a ``FinalOutput`` into the user's output type.

        Handles both ``BaseModel`` subclasses (multi-field) and simple scalar
        types like ``str`` (single ``output`` field).

        Args:
            result: The ``FinalOutput`` returned by ``SUBMIT()`` in the sandbox.

        Returns:
            Tuple of ``(parsed_output, None)`` on success, or
            ``(None, error_message)`` on failure.
        """
        raw = result.output

        if not isinstance(raw, dict):
            return None, (
                f"[Error] SUBMIT returned {type(raw).__name__}, expected keyword arguments."
            )

        # --- Simple / scalar output type ---
        if not self._is_model_output:
            # For scalar types the sandbox calls SUBMIT(output=value)
            if "output" not in raw:
                # Try to grab the first (and hopefully only) value
                if len(raw) == 1:
                    value = next(iter(raw.values()))
                else:
                    return None, (
                        "[Error] For scalar output use SUBMIT(output=value) "
                        "or SUBMIT with a single keyword argument."
                    )
            else:
                value = raw["output"]

            try:
                parsed = self._output_type(value)
                return parsed, None
            except (TypeError, ValueError) as e:
                type_name = getattr(
                    self._output_type, "__name__", str(self._output_type)
                )
                return None, f"[Type Error] Cannot convert to {type_name}: {e}"

        # --- BaseModel output type ---
        expected = set(self._output_type.model_fields.keys())
        missing = expected - set(raw.keys())
        if missing:
            return None, (
                f"[Error] Missing output fields: {sorted(missing)}.  "
                f"Use SUBMIT({', '.join(expected)})"
            )

        try:
            parsed = self._output_type.model_validate(raw)
            return parsed, None
        except pydantic.ValidationError as e:
            return None, f"[Type Error] {e}"

    def _process_execution_result(
        self,
        action: ActionResponse,
        exec_result: Any,
        history: REPLHistory,
    ) -> RLMResult[OutputT] | REPLHistory:
        """Process an interpreter result and decide whether the run is done.

        Args:
            action: The LLM's action (reasoning + code).
            exec_result: Return value from ``interpreter.execute()``.
            history: Current REPL history.

        Returns:
            ``RLMResult`` if ``SUBMIT`` was called successfully and output
            validated, otherwise an updated ``REPLHistory`` to continue.
        """
        code = _strip_code_fences(action.code)

        # Handle error strings from caught exceptions
        if isinstance(exec_result, str) and exec_result.startswith("[Error]"):
            output = self._format_output(exec_result)
            return history.append(reasoning=action.reasoning, code=code, output=output)

        # Handle SUBMIT -> FinalOutput
        if isinstance(exec_result, FinalOutput):
            parsed, error = self._process_final_output(exec_result)

            if error:
                return history.append(
                    reasoning=action.reasoning, code=code, output=error
                )

            # Success -- build final result
            final_history = history.append(
                reasoning=action.reasoning,
                code=code,
                output=f"SUBMIT: {parsed}",
            )
            return RLMResult(
                output=parsed,
                trajectory=[e.model_dump() for e in final_history],
                final_reasoning=action.reasoning,
                usage=self._tracker.usage,
            )

        # Normal (non-final) output
        if isinstance(exec_result, list):
            output = "\n".join(map(str, exec_result))
        else:
            output = str(exec_result) if exec_result else ""

        output = self._format_output(output)
        if self._verbose:
            logger.info(output[: self._max_output_chars])
        return history.append(reasoning=action.reasoning, code=code, output=output)

    # =====================================================================
    # Main async execution loop
    # =====================================================================

    async def run(self, **inputs: Any) -> RLMResult[OutputT]:
        """Execute the RLM loop and return a structured result.

        Each keyword argument becomes a named variable inside the Monty
        sandbox.  The LLM iteratively writes code to examine the inputs,
        call sub-LLMs, and ultimately ``SUBMIT()`` its answer.

        Args:
            **inputs: Named input values (e.g. ``query="..."``,
                ``context="...long text..."``).

        Returns:
            :class:`RLMResult` containing the parsed output, trajectory,
            and accumulated token usage.

        Raises:
            ValueError: If no inputs are provided.

        Example:
            >>> result = await rlm.run(
            ...     query="Summarise the document",
            ...     context=long_document_text,
            ... )
            >>> print(result.output)
        """
        if not inputs:
            raise ValueError("At least one input keyword argument is required.")

        # Reset per-run state
        self._tracker.reset()

        # Build variable metadata for the LLM prompt
        variables = [
            REPLVariable.from_value(name, value) for name, value in inputs.items()
        ]
        input_names = list(inputs.keys())

        # Build system prompts
        action_system_prompt = self._build_action_system_prompt(input_names)
        extract_system_prompt = self._build_extract_system_prompt()

        # Create a fresh interpreter for this run
        interpreter = MontyCodeInterpreter(
            type_check=self._type_check,
            limits=self._limits,
        )
        interpreter.output_fields = get_output_fields_info(self._output_type)

        # Inject execution tools (llm_query + user tools) into the interpreter
        execution_tools = self._prepare_execution_tools(self._tracker)
        interpreter.tools.update(execution_tools)

        history = REPLHistory(max_output_chars=self._max_output_chars)

        for iteration in range(self._max_iterations):
            # Build user prompt for this iteration
            user_prompt = self._build_action_user_prompt(
                variables,
                history,
                iteration,
                self._max_iterations,
            )

            # Call the action agent (async)
            with self._action_agent.override(instructions=action_system_prompt):
                agent_result = await self._action_agent.run(user_prompt)
            self._tracker.incr(agent_result.usage())
            action = agent_result.output

            if self._verbose:
                logger.info(
                    "RLM iteration %d/%d\nReasoning: %s\nCode:\n%s",
                    iteration + 1,
                    self._max_iterations,
                    action.reasoning,
                    action.code,
                )

            # Execute code in the Monty sandbox.  Async tools (llm_query)
            # are dispatched as Monty futures and resolved concurrently.
            try:
                code = _strip_code_fences(action.code)
                exec_result = await interpreter.execute(
                    code,
                    dict(inputs),
                )
            except (CodeExecutionError, SyntaxError) as e:
                exec_result = f"[Error] {e}"

            # Decide whether we're done
            outcome = self._process_execution_result(action, exec_result, history)
            if isinstance(outcome, RLMResult):
                return outcome
            history = outcome

        # Max iterations exhausted -- use the extract agent as a fallback
        return await self._extract_fallback(
            variables,
            history,
            extract_system_prompt,
        )

    async def _extract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        extract_system_prompt: str,
    ) -> RLMResult[OutputT]:
        """Use the extract agent to produce output when max iterations are hit.

        Args:
            variables: Input variable metadata.
            history: Full REPL trajectory.
            extract_system_prompt: System prompt for the extract agent.

        Returns:
            :class:`RLMResult` with the extracted output.
        """
        logger.warning(
            "RLM reached max iterations (%d), using extract agent for final output.",
            self._max_iterations,
        )
        user_prompt = self._build_extract_user_prompt(variables, history)

        with self._extract_agent.override(instructions=extract_system_prompt):
            agent_result = await self._extract_agent.run(user_prompt)
        self._tracker.incr(agent_result.usage())

        return RLMResult(
            output=agent_result.output,
            trajectory=[e.model_dump() for e in history],
            final_reasoning="Extract forced final output (max iterations reached)",
            usage=self._tracker.usage,
        )

    # =====================================================================
    # Agent-tool surface
    # =====================================================================

    def as_toolset(self, name: str = "rlm_analyze") -> FunctionToolset:
        """Return a FunctionToolset suitable for use as a pydantic-ai toolset."""
        rlm = self  # capture for closure

        async def run_rlm(query: str, context: str) -> str:
            """Analyse *context* using an RLM agent to answer *query*."""
            result = await rlm.run(query=query, context=context)
            # Return JSON so the parent agent can parse it
            if isinstance(result.output, BaseModel):
                return result.output.model_dump_json()
            return str(result.output)

        run_rlm.__name__ = name
        run_rlm.__qualname__ = name
        # Get a readable name for the output type
        output_type_name = getattr(
            self._output_type, "__name__", str(self._output_type)
        )

        run_rlm.__doc__ = (
            f"Analyse large context using a Recursive Language Model (RLM).\n\n"
            f"The RLM iteratively writes and executes Python code in a sandbox to explore\n"
            f"the context programmatically, call sub-LLMs for semantic analysis, and build\n"
            f"up structured answers without feeding the entire context into a single prompt.\n\n"
            f"Args:\n"
            f"    query (str): The question or task to answer using the provided context.\n"
            f"    context (str): The data to analyse (can be very large, e.g., documents, logs).\n\n"
            f"Returns:\n"
            f"    An instance of {output_type_name}."
        )
        toolset = FunctionToolset()
        toolset.add_function(
            run_rlm, takes_context=True, name=name, description=run_rlm.__doc__
        )
        return toolset

    # =====================================================================
    # Convenience properties
    # =====================================================================

    @property
    def tracker(self) -> UsageTracker:
        """The per-run usage tracker (useful for inspecting usage after ``run()``)."""
        return self._tracker

    @property
    def output_type(self) -> type[OutputT]:
        """The output type (``BaseModel`` subclass or simple scalar type)."""
        return self._output_type
