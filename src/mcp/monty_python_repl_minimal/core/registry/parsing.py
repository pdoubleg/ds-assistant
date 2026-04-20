"""Docstring and signature parsing helpers for standalone registry tools."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from griffe import (
    Docstring,
    DocstringSectionKind,
    GoogleOptions,
    Object as GriffeObject,
)

from ...privacy import safe_json_value


@dataclass(slots=True)
class ToolArgument:
    """Structured metadata for one registered tool argument."""

    name: str
    kind: str
    annotation: str | None = None
    description: str | None = None
    default: Any | None = None
    required: bool = True

    def to_help_dict(self) -> dict[str, Any]:
        """Render argument metadata for help responses."""
        return {
            "name": self.name,
            "kind": self.kind,
            "annotation": self.annotation,
            "description": self.description,
            "default": self.default,
            "required": self.required,
        }

    def render_signature_fragment(self) -> str:
        """Render the argument as a Python-style signature fragment.

        Returns:
            Signature fragment suitable for a callable signature.
        """
        prefix = ""
        if self.kind == "var_positional":
            prefix = "*"
        elif self.kind == "var_keyword":
            prefix = "**"

        parts = [f"{prefix}{self.name}"]
        if self.annotation:
            parts.append(f": {self.annotation}")
        if not self.required:
            parts.append(f" = {self._render_default_value()}")
        return "".join(parts)

    def render_argument_help(self) -> str:
        """Render the argument as a human-readable help line.

        Returns:
            One formatted argument line for tool help output.
        """
        label_parts = [
            self.annotation or "Any",
            "required" if self.required else "optional",
        ]
        if not self.required:
            label_parts.append(f"default={self._render_default_value()}")
        description = self.description or "No description provided."
        return f"- {self.name} ({', '.join(label_parts)}): {description}"

    def _render_default_value(self) -> str:
        """Render a stable default value representation.

        Returns:
            Python-like representation of the default value.
        """
        if isinstance(self.default, str):
            return json.dumps(self.default)
        return repr(self.default)


@dataclass(slots=True)
class ParsedToolDocstring:
    """Structured metadata parsed from a Google-style docstring."""

    summary: str
    details: str | None = None
    parameter_descriptions: dict[str, str] = field(default_factory=dict)
    returns_description: str | None = None
    example: str | None = None
    has_parameters_section: bool = False
    has_returns_section: bool = False
    has_examples_section: bool = False


class ToolDocstringValidationError(ValueError):
    """Raised when a registered tool docstring is missing required sections."""


@dataclass(slots=True)
class ToolSpec:
    """Structured tool registration payload before registry insertion."""

    name: str
    func: Callable[..., Any]
    description: str
    detailed_description: str | None = None
    usage_example: str | None = None
    collection: str | None = None
    collection_description: str | None = None
    arguments: tuple[ToolArgument, ...] = field(default_factory=tuple)
    return_annotation: str | None = None
    return_description: str | None = None


def _unwrap_callable(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the undecorated callable used for metadata extraction."""
    raw_callable = getattr(func, "__func__", func)
    return inspect.unwrap(raw_callable)


def _stringify_annotation(annotation: Any) -> str | None:
    """Render a type annotation into a stable help-friendly string."""
    if annotation is inspect.Signature.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    try:
        return inspect.formatannotation(annotation)
    except Exception:  # pragma: no cover
        return getattr(annotation, "__name__", repr(annotation))


def _clean_docstring_text(value: Any) -> str:
    """Normalize parsed docstring text for help payloads.

    Args:
        value: Parsed section content returned by `griffe`.

    Returns:
        Cleaned string with indentation normalized and empty values removed.
    """
    if value is None:
        return ""
    return inspect.cleandoc(str(value)).strip()


def _collapse_docstring_text(value: Any) -> str:
    """Normalize parsed inline docstring text into one readable line.

    Args:
        value: Parsed section content returned by `griffe`.

    Returns:
        Single-line text with internal whitespace collapsed.
    """
    cleaned_value = _clean_docstring_text(value)
    return " ".join(cleaned_value.split())


def _parse_parameter_descriptions(parameters_section: Any) -> dict[str, str]:
    """Extract per-parameter descriptions from a parsed Args section.

    Args:
        parameters_section: Parsed `griffe` parameters section.

    Returns:
        Mapping of parameter name to cleaned description text.
    """
    descriptions: dict[str, str] = {}
    for parameter in parameters_section.value:
        description = _collapse_docstring_text(getattr(parameter, "description", None))
        if description:
            descriptions[str(parameter.name)] = description
    return descriptions


def _parse_returns_description(returns_section: Any) -> str | None:
    """Extract one help-friendly returns description from a parsed section.

    Args:
        returns_section: Parsed `griffe` returns section.

    Returns:
        Joined return description text, including type annotations when present.
    """
    return_chunks: list[str] = []

    for returned_value in returns_section.value:
        description = _collapse_docstring_text(
            getattr(returned_value, "description", None)
        )
        annotation = getattr(returned_value, "annotation", None)
        rendered_annotation = _clean_docstring_text(annotation)

        if description and rendered_annotation:
            return_chunks.append(f"{rendered_annotation}: {description}")
        elif description:
            return_chunks.append(description)
        elif rendered_annotation:
            return_chunks.append(rendered_annotation)

    return "\n".join(return_chunks) if return_chunks else None


def _parse_example_block(examples_section: Any) -> str | None:
    """Extract one representative usage example from a parsed Examples section.

    Args:
        examples_section: Parsed `griffe` examples section.

    Returns:
        Example code when present, otherwise fallback example prose.
    """
    example_blocks: list[str] = []
    text_blocks: list[str] = []

    # Griffe may represent examples as code/example blocks or plain text blocks
    # depending on the source formatting. Prefer code-like examples when present.
    for kind, value in examples_section.value:
        cleaned_value = _clean_docstring_text(value)
        if not cleaned_value:
            continue
        if kind == DocstringSectionKind.examples:
            example_blocks.append(cleaned_value)
        elif kind == DocstringSectionKind.text:
            text_blocks.append(cleaned_value)

    rendered_blocks = example_blocks or text_blocks
    return "\n\n".join(rendered_blocks) if rendered_blocks else None


def parse_tool_docstring(func: Callable[..., Any]) -> ParsedToolDocstring:
    """Parse a Google-style docstring directly with `griffe`.

    Args:
        func: Registered tool callable whose docstring should be parsed.

    Returns:
        Structured docstring metadata derived from the Google-style sections.
    """
    raw_callable = _unwrap_callable(func)
    docstring = inspect.getdoc(raw_callable)
    if not docstring:
        return ParsedToolDocstring(summary="No description provided.")

    # Griffe expects a parent object for some Google parser operations.
    # See: https://github.com/mkdocstrings/griffe/issues/293
    parent = cast(GriffeObject, inspect.signature(raw_callable))
    parser_options = GoogleOptions(
        returns_named_value=False,
        returns_multiple_items=False,
    )
    sections = Docstring(
        docstring,
        lineno=1,
        parser="google",
        parent=parent,
        parser_options=parser_options,
    ).parse()

    text_section = next(
        (section for section in sections if section.kind == DocstringSectionKind.text),
        None,
    )
    parameters_section = next(
        (
            section
            for section in sections
            if section.kind == DocstringSectionKind.parameters
        ),
        None,
    )
    returns_section = next(
        (
            section
            for section in sections
            if section.kind == DocstringSectionKind.returns
        ),
        None,
    )
    examples_section = next(
        (
            section
            for section in sections
            if section.kind == DocstringSectionKind.examples
        ),
        None,
    )

    details = (
        _clean_docstring_text(text_section.value) or None if text_section else None
    )
    summary = details.split("\n\n", maxsplit=1)[0].strip() if details else ""
    parameter_descriptions = (
        _parse_parameter_descriptions(parameters_section) if parameters_section else {}
    )
    returns_description = (
        _parse_returns_description(returns_section) if returns_section else None
    )
    example = _parse_example_block(examples_section) if examples_section else None

    return ParsedToolDocstring(
        summary=summary or "No description provided.",
        details=details,
        parameter_descriptions=parameter_descriptions,
        returns_description=returns_description,
        example=example,
        has_parameters_section=parameters_section is not None,
        has_returns_section=returns_section is not None,
        has_examples_section=examples_section is not None,
    )


def validate_tool_docstring(
    func: Callable[..., Any],
    *,
    name: str | None = None,
) -> ParsedToolDocstring:
    """Validate that a tool docstring includes the required help sections."""
    parsed_docstring = parse_tool_docstring(func)
    exported_name = name or _unwrap_callable(func).__name__
    parameter_names = tuple(inspect.signature(func).parameters)
    undocumented_parameters = tuple(
        parameter_name
        for parameter_name in parameter_names
        if parameter_name not in parsed_docstring.parameter_descriptions
    )

    missing_sections: list[str] = []
    if parameter_names and not parsed_docstring.has_parameters_section:
        missing_sections.append("Args")
    if not parsed_docstring.has_returns_section:
        missing_sections.append("Returns")
    if not parsed_docstring.has_examples_section or not parsed_docstring.example:
        missing_sections.append("Examples")

    problems: list[str] = []
    if missing_sections:
        problems.append(
            "missing required docstring sections: "
            + ", ".join(section_name for section_name in missing_sections)
        )
    if undocumented_parameters:
        problems.append(
            "missing argument descriptions for: "
            + ", ".join(parameter_name for parameter_name in undocumented_parameters)
        )

    if problems:
        raise ToolDocstringValidationError(
            f"Tool {exported_name!r} has an invalid docstring; "
            + "; ".join(problems)
            + "."
        )

    return parsed_docstring


def build_tool_arguments(
    func: Callable[..., Any],
    *,
    parameter_descriptions: Mapping[str, str],
) -> tuple[ToolArgument, ...]:
    """Build structured argument metadata from the callable signature."""
    arguments: list[ToolArgument] = []
    for parameter in inspect.signature(func).parameters.values():
        default = None
        required = parameter.default is inspect.Parameter.empty
        if not required:
            default = safe_json_value(parameter.default, max_items=5, max_chars=120)

        arguments.append(
            ToolArgument(
                name=parameter.name,
                kind=parameter.kind.name.lower(),
                annotation=_stringify_annotation(parameter.annotation),
                description=parameter_descriptions.get(parameter.name),
                default=default,
                required=required,
            )
        )
    return tuple(arguments)


def build_tool_spec(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    collection: str | None = None,
    collection_description: str | None = None,
) -> ToolSpec:
    """Build a tool spec from a callable and optional overrides."""
    parsed_docstring = validate_tool_docstring(func, name=name)
    resolved_signature = inspect.signature(func)
    return ToolSpec(
        name=name or func.__name__,
        func=func,
        description=parsed_docstring.summary.strip(),
        detailed_description=(parsed_docstring.details or "").strip() or None,
        usage_example=parsed_docstring.example,
        collection=collection,
        collection_description=(collection_description or "").strip() or None,
        arguments=build_tool_arguments(
            func,
            parameter_descriptions=parsed_docstring.parameter_descriptions,
        ),
        return_annotation=_stringify_annotation(resolved_signature.return_annotation),
        return_description=(parsed_docstring.returns_description or "").strip() or None,
    )


__all__ = [
    "ParsedToolDocstring",
    "ToolArgument",
    "ToolDocstringValidationError",
    "ToolSpec",
    "build_tool_arguments",
    "build_tool_spec",
    "parse_tool_docstring",
    "validate_tool_docstring",
]
