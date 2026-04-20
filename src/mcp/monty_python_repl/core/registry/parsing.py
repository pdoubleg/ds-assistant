"""Docstring and signature parsing helpers for registry tools."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from griffe import Docstring, DocstringSectionKind, parse_google

from .utils import safe_json_value


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
            str: Signature fragment suitable for a callable signature.
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
            str: One formatted argument line for tool help output.
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
            str: Python-like representation of the default value.
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
    except Exception:  # pragma: no cover - defensive fallback
        return getattr(annotation, "__name__", repr(annotation))


def _split_google_docstring_sections(docstring: str) -> dict[str, str]:
    """Split a Google-style docstring into coarse top-level sections."""
    section_aliases = {
        "Args:": "args",
        "Arguments:": "args",
        "Returns:": "returns",
        "Examples:": "examples",
        "Example:": "examples",
    }
    sections: dict[str, list[str]] = {
        "text": [],
        "args": [],
        "returns": [],
        "examples": [],
    }
    current_section = "text"

    for line in docstring.splitlines():
        normalized = line.strip()
        if normalized in section_aliases:
            current_section = section_aliases[normalized]
            continue
        sections[current_section].append(line)

    return {
        section_name: "\n".join(lines).strip()
        for section_name, lines in sections.items()
    }


def _parse_parameter_block(section_text: str) -> dict[str, str]:
    """Parse a Google-style ``Args:`` block into per-parameter descriptions."""
    if not section_text:
        return {}

    parameter_pattern = re.compile(
        r"^\s*(?P<name>[\w_]+)\s*(?:\([^)]*\))?:\s*(?P<description>.*)$"
    )
    descriptions: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in section_text.splitlines():
        matched = parameter_pattern.match(line)
        if matched:
            if current_name is not None:
                descriptions[current_name] = " ".join(
                    part.strip() for part in current_lines if part.strip()
                ).strip()
            current_name = matched.group("name")
            current_lines = [matched.group("description")]
            continue

        if current_name is not None:
            current_lines.append(line.strip())

    if current_name is not None:
        descriptions[current_name] = " ".join(
            part.strip() for part in current_lines if part.strip()
        ).strip()

    return {
        name: description for name, description in descriptions.items() if description
    }


def _parse_returns_block(section_text: str) -> str | None:
    """Parse a Google-style ``Returns:`` block into one help string."""
    if not section_text:
        return None
    return (
        "\n".join(line.rstrip() for line in section_text.splitlines()).strip() or None
    )


def _parse_examples_block(section_text: str) -> str | None:
    """Parse a Google-style ``Examples:`` block into the help example payload."""
    if not section_text:
        return None
    return (
        "\n".join(line.rstrip() for line in section_text.splitlines()).strip() or None
    )


def parse_tool_docstring(func: Callable[..., Any]) -> ParsedToolDocstring:
    """Parse a Google-style docstring with griffe metadata helpers."""
    docstring = inspect.getdoc(_unwrap_callable(func))
    if not docstring:
        return ParsedToolDocstring(summary="No description provided.")

    fallback_sections = _split_google_docstring_sections(docstring)
    sections = parse_google(Docstring(docstring), warnings=False)
    summary = ""
    details: str | None = None
    parameter_descriptions: dict[str, str] = {}
    returns_description: str | None = None
    example: str | None = None
    has_parameters_section = False
    has_returns_section = False
    has_examples_section = False

    for section in sections:
        if section.kind == DocstringSectionKind.text and not summary:
            details = str(section.value).strip()
            summary = details.split("\n\n", maxsplit=1)[0].strip()
            continue

        if section.kind == DocstringSectionKind.parameters:
            has_parameters_section = True
            for parameter in section.value:
                description = (parameter.description or "").strip()
                if description:
                    parameter_descriptions[str(parameter.name)] = description
            continue

        if section.kind == DocstringSectionKind.examples and example is None:
            has_examples_section = True
            example_chunks = [
                content.strip()
                for kind, content in section.value
                if kind == DocstringSectionKind.text and content.strip()
            ]
            if example_chunks:
                example = "\n".join(example_chunks)
            continue

        if section.kind == DocstringSectionKind.returns and returns_description is None:
            has_returns_section = True
            return_chunks: list[str] = []
            for returned_value in section.value:
                description = getattr(returned_value, "description", None)
                annotation = getattr(returned_value, "annotation", None)
                rendered_annotation = None
                if annotation is not None:
                    rendered_annotation = _stringify_annotation(annotation)
                if description and rendered_annotation:
                    return_chunks.append(
                        f"{rendered_annotation}: {str(description).strip()}"
                    )
                elif description:
                    return_chunks.append(str(description).strip())
                elif rendered_annotation:
                    return_chunks.append(str(rendered_annotation).strip())
            if return_chunks:
                returns_description = "\n".join(
                    chunk for chunk in return_chunks if chunk
                )

    if not summary and fallback_sections["text"]:
        details = fallback_sections["text"]
        summary = details.split("\n\n", maxsplit=1)[0].strip()
    if not has_parameters_section and fallback_sections["args"]:
        parameter_descriptions = _parse_parameter_block(fallback_sections["args"])
        has_parameters_section = bool(parameter_descriptions)
    if not has_returns_section and fallback_sections["returns"]:
        returns_description = _parse_returns_block(fallback_sections["returns"])
        has_returns_section = returns_description is not None
    if not has_examples_section and fallback_sections["examples"]:
        example = _parse_examples_block(fallback_sections["examples"])
        has_examples_section = example is not None

    return ParsedToolDocstring(
        summary=summary or "No description provided.",
        details=details,
        parameter_descriptions=parameter_descriptions,
        returns_description=returns_description,
        example=example,
        has_parameters_section=has_parameters_section,
        has_returns_section=has_returns_section,
        has_examples_section=has_examples_section,
    )


def validate_tool_docstring(
    func: Callable[..., Any],
    *,
    name: str | None = None,
) -> ParsedToolDocstring:
    """Validate that a tool docstring includes the required help sections.

    Args:
        func (Callable[..., Any]): Tool callable being registered.
        name (str | None): Optional exported tool name override for error messages.

    Returns:
        ParsedToolDocstring: Parsed docstring metadata when validation succeeds.

    Raises:
        ToolDocstringValidationError: If required docstring sections or parameter
            descriptions are missing.
    """
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
