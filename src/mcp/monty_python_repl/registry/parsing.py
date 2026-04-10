"""Docstring and signature parsing helpers for registry tools."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
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


@dataclass(slots=True)
class ParsedToolDocstring:
    """Structured metadata parsed from a Google-style docstring."""

    summary: str
    parameter_descriptions: dict[str, str] = field(default_factory=dict)
    example: str | None = None


@dataclass(slots=True)
class ToolSpec:
    """Structured tool registration payload before registry insertion."""

    name: str
    func: Callable[..., Any]
    description: str
    usage_example: str | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)
    collection: str | None = None
    collection_description: str | None = None
    arguments: tuple[ToolArgument, ...] = field(default_factory=tuple)
    return_annotation: str | None = None


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


def parse_tool_docstring(func: Callable[..., Any]) -> ParsedToolDocstring:
    """Parse a Google-style docstring with griffe metadata helpers."""
    docstring = inspect.getdoc(_unwrap_callable(func))
    if not docstring:
        return ParsedToolDocstring(summary="No description provided.")

    sections = parse_google(Docstring(docstring), warnings=False)
    summary = ""
    parameter_descriptions: dict[str, str] = {}
    example: str | None = None

    for section in sections:
        if section.kind == DocstringSectionKind.text and not summary:
            summary = str(section.value).strip()
            continue

        if section.kind == DocstringSectionKind.parameters:
            for parameter in section.value:
                description = (parameter.description or "").strip()
                if description:
                    parameter_descriptions[str(parameter.name)] = description
            continue

        if section.kind == DocstringSectionKind.examples and example is None:
            example_chunks = [
                content.strip()
                for kind, content in section.value
                if kind == DocstringSectionKind.text and content.strip()
            ]
            if example_chunks:
                example = "\n".join(example_chunks)

    return ParsedToolDocstring(
        summary=summary or "No description provided.",
        parameter_descriptions=parameter_descriptions,
        example=example,
    )


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
    description: str | None = None,
    usage_example: str | None = None,
    categories: Iterable[str] = (),
    collection: str | None = None,
    collection_description: str | None = None,
) -> ToolSpec:
    """Build a tool spec from a callable and optional overrides."""
    parsed_docstring = parse_tool_docstring(func)
    resolved_signature = inspect.signature(func)
    return ToolSpec(
        name=name or func.__name__,
        func=func,
        description=(description or parsed_docstring.summary).strip(),
        usage_example=usage_example or parsed_docstring.example,
        categories=tuple(str(category) for category in categories),
        collection=collection,
        collection_description=(collection_description or "").strip() or None,
        arguments=build_tool_arguments(
            func,
            parameter_descriptions=parsed_docstring.parameter_descriptions,
        ),
        return_annotation=_stringify_annotation(resolved_signature.return_annotation),
    )


__all__ = [
    "ParsedToolDocstring",
    "ToolArgument",
    "ToolSpec",
    "build_tool_arguments",
    "build_tool_spec",
    "parse_tool_docstring",
]
