"""Core registry types for the standalone hackathon Monty REPL."""

from __future__ import annotations

import inspect
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ClassVar

from ...privacy import safe_json_value
from .parsing import ToolArgument, ToolSpec, build_tool_spec


@dataclass(slots=True)
class ToolMetadata:
    """Marker metadata attached to decorated collection tools."""


@dataclass(slots=True)
class RegisteredFunction:
    """Metadata describing a sandbox function exposed to Monty."""

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

    @property
    def signature(self) -> str:
        """Return the Python signature for the registered callable."""
        return str(inspect.signature(self.func))

    @property
    def full_signature(self) -> str:
        """Return the exported callable signature including the tool name."""
        return self.render_signature(multiline=False)

    @property
    def usage_guidance(self) -> list[str]:
        """Return user-facing usage guidance for the registered function."""
        return self._build_usage_guidance()

    def render_signature(self, *, multiline: bool = False, indent: str = "    ") -> str:
        """Render a help-friendly callable signature.

        Args:
            multiline: Whether to render one argument per line.
            indent: Indentation used for multiline argument lines.

        Returns:
            Formatted signature text.
        """
        parts: list[str] = []
        inserted_kw_marker = False
        for argument in self.arguments:
            if argument.kind == "keyword_only" and not inserted_kw_marker:
                parts.append("*")
                inserted_kw_marker = True
            parts.append(argument.render_signature_fragment())

        suffix = f" -> {self.return_annotation}" if self.return_annotation else ""
        if not multiline:
            return f"{self.name}({', '.join(parts)}){suffix}"

        if not parts:
            return f"{self.name}(){suffix}"

        rendered_arguments = ",\n".join(f"{indent}{part}" for part in parts)
        return f"{self.name}(\n{rendered_arguments},\n){suffix}"

    def _build_usage_guidance(self) -> list[str]:
        """Build Monty-specific guidance for using a registered function."""
        guidance = [
            "Call this helper directly inside `execute(...)` code, not as a method on a dataframe or collection object.",
            "Keep `execute(...)` focused on orchestrating registered helpers, assigning handle-bearing results, and inspecting structured summaries.",
            "The `execute(...)` runtime is sandboxed, so standard Python file, compilation, or introspection operations may be restricted and may fail with privacy-sanitized errors.",
        ]
        if self.collection:
            guidance.append(
                f"Use `help({self.collection!r})` to discover related helpers in the same collection."
            )
        if self.collection == "workspace":
            guidance.append(
                "Prefer these workspace helpers over direct `open(...)`, `Path.write_text(...)`, `Path.read_text(...)`, or manual file persistence logic inside `execute(...)`."
            )
        if any(argument.name.endswith("_handle") for argument in self.arguments):
            guidance.append(
                "Arguments ending in `_handle` expect a stored handle string returned by an earlier Monty step."
            )
        if self.usage_example:
            guidance.append(
                "Start from the usage example, then adapt the variable names and paths to the current session."
            )
        return guidance

    def to_help_dict(self, *, detailed: bool = False) -> dict[str, Any]:
        """Render the function metadata for help responses."""
        payload = {
            "name": self.name,
            "signature": self.full_signature,
            "description": self.description,
            "collection": self.collection,
            "usage_example": self.usage_example,
            "arguments": [argument.to_help_dict() for argument in self.arguments],
            "return_annotation": self.return_annotation,
        }
        if detailed:
            payload.update(
                {
                    "detailed_description": self.detailed_description
                    or self.description,
                    "collection_description": self.collection_description,
                    "return_value": {
                        "annotation": self.return_annotation,
                        "description": self.return_description,
                    },
                    "usage_guidance": self.usage_guidance,
                }
            )
        return payload


@dataclass(slots=True)
class RegisteredCollection:
    """Metadata describing a named collection of registry tools."""

    name: str
    description: str
    tool_names: list[str] = field(default_factory=list)

    def to_help_dict(self) -> dict[str, Any]:
        """Render collection metadata for help responses."""
        return {
            "name": self.name,
            "description": self.description,
            "tool_count": len(self.tool_names),
            "tools": self.sorted_tool_names(),
        }

    def sorted_tool_names(self) -> list[str]:
        """Return collection tool names in stable alphabetical order."""
        return sorted(self.tool_names)


_TOOL_METADATA_ATTR = "__monty_tool_metadata__"


def tool(
    func: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
    """Mark a collection method as a Monty tool."""
    metadata = ToolMetadata()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Attach registry metadata to the target callable."""

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        setattr(func, _TOOL_METADATA_ATTR, metadata)
        setattr(wrapped, _TOOL_METADATA_ATTR, metadata)
        return wrapped

    if func is None:
        return decorator
    return decorator(func)


def _get_tool_metadata(func: Callable[..., Any]) -> ToolMetadata | None:
    """Return decorator metadata attached to a tool callable, if any."""
    raw_callable = getattr(func, "__func__", func)
    return getattr(inspect.unwrap(raw_callable), _TOOL_METADATA_ATTR, None)


class ToolCollection(ABC):
    """Base class for grouping related Monty tools by task area."""

    name: ClassVar[str]
    description: ClassVar[str] = ""

    @property
    def collection_name(self) -> str:
        """Return the exported collection name."""
        if getattr(type(self), "name", None):
            return str(type(self).name)
        return self.__class__.__name__.removesuffix("Collection").lower()

    @property
    def collection_description(self) -> str:
        """Return the exported collection description."""
        if getattr(type(self), "description", None):
            return str(type(self).description).strip()
        return (inspect.getdoc(type(self)) or "").strip()

    def tools(self) -> list[ToolSpec]:
        """Return decorated tool specs defined on the collection."""
        specs: list[ToolSpec] = []
        for _, member in inspect.getmembers(self, predicate=callable):
            if _get_tool_metadata(member) is None:
                continue
            specs.append(
                build_tool_spec(
                    member,
                    collection=self.collection_name,
                    collection_description=self.collection_description,
                )
            )
        return specs


class FunctionRegistry:
    """Declarative registry of sandbox functions and tool collections."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._functions: dict[str, RegisteredFunction] = {}
        self._collections: dict[str, RegisteredCollection] = {}

    def _ensure_collection(
        self,
        name: str,
        *,
        description: str | None = None,
    ) -> RegisteredCollection:
        """Create or update collection metadata before tool registration."""
        collection = self._collections.get(name)
        if collection is None:
            collection = RegisteredCollection(
                name=name,
                description=(description or "").strip(),
            )
            self._collections[name] = collection
            return collection

        if description and not collection.description:
            collection.description = description.strip()
        return collection

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        collection: str | None = None,
        collection_description: str | None = None,
    ) -> RegisteredFunction:
        """Register a callable for sandbox use."""
        return self.register_tool(
            build_tool_spec(
                func,
                name=name,
                collection=collection,
                collection_description=collection_description,
            )
        )

    def register_tool(self, tool_spec: ToolSpec) -> RegisteredFunction:
        """Register a pre-built tool spec."""
        if tool_spec.name in self._functions:
            raise ValueError(f"Function {tool_spec.name!r} is already registered.")

        entry = RegisteredFunction(
            name=tool_spec.name,
            func=tool_spec.func,
            description=tool_spec.description,
            detailed_description=tool_spec.detailed_description,
            usage_example=tool_spec.usage_example,
            collection=tool_spec.collection,
            collection_description=tool_spec.collection_description,
            arguments=tool_spec.arguments,
            return_annotation=tool_spec.return_annotation,
            return_description=tool_spec.return_description,
        )
        self._functions[entry.name] = entry

        if tool_spec.collection:
            collection = self._ensure_collection(
                tool_spec.collection,
                description=tool_spec.collection_description,
            )
            collection.tool_names.append(entry.name)

        return entry

    def register_collection(
        self, collection: ToolCollection
    ) -> list[RegisteredFunction]:
        """Register all decorated tools exposed by a collection."""
        self._ensure_collection(
            collection.collection_name,
            description=collection.collection_description,
        )
        return [self.register_tool(tool_spec) for tool_spec in collection.tools()]

    def get(self, name: str) -> RegisteredFunction | None:
        """Return a single registry entry by name."""
        return self._functions.get(name)

    def get_collection(self, name: str) -> RegisteredCollection | None:
        """Return a single registered collection by name."""
        return self._collections.get(name)

    def entries(self, *, collection: str | None = None) -> list[RegisteredFunction]:
        """Return registry entries in alphabetical order."""
        if collection is None:
            names = sorted(self._functions)
        else:
            names = sorted(
                name
                for name, entry in self._functions.items()
                if entry.collection == collection
            )
        return [self._functions[name] for name in names]

    def collections(self) -> list[RegisteredCollection]:
        """Return registered collections in alphabetical order."""
        return [self._collections[name] for name in sorted(self._collections)]

    def exported_tools(self) -> dict[str, Callable[..., Any]]:
        """Return the callable mapping needed by the interpreter."""
        return {entry.name: entry.func for entry in self.entries()}


class ObjectStore:
    """Store host-side Python objects behind Monty-safe string handles."""

    def __init__(self) -> None:
        """Initialize an empty in-memory object store."""
        self._objects: dict[str, Any] = {}
        self._counters: dict[str, int] = {}

    def put(self, value: Any, *, prefix: str) -> str:
        """Persist an object and return its generated handle."""
        next_index = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = next_index
        handle = f"{prefix}_{next_index}"
        self._objects[handle] = value
        return handle

    def get(self, handle: str, *, expected_type: type[Any] | None = None) -> Any:
        """Return a stored object by handle."""
        if handle not in self._objects:
            raise KeyError(f"Unknown object handle: {handle}")

        value = self._objects[handle]
        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(
                f"Handle {handle!r} does not reference a {expected_type.__name__} object."
            )
        return value

    def summary(self, handle: str) -> dict[str, Any]:
        """Return a JSON-friendly summary for a stored object."""
        return {"handle": handle, "value": safe_json_value(self.get(handle))}

    def list_handles(self) -> list[str]:
        """Return all stored handles in insertion order."""
        return list(self._objects)


__all__ = [
    "FunctionRegistry",
    "ObjectStore",
    "RegisteredCollection",
    "RegisteredFunction",
    "ToolArgument",
    "ToolCollection",
    "ToolMetadata",
    "_get_tool_metadata",
    "tool",
]
