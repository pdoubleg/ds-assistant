"""Core registry types for the Monty Python REPL."""

from __future__ import annotations

import inspect
from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ClassVar

from .parsing import ToolArgument, ToolSpec, build_tool_spec
from .utils import safe_json_value


@dataclass(slots=True)
class ToolMetadata:
    """Decorator-supplied metadata for a collection tool."""

    name: str | None = None
    description: str | None = None
    usage_example: str | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class RegisteredFunction:
    """Metadata describing a sandbox function exposed to Monty."""

    name: str
    func: Callable[..., Any]
    description: str
    usage_example: str | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)
    collection: str | None = None
    arguments: tuple[ToolArgument, ...] = field(default_factory=tuple)
    return_annotation: str | None = None

    @property
    def signature(self) -> str:
        """Return the Python signature for the registered callable."""
        return str(inspect.signature(self.func))

    def to_help_dict(self) -> dict[str, Any]:
        """Render the function metadata for help responses.

        Returns:
            dict[str, Any]: JSON-friendly function metadata.
        """
        return {
            "name": self.name,
            "signature": f"{self.name}{self.signature}",
            "description": self.description,
            "categories": list(self.categories),
            "collection": self.collection,
            "usage_example": self.usage_example,
            "arguments": [argument.to_help_dict() for argument in self.arguments],
            "return_annotation": self.return_annotation,
        }


@dataclass(slots=True)
class RegisteredCollection:
    """Metadata describing a named collection of registry tools."""

    name: str
    description: str
    tool_names: list[str] = field(default_factory=list)

    def to_help_dict(self) -> dict[str, Any]:
        """Render collection metadata for help responses.

        Returns:
            dict[str, Any]: JSON-friendly collection metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "tool_count": len(self.tool_names),
            "tools": sorted(self.tool_names),
        }


_TOOL_METADATA_ATTR = "__monty_tool_metadata__"


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    usage_example: str | None = None,
    categories: Iterable[str] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a collection method as a Monty tool.

    Args:
        name (str | None): Optional exported tool name override.
        description (str | None): Optional summary override when the docstring is
            unavailable or should be customized.
        usage_example (str | None): Optional example surfaced in the help output.
        categories (Iterable[str]): Categories used to group related tools.

    Returns:
        Callable[[Callable[..., Any]], Callable[..., Any]]: Decorator that stores
        tool metadata on the wrapped callable.
    """
    metadata = ToolMetadata(
        name=name,
        description=description,
        usage_example=usage_example,
        categories=tuple(str(category) for category in categories),
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Attach registry metadata to the target callable.

        Args:
            func (Callable[..., Any]): Function or bound method being decorated.

        Returns:
            Callable[..., Any]: Wrapped callable with attached metadata.
        """

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Pass through to the wrapped tool implementation."""
            return func(*args, **kwargs)

        # Keep metadata discoverable on both the wrapper and the original
        # callable so inspect.unwrap-based lookups remain stable.
        setattr(func, _TOOL_METADATA_ATTR, metadata)
        setattr(wrapped, _TOOL_METADATA_ATTR, metadata)
        return wrapped

    return decorator


def _get_tool_metadata(func: Callable[..., Any]) -> ToolMetadata | None:
    """Return decorator metadata attached to a tool callable, if any.

    Args:
        func (Callable[..., Any]): Bound or unbound callable.

    Returns:
        ToolMetadata | None: Attached decorator metadata when present.
    """
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
        """Return decorated tool specs defined on the collection.

        Returns:
            list[ToolSpec]: Decorated tools registered by the collection.
        """
        specs: list[ToolSpec] = []
        for _, member in inspect.getmembers(self, predicate=callable):
            metadata = _get_tool_metadata(member)
            if metadata is None:
                continue
            specs.append(
                build_tool_spec(
                    member,
                    name=metadata.name,
                    description=metadata.description,
                    usage_example=metadata.usage_example,
                    categories=metadata.categories,
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
        """Create or update collection metadata before tool registration.

        Args:
            name (str): Collection name.
            description (str | None): Optional collection description.

        Returns:
            RegisteredCollection: Stored collection metadata.
        """
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
        description: str | None = None,
        usage_example: str | None = None,
        categories: Iterable[str] = (),
        collection: str | None = None,
        collection_description: str | None = None,
    ) -> RegisteredFunction:
        """Register a callable for sandbox use.

        Args:
            func (Callable[..., Any]): Tool callable to register.
            name (str | None): Optional exported name override.
            description (str | None): Optional summary override.
            usage_example (str | None): Optional usage example.
            categories (Iterable[str]): Tool categories.
            collection (str | None): Optional collection grouping.
            collection_description (str | None): Optional collection summary.

        Returns:
            RegisteredFunction: Stored registry entry.
        """
        return self.register_tool(
            build_tool_spec(
                func,
                name=name,
                description=description,
                usage_example=usage_example,
                categories=categories,
                collection=collection,
                collection_description=collection_description,
            )
        )

    def register_tool(self, tool_spec: ToolSpec) -> RegisteredFunction:
        """Register a pre-built tool spec.

        Args:
            tool_spec (ToolSpec): Normalized tool registration payload.

        Returns:
            RegisteredFunction: Stored registry entry.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool_spec.name in self._functions:
            raise ValueError(f"Function {tool_spec.name!r} is already registered.")

        entry = RegisteredFunction(
            name=tool_spec.name,
            func=tool_spec.func,
            description=tool_spec.description,
            usage_example=tool_spec.usage_example,
            categories=tool_spec.categories,
            collection=tool_spec.collection,
            arguments=tool_spec.arguments,
            return_annotation=tool_spec.return_annotation,
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
        """Register all decorated tools exposed by a collection.

        Args:
            collection (ToolCollection): Collection instance exposing tools.

        Returns:
            list[RegisteredFunction]: Registered tool entries.
        """
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
        """Return registry entries in alphabetical order.

        Args:
            collection (str | None): Optional collection filter.

        Returns:
            list[RegisteredFunction]: Sorted registry entries.
        """
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
        """Persist an object and return its generated handle.

        Args:
            value (Any): Object to persist.
            prefix (str): Handle prefix, such as `df` or `fig`.

        Returns:
            str: Generated handle.
        """
        next_index = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = next_index
        handle = f"{prefix}_{next_index}"
        self._objects[handle] = value
        return handle

    def get(self, handle: str, *, expected_type: type[Any] | None = None) -> Any:
        """Return a stored object by handle.

        Args:
            handle (str): Object handle.
            expected_type (type[Any] | None): Optional runtime type check.

        Returns:
            Any: Stored object.

        Raises:
            KeyError: If the handle is unknown.
            TypeError: If the stored object does not match the expected type.
        """
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
    "tool",
]
