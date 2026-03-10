"""A2UI transport contracts shared across the backend."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class A2UIComponent(BaseModel):
    """Represents a single A2UI component to be rendered on the frontend.

    Notes:
        - This is the common interface for all A2UI components.
        - Should be serialized as a dict, e.g. ``component.model_dump()`` when adding to state.

    Attributes:
        id: Unique component identifier.
        type: Component type string (e.g., 'a2ui.DocumentCard').
        props: Component-specific properties passed to the React renderer.
        layout: Optional layout hints (width, position, className).
        zone: Semantic zone for layout grouping.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    props: dict[str, Any] = Field(default_factory=dict)
    layout: dict[str, Any] | None = None
    zone: str | None = None


class A2UIConvertible(BaseModel, ABC):
    """Base model for anything that can render to an A2UI component."""

    @abstractmethod
    def to_a2ui_component(self) -> A2UIComponent:
        """Return this model as an A2UI component."""
        raise NotImplementedError
