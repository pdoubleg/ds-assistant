"""Standalone exception types for the hackathon Monty package."""

from __future__ import annotations


class CodeExecutionError(Exception):
    """Raised when sandboxed code execution fails at runtime."""


__all__ = ["CodeExecutionError"]
