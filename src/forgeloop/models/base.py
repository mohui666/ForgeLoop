from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from forgeloop.types import Message, ModelResponse


class ModelProviderError(RuntimeError):
    """Provider failure with a concise UI message and redacted diagnostic detail."""

    def __init__(self, message: str, *, details: str = "") -> None:
        super().__init__(message)
        self.details = details or message


class ModelProvider(Protocol):
    """Stable boundary between the agent loop and a model implementation."""

    @property
    def model_id(self) -> str: ...

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict],
        *,
        timeout_seconds: float,
    ) -> ModelResponse: ...
