from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from forgeloop.model_capabilities import ModelCapability
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

    @property
    def policy_identity(self) -> Any: ...

    @property
    def capability(self) -> ModelCapability | None: ...

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict],
        *,
        timeout_seconds: float,
    ) -> ModelResponse: ...
