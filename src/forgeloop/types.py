from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_message_value(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    reported_total_tokens: int | None = None
    usage_source: str | None = None
    cost_source: str = "unknown"
    model: str | None = None
    provider: str | None = None
    latency_seconds: float | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.reported_total_tokens is not None:
            return self.reported_total_tokens
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage = field(default_factory=ModelUsage)
    finish_reason: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    assistant_message_fields: dict[str, Any] = field(default_factory=dict)

    def as_assistant_message(self) -> Message:
        message: Message = {"role": "assistant", "content": self.content or ""}
        message.update(self.assistant_message_fields)
        if self.tool_calls:
            message["tool_calls"] = [
                call.as_message_value() for call in self.tool_calls
            ]
        return message
