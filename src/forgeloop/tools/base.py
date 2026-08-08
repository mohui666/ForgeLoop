from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from forgeloop.effects import EffectContext, EffectDraft, EffectRecorder


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    metadata: dict[str, Any] | None = None
    effects: tuple[EffectDraft, ...] = ()

    def as_observation(self) -> str:
        prefix = "OK" if self.ok else "ERROR"
        return f"{prefix}\n{self.output}"


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def execute(
        self, arguments: dict[str, Any], *, timeout_seconds: float
    ) -> ToolResult: ...

    def schema(self) -> dict[str, Any]: ...


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._effect_recorder: EffectRecorder | None = None
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def bind_effect_recorder(self, recorder: EffectRecorder) -> None:
        self._effect_recorder = recorder

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float,
        effect_context: EffectContext | None = None,
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"Unknown tool: {name}")
        started = time.perf_counter()
        try:
            result = tool.execute(arguments, timeout_seconds=timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - tool failures must become observations
            result = ToolResult(False, f"{type(exc).__name__}: {exc}")
        metadata = dict(result.metadata or {})
        metadata["duration_seconds"] = round(time.perf_counter() - started, 6)
        if self._effect_recorder and effect_context:
            errors: list[str] = []
            for draft in result.effects:
                try:
                    self._effect_recorder.record(
                        draft,
                        context=effect_context,
                        tool_name=name,
                    )
                except Exception as exc:  # observability must not break tool execution
                    errors.append(f"{type(exc).__name__}: {exc}")
            if errors:
                metadata["effect_recording_errors"] = errors
        return ToolResult(result.ok, result.output, metadata, result.effects)


class BaseTool:
    name = ""
    description = ""
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
