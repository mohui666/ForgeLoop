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
        self._run_context: dict[str, Any] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool
        binder = getattr(tool, "bind_run_context", None)
        if self._run_context and callable(binder):
            binder(**self._run_context)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def bind_effect_recorder(self, recorder: EffectRecorder) -> None:
        self._effect_recorder = recorder

    def bind_run_context(self, *, base_head: str | None) -> None:
        """Bind stable run-start identity to tools that consume it."""

        self._run_context = {"base_head": base_head}
        for tool in self._tools.values():
            binder = getattr(tool, "bind_run_context", None)
            if callable(binder):
                binder(base_head=base_head)

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
        argument_error = validate_tool_arguments(tool.parameters, arguments)
        if argument_error:
            return ToolResult(
                False,
                f"Invalid arguments for {name}: {argument_error}",
                {
                    "execution_blocked": True,
                    "reason": "invalid_tool_arguments",
                    "argument_validation": "structural_schema",
                },
            )
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


def validate_tool_arguments(
    parameters: dict[str, Any], arguments: dict[str, Any]
) -> str | None:
    """Validate structural schema constraints before side effects.

    Bounds such as ``minItems`` and ``minLength`` remain tool-owned semantics.
    Controller tools intentionally use those values to produce their own recovery
    and terminal decisions, so the registry must not pre-empt them.
    """

    required = parameters.get("required") or ()
    missing = [str(name) for name in required if name not in arguments]
    if missing:
        return "missing required properties: " + ", ".join(missing)
    properties = parameters.get("properties") or {}
    if parameters.get("additionalProperties") is False:
        extra = sorted(set(arguments) - set(properties))
        if extra:
            return "unexpected properties: " + ", ".join(extra)
    expected_types: dict[str, tuple[type, ...]] = {
        "object": (dict,),
        "array": (list, tuple),
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
    }
    for name, value in arguments.items():
        schema = properties.get(name)
        if not isinstance(schema, dict):
            continue
        expected = schema.get("type")
        accepted = expected_types.get(str(expected))
        if accepted and (
            not isinstance(value, accepted)
            or expected in {"integer", "number"}
            and isinstance(value, bool)
        ):
            return f"{name} must be {expected}"
        allowed = schema.get("enum")
        if isinstance(allowed, list) and value not in allowed:
            return f"{name} must be one of: {', '.join(map(str, allowed))}"
    return None


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
