from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from forgeloop.agent_types import RunMode, RunResult, RunStatus
from forgeloop.budget import BudgetExceeded, BudgetLimits, BudgetState
from forgeloop.models.base import ModelProvider
from forgeloop.prompts import build_system_prompt
from forgeloop.tools.base import ToolRegistry
from forgeloop.trajectory import TrajectoryStore
from forgeloop.types import Message, ModelResponse, ToolCall
from forgeloop.workspace import Workspace

FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "End this run with an explicit terminal status, summary, and verification evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["completed", "blocked", "failed"],
                },
                "summary": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["status", "summary"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class AgentLoop:
    provider: ModelProvider
    tools: ToolRegistry
    workspace: Workspace
    trajectory: TrajectoryStore
    limits: BudgetLimits
    event_sink: Callable[[str, dict[str, Any]], None] | None = None
    cancel_check: Callable[[], bool] | None = None

    def run(
        self,
        mode: RunMode,
        request: str,
        *,
        context_messages: tuple[Message, ...] | list[Message] = (),
        instructions: str = "",
    ) -> RunResult:
        if not request.strip():
            raise ValueError("request cannot be empty")
        budget = BudgetState(self.limits)
        self._session_context = tuple(dict(item) for item in context_messages)
        self._request = request.strip()
        tool_schemas = self.tools.schemas()
        shell_environment = next(
            (
                schema["function"]["description"]
                for schema in tool_schemas
                if schema["function"]["name"] == "shell"
            ),
            "",
        )
        messages: list[Message] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    mode,
                    str(self.workspace.root),
                    instructions,
                    shell_environment,
                ),
            },
            *[dict(item) for item in context_messages],
            {"role": "user", "content": self._request},
        ]
        detector = {
            "calls": Counter(),
            "errors": Counter(),
            "no_progress": 0,
        }
        schemas = [*tool_schemas, FINISH_SCHEMA]
        self.trajectory.append(
            "run_started",
            {
                "mode": mode,
                "request": request.strip(),
                "model": self.provider.model_id,
                "workspace": str(self.workspace.root),
                "git": self.workspace.git_snapshot(),
                "budget": budget.snapshot(),
            },
        )

        while True:
            try:
                if self._cancelled():
                    return self._finish(
                        RunStatus.INTERRUPTED,
                        "Agent 已中断。",
                        "Session 与已完成的修改已保留。",
                        budget,
                        stop_reason="user_interrupt",
                    )
                budget.check_before_step()
                budget.begin_model_call()
                self._emit("model_started", {"step": budget.steps})
                self.trajectory.append(
                    "model_request",
                    {"step": budget.steps, "messages": messages, "tools": schemas},
                )
                response = self.provider.complete(
                    messages,
                    schemas,
                    timeout_seconds=max(0.1, budget.remaining_seconds),
                )
                self._emit(
                    "model_finished",
                    {
                        "step": budget.steps,
                        "tool_calls": len(response.tool_calls),
                        "has_content": bool(response.content),
                    },
                )
                self.trajectory.append(
                    "model_response", self._response_payload(response)
                )
                budget.record_usage(response.usage)
                messages.append(response.as_assistant_message())

                if not response.tool_calls:
                    summary = (
                        response.content or "Model returned no action or final message."
                    ).strip()
                    status = (
                        RunStatus.COMPLETED if response.content else RunStatus.FAILED
                    )
                    return self._finish(
                        status,
                        summary,
                        "Model ended without an explicit finish tool.",
                        budget,
                        stop_reason="model_final_message",
                    )

                budget.reserve_tool_calls(len(response.tool_calls))
                terminal = self._execute_calls(
                    response.tool_calls, messages, budget, detector
                )
                if terminal is not None:
                    return terminal
            except BudgetExceeded as exc:
                return self._finish(
                    RunStatus.BUDGET_EXCEEDED,
                    str(exc),
                    "Budget guard stopped the loop.",
                    budget,
                    stop_reason="budget_guard",
                )
            except Exception as exc:  # noqa: BLE001 - convert boundary failures into terminal results
                self._emit(
                    "error",
                    {
                        "message": str(exc),
                        "details": getattr(
                            exc, "details", f"{type(exc).__name__}: {exc}"
                        ),
                    },
                )
                self.trajectory.append(
                    "run_error", {"type": type(exc).__name__, "message": str(exc)}
                )
                return self._finish(
                    RunStatus.FAILED,
                    f"Agent loop failed: {type(exc).__name__}: {exc}",
                    "Unhandled model or orchestration error.",
                    budget,
                    stop_reason="orchestration_error",
                )

    def _execute_calls(
        self,
        calls: tuple[ToolCall, ...],
        messages: list[Message],
        budget: BudgetState,
        detector: dict[str, Any],
    ) -> RunResult | None:
        if any(call.name == "finish" for call in calls) and len(calls) != 1:
            for call in calls:
                self.trajectory.append("tool_call", call)
                observation = "ERROR\nfinish must be the only tool call in a response."
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": observation}
                )
                self.trajectory.append(
                    "observation", {"tool_call_id": call.id, "output": observation}
                )
            budget.tool_calls += len(calls)
            return None

        for call in calls:
            if self._cancelled():
                return self._finish(
                    RunStatus.INTERRUPTED,
                    "Agent 已中断。",
                    "Session 与已完成的修改已保留。",
                    budget,
                    stop_reason="user_interrupt",
                )
            budget.check_time()
            budget.tool_calls += 1
            self.trajectory.append("tool_call", call)
            if call.name == "finish":
                return self._finish_from_call(call, budget)
            self._emit(
                "tool_started",
                {"id": call.id, "name": call.name, "arguments": call.arguments},
            )
            signature = json.dumps(
                {"name": call.name, "arguments": call.arguments},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            detector["calls"][signature] += 1
            if detector["calls"][signature] > self.limits.max_repeated_tool_calls:
                return self._finish(
                    RunStatus.BLOCKED,
                    "Stopped after repeated identical tool calls.",
                    f"Tool {call.name} repeated without a different action.",
                    budget,
                    stop_reason="repeated_tool_call",
                )
            before_status = self.workspace.git_progress_fingerprint()
            result = self.tools.execute(
                call.name,
                call.arguments,
                timeout_seconds=max(0.1, budget.remaining_seconds),
            )
            observation = result.as_observation()
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
            )
            self.trajectory.append(
                "observation",
                {
                    "tool_call_id": call.id,
                    "tool": call.name,
                    "ok": result.ok,
                    "output": result.output,
                    "metadata": result.metadata,
                },
            )
            self._emit(
                "tool_finished",
                {
                    "id": call.id,
                    "name": call.name,
                    "ok": result.ok,
                    "output": result.output,
                    "metadata": result.metadata or {},
                },
            )
            if not result.ok:
                error_key = f"{call.name}:{result.output}"
                detector["errors"][error_key] += 1
                if detector["errors"][error_key] >= self.limits.max_repeated_errors:
                    return self._finish(
                        RunStatus.BLOCKED,
                        "Stopped after the same tool error repeated.",
                        result.output,
                        budget,
                        stop_reason="repeated_error",
                    )
            if call.name in {"apply_patch", "shell"}:
                after_status = self.workspace.git_progress_fingerprint()
                detector["no_progress"] = (
                    detector["no_progress"] + 1 if before_status == after_status else 0
                )
                if detector["no_progress"] >= self.limits.max_no_progress_steps:
                    return self._finish(
                        RunStatus.BLOCKED,
                        "Stopped because mutation steps made no observable Git progress.",
                        "Review the last tool observations before retrying.",
                        budget,
                        stop_reason="no_progress",
                    )
        return None

    def _finish_from_call(self, call: ToolCall, budget: BudgetState) -> RunResult:
        raw_status = str(call.arguments.get("status", "failed"))
        try:
            status = RunStatus(raw_status)
        except ValueError:
            status = RunStatus.FAILED
        summary = str(
            call.arguments.get("summary") or "Model ended the run without a summary."
        )
        evidence = str(call.arguments.get("evidence") or "")
        return self._finish(
            status, summary, evidence, budget, stop_reason="model_finish_tool"
        )

    def _finish(
        self,
        status: RunStatus,
        summary: str,
        evidence: str,
        budget: BudgetState,
        *,
        stop_reason: str,
    ) -> RunResult:
        summary = self.trajectory.redact_text(summary)
        evidence = self.trajectory.redact_text(evidence)
        snapshot = budget.snapshot()
        provider = (
            min(budget.providers)
            if budget.providers
            else self.provider.model_id.partition("/")[0]
        )
        self.trajectory.append(
            "run_finished",
            {
                "status": status,
                "summary": summary,
                "evidence": evidence,
                "stop_reason": stop_reason,
                "git": self.workspace.git_snapshot(),
                "budget": snapshot,
            },
        )
        self._emit(
            "run_finished",
            {
                "status": status.value,
                "summary": summary,
                "evidence": evidence,
                "stop_reason": stop_reason,
                "budget": snapshot,
            },
        )
        return RunResult(
            status,
            summary,
            evidence,
            self.trajectory.path,
            snapshot,
            stop_reason=stop_reason,
            model=self.provider.model_id,
            provider=provider,
            conversation=(
                *getattr(self, "_session_context", ()),
                {"role": "user", "content": getattr(self, "_request", "")},
                {"role": "assistant", "content": summary},
            ),
        )

    @staticmethod
    def _response_payload(response: ModelResponse) -> dict[str, Any]:
        return {
            "content": response.content,
            "tool_calls": response.tool_calls,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "provider_metadata": response.provider_metadata,
            "assistant_message_fields": response.assistant_message_fields,
        }

    def _cancelled(self) -> bool:
        return bool(self.cancel_check and self.cancel_check())

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(event_type, payload)
        except Exception:  # noqa: BLE001 - presentation observers cannot break the loop
            return


__all__ = ["AgentLoop", "RunMode", "RunResult", "RunStatus"]
