from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from forgeloop.agent_types import RunMode, RunResult, RunStatus
from forgeloop.budget import BudgetExceeded, BudgetLimits, BudgetState
from forgeloop.controller import ControllerRecovery, ControllerTerminal, ControllerV1
from forgeloop.context import (
    AgentMessageHistory,
    agent_compaction_threshold,
    prepare_agent_context,
)
from forgeloop.delivery import RunDelivery
from forgeloop.effects import EffectContext, EffectRecorder
from forgeloop.guards import RepeatedActionDetector, guard_semantics
from forgeloop.models.base import ModelProvider, ModelProviderError
from forgeloop.models.reliability import (
    ProviderRetryPolicy,
    classify_provider_error,
    normalize_provider_error,
)
from forgeloop.prompts import build_system_prompt
from forgeloop.policy import provider_policy_identity
from forgeloop.tools.base import ToolRegistry, validate_tool_arguments
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

_OUTPUT_LIMIT_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}
_SAFETY_LIMIT_FINISH_REASONS = {"content_filter", "safety", "blocked"}


class _AgentCancelled(Exception):
    pass


@dataclass
class AgentLoop:
    provider: ModelProvider
    tools: ToolRegistry
    workspace: Workspace
    trajectory: TrajectoryStore
    limits: BudgetLimits
    event_sink: Callable[[str, dict[str, Any]], None] | None = None
    cancel_check: Callable[[], bool] | None = None
    controller: ControllerV1 | None = None
    delivery: RunDelivery | None = None
    provider_reliability: ProviderRetryPolicy | None = None
    retry_sleep: Callable[[float], None] = time.sleep
    retry_random: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        self._provider_retry_policy = (
            self.provider_reliability
            or ProviderRetryPolicy.from_provider(self.provider)
        )
        initial = self.workspace.git_snapshot()
        self._base_head = initial.head if initial.is_repository else None
        if self.controller:
            self.controller.start(self.workspace)
            for tool in self.controller.additional_tools(self.workspace):
                self.tools.register(tool)
        if self.delivery:
            self.delivery.start(self.workspace)
        self.tools.bind_effect_recorder(
            EffectRecorder(self.trajectory, self.workspace.root)
        )

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
        self._reset_provider_reliability_stats()
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
        messages = AgentMessageHistory(
            [
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
        )
        base_message_count = len(messages)
        compact_threshold_tokens = agent_compaction_threshold(self.provider)
        detector = {
            "repeated_actions": RepeatedActionDetector(),
            "last_error": "",
            "error_streak": 0,
            "no_progress": 0,
        }
        schemas = [*tool_schemas, FINISH_SCHEMA]
        self.trajectory.append(
            "run_started",
            {
                "mode": mode,
                "request": request.strip(),
                "model": self.provider.model_id,
                "policy_identity": provider_policy_identity(self.provider),
                "workspace": str(self.workspace.root),
                "git": self.workspace.git_snapshot(),
                "budget": budget.snapshot(),
                "loop_guards": guard_semantics(
                    max_repeated_tool_calls=self.limits.max_repeated_tool_calls,
                    max_repeated_errors=self.limits.max_repeated_errors,
                    max_no_progress_steps=self.limits.max_no_progress_steps,
                ),
                "provider_reliability": self._provider_retry_policy.to_dict(),
                "context_management": {
                    "schema_version": "forgeloop.context.pi-parity.v1",
                    "strategy": "stable_compaction_epoch",
                    "compact_threshold_tokens": compact_threshold_tokens,
                    "canonical_history": "append_only",
                },
                "controller": self.controller.identity if self.controller else None,
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
                if self.controller:
                    decision = self.controller.before_model_call(
                        current_fingerprint=self._fingerprint(),
                        budget_snapshot=budget.snapshot(),
                    )
                    self._record_controller_events(budget)
                    if isinstance(decision, ControllerRecovery):
                        self._apply_controller_recoveries((decision,), messages, budget)
                    elif isinstance(decision, ControllerTerminal):
                        return self._finish_controller_terminal(decision, budget)
                budget.check_before_step()
                budget.begin_model_call()
                available_schemas = (
                    self.controller.filter_tool_schemas(schemas)
                    if self.controller
                    else schemas
                )
                request_messages, context_report = prepare_agent_context(
                    messages,
                    available_schemas,
                    base_message_count=base_message_count,
                    compact_threshold_tokens=compact_threshold_tokens,
                    redactor=self.trajectory.redactor,
                )
                if context_report["applied"]:
                    messages.commit_compaction(request_messages)
                    # The committed ledger becomes part of the stable prefix for
                    # the next epoch, matching Pi's summary + kept-messages shape.
                    base_message_count += 1
                context_report.update(
                    {
                        "compaction_committed": context_report["applied"],
                        "compaction_epoch": messages.compaction_epochs,
                        "canonical_messages": len(messages.canonical),
                        "compact_threshold_tokens": compact_threshold_tokens,
                    }
                )
                context_report = {"step": budget.steps, **context_report}
                self._emit(
                    "model_started",
                    {
                        "step": budget.steps,
                        "estimated_input_tokens": context_report[
                            "estimated_input_tokens"
                        ],
                        "context_compacted": context_report["applied"],
                    },
                )
                self.trajectory.append(
                    "model_request",
                    {
                        "step": budget.steps,
                        "messages": request_messages,
                        "tools": available_schemas,
                        "context": context_report,
                    },
                )
                response = self._complete_current_request(
                    request_messages, available_schemas, budget
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
                context_usage = {
                    **context_report,
                    "input_tokens": response.usage.input_tokens,
                    "cached_tokens": response.usage.cached_tokens,
                }
                self.trajectory.append("context_usage", context_usage)
                budget.record_usage(response.usage)
                messages.append(response.as_assistant_message())

                response_limit = self._response_limit(response)
                if response_limit:
                    self.trajectory.append(
                        response_limit,
                        {
                            "model_call": budget.model_calls,
                            "finish_reason": response.finish_reason,
                            "tool_calls_blocked": len(response.tool_calls),
                            "action": (
                                "tool_calls_rejected"
                                if response.tool_calls
                                else "terminal"
                            ),
                        },
                    )
                    if response.tool_calls:
                        budget.reserve_tool_calls(len(response.tool_calls))
                        self._reject_truncated_tool_calls(
                            response.tool_calls,
                            messages,
                            budget,
                            reason=response_limit,
                        )
                        continue
                    return self._finish(
                        RunStatus.FAILED,
                        "Model response reached a provider output or safety limit.",
                        "No limited response was accepted as task completion.",
                        budget,
                        stop_reason=response_limit,
                    )

                if not response.tool_calls:
                    summary = (
                        response.content or "Model returned no action or final message."
                    ).strip()
                    if self.controller:
                        decision = self.controller.review_final(
                            response.content,
                            current_fingerprint=self._fingerprint(),
                        )
                        if isinstance(decision, ControllerRecovery):
                            self._apply_controller_recoveries(
                                (decision,), messages, budget
                            )
                            continue
                        return self._finish_controller_terminal(decision, budget)
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

                if not (
                    len(response.tool_calls) == 1
                    and response.tool_calls[0].name == "finish"
                ):
                    budget.reserve_tool_calls(len(response.tool_calls))
                terminal = self._execute_calls(
                    response.tool_calls, messages, budget, detector
                )
                if terminal is not None:
                    return terminal
            except _AgentCancelled:
                return self._finish(
                    RunStatus.INTERRUPTED,
                    "Agent 已中断。",
                    "Session 与已完成的修改已保留。",
                    budget,
                    stop_reason="user_interrupt",
                )
            except BudgetExceeded as exc:
                stop_reason = (
                    "timeout_guard" if "time budget" in str(exc) else "budget_guard"
                )
                return self._finish(
                    RunStatus.BUDGET_EXCEEDED,
                    str(exc),
                    "Budget guard stopped the loop.",
                    budget,
                    stop_reason=stop_reason,
                )
            except ModelProviderError as exc:
                classification = classify_provider_error(exc)
                self.trajectory.append(
                    "provider_error",
                    {
                        "type": classification.error_type,
                        "message": str(exc),
                        "details": exc.details,
                        "status_code": classification.status_code,
                        "retryable": classification.retryable,
                        "retry_reason": classification.retry_reason,
                        "provider_reliability": self.provider_reliability_summary,
                    },
                )
                return self._finish(
                    RunStatus.FAILED,
                    str(exc),
                    exc.details,
                    budget,
                    stop_reason="provider_failure",
                )
            except TimeoutError as exc:
                self.trajectory.append(
                    "provider_error", {"type": type(exc).__name__, "message": str(exc)}
                )
                return self._finish(
                    RunStatus.FAILED,
                    f"Provider request timed out: {exc}",
                    "The provider call exceeded its timeout; no retry was attempted.",
                    budget,
                    stop_reason="provider_timeout",
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

    def _complete_current_request(
        self,
        messages: list[Message],
        tools: list[dict],
        budget: BudgetState,
    ) -> ModelResponse:
        policy = self._provider_retry_policy
        logical_call = budget.model_calls
        for attempt in range(1, policy.max_attempts + 1):
            budget.check_time()
            self._provider_reliability_stats["attempt_count"] += 1
            if attempt > 1:
                self._provider_reliability_stats["retry_attempts"] += 1
            timeout_seconds = max(
                0.1,
                min(policy.attempt_timeout_seconds, budget.remaining_seconds),
            )
            attempt_payload = {
                "model_call": logical_call,
                "attempt_count": attempt,
                "max_attempts": policy.max_attempts,
                "timeout_seconds": round(timeout_seconds, 3),
            }
            self.trajectory.append("provider_attempt_started", attempt_payload)
            self._emit("provider_attempt_started", attempt_payload)
            try:
                response = self.provider.complete(
                    messages,
                    tools,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as raw_error:  # noqa: BLE001 - provider boundary
                error = normalize_provider_error(raw_error)
                classification = classify_provider_error(error)
                self._provider_reliability_stats["failed_attempts"] += 1
                reasons = self._provider_reliability_stats["failures_by_reason"]
                reasons[classification.retry_reason] = (
                    reasons.get(classification.retry_reason, 0) + 1
                )
                failure_payload = {
                    "model_call": logical_call,
                    "attempt_count": attempt,
                    "max_attempts": policy.max_attempts,
                    "error_type": classification.error_type,
                    "message": str(error),
                    "details": error.details,
                    "status_code": classification.status_code,
                    "retryable": classification.retryable,
                    "retry_reason": classification.retry_reason,
                    "usage": {"status": "unavailable"},
                }
                self.trajectory.append("provider_attempt_failed", failure_payload)
                self._emit("provider_attempt_failed", failure_payload)

                if not classification.retryable:
                    self._provider_reliability_stats["permanent_failures"] += 1
                    self.trajectory.append(
                        "provider_request_failed",
                        {
                            **failure_payload,
                            "outcome": "permanent_failure",
                            "recovered": False,
                            "exhausted": False,
                        },
                    )
                    raise error
                if attempt >= policy.max_attempts:
                    self._provider_reliability_stats["exhausted_requests"] += 1
                    self.trajectory.append(
                        "provider_retry_exhausted",
                        {
                            **failure_payload,
                            "outcome": "exhausted",
                            "recovered": False,
                            "exhausted": True,
                        },
                    )
                    raise error

                random_value = max(0.0, min(1.0, float(self.retry_random())))
                planned_backoff = policy.backoff_seconds(
                    attempt, random_value=random_value
                )
                backoff = min(planned_backoff, budget.remaining_seconds)
                retry_payload = {
                    **failure_payload,
                    "next_attempt": attempt + 1,
                    "backoff_seconds": round(backoff, 6),
                    "planned_backoff_seconds": planned_backoff,
                    "outcome": "scheduled",
                }
                self.trajectory.append("provider_retry_scheduled", retry_payload)
                self._emit("provider_retry_scheduled", retry_payload)
                if backoff > 0:
                    self._wait_before_retry(backoff, budget)
                budget.check_time()
                continue

            success_payload = {
                **attempt_payload,
                "outcome": "success",
                "usage_recorded_in": "model_response",
            }
            self.trajectory.append("provider_attempt_succeeded", success_payload)
            if attempt > 1:
                self._provider_reliability_stats["recovered_requests"] += 1
                recovered_payload = {
                    **success_payload,
                    "retry_attempts": attempt - 1,
                    "recovered": True,
                    "exhausted": False,
                }
                self.trajectory.append("provider_request_recovered", recovered_payload)
                self._emit("provider_request_recovered", recovered_payload)
            return response
        raise AssertionError("bounded provider retry loop did not return or raise")

    def _wait_before_retry(self, seconds: float, budget: BudgetState) -> None:
        """Make production backoff interruptible without changing test clocks."""

        if self.retry_sleep is not time.sleep:
            self.retry_sleep(seconds)
            if self._cancelled():
                raise _AgentCancelled
            return
        deadline = time.perf_counter() + seconds
        while True:
            if self._cancelled():
                raise _AgentCancelled
            budget.check_time()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    @staticmethod
    def _response_limit(response: ModelResponse) -> str | None:
        finish_reason = str(response.finish_reason or "").lower()
        if finish_reason in _OUTPUT_LIMIT_FINISH_REASONS:
            return "provider_output_limit"
        if finish_reason in _SAFETY_LIMIT_FINISH_REASONS:
            return "provider_safety_limit"
        return None

    def _reject_truncated_tool_calls(
        self,
        calls: tuple[ToolCall, ...],
        messages: list[Message],
        budget: BudgetState,
        *,
        reason: str,
    ) -> None:
        for call in calls:
            budget.tool_calls += 1
            self.trajectory.append("tool_call", call)
            output = (
                f'Tool call "{call.name}" was not executed because the model '
                "response reached a provider output or safety limit; its arguments "
                "may be incomplete. Re-issue the complete tool call."
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"ERROR\n{output}",
                }
            )
            metadata = {
                "execution_blocked": True,
                "reason": reason,
            }
            self.trajectory.append(
                "observation",
                {
                    "tool_call_id": call.id,
                    "tool": call.name,
                    "ok": False,
                    "output": output,
                    "metadata": metadata,
                },
            )
            self._emit(
                "tool_finished",
                {
                    "id": call.id,
                    "name": call.name,
                    "ok": False,
                    "output": output,
                    "metadata": metadata,
                },
            )

    def _reset_provider_reliability_stats(self) -> None:
        self._provider_reliability_stats: dict[str, Any] = {
            "attempt_count": 0,
            "retry_attempts": 0,
            "failed_attempts": 0,
            "recovered_requests": 0,
            "exhausted_requests": 0,
            "permanent_failures": 0,
            "failures_by_reason": {},
        }

    @property
    def provider_reliability_summary(self) -> dict[str, Any]:
        stats = getattr(self, "_provider_reliability_stats", {})
        return {
            "config": self._provider_retry_policy.to_dict(),
            **stats,
            "failures_by_reason": dict(stats.get("failures_by_reason", {})),
        }

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

        recoveries: list[ControllerRecovery] = []
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
            argument_error = call.argument_error
            if argument_error is None and call.name == "finish":
                argument_error = validate_tool_arguments(
                    FINISH_SCHEMA["function"]["parameters"], call.arguments
                )
            if argument_error:
                output = f"Invalid arguments for {call.name}: {argument_error}"
                metadata = {
                    "execution_blocked": True,
                    "reason": "invalid_tool_arguments",
                }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"ERROR\n{output}",
                    }
                )
                self.trajectory.append(
                    "observation",
                    {
                        "tool_call_id": call.id,
                        "tool": call.name,
                        "ok": False,
                        "output": output,
                        "metadata": metadata,
                    },
                )
                self._emit(
                    "tool_finished",
                    {
                        "id": call.id,
                        "name": call.name,
                        "ok": False,
                        "output": output,
                        "metadata": metadata,
                    },
                )
                continue
            if call.name == "finish":
                if self.controller:
                    decision = self.controller.review_finish(
                        call,
                        current_fingerprint=self._fingerprint(),
                    )
                    if isinstance(decision, ControllerRecovery):
                        observation = f"ERROR\n{decision.feedback}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": observation,
                            }
                        )
                        self.trajectory.append(
                            "observation",
                            {
                                "tool_call_id": call.id,
                                "tool": call.name,
                                "ok": False,
                                "output": decision.feedback,
                                "metadata": {"controller": self.controller.identity},
                            },
                        )
                        self._apply_controller_recoveries((decision,), messages, budget)
                        return None
                    if isinstance(decision, ControllerTerminal):
                        return self._finish_controller_terminal(decision, budget)
                return self._finish_from_call(call, budget)
            if self.controller:
                guard = self.controller.guard_action(
                    call,
                    current_fingerprint=self._fingerprint(),
                )
                self._record_controller_events(budget)
                if guard:
                    observation = f"ERROR\n{guard.feedback}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": observation,
                        }
                    )
                    self.trajectory.append(
                        "observation",
                        {
                            "tool_call_id": call.id,
                            "tool": call.name,
                            "ok": False,
                            "output": guard.feedback,
                            "metadata": {
                                "controller": self.controller.identity,
                                "execution_blocked": True,
                            },
                        },
                    )
                    self._emit(
                        "tool_finished",
                        {
                            "id": call.id,
                            "name": call.name,
                            "ok": False,
                            "output": guard.feedback,
                            "metadata": {
                                "controller": self.controller.identity,
                                "execution_blocked": True,
                            },
                        },
                    )
                    terminal = self.controller.post_tool_terminal()
                    if terminal:
                        recoveries.append(guard)
                        self._apply_controller_recoveries(recoveries, messages, budget)
                        return self._finish_controller_terminal(terminal, budget)
                    recoveries.append(guard)
                    continue
            self._emit(
                "tool_started",
                {"id": call.id, "name": call.name, "arguments": call.arguments},
            )
            before_status = self._fingerprint()
            result = self.tools.execute(
                call.name,
                call.arguments,
                timeout_seconds=max(0.1, budget.remaining_seconds),
                effect_context=EffectContext(
                    step=budget.steps,
                    tool_call_id=call.id,
                ),
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
            after_status = self._fingerprint()
            repeated = detector["repeated_actions"].observe(
                call,
                result,
                before_fingerprint=before_status,
                after_fingerprint=after_status,
            )
            if repeated.streak == 2:
                self._record_repeated_action_guard(
                    call,
                    repeated,
                    budget,
                    outcome="warning",
                )
                if not self.controller:
                    recoveries.append(
                        ControllerRecovery(
                            "repeated_action",
                            f"{call.name} repeated with identical evidence",
                            "Repeated-action warning: the immediately preceding "
                            "identical tool call produced the same observation and no "
                            "workspace change. Use the evidence or choose a materially "
                            "different action.",
                        )
                    )

            error_key = result.output if not result.ok else ""
            if error_key:
                detector["error_streak"] = (
                    detector["error_streak"] + 1
                    if error_key == detector["last_error"]
                    and before_status == after_status
                    else 1
                )
                detector["last_error"] = error_key
                if (
                    detector["error_streak"] == self.limits.max_repeated_errors
                    and not self.controller
                ):
                    recoveries.append(
                        ControllerRecovery(
                            "repeated_error_warning",
                            "consecutive identical tool errors",
                            "The same tool error has repeated without a workspace "
                            "change. Reinspect the current state or change the action; "
                            "this warning does not terminate execution.",
                        )
                    )
            else:
                detector["last_error"] = ""
                detector["error_streak"] = 0

            if call.name == "apply_patch" and before_status == after_status:
                detector["no_progress"] += 1
                if detector["no_progress"] == self.limits.max_no_progress_steps:
                    recoveries.append(
                        ControllerRecovery(
                            "no_progress_warning",
                            "consecutive apply_patch actions left the workspace unchanged",
                            "Mutation warning: recent apply_patch actions made no "
                            "Git-visible change. Re-read the target and use the latest "
                            "observation before patching again. Execution remains open.",
                        )
                    )
            else:
                detector["no_progress"] = 0
            if self.controller:
                recoveries.extend(
                    self.controller.observe_tool(
                        call,
                        result,
                        before_fingerprint=before_status,
                        after_fingerprint=after_status,
                        budget_snapshot=budget.snapshot(),
                    )
                )
                self._record_controller_events(budget)
                terminal = self.controller.post_tool_terminal()
                if terminal:
                    self._apply_controller_recoveries(recoveries, messages, budget)
                    return self._finish_controller_terminal(terminal, budget)
            if repeated.streak > self.limits.max_repeated_tool_calls:
                self._record_repeated_action_guard(
                    call,
                    repeated,
                    budget,
                    outcome="terminal",
                )
                self._apply_controller_recoveries(recoveries, messages, budget)
                return self._finish(
                    RunStatus.BLOCKED,
                    "Stopped after a proven no-change tool loop.",
                    (
                        f"Tool {call.name} produced the same observation with an "
                        f"unchanged workspace for {repeated.streak} consecutive calls."
                    ),
                    budget,
                    stop_reason="repeated_tool_call",
                )
        if self.controller:
            self.controller.end_tool_batch()
            self._record_controller_events(budget)
        self._apply_controller_recoveries(recoveries, messages, budget)
        return None

    def _record_repeated_action_guard(
        self,
        call: ToolCall,
        evidence: Any,
        budget: BudgetState,
        *,
        outcome: str,
    ) -> None:
        payload = {
            "step": budget.steps,
            "guard": "repeated_tool_call",
            "outcome": outcome,
            "tool": call.name,
            "streak": evidence.streak,
            "configured_repeat_limit": self.limits.max_repeated_tool_calls,
            "hard_stop_streak": self.limits.max_repeated_tool_calls + 1,
            "window": "contiguous",
            "same_observation": True,
            "workspace_unchanged": True,
            "action_fingerprint": evidence.action_fingerprint,
            "observation_fingerprint": evidence.observation_fingerprint,
        }
        self.trajectory.append("repeated_action_guard", payload)
        self._emit("repeated_action_guard", payload)

    def _record_controller_events(self, budget: BudgetState) -> None:
        if not self.controller:
            return
        for event_type, event_payload in self.controller.drain_events():
            payload = {"step": budget.steps, **event_payload}
            self.trajectory.append(event_type, payload)
            self._emit(event_type, payload)

    def _apply_controller_recoveries(
        self,
        recoveries: list[ControllerRecovery] | tuple[ControllerRecovery, ...],
        messages: list[Message],
        budget: BudgetState,
    ) -> None:
        if not recoveries:
            return
        feedback = "\n\n".join(recovery.feedback for recovery in recoveries)
        messages.append({"role": "user", "content": feedback})
        for recovery in recoveries:
            payload = {
                "controller": self.controller.identity if self.controller else None,
                "step": budget.steps,
                "strategy": recovery.strategy,
                "trigger": recovery.trigger,
                "feedback": recovery.feedback,
            }
            self.trajectory.append("controller_recovery", payload)
            self._emit("controller_recovery", payload)

    def _finish_controller_terminal(
        self, decision: ControllerTerminal, budget: BudgetState
    ) -> RunResult:
        return self._finish(
            decision.status,
            decision.summary,
            decision.evidence,
            budget,
            stop_reason=decision.stop_reason,
        )

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
        delivery_payload = None
        if self.delivery:
            delivery_result = self.delivery.deliver(self.workspace, status)
            delivery_payload = delivery_result.to_dict()
            self.trajectory.append("patch_delivery", delivery_payload)
            if status is RunStatus.COMPLETED and not delivery_result.ok:
                status = RunStatus.FAILED
                stop_reason = "patch_delivery_failure"
                summary = "Validated work could not be delivered as a collector patch."
                evidence = delivery_result.detail or "Patch delivery failed."
            elif delivery_result.detail:
                evidence = "\n".join(filter(None, (evidence, delivery_result.detail)))
        summary = self.trajectory.redact_text(summary)
        evidence = self.trajectory.redact_text(evidence)
        snapshot = budget.snapshot()
        if self.controller:
            self.controller.finalize_budget(snapshot)
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
                "provider_reliability": self.provider_reliability_summary,
                "controller": self.controller.summary() if self.controller else None,
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
            delivery=delivery_payload,
            provider_reliability=self.provider_reliability_summary,
            conversation=(
                *getattr(self, "_session_context", ()),
                {"role": "user", "content": getattr(self, "_request", "")},
                {"role": "assistant", "content": summary},
            ),
        )

    def _fingerprint(self) -> str:
        return self.workspace.git_progress_fingerprint(base_head=self._base_head)

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
