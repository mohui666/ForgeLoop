from __future__ import annotations

import copy
import json
import ssl
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from forgeloop.agent import AgentLoop, RunMode, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.models import LiteLLMProvider, ProviderRetryPolicy
from forgeloop.runtime import LocalRuntime
from forgeloop.tools import build_default_tools
from forgeloop.trajectory import TrajectoryStore
from forgeloop.types import ModelResponse, ModelUsage, ToolCall
from forgeloop.workspace import Workspace


class FaultProvider:
    model_id = "test/fault-injection"

    def __init__(self, outcomes: list[BaseException | ModelResponse]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[list[dict]] = []

    def complete(self, messages, tools, *, timeout_seconds):
        del tools
        assert timeout_seconds > 0
        self.requests.append(copy.deepcopy(list(messages)))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class StatusError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(RuntimeError):
    pass


def _agent(
    tmp_path: Path,
    provider,
    *,
    retry: ProviderRetryPolicy | None = None,
    sleeps: list[float] | None = None,
    max_output_limit_recoveries: int = 2,
) -> AgentLoop:
    workspace = Workspace(tmp_path)
    recorded_sleeps = sleeps if sleeps is not None else []
    return AgentLoop(
        provider=provider,
        tools=build_default_tools(workspace, LocalRuntime()),
        workspace=workspace,
        trajectory=TrajectoryStore(tmp_path / ".forgeloop" / "runs", run_id="retry"),
        limits=BudgetLimits(
            max_steps=10,
            max_model_calls=10,
            max_tool_calls=10,
            max_seconds=60,
        ),
        provider_reliability=retry or ProviderRetryPolicy(),
        max_output_limit_recoveries=max_output_limit_recoveries,
        retry_sleep=recorded_sleeps.append,
        retry_random=lambda: 0.5,
    )


def _final(content: str = "done") -> ModelResponse:
    return ModelResponse(
        content=content,
        usage=ModelUsage(
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.001,
            cached_tokens=3,
            usage_source="provider_response",
            cost_source="provider_reported",
        ),
        finish_reason="stop",
    )


def _finish_call(call_id: str = "finish") -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                call_id,
                "finish",
                {"status": "completed", "summary": "done", "evidence": "verified"},
            ),
        ),
        usage=ModelUsage(10, 2, 0.001, cached_tokens=3),
        finish_reason="tool_calls",
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_two_ssl_eofs_then_success_retries_same_logical_request(tmp_path: Path) -> None:
    provider = FaultProvider(
        [
            ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
            ssl.SSLEOFError(8, "UNEXPECTED_EOF_WHILE_READING"),
            _final(),
        ]
    )
    sleeps: list[float] = []
    retry = ProviderRetryPolicy(
        max_attempts=4,
        initial_backoff_seconds=0.25,
        max_backoff_seconds=2,
        backoff_multiplier=2,
        jitter_ratio=0.2,
    )

    result = _agent(tmp_path, provider, retry=retry, sleeps=sleeps).run(
        RunMode.TASK, "finish"
    )

    assert result.status is RunStatus.COMPLETED
    assert result.budget["usage"]["model_calls"] == 1
    assert result.budget["usage"]["input_tokens"] == 10
    assert result.budget["usage"]["cost_usd"] == 0.001
    assert sleeps == [0.25, 0.5]
    events = _events(result.trajectory_path)
    assert [event["type"] for event in events].count("model_request") == 1
    assert [event["type"] for event in events].count("model_response") == 1
    assert [event["type"] for event in events].count("provider_attempt_started") == 3
    recovered = next(
        event["payload"]
        for event in events
        if event["type"] == "provider_request_recovered"
    )
    assert recovered["attempt_count"] == 3
    assert recovered["retry_attempts"] == 2
    assert result.provider_reliability["failures_by_reason"] == {"ssl_eof": 2}
    assert all(
        event["payload"]["timeout_seconds"] <= retry.attempt_timeout_seconds
        for event in events
        if event["type"] == "provider_attempt_started"
    )


def test_incomplete_stream_is_discarded_then_complete_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    partial = {
        "model": "mock/model",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "id": "partial-call",
                            "function": {
                                "name": "apply_patch",
                                "arguments": '{"path":"never.py"',
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
        "usage": None,
    }
    provider_usage = SimpleNamespace(
        prompt_tokens=7, completion_tokens=2, total_tokens=9
    )
    terminal = {
        "model": "mock/model",
        "choices": [{"delta": {"content": "complete"}, "finish_reason": "stop"}],
        "usage": provider_usage,
    }
    streams = iter((iter([partial]), iter([terminal])))

    def assembled_response(chunks, messages):
        del chunks, messages
        return SimpleNamespace(
            id="stream-response",
            model="mock/model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="complete", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=999, completion_tokens=999),
            _hidden_params={"custom_llm_provider": "mock"},
        )

    fake_litellm = SimpleNamespace(
        completion=lambda **kwargs: next(streams),
        stream_chunk_builder=assembled_response,
        model_cost={},
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    provider = LiteLLMProvider("mock/model", extra={"stream": True})
    result = _agent(
        tmp_path,
        provider,
        retry=ProviderRetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        ),
    ).run(RunMode.TASK, "stream safely")

    assert result.status is RunStatus.COMPLETED
    assert result.summary == "complete"
    assert result.budget["usage"]["input_tokens"] == 7
    assert result.budget["usage"]["output_tokens"] == 2
    events = _events(result.trajectory_path)
    failures = [
        event["payload"]
        for event in events
        if event["type"] == "provider_attempt_failed"
    ]
    assert failures[0]["retry_reason"] == "incomplete_stream"
    assert failures[0]["usage"] == {"status": "unavailable"}
    assert len([event for event in events if event["type"] == "model_response"]) == 1
    assert not [event for event in events if event["type"] == "tool_call"]
    assert "partial-call" not in result.trajectory_path.read_text(encoding="utf-8")


def test_retryable_5xx_recovers(tmp_path: Path) -> None:
    provider = FaultProvider([StatusError(503, "service unavailable"), _final()])
    result = _agent(
        tmp_path,
        provider,
        retry=ProviderRetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        ),
    ).run(RunMode.TASK, "finish")

    assert result.status is RunStatus.COMPLETED
    events = _events(result.trajectory_path)
    failure = next(
        event["payload"]
        for event in events
        if event["type"] == "provider_attempt_failed"
    )
    assert failure["status_code"] == 503
    assert failure["retry_reason"] == "provider_5xx"
    assert result.provider_reliability["recovered_requests"] == 1


def test_permanent_4xx_fails_immediately_without_retry(tmp_path: Path) -> None:
    provider = FaultProvider([StatusError(400, "invalid parameter"), _final()])
    result = _agent(tmp_path, provider).run(RunMode.TASK, "finish")

    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "provider_failure"
    assert len(provider.requests) == 1
    assert result.budget["usage"]["input_tokens"] is None
    assert result.budget["usage"]["usage_complete"] is False
    assert result.budget["usage"]["unavailable_model_calls"] == 1
    events = _events(result.trajectory_path)
    terminal = next(
        event["payload"]
        for event in events
        if event["type"] == "provider_request_failed"
    )
    assert terminal["retryable"] is False
    assert terminal["recovered"] is False
    assert terminal["exhausted"] is False
    assert not [
        event for event in events if event["type"] == "provider_retry_scheduled"
    ]


def test_known_success_usage_survives_later_unavailable_request(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    read = ModelResponse(
        tool_calls=(ToolCall("read-once", "read_file", {"path": "sample.txt"}),),
        usage=ModelUsage(
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.001,
            cached_tokens=3,
        ),
        finish_reason="tool_calls",
    )
    provider = FaultProvider([read, StatusError(400, "invalid parameter")])

    result = _agent(tmp_path, provider).run(RunMode.TASK, "preserve known usage")

    usage = result.budget["usage"]
    assert result.stop_reason == "provider_failure"
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 2
    assert usage["cached_tokens"] == 3
    assert usage["cost_usd"] == 0.001
    assert usage["usage_complete"] is False
    assert usage["usage_records"] == 1
    assert usage["unavailable_model_calls"] == 1


def test_permanent_sdk_class_wins_over_incidental_number(tmp_path: Path) -> None:
    provider = FaultProvider(
        [
            AuthenticationError("credential rejected for a model with 500 dimensions"),
            _final(),
        ]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "fail fast")

    assert result.stop_reason == "provider_failure"
    assert len(provider.requests) == 1
    assert result.provider_reliability["permanent_failures"] == 1


def test_quota_429_is_not_retried(tmp_path: Path) -> None:
    provider = FaultProvider(
        [StatusError(429, "insufficient_quota: check billing"), _final()]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "fail fast")

    assert result.stop_reason == "provider_failure"
    assert len(provider.requests) == 1
    assert result.provider_reliability["failures_by_reason"] == {"quota_or_billing": 1}


def test_retry_exhaustion_terminates_as_provider_failure(tmp_path: Path) -> None:
    provider = FaultProvider(
        [
            ConnectionResetError("connection reset by peer"),
            ConnectionResetError("connection reset by peer"),
            ConnectionResetError("connection reset by peer"),
        ]
    )
    sleeps: list[float] = []
    result = _agent(
        tmp_path,
        provider,
        retry=ProviderRetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0.1,
            max_backoff_seconds=1,
            jitter_ratio=0,
        ),
        sleeps=sleeps,
    ).run(RunMode.TASK, "finish")

    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "provider_failure"
    assert len(provider.requests) == 3
    assert sleeps == [0.1, 0.2]
    assert result.provider_reliability["exhausted_requests"] == 1
    exhausted = next(
        event["payload"]
        for event in _events(result.trajectory_path)
        if event["type"] == "provider_retry_exhausted"
    )
    assert exhausted["attempt_count"] == 3
    assert exhausted["usage"] == {"status": "unavailable"}


def test_cancellation_during_retry_backoff_preserves_trajectory(tmp_path: Path) -> None:
    provider = FaultProvider(
        [ConnectionResetError("connection reset by peer"), _final()]
    )
    agent = _agent(
        tmp_path,
        provider,
        retry=ProviderRetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=1,
            max_backoff_seconds=1,
        ),
    )
    cancelled = False

    def cancel_after_sleep(seconds: float) -> None:
        nonlocal cancelled
        assert seconds == 1
        cancelled = True

    agent.retry_sleep = cancel_after_sleep
    agent.cancel_check = lambda: cancelled

    result = agent.run(RunMode.TASK, "cancel safely")

    assert result.status is RunStatus.INTERRUPTED
    assert result.stop_reason == "user_interrupt"
    assert len(provider.requests) == 1
    assert result.provider_reliability["failed_attempts"] == 1


def test_completed_tool_call_is_not_reexecuted_by_next_request_retry(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    edit = ModelResponse(
        tool_calls=(
            ToolCall(
                "edit-once",
                "apply_patch",
                {
                    "path": "sample.txt",
                    "old_text": "hello",
                    "new_text": "world",
                },
            ),
        ),
        usage=ModelUsage(10, 2, 0.001, cached_tokens=3),
        finish_reason="tool_calls",
    )
    provider = FaultProvider(
        [edit, ConnectionResetError("connection reset by peer"), _finish_call()]
    )
    result = _agent(
        tmp_path,
        provider,
        retry=ProviderRetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        ),
    ).run(RunMode.TASK, "edit once")

    assert result.status is RunStatus.COMPLETED
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "world\n"
    assert provider.requests[1] == provider.requests[2]
    assistant_tools = [
        message
        for message in provider.requests[2]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    tool_results = [
        message for message in provider.requests[2] if message.get("role") == "tool"
    ]
    assert len(assistant_tools) == 1
    assert len(tool_results) == 1
    events = _events(result.trajectory_path)
    assert [
        event["payload"]["name"] for event in events if event["type"] == "tool_call"
    ] == ["apply_patch", "finish"]
    assert result.budget["usage"]["model_calls"] == 2
    assert result.budget["usage"]["input_tokens"] == 20
    assert result.provider_reliability["attempt_count"] == 3
    assert result.provider_reliability["retry_attempts"] == 1


def test_output_limited_tool_call_is_rejected_without_execution(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    truncated = ModelResponse(
        tool_calls=(
            ToolCall(
                "unsafe-edit",
                "apply_patch",
                {
                    "path": "sample.txt",
                    "old_text": "hello",
                    "new_text": "world",
                },
            ),
        ),
        usage=ModelUsage(10, 2, 0.001),
        finish_reason="length",
    )
    provider = FaultProvider([truncated, _finish_call()])

    result = _agent(tmp_path, provider).run(RunMode.TASK, "do not truncate")

    assert result.status is RunStatus.COMPLETED
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "hello\n"
    assert "provider output or safety limit" in provider.requests[1][-2]["content"]
    assert provider.requests[1][-2]["role"] == "tool"
    assert provider.requests[1][-1]["role"] == "user"
    events = _events(result.trajectory_path)
    limited = next(
        event["payload"] for event in events if event["type"] == "provider_output_limit"
    )
    assert limited["tool_calls_blocked"] == 1
    observation = next(
        event["payload"] for event in events if event["type"] == "observation"
    )
    assert observation["metadata"]["execution_blocked"] is True
    assert observation["metadata"]["reason"] == "provider_output_limit"


def test_schema_invalid_tool_call_becomes_observation_not_terminal(
    tmp_path: Path,
) -> None:
    invalid = ModelResponse(
        tool_calls=(ToolCall("bad-read", "read_file", {}),),
        usage=ModelUsage(10, 2, 0.001),
        finish_reason="tool_calls",
    )
    provider = FaultProvider([invalid, _finish_call()])

    result = _agent(tmp_path, provider).run(RunMode.TASK, "recover from bad args")

    assert result.status is RunStatus.COMPLETED
    assert "missing required properties: path" in provider.requests[1][-1]["content"]
    observations = [
        event["payload"]
        for event in _events(result.trajectory_path)
        if event["type"] == "observation"
    ]
    assert observations[0]["metadata"]["reason"] == "invalid_tool_arguments"


def test_output_limited_final_message_recovers_on_new_logical_call(
    tmp_path: Path,
) -> None:
    provider = FaultProvider(
        [
            ModelResponse(
                content="partial answer",
                usage=ModelUsage(10, 8, 0.001),
                finish_reason="length",
            ),
            _finish_call(),
        ]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "finish safely")

    assert result.status is RunStatus.COMPLETED
    assert result.budget["usage"]["model_calls"] == 2
    assert result.budget["usage"]["input_tokens"] == 20
    recovery_request = provider.requests[1]
    assert recovery_request[-2]["role"] == "assistant"
    assert recovery_request[-2]["content"] == "partial answer"
    assert recovery_request[-1]["role"] == "user"
    assert "exactly one complete next action" in recovery_request[-1]["content"]
    events = _events(result.trajectory_path)
    recovery = next(
        event["payload"]
        for event in events
        if event["type"] == "provider_output_limit_recovery"
    )
    assert recovery["recovery_count"] == 1
    assert recovery["next_action"] == "new_logical_model_call"
    assert recovery["usage_recorded"] is True
    started = next(
        event["payload"] for event in events if event["type"] == "run_started"
    )
    assert started["output_limit_recovery"] == {
        "schema_version": "forgeloop.output-limit-recovery.v1",
        "max_recoveries": 2,
        "scope": "completed_response_with_output_limit",
        "safety_limits_recoverable": False,
        "truncated_tool_calls_executable": False,
    }


def test_reasoning_only_output_limit_recovers_without_duplicate_tool_execution(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("evidence\n", encoding="utf-8")
    reasoning = "unfinished design reasoning" * 100
    provider = FaultProvider(
        [
            ModelResponse(
                content=None,
                usage=ModelUsage(100, 8_192, 0.01, reasoning_tokens=8_192),
                finish_reason="length",
                assistant_message_fields={"reasoning_content": reasoning},
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall("inspect-once", "read_file", {"path": "sample.txt"}),
                ),
                usage=ModelUsage(110, 12, 0.001, reasoning_tokens=0),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "finish",
                        "finish",
                        {
                            "status": "completed",
                            "summary": "done",
                            "evidence": "verified",
                        },
                    ),
                ),
                usage=ModelUsage(10, 2, 0.001, reasoning_tokens=0),
                finish_reason="tool_calls",
            ),
        ]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "inspect then finish")

    assert result.status is RunStatus.COMPLETED
    assert result.budget["usage"]["model_calls"] == 3
    assert result.budget["usage"]["input_tokens"] == 220
    assert result.budget["usage"]["output_tokens"] == 8_206
    assert result.budget["usage"]["reasoning_tokens"] == 8_192
    recovery_request = provider.requests[1]
    assert recovery_request[-2] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
    }
    assert recovery_request[-1]["role"] == "user"
    final_request = provider.requests[2]
    assert [
        message["tool_call_id"]
        for message in final_request
        if message.get("role") == "tool"
    ] == ["inspect-once"]
    executed = [
        event["payload"]["name"]
        for event in _events(result.trajectory_path)
        if event["type"] == "tool_call"
    ]
    assert executed == ["read_file", "finish"]


def test_output_limited_final_message_fails_closed_after_recovery_exhaustion(
    tmp_path: Path,
) -> None:
    provider = FaultProvider(
        [
            ModelResponse(
                content="partial answer",
                usage=ModelUsage(10, 2, 0.001),
                finish_reason="length",
            ),
            ModelResponse(
                content="still partial",
                usage=ModelUsage(20, 3, 0.002),
                finish_reason="length",
            ),
        ]
    )

    result = _agent(tmp_path, provider, max_output_limit_recoveries=1).run(
        RunMode.TASK, "finish safely"
    )

    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "provider_output_limit"
    assert result.budget["usage"]["model_calls"] == 2
    assert result.budget["usage"]["input_tokens"] == 30
    events = _events(result.trajectory_path)
    limited = [
        event["payload"] for event in events if event["type"] == "provider_output_limit"
    ]
    assert [event["action"] for event in limited] == ["recovery", "terminal"]
    assert (
        len(
            [
                event
                for event in events
                if event["type"] == "provider_output_limit_recovery"
            ]
        )
        == 1
    )


def test_safety_limited_response_never_uses_output_recovery(tmp_path: Path) -> None:
    provider = FaultProvider(
        [
            ModelResponse(
                content="blocked",
                usage=ModelUsage(10, 2, 0.001),
                finish_reason="content_filter",
            )
        ]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "finish safely")

    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "provider_safety_limit"
    assert len(provider.requests) == 1
    assert not any(
        event["type"] == "provider_output_limit_recovery"
        for event in _events(result.trajectory_path)
    )
