from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from forgeloop.agent import AgentLoop, RunMode, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.runtime import DockerRuntime, LocalRuntime
from forgeloop.tools import build_default_tools
from forgeloop.trajectory import TrajectoryStore
from forgeloop.types import Message, ModelResponse, ModelUsage, ToolCall
from forgeloop.workspace import Workspace


class ScriptedProvider:
    model_id = "test/scripted"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = iter(responses)
        self.requests: list[Sequence[Message]] = []

    def complete(self, messages, tools, *, timeout_seconds):
        assert timeout_seconds > 0
        assert any(tool["function"]["name"] == "finish" for tool in tools)
        self.requests.append(list(messages))
        return next(self.responses)


def call(call_id: str, name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(call_id, name, arguments),),
        usage=ModelUsage(10, 5, 0.001),
        finish_reason="tool_calls",
    )


def make_agent(
    tmp_path: Path, provider: ScriptedProvider, **limit_overrides
) -> AgentLoop:
    workspace = Workspace(tmp_path)
    limits = {
        "max_steps": 10,
        "max_model_calls": 10,
        "max_tool_calls": 10,
        "max_seconds": 60,
    }
    limits.update(limit_overrides)
    return AgentLoop(
        provider,
        build_default_tools(workspace, LocalRuntime()),
        workspace,
        TrajectoryStore(tmp_path / ".forgeloop" / "runs", run_id="test-run"),
        BudgetLimits(**limits),
    )


def test_loop_executes_actions_and_records_trajectory(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            call("1", "read_file", path="sample.txt"),
            call(
                "2",
                "apply_patch",
                path="sample.txt",
                old_text="hello",
                new_text="world",
            ),
            call("3", "read_file", path="sample.txt"),
            call(
                "4",
                "finish",
                status="completed",
                summary="Updated sample",
                evidence="Read-back showed world",
            ),
        ]
    )
    agent = make_agent(tmp_path, provider)

    result = agent.run(RunMode.TASK, "Change hello to world")

    assert result.status is RunStatus.COMPLETED
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "world\n"
    assert result.budget["usage"]["model_calls"] == 4
    events = [
        json.loads(line)
        for line in result.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["type"] == "run_started"
    assert (
        events[0]["payload"]["loop_guards"]["repeated_action"]["hard_stop_streak"] == 4
    )
    assert events[-1]["type"] == "run_finished"
    assert [event["type"] for event in events].count("observation") == 3
    context_events = [event for event in events if event["type"] == "context_usage"]
    assert len(context_events) == 4
    assert context_events[0]["payload"]["input_tokens"] == 10
    assert context_events[0]["payload"]["estimated_input_tokens"] > 0
    assert context_events[0]["payload"]["dominant_sources"]


def test_step_budget_stops_before_an_extra_model_call(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider([call("1", "read_file", path="sample.txt")])
    agent = make_agent(tmp_path, provider, max_steps=1, max_model_calls=5)

    result = agent.run(RunMode.GOAL, "Inspect forever")

    assert result.status is RunStatus.BUDGET_EXCEEDED
    assert "step budget" in result.summary
    assert len(provider.requests) == 1


def test_cumulative_tokens_are_telemetry_not_an_execution_stop(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("hello", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("1", "read_file", {"path": "sample.txt"}),),
                usage=ModelUsage(250_000, 5, 0.001, cached_tokens=200_000),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "2",
                        "finish",
                        {
                            "status": "completed",
                            "summary": "Done",
                            "evidence": "Read completed",
                        },
                    ),
                ),
                usage=ModelUsage(10, 5, 0.001, cached_tokens=0),
                finish_reason="tool_calls",
            ),
        ]
    )
    result = make_agent(tmp_path, provider).run(RunMode.TASK, "Inspect and finish")

    assert result.status is RunStatus.COMPLETED
    assert result.stop_reason == "model_finish_tool"
    assert result.budget["usage"]["input_tokens"] == 250_010
    assert result.budget["usage"]["cached_tokens"] == 200_000
    assert result.budget["usage"]["output_tokens"] == 10
    assert result.budget["usage"]["total_tokens"] == 250_020


def test_plain_final_response_is_supported(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [ModelResponse(content="Nothing needed.", usage=ModelUsage(2, 2))]
    )
    result = make_agent(tmp_path, provider).run(RunMode.TASK, "Check")
    assert result.status is RunStatus.COMPLETED
    assert result.summary == "Nothing needed."


def test_system_prompt_uses_runtime_shell_environment(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [ModelResponse(content="Nothing needed.", usage=ModelUsage(2, 2))]
    )
    workspace = Workspace(tmp_path)
    agent = AgentLoop(
        provider,
        build_default_tools(workspace, DockerRuntime()),
        workspace,
        TrajectoryStore(tmp_path / ".forgeloop" / "runs", run_id="docker-prompt"),
        BudgetLimits(max_seconds=60),
    )

    agent.run(RunMode.TASK, "Check")

    system_prompt = provider.requests[0][0]["content"]
    assert "POSIX shell" in system_prompt
    assert "/bin/sh" in system_prompt
    assert "PowerShell commands" in system_prompt
    assert "unavailable" in system_prompt


def test_editing_mode_system_prompt_uses_execution_first_policy(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [ModelResponse(content="Nothing needed.", usage=ModelUsage(2, 2))]
    )

    make_agent(tmp_path, provider).run(RunMode.TASK, "Fix the focused behavior")

    system_prompt = provider.requests[0][0]["content"]
    assert "Execution-first coding policy (v1)" in system_prompt
    assert "inspect -> hypothesis -> minimal edit -> validate" in system_prompt
    assert "Do not wait to understand the whole repository" in system_prompt
    assert "Treat focused validation as an information-gathering experiment" in (
        system_prompt
    )
    assert "rereading unchanged content merely to feel more confident" in system_prompt


def test_plan_mode_does_not_receive_editing_policy(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [ModelResponse(content="Plan only.", usage=ModelUsage(2, 2))]
    )

    make_agent(tmp_path, provider).run(RunMode.PLAN, "Plan the change")

    system_prompt = provider.requests[0][0]["content"]
    assert "Execution-first coding policy" not in system_prompt
    assert "Plan Mode is strictly read-only" in system_prompt


def test_failed_model_call_marks_usage_unknown(tmp_path: Path) -> None:
    class FailingProvider:
        model_id = "test/failing"

        def complete(self, messages, tools, *, timeout_seconds):
            del messages, tools, timeout_seconds
            raise RuntimeError("provider unavailable")

    result = make_agent(tmp_path, FailingProvider()).run(RunMode.TASK, "Check")
    usage = result.budget["usage"]
    assert usage["model_calls"] == 1
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["cost_usd"] is None


def test_repeated_identical_tool_call_is_stopped(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("hello", encoding="utf-8")
    repeated = call("1", "read_file", path="sample.txt")
    provider = ScriptedProvider([repeated, repeated, repeated, repeated])
    result = make_agent(tmp_path, provider).run(RunMode.TASK, "Repeat forever")
    assert result.status is RunStatus.BLOCKED
    assert result.stop_reason == "repeated_tool_call"
    events = [
        json.loads(line)
        for line in result.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["type"] for event in events].count("observation") == 4
    guard_events = [
        event["payload"] for event in events if event["type"] == "repeated_action_guard"
    ]
    assert [event["outcome"] for event in guard_events] == ["warning", "terminal"]
    assert guard_events[-1]["streak"] == 4
    assert guard_events[-1]["hard_stop_streak"] == 4
    assert guard_events[-1]["window"] == "contiguous"


def test_repeated_tool_error_warns_before_a_proven_loop_stop(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            call("1", "read_file", path="missing-1.txt"),
            call("2", "read_file", path="missing-1.txt"),
            call("3", "read_file", path="missing-1.txt"),
            call(
                "4",
                "finish",
                status="completed",
                summary="Stopped retrying",
                evidence="The missing path was confirmed.",
            ),
        ]
    )
    result = make_agent(tmp_path, provider).run(RunMode.TASK, "Fail forever")
    assert result.status is RunStatus.COMPLETED
    assert result.stop_reason == "model_finish_tool"
    assert "same tool error has repeated" in provider.requests[3][-1]["content"]


def test_distinct_shell_observations_are_not_no_progress_terminal(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [
            *[
                call(str(index), "shell", command=f"Write-Output {index}")
                for index in range(6)
            ],
            call(
                "finish",
                "finish",
                status="completed",
                summary="Inspection complete",
                evidence="Six distinct observations were collected.",
            ),
        ]
    )
    result = make_agent(tmp_path, provider).run(RunMode.TASK, "Make no progress")
    assert result.status is RunStatus.COMPLETED
    assert result.stop_reason == "model_finish_tool"


def test_noncontiguous_identical_reads_do_not_accumulate_toward_stop(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("sample", encoding="utf-8")
    (tmp_path / "other.txt").write_text("other", encoding="utf-8")
    provider = ScriptedProvider(
        [
            call("1", "read_file", path="sample.txt"),
            call("2", "read_file", path="other.txt"),
            call("3", "read_file", path="sample.txt"),
            call("4", "read_file", path="other.txt"),
            call("5", "read_file", path="sample.txt"),
            call("6", "read_file", path="other.txt"),
            call("7", "read_file", path="sample.txt"),
            call(
                "8",
                "finish",
                status="completed",
                summary="Inspection complete",
                evidence="Both files were inspected repeatedly with intervening evidence.",
            ),
        ]
    )

    result = make_agent(tmp_path, provider).run(RunMode.TASK, "Compare both files")

    assert result.status is RunStatus.COMPLETED
    assert result.stop_reason == "model_finish_tool"
    events = [
        json.loads(line)
        for line in result.trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    assert not [event for event in events if event["type"] == "repeated_action_guard"]


def test_cancel_check_interrupts_without_losing_result(tmp_path: Path) -> None:
    provider = ScriptedProvider([])
    agent = make_agent(tmp_path, provider)
    agent.cancel_check = lambda: True
    result = agent.run(RunMode.TASK, "Stop now")
    assert result.status is RunStatus.INTERRUPTED
    assert result.stop_reason == "user_interrupt"
    assert not provider.requests
