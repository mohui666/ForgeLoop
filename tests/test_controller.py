from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from forgeloop.agent import AgentLoop, RunMode, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.controller import ControllerV1
from forgeloop.models.base import ModelProviderError
from forgeloop.runtime import LocalRuntime
from forgeloop.tools import build_default_tools
from forgeloop.trajectory import TrajectoryStore
from forgeloop.types import Message, ModelResponse, ModelUsage, ToolCall
from forgeloop.workspace import Workspace


class ScriptedProvider:
    model_id = "test/controller"
    policy_identity = None
    capability = None

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = iter(responses)
        self.requests: list[Sequence[Message]] = []

    def complete(self, messages, tools, *, timeout_seconds):
        del tools, timeout_seconds
        self.requests.append(list(messages))
        return next(self.responses)


def _call(call_id: str, name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(call_id, name, arguments),),
        usage=ModelUsage(10, 5, 0.001),
        finish_reason="tool_calls",
    )


def _git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@forgeloop.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "ForgeLoop Test"],
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)


def _agent(path: Path, provider) -> AgentLoop:
    workspace = Workspace(path)
    return AgentLoop(
        provider,
        build_default_tools(workspace, LocalRuntime()),
        workspace,
        TrajectoryStore(path / ".forgeloop" / "runs", run_id="controller"),
        BudgetLimits(max_steps=12, max_model_calls=12, max_tool_calls=12),
        controller=ControllerV1(),
    )


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_controller_recovers_repeated_action_and_requires_finish(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    _git_repo(tmp_path)
    provider = ScriptedProvider(
        [
            _call("read-1", "read_file", path="sample.txt"),
            _call("read-2", "read_file", path="sample.txt"),
            _call(
                "patch",
                "apply_patch",
                path="sample.txt",
                old_text="old",
                new_text="new",
            ),
            ModelResponse(content="Done.", usage=ModelUsage(4, 2, 0.001)),
            _call(
                "finish",
                "finish",
                status="completed",
                summary="Updated sample.",
                evidence="Current diff contains new.",
            ),
        ]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.COMPLETED
    assert "identical repeated action" in provider.requests[2][-1]["content"]
    assert "explicit terminal decision" in provider.requests[4][-1]["content"]
    recoveries = [
        event["payload"]["strategy"]
        for event in _events(result.trajectory_path)
        if event["type"] == "controller_recovery"
    ]
    assert recoveries == ["repeated_action", "missing_explicit_finish"]


def test_controller_reinspects_after_consecutive_patch_failures(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    _git_repo(tmp_path)
    provider = ScriptedProvider(
        [
            _call(
                "bad-1",
                "apply_patch",
                path="sample.txt",
                old_text="not here one",
                new_text="new",
            ),
            _call(
                "bad-2",
                "apply_patch",
                path="sample.txt",
                old_text="not here two",
                new_text="new",
            ),
            _call("read", "read_file", path="sample.txt"),
            _call(
                "good",
                "apply_patch",
                path="sample.txt",
                old_text="old",
                new_text="new",
            ),
            _call(
                "finish",
                "finish",
                status="completed",
                summary="Fixed.",
                evidence="Patch applied.",
            ),
        ]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.COMPLETED
    feedback = provider.requests[2][-1]["content"]
    assert "ERROR tool observation" in feedback
    assert "stale context" in feedback
    summary = _events(result.trajectory_path)[-1]["payload"]["controller"]
    assert summary["recoveries"]["edit_failure_reinspect"] == 1
    assert summary["recoveries"]["tool_error_feedback"] == 2


def test_controller_stops_repeated_plain_final_without_change(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    _git_repo(tmp_path)
    provider = ScriptedProvider(
        [
            ModelResponse(content="Looks fine.", usage=ModelUsage(4, 2, 0.001)),
            ModelResponse(content="Still done.", usage=ModelUsage(4, 2, 0.001)),
        ]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "controller_no_change_final"
    assert "no Git-visible change" in provider.requests[1][-1]["content"]


def test_controller_does_not_force_a_premature_edit_after_inspection(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    _git_repo(tmp_path)
    provider = ScriptedProvider(
        [
            *[
                _call(str(index), "shell", command=f"Write-Output inspect-{index}")
                for index in range(6)
            ],
            _call("inspect-6", "read_file", path="sample.txt"),
            _call("inspect-7", "shell", command="Write-Output focused-inspect"),
            _call("blocked-inspect", "read_file", path="sample.txt"),
            _call(
                "patch",
                "apply_patch",
                path="sample.txt",
                old_text="old",
                new_text="new",
            ),
            _call(
                "finish",
                "finish",
                status="completed",
                summary="Fixed.",
                evidence="Patch applied after recovery.",
            ),
        ]
    )

    result = _agent(tmp_path, provider).run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.COMPLETED
    recoveries = [
        event["payload"]["strategy"]
        for event in _events(result.trajectory_path)
        if event["type"] == "controller_recovery"
    ]
    assert not {
        "no_progress_reinspect",
        "no_progress_action_required",
        "exploration_action_blocked",
    }.intersection(recoveries)


def test_provider_timeout_has_explicit_terminal_reason(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("old\n", encoding="utf-8")
    _git_repo(tmp_path)

    class FailingProvider:
        model_id = "deepseek/deepseek-v4-flash"
        policy_identity = None
        capability = None

        def complete(self, messages, tools, *, timeout_seconds):
            del messages, tools, timeout_seconds
            raise ModelProviderError(
                "Provider request timed out.", details="Timeout: upstream deadline"
            )

    result = _agent(tmp_path, FailingProvider()).run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "provider_timeout"
    assert result.budget["usage"]["model_calls"] == 1
    assert result.budget["usage"]["input_tokens"] is None
