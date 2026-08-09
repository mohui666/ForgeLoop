from __future__ import annotations

import json
import subprocess
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from forgeloop.agent import AgentLoop, RunMode, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.hybrid_controller import (
    ControllerPolicyConfig,
    ControllerPolicyResult,
    HybridControllerV12,
    HybridDecision,
)
from forgeloop.runtime import LocalRuntime
from forgeloop.tools import build_default_tools
from forgeloop.tools.base import ToolResult
from forgeloop.trajectory import TrajectoryStore
from forgeloop.types import Message, ModelResponse, ModelUsage, ToolCall
from forgeloop.workspace import Workspace


class ScriptedPolicy:
    config = ControllerPolicyConfig.load()

    def __init__(self, decisions: list[HybridDecision]) -> None:
        self.decisions = deque(decisions)

    def decide(self, snapshot) -> ControllerPolicyResult:
        del snapshot
        return ControllerPolicyResult(self.decisions.popleft(), 0.01, 10, 3)


class ScriptedProvider:
    model_id = "test/hybrid-v1.2"
    policy_identity = None
    capability = None

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = iter(responses)
        self.requests: list[tuple[Sequence[Message], list[dict]]] = []

    def complete(self, messages, tools, *, timeout_seconds):
        del timeout_seconds
        self.requests.append((list(messages), list(tools)))
        return next(self.responses)


def _call(call_id: str, name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(call_id, name, arguments),),
        usage=ModelUsage(10, 5, 0.001),
        finish_reason="tool_calls",
    )


def _git_repo(path: Path) -> Workspace:
    (path / "sample.py").write_text("value = 1\n", encoding="utf-8")
    tests = path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        "import unittest\n\n"
        "class SampleTest(unittest.TestCase):\n"
        "    def test_value(self):\n"
        "        self.assertEqual(2, 2)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@invalid"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)
    return Workspace(path)


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _names(schemas: list[dict]) -> set[str]:
    return {str(schema["function"]["name"]) for schema in schemas}


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_implement_gate_filters_broad_tools_and_allows_one_replan() -> None:
    controller = HybridControllerV12(ScriptedPolicy([]))
    controller._state = "implement"
    controller._located_paths.add("src/worker.py")
    schemas = [
        _schema(name)
        for name in (
            "read_file",
            "search_files",
            "list_files",
            "apply_patch",
            "git_diff",
            "shell",
            "finish",
        )
    ]

    assert _names(controller.filter_tool_schemas(schemas)) == {
        "read_file",
        "apply_patch",
        "git_diff",
        "shell",
        "finish",
    }
    assert (
        controller.guard_action(
            ToolCall("scoped", "read_file", {"path": "src/helper.py"}),
            current_fingerprint="same",
        )
        is None
    )
    assert (
        controller.guard_action(
            ToolCall("targeted", "read_file", {"path": "tests/test_worker.py"}),
            current_fingerprint="same",
        )
        is None
    )

    blocked = controller.guard_action(
        ToolCall("broad", "search_files", {"pattern": "selector", "path": "."}),
        current_fingerprint="same",
    )
    assert blocked is not None
    assert blocked.strategy == "controller_action_blocked"
    assert {"search_files", "list_files"} <= _names(
        controller.filter_tool_schemas(schemas)
    )

    assert (
        controller.guard_action(
            ToolCall("replan", "search_files", {"pattern": "selector", "path": "."}),
            current_fingerprint="same",
        )
        is None
    )
    assert "search_files" not in _names(controller.filter_tool_schemas(schemas))

    blocked_again = controller.guard_action(
        ToolCall("broad-2", "list_files", {"path": "."}),
        current_fingerprint="same",
    )
    assert blocked_again is not None
    events = controller.drain_events()
    assert [event[0] for event in events] == [
        "controller_action_blocked",
        "controller_replan_allowed",
        "controller_action_blocked",
    ]
    assert events[0][1]["replan_available"] is True
    assert events[2][1]["replan_available"] is False


def test_non_explore_schema_removes_git_history_operation() -> None:
    controller = HybridControllerV12(ScriptedPolicy([]))
    controller._state = "implement"
    schema = {
        "type": "function",
        "function": {
            "name": "git_inspect",
            "parameters": {
                "properties": {
                    "operation": {"enum": ["status", "diff", "log"]},
                }
            },
        },
    }

    filtered = controller.filter_tool_schemas([schema])

    assert filtered[0]["function"]["parameters"]["properties"]["operation"]["enum"] == [
        "status",
        "diff",
    ]
    assert schema["function"]["parameters"]["properties"]["operation"]["enum"] == [
        "status",
        "diff",
        "log",
    ]


def test_verify_and_finalize_gate_shell_actions_by_category() -> None:
    controller = HybridControllerV12(ScriptedPolicy([]))
    controller._located_paths.add("src/worker.py")

    controller._state = "verify"
    assert (
        controller.guard_action(
            ToolCall("test", "shell", {"command": "cargo nextest run"}),
            current_fingerprint="same",
        )
        is None
    )
    assert (
        controller.guard_action(
            ToolCall("read", "shell", {"command": "sed -n '1,80p' src/worker.py"}),
            current_fingerprint="same",
        )
        is None
    )
    assert (
        controller.guard_action(
            ToolCall("find", "shell", {"command": "find . -name '*.rs'"}),
            current_fingerprint="same",
        )
        is not None
    )
    assert (
        controller.guard_action(
            ToolCall(
                "test-and-browse",
                "shell",
                {"command": "cargo nextest run && find . -name '*.rs'"},
            ),
            current_fingerprint="same",
        )
        is not None
    )
    assert (
        controller.guard_action(
            ToolCall("log", "shell", {"command": "git log --oneline -5"}),
            current_fingerprint="same",
        )
        is not None
    )
    assert (
        controller.guard_action(
            ToolCall("git-log", "git_inspect", {"operation": "log"}),
            current_fingerprint="same",
        )
        is not None
    )
    assert (
        controller.guard_action(
            ToolCall("git-status", "git_inspect", {"operation": "status"}),
            current_fingerprint="same",
        )
        is None
    )
    assert (
        controller.guard_action(
            ToolCall("replan", "list_files", {"path": "src"}),
            current_fingerprint="same",
        )
        is not None
    )
    assert (
        controller.guard_action(
            ToolCall("replan-use", "list_files", {"path": "src"}),
            current_fingerprint="same",
        )
        is None
    )

    controller._state = "finalize"
    controller._replan_granted = False
    controller._replan_used = False
    assert (
        controller.guard_action(
            ToolCall("commit", "shell", {"command": "git add -A && git commit -m fix"}),
            current_fingerprint="same",
        )
        is None
    )
    assert (
        controller.guard_action(
            ToolCall("list", "list_files", {"path": "."}),
            current_fingerprint="same",
        )
        is not None
    )


def test_classifier_replan_resets_deterministic_window_only_once() -> None:
    controller = HybridControllerV12(
        ScriptedPolicy(
            [
                HybridDecision("explore", "inspect"),
                HybridDecision("explore", "inspect"),
            ]
        )
    )
    controller._state = "implement"
    controller._next_action = "edit"
    controller._no_progress_actions = 7
    controller._action_required = True

    controller.observe_tool(
        ToolCall("missing", "read_file", {"path": "src/missing.py"}),
        ToolResult(False, "not found"),
        before_fingerprint="same",
        after_fingerprint="same",
    )

    assert controller._no_progress_actions == 0
    assert controller._action_required is False
    reset_events = [
        event
        for event in controller.drain_events()
        if event[0] == "controller_replan_window_reset"
    ]
    assert len(reset_events) == 1
    assert reset_events[0][1]["no_progress_actions"] == 8

    controller._state = "implement"
    controller._next_action = "edit"
    controller._no_progress_actions = 7
    controller._action_required = True
    controller.observe_tool(
        ToolCall("missing-again", "read_file", {"path": "src/still-missing.py"}),
        ToolResult(False, "not found"),
        before_fingerprint="same",
        after_fingerprint="same",
    )

    assert controller._no_progress_actions == 8
    assert controller._action_required is True
    assert not any(
        event[0] == "controller_replan_window_reset"
        for event in controller.drain_events()
    )
    assert (
        controller.summary()["state_aware_gating"]["deterministic_replan_window_resets"]
        == 1
    )


def test_implement_gate_exhausts_scoped_reads_then_requires_action() -> None:
    controller = HybridControllerV12(
        ScriptedPolicy([HybridDecision("implement", "edit") for _ in range(3)])
    )
    controller._state = "implement"
    controller._next_action = "edit"
    schemas = [_schema("read_file"), _schema("apply_patch"), _schema("shell")]

    for index in range(3):
        controller.observe_tool(
            ToolCall(
                f"read-{index}",
                "read_file",
                {"path": f"src/target-{index}.py"},
            ),
            ToolResult(True, "source"),
            before_fingerprint="same",
            after_fingerprint="same",
        )

    assert "read_file" not in _names(controller.filter_tool_schemas(schemas))
    blocked = controller.guard_action(
        ToolCall("read-more", "read_file", {"path": "src/more.py"}),
        current_fingerprint="same",
    )
    assert blocked is not None
    assert blocked.strategy == "controller_action_blocked"
    assert "allowance is exhausted" in blocked.feedback
    blocked_event = controller.drain_events()[-1]
    assert blocked_event[0] == "controller_action_blocked"
    assert "read_file" not in blocked_event[1]["allowed_actions"]
    assert (
        controller.guard_action(
            ToolCall(
                "shell-read",
                "shell",
                {"command": "sed -n '1,80p' src/target-0.py"},
            ),
            current_fingerprint="same",
        )
        is not None
    )
    assert (
        controller.guard_action(
            ToolCall(
                "patch",
                "apply_patch",
                {"path": "src/target-0.py", "old_text": "a", "new_text": "b"},
            ),
            current_fingerprint="same",
        )
        is None
    )


def test_agent_exposes_state_filtered_schemas_and_records_blocked_replan(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    provider = ScriptedProvider(
        [
            _call("read", "read_file", path="sample.py"),
            _call("blocked", "search_files", pattern="value", path="."),
            _call("replan", "search_files", pattern="value", path="."),
            _call(
                "patch",
                "apply_patch",
                path="sample.py",
                old_text="value = 1",
                new_text="value = 2",
            ),
            _call(
                "test",
                "shell",
                command="python -m unittest discover -s tests -v",
            ),
            _call(
                "finish",
                "finish",
                status="completed",
                summary="Updated sample.",
                evidence="Tests passed.",
            ),
        ]
    )
    controller = HybridControllerV12(
        ScriptedPolicy(
            [
                HybridDecision("implement", "edit"),
                HybridDecision("implement", "edit"),
                HybridDecision("verify", "test"),
                HybridDecision("finalize", "finalize"),
            ]
        )
    )
    agent = AgentLoop(
        provider,
        build_default_tools(workspace, LocalRuntime()),
        workspace,
        TrajectoryStore(tmp_path / ".forgeloop" / "runs", run_id="v12"),
        BudgetLimits(max_steps=10, max_model_calls=10, max_tool_calls=12),
        controller=controller,
    )

    result = agent.run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.COMPLETED
    request_tools = [_names(request[1]) for request in provider.requests]
    assert {"search_files", "list_files"} <= request_tools[0]
    assert "search_files" not in request_tools[1]
    assert "search_files" in request_tools[2]
    assert "search_files" not in request_tools[3]
    assert "search_files" not in request_tools[4]
    assert "search_files" not in request_tools[5]
    events = _events(result.trajectory_path)
    assert sum(event["type"] == "controller_action_blocked" for event in events) == 1
    assert sum(event["type"] == "controller_replan_allowed" for event in events) == 1
    finished = events[-1]["payload"]["controller"]
    assert finished["state_aware_gating"]["blocked"] == 1
    assert finished["state_aware_gating"]["controlled_replans_used"] == 1
