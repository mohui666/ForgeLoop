from __future__ import annotations

import json
import subprocess
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from forgeloop.agent import AgentLoop, RunMode, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.hybrid_controller import (
    EDIT_INTENT_TOOL_NAME,
    ControllerPolicyConfig,
    ControllerPolicyResult,
    HybridControllerEditIntent,
    HybridDecision,
)
from forgeloop.runtime import LocalRuntime
from forgeloop.tools import build_default_tools
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
    model_id = "test/edit-intent"
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


def _names(schemas: list[dict]) -> set[str]:
    return {str(schema["function"]["name"]) for schema in schemas}


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _agent(
    tmp_path: Path,
    provider: ScriptedProvider,
    controller: HybridControllerEditIntent,
) -> AgentLoop:
    workspace = _git_repo(tmp_path)
    return AgentLoop(
        provider,
        build_default_tools(workspace, LocalRuntime()),
        workspace,
        TrajectoryStore(tmp_path / ".forgeloop" / "runs", run_id="edit-intent"),
        BudgetLimits(max_steps=10, max_model_calls=10, max_tool_calls=12),
        controller=controller,
    )


def test_valid_intent_is_recorded_and_becomes_compact_implement_context(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("list", "list_files", {}),
                    ToolCall("search", "search_files", {"pattern": "value"}),
                ),
                usage=ModelUsage(10, 5, 0.001),
                finish_reason="tool_calls",
            ),
            _call(
                "intent",
                EDIT_INTENT_TOOL_NAME,
                target_files=["sample.py"],
                diagnosis="The sample value is stale.",
                intended_change="Change value from 1 to 2.",
                validation_command="python -m unittest discover -s tests -v",
            ),
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
    controller = HybridControllerEditIntent(
        ScriptedPolicy(
            [
                HybridDecision("implement", "edit"),
                HybridDecision("verify", "test"),
                HybridDecision("finalize", "finalize"),
            ]
        )
    )

    result = _agent(tmp_path, provider, controller).run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.COMPLETED
    schemas = [_names(request[1]) for request in provider.requests]
    assert EDIT_INTENT_TOOL_NAME not in schemas[0]
    assert schemas[1] == {
        EDIT_INTENT_TOOL_NAME,
        "finish",
        "read_file",
        "search_files",
        "list_files",
        "git_inspect",
    }
    assert EDIT_INTENT_TOOL_NAME not in schemas[2]
    assert "search_files" not in schemas[2]
    working_context = provider.requests[2][0][-1]["content"]
    assert "Edit intent accepted" in working_context
    assert "target_files: sample.py" in working_context
    events = _events(result.trajectory_path)
    accepted = [event for event in events if event["type"] == "edit_intent_accepted"]
    assert len(accepted) == 1
    assert (
        sum(event["type"] == "edit_intent_handoff_activated" for event in events) == 1
    )
    assert not any(event["type"] == "edit_intent_rejected" for event in events)
    assert accepted[0]["payload"]["intent"]["target_files"] == ["sample.py"]
    summary = events[-1]["payload"]["controller"]["edit_intent_handoff"]
    assert summary["requested"] == 1
    assert summary["accepted"] == 1
    assert summary["rejected"] == 0


def test_invalid_intent_allows_one_replan_then_stops_early(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            _call("read", "read_file", path="sample.py"),
            _call(
                "bad-intent-1",
                EDIT_INTENT_TOOL_NAME,
                target_files=["missing.py"],
                diagnosis="Guessing.",
                intended_change="Change a missing file.",
                validation_command="python -m unittest",
            ),
            _call("focused-read", "read_file", path="sample.py"),
            _call(
                "bad-intent-2",
                EDIT_INTENT_TOOL_NAME,
                target_files=[],
                diagnosis="",
                intended_change="",
                validation_command="",
            ),
        ]
    )
    controller = HybridControllerEditIntent(
        ScriptedPolicy([HybridDecision("implement", "edit")])
    )

    result = _agent(tmp_path, provider, controller).run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "controller_invalid_edit_intent"
    assert result.budget["usage"]["model_calls"] == 4
    schemas = [_names(request[1]) for request in provider.requests]
    assert schemas[1] == {
        EDIT_INTENT_TOOL_NAME,
        "finish",
        "read_file",
        "search_files",
        "list_files",
        "git_inspect",
    }
    assert schemas[2] == {
        EDIT_INTENT_TOOL_NAME,
        "finish",
        "read_file",
        "search_files",
    }
    assert schemas[3] == {EDIT_INTENT_TOOL_NAME, "finish"}
    events = _events(result.trajectory_path)
    assert sum(event["type"] == "edit_intent_rejected" for event in events) == 2
    assert sum(event["type"] == "edit_intent_focused_replan" for event in events) == 1
    summary = events[-1]["payload"]["controller"]["edit_intent_handoff"]
    assert summary["accepted"] == 0
    assert summary["rejected"] == 2
    assert summary["focused_replans_used"] == 1


def test_intent_pending_keeps_context_read_only_and_bounded(tmp_path: Path) -> None:
    workspace = _git_repo(tmp_path)
    controller = HybridControllerEditIntent(ScriptedPolicy([]))
    controller.start(workspace)
    controller._intent_required = True
    schemas = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "properties": {"operation": {"enum": ["status", "diff", "log"]}}
                },
            },
        }
        for name in (
            "read_file",
            "search_files",
            "list_files",
            "git_inspect",
            "apply_patch",
            "shell",
            EDIT_INTENT_TOOL_NAME,
            "finish",
        )
    ]

    filtered = controller.filter_tool_schemas(schemas)

    assert _names(filtered) == {
        "read_file",
        "search_files",
        "list_files",
        "git_inspect",
        EDIT_INTENT_TOOL_NAME,
        "finish",
    }
    git_schema = next(
        schema for schema in filtered if schema["function"]["name"] == "git_inspect"
    )
    assert git_schema["function"]["parameters"]["properties"]["operation"]["enum"] == [
        "status",
        "diff",
    ]
    for index in range(3):
        assert (
            controller.guard_action(
                ToolCall(f"read-{index}", "read_file", {"path": "sample.py"}),
                current_fingerprint="same",
            )
            is None
        )
    assert _names(controller.filter_tool_schemas(schemas)) == {
        EDIT_INTENT_TOOL_NAME,
        "finish",
    }
    blocked = controller.guard_action(
        ToolCall("read-four", "read_file", {"path": "sample.py"}),
        current_fingerprint="same",
    )
    assert blocked is not None
    assert controller.summary()["edit_intent_handoff"]["context_actions"] == 3


def test_plain_response_cannot_bypass_required_intent(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            _call("read", "read_file", path="sample.py"),
            ModelResponse(content="I will edit it.", usage=ModelUsage(10, 5, 0.001)),
            ModelResponse(content="Still no intent.", usage=ModelUsage(10, 5, 0.001)),
        ]
    )
    controller = HybridControllerEditIntent(
        ScriptedPolicy([HybridDecision("implement", "edit")])
    )

    result = _agent(tmp_path, provider, controller).run(RunMode.TASK, "Update sample")

    assert result.status is RunStatus.FAILED
    assert result.stop_reason == "controller_invalid_edit_intent"
    assert result.budget["usage"]["model_calls"] == 3
    assert "One focused replan is available" in provider.requests[2][0][-1]["content"]
