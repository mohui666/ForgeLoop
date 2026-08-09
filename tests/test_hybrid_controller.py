from __future__ import annotations

import json
import subprocess
from collections import deque
from pathlib import Path

import httpx
import pytest

from forgeloop.hybrid_controller import (
    ControllerPolicyConfig,
    ControllerPolicyError,
    ControllerPolicyResult,
    HybridControllerV11,
    HybridControllerV13Simplified,
    HybridDecision,
    OllamaControllerPolicy,
)
from forgeloop.tools.base import ToolResult
from forgeloop.types import ToolCall
from forgeloop.workspace import Workspace


class ScriptedPolicy:
    config = ControllerPolicyConfig.load()

    def __init__(self, values) -> None:
        self.values = deque(values)

    def decide(self, snapshot):
        value = self.values.popleft()
        if isinstance(value, Exception):
            raise value
        return ControllerPolicyResult(value, 0.01, 12, 4)


def _git_repo(path: Path) -> Workspace:
    (path / "sample.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@invalid"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)
    return Workspace(path)


def _budget() -> dict:
    return {
        "limits": {
            "max_steps": 20,
            "max_model_calls": 20,
            "max_tool_calls": 50,
            "max_seconds": 300,
            "max_tokens": 10000,
        },
        "usage": {
            "steps": 2,
            "model_calls": 2,
            "tool_calls": 1,
            "elapsed_seconds": 3.5,
            "total_tokens": 500,
        },
    }


def test_ollama_policy_uses_native_schema_and_compact_input() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": '{"state":"implement","next_action":"edit"}',
                },
                "prompt_eval_count": 91,
                "eval_count": 12,
            },
        )

    policy = OllamaControllerPolicy(transport=httpx.MockTransport(handler))
    snapshot = {
        "current": {"state": "explore", "next_action": "inspect"},
        "recent_tools": [{"action": "inspect", "ok": True, "source_changed": False}],
        "progress_signal": "inspected_no_diff",
        "source_diff": False,
        "test_status": "unknown",
        "implementation_readiness": {
            "ready": False,
            "source_content_read": False,
            "source_files_read": [],
            "candidate_target_files": [],
            "saw_test_evidence": False,
            "saw_error_evidence": False,
            "has_diff": False,
            "has_intent": False,
        },
        "remaining_budget": {"steps": 18},
    }
    result = policy.decide(snapshot)

    assert result.decision == HybridDecision("implement", "edit")
    assert len(captured["format"]["oneOf"]) == 5
    assert all(
        branch["additionalProperties"] is False
        for branch in captured["format"]["oneOf"]
    )
    assert captured["options"] == {
        "temperature": 0,
        "num_ctx": 2048,
        "num_predict": 64,
    }
    user_input = captured["messages"][1]["content"]
    assert "repository" not in user_input
    assert "source code" not in user_input
    assert '"source_content_read":false' in user_input
    assert '"candidate_target_files":[]' in user_input


def test_decision_schema_rejects_extra_fields_and_invalid_pairs() -> None:
    for value in (
        {"state": "explore", "next_action": "inspect", "prompt": "arbitrary"},
        {"state": "finalize", "next_action": "edit"},
        {"state": "unknown", "next_action": "inspect"},
    ):
        try:
            HybridDecision.from_value(value)
        except ControllerPolicyError:
            pass
        else:
            raise AssertionError(f"invalid decision accepted: {value}")


def test_valid_pair_that_conflicts_with_progress_signal_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"message": {"content": '{"state":"implement","next_action":"edit"}'}},
        )

    policy = OllamaControllerPolicy(transport=httpx.MockTransport(handler))
    with pytest.raises(ControllerPolicyError) as exc_info:
        policy.decide(
            {
                "progress_signal": "tests_passed",
                "recent_tools": [
                    {"action": "test", "ok": True, "source_changed": False}
                ],
                "source_diff": True,
                "test_status": "pass",
                "remaining_budget": {"steps": 5},
            }
        )

    assert exc_info.value.category == "semantic_validation"


def test_hybrid_transitions_only_emit_fixed_guidance_and_limit_exploration(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    policy = ScriptedPolicy(
        [
            HybridDecision("implement", "edit"),
            HybridDecision("verify", "test"),
            HybridDecision("finalize", "finalize"),
        ]
    )
    controller = HybridControllerV11(policy)
    controller.start(workspace)

    first = controller.observe_tool(
        ToolCall("1", "read_file", {"path": "sample.py"}),
        ToolResult(True, "value = 1"),
        before_fingerprint=initial,
        after_fingerprint=initial,
        budget_snapshot=_budget(),
    )
    assert first[-1].feedback.startswith("Hybrid Controller v1.1: phase=implement")

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    changed = workspace.git_progress_fingerprint()
    second = controller.observe_tool(
        ToolCall("2", "apply_patch", {}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=changed,
        budget_snapshot=_budget(),
    )
    assert second[-1].feedback.startswith("Hybrid Controller v1.1: phase=verify")
    blocked = controller.guard_action(
        ToolCall("3", "search_files", {"query": "anything"}),
        current_fingerprint=changed,
    )
    assert blocked is not None
    assert blocked.strategy == "hybrid_phase_action_blocked"
    assert (
        controller.guard_action(
            ToolCall("4", "read_file", {"path": "sample.py"}),
            current_fingerprint=changed,
        )
        is None
    )

    third = controller.observe_tool(
        ToolCall("5", "shell", {"command": "pytest -q"}),
        ToolResult(True, "1 passed"),
        before_fingerprint=changed,
        after_fingerprint=changed,
        budget_snapshot=_budget(),
    )
    assert "call finish explicitly" in third[-1].feedback
    events = controller.drain_events()
    assert [event[1]["state"] for event in events] == [
        "implement",
        "verify",
        "finalize",
    ]
    assert all(
        set(event[1]["input"])
        == {
            "current",
            "recent_tools",
            "progress_signal",
            "source_diff",
            "test_status",
            "implementation_readiness",
            "remaining_budget",
        }
        for event in events
    )
    summary = controller.summary()
    assert summary["decisions"] == 3
    assert summary["fallbacks"] == 0
    assert summary["policy_usage"]["cost_usd"] == 0.0


def test_invalid_local_policy_falls_back_without_blocking_deterministic_logic(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    fingerprint = workspace.git_progress_fingerprint()
    controller = HybridControllerV11(
        ScriptedPolicy(
            [ControllerPolicyError("schema_validation", "bad model response")]
        )
    )
    controller.start(workspace)

    recoveries = controller.observe_tool(
        ToolCall("1", "read_file", {"path": "sample.py"}),
        ToolResult(True, "value = 1"),
        before_fingerprint=fingerprint,
        after_fingerprint=fingerprint,
        budget_snapshot=_budget(),
    )

    assert recoveries == ()
    assert (
        controller.guard_action(
            ToolCall("2", "read_file", {"path": "sample.py"}),
            current_fingerprint=fingerprint,
        )
        is None
    )
    events = controller.drain_events()
    assert events[0][0] == "controller_policy_fallback"
    assert events[0][1]["error_category"] == "schema_validation"
    assert controller.summary()["fallbacks"] == 1


def test_v13_classifier_is_advisory_and_source_diff_only_guides_next_steps(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(
        ScriptedPolicy(
            [
                HybridDecision("implement", "edit"),
                HybridDecision("verify", "test"),
            ]
        )
    )
    controller.start(workspace)
    schemas = [
        {"type": "function", "function": {"name": name}}
        for name in ("list_files", "search_files", "read_file", "apply_patch", "shell")
    ]

    assert controller.additional_tools(workspace) == ()
    assert controller.filter_tool_schemas(schemas) == schemas
    first = controller.observe_tool(
        ToolCall("read", "read_file", {"path": "sample.py"}),
        ToolResult(True, "value = 1\n"),
        before_fingerprint=initial,
        after_fingerprint=initial,
        budget_snapshot=_budget(),
    )
    assert first == ()
    assert (
        controller.guard_action(
            ToolCall("search", "search_files", {"pattern": "value"}),
            current_fingerprint=initial,
        )
        is None
    )

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    changed = workspace.git_progress_fingerprint()
    second = controller.observe_tool(
        ToolCall("patch", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=changed,
        budget_snapshot=_budget(),
    )

    assert [recovery.strategy for recovery in second] == ["source_diff_next_steps"]
    assert "focused test" in second[0].feedback
    assert (
        controller.guard_action(
            ToolCall("list", "list_files", {}), current_fingerprint=changed
        )
        is None
    )
    events = controller.drain_events()
    decisions = [
        payload for name, payload in events if name == "controller_policy_decision"
    ]
    assert len(decisions) == 2
    assert all(decision["advisory"] is True for decision in decisions)
    assert sum(name == "controller_source_diff_detected" for name, _ in events) == 1
    summary = controller.summary()["simplified_control"]
    assert summary == {
        "classifier_advisory_only": True,
        "classifier_action_gating": False,
        "edit_intent_required": False,
        "tool_schemas_filtered_by_state": False,
        "source_diff_guidance_count": 1,
    }
