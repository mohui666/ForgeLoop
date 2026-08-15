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
    HybridControllerV14ExplicitCloseout,
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


class SignalPolicy:
    config = ControllerPolicyConfig.load()

    def decide(self, snapshot):
        decision = {
            "needs_inspection": HybridDecision("explore", "inspect"),
            "inspected_no_diff": HybridDecision("implement", "edit"),
            "modified_untested": HybridDecision("verify", "test"),
            "tests_failed": HybridDecision("verify", "replan"),
            "tests_passed": HybridDecision("finalize", "finalize"),
        }[snapshot["progress_signal"]]
        return ControllerPolicyResult(decision, 0.01, 12, 4)


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


def _budget_at(model_calls: int, total_tokens: int) -> dict:
    budget = _budget()
    budget["usage"]["model_calls"] = model_calls
    budget["usage"]["steps"] = model_calls
    budget["usage"]["total_tokens"] = total_tokens
    return budget


def test_v13_classifier_is_advisory_and_coherent_edit_batches_are_not_blocked(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(SignalPolicy())
    controller.start(workspace)

    controller.observe_tool(
        ToolCall("read", "read_file", {"path": "sample.py"}),
        ToolResult(True, "value = 1\n"),
        before_fingerprint=initial,
        after_fingerprint=initial,
        budget_snapshot=_budget(),
    )
    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    first_edit = workspace.git_progress_fingerprint()
    recoveries = controller.observe_tool(
        ToolCall("patch-1", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=first_edit,
        budget_snapshot=_budget(),
    )
    assert [item.strategy for item in recoveries] == ["source_edit_detected"]

    assert (
        controller.guard_action(
            ToolCall("patch-2", "apply_patch", {"path": "sample.py"}),
            current_fingerprint=first_edit,
        )
        is None
    )
    (tmp_path / "sample.py").write_text("value = 3\n", encoding="utf-8")
    second_edit = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch-2", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=first_edit,
        after_fingerprint=second_edit,
        budget_snapshot=_budget(),
    )

    events = controller.drain_events()
    decisions = [
        payload for name, payload in events if name == "controller_policy_decision"
    ]
    assert decisions
    assert all(item["advisory"] is True for item in decisions)
    closure = controller.summary()["execution_closure"]
    assert closure["phase"] == "needs_validation"
    assert closure["action_guards"] is False
    assert closure["classifier_advisory_only"] is True


def test_v13_long_edit_batch_is_not_terminated_by_call_or_token_deadlines(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(SignalPolicy())
    controller.start(workspace)

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    first_edit = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch-98", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=first_edit,
        budget_snapshot=_budget_at(98, 2_657_873),
    )

    (tmp_path / "sample.py").write_text("value = 3\n", encoding="utf-8")
    second_edit = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch-99", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=first_edit,
        after_fingerprint=second_edit,
        budget_snapshot=_budget_at(99, 2_700_000),
    )

    (tmp_path / "sample.py").write_text("value = 4\n", encoding="utf-8")
    third_edit = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch-100", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=second_edit,
        after_fingerprint=third_edit,
        budget_snapshot=_budget_at(100, 2_742_000),
    )

    assert (
        controller.before_model_call(
            current_fingerprint=third_edit,
            budget_snapshot=_budget_at(101, 2_800_000),
        )
        is None
    )
    advisory = controller.before_model_call(
        current_fingerprint=third_edit,
        budget_snapshot=_budget_at(106, 3_200_000),
    )
    assert advisory is not None
    assert advisory.strategy == "validation_due"
    assert (
        controller.before_model_call(
            current_fingerprint=third_edit,
            budget_snapshot=_budget_at(200, 8_000_000),
        )
        is None
    )

    passed = controller.observe_tool(
        ToolCall("validate-201", "validate", {"command": "pytest -q"}),
        ToolResult(True, "1 passed", {"exit_code": 0}),
        before_fingerprint=third_edit,
        after_fingerprint=third_edit,
        budget_snapshot=_budget_at(201, 8_100_000),
    )
    assert [item.strategy for item in passed][-1] == "validation_passed"
    assert controller.summary()["execution_closure"]["phase"] == "needs_review"


def test_v13_failed_validation_can_repair_and_retest_beyond_old_limits(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(SignalPolicy())
    controller.start(workspace)

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    edited = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(20, 500_000),
    )
    for attempt in range(1, 7):
        controller.observe_tool(
            ToolCall(
                f"validate-{attempt}",
                "validate",
                {"command": f"pytest -q tests/test_{attempt}.py"},
            ),
            ToolResult(False, f"failure {attempt}", {"exit_code": 1}),
            before_fingerprint=edited,
            after_fingerprint=edited,
            budget_snapshot=_budget_at(20 + attempt, 500_000 + attempt * 80_000),
        )

    advisory = controller.before_model_call(
        current_fingerprint=edited,
        budget_snapshot=_budget_at(30, 1_500_000),
    )
    assert advisory is not None
    assert advisory.strategy == "repair_validation_due"
    assert (
        controller.before_model_call(
            current_fingerprint=edited,
            budget_snapshot=_budget_at(80, 5_000_000),
        )
        is None
    )

    passed = controller.observe_tool(
        ToolCall("validate-pass", "validate", {"command": "pytest -q"}),
        ToolResult(True, "1 passed", {"exit_code": 0}),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(81, 5_100_000),
    )
    assert [item.strategy for item in passed][-1] == "validation_passed"
    closure = controller.summary()["execution_closure"]
    assert closure["validation"]["attempts"] == 7
    assert closure["long_horizon_guards"]["phase_deadlines_terminal"] is False


def test_v13_tracks_validation_fix_retest_review_and_finish(tmp_path: Path) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(SignalPolicy())
    controller.start(workspace)

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    edited = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch-1", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(2, 1_000),
    )
    failed = controller.observe_tool(
        ToolCall("validate-1", "validate", {"command": "pytest -q"}),
        ToolResult(False, "1 failed", {"exit_code": 1}),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(3, 2_000),
    )
    assert [item.strategy for item in failed][-1] == "validation_failed"

    (tmp_path / "sample.py").write_text("value = 3\n", encoding="utf-8")
    fixed = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch-2", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=edited,
        after_fingerprint=fixed,
        budget_snapshot=_budget_at(4, 3_000),
    )
    passed = controller.observe_tool(
        ToolCall("validate-2", "validate", {"command": "pytest -q"}),
        ToolResult(True, "1 passed", {"exit_code": 0}),
        before_fingerprint=fixed,
        after_fingerprint=fixed,
        budget_snapshot=_budget_at(5, 4_000),
    )
    assert [item.strategy for item in passed][-1] == "validation_passed"

    controller.observe_tool(
        ToolCall("diff", "git_diff", {}),
        ToolResult(True, "diff --git a/sample.py b/sample.py"),
        before_fingerprint=fixed,
        after_fingerprint=fixed,
        budget_snapshot=_budget_at(6, 5_000),
    )
    assert (
        controller.review_finish(
            ToolCall("finish", "finish", {"status": "completed"}),
            current_fingerprint=fixed,
        )
        is None
    )
    closure = controller.summary()["execution_closure"]
    assert closure["phase"] == "ready_to_finish"
    assert closure["validation"]["attempts"] == 2
    assert closure["validation"]["passes"] == 1
    assert closure["validation"]["failures"] == 1
    assert closure["diff_reviewed"] is True


def test_v13_invalidates_validation_that_or_later_edit_changes_tree(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(SignalPolicy())
    controller.start(workspace)

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    edited = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(2, 1_000),
    )

    (tmp_path / "sample.py").write_text("value = 3\n", encoding="utf-8")
    changed_by_validation = workspace.git_progress_fingerprint()
    changed = controller.observe_tool(
        ToolCall("validate-1", "validate", {"command": "python check.py"}),
        ToolResult(True, "check passed", {"exit_code": 0}),
        before_fingerprint=edited,
        after_fingerprint=changed_by_validation,
        budget_snapshot=_budget_at(3, 2_000),
    )
    assert [item.strategy for item in changed][-1] == "validation_changed_tree"

    controller.observe_tool(
        ToolCall("validate-2", "validate", {"command": "pytest -q"}),
        ToolResult(True, "1 passed", {"exit_code": 0}),
        before_fingerprint=changed_by_validation,
        after_fingerprint=changed_by_validation,
        budget_snapshot=_budget_at(4, 3_000),
    )
    controller.observe_tool(
        ToolCall("diff", "git_diff", {}),
        ToolResult(True, "diff"),
        before_fingerprint=changed_by_validation,
        after_fingerprint=changed_by_validation,
        budget_snapshot=_budget_at(5, 4_000),
    )

    (tmp_path / "sample.py").write_text("value = 4\n", encoding="utf-8")
    review_fix = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("review-fix", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=changed_by_validation,
        after_fingerprint=review_fix,
        budget_snapshot=_budget_at(6, 5_000),
    )
    rejected = controller.review_finish(
        ToolCall("finish", "finish", {"status": "completed"}),
        current_fingerprint=review_fix,
    )
    assert rejected is not None
    assert rejected.strategy == "finish_not_ready"
    closure = controller.summary()["execution_closure"]
    assert closure["phase"] == "needs_validation"
    assert closure["validation"]["source_changes"] == 1


def test_v13_exploration_has_no_token_or_phase_local_terminal(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(SignalPolicy())
    controller.start(workspace)

    decision = controller.before_model_call(
        current_fingerprint=initial,
        budget_snapshot=_budget_at(8, 5_000),
    )
    assert decision is None
    first_turn = {
        schema["function"]["name"]
        for schema in controller.filter_tool_schemas(
            [
                {"type": "function", "function": {"name": name}}
                for name in (
                    "list_files",
                    "search_files",
                    "read_file",
                    "apply_patch",
                    "shell",
                    "finish",
                )
            ]
        )
    }
    assert first_turn == {
        "list_files",
        "search_files",
        "read_file",
        "apply_patch",
        "shell",
        "finish",
    }
    advisory_only = controller.guard_action(
        ToolCall("stale", "search_files", {"pattern": "value"}),
        current_fingerprint=initial,
    )
    assert advisory_only is None
    later = controller.before_model_call(
        current_fingerprint=initial,
        budget_snapshot=_budget_at(12, 800_000),
    )
    assert later is None


def test_v13_hands_off_when_source_evidence_is_ready_before_half_budget(
    tmp_path: Path,
) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(SignalPolicy())
    controller.start(workspace)
    controller.observe_tool(
        ToolCall("read", "read_file", {"path": "sample.py"}),
        ToolResult(True, "value = 1\n"),
        before_fingerprint=initial,
        after_fingerprint=initial,
        budget_snapshot=_budget_at(5, 4_000),
    )

    handoff = controller.before_model_call(
        current_fingerprint=initial,
        budget_snapshot=_budget_at(6, 5_000),
    )

    assert handoff is not None
    assert handoff.strategy == "implementation_due"
    assert "source evidence" in handoff.trigger


def test_v13_auto_finishes_one_decision_after_review(tmp_path: Path) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV13Simplified(SignalPolicy())
    controller.start(workspace)

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    edited = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(2, 1_000),
    )
    controller.observe_tool(
        ToolCall("validate", "validate", {"command": "pytest -q"}),
        ToolResult(True, "1 passed", {"exit_code": 0}),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(3, 2_000),
    )
    controller.observe_tool(
        ToolCall("diff", "git_diff", {}),
        ToolResult(True, "diff"),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(4, 3_000),
    )
    assert (
        controller.before_model_call(
            current_fingerprint=edited,
            budget_snapshot=_budget_at(4, 3_000),
        )
        is None
    )
    terminal = controller.before_model_call(
        current_fingerprint=edited,
        budget_snapshot=_budget_at(5, 4_000),
    )
    assert terminal is not None
    assert terminal.stop_reason == "controller_ready_auto_finish"


def test_v14_requires_explicit_finish_after_complete_review(tmp_path: Path) -> None:
    workspace = _git_repo(tmp_path)
    initial = workspace.git_progress_fingerprint()
    controller = HybridControllerV14ExplicitCloseout(SignalPolicy())
    controller.start(workspace)

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    edited = workspace.git_progress_fingerprint()
    controller.observe_tool(
        ToolCall("patch", "apply_patch", {"path": "sample.py"}),
        ToolResult(True, "applied"),
        before_fingerprint=initial,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(2, 1_000),
    )
    controller.observe_tool(
        ToolCall("validate", "validate", {"command": "pytest -q"}),
        ToolResult(True, "1 passed", {"exit_code": 0}),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(3, 2_000),
    )
    controller.observe_tool(
        ToolCall("partial", "git_diff", {"path": "sample.py"}),
        ToolResult(True, "diff", {"review_scope": "partial"}),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(4, 3_000),
    )
    assert controller.summary()["execution_closure"]["phase"] == "needs_review"
    controller.observe_tool(
        ToolCall("cached", "git_diff", {"cached": True}),
        ToolResult(True, "diff", {"review_scope": "worktree"}),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(4, 3_000),
    )
    controller.observe_tool(
        ToolCall("failed", "git_diff", {}),
        ToolResult(False, "git failed", {"review_scope": "worktree"}),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(4, 3_000),
    )
    assert controller.summary()["execution_closure"]["phase"] == "needs_review"

    controller.observe_tool(
        ToolCall("complete", "git_diff", {}),
        ToolResult(True, "diff", {"review_scope": "worktree"}),
        before_fingerprint=edited,
        after_fingerprint=edited,
        budget_snapshot=_budget_at(5, 4_000),
    )
    assert controller.summary()["execution_closure"]["phase"] == "ready_to_finish"
    assert (
        controller.before_model_call(
            current_fingerprint=edited,
            budget_snapshot=_budget_at(50, 500_000),
        )
        is None
    )
    assert (
        controller.review_finish(
            ToolCall("finish", "finish", {"status": "completed"}),
            current_fingerprint=edited,
        )
        is None
    )
    plain_final = controller.review_final(
        "Validated and reviewed.", current_fingerprint=edited
    )
    assert plain_final.stop_reason == "controller_ready_final_message"
    closure = controller.summary()["execution_closure"]
    assert closure["auto_finished"] is False
    assert closure["long_horizon_guards"]["auto_finish_after_review"] is False


def test_base_relative_fingerprint_survives_commit(tmp_path: Path) -> None:
    workspace = _git_repo(tmp_path)
    base = workspace.git_snapshot().head
    assert base is not None
    initial = workspace.git_progress_fingerprint(base_head=base)

    (tmp_path / "sample.py").write_text("value = 2\n", encoding="utf-8")
    dirty = workspace.git_progress_fingerprint(base_head=base)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "change"], cwd=tmp_path, check=True)
    committed = workspace.git_progress_fingerprint(base_head=base)

    assert dirty != initial
    assert committed == dirty
