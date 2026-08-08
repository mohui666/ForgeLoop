from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from forgeloop.budget import BudgetLimits
from forgeloop.evals import (
    EvalInfrastructureError,
    EvalRunner,
    EvalSuite,
    EvalTaskResult,
    FailureCategory,
    FixtureRepository,
    aggregate_results,
    default_suite_path,
)
from forgeloop.security import SecretRedactor
from forgeloop.trajectory import TrajectoryStore
from forgeloop.types import ModelResponse, ModelUsage, ToolCall
from forgeloop.verifier import VerifierResult
from forgeloop.workspace import Workspace


class ScriptedProvider:
    model_id = "mock/model"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)

    def complete(self, messages, tools, *, timeout_seconds):
        del messages, tools, timeout_seconds
        return next(self._responses)


def response(call_id: str, name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(call_id, name, arguments),),
        usage=ModelUsage(
            input_tokens=10,
            output_tokens=5,
            cost_usd=None,
            cached_tokens=2,
            reasoning_tokens=1,
            usage_source="provider_response",
            model="mock/model",
            provider="mock",
        ),
    )


def test_loads_fixed_smoke_suite() -> None:
    suite = EvalSuite.load(default_suite_path())
    assert suite.id == "python-smoke-v1"
    assert len(suite.tasks) == 30
    assert len(suite.select_stage("a")) == 1
    assert len(suite.select_stage("b")) == 3
    assert len(suite.select_stage("c")) == 30
    assert all(len(task.base_commit) == 40 for task in suite.tasks)
    assert sum(task.difficulty == "easy" for task in suite.tasks) == 10
    assert sum(task.difficulty == "medium" for task in suite.tasks) == 14
    assert sum(task.difficulty == "hard" for task in suite.tasks) == 6


def test_fixture_reset_reproduces_expected_commit(tmp_path: Path) -> None:
    task = EvalSuite.load(default_suite_path()).select_stage("a")[0]
    destination = tmp_path / "workspace"
    head = FixtureRepository().prepare(task, destination)
    assert head == task.base_commit
    assert Workspace(destination).git_snapshot().status == ""

    assert not (destination / "__pycache__").exists()
    (destination / "__pycache__").mkdir()
    (destination / "__pycache__" / "ignored.pyc").write_bytes(b"transient")
    diff, status = FixtureRepository().final_diff(destination)
    assert diff == ""
    assert status == ""

    wrong = replace(task, base_commit="0" * 40)
    try:
        FixtureRepository().prepare(wrong, tmp_path / "wrong")
    except EvalInfrastructureError as exc:
        assert "Base SHA mismatch" in str(exc)
    else:
        raise AssertionError("Expected reset to reject a mismatched base SHA")


def test_eval_runner_uses_verifier_for_success_and_serializes(tmp_path: Path) -> None:
    suite = EvalSuite.load(default_suite_path())
    provider = ScriptedProvider(
        [
            response(
                "patch",
                "apply_patch",
                path="pricing.py",
                old_text="if is_member or subtotal >= 100:",
                new_text="if is_member and subtotal >= 100:",
            ),
            response(
                "finish",
                "finish",
                status="completed",
                summary="fixed",
                evidence="updated condition",
            ),
        ]
    )
    summary, run_dir = EvalRunner(
        provider=provider,
        limits=BudgetLimits(max_seconds=60, max_tokens=1000),
        output_root=tmp_path / "runs",
    ).run(suite, suite.select_stage("a"))

    assert summary.solved == 1
    assert summary.pass_rate == 1.0
    assert summary.total_tokens == 30
    assert summary.total_cost_usd is None
    saved = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["task_results"][0]["verifier"]["passed"] is True
    assert saved["task_results"][0]["final_diff"]
    assert (run_dir / "tasks.jsonl").is_file()
    assert not (run_dir / "workspaces").exists()


def test_eval_runner_can_create_a_runtime_from_task_metadata(tmp_path: Path) -> None:
    suite = EvalSuite.load(default_suite_path())
    seen: list[str] = []
    provider = ScriptedProvider(
        [
            response(
                "patch",
                "apply_patch",
                path="pricing.py",
                old_text="if is_member or subtotal >= 100:",
                new_text="if is_member and subtotal >= 100:",
            ),
            response(
                "finish",
                "finish",
                status="completed",
                summary="fixed",
                evidence="updated condition",
            ),
        ]
    )

    def runtime_for_task(task):
        seen.append(task.id)
        from forgeloop.runtime import LocalRuntime

        return LocalRuntime()

    summary, _ = EvalRunner(
        provider=provider,
        limits=BudgetLimits(max_seconds=60, max_tokens=1000),
        output_root=tmp_path / "runs",
        task_runtime_factory=runtime_for_task,
    ).run(suite, suite.select_stage("a"))

    assert summary.solved == 1
    assert seen == ["condition-boundary"]


def test_verifier_failure_is_model_failure(tmp_path: Path) -> None:
    suite = EvalSuite.load(default_suite_path())
    provider = ScriptedProvider(
        [
            response(
                "finish",
                "finish",
                status="completed",
                summary="claimed success",
                evidence="none",
            )
        ]
    )
    summary, _ = EvalRunner(
        provider=provider,
        limits=BudgetLimits(max_seconds=60, max_tokens=1000),
        output_root=tmp_path / "runs",
    ).run(suite, suite.select_stage("a"))
    task = summary.task_results[0]
    assert summary.solved == 0
    assert task["success"] is False
    assert task["failure_category"] == FailureCategory.MODEL.value


def test_eval_runner_repeats_use_independent_attempt_workspaces(tmp_path: Path) -> None:
    suite = EvalSuite.load(default_suite_path())
    responses = []
    for attempt in range(3):
        responses.extend(
            [
                response(
                    f"patch-{attempt}",
                    "apply_patch",
                    path="pricing.py",
                    old_text="if is_member or subtotal >= 100:",
                    new_text="if is_member and subtotal >= 100:",
                ),
                response(
                    f"finish-{attempt}",
                    "finish",
                    status="completed",
                    summary="fixed",
                    evidence="updated condition",
                ),
            ]
        )
    summary, run_dir = EvalRunner(
        provider=ScriptedProvider(responses),
        limits=BudgetLimits(max_seconds=60, max_tokens=1000),
        output_root=tmp_path / "repeat-runs",
    ).run(suite, suite.select_stage("a"), repeats=3)

    assert summary.tasks == 1
    assert summary.attempts == 3
    assert summary.pass_at_1 == 1.0
    assert summary.pass_at_3 == 1.0
    assert [result["attempt"] for result in summary.task_results] == [1, 2, 3]
    assert len((run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert not (run_dir / "workspaces").exists()


def test_auth_failure_is_environment_with_unknown_usage(tmp_path: Path) -> None:
    class AuthFailingProvider:
        model_id = "moonshot/kimi-k3"

        def complete(self, messages, tools, *, timeout_seconds):
            del messages, tools, timeout_seconds
            raise RuntimeError("AuthenticationError: Invalid Authentication")

    suite = EvalSuite.load(default_suite_path())
    summary, _ = EvalRunner(
        provider=AuthFailingProvider(),
        limits=BudgetLimits(max_seconds=60, max_tokens=1000),
        output_root=tmp_path / "runs",
    ).run(suite, suite.select_stage("a"))
    task = summary.task_results[0]
    assert task["failure_category"] == FailureCategory.ENVIRONMENT.value
    assert task["input_tokens"] is None
    assert task["output_tokens"] is None
    assert task["total_cost_usd"] is None
    assert task["final_diff"] == ""


def test_systemic_environment_failure_stops_remaining_attempts(tmp_path: Path) -> None:
    class AlwaysFailingProvider:
        model_id = "mock/environment-failure"

        def complete(self, messages, tools, *, timeout_seconds):
            del messages, tools, timeout_seconds
            raise RuntimeError("ConnectionError: provider unavailable")

    suite = EvalSuite.load(default_suite_path())
    summary, _ = EvalRunner(
        provider=AlwaysFailingProvider(),
        limits=BudgetLimits(max_seconds=60, max_tokens=1000),
        output_root=tmp_path / "systemic-stop",
    ).run(
        suite,
        suite.select_stage("a"),
        repeats=3,
        stop_on_systemic_failure=True,
    )

    assert summary.stopped_early is True
    assert summary.attempts == 2
    assert summary.planned_attempts == 3
    assert "environment_eval_failure" in (summary.stop_reason or "")


def _result(
    task_id: str,
    *,
    success: bool,
    terminal: str,
    cost: float | None,
    tokens: int | None = 30,
    attempt: int = 1,
    difficulty: str = "medium",
) -> EvalTaskResult:
    return EvalTaskResult(
        task_id=task_id,
        description="task",
        model="mock/model",
        provider="mock",
        success=success,
        terminal_state=terminal,
        stop_reason="model_finish_tool",
        verifier=VerifierResult("verify", success, 0 if success else 1, "", "", 0.1),
        failure_category=(
            FailureCategory.NONE.value if success else FailureCategory.MODEL.value
        ),
        failure_detail=None,
        expected_base_sha="a" * 40,
        actual_base_sha="a" * 40,
        initial_dirty=False,
        model_calls=2,
        tool_calls=3,
        steps=2,
        input_tokens=tokens,
        output_tokens=0 if tokens is not None else None,
        total_tokens=tokens,
        cached_tokens=None,
        reasoning_tokens=None,
        total_cost_usd=cost,
        cost_sources=("litellm_calculated",) if cost is not None else ("unknown",),
        wall_time_seconds=2.0,
        final_diff="diff",
        final_status=" M file.py",
        trajectory_path="trajectory.jsonl",
        attempt=attempt,
        difficulty=difficulty,
    )


def test_aggregation_cost_per_solved_and_solved_zero() -> None:
    summary = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [
            _result("one", success=True, terminal="completed", cost=0.2),
            _result("two", success=False, terminal="failed", cost=0.4),
        ],
    )
    assert summary.total_cost_usd == pytest.approx(0.6)
    assert summary.average_cost_per_task_usd == pytest.approx(0.3)
    assert summary.cost_per_solved_task_usd == pytest.approx(0.6)

    zero = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [_result("none", success=False, terminal="failed", cost=0.1)],
    )
    assert zero.solved == 0
    assert zero.cost_per_solved_task_usd is None
    assert zero.average_tokens_per_solved_task is None


def test_unknown_cost_and_token_propagate_to_summary() -> None:
    summary = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [
            _result("known", success=True, terminal="completed", cost=0.1),
            _result(
                "unknown", success=False, terminal="failed", cost=None, tokens=None
            ),
        ],
    )
    assert summary.total_cost_usd is None
    assert summary.total_tokens is None
    assert summary.average_cost_per_task_usd is None


def test_repeat_aggregation_pass_at_1_pass_at_3_and_per_solved_metrics() -> None:
    summary = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [
            _result(
                "one", success=False, terminal="failed", cost=0.1, tokens=10, attempt=1
            ),
            _result(
                "one",
                success=True,
                terminal="completed",
                cost=0.2,
                tokens=20,
                attempt=2,
            ),
            _result(
                "one", success=False, terminal="failed", cost=0.1, tokens=10, attempt=3
            ),
            _result(
                "two",
                success=True,
                terminal="completed",
                cost=0.2,
                tokens=20,
                attempt=1,
            ),
            _result(
                "two", success=False, terminal="failed", cost=0.1, tokens=10, attempt=2
            ),
            _result(
                "two", success=False, terminal="failed", cost=0.1, tokens=10, attempt=3
            ),
        ],
    )

    assert summary.tasks == 2
    assert summary.attempts == 6
    assert summary.repeats == 3
    assert summary.solved == 2
    assert summary.pass_at_1 == pytest.approx(0.5)
    assert summary.pass_at_3 == pytest.approx(1.0)
    assert summary.total_tokens == 80
    assert summary.tokens_per_solved_task == pytest.approx(40)
    assert summary.total_cost_usd == pytest.approx(0.8)
    assert summary.cost_per_solved_task_usd == pytest.approx(0.4)
    assert summary.difficulty_metrics["medium"]["pass_at_3"] == pytest.approx(1.0)


def test_two_repeats_leave_pass_at_3_unknown() -> None:
    summary = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [
            _result("one", success=False, terminal="failed", cost=0.1, attempt=1),
            _result("one", success=True, terminal="completed", cost=0.1, attempt=2),
        ],
    )
    assert summary.pass_at_1 == 0.0
    assert summary.pass_at_3 is None


def test_secret_redaction_covers_trajectory_and_authorization(tmp_path: Path) -> None:
    secret = "temporary-secret-value"
    store = TrajectoryStore(
        tmp_path, run_id="redaction", redactor=SecretRedactor((secret,))
    )
    store.append(
        "error",
        {
            "message": f"request failed for {secret}",
            "header": f"Authorization: Bearer {secret}",
        },
    )
    text = store.path.read_text(encoding="utf-8")
    assert secret not in text
    assert "[REDACTED]" in text
