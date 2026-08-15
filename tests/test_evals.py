from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

import pytest

from forgeloop import persistence
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


class _EvalFaultStream:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        short_write: bool = False,
        partial_write_failure: bool = False,
    ) -> None:
        self.stream = stream
        self.short_write = short_write
        self.partial_write_failure = partial_write_failure

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        self.stream.close()

    def write(self, data: bytes | memoryview) -> int:
        if self.partial_write_failure:
            self.partial_write_failure = False
            self.stream.write(data[: max(1, len(data) // 2)])
            raise OSError("injected task append failure")
        if self.short_write and len(data) > 1:
            data = data[: max(1, len(data) // 2)]
        return self.stream.write(data)

    def flush(self) -> None:
        self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.stream.seek(offset, whence)

    def tell(self) -> int:
        return self.stream.tell()

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)

    def truncate(self, size: int | None = None) -> int:
        return self.stream.truncate(size)


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
        limits=BudgetLimits(max_seconds=60),
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
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["schema_version"] == "forgeloop.eval-run-state.v1"
    assert state["status"] == "completed"
    assert state["planned_attempts"] == 1
    assert state["completed_attempts"] == 1
    assert state["run_id"] == summary.run_id
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
        limits=BudgetLimits(max_seconds=60),
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
        limits=BudgetLimits(max_seconds=60),
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
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "repeat-runs",
    ).run(suite, suite.select_stage("a"), repeats=3)

    assert summary.tasks == 1
    assert summary.attempts == 3
    assert summary.pass_at_1 == 1.0
    assert summary.pass_at_3 == 1.0
    assert [result["attempt"] for result in summary.task_results] == [1, 2, 3]
    assert len((run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert not (run_dir / "workspaces").exists()


def test_eval_runner_task_records_survive_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = EvalSuite.load(default_suite_path())
    runner = EvalRunner(
        provider=ScriptedProvider([]),
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "short-write-runs",
    )
    monkeypatch.setattr(
        runner,
        "_run_task",
        lambda task, attempt, trajectories, workspaces: replace(
            _result(task.id, success=True, terminal="completed", cost=0.1),
            attempt=attempt,
        ),
    )
    real_open = open

    def short_open(*args: object, **kwargs: object) -> _EvalFaultStream:
        return _EvalFaultStream(real_open(*args, **kwargs), short_write=True)

    monkeypatch.setattr(persistence, "open", short_open, raising=False)
    summary, run_dir = runner.run(suite, suite.select_stage("a"), repeats=2)

    records = [
        json.loads(line)
        for line in (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary.attempts == 2
    assert [record["attempt"] for record in records] == [1, 2]


def test_eval_runner_failed_task_append_preserves_complete_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = EvalSuite.load(default_suite_path())
    runner = EvalRunner(
        provider=ScriptedProvider([]),
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "append-failure-runs",
    )
    monkeypatch.setattr(
        runner,
        "_run_task",
        lambda task, attempt, trajectories, workspaces: replace(
            _result(task.id, success=True, terminal="completed", cost=0.1),
            attempt=attempt,
        ),
    )
    real_open = open
    opens = 0

    def failing_second_open(*args: object, **kwargs: object) -> _EvalFaultStream:
        nonlocal opens
        opens += 1
        return _EvalFaultStream(
            real_open(*args, **kwargs), partial_write_failure=opens == 2
        )

    monkeypatch.setattr(persistence, "open", failing_second_open, raising=False)
    with pytest.raises(OSError, match="injected task append failure"):
        runner.run(suite, suite.select_stage("a"), repeats=2)

    run_dir = next((tmp_path / "append-failure-runs").iterdir())
    lines = (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["attempt"] == 1
    assert not (run_dir / "summary.json").exists()
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["planned_attempts"] == 2
    assert state["completed_attempts"] == 1
    assert state["error_type"] == "OSError"
    assert state["error_message"] == "injected task append failure"


def test_eval_runner_summary_replace_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = EvalSuite.load(default_suite_path())
    runner = EvalRunner(
        provider=ScriptedProvider([]),
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "summary-failure-runs",
    )
    monkeypatch.setattr(
        runner,
        "_run_task",
        lambda task, attempt, trajectories, workspaces: _result(
            task.id, success=True, terminal="completed", cost=0.1
        ),
    )
    real_replace = persistence.os.replace

    def fail_summary_replace(source: object, target: object) -> None:
        if Path(target).name == "summary.json":
            raise OSError("injected summary replace failure")
        real_replace(source, target)

    monkeypatch.setattr(persistence.os, "replace", fail_summary_replace)

    with pytest.raises(OSError, match="injected summary replace failure"):
        runner.run(suite, suite.select_stage("a"))

    run_dir = next((tmp_path / "summary-failure-runs").iterdir())
    lines = (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["task_id"] == suite.select_stage("a")[0].id
    assert not (run_dir / "summary.json").exists()
    assert list(run_dir.glob(".summary.json.*.tmp")) == []
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["completed_attempts"] == 1
    assert state["error_type"] == "OSError"


def test_eval_runner_failed_state_redacts_error_without_swallowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = EvalSuite.load(default_suite_path())
    secret = "sk-test-lifecycle-secret"
    runner = EvalRunner(
        provider=ScriptedProvider([]),
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "redacted-failure-runs",
        redactor=SecretRedactor((secret,)),
    )

    def fail_task(*_args: object) -> EvalTaskResult:
        raise RuntimeError(f"provider credential {secret} was rejected")

    monkeypatch.setattr(runner, "_run_task", fail_task)

    with pytest.raises(RuntimeError, match=secret):
        runner.run(suite, suite.select_stage("a"))

    run_dir = next((tmp_path / "redacted-failure-runs").iterdir())
    state_text = (run_dir / "run-state.json").read_text(encoding="utf-8")
    state = json.loads(state_text)
    assert state["status"] == "failed"
    assert state["completed_attempts"] == 0
    assert state["error_type"] == "RuntimeError"
    assert secret not in state_text
    assert "[REDACTED]" in state["error_message"]
    assert not (run_dir / "tasks.jsonl").exists()
    assert not (run_dir / "summary.json").exists()


def test_eval_lifecycle_formatting_failure_does_not_replace_run_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = EvalSuite.load(default_suite_path())
    runner = EvalRunner(
        provider=ScriptedProvider([]),
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "broken-error-runs",
    )

    class BrokenStringError(RuntimeError):
        def __str__(self) -> str:
            raise ValueError("error formatting failed")

    original = BrokenStringError()

    def fail_task(*_args: object) -> EvalTaskResult:
        raise original

    monkeypatch.setattr(runner, "_run_task", fail_task)

    with pytest.raises(BrokenStringError) as caught:
        runner.run(suite, suite.select_stage("a"))

    assert caught.value is original
    run_dir = next((tmp_path / "broken-error-runs").iterdir())
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "running"
    assert state["completed_attempts"] == 0


def test_eval_interrupt_is_attributed_and_re_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = EvalSuite.load(default_suite_path())
    runner = EvalRunner(
        provider=ScriptedProvider([]),
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "interrupted-runs",
    )

    def interrupt(*_args: object) -> EvalTaskResult:
        raise KeyboardInterrupt("local stop")

    monkeypatch.setattr(runner, "_run_task", interrupt)

    with pytest.raises(KeyboardInterrupt, match="local stop"):
        runner.run(suite, suite.select_stage("a"))

    run_dir = next((tmp_path / "interrupted-runs").iterdir())
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["completed_attempts"] == 0
    assert state["error_type"] == "KeyboardInterrupt"
    assert state["ended_at"]


def test_auth_failure_is_environment_with_unknown_usage(tmp_path: Path) -> None:
    class AuthFailingProvider:
        model_id = "moonshot/kimi-k3"

        def complete(self, messages, tools, *, timeout_seconds):
            del messages, tools, timeout_seconds
            raise RuntimeError("AuthenticationError: Invalid Authentication")

    suite = EvalSuite.load(default_suite_path())
    summary, _ = EvalRunner(
        provider=AuthFailingProvider(),
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "runs",
    ).run(suite, suite.select_stage("a"))
    task = summary.task_results[0]
    assert task["failure_category"] == FailureCategory.ENVIRONMENT.value
    assert task["input_tokens"] is None
    assert task["output_tokens"] is None
    assert task["total_cost_usd"] is None
    assert task["final_diff"] == ""


def test_pre_agent_infrastructure_failure_has_known_zero_usage(
    tmp_path: Path, monkeypatch
) -> None:
    suite = EvalSuite.load(default_suite_path())
    runner = EvalRunner(
        provider=ScriptedProvider([]),
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "pre-agent-failure",
    )

    def fail_initial_state(runtime, workspace):
        del runtime, workspace
        raise EvalInfrastructureError("runtime status unavailable")

    monkeypatch.setattr(runner, "_runtime_initial_state", fail_initial_state)
    summary, _ = runner.run(suite, suite.select_stage("a"))
    task = summary.task_results[0]

    assert task["model_calls"] == 0
    assert task["input_tokens"] == 0
    assert task["output_tokens"] == 0
    assert task["total_tokens"] == 0
    assert task["total_cost_usd"] == 0.0
    assert summary.total_tokens == 0
    assert summary.total_cost_usd == 0.0


def test_post_agent_infrastructure_failure_preserves_known_usage(
    tmp_path: Path, monkeypatch
) -> None:
    suite = EvalSuite.load(default_suite_path())
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "finish",
                        "finish",
                        {
                            "status": "completed",
                            "summary": "done",
                            "evidence": "checked",
                        },
                    ),
                ),
                usage=ModelUsage(
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.001,
                    cost_source="provider_response",
                ),
            )
        ]
    )

    def fail_verifier(*args, **kwargs):
        del args, kwargs
        raise EvalInfrastructureError("verifier runtime unavailable")

    monkeypatch.setattr("forgeloop.evals.Verifier.run", fail_verifier)
    summary, _ = EvalRunner(
        provider=provider,
        limits=BudgetLimits(max_seconds=60),
        output_root=tmp_path / "post-agent-failure",
    ).run(suite, suite.select_stage("a"))
    task = summary.task_results[0]

    assert task["model_calls"] == 1
    assert task["total_tokens"] == 15
    assert task["total_cost_usd"] == 0.001
    assert summary.total_tokens == 15
    assert summary.total_cost_usd == 0.001


def test_systemic_environment_failure_stops_remaining_attempts(tmp_path: Path) -> None:
    class AlwaysFailingProvider:
        model_id = "mock/environment-failure"

        def complete(self, messages, tools, *, timeout_seconds):
            del messages, tools, timeout_seconds
            raise RuntimeError("ConnectionError: provider unavailable")

    suite = EvalSuite.load(default_suite_path())
    summary, _ = EvalRunner(
        provider=AlwaysFailingProvider(),
        limits=BudgetLimits(max_seconds=60),
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
    expected_outcome: str = "completed",
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
        expected_outcome=expected_outcome,
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


def test_aggregation_reports_weighted_warm_prefix_cache_rate() -> None:
    first = replace(
        _result("one", success=True, terminal="completed", cost=0.1),
        input_tokens=1_000,
        cached_tokens=800,
        cache_miss_tokens=200,
        warm_cache_reusable_tokens=900,
        warm_cache_reused_tokens=891,
        warm_cache_missed_tokens=9,
        warm_cache_significant_miss_calls=0,
    )
    second = replace(
        _result("two", success=True, terminal="completed", cost=0.1),
        input_tokens=2_000,
        cached_tokens=1_700,
        cache_miss_tokens=300,
        warm_cache_reusable_tokens=1_800,
        warm_cache_reused_tokens=1_764,
        warm_cache_missed_tokens=36,
        warm_cache_significant_miss_calls=1,
        warm_cache_reset_calls=1,
    )

    summary = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [first, second],
        min_warm_cache_hit_rate=0.98,
    )

    assert summary.total_cached_tokens == 2_500
    assert summary.total_cache_miss_tokens == 500
    assert summary.cached_input_ratio == pytest.approx(2_500 / 3_000)
    assert summary.warm_cache_reusable_tokens == 2_700
    assert summary.warm_cache_reused_tokens == 2_655
    assert summary.warm_cache_missed_tokens == 45
    assert summary.warm_cache_hit_ratio == pytest.approx(2_655 / 2_700)
    assert summary.warm_cache_significant_miss_calls == 1
    assert summary.warm_cache_reset_calls == 1
    assert summary.min_warm_cache_hit_rate == 0.98
    assert summary.warm_cache_gate_passed is True

    failed_gate = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [first, second],
        min_warm_cache_hit_rate=0.99,
    )
    assert failed_gate.warm_cache_gate_passed is False


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


def test_task_outcome_counts_do_not_overlap_across_repeats() -> None:
    summary = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [
            _result("solved", success=True, terminal="completed", cost=0.1, attempt=1),
            _result("solved", success=True, terminal="blocked", cost=0.1, attempt=2),
            _result(
                "blocked",
                success=True,
                terminal="blocked",
                cost=0.1,
                attempt=1,
                expected_outcome="blocked",
            ),
            _result("blocked", success=False, terminal="failed", cost=0.1, attempt=2),
        ],
    )

    assert summary.solved == 1
    assert summary.blocked == 1
    assert summary.failed == 0


def test_verifier_pass_is_solved_independently_of_terminal_state() -> None:
    summary = aggregate_results(
        "suite",
        "run",
        "mock/model",
        [_result("solved", success=True, terminal="blocked", cost=0.1)],
    )

    assert summary.solved == 1
    assert summary.blocked == 0
    assert summary.failed == 0
    assert summary.task_results[0]["terminal_state"] == "blocked"


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
