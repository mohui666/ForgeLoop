from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any

from forgeloop import __version__
from forgeloop.agent import AgentLoop, RunMode, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.models.base import ModelProvider
from forgeloop.runtime import LocalRuntime, Runtime
from forgeloop.security import SecretRedactor
from forgeloop.tools import build_default_tools
from forgeloop.trajectory import TrajectoryStore
from forgeloop.verifier import Verifier, VerifierResult
from forgeloop.workspace import Workspace

EVAL_SCHEMA_VERSION = "forgeloop.eval.v2"
FIXTURE_GIT_DATE = "2026-01-01T00:00:00+00:00"


class EvalInfrastructureError(RuntimeError):
    pass


class FailureCategory(str, Enum):
    NONE = "none"
    MODEL = "model_failure"
    HARNESS = "forgeloop_harness_failure"
    ENVIRONMENT = "environment_eval_failure"


@dataclass(frozen=True)
class EvalTask:
    id: str
    description: str
    fixture: Path
    base_commit: str
    mode: RunMode
    verifier_command: str
    verifier_timeout_seconds: float
    timeout_seconds: float
    expected_outcome: RunStatus
    tags: tuple[str, ...]
    stage: str
    difficulty: str
    source_repository: str | None = None
    source_pr: str | None = None
    source_commit: str | None = None
    source_base_sha: str | None = None
    docker_image: str | None = None
    dockerfile: Path | None = None
    docker_build_context: Path | None = None


@dataclass(frozen=True)
class EvalSuite:
    id: str
    tasks: tuple[EvalTask, ...]
    source_path: Path
    kind: str = "benchmark"

    @classmethod
    def load(cls, path: Path) -> EvalSuite:
        source = path.expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("schema_version") != EVAL_SCHEMA_VERSION:
            raise ValueError(f"Unsupported eval schema: {data.get('schema_version')}")
        tasks: list[EvalTask] = []
        seen: set[str] = set()
        for raw in data.get("tasks", []):
            task_id = str(raw["id"])
            if task_id in seen:
                raise ValueError(f"Duplicate eval task id: {task_id}")
            seen.add(task_id)
            fixture = (source.parent / raw["fixture"]).resolve()
            try:
                fixture.relative_to(source.parent)
            except ValueError as exc:
                raise ValueError(f"Fixture escapes suite directory: {fixture}") from exc
            verifier = raw["verifier"]
            source_info = raw.get("source", {})
            docker_info = raw.get("docker", {})

            def suite_path(value: str | None, label: str) -> Path | None:
                if not value:
                    return None
                resolved = (source.parent / value).resolve()
                try:
                    resolved.relative_to(source.parent)
                except ValueError as exc:
                    raise ValueError(
                        f"{label} escapes suite directory: {resolved}"
                    ) from exc
                return resolved

            tasks.append(
                EvalTask(
                    id=task_id,
                    description=str(raw["description"]),
                    fixture=fixture,
                    base_commit=str(raw["base_commit"]),
                    mode=RunMode(raw["mode"]),
                    verifier_command=str(verifier["command"]),
                    verifier_timeout_seconds=float(verifier.get("timeout_seconds", 30)),
                    timeout_seconds=float(raw.get("timeout_seconds", 300)),
                    expected_outcome=RunStatus(
                        raw.get("expected_outcome", "completed")
                    ),
                    tags=tuple(str(tag) for tag in raw.get("tags", [])),
                    stage=str(raw.get("stage", "c")).lower(),
                    difficulty=str(raw.get("difficulty", "medium")).lower(),
                    source_repository=source_info.get("repository"),
                    source_pr=source_info.get("pr"),
                    source_commit=source_info.get("fix_commit"),
                    source_base_sha=source_info.get("base_sha"),
                    docker_image=docker_info.get("image"),
                    dockerfile=suite_path(docker_info.get("dockerfile"), "Dockerfile"),
                    docker_build_context=suite_path(
                        docker_info.get("build_context"), "Docker build context"
                    ),
                )
            )
        kind = str(data.get("suite_kind", "benchmark")).lower()
        if kind == "smoke" and not 30 <= len(tasks) <= 50:
            raise ValueError("Smoke eval suite must contain 30-50 tasks")
        if not tasks:
            raise ValueError("Eval suite must contain at least one task")
        invalid_difficulties = {
            task.difficulty
            for task in tasks
            if task.difficulty not in {"easy", "medium", "hard"}
        }
        if invalid_difficulties:
            raise ValueError(
                "Unsupported task difficulty: "
                + ", ".join(sorted(invalid_difficulties))
            )
        return cls(str(data["suite_id"]), tuple(tasks), source, kind)

    def select_stage(self, stage: str) -> tuple[EvalTask, ...]:
        normalized = stage.lower()
        if normalized == "a":
            return tuple(task for task in self.tasks if task.stage == "a")
        if normalized == "b":
            return tuple(task for task in self.tasks if task.stage == "b")
        if normalized in {"c", "all"}:
            return self.tasks
        raise ValueError("stage must be a, b, or c")


@dataclass(frozen=True)
class EvalTaskResult:
    task_id: str
    description: str
    model: str
    provider: str | None
    success: bool
    terminal_state: str
    stop_reason: str
    verifier: VerifierResult | None
    failure_category: str
    failure_detail: str | None
    expected_base_sha: str
    actual_base_sha: str | None
    initial_dirty: bool | None
    model_calls: int
    tool_calls: int
    steps: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    total_cost_usd: float | None
    cost_sources: tuple[str, ...]
    wall_time_seconds: float
    final_diff: str
    final_status: str
    trajectory_path: str | None
    attempt: int = 1
    difficulty: str = "medium"
    expected_outcome: str = RunStatus.COMPLETED.value

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class EvalSummary:
    schema_version: str
    suite_id: str
    run_id: str
    model: str
    provider: str | None
    forgeloop_version: str
    litellm_version: str | None
    tasks: int
    attempts: int
    repeats: int
    planned_tasks: int
    planned_attempts: int
    stopped_early: bool
    stop_reason: str | None
    solved: int
    failed: int
    blocked: int
    budget_exceeded: int
    pass_rate: float
    pass_at_1: float
    pass_at_3: float | None
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_tokens: int | None
    average_tokens_per_task: float | None
    average_tokens_per_solved_task: float | None
    tokens_per_solved_task: float | None
    total_cost_usd: float | None
    average_cost_per_task_usd: float | None
    cost_per_solved_task_usd: float | None
    average_steps: float
    average_model_calls: float
    average_tool_calls: float
    average_wall_time_seconds: float
    failure_categories: dict[str, int]
    difficulty_metrics: dict[str, dict[str, Any]]
    task_results: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


class FixtureRepository:
    def prepare(self, task: EvalTask, destination: Path) -> str:
        head = self.materialize(task.id, task.fixture, destination)
        if head != task.base_commit:
            raise EvalInfrastructureError(
                f"Base SHA mismatch for {task.id}: expected {task.base_commit}, got {head}"
            )
        return head

    def materialize(self, task_id: str, fixture: Path, destination: Path) -> str:
        """Copy fixture bytes and create the deterministic eval baseline commit."""
        if destination.exists():
            raise EvalInfrastructureError(f"Workspace already exists: {destination}")
        if not fixture.is_dir():
            raise EvalInfrastructureError(f"Fixture does not exist: {fixture}")
        shutil.copytree(
            fixture,
            destination,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".git"
            ),
        )
        self._git(destination, "init", "-q", "--initial-branch=main")
        self._git(destination, "config", "core.autocrlf", "false")
        exclude = destination / ".git" / "info" / "exclude"
        with exclude.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n__pycache__/\n*.py[cod]\n.pytest_cache/\n")
        self._git(destination, "add", "-A")
        env = os.environ.copy()
        env.update(
            {
                "GIT_AUTHOR_NAME": "ForgeLoop Eval",
                "GIT_AUTHOR_EMAIL": "eval@forgeloop.invalid",
                "GIT_COMMITTER_NAME": "ForgeLoop Eval",
                "GIT_COMMITTER_EMAIL": "eval@forgeloop.invalid",
                "GIT_AUTHOR_DATE": FIXTURE_GIT_DATE,
                "GIT_COMMITTER_DATE": FIXTURE_GIT_DATE,
            }
        )
        self._git(destination, "commit", "-q", "-m", f"fixture: {task_id}", env=env)
        head = self._git(destination, "rev-parse", "HEAD").stdout.strip()
        if self._git(destination, "status", "--porcelain").stdout.strip():
            raise EvalInfrastructureError(f"Fixture is dirty after reset: {task_id}")
        return head

    def final_diff(self, workspace: Path) -> tuple[str, str]:
        self._git(workspace, "add", "-N", ".", check=False)
        diff = self._git(workspace, "diff", "--binary", "HEAD", check=False).stdout
        status = self._git(
            workspace, "status", "--short", "--untracked-files=all", check=False
        ).stdout
        return diff, status

    @staticmethod
    def _git(
        cwd: Path,
        *arguments: str,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-C", str(cwd), *arguments],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            env=env,
            check=False,
        )
        if check and result.returncode != 0:
            raise EvalInfrastructureError(
                f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
            )
        return result


@dataclass
class EvalRunner:
    provider: ModelProvider
    limits: BudgetLimits
    output_root: Path
    runtime_factory: Callable[[], Runtime] | None = None
    task_runtime_factory: Callable[[EvalTask], Runtime] | None = None
    redactor: SecretRedactor | None = None
    keep_workspaces: bool = False

    def run(
        self,
        suite: EvalSuite,
        tasks: tuple[EvalTask, ...],
        *,
        repeats: int = 1,
        stop_on_systemic_failure: bool = False,
    ) -> tuple[EvalSummary, Path]:
        if not tasks:
            raise ValueError("No eval tasks selected")
        if not 1 <= repeats <= 3:
            raise ValueError("repeats must be between 1 and 3")
        run_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.output_root.resolve() / f"{timestamp}-{run_id[:8]}"
        trajectories = run_dir / "trajectories"
        workspaces = run_dir / "workspaces"
        trajectories.mkdir(parents=True)
        workspaces.mkdir(parents=True)
        results_path = run_dir / "tasks.jsonl"
        results: list[EvalTaskResult] = []
        systemic_signature: tuple[str, str | None] | None = None
        systemic_streak = 0
        stop_reason: str | None = None
        for task in tasks:
            for attempt in range(1, repeats + 1):
                result = self._run_task(task, attempt, trajectories, workspaces)
                results.append(result)
                with results_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(
                        json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
                    )
                if result.failure_category in {
                    FailureCategory.HARNESS.value,
                    FailureCategory.ENVIRONMENT.value,
                }:
                    signature = (result.failure_category, result.failure_detail)
                    systemic_streak = (
                        systemic_streak + 1 if signature == systemic_signature else 1
                    )
                    systemic_signature = signature
                else:
                    systemic_signature = None
                    systemic_streak = 0
                if stop_on_systemic_failure and systemic_streak >= 2:
                    stop_reason = (
                        "systemic failure: "
                        f"{result.failure_category}: {result.failure_detail or 'unknown'}"
                    )
                    break
            if stop_reason:
                break
        summary = aggregate_results(
            suite.id,
            run_id,
            self.provider.model_id,
            results,
            planned_tasks=len(tasks),
            planned_repeats=repeats,
            stop_reason=stop_reason,
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not self.keep_workspaces:
            self._remove_workspaces(workspaces, run_dir)
        return summary, run_dir

    @staticmethod
    def _remove_workspaces(workspaces: Path, run_dir: Path) -> None:
        target = workspaces.resolve()
        target.relative_to(run_dir.resolve())

        def make_writable(function, path, error_info):
            del error_info
            os.chmod(path, stat.S_IWRITE)
            function(path)

        shutil.rmtree(target, onerror=make_writable)

    def _run_task(
        self, task: EvalTask, attempt: int, trajectories: Path, workspaces: Path
    ) -> EvalTaskResult:
        started = time.perf_counter()
        workspace_path = workspaces / f"{task.id}-attempt-{attempt}"
        fixture_repo = FixtureRepository()
        actual_sha: str | None = None
        trajectory: TrajectoryStore | None = None
        runtime: Runtime | None = None
        run_result = None
        try:
            actual_sha = fixture_repo.prepare(task, workspace_path)
            workspace = Workspace(workspace_path)
            initial = workspace.git_snapshot()
            if not initial.is_repository or initial.status:
                raise EvalInfrastructureError("Initial repository state is not clean")
            trajectory = TrajectoryStore(
                trajectories,
                run_id=f"{task.id}-attempt-{attempt}-{uuid.uuid4().hex[:8]}",
                redactor=self.redactor,
            )
            if self.task_runtime_factory:
                runtime = self.task_runtime_factory(task)
            else:
                runtime = (
                    self.runtime_factory() if self.runtime_factory else LocalRuntime()
                )
            runtime.start(workspace.root)
            trajectory.append("eval_runtime_started", runtime.metadata)
            runtime_head, runtime_dirty = self._runtime_initial_state(
                runtime, workspace
            )
            if runtime_head != actual_sha:
                raise EvalInfrastructureError(
                    f"Runtime base SHA mismatch: expected {actual_sha}, got {runtime_head}"
                )
            if runtime_dirty:
                raise EvalInfrastructureError(
                    "Runtime initial repository state is dirty"
                )
            task_limits = replace(
                self.limits,
                max_seconds=min(self.limits.max_seconds, task.timeout_seconds),
            )
            agent = AgentLoop(
                provider=self.provider,
                tools=build_default_tools(workspace, runtime),
                workspace=workspace,
                trajectory=trajectory,
                limits=task_limits,
            )
            run_result = agent.run(task.mode, task.description)
            verifier = Verifier(runtime).run(
                workspace, task.verifier_command, task.verifier_timeout_seconds
            )
            final_diff, final_status = fixture_repo.final_diff(workspace_path)
            correct_block = (
                task.expected_outcome is RunStatus.BLOCKED
                and run_result.status is RunStatus.BLOCKED
                and verifier.passed
                and not final_diff.strip()
            )
            success = (
                verifier.passed
                if task.expected_outcome is RunStatus.COMPLETED
                else correct_block
            )
            failure_category, failure_detail = self._attribute_failure(
                success,
                run_result.status,
                run_result.stop_reason,
                run_result.summary,
                verifier,
            )
            trajectory.append("eval_verifier", verifier)
            trajectory.append(
                "eval_final_diff", {"diff": final_diff, "status": final_status}
            )
            usage = run_result.budget["usage"]
            task_result = EvalTaskResult(
                task_id=task.id,
                description=task.description,
                model=run_result.model or self.provider.model_id,
                provider=run_result.provider,
                success=success,
                terminal_state=run_result.status.value,
                stop_reason=run_result.stop_reason,
                verifier=verifier,
                failure_category=failure_category.value,
                failure_detail=failure_detail,
                expected_base_sha=task.base_commit,
                actual_base_sha=actual_sha,
                initial_dirty=False,
                model_calls=usage["model_calls"],
                tool_calls=usage["tool_calls"],
                steps=usage["steps"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                total_cost_usd=usage["cost_usd"],
                cost_sources=tuple(usage["cost_sources"] or ["unknown"]),
                wall_time_seconds=round(time.perf_counter() - started, 3),
                final_diff=final_diff,
                final_status=final_status,
                trajectory_path=str(trajectory.path),
                attempt=attempt,
                difficulty=task.difficulty,
                expected_outcome=task.expected_outcome.value,
            )
            runtime.close()
            trajectory.append(
                "eval_runtime_stopped", {"runtime": runtime.metadata["type"]}
            )
            runtime = None
            return task_result
        except Exception as exc:  # noqa: BLE001 - record infrastructure failures per task
            detail = self._redact(str(exc))
            if runtime is not None:
                try:
                    runtime_type = runtime.metadata["type"]
                    runtime.close()
                    if trajectory:
                        trajectory.append(
                            "eval_runtime_stopped", {"runtime": runtime_type}
                        )
                except Exception as cleanup_exc:  # noqa: BLE001 - report leaked containers
                    detail += "; runtime cleanup failed: " + self._redact(
                        str(cleanup_exc)
                    )
            if trajectory:
                trajectory.append(
                    "eval_infrastructure_error",
                    {"type": type(exc).__name__, "message": detail},
                )
            usage = (
                run_result.budget["usage"]
                if run_result is not None
                else {
                    "model_calls": 0,
                    "tool_calls": 0,
                    "steps": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_sources": [],
                }
            )
            return EvalTaskResult(
                task_id=task.id,
                description=task.description,
                model=self.provider.model_id,
                provider=self.provider.model_id.partition("/")[0],
                success=False,
                terminal_state="infrastructure_failed",
                stop_reason="eval_infrastructure_error",
                verifier=None,
                failure_category=FailureCategory.ENVIRONMENT.value,
                failure_detail=detail,
                expected_base_sha=task.base_commit,
                actual_base_sha=actual_sha,
                initial_dirty=None,
                model_calls=usage["model_calls"],
                tool_calls=usage["tool_calls"],
                steps=usage["steps"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                total_cost_usd=usage["cost_usd"],
                cost_sources=tuple(
                    usage["cost_sources"]
                    or ([] if usage["model_calls"] == 0 else ["unknown"])
                ),
                wall_time_seconds=round(time.perf_counter() - started, 3),
                final_diff="",
                final_status="",
                trajectory_path=str(trajectory.path) if trajectory else None,
                attempt=attempt,
                difficulty=task.difficulty,
                expected_outcome=task.expected_outcome.value,
            )

    @staticmethod
    def _runtime_initial_state(
        runtime: Runtime, workspace: Workspace
    ) -> tuple[str, bool]:
        head = runtime.run("git rev-parse HEAD", workspace.root, 15)
        if head.exit_code != 0:
            raise EvalInfrastructureError(
                "Runtime cannot read Git HEAD: "
                + (head.stderr.strip() or head.stdout.strip())
            )
        status = runtime.run(
            "git status --porcelain --untracked-files=all", workspace.root, 15
        )
        if status.exit_code != 0:
            raise EvalInfrastructureError(
                "Runtime cannot read Git status: "
                + (status.stderr.strip() or status.stdout.strip())
            )
        return head.stdout.strip(), bool(status.stdout.strip())

    def _redact(self, value: str) -> str:
        return (self.redactor or SecretRedactor()).redact_text(value)

    @staticmethod
    def _attribute_failure(
        success: bool,
        status: RunStatus,
        stop_reason: str,
        summary: str,
        verifier: VerifierResult,
    ) -> tuple[FailureCategory, str | None]:
        if success:
            return FailureCategory.NONE, None
        if stop_reason == "orchestration_error":
            environment_markers = (
                "AuthenticationError",
                "Invalid Authentication",
                "RateLimitError",
                "ConnectionError",
                "Timeout",
            )
            if any(marker in summary for marker in environment_markers):
                return FailureCategory.ENVIRONMENT, "Provider/API environment failed"
            return FailureCategory.HARNESS, "Agent/provider/tool orchestration failed"
        if verifier.timed_out:
            return FailureCategory.MODEL, "Verifier timed out after agent changes"
        if status is RunStatus.BUDGET_EXCEEDED:
            return FailureCategory.MODEL, "Agent exhausted its configured budget"
        if status is RunStatus.BLOCKED:
            return FailureCategory.MODEL, "Agent reported blocked"
        return FailureCategory.MODEL, "Objective verifier failed"


def aggregate_results(
    suite_id: str,
    run_id: str,
    model: str,
    results: list[EvalTaskResult],
    *,
    planned_tasks: int | None = None,
    planned_repeats: int | None = None,
    stop_reason: str | None = None,
) -> EvalSummary:
    grouped: dict[str, list[EvalTaskResult]] = {}
    for result in results:
        grouped.setdefault(result.task_id, []).append(result)
    for attempts in grouped.values():
        attempts.sort(key=lambda item: item.attempt)
    task_count = len(grouped)
    attempt_count = len(results)
    repeats = max((result.attempt for result in results), default=0)
    solved = sum(
        any(
            result.success and result.expected_outcome != RunStatus.BLOCKED.value
            for result in attempts
        )
        for attempts in grouped.values()
    )
    blocked = sum(
        not any(
            result.success and result.expected_outcome != RunStatus.BLOCKED.value
            for result in attempts
        )
        and any(
            result.success and result.expected_outcome == RunStatus.BLOCKED.value
            for result in attempts
        )
        for attempts in grouped.values()
    )
    budget_exceeded = sum(
        not any(result.success for result in attempts)
        and any(
            result.terminal_state == RunStatus.BUDGET_EXCEEDED.value
            for result in attempts
        )
        for attempts in grouped.values()
    )
    failed = task_count - solved - blocked - budget_exceeded
    pass_at_1 = (
        sum(attempts[0].success for attempts in grouped.values()) / task_count
        if task_count
        else 0.0
    )
    pass_at_3 = (
        sum(
            any(result.success for result in attempts if result.attempt <= 3)
            for attempts in grouped.values()
        )
        / task_count
        if task_count and repeats >= 3
        else None
    )
    total_input = _sum_known(result.input_tokens for result in results)
    total_output = _sum_known(result.output_tokens for result in results)
    total_tokens = _sum_known(result.total_tokens for result in results)
    total_cost = _sum_known(result.total_cost_usd for result in results)
    provider = next((result.provider for result in results if result.provider), None)
    categories: dict[str, int] = {}
    for result in results:
        categories[result.failure_category] = (
            categories.get(result.failure_category, 0) + 1
        )
    difficulty_metrics = {
        difficulty: _aggregate_difficulty(
            {
                task_id: attempts
                for task_id, attempts in grouped.items()
                if attempts[0].difficulty == difficulty
            },
            repeats,
        )
        for difficulty in ("easy", "medium", "hard")
    }
    return EvalSummary(
        schema_version=EVAL_SCHEMA_VERSION,
        suite_id=suite_id,
        run_id=run_id,
        model=model,
        provider=provider,
        forgeloop_version=__version__,
        litellm_version=_package_version("litellm"),
        tasks=task_count,
        attempts=attempt_count,
        repeats=repeats,
        planned_tasks=planned_tasks if planned_tasks is not None else task_count,
        planned_attempts=(
            planned_tasks * planned_repeats
            if planned_tasks is not None and planned_repeats is not None
            else attempt_count
        ),
        stopped_early=stop_reason is not None,
        stop_reason=stop_reason,
        solved=solved,
        failed=failed,
        blocked=blocked,
        budget_exceeded=budget_exceeded,
        pass_rate=solved / task_count if task_count else 0.0,
        pass_at_1=pass_at_1,
        pass_at_3=pass_at_3,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        average_tokens_per_task=_divide(total_tokens, task_count),
        average_tokens_per_solved_task=_divide(total_tokens, solved),
        tokens_per_solved_task=_divide(total_tokens, solved),
        total_cost_usd=total_cost,
        average_cost_per_task_usd=_divide(total_cost, task_count),
        cost_per_solved_task_usd=_divide(total_cost, solved),
        average_steps=sum(result.steps for result in results) / attempt_count,
        average_model_calls=sum(result.model_calls for result in results)
        / attempt_count,
        average_tool_calls=sum(result.tool_calls for result in results) / attempt_count,
        average_wall_time_seconds=sum(result.wall_time_seconds for result in results)
        / attempt_count,
        failure_categories=categories,
        difficulty_metrics=difficulty_metrics,
        task_results=tuple(result.to_dict() for result in results),
    )


def _aggregate_difficulty(
    grouped: dict[str, list[EvalTaskResult]], repeats: int
) -> dict[str, Any]:
    results = [result for attempts in grouped.values() for result in attempts]
    tasks = len(grouped)
    solved = sum(
        any(
            result.success and result.expected_outcome != RunStatus.BLOCKED.value
            for result in attempts
        )
        for attempts in grouped.values()
    )
    total_tokens = _sum_known(result.total_tokens for result in results)
    total_cost = _sum_known(result.total_cost_usd for result in results)
    categories: dict[str, int] = {}
    for result in results:
        categories[result.failure_category] = (
            categories.get(result.failure_category, 0) + 1
        )
    return {
        "tasks": tasks,
        "attempts": len(results),
        "solved": solved,
        "pass_at_1": (
            sum(attempts[0].success for attempts in grouped.values()) / tasks
            if tasks
            else None
        ),
        "pass_at_3": (
            sum(
                any(result.success for result in attempts if result.attempt <= 3)
                for attempts in grouped.values()
            )
            / tasks
            if tasks and repeats >= 3
            else None
        ),
        "total_tokens": total_tokens,
        "tokens_per_solved": _divide(total_tokens, solved),
        "total_cost_usd": total_cost,
        "cost_per_solved_usd": _divide(total_cost, solved),
        "failure_categories": categories,
    }


def default_suite_path() -> Path:
    return Path(__file__).parent / "eval_suite" / "smoke" / "tasks.json"


def resolve_suite_path(value: Path) -> Path:
    """Resolve a built-in suite alias without hiding arbitrary manifest paths."""
    alias = str(value).lower()
    if alias in {"smoke", "python-smoke-v1"}:
        return default_suite_path()
    if alias in {"real-swe", "real-swe-stage-a"}:
        generated = Path(".forgeloop/foundry/real-swe/tasks.json").resolve()
        if not generated.is_file():
            raise ValueError(
                "real-swe has not been built; run `forgeloop foundry build` first"
            )
        return generated
    return value


def _sum_known(values) -> int | float | None:
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return sum(materialized)


def _divide(value: float | None, denominator: int) -> float | None:
    if value is None or denominator == 0:
        return None
    return value / denominator


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
