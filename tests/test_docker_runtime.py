from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from forgeloop.budget import BudgetLimits
from forgeloop.evals import EvalRunner, EvalSuite, default_suite_path
from forgeloop.runtime import DockerRuntime, LocalRuntime
from forgeloop.tools.builtin import (
    ApplyPatchTool,
    GitDiffTool,
    ReadFileTool,
    SearchFilesTool,
    ShellTool,
)
from forgeloop.types import ModelResponse, ModelUsage, ToolCall
from forgeloop.workspace import Workspace

docker_required = pytest.mark.skipif(
    not DockerRuntime.available(), reason="Docker Engine is not available"
)


class SmokeProvider:
    model_id = "mock/docker-smoke"

    def __init__(self, attempts: int = 1) -> None:
        responses = []
        for attempt in range(attempts):
            responses.extend(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                f"patch-{attempt}",
                                "apply_patch",
                                {
                                    "path": "pricing.py",
                                    "old_text": "if is_member or subtotal >= 100:",
                                    "new_text": "if is_member and subtotal >= 100:",
                                },
                            ),
                        ),
                        usage=ModelUsage(input_tokens=10, output_tokens=5),
                    ),
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                f"finish-{attempt}",
                                "finish",
                                {
                                    "status": "completed",
                                    "summary": "fixed",
                                    "evidence": "verifier",
                                },
                            ),
                        ),
                        usage=ModelUsage(input_tokens=10, output_tokens=5),
                    ),
                ]
            )
        self._responses = iter(responses)

    def complete(self, messages, tools, *, timeout_seconds):
        del messages, tools, timeout_seconds
        return next(self._responses)


class FailingProvider:
    model_id = "mock/docker-failure"

    def complete(self, messages, tools, *, timeout_seconds):
        del messages, tools, timeout_seconds
        raise RuntimeError("provider unavailable")


def _init_repository(root: Path) -> None:
    (root / "code.py").write_text("alpha\nbeta\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.name", "ForgeLoop Test"], cwd=root, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@forgeloop.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True)


def _eval_containers() -> set[str]:
    docker = DockerRuntime._find_docker()
    result = subprocess.run(
        [
            docker,
            "container",
            "list",
            "--all",
            "--filter",
            "label=forgeloop.eval=true",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=DockerRuntime._docker_env(docker),
    )
    return set(result.stdout.splitlines())


@docker_required
@pytest.mark.docker
def test_docker_runtime_routes_all_workspace_tools_and_removes_container(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    workspace = Workspace(tmp_path)
    runtime = DockerRuntime()
    container_name: str | None = None
    try:
        runtime.start(workspace.root)
        container_name = runtime.container_name
        read = ReadFileTool(workspace, runtime).execute(
            {"path": "code.py", "start_line": 2}, timeout_seconds=5
        )
        search = SearchFilesTool(workspace, runtime).execute(
            {"pattern": "beta", "glob": "*.py"}, timeout_seconds=5
        )
        patch = ApplyPatchTool(workspace, runtime).execute(
            {"path": "code.py", "old_text": "beta", "new_text": "gamma"},
            timeout_seconds=5,
        )
        shell = ShellTool(workspace, runtime).execute(
            {"command": "python -m py_compile code.py"}, timeout_seconds=10
        )
        diff = GitDiffTool(workspace, runtime).execute({}, timeout_seconds=10)
    finally:
        runtime.close()

    assert read.ok and read.output == "2: beta"
    assert search.ok and "code.py:2:beta" in search.output
    assert patch.ok and shell.ok
    assert diff.ok and "gamma" in diff.output
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert container_name
    docker = DockerRuntime._find_docker()
    inspect = subprocess.run(
        [docker, "container", "inspect", container_name],
        capture_output=True,
        check=False,
        env=DockerRuntime._docker_env(docker),
    )
    assert inspect.returncode != 0


@docker_required
@pytest.mark.docker
def test_same_smoke_task_runs_local_and_repeated_docker_from_fixed_sha(
    tmp_path: Path,
) -> None:
    suite = EvalSuite.load(default_suite_path())
    task = suite.select_stage("a")
    configurations = [(LocalRuntime, 1), (DockerRuntime, 2)]
    results = []

    for index, (factory, repeats) in enumerate(configurations):
        summary, run_dir = EvalRunner(
            provider=SmokeProvider(attempts=repeats),
            limits=BudgetLimits(max_seconds=60, max_tokens=1000),
            output_root=tmp_path / f"run-{index}",
            runtime_factory=factory,
        ).run(suite, task, repeats=repeats)
        results.extend(summary.task_results)
        for result in summary.task_results:
            trajectory = Path(result["trajectory_path"])
            event_types = {
                json.loads(line)["type"]
                for line in trajectory.read_text(encoding="utf-8").splitlines()
            }
            assert "eval_runtime_started" in event_types
            assert "eval_runtime_stopped" in event_types
        assert not (run_dir / "workspaces").exists()

    assert all(result["success"] for result in results)
    assert all(result["actual_base_sha"] == task[0].base_commit for result in results)
    assert all(result["initial_dirty"] is False for result in results)
    assert (
        results[0]["final_diff"] == results[1]["final_diff"] == results[2]["final_diff"]
    )


@docker_required
@pytest.mark.docker
def test_eval_runner_removes_docker_container_after_provider_failure(
    tmp_path: Path,
) -> None:
    before = _eval_containers()
    suite = EvalSuite.load(default_suite_path())
    summary, _ = EvalRunner(
        provider=FailingProvider(),
        limits=BudgetLimits(max_seconds=60, max_tokens=1000),
        output_root=tmp_path / "failure-run",
        runtime_factory=DockerRuntime,
    ).run(suite, suite.select_stage("a"))

    assert summary.failed == 1
    assert _eval_containers() == before
