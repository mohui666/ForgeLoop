from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import forgeloop.foundry as foundry_module
from forgeloop.foundry import FoundryBuilder, SourceTask
from forgeloop.runtime import CommandResult
from forgeloop.verifier import Verifier
from forgeloop.workspace import Workspace


class StaticRuntime:
    def __init__(self, result: CommandResult) -> None:
        self.result = result

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        del command, cwd, timeout_seconds
        return self.result


def test_verifier_propagates_runtime_truncation_without_changing_pass_fail(
    tmp_path: Path,
) -> None:
    runtime = StaticRuntime(
        CommandResult(
            command="check",
            cwd=str(tmp_path),
            exit_code=0,
            stdout="partial stdout",
            stderr="partial stderr",
            stdout_truncated=True,
            stderr_truncated=True,
        )
    )

    result = Verifier(runtime).run(Workspace(tmp_path), "check", 10)

    assert result.passed is True
    assert result.exit_code == 0
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_verifier_failure_remains_exit_code_driven_when_output_is_complete(
    tmp_path: Path,
) -> None:
    runtime = StaticRuntime(
        CommandResult(
            command="check",
            cwd=str(tmp_path),
            exit_code=3,
            stdout="complete stdout",
            stderr="complete stderr",
        )
    )

    result = Verifier(runtime).run(Workspace(tmp_path), "check", 10)

    assert result.passed is False
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False


def test_foundry_records_runtime_and_local_evidence_truncation(
    tmp_path: Path, monkeypatch
) -> None:
    stdout = "head" + "x" * 5000
    runtime_result = SimpleNamespace(
        exit_code=1,
        timed_out=False,
        stdout=stdout,
        stderr="runtime-capped stderr",
        stdout_truncated=False,
        stderr_truncated=True,
    )

    class FakeDockerRuntime:
        def __init__(self, *, image: str) -> None:
            assert image == "offline-image"

        def start(self, workspace: Path) -> None:
            assert workspace == tmp_path

        def run(self, command: str, workspace: Path, timeout_seconds: float):
            assert command == "offline-check"
            assert workspace == tmp_path
            assert timeout_seconds == 12
            return runtime_result

        def close(self) -> None:
            pass

    monkeypatch.setattr(foundry_module, "DockerRuntime", FakeDockerRuntime)
    task = SourceTask(
        id="offline",
        repository="unused",
        fix_commit="unused",
        source_pr=None,
        description="offline truncation test",
        test_paths=("tests/test_offline.py",),
        solution_paths=("offline.py",),
        verifier_command="offline-check",
        verifier_timeout_seconds=12,
        timeout_seconds=30,
        difficulty="small",
        tags=(),
    )

    evidence = FoundryBuilder._verify_in_container(task, tmp_path, "offline-image")

    assert evidence["exit_code"] == 1
    assert evidence["stdout"] == stdout[-4000:]
    assert evidence["stderr"] == "runtime-capped stderr"
    assert evidence["stdout_truncated"] is True
    assert evidence["stderr_truncated"] is True
