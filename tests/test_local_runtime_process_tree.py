from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

import forgeloop.runtime as runtime_module
from forgeloop.runtime import LocalRuntime


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group behavior")
def test_local_timeout_kills_delayed_child_before_workspace_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "late.txt"
    child = (
        "import pathlib,time; time.sleep(0.5); "
        f"pathlib.Path({str(output)!r}).write_text('late', encoding='utf-8')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child)}"

    result = LocalRuntime().run(command, tmp_path, timeout_seconds=0.05)
    time.sleep(0.7)

    assert result.timed_out is True
    assert result.exit_code == 124
    assert not output.exists()


def test_posix_termination_targets_the_process_group(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        runtime_module.os, "getpgid", lambda pid: pid + 1, raising=False
    )
    monkeypatch.setattr(
        runtime_module.os,
        "killpg",
        lambda pgid, sig: calls.append((pgid, sig)),
        raising=False,
    )
    monkeypatch.setattr(runtime_module.signal, "SIGKILL", 9, raising=False)

    LocalRuntime._terminate_posix_process_group(321)

    assert calls == [(322, 9)]


def test_windows_local_process_does_not_request_a_posix_session(
    tmp_path: Path, monkeypatch
) -> None:
    if os.name != "nt":
        pytest.skip("Windows Popen configuration")
    captured: dict[str, object] = {}

    class FinishedProcess:
        returncode = 0

        def __init__(self, _argv, **kwargs) -> None:
            captured.update(kwargs)

        def wait(self, timeout: float) -> None:
            del timeout

    monkeypatch.setattr(runtime_module.subprocess, "Popen", FinishedProcess)

    result = LocalRuntime().run("Write-Output ok", tmp_path, timeout_seconds=1)

    assert result.exit_code == 0
    assert captured["start_new_session"] is False


@pytest.mark.skipif(os.name != "nt", reason="Windows taskkill fallback")
def test_windows_taskkill_failure_falls_back_to_process_kill(monkeypatch) -> None:
    killed: list[bool] = []

    class Process:
        pid = 321

        @staticmethod
        def kill() -> None:
            killed.append(True)

    monkeypatch.setattr(
        runtime_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )

    LocalRuntime._terminate_process_tree(Process())  # type: ignore[arg-type]

    assert killed == [True]
