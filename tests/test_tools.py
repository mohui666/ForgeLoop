from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from forgeloop.runtime import DockerRuntime, LocalRuntime
from forgeloop.tools.builtin import (
    ApplyPatchTool,
    GitDiffTool,
    ReadFileTool,
    SearchFilesTool,
    ShellTool,
)
from forgeloop.workspace import Workspace, WorkspaceError


def test_workspace_rejects_escape(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    with pytest.raises(WorkspaceError, match="escapes"):
        workspace.resolve("../outside.txt")


def test_read_search_and_patch(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    (tmp_path / "code.py").write_text("alpha\nbeta\n", encoding="utf-8")

    read = ReadFileTool(workspace).execute(
        {"path": "code.py", "start_line": 2}, timeout_seconds=1
    )
    search = SearchFilesTool(workspace).execute(
        {"pattern": "beta", "glob": "*.py"}, timeout_seconds=5
    )
    patch = ApplyPatchTool(workspace).execute(
        {"path": "code.py", "old_text": "beta", "new_text": "gamma"}, timeout_seconds=1
    )

    assert read.ok and read.output == "2: beta"
    assert search.ok and "code.py:2:beta" in search.output
    assert patch.ok
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "alpha\ngamma\n"


def test_patch_refuses_ambiguous_replacement(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    (tmp_path / "a.txt").write_text("x x", encoding="utf-8")
    result = ApplyPatchTool(workspace).execute(
        {"path": "a.txt", "old_text": "x", "new_text": "y"}, timeout_seconds=1
    )
    assert not result.ok
    assert "matched 2 times" in result.output


def test_patch_preserves_crlf(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    (tmp_path / "windows.txt").write_bytes(b"first\r\nsecond\r\n")
    result = ApplyPatchTool(workspace).execute(
        {"path": "windows.txt", "old_text": "first\nsecond", "new_text": "one\ntwo"},
        timeout_seconds=1,
    )
    assert result.ok
    assert (tmp_path / "windows.txt").read_bytes() == b"one\r\ntwo\r\n"


def test_shell_and_git_diff(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    workspace = Workspace(tmp_path)
    runtime = LocalRuntime()
    shell = ShellTool(workspace, runtime).execute(
        {"command": "Set-Content -LiteralPath created.txt -Value hello"},
        timeout_seconds=10,
    )
    diff = GitDiffTool(workspace, runtime).execute({}, timeout_seconds=10)
    assert shell.ok
    assert diff.ok
    assert "?? created.txt" in diff.output
    snapshot = workspace.git_snapshot()
    assert snapshot.is_repository
    assert "?? created.txt" in snapshot.status


def test_shell_schema_describes_docker_shell(tmp_path: Path) -> None:
    schema = ShellTool(Workspace(tmp_path), DockerRuntime()).schema()
    description = schema["function"]["description"]

    assert "POSIX shell" in description
    assert "Docker container" in description
    assert "/bin/sh" in description
    assert "PowerShell commands" in description
    assert "unavailable" in description


@pytest.mark.skipif(os.name != "nt", reason="Windows LocalRuntime semantics")
def test_shell_schema_describes_windows_local_shell(tmp_path: Path) -> None:
    schema = ShellTool(Workspace(tmp_path), LocalRuntime()).schema()
    description = schema["function"]["description"]

    assert "Windows" in description
    assert "PowerShell" in description
    assert "Use PowerShell commands and variables" in description


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree semantics")
def test_local_runtime_timeout_terminates_child_process_tree(tmp_path: Path) -> None:
    script = tmp_path / "spawn_child.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'])\n"
        "open('child.pid', 'w', encoding='utf-8').write(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    result = LocalRuntime().run("python spawn_child.py", tmp_path, 1)
    elapsed = time.monotonic() - started

    assert result.timed_out
    assert result.exit_code == 124
    assert elapsed < 10
    child_pid = (tmp_path / "child.pid").read_text(encoding="utf-8")
    listing = subprocess.run(
        ["tasklist", "/FI", f"PID eq {child_pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert f'"{child_pid}"' not in listing.stdout
