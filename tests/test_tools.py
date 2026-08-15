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
    _git_review_pathspec,
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
    (tmp_path / ".forgeloop").mkdir()
    (tmp_path / ".forgeloop" / "trajectory.jsonl").write_text(
        "internal\n", encoding="utf-8"
    )
    diff = GitDiffTool(workspace, runtime).execute({}, timeout_seconds=10)
    assert shell.ok
    assert diff.ok
    assert "?? created.txt" in diff.output
    assert "diff --git a/created.txt b/created.txt" in diff.output
    assert "+hello" in diff.output
    assert diff.metadata["review_scope"] == "worktree"
    assert diff.metadata["untracked_files"] == 1
    assert ".forgeloop" not in diff.output
    snapshot = workspace.git_snapshot()
    assert snapshot.is_repository
    assert "?? created.txt" in snapshot.status


def test_git_diff_marks_path_filtered_and_sensitive_reviews_incomplete(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    workspace = Workspace(tmp_path)
    runtime = LocalRuntime()
    (tmp_path / "created.txt").write_text("hello\n", encoding="utf-8")
    partial = GitDiffTool(workspace, runtime).execute(
        {"path": "created.txt"}, timeout_seconds=10
    )

    assert partial.ok
    assert partial.metadata["review_scope"] == "partial"
    assert "+hello" in partial.output

    (tmp_path / ".env").write_text("API_KEY=do-not-expose\n", encoding="utf-8")
    sensitive = GitDiffTool(workspace, runtime).execute({}, timeout_seconds=10)

    assert sensitive.ok
    assert sensitive.metadata["review_scope"] == "partial"
    assert "do-not-expose" not in sensitive.output
    assert "sensitive untracked content withheld" in sensitive.output


def test_git_diff_reviews_staged_changes_and_withholds_tracked_secrets(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "public.txt").write_text("before\n", encoding="utf-8")
    (tmp_path / ".env").write_text("placeholder\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ForgeLoop Tests",
            "-c",
            "user.email=tests@forgeloop.local",
            "commit",
            "-qm",
            "base",
        ],
        cwd=tmp_path,
        check=True,
    )
    workspace = Workspace(tmp_path)
    runtime = LocalRuntime()
    (tmp_path / "public.txt").write_text("staged-value\n", encoding="utf-8")
    subprocess.run(["git", "add", "public.txt"], cwd=tmp_path, check=True)

    staged = GitDiffTool(workspace, runtime).execute({}, timeout_seconds=10)

    assert staged.ok
    assert staged.metadata["review_scope"] == "worktree"
    assert "Staged changes:" in staged.output
    assert "+staged-value" in staged.output

    (tmp_path / ".env").write_text("NEVER_SHOW_TRACKED_SECRET\n", encoding="utf-8")
    sensitive = GitDiffTool(workspace, runtime).execute({}, timeout_seconds=10)

    assert sensitive.ok
    assert sensitive.metadata["review_scope"] == "partial"
    assert sensitive.metadata["tracked_sensitive_files_withheld"] == 1
    assert "NEVER_SHOW_TRACKED_SECRET" not in sensitive.output
    assert "Sensitive tracked diff content withheld: .env" in sensitive.output


def test_git_diff_path_filter_treats_shell_quotes_as_literal(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    quoted = "odd'name.txt"
    (tmp_path / quoted).write_text("literal path\n", encoding="utf-8")

    result = GitDiffTool(Workspace(tmp_path), LocalRuntime()).execute(
        {"path": quoted}, timeout_seconds=10
    )

    assert result.ok
    assert result.metadata["review_scope"] == "partial"
    assert "+literal path" in result.output


def test_git_diff_posix_pathspec_uses_shell_quoting_and_literal_magic() -> None:
    class PosixRuntime:
        shell_environment = type("Shell", (), {"syntax": "POSIX shell"})()

    pathspec = _git_review_pathspec(PosixRuntime(), "odd\\path':(top)name.txt")

    assert pathspec.startswith(" -- ")
    assert "odd\\path" in pathspec
    assert "odd/path" not in pathspec
    assert ":(literal)odd" in pathspec
    assert "\"'\"'" in pathspec
    assert ":(top)name.txt" in pathspec


def test_git_diff_marks_truncated_review_evidence_incomplete(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    workspace = Workspace(tmp_path)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ForgeLoop Tests",
            "-c",
            "user.email=tests@forgeloop.local",
            "commit",
            "-qm",
            "base",
        ],
        cwd=tmp_path,
        check=True,
    )
    tracked.write_text("changed\n" + "x" * 500 + "\n", encoding="utf-8")

    tracked_result = GitDiffTool(workspace, LocalRuntime(max_output_chars=120)).execute(
        {}, timeout_seconds=10
    )

    assert tracked_result.ok
    assert tracked_result.metadata["review_scope"] == "partial"
    assert tracked_result.metadata["output_complete"] is False
    assert "chars omitted" in tracked_result.output

    markerless = GitDiffTool(workspace, LocalRuntime(max_output_chars=5)).execute(
        {}, timeout_seconds=10
    )

    assert markerless.ok
    assert markerless.metadata["review_scope"] == "partial"
    assert markerless.metadata["output_complete"] is False
    assert markerless.metadata["stdout_truncated"] is True

    (tmp_path / "large untracked.txt").write_text("y" * 500, encoding="utf-8")
    untracked_result = GitDiffTool(
        workspace, LocalRuntime(), max_untracked_diff_chars=120
    ).execute({}, timeout_seconds=10)

    assert untracked_result.ok
    assert untracked_result.metadata["review_scope"] == "partial"
    assert untracked_result.metadata["output_complete"] is False
    assert "untracked diff truncated" in untracked_result.output


def test_command_output_truncation_preserves_head_and_tail(tmp_path: Path) -> None:
    result = LocalRuntime(max_output_chars=100).run(
        "python -c \"print('HEAD' + 'x' * 300 + 'TAIL')\"",
        tmp_path,
        10,
    )

    assert result.exit_code == 0
    assert result.stdout.startswith("HEAD")
    assert result.stdout.rstrip().endswith("TAIL")
    assert "chars omitted from middle" in result.stdout
    assert result.stdout_truncated is True
    assert result.stderr_truncated is False


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
