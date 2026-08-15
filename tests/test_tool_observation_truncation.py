from __future__ import annotations

from pathlib import Path

import pytest

from forgeloop.runtime import CommandResult, ShellEnvironment
from forgeloop.tools.builtin import GitInspectTool, ShellTool, ValidateTool
from forgeloop.workspace import Workspace


class MarkerlessTruncatingRuntime:
    shell_environment = ShellEnvironment(
        platform="Windows",
        executable="pwsh",
        syntax="PowerShell",
        location="offline test runtime",
    )

    def __init__(
        self, *, stdout_truncated: bool = False, stderr_truncated: bool = False
    ) -> None:
        self.stdout_truncated = stdout_truncated
        self.stderr_truncated = stderr_truncated

    def path_kind(self, path: Path) -> str:
        return "directory" if path.is_dir() else "missing"

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        del timeout_seconds
        return CommandResult(
            command=command,
            cwd=str(cwd),
            exit_code=0,
            stdout="markerless partial stdout",
            stderr="markerless partial stderr",
            stdout_truncated=self.stdout_truncated,
            stderr_truncated=self.stderr_truncated,
        )


@pytest.mark.parametrize("tool_type", [ShellTool, ValidateTool])
def test_shell_tools_warn_on_markerless_stdout_truncation_without_failing(
    tmp_path: Path, tool_type
) -> None:
    runtime = MarkerlessTruncatingRuntime(stdout_truncated=True)

    result = tool_type(Workspace(tmp_path), runtime).execute(
        {"command": "Write-Output partial"}, timeout_seconds=10
    )

    assert result.ok is True
    assert result.metadata["exit_code"] == 0
    assert result.metadata["stdout_truncated"] is True
    assert "markerless partial stdout" in result.output
    assert "output is incomplete" in result.output
    assert "runtime truncated stdout" in result.output


def test_git_inspect_warns_on_markerless_stderr_truncation_without_failing(
    tmp_path: Path,
) -> None:
    runtime = MarkerlessTruncatingRuntime(stderr_truncated=True)

    result = GitInspectTool(Workspace(tmp_path), runtime).execute(
        {"operation": "status"}, timeout_seconds=10
    )

    assert result.ok is True
    assert result.metadata["exit_code"] == 0
    assert result.metadata["stderr_truncated"] is True
    assert "markerless partial stderr" in result.output
    assert "output is incomplete" in result.output
    assert "runtime truncated stderr" in result.output
