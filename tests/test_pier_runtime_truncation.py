from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeloop.deepswe import DeepSWEError, PierRuntime
from forgeloop.runtime import CommandResult


class _FakePierRuntime(PierRuntime):
    def __init__(self, root: Path, *, stream: str | None) -> None:
        super().__init__(environment=None, loop=None)  # type: ignore[arg-type]
        self.workspace_root = root.resolve()
        self.stream = stream

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        del cwd, timeout_seconds
        if command.startswith("if [ -f"):
            stdout = "file\n"
        elif "enumerate" in command:
            stdout = json.dumps(["a.py:1:value", ".env:1:secret", "b.py:2:value"])
        else:
            stdout = json.dumps(["a.py", ".env", "credentials.json", "b.py"])
        return CommandResult(
            command=command,
            cwd="/app",
            exit_code=0,
            stdout=stdout,
            stderr="partial diagnostic" if self.stream == "stderr" else "",
            stdout_truncated=self.stream == "stdout",
            stderr_truncated=self.stream == "stderr",
        )


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_path_kind_fails_closed_on_structured_truncation(
    stream: str,
) -> None:
    root = Path.cwd()
    runtime = _FakePierRuntime(root, stream=stream)

    with pytest.raises(
        DeepSWEError, match=rf"Remote path inspection.*{stream}.*incomplete"
    ):
        runtime.path_kind(root)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_search_fails_closed_on_structured_truncation(stream: str) -> None:
    root = Path.cwd()
    runtime = _FakePierRuntime(root, stream=stream)

    result = runtime.search_text("value", root, "*.py", 10, 5)

    assert result.matches == ()
    assert result.error is not None
    assert "Remote search" in result.error
    assert stream in result.error
    assert "incomplete" in result.error


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_list_files_fails_closed_on_structured_truncation(stream: str) -> None:
    root = Path.cwd()
    runtime = _FakePierRuntime(root, stream=stream)

    with pytest.raises(
        DeepSWEError, match=rf"Remote file listing.*{stream}.*incomplete"
    ):
        runtime.list_files(root, "*.py", 10)


def test_structured_consumers_accept_complete_results() -> None:
    root = Path.cwd()
    runtime = _FakePierRuntime(root, stream=None)

    assert runtime.path_kind(root) == "file"
    assert runtime.search_text("value", root, "*.py", 10, 5).matches == (
        "a.py:1:value",
        "b.py:2:value",
    )
    assert runtime.list_files(root, "*.py", 10) == ("a.py", "b.py")


def test_path_kind_fails_closed_on_command_failure() -> None:
    root = Path.cwd()
    runtime = _FakePierRuntime(root, stream=None)
    original = runtime.run

    def failed(command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        result = original(command, cwd, timeout_seconds)
        return CommandResult(
            command=result.command,
            cwd=result.cwd,
            exit_code=2,
            stdout="",
            stderr="remote inspection failed",
        )

    runtime.run = failed  # type: ignore[method-assign]

    with pytest.raises(DeepSWEError, match="remote inspection failed"):
        runtime.path_kind(root)


def test_pier_legacy_truncation_format_remains_frozen() -> None:
    value = "x" * 40_001

    truncated = PierRuntime._truncate(value)

    assert truncated == "x" * 40_000 + "\n... <1 chars omitted>"
