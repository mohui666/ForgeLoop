from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from forgeloop.agent_types import RunStatus
from forgeloop.delivery import GitPatchDelivery
from forgeloop.runtime import CommandResult


_BASE_SHA = "a" * 40
_PATCH_SHA = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class _Snapshot:
    is_repository: bool = True
    head: str = _BASE_SHA
    branch: str = "main"
    status: str = ""


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def git_snapshot() -> _Snapshot:
        return _Snapshot()


class _FaultRuntime:
    def __init__(self, target: str, stream: str) -> None:
        self.target = target
        self.stream = stream
        self.commands: list[str] = []

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        del timeout_seconds
        self.commands.append(command)
        stdout = self._stdout(command)
        targeted = self.target in command
        return CommandResult(
            command=command,
            cwd=str(cwd),
            exit_code=0,
            stdout=stdout,
            stderr="partial diagnostic" if targeted and self.stream == "stderr" else "",
            stdout_truncated=targeted and self.stream == "stdout",
            stderr_truncated=targeted and self.stream == "stderr",
        )

    @staticmethod
    def _stdout(command: str) -> str:
        if "symbolic-ref" in command:
            return "main\n"
        if "status --porcelain" in command:
            return ""
        if "rev-parse HEAD" in command:
            return _BASE_SHA + "\n"
        if command.startswith("python -c"):
            return json.dumps({"bytes": 0, "sha256": _PATCH_SHA}) + "\n"
        raise AssertionError(f"Unexpected command: {command}")


@pytest.mark.parametrize(
    "target",
    [
        "symbolic-ref",
        "status --porcelain",
        "rev-parse HEAD",
        "python -c",
    ],
    ids=["branch", "status", "head", "patch-identity"],
)
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_delivery_fails_closed_on_truncated_control_output(
    target: str, stream: str
) -> None:
    runtime = _FaultRuntime(target, stream)
    workspace = _Workspace(Path("unused-offline-workspace"))
    delivery = GitPatchDelivery(runtime, require_patch_on_completed=False)
    delivery.start(workspace)

    result = delivery.deliver(workspace, RunStatus.FAILED)

    assert result.ok is False
    assert result.status == "delivery_failed"
    assert result.patch_bytes == 0
    assert "Incomplete command output while preparing patch delivery" in result.detail
    assert target in result.detail
