from __future__ import annotations

import time
from dataclasses import dataclass

from forgeloop.runtime import Runtime
from forgeloop.workspace import Workspace


@dataclass(frozen=True)
class VerifierResult:
    command: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


@dataclass
class Verifier:
    runtime: Runtime

    def run(
        self, workspace: Workspace, command: str, timeout_seconds: float
    ) -> VerifierResult:
        started = time.perf_counter()
        result = self.runtime.run(command, workspace.root, timeout_seconds)
        return VerifierResult(
            command=command,
            passed=result.exit_code == 0,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=round(time.perf_counter() - started, 6),
            timed_out=result.timed_out,
        )
