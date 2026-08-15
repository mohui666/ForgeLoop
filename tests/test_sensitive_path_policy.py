from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import forgeloop.deepswe as deepswe
from forgeloop.deepswe import PierRuntime
from forgeloop.runtime import CommandResult, _CONTAINER_SEARCH_SCRIPT
from forgeloop.security import is_sensitive_path, sensitive_path_python_source


SENSITIVE_PATHS = (
    ".env",
    "nested/.env.local",
    "credential",
    "credentials.json",
    "keys/private-key",
    "keys/private.key",
    "keys/certificate.pem",
    ".git/config",
)


def _populate(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "public.py").write_text("needle public\n", encoding="utf-8")
    for relative in SENSITIVE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("needle secret\n", encoding="utf-8")


def test_isolated_predicate_matches_host_policy() -> None:
    namespace: dict[str, object] = {}
    exec(sensitive_path_python_source(), namespace)
    isolated = namespace["is_sensitive_path"]

    candidates = (*SENSITIVE_PATHS, "src/public.py", "keys/public.txt")
    assert [isolated(path) for path in candidates] == [  # type: ignore[operator]
        is_sensitive_path(path) for path in candidates
    ]


def test_standalone_search_filters_canonical_sensitive_paths(tmp_path: Path) -> None:
    _populate(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _CONTAINER_SEARCH_SCRIPT,
            str(tmp_path),
            ".",
            "needle",
            "",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert json.loads(completed.stdout)["matches"] == ["src/public.py:1:needle public"]


class _ExecutingPierRuntime(PierRuntime):
    def __init__(self, root: Path) -> None:
        super().__init__(environment=None, loop=None)  # type: ignore[arg-type]
        self.workspace_root = root.resolve()
        self.raw_stdout = ""

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        del cwd
        argv = shlex.split(command)
        completed = subprocess.run(
            [sys.executable, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        self.raw_stdout = completed.stdout
        return CommandResult(
            command=command,
            cwd=str(self.workspace_root),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@pytest.mark.parametrize("operation", ["search", "list"])
def test_pier_filters_sensitive_paths_before_host_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    _populate(tmp_path)
    monkeypatch.setattr(deepswe, "REMOTE_ROOT", tmp_path.as_posix())
    runtime = _ExecutingPierRuntime(tmp_path)

    if operation == "search":
        visible = runtime.search_text("needle", tmp_path, None, 100, 10).matches
        assert visible == ("src/public.py:1:needle public",)
    else:
        visible = runtime.list_files(tmp_path, "*", 100)
        assert visible == ("src/public.py",)

    # This is the raw remote process payload, before PierRuntime's defensive
    # host-side filter. Sensitive path names never cross that boundary.
    assert json.loads(runtime.raw_stdout) == list(visible)
    assert not any(name in runtime.raw_stdout for name in (".env", ".git", ".key"))
