from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from forgeloop.runtime import _CONTAINER_SEARCH_SCRIPT, LocalRuntime


def test_local_search_timeout_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("needle\n", encoding="utf-8")

    result = LocalRuntime().search_text(
        "needle", tmp_path, "*.py", max_results=100, timeout_seconds=0
    )

    assert result.error == "Search timed out"
    assert result.timed_out is True
    assert result.matches == ()


def test_local_search_timeout_preempts_pathological_regex(tmp_path: Path) -> None:
    (tmp_path / "slow.txt").write_text("a" * 32 + "!\n", encoding="utf-8")
    started = time.perf_counter()

    result = LocalRuntime().search_text(
        "(a+)+$", tmp_path, "*.txt", max_results=100, timeout_seconds=0.05
    )

    assert result.error == "Search timed out"
    assert result.timed_out is True
    assert result.matches == ()
    assert time.perf_counter() - started < 3


def test_container_search_script_filters_sensitive_recursive_candidates(
    tmp_path: Path,
) -> None:
    (tmp_path / "code.py").write_text("needle public\n", encoding="utf-8")
    (tmp_path / ".env").write_text("needle env-secret\n", encoding="utf-8")
    (tmp_path / "credentials.json").write_text(
        "needle credentials-secret\n", encoding="utf-8"
    )
    (tmp_path / "private.pem").write_text("needle pem-secret\n", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("needle git-secret\n", encoding="utf-8")
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
    payload = json.loads(completed.stdout)

    assert payload["error"] is None
    assert payload["matches"] == ["code.py:1:needle public"]
