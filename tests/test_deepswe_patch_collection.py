from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pier")

from forgeloop.deepswe import _audit_artifact_collection  # noqa: E402


@pytest.mark.docker
def test_pier_collects_delivery_patch_and_applies_in_clean_verifier(
    tmp_path: Path,
) -> None:
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    info = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, check=False, timeout=30
    )
    if info.returncode != 0:
        pytest.skip("Docker Engine is unavailable")

    task = Path("tests/fixtures/deepswe_patch_collection").resolve()
    jobs = tmp_path / "jobs"
    completed = subprocess.run(
        [
            "pier",
            "run",
            "--path",
            str(task),
            "--agent-import-path",
            "forgeloop.deepswe_fixture:DeterministicPatchCollectionAgent",
            "--env",
            "docker",
            "--n-attempts",
            "1",
            "--n-concurrent",
            "1",
            "--jobs-dir",
            str(jobs),
            "--job-name",
            "deterministic",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    job = jobs / "deterministic"
    trial = next(path.parent for path in job.rglob("result.json") if path.parent != job)
    result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    trajectory = next((trial / "agent" / "forgeloop-trajectories").glob("*.jsonl"))
    stdout = (trial / "verifier" / "test-stdout.txt").read_text(encoding="utf-8")
    audit = _audit_artifact_collection(
        trial,
        expected_base_sha="cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a",
        trajectory_path=trajectory,
        verifier_stdout=stdout,
        verifier_rewards=(result.get("verifier_result") or {}).get("rewards") or {},
    )

    assert audit.ok is True, audit.detail
    assert audit.patch_bytes > 0
    assert audit.manifest_status == "ok"
    assert audit.verifier_apply_bytes == audit.patch_bytes
    assert (result["verifier_result"]["rewards"]["reward"]) == 1.0
    assert "verifier observed ForgeLoop source state" in stdout
