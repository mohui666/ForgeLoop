from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import forgeloop.deepswe as deepswe
from forgeloop.deepswe import DeepSWEError
from forgeloop.verifier import VerifierResult


def _base_event() -> dict[str, object]:
    return {
        "schema_version": "forgeloop.trajectory.v2",
        "run_id": "offline-import",
        "sequence": 7,
        "timestamp": "2026-08-15T00:00:00+00:00",
        "type": "run_finished",
        "payload": {"status": "completed"},
    }


def _audit() -> SimpleNamespace:
    return SimpleNamespace(
        ok=True,
        status="ok",
        to_dict=lambda: {"ok": True, "status": "ok"},
    )


def _verifier() -> VerifierResult:
    return VerifierResult(
        command="offline verifier",
        passed=True,
        exit_code=0,
        stdout="PASS",
        stderr="",
        duration_seconds=0.1,
    )


def test_external_evidence_is_published_as_one_atomic_trajectory(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trace.jsonl"
    trajectory.write_text(json.dumps(_base_event()) + "\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.patch").write_text("diff --git a/a b/a\n", encoding="utf-8")

    deepswe._append_external_events(trajectory, _verifier(), tmp_path, _audit())

    events = [json.loads(line) for line in trajectory.read_text().splitlines()]
    assert [event["sequence"] for event in events] == [7, 8, 9, 10]
    assert [event["type"] for event in events[1:]] == [
        "eval_artifact_collection",
        "eval_verifier",
        "eval_final_diff",
    ]


def test_external_evidence_publish_failure_preserves_original_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = tmp_path / "trace.jsonl"
    original = (json.dumps(_base_event()) + "\n").encode()
    trajectory.write_bytes(original)

    def fail_publish(_path: Path, _text: str) -> None:
        raise OSError("injected atomic publish failure")

    monkeypatch.setattr(deepswe, "atomic_write_text", fail_publish)

    with pytest.raises(OSError, match="injected atomic publish failure"):
        deepswe._append_external_events(trajectory, _verifier(), tmp_path, _audit())

    assert trajectory.read_bytes() == original


def test_external_evidence_rejects_incomplete_trajectory_tail(tmp_path: Path) -> None:
    trajectory = tmp_path / "trace.jsonl"
    trajectory.write_text(json.dumps(_base_event()), encoding="utf-8")

    with pytest.raises(DeepSWEError, match="incomplete trajectory"):
        deepswe._append_external_events(trajectory, _verifier(), tmp_path, _audit())

    assert trajectory.read_text(encoding="utf-8") == json.dumps(_base_event())
