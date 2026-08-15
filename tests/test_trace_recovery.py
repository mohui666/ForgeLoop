from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeloop.trace import (
    TraceError,
    explain_trajectory,
    load_trajectory,
    replay_trajectory,
)


def _event(sequence: int, event_type: str, payload: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "schema_version": "forgeloop.trajectory.v2",
            "run_id": "offline-recovery",
            "sequence": sequence,
            "timestamp": "2026-08-15T00:00:00+00:00",
            "type": event_type,
            "payload": payload,
        }
    ).encode("utf-8")


def test_forensic_views_recover_complete_events_before_truncated_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(
        _event(0, "run_started", {"request": "offline task"})
        + b"\n"
        + _event(1, "tool_call", {"name": "read_file"})
        + b"\n"
        + b'{"schema_version":"forgeloop.trajectory.v2","sequence":2'
    )

    replay = replay_trajectory(path)
    explanation = explain_trajectory(path)

    assert "Trajectory: offline-recovery" in replay
    assert "Evidence warning: ignored an incomplete final trajectory record" in replay
    assert (
        "Evidence warning: ignored an incomplete final trajectory record" in explanation
    )
    assert "Termination: unknown/unknown" in explanation


def test_strict_loader_rejects_truncated_tail(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(_event(0, "run_started", {"request": "offline task"}) + b"\n{")

    with pytest.raises(TraceError, match="Invalid trajectory JSON.*:2"):
        load_trajectory(path)


def test_forensic_recovery_rejects_malformed_complete_record(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(
        _event(0, "run_started", {"request": "offline task"}) + b"\n{bad}\n"
    )

    with pytest.raises(TraceError, match="Invalid trajectory JSON.*:2"):
        replay_trajectory(path)


def test_forensic_recovery_handles_partial_utf8_character(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(
        _event(0, "run_started", {"request": "offline task"})
        + b"\n"
        + b'{"payload":"\xe4\xb8'
    )

    replay = replay_trajectory(path)

    assert "Evidence warning: ignored an incomplete final trajectory record" in replay


def test_forensic_recovery_requires_at_least_one_complete_event(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(b'{"sequence":0')

    with pytest.raises(TraceError, match="Invalid trajectory JSON.*:1"):
        replay_trajectory(path)
