from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("pier")

from forgeloop.deepswe import (  # noqa: E402
    DEEPSWE_MAX_MODEL_CALLS,
    DEEPSWE_MAX_SECONDS,
    DEEPSWE_MAX_TOOL_CALLS,
    DEFAULT_SUBSET_PATH,
    DeepSWESubset,
    PierListFilesTool,
    _audit_artifact_collection,
    import_pier_results,
    pier_command,
    select_task_ids,
)


def test_frozen_subset_is_unique_and_reproducible() -> None:
    subset = DeepSWESubset.load(DEFAULT_SUBSET_PATH)

    assert len(subset.tasks) == 20
    assert len(set(subset.tasks)) == 20
    assert subset.revision == "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9"
    assert subset.pier_revision == "34c18f0e4eed88877c28721f5c5871a950bec637"
    assert select_task_ids(["c", "a", "b"], 7, 2) == select_task_ids(
        ["b", "c", "a"], 7, 2
    )


def test_pier_command_uses_official_task_path_and_fixed_single_attempt(
    tmp_path: Path,
) -> None:
    command = pier_command(
        tmp_path / "deep-swe",
        tmp_path / "jobs",
        "daily",
        ("one", "two"),
        "qwen3.5-4b-local",
    )

    assert command[1] == "run"
    assert "forgeloop.deepswe:ForgeLoopPierAgent" in command
    assert command.count("--include-task-name") == 2
    assert command[command.index("--n-attempts") + 1] == "1"
    assert command[command.index("--n-concurrent") + 1] == "1"
    kwargs = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--agent-kwarg"
    ]
    assert f"max_steps={DEEPSWE_MAX_MODEL_CALLS}" in kwargs
    assert f"max_model_calls={DEEPSWE_MAX_MODEL_CALLS}" in kwargs
    assert f"max_tool_calls={DEEPSWE_MAX_TOOL_CALLS}" in kwargs
    assert f"max_seconds={DEEPSWE_MAX_SECONDS:g}" in kwargs
    assert not any(value.startswith("max_tokens=") for value in kwargs)
    assert PierListFilesTool.max_results == 100


def test_import_pier_result_maps_verifier_and_trajectory(tmp_path: Path) -> None:
    subset = DeepSWESubset.load(DEFAULT_SUBSET_PATH)
    job_dir = tmp_path / "jobs" / "job-one"
    trial = job_dir / "trial-one"
    (trial / "agent" / "forgeloop-trajectories").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "artifacts").mkdir()
    trajectory = trial / "agent" / "forgeloop-trajectories" / "trace.jsonl"
    patch = b"diff --git a/a.py b/a.py\n"
    trajectory.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {
                    "schema_version": "forgeloop.trajectory.v2",
                    "run_id": "trace-one",
                    "sequence": 0,
                    "timestamp": "2026-08-09T00:00:01+00:00",
                    "type": "run_finished",
                    "payload": {},
                },
                {
                    "schema_version": "forgeloop.trajectory.v2",
                    "run_id": "trace-one",
                    "sequence": 1,
                    "timestamp": "2026-08-09T00:00:02+00:00",
                    "type": "patch_delivery",
                    "payload": {
                        "ok": True,
                        "has_patch": True,
                        "base_sha": "de139fd51c4d347666d109a8aea9d25451d908f6",
                        "head_sha": "a" * 40,
                        "branch": "forgeloop/deepswe-delivery-dd7ff13e",
                        "patch_bytes": len(patch.rstrip()),
                        "patch_sha256": hashlib.sha256(patch.rstrip()).hexdigest(),
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (trial / "verifier" / "test-stdout.txt").write_text(
        f"[verifier] model.patch applied ({len(patch)} bytes)\nofficial tests passed",
        encoding="utf-8",
    )
    (trial / "artifacts" / "model.patch").write_bytes(patch)
    (trial / "artifacts" / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": "/logs/artifacts/model.patch",
                    "destination": "artifacts/model.patch",
                    "type": "file",
                    "status": "ok",
                }
            ]
        ),
        encoding="utf-8",
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/mashumaro-flattened-dataclass-fields",
                "task_id": {
                    "path": str(
                        Path(".forgeloop/external/deep-swe/tasks")
                        / "mashumaro-flattened-dataclass-fields"
                    )
                },
                "started_at": "2026-08-09T00:00:00+00:00",
                "finished_at": "2026-08-09T00:00:10+00:00",
                "agent_result": {
                    "n_input_tokens": 100,
                    "n_output_tokens": 20,
                    "n_cache_tokens": 5,
                    "cost_usd": 0.0,
                    "n_agent_steps": 2,
                    "metadata": {
                        "forgeloop": {
                            "terminal_state": "completed",
                            "stop_reason": "finish_tool",
                            "model_calls": 2,
                            "tool_calls": 3,
                            "usage_complete": True,
                            "cost_sources": ["policy_local_zero"],
                        }
                    },
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
                "verifier": {
                    "started_at": "2026-08-09T00:00:08+00:00",
                    "finished_at": "2026-08-09T00:00:10+00:00",
                },
            }
        ),
        encoding="utf-8",
    )

    report = import_pier_results(
        job_dir, tmp_path / "reports", subset, "qwen3.5-4b-local"
    )

    assert report is not None
    task = json.loads((report / "tasks.jsonl").read_text(encoding="utf-8"))
    assert task["task_id"] == "mashumaro-flattened-dataclass-fields"
    assert task["success"] is True
    assert task["final_status"] == "patch_collected_verified"
    assert task["verifier"]["command"] == "DeepSWE official Pier verifier"
    assert task["total_cost_usd"] == 0.0
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    assert summary["solved"] == 1
    provenance = json.loads((report / "provenance.json").read_text(encoding="utf-8"))
    execution = provenance["execution_budget"]
    assert execution["schema_version"] == "forgeloop.execution-budget.v2"
    assert execution["cumulative_tokens"] == "telemetry_only"
    assert execution["limits"]["max_model_calls"] == DEEPSWE_MAX_MODEL_CALLS
    assert execution["limits"]["max_seconds"] == DEEPSWE_MAX_SECONDS
    assert "max_tokens" not in execution["limits"]
    collection = provenance["artifact_collection"]
    assert collection["fail_closed"] is True
    assert collection["tasks"][0]["status"] == "ok"
    assert provenance["pier_revision"] == subset.pier_revision
    guards = provenance["guard_semantics"]
    assert guards["schema_version"] == "forgeloop.long-horizon-guards.v1"
    assert guards["repeated_action"]["window"] == "contiguous"
    assert guards["repeated_action"]["hard_stop_streak"] == 4
    assert guards["repeated_error"]["terminal"] is False
    mapped_trace = Path(task["trajectory_path"])
    event_types = [
        json.loads(line)["type"]
        for line in mapped_trace.read_text(encoding="utf-8").splitlines()
    ]
    assert event_types[-3:] == [
        "eval_artifact_collection",
        "eval_verifier",
        "eval_final_diff",
    ]

    (trial / "artifacts" / "model.patch").unlink()
    (trial / "artifacts" / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": "/logs/artifacts/model.patch",
                    "destination": "artifacts/model.patch",
                    "type": "file",
                    "status": "failed",
                }
            ]
        ),
        encoding="utf-8",
    )
    (trial / "verifier" / "test-stdout.txt").write_text(
        "[verifier] no model.patch submitted", encoding="utf-8"
    )
    failed_result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    failed_result["verifier_result"] = {"rewards": {"reward": 0.0}}
    (trial / "result.json").write_text(json.dumps(failed_result), encoding="utf-8")

    failed_report = import_pier_results(
        job_dir, tmp_path / "failed-reports", subset, "qwen3.5-4b-local"
    )
    failed_task = json.loads(
        (failed_report / "tasks.jsonl").read_text(encoding="utf-8")
    )
    assert failed_task["terminal_state"] == "infrastructure_failed"
    assert failed_task["stop_reason"] == ("artifact_collection_patch_missing_or_empty")
    failed_provenance = json.loads(
        (failed_report / "provenance.json").read_text(encoding="utf-8")
    )
    assert failed_provenance["artifact_collection"]["tasks"][0]["status"] == (
        "patch_missing_or_empty"
    )


def test_artifact_audit_fails_closed_for_empty_patch(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    artifacts = trial / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "model.patch").write_bytes(b"")
    (artifacts / "manifest.json").write_text("[]", encoding="utf-8")
    trajectory = trial / "trace.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "type": "patch_delivery",
                "payload": {
                    "ok": True,
                    "has_patch": True,
                    "base_sha": "a" * 40,
                    "head_sha": "b" * 40,
                    "patch_bytes": 20,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = _audit_artifact_collection(
        trial,
        expected_base_sha="a" * 40,
        trajectory_path=trajectory,
        verifier_stdout="no model.patch submitted",
        verifier_rewards={"reward": 0},
    )

    assert audit.ok is False
    assert audit.status == "patch_missing_or_empty"


def test_artifact_audit_fails_closed_for_wrong_base(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    artifacts = trial / "artifacts"
    artifacts.mkdir(parents=True)
    patch = b"diff --git a/a b/a\n"
    (artifacts / "model.patch").write_bytes(patch)
    trajectory = trial / "trace.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "type": "patch_delivery",
                "payload": {
                    "ok": True,
                    "has_patch": True,
                    "base_sha": "b" * 40,
                    "head_sha": "c" * 40,
                    "patch_bytes": len(patch.rstrip()),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = _audit_artifact_collection(
        trial,
        expected_base_sha="a" * 40,
        trajectory_path=trajectory,
        verifier_stdout=f"model.patch applied ({len(patch)} bytes)",
        verifier_rewards={"reward": 1},
    )

    assert audit.ok is False
    assert audit.status == "base_mismatch"


def test_artifact_audit_fails_closed_for_unapplyable_patch(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    artifacts = trial / "artifacts"
    artifacts.mkdir(parents=True)
    patch = b"diff --git a/a b/a\n"
    (artifacts / "model.patch").write_bytes(patch)
    (artifacts / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "source": "/logs/artifacts/model.patch",
                    "destination": "artifacts/model.patch",
                    "type": "file",
                    "status": "ok",
                }
            ]
        ),
        encoding="utf-8",
    )
    trajectory = trial / "trace.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "type": "patch_delivery",
                "payload": {
                    "ok": True,
                    "has_patch": True,
                    "base_sha": "a" * 40,
                    "head_sha": "b" * 40,
                    "patch_bytes": len(patch.rstrip()),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = _audit_artifact_collection(
        trial,
        expected_base_sha="a" * 40,
        trajectory_path=trajectory,
        verifier_stdout="submitted model.patch failed to apply",
        verifier_rewards={"reward": 0, "apply_failed": 1},
    )

    assert audit.ok is False
    assert audit.status == "patch_apply_failed"
