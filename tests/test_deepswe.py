from __future__ import annotations

import json
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
    import_pier_results,
    pier_command,
    select_task_ids,
)


def test_frozen_subset_is_unique_and_reproducible() -> None:
    subset = DeepSWESubset.load(DEFAULT_SUBSET_PATH)

    assert len(subset.tasks) == 20
    assert len(set(subset.tasks)) == 20
    assert subset.revision == "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9"
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
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "forgeloop.trajectory.v2",
                "run_id": "trace-one",
                "sequence": 0,
                "timestamp": "2026-08-09T00:00:01+00:00",
                "type": "run_finished",
                "payload": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (trial / "verifier" / "test-stdout.txt").write_text(
        "official tests passed", encoding="utf-8"
    )
    (trial / "artifacts" / "model.patch").write_text(
        "diff --git a/a.py b/a.py\n", encoding="utf-8"
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/mashumaro-flattened-dataclass-fields",
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
    mapped_trace = Path(task["trajectory_path"])
    event_types = [
        json.loads(line)["type"]
        for line in mapped_trace.read_text(encoding="utf-8").splitlines()
    ]
    assert event_types[-2:] == ["eval_verifier", "eval_final_diff"]
