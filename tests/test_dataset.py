import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import forgeloop.persistence as persistence
from forgeloop.cli import app
from forgeloop.dataset import (
    DATASET_SCHEMA_VERSION,
    INFRASTRUCTURE_FAILURE,
    MODEL_FAILURE,
    SFT_CANDIDATE,
    SFT_SCHEMA_VERSION,
    SUCCESSFUL_BUT_INEFFICIENT,
    DatasetBuilder,
    TrainingDataSanitizer,
    export_dataset,
    inspect_dataset,
    load_dataset,
)
from forgeloop.security import SecretRedactor


def _inject_atomic_failure(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"injected {stage} failure")

    if stage == "write":
        monkeypatch.setattr(persistence, "_write_all", fail)
    elif stage == "fsync":
        monkeypatch.setattr(persistence.os, "fsync", fail)
    elif stage == "replace":
        monkeypatch.setattr(persistence.os, "replace", fail)
    else:  # pragma: no cover - protects the test helper itself
        raise AssertionError(stage)


def _assert_no_atomic_temps(directory: Path, target_name: str) -> None:
    assert list(directory.glob(f".{target_name}.*.tmp")) == []


def _event(run_id: str, sequence: int, event_type: str, payload: dict) -> dict:
    return {
        "schema_version": "forgeloop.trajectory.v1",
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": "2026-08-08T00:00:00+00:00",
        "type": event_type,
        "payload": payload,
    }


def _write_trajectory(
    path: Path,
    run_id: str,
    *,
    terminal: str,
    stop_reason: str,
    verifier_passed: bool,
    secret: str = "",
    with_effect: bool = False,
) -> None:
    workspace = path.parent.parent / "workspaces" / run_id
    events = [
        _event(
            run_id,
            0,
            "run_started",
            {
                "mode": "task",
                "request": f"Fix the bug {secret}",
                "model": "mock/model",
                "policy_identity": {
                    "schema_version": "forgeloop.policy.v1",
                    "policy_id": "mock-base-v1",
                    "stage": "base",
                    "base_model": "mock/base-model",
                    "model_revision": "revision-1",
                    "tokenizer": "mock/tokenizer",
                    "tokenizer_revision": "tokenizer-revision-1",
                    "inference_backend": "vllm",
                    "litellm_model": "mock/model",
                    "capabilities": {"tool_calling": True},
                    "serving_config": {"tool_call_parser": "mock"},
                    "generation_config": {"temperature": 0.2},
                },
                "workspace": str(workspace),
                "git": {"head": "a" * 40, "repository_root": str(workspace)},
            },
        ),
        _event(
            run_id,
            1,
            "model_request",
            {
                "step": 1,
                "messages": [
                    {"role": "system", "content": f"Workspace: {workspace}"},
                    {"role": "user", "content": f"Fix the bug {secret}"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "read_file", "parameters": {}},
                    }
                ],
            },
        ),
        _event(
            run_id,
            2,
            "model_response",
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "read_file",
                        "arguments": {"path": "bug.py"},
                    }
                ],
            },
        ),
        _event(
            run_id,
            3,
            "tool_call",
            {"id": "call-1", "name": "read_file", "arguments": {"path": "bug.py"}},
        ),
        _event(
            run_id,
            4,
            "observation",
            {
                "tool_call_id": "call-1",
                "tool": "read_file",
                "ok": True,
                "output": f"buggy code {secret}",
                "metadata": {},
            },
        ),
        _event(
            run_id,
            5,
            "model_response",
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "finish-1",
                        "name": "finish",
                        "arguments": {"status": terminal, "summary": "done"},
                    }
                ],
            },
        ),
        _event(
            run_id,
            6,
            "tool_call",
            {
                "id": "finish-1",
                "name": "finish",
                "arguments": {"status": terminal, "summary": "done"},
            },
        ),
        _event(
            run_id,
            7,
            "run_finished",
            {"status": terminal, "stop_reason": stop_reason, "summary": "done"},
        ),
        _event(run_id, 8, "eval_runtime_started", {"type": "local"}),
        _event(
            run_id,
            9,
            "eval_verifier",
            {
                "command": "pytest",
                "passed": verifier_passed,
                "exit_code": 0 if verifier_passed else 1,
                "stdout": "ok" if verifier_passed else "failed",
                "stderr": "",
                "timed_out": False,
            },
        ),
        _event(
            run_id,
            10,
            "eval_final_diff",
            {
                "diff": f"diff --git a/bug.py b/bug.py\n+fixed {secret}\n",
                "status": " M bug.py",
            },
        ),
    ]
    if with_effect:
        events.append(
            _event(
                run_id,
                11,
                "effect",
                {
                    "schema_version": "forgeloop.effect.v1",
                    "event_id": f"eff_{run_id}_0001",
                    "trajectory_id": run_id,
                    "step": 1,
                    "timestamp": "2026-08-08T00:00:00+00:00",
                    "type": "file.write",
                    "tool_name": "apply_patch",
                    "tool_call_id": "call-1",
                    "target": "bug.py",
                    "action": {"operation": "update"},
                    "result": {"status": "success"},
                    "risk": {"level": "medium", "flags": ["review_change"]},
                    "evidence": {"sha256": "a" * 64},
                },
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def _make_source(tmp_path: Path) -> tuple[Path, Path, str]:
    secret = "test-provider-secret"
    source = tmp_path / "eval-runs"
    run = source / "run-one"
    trajectories = run / "trajectories"
    suite_path = tmp_path / "suite" / "tasks.json"
    suite_path.parent.mkdir(parents=True)
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": "forgeloop.eval.v2",
                "suite_id": "suite-one",
                "suite_kind": "real-swe",
                "tasks": [
                    {
                        "id": "task-one",
                        "description": "Fix the bug",
                        "fixture": "fixture",
                        "base_commit": "a" * 40,
                        "mode": "task",
                        "verifier": {"command": "pytest"},
                        "source": {
                            "repository": "https://example.test/repo.git",
                            "pr": "https://example.test/repo/pull/1",
                            "fix_commit": "b" * 40,
                            "base_sha": "c" * 40,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cases = [
        ("sft-run", True, "completed", "model_finish_tool", "none"),
        ("inefficient-run", True, "blocked", "no_progress", "none"),
        ("model-failure-run", False, "failed", "model_finish_tool", "model_failure"),
        (
            "infrastructure-run",
            False,
            "infrastructure_failed",
            "eval_infrastructure_error",
            "environment_eval_failure",
        ),
    ]
    records = []
    for run_id, success, terminal, stop_reason, failure_category in cases:
        trajectory_path = trajectories / f"{run_id}.jsonl"
        _write_trajectory(
            trajectory_path,
            run_id,
            terminal=terminal,
            stop_reason=stop_reason,
            verifier_passed=success,
            secret=secret if run_id == "sft-run" else "",
            with_effect=run_id == "sft-run",
        )
        records.append(
            {
                "task_id": "task-one",
                "description": "Fix the bug",
                "model": "mock/model",
                "provider": "mock",
                "success": success,
                "terminal_state": terminal,
                "stop_reason": stop_reason,
                "verifier": {
                    "command": "pytest",
                    "passed": success,
                    "exit_code": 0 if success else 1,
                },
                "failure_category": failure_category,
                "failure_detail": None,
                "expected_base_sha": "a" * 40,
                "model_calls": 2,
                "tool_calls": 2,
                "steps": 2,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "total_cost_usd": 0.01,
                "cost_sources": ["provider_response"],
                "wall_time_seconds": 1.0,
                "final_diff": f"diff --git a/bug.py b/bug.py\n+fixed {secret}\n",
                "final_status": " M bug.py",
                "trajectory_path": str(trajectory_path),
                "attempt": 1,
            }
        )
    records.append({"task_id": "missing", "trajectory_path": None})
    run.mkdir(parents=True, exist_ok=True)
    (run / "tasks.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (run / "summary.json").write_text(
        json.dumps({"suite_id": "suite-one", "run_id": "eval-run-one"}),
        encoding="utf-8",
    )
    return source, suite_path, secret


def test_build_classifies_and_preserves_complete_provenance(tmp_path: Path) -> None:
    source, suite_path, secret = _make_source(tmp_path)
    output = tmp_path / "dataset"
    sanitizer = TrainingDataSanitizer(
        redactor=SecretRedactor((secret,)), local_roots=(tmp_path,)
    )
    result = DatasetBuilder(
        source, output, suite_paths=(suite_path,), sanitizer=sanitizer
    ).build()

    assert result.samples == 4
    assert result.classifications == {
        SFT_CANDIDATE: 1,
        SUCCESSFUL_BUT_INEFFICIENT: 1,
        MODEL_FAILURE: 1,
        INFRASTRUCTURE_FAILURE: 1,
    }
    assert result.skipped == {"missing_trajectory": 1}
    samples = load_dataset(output)
    sft = next(
        sample for sample in samples if sample["classification"] == SFT_CANDIDATE
    )
    assert sft["schema_version"] == DATASET_SCHEMA_VERSION
    assert sft["repo"] == "https://example.test/repo.git"
    assert sft["base_sha"] == "a" * 40
    assert sft["source_trajectory_id"] == "sft-run"
    assert sft["policy_identity"]["policy_id"] == "mock-base-v1"
    assert sft["policy_identity"]["model_revision"] == "revision-1"
    assert sft["task_provenance"]["task_id"] == "task-one"
    assert sft["task_provenance"]["source_base_sha"] == "c" * 40
    assert sft["verifier_result"]["passed"] is True
    assert sft["runtime"]["type"] == "local"
    assert len(sft["tool_calls"]) == 2
    assert len(sft["tool_observations"]) == 1
    assert len(sft["effect_events"]) == 1
    assert sft["effect_summary"]["status"] == "recorded"
    assert sft["effect_summary"]["by_type"] == {"file.write": 1}
    assert sft["effect_summary"]["modified_files"] == ["bug.py"]
    assert sft["safety_flags"] == ["review_change"]
    assert [message["role"] for message in sft["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    serialized = json.dumps(sft)
    assert secret not in serialized
    assert str(tmp_path) not in serialized
    assert "[REDACTED]" in serialized
    assert "[LOCAL_ROOT]" in serialized

    legacy = next(
        sample for sample in samples if sample["classification"] == MODEL_FAILURE
    )
    assert legacy["effect_events"] == []
    assert legacy["effect_summary"]["status"] == "legacy_no_effect_events"

    stats = inspect_dataset(output)
    assert stats["samples"] == 4
    assert stats["total_tokens"] == 60
    assert stats["repositories"] == 1
    assert stats["effect_events"] == 1
    assert stats["effect_types"] == {"file.write": 1}
    assert stats["effect_statuses"] == {
        "legacy_no_effect_events": 3,
        "recorded": 1,
    }
    assert stats["safety_flags"] == {"review_change": 1}


def test_exports_filter_infrastructure_and_keep_sft_adapter_separate(
    tmp_path: Path,
) -> None:
    source, suite_path, secret = _make_source(tmp_path)
    dataset = tmp_path / "dataset"
    sanitizer = TrainingDataSanitizer(
        redactor=SecretRedactor((secret,)), local_roots=(tmp_path,)
    )
    DatasetBuilder(
        source, dataset, suite_paths=(suite_path,), sanitizer=sanitizer
    ).build()

    sft_path = tmp_path / "exports" / "sft.jsonl"
    count, classifications = export_dataset(dataset, sft_path, sanitizer=sanitizer)
    assert count == 1
    assert classifications == {SFT_CANDIDATE: 1}
    exported = json.loads(sft_path.read_text(encoding="utf-8"))
    assert exported["schema_version"] == SFT_SCHEMA_VERSION
    assert exported["metadata"]["trajectory_id"] == "sft-run"
    assert exported["metadata"]["policy_identity"]["stage"] == "base"
    assert exported["metadata"]["verifier_passed"] is True
    assert exported["messages"]
    assert exported["tools"]
    assert exported["outcome"]["final_diff"]
    assert "effect_events" not in exported
    assert secret not in sft_path.read_text(encoding="utf-8")

    internal_path = tmp_path / "exports" / "internal.jsonl"
    internal_count, internal_classifications = export_dataset(
        dataset,
        internal_path,
        export_format="internal",
        sanitizer=sanitizer,
    )
    assert internal_count == 3
    assert INFRASTRUCTURE_FAILURE not in internal_classifications
    assert all(
        json.loads(line)["classification"] != INFRASTRUCTURE_FAILURE
        for line in internal_path.read_text(encoding="utf-8").splitlines()
    )


def test_sanitizer_removes_provider_credentials_and_local_paths(tmp_path: Path) -> None:
    exact = "exact-environment-secret"
    sanitizer = TrainingDataSanitizer(
        redactor=SecretRedactor((exact,)), local_roots=(tmp_path,)
    )
    value = sanitizer.sanitize(
        {
            "api_key": "plain-secret",
            "log": (
                f"Authorization: Bearer bearer-value token=token-value "
                f"OPENAI_API_KEY=historical-provider-value "
                f"https://user:http-password@example.test/path "
                f"sk-1234567890abcdefghijkl {exact} {tmp_path / 'repo' / 'file.py'}"
            ),
        }
    )
    serialized = json.dumps(value)
    for forbidden in (
        "plain-secret",
        "bearer-value",
        "token-value",
        "historical-provider-value",
        "http-password",
        "sk-1234567890abcdefghijkl",
        exact,
        str(tmp_path),
    ):
        assert forbidden not in serialized
    assert serialized.count("[REDACTED]") >= 5
    assert "[LOCAL_ROOT]" in serialized


def test_dataset_cli_build_inspect_and_export(tmp_path: Path) -> None:
    source, suite_path, _ = _make_source(tmp_path)
    dataset = tmp_path / "dataset"
    runner = CliRunner()
    built = runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--source",
            str(source),
            "--output",
            str(dataset),
            "--suite",
            str(suite_path),
        ],
    )
    assert built.exit_code == 0, built.output
    assert "Samples: 4" in built.output

    inspected = runner.invoke(app, ["dataset", "inspect", "--dataset", str(dataset)])
    assert inspected.exit_code == 0, inspected.output
    assert "sft_candidate" in inspected.output
    assert "Repositories: 1" in inspected.output

    output = tmp_path / "sft.jsonl"
    exported = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ],
    )
    assert exported.exit_code == 0, exported.output
    assert "Exported: 1" in exported.output
    assert output.is_file()


def test_loading_pre_effect_dataset_index_adds_legacy_defaults(tmp_path: Path) -> None:
    source, suite_path, _ = _make_source(tmp_path)
    dataset = tmp_path / "dataset"
    DatasetBuilder(source, dataset, suite_paths=(suite_path,)).build()
    sample = json.loads(
        (dataset / "index.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    sample.pop("effect_events")
    sample.pop("effect_summary")
    sample.pop("safety_flags")
    sample.pop("policy_identity")
    legacy_dataset = tmp_path / "legacy-dataset"
    legacy_dataset.mkdir()
    (legacy_dataset / "index.jsonl").write_text(
        json.dumps(sample) + "\n", encoding="utf-8"
    )

    loaded = load_dataset(legacy_dataset)[0]
    assert loaded["effect_events"] == []
    assert loaded["effect_summary"]["status"] == "legacy_no_effect_events"
    assert loaded["safety_flags"] == []
    assert loaded["policy_identity"]["identity_status"] == "legacy_model_only"


@pytest.mark.parametrize("stage", ("write", "fsync", "replace"))
def test_dataset_index_publish_failure_preserves_existing_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    source, suite_path, _ = _make_source(tmp_path)
    output = tmp_path / "dataset"
    output.mkdir()
    old_index = b'{"old":"index"}\n'
    old_manifest = b'{"old":"manifest"}'
    (output / "index.jsonl").write_bytes(old_index)
    (output / "manifest.json").write_bytes(old_manifest)
    _inject_atomic_failure(monkeypatch, stage)

    with pytest.raises(OSError, match=f"injected {stage} failure"):
        DatasetBuilder(source, output, suite_paths=(suite_path,)).build()

    assert (output / "index.jsonl").read_bytes() == old_index
    assert (output / "manifest.json").read_bytes() == old_manifest
    _assert_no_atomic_temps(output, "index.jsonl")


def test_dataset_manifest_replace_failure_preserves_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, suite_path, _ = _make_source(tmp_path)
    output = tmp_path / "dataset"
    output.mkdir()
    old_manifest = b'{"old":"manifest"}'
    (output / "manifest.json").write_bytes(old_manifest)
    real_replace = os.replace

    def fail_manifest_replace(source_path: object, target_path: object) -> None:
        if Path(target_path).name == "manifest.json":
            raise OSError("injected manifest replace failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(persistence.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="injected manifest replace failure"):
        DatasetBuilder(source, output, suite_paths=(suite_path,)).build()

    assert (output / "manifest.json").read_bytes() == old_manifest
    assert json.loads(
        (output / "index.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    _assert_no_atomic_temps(output, "manifest.json")


@pytest.mark.parametrize("stage", ("write", "fsync", "replace"))
def test_dataset_export_publish_failure_preserves_existing_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    source, suite_path, _ = _make_source(tmp_path)
    dataset = tmp_path / "dataset"
    DatasetBuilder(source, dataset, suite_paths=(suite_path,)).build()
    output = tmp_path / "exports" / "sft.jsonl"
    output.parent.mkdir()
    old_export = b'{"old":"export"}\n'
    output.write_bytes(old_export)
    _inject_atomic_failure(monkeypatch, stage)

    with pytest.raises(OSError, match=f"injected {stage} failure"):
        export_dataset(dataset, output)

    assert output.read_bytes() == old_export
    _assert_no_atomic_temps(output.parent, output.name)
