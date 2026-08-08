from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from typer.testing import CliRunner

from forgeloop.agent import AgentLoop, RunMode
from forgeloop.budget import BudgetLimits
from forgeloop.cli import app
from forgeloop.dataset import DatasetBuilder, load_dataset
from forgeloop.effects import EffectContext, EffectDraft, EffectRecorder
from forgeloop.evals import EvalRunner, EvalSuite, default_suite_path
from forgeloop.models.base import ModelProvider
from forgeloop.runtime import LocalRuntime
from forgeloop.security import SecretRedactor
from forgeloop.tools import build_default_tools
from forgeloop.trace import (
    analyze_trajectory,
    explain_trajectory,
    load_trajectory,
    replay_trajectory,
    resolve_trajectory,
)
from forgeloop.trajectory import TrajectoryStore
from forgeloop.types import Message, ModelResponse, ModelUsage, ToolCall
from forgeloop.workspace import Workspace


class ScriptedProvider(ModelProvider):
    model_id = "test/effects"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = iter(responses)
        self.requests: list[Sequence[Message]] = []

    def complete(self, messages, tools, *, timeout_seconds):
        del tools
        assert timeout_seconds > 0
        self.requests.append(list(messages))
        return next(self.responses)


def _call(call_id: str, name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(call_id, name, arguments),),
        usage=ModelUsage(10, 5, 0.001),
        finish_reason="tool_calls",
    )


def _git(cwd: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )


def _agent_task(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "tests").mkdir()
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "def value():\n    return 'hello'\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_sample.py").write_text(
        "import unittest\n"
        "from sample import value\n\n"
        "class SampleTests(unittest.TestCase):\n"
        "    def test_value(self):\n"
        "        self.assertEqual(value(), 'world')\n",
        encoding="utf-8",
    )
    _git(tmp_path, "init", "-q", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "ForgeLoop Test")
    _git(tmp_path, "config", "user.email", "test@forgeloop.invalid")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "baseline")

    provider = ScriptedProvider(
        [
            _call("read-1", "read_file", path="sample.py"),
            _call(
                "edit-1",
                "apply_patch",
                path="sample.py",
                old_text="return 'hello'",
                new_text="return 'world'",
            ),
            _call(
                "test-1",
                "shell",
                command="python -m unittest discover -s tests -v",
            ),
            _call(
                "finish-1",
                "finish",
                status="completed",
                summary="updated and tested",
                evidence="unittest passed",
            ),
        ]
    )
    workspace = Workspace(tmp_path)
    store = TrajectoryStore(tmp_path / ".forgeloop" / "runs", run_id="effects-agent")
    agent = AgentLoop(
        provider,
        build_default_tools(workspace, LocalRuntime()),
        workspace,
        store,
        BudgetLimits(max_seconds=60, max_tokens=1_000),
    )
    result = agent.run(RunMode.TASK, "Change hello to world and run tests")
    assert result.status.value == "completed"
    return result.trajectory_path, tmp_path / "sample.py"


def test_real_agent_task_records_linked_effects_and_replays_offline(
    tmp_path: Path,
) -> None:
    trajectory_path, sample_path = _agent_task(tmp_path)
    events = load_trajectory(trajectory_path)
    effects = [event["payload"] for event in events if event["type"] == "effect"]
    effect_types = {effect["type"] for effect in effects}

    assert events[0]["schema_version"] == "forgeloop.trajectory.v2"
    assert {
        "file.read",
        "file.write",
        "shell.exec",
        "test.run",
        "git.change",
    } <= effect_types
    assert all(effect["trajectory_id"] == "effects-agent" for effect in effects)
    assert all(effect["tool_call_id"] for effect in effects)
    assert all(
        {
            "event_id",
            "trajectory_id",
            "step",
            "timestamp",
            "type",
            "tool_name",
            "tool_call_id",
            "target",
            "action",
            "result",
            "risk",
            "evidence",
        }
        <= effect.keys()
        for effect in effects
    )
    assert all(
        isinstance(effect["step"], int) and effect["step"] > 0 for effect in effects
    )
    effects_per_call = Counter(effect["tool_call_id"] for effect in effects)
    assert effects_per_call["edit-1"] == 2
    assert effects_per_call["test-1"] == 2
    git_effect = next(effect for effect in effects if effect["type"] == "git.change")
    assert "sample.py" in git_effect["action"]["changed_paths"]
    test_effect = next(effect for effect in effects if effect["type"] == "test.run")
    assert test_effect["result"]["status"] == "pass"
    assert test_effect["tool_call_id"] == "test-1"

    sample_path.write_text("sentinel\n", encoding="utf-8")
    replay = replay_trajectory(trajectory_path)
    assert "Mode: offline evidence only" in replay
    assert "Edit sample.py" in replay
    assert "Test PASS" in replay
    assert "Git change: sample.py" in replay
    assert sample_path.read_text(encoding="utf-8") == "sentinel\n"
    assert (
        resolve_trajectory("effects-agent", search_roots=(trajectory_path.parent,))
        == trajectory_path.resolve()
    )


def test_eval_effects_flow_into_dataset_sample(tmp_path: Path) -> None:
    suite = EvalSuite.load(default_suite_path())
    provider = ScriptedProvider(
        [
            _call(
                "edit-1",
                "apply_patch",
                path="pricing.py",
                old_text="if is_member or subtotal >= 100:",
                new_text="if is_member and subtotal >= 100:",
            ),
            _call(
                "test-1",
                "shell",
                command="python -m unittest discover -s tests -v",
            ),
            _call(
                "finish-1",
                "finish",
                status="completed",
                summary="fixed and verified",
                evidence="unittest passed",
            ),
        ]
    )
    source_root = tmp_path / "eval-runs"
    summary, _ = EvalRunner(
        provider=provider,
        limits=BudgetLimits(max_seconds=60, max_tokens=1_000),
        output_root=source_root,
    ).run(suite, suite.select_stage("a"))
    assert summary.solved == 1

    dataset = tmp_path / "dataset"
    result = DatasetBuilder(
        source_root,
        dataset,
        suite_paths=(default_suite_path(),),
    ).build()
    assert result.samples == 1
    sample = load_dataset(dataset)[0]
    effect_types = {effect["type"] for effect in sample["effect_events"]}
    assert {"file.write", "git.change", "shell.exec", "test.run"} <= effect_types
    assert sample["effect_summary"]["status"] == "recorded"
    assert sample["effect_summary"]["total"] == len(sample["effect_events"])
    assert "pricing.py" in sample["effect_summary"]["modified_files"]


def test_shell_delete_and_policy_block_create_effects(tmp_path: Path) -> None:
    (tmp_path / "remove.txt").write_text("remove me", encoding="utf-8")
    _git(tmp_path, "init", "-q", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "ForgeLoop Test")
    _git(tmp_path, "config", "user.email", "test@forgeloop.invalid")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "baseline")
    workspace = Workspace(tmp_path)
    store = TrajectoryStore(tmp_path / "trace", run_id="delete-effects")
    tools = build_default_tools(workspace, LocalRuntime())
    tools.bind_effect_recorder(EffectRecorder(store, tmp_path))

    deleted = tools.execute(
        "shell",
        {
            "command": "python -c \"from pathlib import Path; Path('remove.txt').unlink()\""
        },
        timeout_seconds=10,
        effect_context=EffectContext(step=1, tool_call_id="delete-1"),
    )
    blocked = tools.execute(
        "shell",
        {"command": "rm -rf data"},
        timeout_seconds=10,
        effect_context=EffectContext(step=2, tool_call_id="blocked-1"),
    )

    assert deleted.ok
    assert not blocked.ok
    effects = [
        event["payload"]
        for event in load_trajectory(store.path)
        if event["type"] == "effect"
    ]
    assert any(
        effect["type"] == "file.delete" and effect["target"] == "remove.txt"
        for effect in effects
    )
    violation = next(
        effect for effect in effects if effect["type"] == "policy.violation"
    )
    assert violation["result"]["status"] == "blocked"
    assert "destructive_operation" in violation["risk"]["flags"]


def test_effect_evidence_redacts_credentials_and_local_roots(tmp_path: Path) -> None:
    secret = "effect-secret-value"
    store = TrajectoryStore(
        tmp_path / "trace",
        run_id="sanitized-effects",
        redactor=SecretRedactor((secret,)),
    )
    recorder = EffectRecorder(store, tmp_path)
    recorder.record(
        EffectDraft(
            "shell.exec",
            str(tmp_path),
            action={
                "command": f"tool --api-key={secret} --root={tmp_path}",
                "authorization": f"Bearer {secret}",
            },
            result={"status": "success"},
            evidence={"stdout": f"token={secret}"},
        ),
        context=EffectContext(step=1, tool_call_id="secret-1"),
        tool_name="shell",
    )
    recorder.record(
        EffectDraft(
            "http.request",
            f"https://user:{secret}@example.test/path?token={secret}",
            action={
                "method": "GET",
                "headers": {"Authorization": f"Bearer {secret}"},
            },
            result={"status": "success", "status_code": 200},
            evidence={"body": "not_stored"},
        ),
        context=EffectContext(step=2, tool_call_id="http-1"),
        tool_name="http",
    )

    text = store.path.read_text(encoding="utf-8")
    assert secret not in text
    assert str(tmp_path) not in text
    assert "[REDACTED]" in text
    assert "[LOCAL_ROOT]" in text


def test_explain_detects_repeated_legacy_tool_error_without_effects(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(tmp_path, run_id="legacy-errors")
    store.append("run_started", {"request": "read missing file"})
    for index in (1, 2):
        store.append(
            "tool_call",
            {
                "id": f"read-{index}",
                "name": "read_file",
                "arguments": {"path": "missing.txt"},
            },
        )
        store.append(
            "observation",
            {
                "tool_call_id": f"read-{index}",
                "tool": "read_file",
                "ok": False,
                "output": "Not a file: missing.txt",
            },
        )
    store.append(
        "run_finished",
        {"status": "blocked", "stop_reason": "repeated_error"},
    )

    analysis = analyze_trajectory(load_trajectory(store.path))
    explanation = explain_trajectory(store.path)
    assert analysis["outcome"] == "UNKNOWN"
    assert analysis["error_loops"] == 1
    assert analysis["first_issue"]["sequence"] == 4
    assert "same tool error observation" in analysis["first_issue"]["reason"]
    assert "Termination: blocked/repeated_error" in explanation


def test_explain_finds_post_pass_redundant_action_before_no_progress(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(tmp_path, run_id="explain-effects")
    recorder = EffectRecorder(store, tmp_path)
    store.append("run_started", {"request": "test until done"})
    recorder.record(
        EffectDraft(
            "test.run",
            ".",
            action={"command": "pytest"},
            result={"status": "pass", "exit_code": 0},
        ),
        context=EffectContext(step=8, tool_call_id="test-8"),
        tool_name="shell",
    )
    store.append("eval_verifier", {"command": "pytest", "passed": True, "exit_code": 0})
    recorder.record(
        EffectDraft(
            "test.run",
            ".",
            action={"command": "pytest"},
            result={"status": "pass", "exit_code": 0},
        ),
        context=EffectContext(step=9, tool_call_id="test-9"),
        tool_name="shell",
    )
    store.append(
        "run_finished",
        {"status": "blocked", "stop_reason": "no_progress", "summary": "stopped"},
    )

    events = load_trajectory(store.path)
    analysis = analyze_trajectory(events)
    explanation = explain_trajectory(store.path, events)
    assert analysis["outcome"] == "PASS"
    assert analysis["post_verifier_pass_effect"] is True
    assert analysis["first_issue"]["step"] == 9
    assert analysis["no_progress_context"]["step"] == 9
    assert "Termination: blocked/no_progress" in explanation
    assert "Step 9" in explanation
    assert "Verifier had already passed" in explanation
    assert "continued a repeated action" in explanation

    runner = CliRunner()
    replayed = runner.invoke(app, ["trace", "replay", str(store.path)])
    explained = runner.invoke(app, ["trace", "explain", str(store.path)])
    assert replayed.exit_code == 0, replayed.output
    assert "Verifier PASS" in replayed.output
    assert "Terminal blocked (no_progress)" in replayed.output
    assert explained.exit_code == 0, explained.output
    assert "Outcome: PASS" in explained.output
