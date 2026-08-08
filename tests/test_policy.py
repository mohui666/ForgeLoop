from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from forgeloop.agent import AgentLoop, RunMode, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.dataset import DatasetBuilder, SFT_CANDIDATE, load_dataset
from forgeloop.evals import EvalRunner, EvalSuite, default_suite_path
from forgeloop.models import LiteLLMProvider
from forgeloop.policy import PolicyIdentity, PolicyManifestError
from forgeloop.runtime import LocalRuntime
from forgeloop.tools import build_default_tools
from forgeloop.trace import load_trajectory
from forgeloop.trajectory import TrajectoryStore
from forgeloop.workspace import Workspace


POLICY_PATH = "qwen3.5-9b"


def _response(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        id=f"response-{call_id}",
        model="forgeloop-qwen35-9b",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[call],
                    reasoning_content="private chain of thought",
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
        _hidden_params={"custom_llm_provider": "openai"},
    )


def test_policy_manifest_is_pinned_capable_and_non_secret(tmp_path: Path) -> None:
    policy = PolicyIdentity.load(POLICY_PATH)

    assert policy.base_model == "Qwen/Qwen3.5-9B"
    assert len(policy.model_revision) == 40
    assert policy.stage == "base"
    assert policy.capabilities.context_window == 131_072
    assert policy.capabilities.max_output_tokens == 32_768
    assert policy.capabilities.tool_calling is True
    assert policy.capabilities.thinking is True
    assert policy.serving_config["tool_call_parser"] == "qwen3_coder"

    raw = PolicyIdentity.load(POLICY_PATH).to_dict()
    raw["serving_config"]["api_key"] = "must-not-be-persisted"
    bad = tmp_path / "bad-policy.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PolicyManifestError, match="cannot contain credentials"):
        PolicyIdentity.load(bad)


def test_qwen_policy_drives_litellm_tools_and_records_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@forgeloop.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "ForgeLoop Test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "maths.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_maths.py").write_text(
        "from maths import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)

    scripted = iter(
        [
            _response("1", "read_file", {"path": "maths.py"}),
            _response("2", "search_files", {"pattern": "return", "path": "."}),
            _response(
                "3",
                "apply_patch",
                {
                    "path": "maths.py",
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                },
            ),
            _response("4", "shell", {"command": "python -m pytest -q"}),
            _response("5", "git_diff", {}),
            _response(
                "6",
                "finish",
                {
                    "status": "completed",
                    "summary": "Fixed add and verified tests.",
                    "evidence": "pytest passed and git diff contains the fix.",
                },
            ),
        ]
    )
    captured: list[dict] = []

    def completion(**kwargs):
        captured.append(kwargs)
        return next(scripted)

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=completion, model_cost={}),
    )
    policy = PolicyIdentity.load(POLICY_PATH)
    provider = LiteLLMProvider(
        model=policy.litellm_model,
        api_base="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        policy=policy,
    )
    workspace = Workspace(tmp_path)
    trajectory = TrajectoryStore(tmp_path / ".forgeloop" / "runs", run_id="qwen")
    result = AgentLoop(
        provider,
        build_default_tools(workspace, LocalRuntime()),
        workspace,
        trajectory,
        BudgetLimits(max_steps=10, max_model_calls=10, max_tool_calls=10),
    ).run(RunMode.TASK, "Fix add and run the tests")

    assert result.status is RunStatus.COMPLETED
    assert "return a + b" in (tmp_path / "maths.py").read_text(encoding="utf-8")
    assert [call["model"] for call in captured] == [policy.litellm_model] * 6
    assert captured[0]["temperature"] == 0.6
    assert captured[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    second_messages = captured[1]["messages"]
    assistant = next(
        message for message in second_messages if message["role"] == "assistant"
    )
    assert "reasoning_content" not in assistant

    events = load_trajectory(result.trajectory_path)
    started = events[0]["payload"]
    assert started["policy_identity"]["policy_id"] == "qwen3.5-9b-base-v1"
    assert started["policy_identity"]["model_revision"] == policy.model_revision
    calls = [
        event["payload"]["name"] for event in events if event["type"] == "tool_call"
    ]
    assert calls == [
        "read_file",
        "search_files",
        "apply_patch",
        "shell",
        "git_diff",
        "finish",
    ]
    effect_types = {
        event["payload"]["type"] for event in events if event["type"] == "effect"
    }
    assert {
        "file.read",
        "file.write",
        "git.change",
        "shell.exec",
        "test.run",
    } <= effect_types


def test_policy_eval_trajectory_flows_into_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = iter(
        [
            _response(
                "patch",
                "apply_patch",
                {
                    "path": "pricing.py",
                    "old_text": "if is_member or subtotal >= 100:",
                    "new_text": "if is_member and subtotal >= 100:",
                },
            ),
            _response(
                "finish",
                "finish",
                {
                    "status": "completed",
                    "summary": "Fixed the membership condition.",
                    "evidence": "The focused verifier passes.",
                },
            ),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=lambda **kwargs: next(scripted), model_cost={}),
    )
    policy = PolicyIdentity.load(POLICY_PATH)
    provider = LiteLLMProvider(
        policy.litellm_model,
        api_base="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        policy=policy,
    )
    suite = EvalSuite.load(default_suite_path())
    summary, run_dir = EvalRunner(
        provider=provider,
        limits=BudgetLimits(max_seconds=60, max_tokens=10_000),
        output_root=tmp_path / "runs",
    ).run(suite, suite.select_stage("a"), repeats=1)

    assert summary.solved == 1
    assert summary.policy_identity["model_revision"] == policy.model_revision
    task_record = json.loads(
        (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert task_record["policy_identity"]["inference_backend"] == "vllm"

    dataset_dir = tmp_path / "dataset"
    result = DatasetBuilder(
        run_dir,
        dataset_dir,
        suite_paths=(default_suite_path(),),
    ).build()
    assert result.samples == 1
    sample = load_dataset(dataset_dir)[0]
    assert sample["classification"] == SFT_CANDIDATE
    assert sample["policy_identity"]["policy_id"] == policy.policy_id
    assert sample["policy_identity"]["tokenizer_revision"] == policy.tokenizer_revision
    assert sample["effect_events"]
