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
from forgeloop.policy import (
    ACTIVE_OPEN_WEIGHT_POLICY,
    BUNDLED_POLICIES,
    PolicyIdentity,
    PolicyManifestError,
)
from forgeloop.runtime import LocalRuntime
from forgeloop.tools import build_default_tools
from forgeloop.trace import load_trajectory
from forgeloop.trajectory import TrajectoryStore
from forgeloop.workspace import Workspace


POLICY_PATH = "qwen3.5-4b-local"
SFT_POLICY_PATH = "qwen3.5-4b-sft-v1"
SFT_V2_POLICY_PATH = "qwen3.5-4b-sft-v2"
V4_CONTROLLER_POLICY_PATH = "deepseek-v4-flash-controller-v1"
V4_HYBRID_POLICY_PATH = "deepseek-v4-flash-hybrid-controller-v1.1"
V4_HYBRID_V12_POLICY_PATH = "deepseek-v4-flash-hybrid-controller-v1.2"
V4_EDIT_INTENT_POLICY_PATH = "deepseek-v4-flash-edit-intent-v1"


def _response(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        id=f"response-{call_id}",
        model="qwen3.5:4b",
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

    assert ACTIVE_OPEN_WEIGHT_POLICY == POLICY_PATH
    assert set(BUNDLED_POLICIES) == {
        POLICY_PATH,
        SFT_POLICY_PATH,
        SFT_V2_POLICY_PATH,
        V4_CONTROLLER_POLICY_PATH,
        V4_HYBRID_POLICY_PATH,
        V4_HYBRID_V12_POLICY_PATH,
        V4_EDIT_INTENT_POLICY_PATH,
    }
    assert policy.policy_id == POLICY_PATH
    assert policy.base_model == "Qwen/Qwen3.5-4B"
    assert len(policy.model_revision) == 64
    assert policy.stage == "base"
    assert policy.inference_backend == "ollama"
    assert policy.litellm_model == "openai/qwen3.5:4b"
    assert policy.capabilities.context_window == 8_192
    assert policy.capabilities.max_output_tokens == 2_048
    assert policy.capabilities.tool_calling is True
    assert policy.capabilities.thinking is True
    assert policy.serving_config["api_base"] == "http://127.0.0.1:11434/v1"
    assert policy.serving_config["model_quantization"] == "Q4_K_M"
    assert policy.serving_config["bypass_environment_proxy_for_loopback"] is True
    assert policy.serving_config["local_api_cost_usd"] == 0.0

    raw = PolicyIdentity.load(POLICY_PATH).to_dict()
    raw["serving_config"]["api_key"] = "must-not-be-persisted"
    bad = tmp_path / "bad-policy.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PolicyManifestError, match="cannot contain credentials"):
        PolicyIdentity.load(bad)


def test_sft_policy_is_independent_but_not_the_active_base_policy() -> None:
    policy = PolicyIdentity.load(SFT_POLICY_PATH)

    assert ACTIVE_OPEN_WEIGHT_POLICY == POLICY_PATH
    assert policy.policy_id == SFT_POLICY_PATH
    assert policy.stage == "sft"
    assert policy.base_model == "Qwen/Qwen3.5-4B"
    assert policy.litellm_model == "openai/qwen3.5-4b-sft-v1"
    assert policy.capabilities.context_window == 8_192
    assert policy.capabilities.max_output_tokens == 2_048
    assert policy.serving_config["adapter_revision"] == (
        "ea73f2c44e68dcf10c8c381a662888ed284953f1cb84f3b9e4156201db0308c3"
    )
    assert policy.serving_config["local_api_cost_usd"] == 0.0


def test_sft_v2_policy_pins_training_and_deployment_artifacts() -> None:
    policy = PolicyIdentity.load(SFT_V2_POLICY_PATH)

    assert ACTIVE_OPEN_WEIGHT_POLICY == POLICY_PATH
    assert policy.policy_id == SFT_V2_POLICY_PATH
    assert policy.stage == "sft"
    assert policy.base_model == "Qwen/Qwen3.5-4B"
    assert policy.litellm_model == "openai/qwen3.5-4b-sft-v2"
    assert policy.capabilities.context_window == 8_192
    assert policy.capabilities.max_output_tokens == 2_048
    assert policy.serving_config["adapter_revision"] == (
        "2f8f34073355bbd7eecc46576fe36adc9608b92f144a94abffc8dd0d68278561"
    )
    assert policy.serving_config["adapter_model_sha256"] == (
        "6f3d80a140171114af7dd56d38d8ba36fa17bbfb63c0d3bd8027256052016418"
    )
    assert policy.serving_config["ollama_model_id"] == "dcaf19b8ec99"
    assert policy.model_revision == (
        "67116bcdf1c60649dc88cfe53439588a002debc8fd0f4437c13c5b9428858def"
    )
    assert policy.serving_config["local_api_cost_usd"] == 0.0


def test_historical_qwen_9b_manifest_remains_provenance_compatible() -> None:
    historical_path = (
        Path(__file__).parents[1]
        / "src"
        / "forgeloop"
        / "policy_assets"
        / "qwen3.5-9b-vllm.json"
    )
    historical = PolicyIdentity.load(historical_path)

    assert "qwen3.5-9b" not in BUNDLED_POLICIES
    assert historical.policy_id == "qwen3.5-9b-base-v1"
    assert historical.inference_backend == "vllm"


def test_v4_flash_controller_policy_is_remote_non_secret_and_reproducible() -> None:
    policy = PolicyIdentity.load(V4_CONTROLLER_POLICY_PATH)

    assert ACTIVE_OPEN_WEIGHT_POLICY == POLICY_PATH
    assert policy.policy_id == V4_CONTROLLER_POLICY_PATH
    assert policy.stage == "base"
    assert policy.litellm_model == "deepseek/deepseek-v4-flash"
    assert policy.inference_backend == "deepseek-api"
    assert policy.capabilities.context_window == 1_000_000
    assert policy.capabilities.max_output_tokens == 384_000
    assert policy.capabilities.tool_calling is True
    assert policy.serving_config["api_base"] == "https://api.deepseek.com/v1"
    assert policy.serving_config["credential_env"] == "DEEPSEEK_API_KEY"
    assert policy.serving_config["controller"] == "v1"
    assert policy.serving_config["thinking_level"] == "max"
    assert "api_key" not in policy.serving_config
    assert "api_key" not in policy.generation_config


def test_v4_flash_hybrid_policy_keeps_main_route_and_pins_controller() -> None:
    policy = PolicyIdentity.load(V4_HYBRID_POLICY_PATH)

    assert policy.policy_id == V4_HYBRID_POLICY_PATH
    assert policy.litellm_model == "deepseek/deepseek-v4-flash"
    assert policy.serving_config["controller"] == "hybrid-v1.1"
    assert policy.serving_config["controller_policy"] == "qwen2.5-1.5b-controller-local"
    assert "api_key" not in policy.serving_config


def test_v4_flash_hybrid_v12_adds_gating_without_changing_models() -> None:
    policy = PolicyIdentity.load(V4_HYBRID_V12_POLICY_PATH)

    assert policy.policy_id == V4_HYBRID_V12_POLICY_PATH
    assert policy.litellm_model == "deepseek/deepseek-v4-flash"
    assert policy.serving_config["controller"] == "hybrid-v1.2"
    assert policy.serving_config["controller_policy"] == (
        "qwen2.5-1.5b-controller-local"
    )
    assert (
        policy.generation_config
        == PolicyIdentity.load(V4_HYBRID_POLICY_PATH).generation_config
    )


def test_v4_flash_edit_intent_keeps_v12_models_and_generation() -> None:
    policy = PolicyIdentity.load(V4_EDIT_INTENT_POLICY_PATH)
    v12 = PolicyIdentity.load(V4_HYBRID_V12_POLICY_PATH)

    assert policy.policy_id == V4_EDIT_INTENT_POLICY_PATH
    assert policy.litellm_model == v12.litellm_model
    assert policy.serving_config["controller"] == "hybrid-v1.2-edit-intent"
    assert policy.serving_config["controller_policy"] == (
        "qwen2.5-1.5b-controller-local"
    )
    assert policy.generation_config == v12.generation_config
    assert "api_key" not in policy.serving_config


def test_local_policy_bypasses_environment_proxy_only_for_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, dict] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            created["httpx"] = kwargs

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created["openai"] = kwargs

        def close(self) -> None:
            created["closed"] = {}

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=FakeClient))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    policy = PolicyIdentity.load(POLICY_PATH)
    provider = LiteLLMProvider(
        policy.litellm_model,
        api_base="http://127.0.0.1:11434/v1",
        api_key="EMPTY",
        policy=policy,
    )

    client = provider._local_openai_client(12.5)

    assert isinstance(client, FakeOpenAI)
    assert created["httpx"] == {"trust_env": False}
    assert created["openai"]["base_url"] == "http://127.0.0.1:11434/v1"
    assert created["openai"]["timeout"] == 12.5
    provider.api_base = "http://model-host:11434/v1"
    assert provider._local_openai_client(12.5) is None


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
        api_base="http://127.0.0.1:11434/v1",
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
    assert captured[0]["temperature"] == 0.2
    assert captured[0]["max_tokens"] == 2048
    assert "extra_body" not in captured[0]
    second_messages = captured[1]["messages"]
    assistant = next(
        message for message in second_messages if message["role"] == "assistant"
    )
    assert "reasoning_content" not in assistant

    events = load_trajectory(result.trajectory_path)
    started = events[0]["payload"]
    assert started["policy_identity"]["policy_id"] == "qwen3.5-4b-local"
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
        api_base="http://127.0.0.1:11434/v1",
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
    assert task_record["policy_identity"]["inference_backend"] == "ollama"
    assert task_record["total_cost_usd"] == 0.0

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
