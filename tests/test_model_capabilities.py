from __future__ import annotations

import json
import subprocess
import urllib.error
from pathlib import Path

import pytest

from forgeloop.config import ConfigStore, GlobalConfig
from forgeloop.context import compact_messages, context_budget, estimate_tokens
from forgeloop.interactive import InteractiveCLI
from forgeloop.model_capabilities import (
    CapabilityResolver,
    ModelCache,
    ModelCapability,
    thinking_parameters,
)
from forgeloop.provider_config import (
    configured_provider_names,
    fetch_provider_models,
)
from forgeloop.tui import SLASH_COMMANDS


class Credentials:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}

    def get_api_key(self, provider: str) -> str | None:
        return self.values.get(provider)

    def set_api_key(self, provider: str, value: str) -> None:
        self.values[provider] = value

    def delete_api_key(self, provider: str) -> None:
        self.values.pop(provider, None)


class Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_only_complete_provider_is_exposed_to_model_selector() -> None:
    config = GlobalConfig(provider_configs={"deepseek": {}})
    credentials = Credentials({"deepseek": "secret"})

    assert configured_provider_names(config, credentials) == ["deepseek"]
    assert "/provider" not in SLASH_COMMANDS
    assert "/providers" not in SLASH_COMMANDS
    assert "/thinking" in SLASH_COMMANDS


def test_refresh_models_updates_isolated_cache_and_old_cache_survives_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GlobalConfig(provider_configs={"deepseek": {}})
    credentials = Credentials({"deepseek": "secret"})
    cache = ModelCache(tmp_path)
    payload = {
        "object": "list",
        "data": [
            {"id": "deepseek-v4-flash", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-v4-pro", "object": "model", "owned_by": "deepseek"},
        ],
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: Response(payload)
    )

    models = fetch_provider_models(config, credentials, "deepseek", cache=cache)
    assert models == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert cache.models("deepseek", "https://api.deepseek.com") == models
    assert cache.models("deepseek", "https://mirror.invalid") == []
    assert "secret" not in cache.path.read_text(encoding="utf-8")

    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    with pytest.raises(ValueError, match="获取模型失败"):
        fetch_provider_models(config, credentials, "deepseek", cache=cache)
    assert cache.models("deepseek", "https://api.deepseek.com") == models


def test_capability_priority_and_unknown_are_explicit(tmp_path: Path) -> None:
    cache = ModelCache(tmp_path)
    cache.update(
        "deepseek",
        "https://api.deepseek.com",
        [("deepseek-v4-flash", {"id": "deepseek-v4-flash", "context_window": 900_000})],
    )
    resolver = CapabilityResolver(cache)
    capability = resolver.resolve(
        "deepseek",
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        {"context_window": 800_000},
    )
    assert capability.context_window == 800_000
    assert capability.max_output_tokens == 384_000
    assert capability.source["context_window"] == "provider_api"

    unknown = resolver.resolve("custom", "https://example.invalid", "private-model")
    assert unknown.context_window is None
    assert unknown.tool_calling is None
    assert context_budget(unknown).usable_context is None


def test_thinking_is_model_aware_and_mapped_only_to_real_levels() -> None:
    assert thinking_parameters("deepseek", "deepseek-v4-flash", "high") == {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert thinking_parameters("deepseek", "deepseek-v4-flash", "auto") == {}
    with pytest.raises(ValueError, match="不支持"):
        thinking_parameters("deepseek", "deepseek-v4-flash", "medium")


def test_context_budget_changes_with_model_limits() -> None:
    small = context_budget(
        ModelCapability(context_window=100_000, max_output_tokens=10_000)
    )
    large = context_budget(
        ModelCapability(context_window=1_000_000, max_output_tokens=100_000)
    )
    assert small.usable_context == 84_000
    assert small.auto_compact_threshold == 71_400
    assert large.usable_context == 840_000
    assert large.auto_compact_threshold == 714_000


def test_manual_compact_summarizes_even_one_conversation_turn() -> None:
    messages = [
        {"role": "user", "content": "修复问题"},
        {"role": "assistant", "content": "已完成"},
    ]
    compacted, stats = compact_messages(
        messages,
        force=True,
        context_state={"original_task": "修复问题", "completed": ["已完成"]},
    )
    assert stats["compacted"] == 1
    assert compacted[0]["role"] == "system"
    assert "Original task" in compacted[0]["content"]


def test_switch_to_smaller_context_compacts_before_selecting(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    store = ConfigStore(tmp_path / "home")
    config = GlobalConfig(provider_configs={"deepseek": {}})
    store.save(config)
    controller = InteractiveCLI(
        cwd=repo,
        config_store=store,
        credential_store=Credentials({"deepseek": "secret"}),
        write=lambda value: None,
    )
    assert controller._create_session(repo)
    assert controller.session is not None
    controller.model_cache.update(
        "deepseek",
        "https://api.deepseek.com",
        [("small-model", {"context_window": 100_000, "max_output_tokens": 10_000})],
    )
    controller.session.conversation = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 24_000}
        for index in range(12)
    ]
    before = estimate_tokens(controller.session.conversation)

    controller.select_model("deepseek", "small-model")

    assert before > 71_400
    assert controller.session.compact_count == 1
    assert estimate_tokens(controller.session.conversation) < 84_000
    assert controller.session.model == "small-model"
