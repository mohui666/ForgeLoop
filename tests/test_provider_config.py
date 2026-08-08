from __future__ import annotations

from forgeloop.config import GlobalConfig
from forgeloop.provider_config import (
    PreflightError,
    canonical_model_route,
    preflight_provider,
    test_provider_api as run_api_test,
)
from forgeloop.types import ModelResponse, ModelUsage, ToolCall


class Credentials:
    def __init__(self, key: str | None = "secret") -> None:
        self.key = key

    def get_api_key(self, provider: str) -> str | None:
        del provider
        return self.key


def test_bare_deepseek_model_becomes_canonical_route() -> None:
    assert (
        canonical_model_route("deepseek", "deepseek-v4-flash")
        == "deepseek/deepseek-v4-flash"
    )


def test_custom_openai_compatible_route_is_internalized() -> None:
    assert canonical_model_route("custom", "private-model") == "openai/private-model"


def test_preflight_rejects_mismatched_or_incomplete_route() -> None:
    try:
        canonical_model_route("deepseek", "openai/gpt-4.1")
    except PreflightError as exc:
        assert "不匹配" in str(exc)
    else:
        raise AssertionError("mismatched route was accepted")
    try:
        preflight_provider(
            GlobalConfig(provider="deepseek", model="deepseek-v4-flash"),
            Credentials(None),
        )
    except PreflightError as exc:
        assert "API Key" in str(exc)
    else:
        raise AssertionError("missing key was accepted")


def test_api_test_requires_valid_tool_call(monkeypatch) -> None:
    def complete(self, messages, tools, *, timeout_seconds):
        del self, messages, tools, timeout_seconds
        return ModelResponse(
            tool_calls=(
                ToolCall("probe", "forgeloop_probe", {"nonce": "forgeloop-ok"}),
            ),
            usage=ModelUsage(2, 1, 0.0),
        )

    monkeypatch.setattr("forgeloop.provider_config.LiteLLMProvider.complete", complete)
    result = run_api_test(
        GlobalConfig(provider="deepseek", model="deepseek-v4-flash"),
        Credentials(),
    )
    assert result.route == "deepseek/deepseek-v4-flash"
    assert result.tool_calling
