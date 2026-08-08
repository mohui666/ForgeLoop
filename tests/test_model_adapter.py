import json
import sys
from types import SimpleNamespace

import pytest

from forgeloop.models.litellm_provider import LiteLLMProvider


def test_tool_arguments_are_encoded_for_provider() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "a.py"}},
                }
            ],
        }
    ]
    converted = LiteLLMProvider._to_litellm_messages(messages)
    arguments = converted[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"path": "a.py"}
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], dict)


def test_complete_normalizes_litellm_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="read_file", arguments='{"path":"a.py"}'),
        )
        message = SimpleNamespace(
            content="thinking",
            tool_calls=[tool_call],
            reasoning_content="private reasoning state",
        )
        return SimpleNamespace(
            id="response-1",
            model="model",
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
                prompt_cache_hit_tokens=4,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
            _hidden_params={
                "response_cost": 0.02,
                "custom_llm_provider": "mock",
            },
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=completion, model_cost={"mock/model": {}}),
    )
    provider = LiteLLMProvider(
        "mock/model",
        api_base="https://example.invalid",
        temperature=0.1,
        extra={"seed": 7},
    )
    response = provider.complete(
        [{"role": "user", "content": "read"}],
        [{"type": "function"}],
        timeout_seconds=5,
    )

    assert captured["model"] == "mock/model"
    assert captured["api_base"] == "https://example.invalid"
    assert captured["seed"] == 7
    assert response.tool_calls[0].arguments == {"path": "a.py"}
    assert response.usage.total_tokens == 15
    assert response.usage.cost_usd == 0.02
    assert response.usage.cached_tokens == 4
    assert response.usage.reasoning_tokens == 2
    assert response.usage.usage_source == "provider_response"
    assert response.usage.cost_source == "litellm_calculated"
    assert response.usage.latency_seconds is not None
    assert (
        response.as_assistant_message()["reasoning_content"]
        == "private reasoning state"
    )
    assert response.provider_metadata == {
        "response_id": "response-1",
        "provider": "mock",
    }


def test_provider_error_redacts_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "temporary-provider-secret"

    def completion(**kwargs):
        raise RuntimeError(f"Authorization: Bearer {kwargs['api_key']}")

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    provider = LiteLLMProvider("mock/model", api_key=secret)
    with pytest.raises(RuntimeError) as caught:
        provider.complete([], [], timeout_seconds=1)
    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_deepseek_tool_turn_keeps_required_reasoning_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completion(**kwargs):
        del kwargs
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="read_file", arguments='{"path":"a.py"}'),
        )
        return SimpleNamespace(
            id="response-1",
            model="deepseek-v4-flash",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="", tool_calls=[tool_call], reasoning_content=None
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
            _hidden_params={"custom_llm_provider": "deepseek"},
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=completion, model_cost={}),
    )
    response = LiteLLMProvider("deepseek/deepseek-v4-flash").complete(
        [], [], timeout_seconds=1
    )

    assert response.as_assistant_message()["reasoning_content"] == " "


def test_model_provider_maps_deepseek_thinking_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="response-thinking",
            model="deepseek-v4-flash",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
            _hidden_params={"custom_llm_provider": "deepseek"},
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=completion, model_cost={}),
    )
    LiteLLMProvider("deepseek/deepseek-v4-flash", thinking_level="max").complete(
        [], [], timeout_seconds=1
    )

    assert captured["reasoning_effort"] == "max"
    assert captured["extra_body"] == {"thinking": {"type": "enabled"}}
