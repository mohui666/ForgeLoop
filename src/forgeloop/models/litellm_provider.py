from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from forgeloop.model_capabilities import thinking_parameters
from forgeloop.types import Message, ModelResponse, ModelUsage, ToolCall
from forgeloop.models.base import ModelProviderError


@dataclass
class LiteLLMProvider:
    """LiteLLM adapter. No LiteLLM response object crosses this boundary."""

    model: str
    api_base: str | None = None
    temperature: float | None = None
    api_key: str | None = field(default=None, repr=False)
    thinking_level: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return self.model

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict],
        *,
        timeout_seconds: float,
    ) -> ModelResponse:
        try:
            import litellm

            litellm.suppress_debug_info = True
            completion = litellm.completion
        except ImportError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError(
                "LiteLLM is not installed; run `uv sync` or install forgeloop"
            ) from exc

        provider_name, _, bare_model = self.model.partition("/")
        thinking = thinking_parameters(
            provider_name, bare_model or provider_name, self.thinking_level
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_litellm_messages(messages),
            "tools": list(tools),
            "timeout": timeout_seconds,
            **self.extra,
        }
        for key, value in thinking.items():
            if key == "extra_body" and isinstance(kwargs.get(key), dict):
                kwargs[key] = {**kwargs[key], **value}
            else:
                kwargs[key] = value
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.api_key:
            kwargs["api_key"] = self.api_key

        started = time.perf_counter()
        try:
            response = completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize provider failures
            message = str(exc)
            if self.api_key:
                message = message.replace(self.api_key, "[REDACTED]")
            message = re.sub(
                r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,\"']+",
                r"\1[REDACTED]",
                message,
            )
            raise ModelProviderError(
                self._friendly_error(message),
                details=f"{type(exc).__name__}: {message}",
            ) from None
        latency = time.perf_counter() - started
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            function = raw_call.function
            arguments = function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Model returned invalid JSON arguments for tool {function.name}: {exc}"
                    ) from exc
            calls.append(
                ToolCall(id=raw_call.id, name=function.name, arguments=arguments or {})
            )

        raw_usage = getattr(response, "usage", None)
        input_tokens = self._optional_int(
            self._get(raw_usage, "prompt_tokens")
            or self._get(raw_usage, "input_tokens")
        )
        output_tokens = self._optional_int(
            self._get(raw_usage, "completion_tokens")
            or self._get(raw_usage, "output_tokens")
        )
        total_tokens = self._optional_int(self._get(raw_usage, "total_tokens"))
        prompt_details = self._get(raw_usage, "prompt_tokens_details")
        completion_details = self._get(raw_usage, "completion_tokens_details")
        cached_tokens = self._optional_int(
            self._get(raw_usage, "cached_tokens")
            or self._get(prompt_details, "cached_tokens")
            or self._get(raw_usage, "prompt_cache_hit_tokens")
            or self._get(raw_usage, "cache_read_input_tokens")
        )
        reasoning_tokens = self._optional_int(
            self._get(raw_usage, "reasoning_tokens")
            or self._get(completion_details, "reasoning_tokens")
        )
        hidden = getattr(response, "_hidden_params", {}) or {}
        provider = hidden.get("custom_llm_provider") or self.model.partition("/")[0]
        response_model = getattr(response, "model", None) or self.model
        cost, cost_source = self._cost(response, hidden, str(response_model))
        assistant_fields: dict[str, Any] = {}
        reasoning_content = getattr(message, "reasoning_content", None)
        provider_fields = getattr(message, "provider_specific_fields", None) or {}
        if not reasoning_content and isinstance(provider_fields, dict):
            reasoning_content = provider_fields.get("reasoning_content")
        if reasoning_content is not None:
            assistant_fields["reasoning_content"] = reasoning_content
        elif calls and provider == "deepseek":
            # DeepSeek thinking-mode tool turns require this field on every
            # subsequent request, even when the API omitted an empty chain.
            assistant_fields["reasoning_content"] = " "
        metadata = {
            "response_id": getattr(response, "id", None),
            "provider": provider,
        }
        return ModelResponse(
            content=getattr(message, "content", None),
            tool_calls=tuple(calls),
            usage=ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                reported_total_tokens=total_tokens,
                usage_source="provider_response" if raw_usage is not None else None,
                cost_source=cost_source,
                model=str(response_model),
                provider=str(provider) if provider else None,
                latency_seconds=latency,
            ),
            finish_reason=getattr(choice, "finish_reason", None),
            provider_metadata={k: v for k, v in metadata.items() if v is not None},
            assistant_message_fields=assistant_fields,
        )

    def _cost(
        self, response: Any, hidden: dict[str, Any], response_model: str
    ) -> tuple[float | None, str]:
        provider_reported = self._get(getattr(response, "usage", None), "cost")
        if provider_reported is not None:
            return float(provider_reported), "provider_reported"
        try:
            import litellm

            candidates = {self.model, response_model}
            if "/" not in response_model and "/" in self.model:
                candidates.add(f"{self.model.partition('/')[0]}/{response_model}")
            has_exact_pricing = any(
                candidate in litellm.model_cost for candidate in candidates
            )
            if not has_exact_pricing:
                return None, "unknown"
            hidden_cost = hidden.get("response_cost")
            if hidden_cost is not None:
                return float(hidden_cost), "litellm_calculated"
            return (
                float(
                    litellm.completion_cost(
                        completion_response=response, model=self.model
                    )
                ),
                "litellm_calculated",
            )
        except Exception:  # noqa: BLE001 - unavailable pricing must remain unknown
            return None, "unknown"

    @staticmethod
    def _get(value: Any, key: str) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return int(value) if value is not None else None

    @staticmethod
    def _to_litellm_messages(messages: Sequence[Message]) -> list[Message]:
        converted: list[Message] = []
        for source in messages:
            message = dict(source)
            if "tool_calls" in message:
                calls = []
                for source_call in message["tool_calls"]:
                    call = dict(source_call)
                    function = dict(call["function"])
                    if not isinstance(function.get("arguments"), str):
                        function["arguments"] = json.dumps(
                            function.get("arguments", {}), ensure_ascii=False
                        )
                    call["function"] = function
                    calls.append(call)
                message["tool_calls"] = calls
            converted.append(message)
        return converted

    @staticmethod
    def _friendly_error(message: str) -> str:
        lowered = message.lower()
        redacted = " [REDACTED]" if "[redacted]" in lowered else ""
        if any(
            marker in lowered
            for marker in (
                "401",
                "unauthorized",
                "authorization",
                "authentication",
                "api key",
            )
        ):
            return f"Provider authentication failed. Check /api key, then run /api test.{redacted}"
        if any(
            marker in lowered
            for marker in ("404", "model_not_found", "does not exist", "model access")
        ):
            return (
                "Model is unavailable for this account or route. Check /api and /model."
            )
        if any(marker in lowered for marker in ("timeout", "timed out")):
            return "Provider request timed out. Retry or increase timeout_seconds."
        if any(marker in lowered for marker in ("connection", "dns", "network")):
            return "Could not connect to the provider endpoint. Check /api base and network access."
        return f"Provider request failed. Open details for the redacted diagnostic message.{redacted}"
