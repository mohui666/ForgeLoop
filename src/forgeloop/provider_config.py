from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from forgeloop.config import (
    PROVIDERS,
    ConfigError,
    CredentialStore,
    GlobalConfig,
    provider_api_base,
)
from forgeloop.model_capabilities import ModelCache
from forgeloop.models import LiteLLMProvider
from forgeloop.types import Message, ModelUsage


class PreflightError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderRoute:
    provider: str
    model: str
    canonical_model: str
    api_base: str | None
    api_key: str | None


@dataclass(frozen=True)
class ApiTestResult:
    route: str
    tool_calling: bool
    usage: ModelUsage
    detail: str


def canonical_model_route(provider: str, model: str) -> str:
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in PROVIDERS:
        raise PreflightError(f"未知 Provider：{provider or '未设置'}")
    if not model:
        raise PreflightError("尚未配置 Model，请先使用 /model。")
    route_prefix = str(PROVIDERS[provider].get("route_prefix", provider))
    if "/" not in model:
        return f"{route_prefix}/{model}"
    route_provider, _, bare = model.partition("/")
    if route_provider.lower() not in {provider, route_prefix}:
        raise PreflightError(
            f"Model route '{route_provider}/{bare}' 与 Provider '{provider}' 不匹配。"
        )
    return f"{route_prefix}/{bare}"


def model_name_from_route(provider: str, model: str) -> str:
    metadata = PROVIDERS.get(provider.strip().lower(), {})
    prefix = str(metadata.get("route_prefix", provider)).strip().lower() + "/"
    return model[len(prefix) :] if model.lower().startswith(prefix) else model


def configured_provider_names(
    config: GlobalConfig, credentials: CredentialStore
) -> list[str]:
    usable: list[str] = []
    for provider, metadata in PROVIDERS.items():
        try:
            api_key = credentials.get_api_key(provider)
        except ConfigError:
            continue
        settings = config.provider_configs.get(provider, {})
        base = provider_api_base(config, provider)
        requires_key = bool(metadata.get("key_env")) and not metadata.get(
            "key_optional"
        )
        explicitly_configured = provider in config.provider_configs or bool(api_key)
        if requires_key and not api_key:
            continue
        if not requires_key and not explicitly_configured:
            continue
        if not base:
            continue
        if not requires_key and not api_key and not settings.get("connection_ok"):
            if not _endpoint_reachable(base):
                continue
        usable.append(provider)
    return usable


def preflight_provider(
    config: GlobalConfig,
    credentials: CredentialStore,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> ProviderRoute:
    selected_provider = (provider or config.provider).strip().lower()
    selected_model = model if model is not None else config.model
    canonical = canonical_model_route(selected_provider, selected_model)
    try:
        api_key = credentials.get_api_key(selected_provider)
    except ConfigError as exc:
        raise PreflightError(str(exc)) from exc
    metadata = PROVIDERS[selected_provider]
    requires_key = bool(metadata.get("key_env")) and not metadata.get("key_optional")
    if requires_key and not api_key:
        raise PreflightError(
            f"{metadata['label']} API Key 未配置，请使用 /api 配置后再试。"
        )
    api_base = provider_api_base(config, selected_provider)
    if not api_base:
        raise PreflightError(f"{metadata['label']} Base URL 未配置，请使用 /api 配置。")
    return ProviderRoute(
        provider=selected_provider,
        model=model_name_from_route(selected_provider, canonical),
        canonical_model=canonical,
        api_base=api_base,
        api_key=api_key,
    )


def fetch_provider_models(
    config: GlobalConfig,
    credentials: CredentialStore,
    provider: str,
    *,
    cache: ModelCache | None = None,
    timeout_seconds: float = 15.0,
) -> list[str]:
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise PreflightError(f"未知 Provider：{provider}")
    try:
        api_key = credentials.get_api_key(provider)
    except ConfigError as exc:
        raise PreflightError(str(exc)) from exc
    metadata = PROVIDERS[provider]
    if metadata.get("key_env") and not metadata.get("key_optional") and not api_key:
        raise PreflightError(f"{metadata['label']} API Key 未配置。")
    base = provider_api_base(config, provider)
    if not base:
        raise PreflightError(f"{metadata['label']} Base URL 未配置。")
    style = metadata.get("models_api")
    if style == "manual":
        raise PreflightError(
            f"{metadata['label']} 暂不支持自动获取模型，请手动输入 Model ID。"
        )

    headers = {"Accept": "application/json", "User-Agent": "ForgeLoop/0.1"}
    if style == "anthropic":
        endpoint = f"{base}/models"
        headers["x-api-key"] = api_key or ""
        headers["anthropic-version"] = "2023-06-01"
    elif style == "gemini":
        endpoint = f"{base}/models"
        headers["x-goog-api-key"] = api_key or ""
    elif style == "ollama":
        endpoint = f"{base}/api/tags"
    else:
        endpoint = f"{base}/models"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _http_error_detail(exc, api_key)
        raise PreflightError(
            f"获取模型失败：HTTP {exc.code} {detail}".rstrip()
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise PreflightError(f"获取模型失败：{_safe_error(exc, api_key)}") from None

    records: list[tuple[str, dict[str, Any]]] = []
    if style == "ollama":
        for item in payload.get("models", []):
            if isinstance(item, dict) and item.get("name"):
                records.append((str(item["name"]), {"id": item["name"]}))
    else:
        for item in payload.get("data", payload.get("models", [])):
            if not isinstance(item, dict):
                continue
            model_id = item.get("id") or item.get("name")
            if style == "gemini" and isinstance(model_id, str):
                model_id = model_id.removeprefix("models/")
            if model_id:
                records.append((str(model_id), item))
    if not records:
        raise PreflightError("Provider 返回成功，但响应中没有可识别的模型。")
    (cache or ModelCache()).update(provider, base, records)
    settings = dict(config.provider_configs.get(provider, {}))
    settings["connection_ok"] = True
    config.provider_configs[provider] = settings
    return sorted({model for model, _ in records}, key=str.casefold)


def test_provider_api(
    config: GlobalConfig,
    credentials: CredentialStore,
    *,
    provider: str | None = None,
    model: str | None = None,
    thinking_level: str = "auto",
    timeout_seconds: float = 45.0,
) -> ApiTestResult:
    route = preflight_provider(config, credentials, provider=provider, model=model)
    nonce = "forgeloop-ok"
    probe_schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "forgeloop_probe",
            "description": "Return the supplied nonce to verify tool calling.",
            "parameters": {
                "type": "object",
                "properties": {"nonce": {"type": "string"}},
                "required": ["nonce"],
                "additionalProperties": False,
            },
        },
    }
    messages: list[Message] = [
        {
            "role": "user",
            "content": f"Call forgeloop_probe exactly once with nonce '{nonce}'. Do not answer in text.",
        }
    ]
    provider_adapter = LiteLLMProvider(
        model=route.canonical_model,
        api_base=route.api_base,
        api_key=route.api_key,
        thinking_level=thinking_level,
    )
    response = provider_adapter.complete(
        messages, [probe_schema], timeout_seconds=timeout_seconds
    )
    valid = any(
        call.name == "forgeloop_probe" and call.arguments.get("nonce") == nonce
        for call in response.tool_calls
    )
    if not valid:
        raise PreflightError("认证和 Model 访问成功，但模型没有返回有效的 tool call。")
    return ApiTestResult(
        route.canonical_model,
        True,
        response.usage,
        "Authentication、model access 与 tool calling 均验证通过。",
    )


def _http_error_detail(exc: urllib.error.HTTPError, api_key: str | None) -> str:
    try:
        raw = exc.read(512).decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        message = parsed.get("error", {}).get("message") or parsed.get("message")
        return _safe_error(ValueError(str(message or "")), api_key)[:240]
    except Exception:  # noqa: BLE001 - diagnostic only
        return ""


def _safe_error(exc: BaseException, api_key: str | None) -> str:
    text = str(exc)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return text[:300]


def _endpoint_reachable(api_base: str, timeout_seconds: float = 0.4) -> bool:
    try:
        parsed = urllib.parse.urlparse(api_base)
        if not parsed.hostname:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((parsed.hostname, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False
