from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from forgeloop.config import forgeloop_home
from forgeloop.persistence import advisory_file_lock, atomic_write_text


THINKING_LEVELS = ("auto", "low", "medium", "high", "max")


@dataclass(frozen=True)
class ModelCapability:
    context_window: int | None = None
    max_output_tokens: int | None = None
    thinking: bool | None = None
    thinking_levels: tuple[str, ...] = ()
    tool_calling: bool | None = None
    streaming: bool | None = None
    source: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelCapability":
        values = dict(raw)
        values["thinking_levels"] = tuple(values.get("thinking_levels", ()))
        values["source"] = dict(values.get("source", {}))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: values[key] for key in allowed if key in values})


# Entries here are deliberately small and backed by provider documentation.
# Unknown models remain unknown; ForgeLoop never inherits a nearby model's limits.
VERIFIED_REGISTRY: dict[tuple[str, str], ModelCapability] = {
    ("deepseek", "deepseek-v4-flash"): ModelCapability(
        context_window=1_000_000,
        max_output_tokens=384_000,
        thinking=True,
        thinking_levels=("auto", "high", "max"),
        tool_calling=True,
        streaming=True,
        source={
            "context_window": "forgeloop_registry:deepseek_official",
            "max_output_tokens": "forgeloop_registry:deepseek_official",
            "thinking": "forgeloop_registry:deepseek_official",
            "thinking_levels": "forgeloop_registry:deepseek_official",
            "tool_calling": "forgeloop_registry:deepseek_official",
            "streaming": "forgeloop_registry:deepseek_official",
        },
    ),
    ("deepseek", "deepseek-v4-pro"): ModelCapability(
        context_window=1_000_000,
        max_output_tokens=384_000,
        thinking=True,
        thinking_levels=("auto", "high", "max"),
        tool_calling=True,
        streaming=True,
        source={
            "context_window": "forgeloop_registry:deepseek_official",
            "max_output_tokens": "forgeloop_registry:deepseek_official",
            "thinking": "forgeloop_registry:deepseek_official",
            "thinking_levels": "forgeloop_registry:deepseek_official",
            "tool_calling": "forgeloop_registry:deepseek_official",
            "streaming": "forgeloop_registry:deepseek_official",
        },
    ),
}


class ModelCache:
    """Non-secret model/capability cache isolated by provider and base URL."""

    def __init__(self, home: Path | None = None, *, lock_timeout: float = 5.0) -> None:
        if lock_timeout < 0:
            raise ValueError("lock_timeout must be non-negative")
        self.path = (
            home or forgeloop_home()
        ).expanduser().resolve() / "model_cache.json"
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.lock_timeout = lock_timeout
        self._lock = RLock()

    @staticmethod
    def route_key(provider: str, api_base: str) -> str:
        return f"{provider.strip().lower()}|{api_base.strip().rstrip('/').lower()}"

    def models(self, provider: str, api_base: str) -> list[str]:
        with self._lock:
            route = (
                self._load()
                .get("routes", {})
                .get(self.route_key(provider, api_base), {})
            )
        return sorted(route.get("models", {}).keys(), key=str.casefold)

    def capability(
        self, provider: str, api_base: str, model: str
    ) -> ModelCapability | None:
        with self._lock:
            route = (
                self._load()
                .get("routes", {})
                .get(self.route_key(provider, api_base), {})
            )
        raw = route.get("models", {}).get(model, {}).get("capability")
        return ModelCapability.from_dict(raw) if isinstance(raw, dict) else None

    def update(
        self,
        provider: str,
        api_base: str,
        models: list[tuple[str, dict[str, Any]]],
    ) -> None:
        with self._lock:
            with advisory_file_lock(self.lock_path, timeout=self.lock_timeout):
                payload = self._load()
                routes = payload.setdefault("routes", {})
                key = self.route_key(provider, api_base)
                previous = routes.get(key, {}).get("models", {})
                records: dict[str, Any] = {}
                for model, metadata in models:
                    old = previous.get(model, {})
                    capability = capability_from_provider_metadata(metadata)
                    records[model] = {
                        "provider_metadata": sanitized_metadata(metadata),
                        "capability": asdict(capability)
                        if capability
                        else old.get("capability"),
                    }
                for model, old in previous.items():
                    if old.get("manual") and model not in records:
                        records[model] = old
                routes[key] = {
                    "provider": provider,
                    "api_base": api_base,
                    "refreshed_at": datetime.now(timezone.utc).isoformat(),
                    "models": records,
                }
                self._save(payload)

    def remember_manual(self, provider: str, api_base: str, model: str) -> None:
        with self._lock:
            with advisory_file_lock(self.lock_path, timeout=self.lock_timeout):
                payload = self._load()
                routes = payload.setdefault("routes", {})
                key = self.route_key(provider, api_base)
                route = routes.setdefault(
                    key,
                    {"provider": provider, "api_base": api_base, "models": {}},
                )
                route.setdefault("models", {}).setdefault(model, {"manual": True})
                self._save(payload)

    def _save(self, payload: dict[str, Any]) -> None:
        atomic_write_text(
            self.path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "routes": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {"version": 1, "routes": {}}
        except (OSError, ValueError, json.JSONDecodeError):
            return {"version": 1, "routes": {}}


class CapabilityResolver:
    def __init__(self, cache: ModelCache) -> None:
        self.cache = cache

    def resolve(
        self,
        provider: str,
        api_base: str,
        model: str,
        provider_metadata: dict[str, Any] | None = None,
    ) -> ModelCapability:
        # Merge field-by-field in strict priority order. A missing field does not
        # erase lower-priority verified information.
        candidates = [
            capability_from_provider_metadata(provider_metadata or {}),
            self.cache.capability(provider, api_base, model),
            VERIFIED_REGISTRY.get((provider, model)),
        ]
        fields = (
            "context_window",
            "max_output_tokens",
            "thinking",
            "thinking_levels",
            "tool_calling",
            "streaming",
        )
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for name in fields:
            for candidate in candidates:
                if candidate is None:
                    continue
                value = getattr(candidate, name)
                known = bool(value) if name == "thinking_levels" else value is not None
                if known:
                    values[name] = value
                    sources[name] = candidate.source.get(name, "unknown")
                    break
        values["source"] = sources
        return ModelCapability(**values)


def capability_from_provider_metadata(
    metadata: dict[str, Any],
) -> ModelCapability | None:
    if not metadata:
        return None
    aliases = {
        "context_window": ("context_window", "context_length", "max_context_tokens"),
        "max_output_tokens": ("max_output_tokens", "output_token_limit"),
        "tool_calling": ("tool_calling", "supports_function_calling"),
        "streaming": ("streaming", "supports_streaming"),
        "thinking": ("thinking", "supports_reasoning"),
    }
    values: dict[str, Any] = {}
    source: dict[str, str] = {}
    for target, names in aliases.items():
        for name in names:
            if name in metadata and metadata[name] is not None:
                raw = metadata[name]
                values[target] = (
                    int(raw)
                    if target.endswith("tokens") or target == "context_window"
                    else bool(raw)
                )
                source[target] = "provider_api"
                break
    levels = metadata.get("thinking_levels") or metadata.get(
        "supported_reasoning_effort"
    )
    if isinstance(levels, (list, tuple)):
        valid = tuple(
            str(item).lower() for item in levels if str(item).lower() in THINKING_LEVELS
        )
        if valid:
            values["thinking_levels"] = tuple(dict.fromkeys(("auto", *valid)))
            source["thinking_levels"] = "provider_api"
    elif values.get("thinking") is True:
        values["thinking_levels"] = ("auto",)
        source["thinking_levels"] = "provider_api"
    if not values:
        return None
    values["source"] = source
    return ModelCapability(**values)


def sanitized_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "id",
        "owned_by",
        "context_window",
        "context_length",
        "max_context_tokens",
        "max_output_tokens",
        "output_token_limit",
        "tool_calling",
        "supports_function_calling",
        "streaming",
        "supports_streaming",
        "thinking",
        "supports_reasoning",
        "thinking_levels",
        "supported_reasoning_effort",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def thinking_parameters(provider: str, model: str, level: str) -> dict[str, Any]:
    """Map ForgeLoop levels to real provider parameters; no synthetic levels."""
    if level == "auto":
        return {}
    if provider == "deepseek" and model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        if level not in {"high", "max"}:
            raise ValueError(f"{model} 不支持 thinking: {level}")
        return {
            "reasoning_effort": level,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    raise ValueError(f"{provider}/{model} 的 thinking 参数映射未知")
