from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


def forgeloop_home() -> Path:
    override = os.getenv("FORGELOOP_HOME")
    return (
        Path(override).expanduser().resolve()
        if override
        else Path.home() / ".forgeloop"
    )


@dataclass
class GlobalConfig:
    provider: str = "openai"
    model: str = ""
    api_base: str = ""
    max_steps: int = 30
    max_model_calls: int = 30
    max_tool_calls: int = 80
    timeout_seconds: float = 900.0
    max_cost_usd: float = 0.0
    auto_compact_tokens: int = 24_000
    provider_configs: dict[str, dict[str, Any]] = field(default_factory=dict)


class ConfigStore:
    """Global, non-secret ForgeLoop preferences."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = (home or forgeloop_home()).expanduser().resolve()
        self.path = self.home / "config.json"

    def load(self) -> GlobalConfig:
        if not self.path.exists():
            return GlobalConfig()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            allowed = GlobalConfig.__dataclass_fields__.keys()
            config = GlobalConfig(**{key: raw[key] for key in allowed if key in raw})
            # Preserve old non-secret provider/base settings without ever moving
            # credentials into JSON. The legacy fields remain for headless CLI
            # compatibility and as the last selected interactive route.
            if config.api_base and config.provider not in config.provider_configs:
                config.provider_configs[config.provider] = {"api_base": config.api_base}
            return config
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Cannot read {self.path}: {exc}") from exc

    def save(self, config: GlobalConfig) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


class CredentialStore:
    """OS-backed API keys. No credential is ever written to ForgeLoop JSON files."""

    service = "ForgeLoop"

    def set_api_key(self, provider: str, value: str) -> None:
        if not value.strip():
            raise ConfigError("API key cannot be empty")
        try:
            self._keyring().set_password(self.service, provider, value.strip())
        except Exception as exc:  # noqa: BLE001 - normalize backend failures
            raise ConfigError(f"OS credential store is unavailable: {exc}") from exc

    def get_api_key(self, provider: str) -> str | None:
        env_name = f"{provider.upper().replace('-', '_')}_API_KEY"
        environment = os.getenv(env_name)
        if environment:
            return environment
        try:
            return self._keyring().get_password(self.service, provider)
        except Exception as exc:  # noqa: BLE001 - normalize backend failures
            raise ConfigError(f"OS credential store is unavailable: {exc}") from exc

    def delete_api_key(self, provider: str) -> None:
        keyring = self._keyring()
        try:
            keyring.delete_password(self.service, provider)
        except keyring.errors.PasswordDeleteError:
            pass
        except Exception as exc:  # noqa: BLE001 - normalize backend failures
            raise ConfigError(f"OS credential store is unavailable: {exc}") from exc

    @staticmethod
    def _keyring() -> Any:
        try:
            import keyring
        except ImportError as exc:
            raise ConfigError(
                "Secure credential storage is unavailable; install the 'keyring' dependency "
                "or provide the provider API key through an environment variable."
            ) from exc
        return keyring


PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "key_env": "OPENAI_API_KEY",
        "default_base": "https://api.openai.com/v1",
        "models_api": "openai",
    },
    "anthropic": {
        "label": "Anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "default_base": "https://api.anthropic.com/v1",
        "models_api": "anthropic",
    },
    "deepseek": {
        "label": "DeepSeek",
        "key_env": "DEEPSEEK_API_KEY",
        "default_base": "https://api.deepseek.com",
        "models_api": "openai",
    },
    "gemini": {
        "label": "Gemini",
        "key_env": "GEMINI_API_KEY",
        "default_base": "https://generativelanguage.googleapis.com/v1beta",
        "models_api": "gemini",
    },
    "azure": {
        "label": "Azure OpenAI",
        "key_env": "AZURE_API_KEY",
        "default_base": "",
        "models_api": "manual",
    },
    "ollama": {
        "label": "Ollama",
        "key_env": "",
        "default_base": "http://localhost:11434",
        "models_api": "ollama",
    },
    "custom": {
        "label": "OpenAI Compatible",
        "key_env": "CUSTOM_API_KEY",
        "default_base": "",
        "models_api": "openai",
        "key_optional": True,
        "route_prefix": "openai",
    },
}


def provider_api_base(config: GlobalConfig, provider: str) -> str:
    settings = config.provider_configs.get(provider, {})
    configured = str(settings.get("api_base", "")).strip()
    if configured:
        return configured.rstrip("/")
    if provider == config.provider and config.api_base:
        return config.api_base.rstrip("/")
    return str(PROVIDERS.get(provider, {}).get("default_base", "")).rstrip("/")


def set_provider_api_base(config: GlobalConfig, provider: str, api_base: str) -> None:
    settings = dict(config.provider_configs.get(provider, {}))
    settings["api_base"] = api_base.strip().rstrip("/")
    config.provider_configs[provider] = settings
    if provider == config.provider:
        config.api_base = settings["api_base"]
