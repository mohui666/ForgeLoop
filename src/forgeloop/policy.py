from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from forgeloop.model_capabilities import ModelCapability


POLICY_SCHEMA_VERSION = "forgeloop.policy.v1"
POLICY_STAGES = {"base", "sft", "rl"}
ACTIVE_OPEN_WEIGHT_POLICY = "qwen3.5-4b-local"
BUNDLED_POLICIES = {
    ACTIVE_OPEN_WEIGHT_POLICY: (
        Path(__file__).with_name("policy_assets") / "qwen3.5-4b-ollama.json"
    ),
    "qwen3.5-4b-sft-v1": (
        Path(__file__).with_name("policy_assets") / "qwen3.5-4b-sft-v1.json"
    ),
    "qwen3.5-4b-sft-v2": (
        Path(__file__).with_name("policy_assets") / "qwen3.5-4b-sft-v2.json"
    ),
    "deepseek-v4-flash-controller-v1": (
        Path(__file__).with_name("policy_assets")
        / "deepseek-v4-flash-controller-v1.json"
    ),
    "deepseek-v4-flash-hybrid-controller-v1.1": (
        Path(__file__).with_name("policy_assets")
        / "deepseek-v4-flash-hybrid-controller-v1.1.json"
    ),
    "deepseek-v4-flash-hybrid-controller-v1.2": (
        Path(__file__).with_name("policy_assets")
        / "deepseek-v4-flash-hybrid-controller-v1.2.json"
    ),
    "deepseek-v4-flash-edit-intent-v1": (
        Path(__file__).with_name("policy_assets")
        / "deepseek-v4-flash-edit-intent-v1.json"
    ),
    "deepseek-v4-flash-edit-intent-readiness-v1": (
        Path(__file__).with_name("policy_assets")
        / "deepseek-v4-flash-edit-intent-readiness-v1.json"
    ),
    "deepseek-v4-flash-controller-v1.3-simplified": (
        Path(__file__).with_name("policy_assets")
        / "deepseek-v4-flash-controller-v1.3-simplified.json"
    ),
    "deepseek-v4-flash-controller-v1.4-pi-output-window": (
        Path(__file__).with_name("policy_assets")
        / "deepseek-v4-flash-controller-v1.4-pi-output-window.json"
    ),
}
_GENERATION_KEYS = {
    "extra_body",
    "frequency_penalty",
    "max_tokens",
    "presence_penalty",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_p",
}
_SECRET_KEYS = {
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


class PolicyManifestError(ValueError):
    pass


@dataclass(frozen=True)
class PolicyIdentity:
    """Non-secret identity for a reproducible, redeployable model policy."""

    policy_id: str
    stage: str
    base_model: str
    model_revision: str
    tokenizer: str
    tokenizer_revision: str
    inference_backend: str
    litellm_model: str
    capabilities: ModelCapability
    serving_config: dict[str, Any] = field(default_factory=dict)
    generation_config: dict[str, Any] = field(default_factory=dict)
    schema_version: str = POLICY_SCHEMA_VERSION

    @classmethod
    def load(cls, path: Path | str) -> "PolicyIdentity":
        supplied = str(path)
        resolved = BUNDLED_POLICIES.get(supplied.lower(), Path(supplied))
        resolved = resolved.expanduser().resolve()
        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicyManifestError(
                f"Cannot read policy manifest {resolved}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise PolicyManifestError("Policy manifest must be a JSON object")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PolicyIdentity":
        if raw.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise PolicyManifestError(f"Policy schema must be {POLICY_SCHEMA_VERSION}")
        required = (
            "policy_id",
            "stage",
            "base_model",
            "model_revision",
            "tokenizer",
            "tokenizer_revision",
            "inference_backend",
            "litellm_model",
            "capabilities",
        )
        missing = [name for name in required if not raw.get(name)]
        if missing:
            raise PolicyManifestError(
                "Policy manifest missing fields: " + ", ".join(missing)
            )
        stage = str(raw["stage"]).lower()
        if stage not in POLICY_STAGES:
            raise PolicyManifestError("Policy stage must be base, sft, or rl")
        serving = _object(raw.get("serving_config"), "serving_config")
        generation = _object(raw.get("generation_config"), "generation_config")
        _reject_secrets(raw)
        unsupported = sorted(set(generation) - _GENERATION_KEYS)
        if unsupported:
            raise PolicyManifestError(
                "Unsupported generation fields: " + ", ".join(unsupported)
            )
        return cls(
            policy_id=str(raw["policy_id"]),
            stage=stage,
            base_model=str(raw["base_model"]),
            model_revision=str(raw["model_revision"]),
            tokenizer=str(raw["tokenizer"]),
            tokenizer_revision=str(raw["tokenizer_revision"]),
            inference_backend=str(raw["inference_backend"]),
            litellm_model=str(raw["litellm_model"]),
            capabilities=ModelCapability.from_dict(
                _object(raw["capabilities"], "capabilities")
            ),
            serving_config=serving,
            generation_config=generation,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = asdict(self.capabilities)
        return payload

    def with_serving_config(self, **overrides: Any) -> "PolicyIdentity":
        """Bind non-secret runtime route details to the recorded identity."""

        updated = {**self.serving_config, **overrides}
        _reject_secrets(updated, ("serving_config",))
        return replace(self, serving_config=updated)


def provider_policy_identity(provider: Any) -> dict[str, Any]:
    """Return explicit identity, or an honest legacy model-only fallback."""

    identity = getattr(provider, "policy_identity", None)
    if isinstance(identity, PolicyIdentity):
        return identity.to_dict()
    if isinstance(identity, dict):
        return dict(identity)
    model = str(getattr(provider, "model_id", "unknown"))
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": None,
        "stage": "unknown",
        "base_model": model,
        "model_revision": None,
        "tokenizer": None,
        "tokenizer_revision": None,
        "inference_backend": None,
        "litellm_model": model,
        "capabilities": {},
        "serving_config": {},
        "generation_config": {},
        "identity_status": "legacy_model_only",
    }


def resolve_policy_api_key(policy: PolicyIdentity) -> str:
    """Resolve a policy's credential by env-var name without persisting the secret."""

    credential_env = str(policy.serving_config.get("credential_env") or "").strip()
    if credential_env:
        value = os.getenv(credential_env)
        if not value:
            raise PolicyManifestError(
                f"Policy {policy.policy_id} requires environment variable {credential_env}."
            )
        return value
    return os.getenv("FORGELOOP_SELF_HOSTED_API_KEY") or "EMPTY"


def _object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyManifestError(f"Policy {name} must be an object")
    return dict(value)


def _reject_secrets(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SECRET_KEYS or normalized.endswith("_api_key"):
                location = ".".join((*path, str(key)))
                raise PolicyManifestError(
                    f"Policy manifests cannot contain credentials: {location}"
                )
            _reject_secrets(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, (*path, str(index)))


__all__ = [
    "ACTIVE_OPEN_WEIGHT_POLICY",
    "POLICY_SCHEMA_VERSION",
    "BUNDLED_POLICIES",
    "PolicyIdentity",
    "PolicyManifestError",
    "provider_policy_identity",
    "resolve_policy_api_key",
]
