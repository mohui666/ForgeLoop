from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from forgeloop.types import Message, ModelUsage

PI_CACHE_MISS_NOISE_FLOOR_TOKENS = 1_024


def tool_schema_fingerprint(tools: Sequence[dict[str, Any]]) -> str:
    """Return a stable, content-only identity for the provider tool prefix."""

    payload = json.dumps(
        list(tools),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def request_prefix_fingerprint(
    messages: Sequence[Message],
    tools: Sequence[dict[str, Any]],
    *,
    base_message_count: int,
) -> str:
    """Identify the request prefix that should remain stable inside an epoch.

    The digest is telemetry only. It deliberately includes the tool schemas and
    the immutable task/session prefix, but excludes the append-only action tail.
    """

    count = max(0, min(base_message_count, len(messages)))
    payload = json.dumps(
        {"messages": list(messages[:count]), "tools": list(tools)},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class PromptCacheTracker:
    """Measure warm prefix reuse separately from unavoidable cold input.

    This follows Pi's cache-waste semantics: after the first request, at most
    ``min(previous prompt, current prompt)`` tokens were reusable. Compaction or
    a request-prefix/schema change starts a new cache epoch and therefore does
    not count as waste.
    """

    previous_prompt_tokens: int | None = None
    previous_epoch: int | None = None
    previous_prefix_fingerprint: str | None = None
    previous_request_started_at: float | None = None
    previous_model_key: str | None = None
    previous_backend_fingerprint: str | None = None
    warm_reusable_tokens: int = 0
    warm_reused_tokens: int = 0
    warm_missed_tokens: int = 0
    measured_warm_calls: int = 0
    significant_miss_calls: int = 0
    reset_calls: int = 0

    def record(
        self,
        usage: ModelUsage,
        *,
        compaction_epoch: int,
        prefix_fingerprint: str,
        request_started_at: float | None = None,
        backend_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        prompt_tokens = usage.input_tokens
        cached_tokens = usage.cached_tokens
        provider_miss_tokens = usage.cache_miss_tokens
        previous_prompt_tokens = self.previous_prompt_tokens
        model_key = (
            f"{usage.provider}/{usage.model}"
            if usage.provider or usage.model
            else None
        )
        request_interval_seconds = (
            max(0.0, request_started_at - self.previous_request_started_at)
            if request_started_at is not None
            and self.previous_request_started_at is not None
            else None
        )
        model_changed = bool(
            self.previous_model_key
            and model_key
            and model_key != self.previous_model_key
        )
        backend_changed = bool(
            self.previous_backend_fingerprint
            and backend_fingerprint
            and backend_fingerprint != self.previous_backend_fingerprint
        )
        status = "warm"
        reset_reason: str | None = None

        if prompt_tokens is None or cached_tokens is None:
            status = "unavailable"
        elif self.previous_prompt_tokens is None:
            status = "cold_start"
        elif self.previous_epoch != compaction_epoch:
            status = "reset"
            reset_reason = "compaction_epoch_changed"
        elif self.previous_prefix_fingerprint != prefix_fingerprint:
            status = "reset"
            reset_reason = "request_prefix_changed"

        reusable_tokens: int | None = None
        reused_tokens: int | None = None
        missed_tokens: int | None = None
        warm_hit_ratio: float | None = None
        significant_miss = False
        miss_classification: str | None = None
        if status == "warm":
            assert self.previous_prompt_tokens is not None
            assert prompt_tokens is not None
            assert cached_tokens is not None
            reusable_tokens = min(self.previous_prompt_tokens, prompt_tokens)
            reused_tokens = min(reusable_tokens, max(0, cached_tokens))
            missed_tokens = max(0, reusable_tokens - reused_tokens)
            warm_hit_ratio = (
                round(reused_tokens / reusable_tokens, 6)
                if reusable_tokens
                else None
            )
            significant_miss = missed_tokens > PI_CACHE_MISS_NOISE_FLOOR_TOKENS
            self.warm_reusable_tokens += reusable_tokens
            self.warm_reused_tokens += reused_tokens
            self.warm_missed_tokens += missed_tokens
            self.measured_warm_calls += 1
            if significant_miss:
                self.significant_miss_calls += 1
                if model_changed:
                    miss_classification = "model_changed"
                elif backend_changed:
                    miss_classification = "backend_changed"
                else:
                    miss_classification = "stable_prefix_provider_miss"
            elif missed_tokens:
                miss_classification = "prefix_unit_noise"
        elif status == "reset":
            self.reset_calls += 1

        if prompt_tokens is not None:
            self.previous_prompt_tokens = prompt_tokens
            self.previous_epoch = compaction_epoch
            self.previous_prefix_fingerprint = prefix_fingerprint
        if request_started_at is not None:
            self.previous_request_started_at = request_started_at
        if model_key is not None:
            self.previous_model_key = model_key
        if backend_fingerprint is not None:
            self.previous_backend_fingerprint = backend_fingerprint

        provider_accounting_valid = (
            prompt_tokens == cached_tokens + provider_miss_tokens
            if prompt_tokens is not None
            and cached_tokens is not None
            and provider_miss_tokens is not None
            else None
        )
        provider_accounting_delta = (
            prompt_tokens - cached_tokens - provider_miss_tokens
            if prompt_tokens is not None
            and cached_tokens is not None
            and provider_miss_tokens is not None
            else None
        )

        return {
            "schema_version": "forgeloop.prompt-cache.pi-parity.v1",
            "status": status,
            "reset_reason": reset_reason,
            "prompt_tokens": prompt_tokens,
            "provider_cached_tokens": cached_tokens,
            "provider_cache_miss_tokens": provider_miss_tokens,
            "provider_prompt_accounting_valid": provider_accounting_valid,
            "provider_prompt_accounting_delta": provider_accounting_delta,
            "previous_prompt_tokens": (
                None if status == "cold_start" else previous_prompt_tokens
            ),
            "reusable_prefix_tokens": reusable_tokens,
            "reused_prefix_tokens": reused_tokens,
            "missed_reusable_tokens": missed_tokens,
            "warm_prefix_hit_ratio": warm_hit_ratio,
            "significant_miss": significant_miss,
            "miss_classification": miss_classification,
            "noise_floor_tokens": PI_CACHE_MISS_NOISE_FLOOR_TOKENS,
            "request_prefix_fingerprint": prefix_fingerprint,
            "request_interval_seconds": (
                round(request_interval_seconds, 3)
                if request_interval_seconds is not None
                else None
            ),
            "model_key": model_key,
            "model_changed": model_changed,
            "backend_fingerprint": backend_fingerprint,
            "backend_changed": backend_changed,
        }

    def snapshot(self) -> dict[str, Any]:
        ratio = (
            round(self.warm_reused_tokens / self.warm_reusable_tokens, 6)
            if self.warm_reusable_tokens
            else None
        )
        return {
            "warm_cache_reusable_tokens": self.warm_reusable_tokens,
            "warm_cache_reused_tokens": self.warm_reused_tokens,
            "warm_cache_missed_tokens": self.warm_missed_tokens,
            "warm_cache_hit_ratio": ratio,
            "warm_cache_measured_calls": self.measured_warm_calls,
            "warm_cache_significant_miss_calls": self.significant_miss_calls,
            "warm_cache_reset_calls": self.reset_calls,
        }
