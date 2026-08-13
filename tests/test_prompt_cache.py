from forgeloop.prompt_cache import (
    PromptCacheTracker,
    request_prefix_fingerprint,
    tool_schema_fingerprint,
)
from forgeloop.types import ModelUsage


def test_warm_prefix_measurement_excludes_cold_start_and_caps_provider_hits() -> None:
    tracker = PromptCacheTracker()

    cold = tracker.record(
        ModelUsage(
            input_tokens=2_000,
            cached_tokens=640,
            cache_miss_tokens=1_360,
            provider="deepseek",
            model="v4",
        ),
        compaction_epoch=0,
        prefix_fingerprint="stable",
        request_started_at=10.0,
        backend_fingerprint="fp-a",
    )
    second = tracker.record(
        ModelUsage(
            input_tokens=2_200,
            cached_tokens=2_048,
            cache_miss_tokens=152,
            provider="deepseek",
            model="v4",
        ),
        compaction_epoch=0,
        prefix_fingerprint="stable",
        request_started_at=12.5,
        backend_fingerprint="fp-a",
    )
    third = tracker.record(
        ModelUsage(input_tokens=2_500, cached_tokens=2_816),
        compaction_epoch=0,
        prefix_fingerprint="stable",
    )

    assert cold["status"] == "cold_start"
    assert cold["provider_prompt_accounting_valid"] is True
    assert second["reusable_prefix_tokens"] == 2_000
    assert second["reused_prefix_tokens"] == 2_000
    assert second["missed_reusable_tokens"] == 0
    assert second["request_interval_seconds"] == 2.5
    assert second["backend_changed"] is False
    assert third["reusable_prefix_tokens"] == 2_200
    assert third["reused_prefix_tokens"] == 2_200
    assert tracker.snapshot() == {
        "warm_cache_reusable_tokens": 4_200,
        "warm_cache_reused_tokens": 4_200,
        "warm_cache_missed_tokens": 0,
        "warm_cache_hit_ratio": 1.0,
        "warm_cache_measured_calls": 2,
        "warm_cache_significant_miss_calls": 0,
        "warm_cache_reset_calls": 0,
    }


def test_compaction_and_request_prefix_changes_reset_measurement() -> None:
    tracker = PromptCacheTracker()
    tracker.record(
        ModelUsage(input_tokens=4_000, cached_tokens=0),
        compaction_epoch=0,
        prefix_fingerprint="a",
    )

    compacted = tracker.record(
        ModelUsage(input_tokens=2_000, cached_tokens=0),
        compaction_epoch=1,
        prefix_fingerprint="b",
    )
    schema_changed = tracker.record(
        ModelUsage(input_tokens=2_100, cached_tokens=0),
        compaction_epoch=1,
        prefix_fingerprint="c",
    )

    assert compacted["status"] == "reset"
    assert compacted["reset_reason"] == "compaction_epoch_changed"
    assert schema_changed["status"] == "reset"
    assert schema_changed["reset_reason"] == "request_prefix_changed"
    assert tracker.snapshot()["warm_cache_measured_calls"] == 0
    assert tracker.snapshot()["warm_cache_reset_calls"] == 2


def test_missing_provider_cache_usage_is_unavailable_not_zero() -> None:
    tracker = PromptCacheTracker()
    first = tracker.record(
        ModelUsage(input_tokens=2_000, cached_tokens=None),
        compaction_epoch=0,
        prefix_fingerprint="stable",
    )
    unavailable = tracker.record(
        ModelUsage(input_tokens=2_100, cached_tokens=None),
        compaction_epoch=0,
        prefix_fingerprint="stable",
    )

    assert first["status"] == "unavailable"
    assert unavailable["status"] == "unavailable"
    assert unavailable["warm_prefix_hit_ratio"] is None
    assert tracker.snapshot()["warm_cache_measured_calls"] == 0


def test_only_misses_above_pi_noise_floor_are_significant() -> None:
    tracker = PromptCacheTracker()
    tracker.record(
        ModelUsage(input_tokens=5_000, cached_tokens=0),
        compaction_epoch=0,
        prefix_fingerprint="stable",
    )
    noise = tracker.record(
        ModelUsage(input_tokens=5_100, cached_tokens=4_000),
        compaction_epoch=0,
        prefix_fingerprint="stable",
    )
    miss = tracker.record(
        ModelUsage(input_tokens=5_200, cached_tokens=3_000),
        compaction_epoch=0,
        prefix_fingerprint="stable",
    )

    assert noise["missed_reusable_tokens"] == 1_000
    assert noise["significant_miss"] is False
    assert miss["missed_reusable_tokens"] == 2_100
    assert miss["significant_miss"] is True
    assert tracker.snapshot()["warm_cache_significant_miss_calls"] == 1


def test_backend_change_is_diagnostic_not_a_measurement_reset() -> None:
    tracker = PromptCacheTracker()
    tracker.record(
        ModelUsage(input_tokens=5_000, cached_tokens=0),
        compaction_epoch=0,
        prefix_fingerprint="stable",
        backend_fingerprint="fp-a",
    )
    changed = tracker.record(
        ModelUsage(input_tokens=5_100, cached_tokens=3_000),
        compaction_epoch=0,
        prefix_fingerprint="stable",
        backend_fingerprint="fp-b",
    )

    assert changed["status"] == "warm"
    assert changed["backend_changed"] is True
    assert changed["miss_classification"] == "backend_changed"


def test_request_prefix_fingerprint_is_stable_for_appended_history() -> None:
    tools = [{"type": "function", "function": {"name": "read"}}]
    base = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "task"},
    ]
    appended = [*base, {"role": "assistant", "content": "tail"}]

    assert request_prefix_fingerprint(
        base, tools, base_message_count=2
    ) == request_prefix_fingerprint(appended, tools, base_message_count=2)
    assert tool_schema_fingerprint(tools) != tool_schema_fingerprint([])
