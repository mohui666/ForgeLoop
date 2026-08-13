from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from forgeloop.types import ModelUsage


@dataclass(frozen=True)
class BudgetLimits:
    max_steps: int = 30
    max_model_calls: int = 30
    max_tool_calls: int = 80
    max_seconds: float = 900.0
    max_cost_usd: float | None = None
    # Repeat limit is a configurable hard boundary only for a contiguous streak
    # with identical action, observation, and unchanged workspace evidence. The
    # error and mutation thresholds are recovery advisories, not terminal budgets.
    max_repeated_tool_calls: int = 3
    max_repeated_errors: int = 3
    max_no_progress_steps: int = 6

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "max_model_calls",
            "max_tool_calls",
            "max_repeated_tool_calls",
            "max_repeated_errors",
            "max_no_progress_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if self.max_cost_usd is not None and self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive or None")


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetState:
    limits: BudgetLimits
    started_at: float = 0.0
    steps: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    cached_tokens: int | None = 0
    reasoning_tokens: int | None = 0
    cost_usd: float | None = 0.0
    usage_records: int = 0
    usage_sources: set[str] = field(default_factory=set)
    cost_sources: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    providers: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.started_at == 0.0:
            self.started_at = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.max_seconds - self.elapsed_seconds)

    def check_before_step(self) -> None:
        if self.steps >= self.limits.max_steps:
            raise BudgetExceeded(f"step budget exceeded ({self.limits.max_steps})")
        if self.model_calls >= self.limits.max_model_calls:
            raise BudgetExceeded(
                f"model call budget exceeded ({self.limits.max_model_calls})"
            )
        self.check_time()
        self.check_cost()

    def check_time(self) -> None:
        if self.elapsed_seconds >= self.limits.max_seconds:
            raise BudgetExceeded(
                f"time budget exceeded ({self.limits.max_seconds:.0f}s)"
            )

    def reserve_tool_calls(self, count: int) -> None:
        if self.tool_calls + count > self.limits.max_tool_calls:
            raise BudgetExceeded(
                f"tool call budget exceeded ({self.limits.max_tool_calls})"
            )

    def begin_model_call(self) -> None:
        self.steps += 1
        self.model_calls += 1

    def record_usage(self, usage: ModelUsage) -> None:
        self.usage_records += 1
        self.input_tokens = self._add_optional(self.input_tokens, usage.input_tokens)
        self.output_tokens = self._add_optional(self.output_tokens, usage.output_tokens)
        self.cached_tokens = self._add_optional(self.cached_tokens, usage.cached_tokens)
        self.reasoning_tokens = self._add_optional(
            self.reasoning_tokens, usage.reasoning_tokens
        )
        self.cost_usd = self._add_optional(self.cost_usd, usage.cost_usd)
        if usage.usage_source:
            self.usage_sources.add(usage.usage_source)
        self.cost_sources.add(usage.cost_source)
        if usage.model:
            self.models.add(usage.model)
        if usage.provider:
            self.providers.add(usage.provider)

    def check_cost(self) -> None:
        """Reject another model call after prior cost exhausted its safety limit.

        Cumulative input, cached, reasoning, and output tokens are accounting
        telemetry, not an execution horizon. Cost remains an optional independent
        safety limit and is checked between calls so a returned validation or
        ``finish`` action is never discarded.
        """
        if self.limits.max_cost_usd is not None and self.cost_usd is None:
            raise BudgetExceeded(
                "cost budget cannot be enforced because provider cost is unknown"
            )
        if (
            self.limits.max_cost_usd is not None
            and self.cost_usd > self.limits.max_cost_usd
        ):
            raise BudgetExceeded(
                f"cost budget exceeded (${self.limits.max_cost_usd:.4f})"
            )

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @staticmethod
    def _add_optional(current: float | None, value: float | None) -> int | float | None:
        if current is None or value is None:
            return None
        return current + value

    def snapshot(self) -> dict:
        complete = self.usage_records == self.model_calls
        has_usage = self.usage_records > 0
        input_tokens = self.input_tokens if has_usage else None
        output_tokens = self.output_tokens if has_usage else None
        total_tokens = self.total_tokens if has_usage else None
        cached_tokens = self.cached_tokens if has_usage else None
        cached_input_ratio = (
            round(cached_tokens / input_tokens, 6)
            if input_tokens and cached_tokens is not None
            else None
        )
        reasoning_tokens = self.reasoning_tokens if has_usage else None
        cost_usd = self.cost_usd if has_usage else None
        return {
            "limits": asdict(self.limits),
            "usage": {
                "steps": self.steps,
                "model_calls": self.model_calls,
                "tool_calls": self.tool_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
                "cached_input_ratio": cached_input_ratio,
                "reasoning_tokens": reasoning_tokens,
                "cost_usd": (round(cost_usd, 8) if cost_usd is not None else None),
                "usage_sources": sorted(self.usage_sources),
                "cost_sources": sorted(self.cost_sources),
                "models": sorted(self.models),
                "providers": sorted(self.providers),
                "usage_complete": complete,
                "usage_records": self.usage_records,
                "unavailable_model_calls": max(
                    0, self.model_calls - self.usage_records
                ),
                "elapsed_seconds": round(self.elapsed_seconds, 3),
            },
        }
