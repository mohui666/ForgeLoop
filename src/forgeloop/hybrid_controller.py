from __future__ import annotations

import json
import re
import time
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from forgeloop.controller import ControllerRecovery, ControllerV1
from forgeloop.tools.base import ToolResult
from forgeloop.types import ToolCall


HYBRID_CONTROLLER_V11_ID = "forgeloop.controller.hybrid.v1.1"
CONTROLLER_POLICY_SCHEMA_VERSION = "forgeloop.controller-policy.v1"
DEFAULT_CONTROLLER_POLICY = "qwen2.5-1.5b-controller-local"
CONTROLLER_POLICY_ASSETS = {
    DEFAULT_CONTROLLER_POLICY: (
        Path(__file__).with_name("controller_assets") / "qwen2.5-1.5b-hybrid-v1.1.json"
    )
}

STATES = ("explore", "implement", "verify", "finalize")
NEXT_ACTIONS = ("inspect", "edit", "test", "replan", "finalize")
ALLOWED_DECISIONS = {
    ("explore", "inspect"),
    ("implement", "edit"),
    ("verify", "test"),
    ("verify", "replan"),
    ("finalize", "finalize"),
}
SIGNAL_DECISIONS = {
    "needs_inspection": ("explore", "inspect"),
    "inspected_no_diff": ("implement", "edit"),
    "modified_untested": ("verify", "test"),
    "tests_failed": ("verify", "replan"),
    "tests_passed": ("finalize", "finalize"),
}
DECISION_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "state": {"const": state},
                "next_action": {"const": next_action},
            },
            "required": ["state", "next_action"],
            "additionalProperties": False,
        }
        for state, next_action in sorted(ALLOWED_DECISIONS)
    ]
}

_TEST_COMMAND = re.compile(
    r"(?:^|[;&|\s])(?:pytest|python(?:3)?\s+-m\s+(?:pytest|unittest)|"
    r"cargo\s+test|go\s+test|dotnet\s+test|npm\s+(?:run\s+)?test|"
    r"pnpm\s+(?:run\s+)?test|yarn\s+(?:run\s+)?test)(?:\s|$)",
    re.IGNORECASE,
)


class ControllerPolicyError(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class ControllerPolicyConfig:
    policy_id: str
    model: str
    api_base: str
    model_identity: str
    quantization: str
    num_ctx: int
    num_predict: int
    timeout_seconds: float
    keep_alive: str
    local_api_cost_usd: float

    @classmethod
    def load(
        cls, reference: str = DEFAULT_CONTROLLER_POLICY
    ) -> "ControllerPolicyConfig":
        source = CONTROLLER_POLICY_ASSETS.get(reference.lower(), Path(reference))
        raw = json.loads(source.expanduser().resolve().read_text(encoding="utf-8"))
        if raw.get("schema_version") != CONTROLLER_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"Controller policy schema must be {CONTROLLER_POLICY_SCHEMA_VERSION}"
            )
        required = (
            "policy_id",
            "model",
            "api_base",
            "model_identity",
            "quantization",
            "num_ctx",
            "num_predict",
            "timeout_seconds",
            "keep_alive",
            "local_api_cost_usd",
        )
        missing = [key for key in required if raw.get(key) is None]
        if missing:
            raise ValueError("Controller policy missing fields: " + ", ".join(missing))
        return cls(**{key: raw[key] for key in required})


@dataclass(frozen=True)
class HybridDecision:
    state: str
    next_action: str

    @classmethod
    def from_value(cls, value: Any) -> "HybridDecision":
        if not isinstance(value, dict):
            raise ControllerPolicyError(
                "schema_validation", "decision must be an object"
            )
        if set(value) != {"state", "next_action"}:
            raise ControllerPolicyError(
                "schema_validation",
                "decision must contain exactly state and next_action",
            )
        state = value.get("state")
        next_action = value.get("next_action")
        if not isinstance(state, str) or state not in STATES:
            raise ControllerPolicyError("schema_validation", "invalid state enum")
        if not isinstance(next_action, str) or next_action not in NEXT_ACTIONS:
            raise ControllerPolicyError("schema_validation", "invalid next_action enum")
        if (state, next_action) not in ALLOWED_DECISIONS:
            raise ControllerPolicyError(
                "semantic_validation", "state and next_action are not an allowed pair"
            )
        return cls(state, next_action)


@dataclass(frozen=True)
class ControllerPolicyResult:
    decision: HybridDecision
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None


class ControllerPolicy(Protocol):
    config: ControllerPolicyConfig

    def decide(self, snapshot: Mapping[str, Any]) -> ControllerPolicyResult: ...


class OllamaControllerPolicy:
    """Strict, local finite-state classifier using Ollama structured outputs."""

    def __init__(
        self,
        config: ControllerPolicyConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config or ControllerPolicyConfig.load()
        self._transport = transport

    def decide(self, snapshot: Mapping[str, Any]) -> ControllerPolicyResult:
        started = time.perf_counter()
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return one JSON decision. Map progress_signal exactly: "
                        "needs_inspection => explore/inspect; inspected_no_diff => "
                        "implement/edit; modified_untested => verify/test; "
                        "tests_failed => verify/replan; tests_passed => "
                        "finalize/finalize. Do not use any other pair. Never write "
                        "code, plans, explanations, repository content, or supervisor "
                        "text."
                    ),
                },
                {
                    "role": "user",
                    "content": _controller_input(snapshot),
                },
            ],
            "stream": False,
            "format": DECISION_SCHEMA,
            "options": {
                "temperature": 0,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
            },
            "keep_alive": self.config.keep_alive,
        }
        try:
            with httpx.Client(
                base_url=self.config.api_base,
                timeout=self.config.timeout_seconds,
                trust_env=False,
                transport=self._transport,
            ) as client:
                response = client.post("/api/chat", json=payload)
                response.raise_for_status()
                raw = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ControllerPolicyError(
                "provider_unavailable",
                f"local controller request failed: {type(exc).__name__}",
            ) from exc
        try:
            content = raw["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ControllerPolicyError(
                "invalid_json", "local controller returned invalid structured content"
            ) from exc
        decision = HybridDecision.from_value(parsed)
        expected = SIGNAL_DECISIONS.get(str(snapshot.get("progress_signal") or ""))
        if expected and (decision.state, decision.next_action) != expected:
            raise ControllerPolicyError(
                "semantic_validation",
                "decision does not match the compact progress signal",
            )
        return ControllerPolicyResult(
            decision=decision,
            latency_seconds=round(time.perf_counter() - started, 6),
            input_tokens=_optional_int(raw.get("prompt_eval_count")),
            output_tokens=_optional_int(raw.get("eval_count")),
        )


class HybridControllerV11(ControllerV1):
    """Model-assisted stage selection with deterministic safety boundaries."""

    identity = HYBRID_CONTROLLER_V11_ID

    def __init__(
        self,
        policy: ControllerPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy or OllamaControllerPolicy()
        self._state = "explore"
        self._next_action = "inspect"
        self._test_status = "unknown"
        self._recent: deque[dict[str, Any]] = deque(maxlen=4)
        self._policy_events: list[tuple[str, dict[str, Any]]] = []
        self._policy_counts: Counter[str] = Counter()
        self._transition_counts: Counter[str] = Counter()
        self._policy_input_tokens = 0
        self._policy_output_tokens = 0
        self._policy_latency_seconds = 0.0

    def observe_tool(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
        budget_snapshot: dict[str, Any] | None = None,
    ) -> tuple[ControllerRecovery, ...]:
        recoveries = list(
            super().observe_tool(
                call,
                result,
                before_fingerprint=before_fingerprint,
                after_fingerprint=after_fingerprint,
                budget_snapshot=budget_snapshot,
            )
        )
        changed = before_fingerprint != after_fingerprint
        category = _action_category(call.name)
        if call.name == "shell" and _TEST_COMMAND.search(
            str(call.arguments.get("command") or "")
        ):
            category = "test"
            self._test_status = "pass" if result.ok else "fail"
        elif changed:
            self._test_status = "unknown"
        self._recent.append(
            {"action": category, "ok": bool(result.ok), "source_changed": changed}
        )
        snapshot = self._snapshot(
            source_diff=bool(
                self._initial_fingerprint
                and after_fingerprint != self._initial_fingerprint
            ),
            budget_snapshot=budget_snapshot,
        )
        previous = (self._state, self._next_action)
        try:
            result_value = self.policy.decide(snapshot)
        except ControllerPolicyError as exc:
            self._policy_counts["fallbacks"] += 1
            self._policy_events.append(
                (
                    "controller_policy_fallback",
                    {
                        "controller": self.identity,
                        "controller_policy": self.policy.config.policy_id,
                        "fallback": True,
                        "error_category": exc.category,
                        "input": snapshot,
                    },
                )
            )
            return tuple(recoveries)

        decision = result_value.decision
        self._state = decision.state
        self._next_action = decision.next_action
        self._policy_counts["decisions"] += 1
        self._policy_input_tokens += result_value.input_tokens or 0
        self._policy_output_tokens += result_value.output_tokens or 0
        self._policy_latency_seconds += result_value.latency_seconds
        transition = (
            f"{previous[0]}/{previous[1]}->{decision.state}/{decision.next_action}"
        )
        changed_decision = previous != (decision.state, decision.next_action)
        if changed_decision:
            self._transition_counts[transition] += 1
        self._policy_events.append(
            (
                "controller_policy_decision",
                {
                    "controller": self.identity,
                    "controller_policy": self.policy.config.policy_id,
                    "model": self.policy.config.model,
                    "model_identity": self.policy.config.model_identity,
                    "state": decision.state,
                    "next_action": decision.next_action,
                    "previous_state": previous[0],
                    "previous_action": previous[1],
                    "transitioned": changed_decision,
                    "schema_valid": True,
                    "fallback": False,
                    "input": snapshot,
                    "input_tokens": result_value.input_tokens,
                    "output_tokens": result_value.output_tokens,
                    "latency_seconds": result_value.latency_seconds,
                    "local_api_cost_usd": self.policy.config.local_api_cost_usd,
                },
            )
        )
        if changed_decision:
            recoveries.append(
                self._recovery(
                    "hybrid_stage_guidance",
                    transition,
                    _fixed_guidance(decision),
                )
            )
        return tuple(recoveries)

    def guard_action(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | None:
        deterministic = super().guard_action(
            call, current_fingerprint=current_fingerprint
        )
        if deterministic:
            return deterministic
        if self._state in {"verify", "finalize"} and call.name in {
            "list_files",
            "search_files",
        }:
            return self._recovery(
                "hybrid_phase_action_blocked",
                f"{call.name} attempted during {self._state}",
                (
                    f"Hybrid Controller v1.1 blocked broad {call.name} during the "
                    f"{self._state} phase. Use the current diff and test evidence; "
                    "only inspect a known file/region when required, then test or "
                    "finish explicitly."
                ),
            )
        return None

    def drain_events(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        events = tuple(self._policy_events)
        self._policy_events.clear()
        return events

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary.update(
            {
                "controller_policy": {
                    "policy_id": self.policy.config.policy_id,
                    "model": self.policy.config.model,
                    "model_identity": self.policy.config.model_identity,
                    "quantization": self.policy.config.quantization,
                    "local_api_cost_usd": self.policy.config.local_api_cost_usd,
                },
                "current_state": self._state,
                "current_next_action": self._next_action,
                "decisions": self._policy_counts["decisions"],
                "fallbacks": self._policy_counts["fallbacks"],
                "transitions": dict(sorted(self._transition_counts.items())),
                "policy_usage": {
                    "input_tokens": self._policy_input_tokens,
                    "output_tokens": self._policy_output_tokens,
                    "latency_seconds": round(self._policy_latency_seconds, 6),
                    "cost_usd": 0.0,
                },
            }
        )
        return summary

    def _snapshot(
        self,
        *,
        source_diff: bool,
        budget_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "current": {"state": self._state, "next_action": self._next_action},
            "recent_tools": list(self._recent),
            "progress_signal": _progress_signal(
                source_diff, self._test_status, self._recent
            ),
            "source_diff": source_diff,
            "test_status": self._test_status,
            "remaining_budget": _remaining_budget(budget_snapshot),
        }


def probe_controller_policy(
    reference: str = DEFAULT_CONTROLLER_POLICY,
) -> dict[str, Any]:
    policy = OllamaControllerPolicy(ControllerPolicyConfig.load(reference))
    expected = {
        "needs_inspection": HybridDecision("explore", "inspect"),
        "inspected_no_diff": HybridDecision("implement", "edit"),
        "modified_untested": HybridDecision("verify", "test"),
        "tests_failed": HybridDecision("verify", "replan"),
        "tests_passed": HybridDecision("finalize", "finalize"),
    }
    decisions: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    latency_seconds = 0.0
    for signal, expected_decision in expected.items():
        recent_action = {
            "needs_inspection": None,
            "inspected_no_diff": "inspect",
            "modified_untested": "edit",
            "tests_failed": "test",
            "tests_passed": "test",
        }[signal]
        snapshot = {
            "current": {"state": "explore", "next_action": "inspect"},
            "recent_tools": (
                [
                    {
                        "action": recent_action,
                        "ok": signal != "tests_failed",
                        "source_changed": signal == "modified_untested",
                    }
                ]
                if recent_action
                else []
            ),
            "progress_signal": signal,
            "source_diff": signal
            in {"modified_untested", "tests_failed", "tests_passed"},
            "test_status": {
                "tests_failed": "fail",
                "tests_passed": "pass",
            }.get(signal, "unknown"),
            "remaining_budget": {
                "steps": 20,
                "model_calls": 20,
                "tool_calls": 60,
                "seconds": 600.0,
                "tokens": 100000,
            },
        }
        result = policy.decide(snapshot)
        if result.decision != expected_decision:
            raise ControllerPolicyError(
                "probe_mismatch",
                f"{signal} returned {result.decision.state}/{result.decision.next_action}",
            )
        input_tokens += result.input_tokens or 0
        output_tokens += result.output_tokens or 0
        latency_seconds += result.latency_seconds
        decisions.append(
            {
                "progress_signal": signal,
                "state": result.decision.state,
                "next_action": result.decision.next_action,
            }
        )
    return {
        "policy": policy.config.policy_id,
        "model": policy.config.model,
        "model_identity": policy.config.model_identity,
        "schema_valid": True,
        "decisions": decisions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": round(latency_seconds, 6),
        "local_api_cost_usd": policy.config.local_api_cost_usd,
    }


def _fixed_guidance(decision: HybridDecision) -> str:
    guidance = {
        "inspect": (
            "Hybrid Controller v1.1: phase=explore. Inspect one targeted file or "
            "symbol needed to identify the smallest supported edit."
        ),
        "edit": (
            "Hybrid Controller v1.1: phase=implement. Make the smallest supported "
            "source edit now; do not broaden repository exploration."
        ),
        "test": (
            "Hybrid Controller v1.1: phase=verify. Run the focused test or verifier "
            "for the current diff; do not broaden repository exploration."
        ),
        "replan": (
            "Hybrid Controller v1.1: phase=verify. The latest test failed. Inspect "
            "only failure-related context, correct the edit, and rerun that test."
        ),
        "finalize": (
            "Hybrid Controller v1.1: phase=finalize. Review the current diff and "
            "passing test evidence, satisfy any commit requirement, then call finish "
            "explicitly. Do not continue exploring."
        ),
    }
    return guidance[decision.next_action]


def _controller_input(snapshot: Mapping[str, Any]) -> str:
    signal = str(snapshot.get("progress_signal") or "needs_inspection")
    required = {
        "needs_inspection": "explore/inspect",
        "inspected_no_diff": "implement/edit",
        "modified_untested": "verify/test",
        "tests_failed": "verify/replan",
        "tests_passed": "finalize/finalize",
    }.get(signal, "explore/inspect")
    recent = snapshot.get("recent_tools") or []
    latest = recent[-1] if isinstance(recent, list) and recent else {}
    budget = snapshot.get("remaining_budget") or {}
    return "\n".join(
        (
            f"Trajectory progress signal: {signal}",
            (
                "Recent tool: "
                f"action={latest.get('action', 'none')}, "
                f"ok={str(latest.get('ok', 'none')).lower()}, "
                f"source_changed={str(latest.get('source_changed', False)).lower()}"
            ),
            f"Source diff exists: {str(bool(snapshot.get('source_diff'))).lower()}",
            f"Test status: {snapshot.get('test_status', 'unknown')}",
            "Remaining budget: " + json.dumps(budget, separators=(",", ":")),
            f"Required mapping for this progress signal: {required}",
        )
    )


def _action_category(tool_name: str) -> str:
    if tool_name == "apply_patch":
        return "edit"
    if tool_name in {"read_file", "search_files", "list_files", "git_inspect"}:
        return "inspect"
    if tool_name == "git_diff":
        return "diff"
    if tool_name == "shell":
        return "command"
    return "other"


def _remaining_budget(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {}
    limits = snapshot.get("limits") or {}
    usage = snapshot.get("usage") or {}

    def remaining(limit_key: str, usage_key: str) -> int | float | None:
        limit = limits.get(limit_key)
        used = usage.get(usage_key)
        if limit is None or used is None:
            return None
        return max(0, limit - used)

    return {
        "steps": remaining("max_steps", "steps"),
        "model_calls": remaining("max_model_calls", "model_calls"),
        "tool_calls": remaining("max_tool_calls", "tool_calls"),
        "seconds": remaining("max_seconds", "elapsed_seconds"),
        "tokens": remaining("max_tokens", "total_tokens"),
    }


def _progress_signal(
    source_diff: bool,
    test_status: str,
    recent: deque[dict[str, Any]],
) -> str:
    if source_diff and test_status == "pass":
        return "tests_passed"
    if source_diff and test_status == "fail":
        return "tests_failed"
    if source_diff:
        return "modified_untested"
    if recent and not recent[-1]["ok"]:
        return "needs_inspection"
    if any(item["action"] == "inspect" and item["ok"] for item in recent):
        return "inspected_no_diff"
    return "needs_inspection"


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


__all__ = [
    "ALLOWED_DECISIONS",
    "CONTROLLER_POLICY_ASSETS",
    "ControllerPolicyConfig",
    "ControllerPolicyError",
    "ControllerPolicyResult",
    "DECISION_SCHEMA",
    "DEFAULT_CONTROLLER_POLICY",
    "HYBRID_CONTROLLER_V11_ID",
    "HybridControllerV11",
    "HybridDecision",
    "OllamaControllerPolicy",
    "probe_controller_policy",
]
