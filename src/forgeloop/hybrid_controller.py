from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import httpx

from forgeloop.agent_types import RunStatus
from forgeloop.controller import ControllerRecovery, ControllerTerminal, ControllerV1
from forgeloop.security import is_sensitive_path
from forgeloop.tools.base import BaseTool, ToolResult
from forgeloop.types import ToolCall


HYBRID_CONTROLLER_V11_ID = "forgeloop.controller.hybrid.v1.1"
HYBRID_CONTROLLER_V12_ID = "forgeloop.controller.hybrid.v1.2"
HYBRID_CONTROLLER_EDIT_INTENT_ID = "forgeloop.controller.hybrid.v1.2.edit-intent.v1"
HYBRID_CONTROLLER_READINESS_ID = (
    "forgeloop.controller.hybrid.v1.2.edit-intent.readiness.v1"
)
HYBRID_CONTROLLER_V13_SIMPLIFIED_ID = "forgeloop.controller.hybrid.v1.3.simplified"
EDIT_INTENT_TOOL_NAME = "submit_edit_intent"
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
    r"(?:^|[;&|\s])(?:pytest|uv\s+run\s+pytest|"
    r"python(?:3)?\s+-m\s+(?:pytest|unittest)|cargo\s+(?:test|nextest)|"
    r"go\s+test|dotnet\s+test|npm\s+(?:run\s+)?test|"
    r"pnpm\s+(?:run\s+)?test|yarn\s+(?:run\s+)?test|"
    r"deno\s+test|bun\s+test|npx\s+[^;&|]*test|tox|ctest|"
    r"mvn\s+test|gradle\s+test|"
    r"\.\/?[^\s]*test\.sh|make\s+(?:test|check))(?:\s|$)",
    re.IGNORECASE,
)
_BROAD_SHELL = re.compile(
    r"(?:^|[;&|\s])(?:ls|dir|tree|find|fd|rg\s+--files|"
    r"grep\s+-R|Get-ChildItem)(?:\s|$)",
    re.IGNORECASE,
)
_HISTORY_SHELL = re.compile(r"\bgit\s+(?:log|show)\b", re.IGNORECASE)
_DIFF_SHELL = re.compile(r"\bgit\s+(?:diff|status)\b", re.IGNORECASE)
_FINALIZE_SHELL = re.compile(r"\bgit\s+(?:add|commit)(?:\s|$)", re.IGNORECASE)
_FORMAT_SHELL = re.compile(
    r"(?:^|[;&|\s])(?:cargo\s+fmt|ruff\s+(?:check|format)|"
    r"prettier|black|isort|gofmt)(?:\s|$)",
    re.IGNORECASE,
)
_READ_SHELL = re.compile(
    r"(?:^|[;&|\s])(?:cat|sed|head|tail|rg|grep|type|Get-Content)(?:\s|$)",
    re.IGNORECASE,
)
_STATE_BASE_TOOLS = {
    "read_file",
    "apply_patch",
    "git_diff",
    "git_inspect",
    "shell",
    "finish",
}
_SOURCE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cjs",
    ".clj",
    ".cljs",
    ".cpp",
    ".cs",
    ".cxx",
    ".erl",
    ".ex",
    ".exs",
    ".fs",
    ".fsx",
    ".go",
    ".h",
    ".hpp",
    ".hrl",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".lua",
    ".mjs",
    ".php",
    ".ps1",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".svelte",
    ".swift",
    ".ts",
    ".tsx",
    ".vue",
}


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
    guidance_version = "v1.1"
    advisory_only = False
    emit_stage_guidance = True

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
        self._source_files_read: set[str] = set()
        self._candidate_target_files: set[str] = set()
        self._saw_test_evidence = False
        self._saw_error_evidence = False
        self._source_diff_exists = False

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
        self._source_diff_exists = bool(
            self._initial_fingerprint and after_fingerprint != self._initial_fingerprint
        )
        self._record_implementation_evidence(call, result, category=category)
        self._recent.append(
            {"action": category, "ok": bool(result.ok), "source_changed": changed}
        )
        snapshot = self._snapshot(
            source_diff=self._source_diff_exists,
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
                    "advisory": self.advisory_only,
                },
            )
        )
        if changed_decision and self.emit_stage_guidance:
            recoveries.append(
                self._recovery(
                    "hybrid_stage_guidance",
                    transition,
                    _fixed_guidance(decision, self.guidance_version),
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
                "implementation_readiness": self._implementation_readiness(),
            }
        )
        return summary

    def _snapshot(
        self,
        *,
        source_diff: bool,
        budget_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        readiness = self._implementation_readiness(source_diff=source_diff)
        return {
            "current": {"state": self._state, "next_action": self._next_action},
            "recent_tools": list(self._recent),
            "progress_signal": _progress_signal(
                source_diff, self._test_status, self._recent
            ),
            "source_diff": source_diff,
            "test_status": self._test_status,
            "implementation_readiness": readiness,
            "remaining_budget": _remaining_budget(budget_snapshot),
        }

    def _record_implementation_evidence(
        self, call: ToolCall, result: ToolResult, *, category: str
    ) -> None:
        if not result.ok:
            self._saw_error_evidence = True
        if category == "test":
            self._saw_test_evidence = True
        if call.name != "read_file" or not result.ok or not result.output.strip():
            return
        path = _intent_path(str(call.arguments.get("path") or ""))
        if not _is_source_file(path):
            return
        self._source_files_read.add(path)
        self._candidate_target_files.add(path)
        if _is_test_path(path):
            self._saw_test_evidence = True

    def _implementation_readiness(
        self, *, source_diff: bool | None = None
    ) -> dict[str, Any]:
        has_diff = self._source_diff_exists if source_diff is None else source_diff
        has_intent = bool(getattr(self, "_edit_intent", None))
        return {
            "ready": bool(self._source_files_read and self._candidate_target_files),
            "source_content_read": bool(self._source_files_read),
            "source_files_read": sorted(self._source_files_read),
            "candidate_target_files": sorted(self._candidate_target_files),
            "saw_test_evidence": self._saw_test_evidence,
            "saw_error_evidence": self._saw_error_evidence,
            "has_diff": has_diff,
            "has_intent": has_intent,
        }


class HybridControllerV13Simplified(HybridControllerV11):
    """Advisory classification around one finite, evidence-backed execution loop."""

    identity = HYBRID_CONTROLLER_V13_SIMPLIFIED_ID
    guidance_version = "execution-closure-v2"
    advisory_only = True
    emit_stage_guidance = False

    implementation_min_exploration_calls = 6
    validation_advisory_calls = 8
    repair_advisory_calls = 8
    review_advisory_calls = 4
    validation_attempt_advisory = 6

    def __init__(
        self,
        policy: ControllerPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)
        self._execution_phase = "explore"
        self._current_fingerprint = ""
        self._validated_fingerprint = ""
        self._reviewed_fingerprint = ""
        self._phase_started_calls = 0
        self._phase_started_tokens = 0
        self._exploration_warned = False
        self._phase_advisories: set[str] = set()
        self._finish_rejections = 0
        self._validation_attempts = 0
        self._validation_passes = 0
        self._validation_failures = 0
        self._validation_source_changes = 0
        self._validation_commands: list[str] = []
        self._first_pass_calls: int | None = None
        self._first_pass_tokens: int | None = None
        self._last_pass_calls: int | None = None
        self._last_pass_tokens: int | None = None
        self._reviewed_calls: int | None = None
        self._reviewed_tokens: int | None = None
        self._ever_validated = False
        self._auto_finished = False
        self._last_budget: dict[str, Any] = {}

    def filter_tool_schemas(
        self, schemas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return ControllerV1.filter_tool_schemas(self, schemas)

    def guard_action(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | None:
        del call, current_fingerprint
        return None

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
        snapshot = budget_snapshot or {}
        self._last_budget = snapshot
        self._current_fingerprint = after_fingerprint
        has_diff = self._has_progress(after_fingerprint)
        changed = before_fingerprint != after_fingerprint
        validation = self._is_validation_call(call)
        diff_review = self._is_diff_review(call)

        if not has_diff:
            self._validated_fingerprint = ""
            self._reviewed_fingerprint = ""
            self._set_phase("explore", snapshot)
            return tuple(recoveries)

        if validation:
            self._validation_attempts += 1
            command = str(call.arguments.get("command") or "")[:2_000]
            self._validation_commands.append(command)
            if changed:
                self._validation_source_changes += 1
                self._validated_fingerprint = ""
                self._reviewed_fingerprint = ""
                self._set_phase("needs_validation", snapshot)
                status = "invalidated"
                recoveries.append(
                    self._recovery(
                        "validation_changed_tree",
                        "validation command changed the deliverable tree",
                        "Execution closure: validation changed the source tree, so it "
                        "cannot validate its own output. Use validate again on the new "
                        "tree without editing files.",
                    )
                )
            elif result.ok:
                self._validation_passes += 1
                self._ever_validated = True
                self._validated_fingerprint = after_fingerprint
                self._reviewed_fingerprint = ""
                calls, tokens = self._usage(snapshot)
                if self._first_pass_calls is None:
                    self._first_pass_calls = calls
                    self._first_pass_tokens = tokens
                self._last_pass_calls = calls
                self._last_pass_tokens = tokens
                self._set_phase("needs_review", snapshot)
                self._test_status = "pass"
                status = "passed"
                recoveries.append(
                    self._recovery(
                        "validation_passed",
                        "explicit validation passed for the current tree",
                        "Execution closure: validation PASS is recorded for the current "
                        "tree. Review the complete diff with git_diff (or git_inspect "
                        "diff). Then either edit and revalidate, or call finish.",
                    )
                )
            else:
                self._validation_failures += 1
                self._validated_fingerprint = ""
                self._reviewed_fingerprint = ""
                self._set_phase("validation_failed", snapshot)
                self._test_status = "fail"
                status = "failed"
                recoveries.append(
                    self._recovery(
                        "validation_failed",
                        "explicit validation failed",
                        "Execution closure: use the concrete validation failure, fix the "
                        "tree, and run validate again. Do not claim completion.",
                    )
                )
            self._policy_events.append(
                (
                    "controller_execution_validation",
                    {
                        "controller": self.identity,
                        "attempt": self._validation_attempts,
                        "status": status,
                        "command": command,
                        "exit_code": (result.metadata or {}).get("exit_code"),
                        "source_changed": changed,
                        "phase": self._execution_phase,
                    },
                )
            )
            return tuple(recoveries)

        if changed:
            self._validated_fingerprint = ""
            self._reviewed_fingerprint = ""
            prior = self._execution_phase
            self._set_phase("needs_validation", snapshot)
            self._test_status = "unknown"
            if prior == "explore":
                recoveries.append(
                    self._recovery(
                        "source_edit_detected",
                        "first deliverable tree change",
                        "Execution closure: complete the coherent edit batch, then use "
                        "validate for a relevant behavioral or repository check. Shell "
                        "exploration is not validation.",
                    )
                )
            self._policy_events.append(
                (
                    "controller_execution_tree_changed",
                    {
                        "controller": self.identity,
                        "action": call.name,
                        "phase": self._execution_phase,
                        "validation_invalidated": prior
                        in {"needs_review", "ready_to_finish"},
                    },
                )
            )
            return tuple(recoveries)

        if (
            diff_review
            and result.ok
            and self._validated_fingerprint == after_fingerprint
            and self._execution_phase == "needs_review"
        ):
            self._reviewed_fingerprint = after_fingerprint
            self._set_phase("ready_to_finish", snapshot)
            self._reviewed_calls, self._reviewed_tokens = self._usage(snapshot)
            self._policy_events.append(
                (
                    "controller_execution_diff_reviewed",
                    {
                        "controller": self.identity,
                        "validation_attempt": self._validation_attempts,
                        "phase": self._execution_phase,
                    },
                )
            )
        return tuple(recoveries)

    def before_model_call(
        self,
        *,
        current_fingerprint: str,
        budget_snapshot: dict[str, Any],
    ) -> ControllerRecovery | ControllerTerminal | None:
        self._current_fingerprint = current_fingerprint
        self._last_budget = budget_snapshot
        calls, _ = self._usage(budget_snapshot)
        phase_calls = calls - self._phase_started_calls

        if self._execution_phase == "ready_to_finish" and phase_calls >= 1:
            self._auto_finished = True
            return ControllerTerminal(
                RunStatus.COMPLETED,
                "Validated changes were reviewed and finalized by the execution controller.",
                "The model received the final diff and made no subsequent tree change.",
                "controller_ready_auto_finish",
            )

        if self._execution_phase == "explore":
            readiness = self._implementation_readiness()
            ready_handoff = (
                calls >= self.implementation_min_exploration_calls
                and readiness["ready"]
            )
            if not self._exploration_warned and ready_handoff:
                self._exploration_warned = True
                return self._recovery(
                    "implementation_due",
                    "source evidence is ready for implementation",
                    "Execution closure advisory: converge on the best-supported "
                    "source edit now, or finish failed/blocked. Tool availability is "
                    "unchanged; this advisory does not authorize or block actions.",
                )

        if self._execution_phase == "needs_validation":
            if phase_calls >= self.validation_advisory_calls:
                return self._phase_advisory(
                    "validation_due",
                    "the current edited tree has not yet been validated",
                    "Execution closure advisory: complete the coherent edit batch, "
                    "then validate the current tree. This is not a deadline and does "
                    "not reduce the execution horizon.",
                )

        if self._execution_phase == "validation_failed":
            if (
                phase_calls >= self.repair_advisory_calls
                or self._validation_attempts >= self.validation_attempt_advisory
            ):
                return self._phase_advisory(
                    "repair_validation_due",
                    "validation is still failing on the current trajectory",
                    "Execution closure advisory: use the latest concrete failure to "
                    "repair and retest. This warning does not terminate execution or "
                    "limit further repair attempts.",
                )

        if (
            self._execution_phase == "needs_review"
            and phase_calls >= self.review_advisory_calls
        ):
            return self._phase_advisory(
                "diff_review_due",
                "the validated tree is awaiting complete diff review",
                "Execution closure advisory: review the complete current diff before "
                "finishing. This warning is not a terminal review deadline.",
            )
        return None

    def allows_token_overrun(self) -> bool:
        return self._ever_validated and self._execution_phase in {
            "needs_validation",
            "validation_failed",
            "needs_review",
            "ready_to_finish",
        }

    def finalize_budget(self, budget_snapshot: dict[str, Any]) -> None:
        self._last_budget = budget_snapshot

    def review_finish(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | ControllerTerminal | None:
        status = str(call.arguments.get("status") or "failed")
        if status != RunStatus.COMPLETED.value:
            return None
        if (
            self._execution_phase == "ready_to_finish"
            and self._validated_fingerprint == current_fingerprint
            and self._reviewed_fingerprint == current_fingerprint
        ):
            return None
        self._finish_rejections += 1
        streak = self._record_terminal_attempt(
            "finish",
            call.arguments,
            current_fingerprint=current_fingerprint,
            controller_state=self._execution_phase,
        )
        if streak >= self.terminal_no_progress_limit:
            return self._phase_terminal(
                "finish_no_progress",
                (
                    "The same finish(completed) request was repeated with an "
                    f"unchanged tree and phase {streak} consecutive times."
                ),
            )
        return self._recovery(
            "finish_not_ready",
            f"finish(completed) requested during {self._execution_phase}",
            self._phase_instruction(),
        )

    def review_final(
        self, content: str | None, *, current_fingerprint: str
    ) -> ControllerRecovery | ControllerTerminal:
        if (
            self._execution_phase == "ready_to_finish"
            and self._validated_fingerprint == current_fingerprint
            and self._reviewed_fingerprint == current_fingerprint
        ):
            return ControllerTerminal(
                RunStatus.COMPLETED,
                (
                    content or "Validated changes reviewed and ready for delivery."
                ).strip(),
                "Plain final response accepted as the finite finalization decision.",
                "controller_ready_final_message",
            )
        streak = self._record_terminal_attempt(
            "plain_final",
            content or "",
            current_fingerprint=current_fingerprint,
            controller_state=self._execution_phase,
        )
        if streak >= self.terminal_no_progress_limit:
            return self._phase_terminal(
                "final_message_no_progress",
                (
                    "The same plain final response was repeated with an unchanged "
                    f"tree and phase {streak} consecutive times."
                ),
            )
        return self._recovery(
            "final_message_not_ready",
            f"plain final response during {self._execution_phase}",
            self._phase_instruction(),
        )

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        calls, tokens = self._usage(self._last_budget)
        summary["execution_closure"] = {
            "phase": self._execution_phase,
            "classifier_advisory_only": True,
            "action_guards": False,
            "explicit_validation_tool": True,
            "validation": {
                "attempts": self._validation_attempts,
                "passes": self._validation_passes,
                "failures": self._validation_failures,
                "source_changes": self._validation_source_changes,
                "commands": list(self._validation_commands),
            },
            "first_pass": {
                "model_calls": self._first_pass_calls,
                "tokens": self._first_pass_tokens,
            },
            "last_pass": {
                "model_calls": self._last_pass_calls,
                "tokens": self._last_pass_tokens,
                "model_calls_after": (
                    calls - self._last_pass_calls
                    if self._last_pass_calls is not None
                    else None
                ),
                "tokens_after": (
                    tokens - self._last_pass_tokens
                    if self._last_pass_tokens is not None
                    else None
                ),
            },
            "diff_reviewed": bool(self._reviewed_fingerprint),
            "finish_rejections": self._finish_rejections,
            "auto_finished": self._auto_finished,
            "long_horizon_guards": {
                "phase_deadlines_terminal": False,
                "phase_token_deadlines_terminal": False,
                "validation_advisory_calls": self.validation_advisory_calls,
                "repair_advisory_calls": self.repair_advisory_calls,
                "review_advisory_calls": self.review_advisory_calls,
                "terminal_no_progress_limit": self.terminal_no_progress_limit,
            },
        }
        return summary

    def _set_phase(self, phase: str, snapshot: dict[str, Any]) -> None:
        if phase == self._execution_phase:
            return
        previous = self._execution_phase
        self._execution_phase = phase
        self._phase_advisories.clear()
        self._phase_started_calls, self._phase_started_tokens = self._usage(snapshot)
        self._policy_events.append(
            (
                "controller_execution_transition",
                {
                    "controller": self.identity,
                    "from": previous,
                    "to": phase,
                    "model_calls": self._phase_started_calls,
                    "tokens": self._phase_started_tokens,
                },
            )
        )

    def _phase_terminal(self, stop_reason: str, evidence: str) -> ControllerTerminal:
        return ControllerTerminal(
            RunStatus.FAILED,
            f"Execution closure stopped in phase {self._execution_phase}.",
            evidence,
            f"controller_{stop_reason}",
        )

    def _phase_advisory(
        self, strategy: str, trigger: str, feedback: str
    ) -> ControllerRecovery | None:
        if strategy in self._phase_advisories:
            return None
        self._phase_advisories.add(strategy)
        return self._recovery(strategy, trigger, feedback)

    def _phase_instruction(self) -> str:
        return {
            "explore": (
                "Inspect only what is needed, implement a source change, then validate."
            ),
            "needs_validation": (
                "Use validate on the current tree before requesting completion."
            ),
            "validation_failed": (
                "Fix the concrete failure and use validate again, or finish failed."
            ),
            "needs_review": (
                "Review the complete current diff with git_diff, then finish or edit."
            ),
            "ready_to_finish": (
                "Call finish now, or edit the tree and validate it again."
            ),
        }[self._execution_phase]

    @staticmethod
    def _usage(snapshot: dict[str, Any]) -> tuple[int, int]:
        usage = snapshot.get("usage") or {}
        calls = usage.get("model_calls")
        tokens = usage.get("total_tokens")
        return (
            calls if isinstance(calls, int) else 0,
            tokens if isinstance(tokens, int) else 0,
        )

    @staticmethod
    def _is_validation_call(call: ToolCall) -> bool:
        if call.name == "validate":
            return True
        return call.name == "shell" and bool(
            _TEST_COMMAND.search(str(call.arguments.get("command") or ""))
        )

    @staticmethod
    def _is_diff_review(call: ToolCall) -> bool:
        if call.name == "git_diff":
            return True
        if call.name == "git_inspect":
            return str(call.arguments.get("operation") or "") == "diff"
        return call.name == "shell" and bool(
            _DIFF_SHELL.search(str(call.arguments.get("command") or ""))
        )


class HybridControllerV12(HybridControllerV11):
    """Hybrid v1.1 plus state-aware tool-schema and action gating."""

    identity = HYBRID_CONTROLLER_V12_ID
    guidance_version = "v1.2"
    implement_scoped_read_limit = 3

    def __init__(
        self,
        policy: ControllerPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)
        self._located_paths: set[str] = set()
        self._replan_granted = False
        self._replan_used = False
        self._gate_counts: Counter[str] = Counter()
        self._state_epoch = 0
        self._deterministic_replan_resets = 0
        self._implement_scoped_reads = 0

    def filter_tool_schemas(
        self, schemas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self._state == "explore":
            return schemas
        allowed = set(_STATE_BASE_TOOLS)
        if (
            self._state == "implement"
            and self._implement_scoped_reads >= self.implement_scoped_read_limit
        ):
            allowed.discard("read_file")
        if self._replan_granted:
            allowed.update({"list_files", "search_files"})
        filtered: list[dict[str, Any]] = []
        for schema in schemas:
            name = str((schema.get("function") or {}).get("name") or "")
            if name not in allowed:
                continue
            if name == "git_inspect":
                schema = copy.deepcopy(schema)
                operation = (
                    schema.get("function", {})
                    .get("parameters", {})
                    .get("properties", {})
                    .get("operation", {})
                )
                if isinstance(operation.get("enum"), list):
                    operation["enum"] = [
                        value for value in operation["enum"] if value != "log"
                    ]
            filtered.append(schema)
        return filtered

    def observe_tool(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
        budget_snapshot: dict[str, Any] | None = None,
    ) -> tuple[ControllerRecovery, ...]:
        previous_state = self._state
        if result.ok and call.name in {"read_file", "apply_patch"}:
            path = _normalise_tool_path(call.arguments.get("path"))
            if path:
                self._located_paths.add(path)
        recoveries = super().observe_tool(
            call,
            result,
            before_fingerprint=before_fingerprint,
            after_fingerprint=after_fingerprint,
            budget_snapshot=budget_snapshot,
        )
        if self._state != previous_state:
            self._state_epoch += 1
            self._replan_granted = False
            self._replan_used = False
            if self._state == "implement":
                self._implement_scoped_reads = 0
            if (
                self._state == "explore"
                and previous_state != "explore"
                and self._deterministic_replan_resets == 0
                and self._no_progress_actions
            ):
                no_progress_actions = self._no_progress_actions
                self._no_progress_actions = 0
                self._action_required = False
                self._deterministic_replan_resets = 1
                self._gate_counts["deterministic_replan_window_reset"] += 1
                self._policy_events.append(
                    (
                        "controller_replan_window_reset",
                        {
                            "controller": self.identity,
                            "previous_state": previous_state,
                            "state": self._state,
                            "state_epoch": self._state_epoch,
                            "no_progress_actions": no_progress_actions,
                            "controlled": True,
                        },
                    )
                )
        if previous_state == "implement" and result.ok and call.name == "read_file":
            self._implement_scoped_reads += 1
        return recoveries

    def guard_action(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | None:
        deterministic = ControllerV1.guard_action(
            self, call, current_fingerprint=current_fingerprint
        )
        if deterministic:
            return deterministic
        reason = self._gate_reason(call)
        if reason is None:
            return None
        replan_candidate = call.name in {"list_files", "search_files"}
        if self._replan_granted and replan_candidate:
            self._replan_granted = False
            self._replan_used = True
            self._gate_counts["controlled_replan_used"] += 1
            self._policy_events.append(
                (
                    "controller_replan_allowed",
                    {
                        "controller": self.identity,
                        "state": self._state,
                        "state_epoch": self._state_epoch,
                        "action": call.name,
                        "reason": reason,
                        "controlled": True,
                    },
                )
            )
            return None
        if not self._replan_used and replan_candidate:
            self._replan_granted = True
        self._gate_counts[f"blocked_{self._state}"] += 1
        allowed = set(_STATE_BASE_TOOLS)
        if (
            self._state == "implement"
            and self._implement_scoped_reads >= self.implement_scoped_read_limit
        ):
            allowed.discard("read_file")
        if self._replan_granted:
            allowed.update({"list_files", "search_files"})
        allowed_names = sorted(allowed)
        replan_available = self._replan_granted
        feedback = (
            f"Hybrid Controller v1.2 blocked {call.name} during state={self._state}: "
            f"{reason}. Choose from the state-allowed actions: "
            f"{', '.join(allowed_names)}."
        )
        if replan_available:
            feedback += (
                " One controlled replan is available on the next turn: use exactly "
                "one targeted read/search action, then return to the current phase."
            )
        self._policy_events.append(
            (
                "controller_action_blocked",
                {
                    "controller": self.identity,
                    "state": self._state,
                    "state_epoch": self._state_epoch,
                    "action": call.name,
                    "reason": reason,
                    "allowed_actions": allowed_names,
                    "replan_available": replan_available,
                },
            )
        )
        return self._recovery(
            "controller_action_blocked",
            f"{call.name} is not allowed during {self._state}",
            feedback,
        )

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary["state_aware_gating"] = {
            "blocked": sum(
                count
                for name, count in self._gate_counts.items()
                if name.startswith("blocked_")
            ),
            "by_state": {
                state: self._gate_counts[f"blocked_{state}"]
                for state in STATES
                if self._gate_counts[f"blocked_{state}"]
            },
            "controlled_replans_used": self._gate_counts["controlled_replan_used"],
            "deterministic_replan_window_resets": self._gate_counts[
                "deterministic_replan_window_reset"
            ],
            "implement_scoped_read_limit": self.implement_scoped_read_limit,
            "implement_scoped_reads_in_current_epoch": self._implement_scoped_reads,
            "located_paths": sorted(self._located_paths),
        }
        return summary

    def _gate_reason(self, call: ToolCall) -> str | None:
        if self._state == "explore":
            return None
        if call.name in {"apply_patch", "git_diff", "finish"}:
            return None
        if call.name == "git_inspect":
            if str(call.arguments.get("operation") or "") == "log":
                return "Git history exploration is not allowed after explore"
            return None
        if call.name == "read_file":
            if (
                self._state == "implement"
                and self._implement_scoped_reads >= self.implement_scoped_read_limit
            ):
                return (
                    "scoped read allowance is exhausted; make the minimal edit or "
                    "finish with concrete evidence"
                )
            return None
        if call.name in {"list_files", "search_files"}:
            return "broad repository discovery is not allowed after explore"
        if call.name == "shell":
            command = str(call.arguments.get("command") or "")
            kind = _shell_action_kind(command, self._located_paths)
            if (
                self._state == "implement"
                and kind == "scoped_read"
                and self._implement_scoped_reads >= self.implement_scoped_read_limit
            ):
                return (
                    "scoped read allowance is exhausted; make the minimal edit or "
                    "finish with concrete evidence"
                )
            allowed = {
                "implement": {"test", "diff", "format", "scoped_read"},
                "verify": {"test", "diff", "format", "scoped_read"},
                "finalize": {
                    "test",
                    "diff",
                    "format",
                    "finalize",
                    "scoped_read",
                },
            }[self._state]
            if kind in allowed:
                return None
            return f"shell action category={kind} is not allowed"
        return f"tool is not available during state={self._state}"


@dataclass
class EditIntentTool(BaseTool):
    controller: "HybridControllerEditIntent"
    workspace: Any
    name = EDIT_INTENT_TOOL_NAME
    description = (
        "Submit the concrete edit plan required before implementation. Target only "
        "existing files and state the diagnosis, intended change, and focused "
        "validation command. This records intent; it does not edit files."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "target_files": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 4,
                "uniqueItems": True,
            },
            "diagnosis": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "intended_change": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2_000,
            },
            "validation_command": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1_000,
            },
        },
        "required": [
            "target_files",
            "diagnosis",
            "intended_change",
            "validation_command",
        ],
        "additionalProperties": False,
    }

    def execute(
        self, arguments: dict[str, Any], *, timeout_seconds: float
    ) -> ToolResult:
        del timeout_seconds
        return self.controller.submit_edit_intent(arguments, self.workspace)


class HybridControllerEditIntent(HybridControllerV12):
    """Hybrid v1.2 with a validated V4-authored edit-intent handoff."""

    identity = HYBRID_CONTROLLER_EDIT_INTENT_ID
    guidance_version = "edit-intent-v1"
    intent_context_action_limit = 3
    intent_context_tools = {"read_file", "search_files", "list_files", "git_inspect"}
    require_implementation_readiness = False

    def __init__(
        self,
        policy: ControllerPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)
        self._intent_required = False
        self._edit_intent: dict[str, Any] | None = None
        self._intent_failures = 0
        self._focused_replan_available = False
        self._focused_replan_used = False
        self._pending_intent_terminal: ControllerTerminal | None = None
        self._intent_counts: Counter[str] = Counter()
        self._intent_activation_pending = False
        self._intent_context_actions = 0

    def additional_tools(self, workspace: Any) -> tuple[EditIntentTool, ...]:
        return (EditIntentTool(self, workspace),)

    def filter_tool_schemas(
        self, schemas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self._intent_required:
            allowed = {EDIT_INTENT_TOOL_NAME, "finish"}
            if self._focused_replan_available:
                allowed.update({"read_file", "search_files"})
            elif (
                self._intent_failures == 0
                and self._intent_context_actions < self.intent_context_action_limit
            ):
                allowed.update(self.intent_context_tools)
            filtered: list[dict[str, Any]] = []
            for schema in schemas:
                name = str((schema.get("function") or {}).get("name") or "")
                if name not in allowed:
                    continue
                if name == "git_inspect":
                    schema = copy.deepcopy(schema)
                    operation = (
                        schema.get("function", {})
                        .get("parameters", {})
                        .get("properties", {})
                        .get("operation", {})
                    )
                    if isinstance(operation.get("enum"), list):
                        operation["enum"] = [
                            value for value in operation["enum"] if value != "log"
                        ]
                filtered.append(schema)
            return filtered
        return [
            schema
            for schema in super().filter_tool_schemas(schemas)
            if str((schema.get("function") or {}).get("name") or "")
            != EDIT_INTENT_TOOL_NAME
        ]

    def observe_tool(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
        budget_snapshot: dict[str, Any] | None = None,
    ) -> tuple[ControllerRecovery, ...]:
        if call.name == EDIT_INTENT_TOOL_NAME:
            return ()
        if self._intent_activation_pending:
            if result.ok and call.name == "read_file":
                path = _normalise_tool_path(call.arguments.get("path"))
                if path:
                    self._located_paths.add(path)
            return ControllerV1.observe_tool(
                self,
                call,
                result,
                before_fingerprint=before_fingerprint,
                after_fingerprint=after_fingerprint,
                budget_snapshot=budget_snapshot,
            )
        if self._intent_required:
            if result.ok and call.name == "read_file":
                path = _normalise_tool_path(call.arguments.get("path"))
                if path:
                    self._located_paths.add(path)
            return ControllerV1.observe_tool(
                self,
                call,
                result,
                before_fingerprint=before_fingerprint,
                after_fingerprint=after_fingerprint,
                budget_snapshot=budget_snapshot,
            )

        previous_state = self._state
        recoveries = list(
            super().observe_tool(
                call,
                result,
                before_fingerprint=before_fingerprint,
                after_fingerprint=after_fingerprint,
                budget_snapshot=budget_snapshot,
            )
        )
        if (
            previous_state == "explore"
            and self._state == "implement"
            and self._edit_intent is None
        ):
            kept: list[ControllerRecovery] = []
            for recovery in recoveries:
                if recovery.strategy == "hybrid_stage_guidance":
                    self._recoveries[recovery.strategy] -= 1
                    if self._recoveries[recovery.strategy] <= 0:
                        del self._recoveries[recovery.strategy]
                    continue
                kept.append(recovery)
            self._state = "explore"
            self._next_action = "inspect"
            readiness = self._implementation_readiness()
            if self.require_implementation_readiness and not readiness["ready"]:
                self._intent_counts["readiness_blocks"] += 1
                self._policy_events.append(
                    (
                        "implement_readiness_blocked",
                        {
                            "controller": self.identity,
                            "classifier_state": "implement",
                            "classifier_next_action": "edit",
                            "readiness": readiness,
                        },
                    )
                )
                kept.append(
                    self._recovery(
                        "implement_readiness_required",
                        "classifier selected implement before source evidence existed",
                        (
                            "Implement Readiness: read the single most relevant "
                            "concrete source file suggested by the task or current "
                            "evidence. A directory listing, Git metadata, file list, "
                            "or another broad search is insufficient. Do not broaden "
                            "the search; read source content, then form the edit intent."
                        ),
                    )
                )
                return tuple(kept)
            if self.require_implementation_readiness:
                self._policy_events.append(
                    (
                        "implement_readiness_satisfied",
                        {
                            "controller": self.identity,
                            "readiness": readiness,
                        },
                    )
                )
            self._intent_activation_pending = True
            self._intent_counts["requested"] += 1
            self._policy_events.append(
                (
                    "edit_intent_requested",
                    {
                        "controller": self.identity,
                        "classifier_state": "implement",
                        "classifier_next_action": "edit",
                        "located_paths": sorted(self._located_paths),
                        "recent_tools": list(self._recent),
                    },
                )
            )
            kept.append(
                self._recovery(
                    "edit_intent_required",
                    "classifier selected implement before an edit intent was accepted",
                    (
                        "Edit Intent Handoff: before implementation, call "
                        "submit_edit_intent with 1-4 existing target_files, a non-empty "
                        "diagnosis, the intended_change, and a focused "
                        "validation_command. ForgeLoop will validate and record it, "
                        "then provide it back as compact working context."
                    ),
                )
            )
            return tuple(kept)
        return tuple(recoveries)

    def end_tool_batch(self) -> None:
        if not self._intent_activation_pending:
            return
        self._intent_activation_pending = False
        self._intent_required = True
        self._policy_events.append(
            (
                "edit_intent_handoff_activated",
                {
                    "controller": self.identity,
                    "located_paths": sorted(self._located_paths),
                    "recent_tools": list(self._recent),
                },
            )
        )

    def guard_action(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | None:
        if not self._intent_required:
            return super().guard_action(call, current_fingerprint=current_fingerprint)
        if call.name == EDIT_INTENT_TOOL_NAME:
            return None
        if self._focused_replan_available and call.name in {
            "read_file",
            "search_files",
        }:
            self._focused_replan_available = False
            self._focused_replan_used = True
            self._intent_counts["focused_replans_used"] += 1
            self._policy_events.append(
                (
                    "edit_intent_focused_replan",
                    {
                        "controller": self.identity,
                        "action": call.name,
                        "arguments": dict(call.arguments),
                        "controlled": True,
                    },
                )
            )
            return None
        if (
            self._intent_failures == 0
            and call.name in self.intent_context_tools
            and not (
                call.name == "git_inspect"
                and str(call.arguments.get("operation") or "") == "log"
            )
            and self._intent_context_actions < self.intent_context_action_limit
        ):
            self._intent_context_actions += 1
            self._intent_counts["context_actions"] += 1
            self._policy_events.append(
                (
                    "edit_intent_context_action",
                    {
                        "controller": self.identity,
                        "action": call.name,
                        "arguments": dict(call.arguments),
                        "index": self._intent_context_actions,
                        "limit": self.intent_context_action_limit,
                    },
                )
            )
            return None

        feedback = self._reject_edit_intent(
            [
                f"{call.name} was requested before the required "
                f"{EDIT_INTENT_TOOL_NAME} handoff"
            ],
            submitted_target_files=[],
        )
        return self._recovery(
            "edit_intent_action_blocked",
            f"{call.name} attempted while edit intent is required",
            feedback,
        )

    def review_final(
        self, content: str | None, *, current_fingerprint: str
    ) -> ControllerRecovery | ControllerTerminal:
        if not self._intent_required:
            return super().review_final(
                content, current_fingerprint=current_fingerprint
            )
        feedback = self._reject_edit_intent(
            [f"model returned text instead of {EDIT_INTENT_TOOL_NAME}"],
            submitted_target_files=[],
        )
        terminal = self.post_tool_terminal()
        if terminal:
            return terminal
        return self._recovery(
            "edit_intent_missing",
            "model did not submit the required structured edit intent",
            feedback,
        )

    def post_tool_terminal(self) -> ControllerTerminal | None:
        terminal = self._pending_intent_terminal
        self._pending_intent_terminal = None
        return terminal

    def submit_edit_intent(
        self, arguments: dict[str, Any], workspace: Any
    ) -> ToolResult:
        errors, intent = self._validate_edit_intent(arguments, workspace)
        if errors:
            feedback = self._reject_edit_intent(
                errors,
                submitted_target_files=list(arguments.get("target_files") or []),
            )
            return ToolResult(
                False,
                feedback,
                {
                    "controller": self.identity,
                    "edit_intent_valid": False,
                    "attempt": self._intent_failures,
                },
            )

        self._edit_intent = intent
        self._intent_required = False
        self._focused_replan_available = False
        self._state = "implement"
        self._next_action = "edit"
        self._state_epoch += 1
        self._replan_granted = False
        self._replan_used = False
        self._implement_scoped_reads = 0
        self._intent_counts["accepted"] += 1
        self._transition_counts["explore/intent->implement/edit"] += 1
        working_context = _format_edit_intent(intent)
        self._policy_events.append(
            (
                "edit_intent_accepted",
                {
                    "controller": self.identity,
                    "intent": dict(intent),
                    "working_context": working_context,
                    "schema_valid": True,
                    "target_files_exist": True,
                },
            )
        )
        return ToolResult(
            True,
            working_context,
            {
                "controller": self.identity,
                "edit_intent_valid": True,
                "intent": dict(intent),
            },
        )

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary["edit_intent_handoff"] = {
            "required": self._intent_required,
            "requested": self._intent_counts["requested"],
            "accepted": self._intent_counts["accepted"],
            "rejected": self._intent_counts["rejected"],
            "focused_replans_used": self._intent_counts["focused_replans_used"],
            "context_actions": self._intent_counts["context_actions"],
            "context_action_limit": self.intent_context_action_limit,
            "readiness_blocks": self._intent_counts["readiness_blocks"],
            "intent": dict(self._edit_intent) if self._edit_intent else None,
        }
        return summary

    def _reject_edit_intent(
        self, errors: list[str], *, submitted_target_files: list[Any]
    ) -> str:
        self._intent_failures += 1
        self._intent_counts["rejected"] += 1
        self._policy_events.append(
            (
                "edit_intent_rejected",
                {
                    "controller": self.identity,
                    "attempt": self._intent_failures,
                    "errors": list(errors),
                    "submitted_target_files": [
                        str(path) for path in submitted_target_files
                    ],
                },
            )
        )
        detail = "; ".join(errors)
        if self._intent_failures == 1:
            self._focused_replan_available = True
            return (
                f"Edit intent rejected: {detail}. One focused replan is available: "
                "use exactly one read_file or scoped search_files action, then "
                f"resubmit {EDIT_INTENT_TOOL_NAME}."
            )
        self._focused_replan_available = False
        self._pending_intent_terminal = ControllerTerminal(
            RunStatus.FAILED,
            "Edit intent handoff failed twice; Controller stopped before implementation.",
            detail,
            "controller_invalid_edit_intent",
        )
        return (
            f"Edit intent rejected again: {detail}. Controller is terminating the run."
        )

    @staticmethod
    def _validate_edit_intent(
        arguments: dict[str, Any], workspace: Any
    ) -> tuple[list[str], dict[str, Any]]:
        errors: list[str] = []
        raw_targets = arguments.get("target_files")
        targets = raw_targets if isinstance(raw_targets, list) else []
        if not 1 <= len(targets) <= 4:
            errors.append("target_files must contain 1-4 paths")

        validated_targets: list[str] = []
        for raw_path in targets[:4]:
            if not isinstance(raw_path, str) or not raw_path.strip():
                errors.append("target_files entries must be non-empty strings")
                continue
            path = _intent_path(raw_path)
            if not path or path == "." or is_sensitive_path(path):
                errors.append(f"target file is not allowed: {raw_path}")
                continue
            try:
                resolved = workspace.resolve(path)
                runtime = getattr(workspace, "runtime", None)
                kind = (
                    runtime.path_kind(resolved)
                    if runtime
                    else ("file" if resolved.is_file() else "missing")
                )
            except Exception as exc:  # noqa: BLE001 - validation becomes observation
                errors.append(f"target file cannot be resolved: {raw_path} ({exc})")
                continue
            if kind != "file":
                errors.append(f"target file does not exist: {raw_path}")
                continue
            validated_targets.append(workspace.relative(resolved))

        if len(set(validated_targets)) != len(validated_targets):
            errors.append("target_files must be unique")

        fields: dict[str, str] = {}
        limits = {
            "diagnosis": 2_000,
            "intended_change": 2_000,
            "validation_command": 1_000,
        }
        for field, limit in limits.items():
            value = arguments.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{field} must be non-empty")
                fields[field] = ""
            elif len(value.strip()) > limit:
                errors.append(f"{field} exceeds {limit} characters")
                fields[field] = value.strip()[:limit]
            else:
                fields[field] = value.strip()

        return errors, {
            "target_files": validated_targets,
            **fields,
        }


class HybridControllerImplementReadiness(HybridControllerEditIntent):
    """Edit Intent Handoff gated on concrete source-reading evidence."""

    identity = HYBRID_CONTROLLER_READINESS_ID
    guidance_version = "edit-intent-readiness-v1"
    require_implementation_readiness = True


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


def _fixed_guidance(decision: HybridDecision, version: str = "v1.1") -> str:
    guidance = {
        "inspect": (
            f"Hybrid Controller {version}: phase=explore. Inspect one targeted file or "
            "symbol needed to identify the smallest supported edit."
        ),
        "edit": (
            f"Hybrid Controller {version}: phase=implement. Make the smallest supported "
            "source edit now; do not broaden repository exploration."
        ),
        "test": (
            f"Hybrid Controller {version}: phase=verify. Run the focused test or verifier "
            "for the current diff; do not broaden repository exploration."
        ),
        "replan": (
            f"Hybrid Controller {version}: phase=verify. The latest test failed. Inspect "
            "only failure-related context, correct the edit, and rerun that test."
        ),
        "finalize": (
            f"Hybrid Controller {version}: phase=finalize. Review the current diff and "
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
    readiness = snapshot.get("implementation_readiness") or {}
    readiness_input = {
        "source_content_read": bool(readiness.get("source_content_read")),
        "source_files_read": list(readiness.get("source_files_read") or []),
        "candidate_target_files": list(readiness.get("candidate_target_files") or []),
        "saw_test_evidence": bool(readiness.get("saw_test_evidence")),
        "saw_error_evidence": bool(readiness.get("saw_error_evidence")),
        "has_diff": bool(readiness.get("has_diff")),
        "has_intent": bool(readiness.get("has_intent")),
        "ready": bool(readiness.get("ready")),
    }
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
            "Implementation readiness: "
            + json.dumps(readiness_input, separators=(",", ":")),
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


def _intent_path(value: str) -> str:
    path = value.replace("\\", "/").strip()
    if path == "/app":
        return "."
    if path.startswith("/app/"):
        path = path[5:]
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/")


def _is_source_file(path: str) -> bool:
    normalized = path.casefold()
    if not normalized or is_sensitive_path(path):
        return False
    return Path(normalized).suffix in _SOURCE_SUFFIXES


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").casefold()
    name = normalized.rsplit("/", 1)[-1]
    return (
        "/test/" in f"/{normalized}/"
        or "/tests/" in f"/{normalized}/"
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.js")
        or name.endswith(".spec.ts")
    )


def _format_edit_intent(intent: Mapping[str, Any]) -> str:
    targets = ", ".join(str(path) for path in intent["target_files"])
    return (
        "Edit intent accepted. Use this compact working context for implementation:\n"
        f"target_files: {targets}\n"
        f"diagnosis: {intent['diagnosis']}\n"
        f"intended_change: {intent['intended_change']}\n"
        f"validation_command: {intent['validation_command']}\n"
        "Proceed with the smallest supported edit, then run the validation command."
    )


def _normalise_tool_path(value: Any) -> str:
    path = str(value or "").replace("\\", "/").strip()
    if path.startswith("/app/"):
        path = path[5:]
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/").casefold()


def _shell_action_kind(command: str, located_paths: set[str]) -> str:
    # A permitted test/diff command must not make a broad discovery command in
    # the same shell payload safe by appearing first.
    if _BROAD_SHELL.search(command):
        return "broad_explore"
    if _TEST_COMMAND.search(command):
        return "test"
    if _DIFF_SHELL.search(command):
        return "diff"
    if _FINALIZE_SHELL.search(command):
        return "finalize"
    if _FORMAT_SHELL.search(command):
        return "format"
    if _HISTORY_SHELL.search(command):
        return "history_explore"
    lowered = command.replace("\\", "/").casefold()
    if _READ_SHELL.search(command) and any(path in lowered for path in located_paths):
        return "scoped_read"
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
    "EDIT_INTENT_TOOL_NAME",
    "HYBRID_CONTROLLER_EDIT_INTENT_ID",
    "HYBRID_CONTROLLER_READINESS_ID",
    "HYBRID_CONTROLLER_V13_SIMPLIFIED_ID",
    "HYBRID_CONTROLLER_V11_ID",
    "HYBRID_CONTROLLER_V12_ID",
    "HybridControllerEditIntent",
    "HybridControllerImplementReadiness",
    "HybridControllerV13Simplified",
    "HybridControllerV11",
    "HybridControllerV12",
    "HybridDecision",
    "OllamaControllerPolicy",
    "probe_controller_policy",
]
