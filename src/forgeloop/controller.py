from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from forgeloop.agent_types import RunStatus
from forgeloop.guards import RepeatedActionDetector, terminal_attempt_fingerprint
from forgeloop.policy import PolicyIdentity
from forgeloop.tools.base import ToolResult
from forgeloop.types import ToolCall
from forgeloop.workspace import Workspace


CONTROLLER_V1_ID = "forgeloop.controller.v1"


@dataclass(frozen=True)
class ControllerRecovery:
    strategy: str
    trigger: str
    feedback: str


@dataclass(frozen=True)
class ControllerTerminal:
    status: RunStatus
    summary: str
    evidence: str
    stop_reason: str


class ControllerV1:
    """Small deterministic stability layer around the existing AgentLoop protocol."""

    identity = CONTROLLER_V1_ID
    terminal_no_progress_limit = 4

    def __init__(
        self,
        *,
        repeat_recovery_at: int = 2,
        edit_failure_recovery_at: int = 2,
        no_progress_recovery_at: int = 6,
    ) -> None:
        self.repeat_recovery_at = repeat_recovery_at
        self.edit_failure_recovery_at = edit_failure_recovery_at
        self.no_progress_recovery_at = no_progress_recovery_at
        self._initial_fingerprint = ""
        self._repeated_actions = RepeatedActionDetector()
        self._recoveries: Counter[str] = Counter()
        self._consecutive_edit_failures = 0
        self._no_progress_actions = 0
        self._last_terminal_attempt = ""
        self._terminal_attempt_streak = 0

    def start(self, workspace: Workspace) -> None:
        self._initial_fingerprint = workspace.git_progress_fingerprint()

    def additional_tools(self, workspace: Workspace) -> tuple[Any, ...]:
        del workspace
        return ()

    def observe_tool(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
        budget_snapshot: dict[str, Any] | None = None,
    ) -> tuple[ControllerRecovery, ...]:
        del budget_snapshot
        recoveries: list[ControllerRecovery] = []
        repeated = self._repeated_actions.observe(
            call,
            result,
            before_fingerprint=before_fingerprint,
            after_fingerprint=after_fingerprint,
        )
        self._last_terminal_attempt = ""
        self._terminal_attempt_streak = 0
        if repeated.streak == self.repeat_recovery_at:
            recoveries.append(
                self._recovery(
                    "repeated_action",
                    f"{call.name} repeated with identical evidence",
                    "Controller v1 detected consecutive identical actions with the "
                    "same observation and unchanged workspace. Use the latest "
                    "evidence, then choose a materially different inspect or edit "
                    "action instead of repeating it.",
                )
            )

        if before_fingerprint == after_fingerprint:
            self._no_progress_actions += 1
        else:
            self._no_progress_actions = 0

        if not result.ok:
            recoveries.append(
                self._recovery(
                    "tool_error_feedback",
                    f"{call.name} returned an error",
                    "Controller v1: the ERROR tool observation immediately above is "
                    "authoritative. Correct the arguments or inspect the target state "
                    "before trying another action.",
                )
            )

        if call.name == "apply_patch":
            if result.ok:
                self._consecutive_edit_failures = 0
            else:
                self._consecutive_edit_failures += 1
                if self._consecutive_edit_failures == self.edit_failure_recovery_at:
                    recoveries.append(
                        self._recovery(
                            "edit_failure_reinspect",
                            "consecutive apply_patch failures",
                            "Controller v1 recovery: stop patching from stale context. "
                            "Re-read the exact target region, then construct a smaller "
                            "patch from the current file contents.",
                        )
                    )

        return tuple(recoveries)

    def guard_action(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | None:
        del call, current_fingerprint
        return None

    def before_model_call(
        self,
        *,
        current_fingerprint: str,
        budget_snapshot: dict[str, Any],
    ) -> ControllerRecovery | ControllerTerminal | None:
        del current_fingerprint, budget_snapshot
        return None

    def allows_token_overrun(self) -> bool:
        return False

    def finalize_budget(self, budget_snapshot: dict[str, Any]) -> None:
        """Record the authoritative terminal budget before rendering the summary."""
        del budget_snapshot

    def review_final(
        self, content: str | None, *, current_fingerprint: str
    ) -> ControllerRecovery | ControllerTerminal:
        has_progress = self._has_progress(current_fingerprint)
        streak = self._record_terminal_attempt(
            "plain_final",
            content or "",
            current_fingerprint=current_fingerprint,
            controller_state="progress" if has_progress else "no_progress",
        )
        if streak < self.terminal_no_progress_limit:
            if has_progress:
                return self._recovery(
                    "missing_explicit_finish",
                    "model returned text after modifying the repository",
                    "Controller v1 requires an explicit terminal decision. Verify the "
                    "current changes, then call finish with status, summary, and "
                    "evidence. Do not respond with a plain final message.",
                )
            return self._recovery(
                "exploration_without_change",
                "model returned text without repository progress",
                "Controller v1 rejected an exploratory final response because no "
                "Git-visible change exists. Inspect the relevant code, make and verify "
                "the requested edit, then call finish. If truly blocked, call finish "
                "with blocked and concrete evidence.",
            )
        reason = (
            "controller_missing_finish"
            if has_progress
            else "controller_no_change_final"
        )
        return ControllerTerminal(
            RunStatus.FAILED,
            "Controller v1 stopped a proven repeated terminal-response loop.",
            (content or "No final content was provided.").strip(),
            reason,
        )

    def review_finish(
        self, call: ToolCall, *, current_fingerprint: str
    ) -> ControllerRecovery | ControllerTerminal | None:
        status = str(call.arguments.get("status", "failed"))
        if status != RunStatus.COMPLETED.value or self._has_progress(
            current_fingerprint
        ):
            return None
        streak = self._record_terminal_attempt(
            "finish",
            call.arguments,
            current_fingerprint=current_fingerprint,
            controller_state="completed_without_change",
        )
        if streak < self.terminal_no_progress_limit:
            return self._recovery(
                "finish_without_change",
                "finish(completed) requested without repository progress",
                "Controller v1 rejected finish(completed) because no Git-visible "
                "change exists. Inspect and implement the requested modification, "
                "verify it, then call finish again. Use blocked or failed only when "
                "supported by concrete evidence.",
            )
        return ControllerTerminal(
            RunStatus.BLOCKED,
            "Controller v1 stopped a proven repeated finish(completed) loop.",
            "No Git-visible modification was produced.",
            "controller_finish_without_change",
        )

    def summary(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "recoveries": dict(sorted(self._recoveries.items())),
            "strategies_triggered": sorted(self._recoveries),
        }

    def drain_events(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        return ()

    def filter_tool_schemas(
        self, schemas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return schemas

    def post_tool_terminal(self) -> ControllerTerminal | None:
        return None

    def end_tool_batch(self) -> None:
        return None

    def _has_progress(self, current_fingerprint: str) -> bool:
        return bool(
            self._initial_fingerprint
            and current_fingerprint != self._initial_fingerprint
        )

    def _recovery(
        self, strategy: str, trigger: str, feedback: str
    ) -> ControllerRecovery:
        self._recoveries[strategy] += 1
        return ControllerRecovery(strategy, trigger, feedback)

    def _record_terminal_attempt(
        self,
        kind: str,
        payload: Any,
        *,
        current_fingerprint: str,
        controller_state: str,
    ) -> int:
        fingerprint = terminal_attempt_fingerprint(
            kind,
            payload,
            current_fingerprint=current_fingerprint,
            controller_state=controller_state,
        )
        self._terminal_attempt_streak = (
            self._terminal_attempt_streak + 1
            if fingerprint == self._last_terminal_attempt
            else 1
        )
        self._last_terminal_attempt = fingerprint
        return self._terminal_attempt_streak


def controller_for_policy(policy: Any) -> ControllerV1 | None:
    if not isinstance(policy, PolicyIdentity):
        return None
    controller = str(policy.serving_config.get("controller") or "").lower()
    if controller == "v1":
        return ControllerV1()
    if controller == "hybrid-v1.1":
        from forgeloop.hybrid_controller import (
            DEFAULT_CONTROLLER_POLICY,
            ControllerPolicyConfig,
            HybridControllerV11,
            OllamaControllerPolicy,
        )

        reference = str(
            policy.serving_config.get("controller_policy") or DEFAULT_CONTROLLER_POLICY
        )
        return HybridControllerV11(
            OllamaControllerPolicy(ControllerPolicyConfig.load(reference))
        )
    if controller == "hybrid-v1.2":
        from forgeloop.hybrid_controller import (
            DEFAULT_CONTROLLER_POLICY,
            ControllerPolicyConfig,
            HybridControllerV12,
            OllamaControllerPolicy,
        )

        reference = str(
            policy.serving_config.get("controller_policy") or DEFAULT_CONTROLLER_POLICY
        )
        return HybridControllerV12(
            OllamaControllerPolicy(ControllerPolicyConfig.load(reference))
        )
    if controller == "hybrid-v1.2-edit-intent":
        from forgeloop.hybrid_controller import (
            DEFAULT_CONTROLLER_POLICY,
            ControllerPolicyConfig,
            HybridControllerEditIntent,
            OllamaControllerPolicy,
        )

        reference = str(
            policy.serving_config.get("controller_policy") or DEFAULT_CONTROLLER_POLICY
        )
        return HybridControllerEditIntent(
            OllamaControllerPolicy(ControllerPolicyConfig.load(reference))
        )
    if controller == "hybrid-v1.2-edit-intent-readiness":
        from forgeloop.hybrid_controller import (
            DEFAULT_CONTROLLER_POLICY,
            ControllerPolicyConfig,
            HybridControllerImplementReadiness,
            OllamaControllerPolicy,
        )

        reference = str(
            policy.serving_config.get("controller_policy") or DEFAULT_CONTROLLER_POLICY
        )
        return HybridControllerImplementReadiness(
            OllamaControllerPolicy(ControllerPolicyConfig.load(reference))
        )
    if controller == "hybrid-v1.3-simplified":
        from forgeloop.hybrid_controller import (
            DEFAULT_CONTROLLER_POLICY,
            ControllerPolicyConfig,
            HybridControllerV13Simplified,
            OllamaControllerPolicy,
        )

        reference = str(
            policy.serving_config.get("controller_policy") or DEFAULT_CONTROLLER_POLICY
        )
        return HybridControllerV13Simplified(
            OllamaControllerPolicy(ControllerPolicyConfig.load(reference))
        )
    return None


__all__ = [
    "CONTROLLER_V1_ID",
    "ControllerRecovery",
    "ControllerTerminal",
    "ControllerV1",
    "controller_for_policy",
]
