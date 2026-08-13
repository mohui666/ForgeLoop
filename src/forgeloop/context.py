from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from forgeloop.model_capabilities import ModelCapability
from forgeloop.security import SecretRedactor
from forgeloop.types import Message

AGENT_COMPACT_THRESHOLD_TOKENS = 6_000
AGENT_KEEP_RECENT_TURNS = 3
PI_COMPACTION_RESERVE_TOKENS = 16_384
COMPACTION_ESTIMATE_SAFETY_RATIO = 0.8
_TEST_COMMAND = re.compile(
    r"(?i)(?:^|[;&|]\s*|\s)(?:python\s+-m\s+)?(?:pytest|unittest|"
    r"cargo\s+test|go\s+test|dotnet\s+test|npm\s+(?:run\s+)?test|"
    r"pnpm\s+(?:run\s+)?test|yarn\s+test)(?:\s|$)"
)


@dataclass(frozen=True)
class ContextBudget:
    context_window: int | None
    reserved_output: int | None
    thinking_tool_reserve: int | None
    safety_margin: int | None
    usable_context: int | None
    auto_compact_threshold: int | None


class AgentMessageHistory(list[Message]):
    """Cache-stable request history with a separate canonical audit history.

    A committed compaction replaces only the request-facing history. The full
    append-only conversation remains available in ``canonical`` for audit and
    provenance. Subsequent messages extend both histories, keeping the compacted
    request prefix byte-stable until another compaction epoch is necessary.
    """

    def __init__(self, messages: Sequence[Message]) -> None:
        initial = [dict(message) for message in messages]
        super().__init__(dict(message) for message in initial)
        self.canonical: list[Message] = initial
        self.compaction_epochs = 0

    def append(self, message: Message) -> None:
        canonical = dict(message)
        self.canonical.append(canonical)
        super().append(dict(message))

    def commit_compaction(self, messages: Sequence[Message]) -> None:
        self[:] = [dict(message) for message in messages]
        self.compaction_epochs += 1


def agent_compaction_threshold(provider: Any) -> int:
    """Choose a cache-friendly threshold near the provider context boundary.

    Pi compacts only near ``contextWindow - reserveTokens`` and then rebuilds a
    stable summary-plus-recent prefix. ForgeLoop uses the same boundary shape,
    with a conservative estimator safety ratio because its request estimate is
    deterministic rather than provider-tokenizer exact.
    """

    identity = getattr(provider, "policy_identity", None)
    serving = getattr(identity, "serving_config", None)
    generation = getattr(identity, "generation_config", None)
    if isinstance(identity, dict):
        serving = identity.get("serving_config")
        generation = identity.get("generation_config")
    if isinstance(serving, dict):
        configured = serving.get("context_compact_threshold_tokens")
        if configured is not None:
            threshold = int(configured)
            if threshold <= 0:
                raise ValueError("context_compact_threshold_tokens must be positive")
            return threshold

    capability = getattr(provider, "capability", None)
    window = getattr(capability, "context_window", None)
    if not isinstance(window, int) or window <= 0:
        return AGENT_COMPACT_THRESHOLD_TOKENS
    configured_output = (
        generation.get("max_tokens") if isinstance(generation, dict) else None
    )
    reserve = max(
        PI_COMPACTION_RESERVE_TOKENS,
        int(configured_output) if configured_output is not None else 0,
    )
    usable = max(AGENT_COMPACT_THRESHOLD_TOKENS, window - reserve)
    return max(
        AGENT_COMPACT_THRESHOLD_TOKENS,
        int(usable * COMPACTION_ESTIMATE_SAFETY_RATIO),
    )


def context_budget(
    capability: ModelCapability, thinking_level: str = "auto"
) -> ContextBudget:
    window = capability.context_window
    if window is None:
        return ContextBudget(None, None, None, None, None, None)
    # These are explicit ForgeLoop safety policies, not claimed model facts.
    # Provider/registry metadata supplies the actual window and output ceiling.
    output = capability.max_output_tokens
    if output is None:
        return ContextBudget(window, None, None, None, None, None)
    tool_reserve = max(2_048, int(window * 0.03))
    if capability.thinking and thinking_level != "auto":
        tool_reserve += max(2_048, int(window * 0.03))
    safety = max(2_048, int(window * 0.03))
    usable = max(0, window - output - tool_reserve - safety)
    threshold = int(usable * 0.85)
    return ContextBudget(window, output, tool_reserve, safety, usable, threshold)


def estimate_tokens(messages: Sequence[Message]) -> int:
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    return max(1, chars // 4) if chars else 0


def estimate_request_tokens(
    messages: Sequence[Message], tools: Sequence[dict[str, Any]] = ()
) -> int:
    """Estimate serialized request size for deterministic context policy decisions."""
    payload = json.dumps(
        {"messages": list(messages), "tools": list(tools)},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return max(1, len(payload) // 4)


def prepare_agent_context(
    messages: Sequence[Message],
    tools: Sequence[dict[str, Any]],
    *,
    base_message_count: int,
    keep_recent_turns: int = AGENT_KEEP_RECENT_TURNS,
    compact_threshold_tokens: int = AGENT_COMPACT_THRESHOLD_TOKENS,
    redactor: SecretRedactor | None = None,
) -> tuple[list[Message], dict[str, Any]]:
    """Prepare a bounded model request while keeping the canonical history intact.

    The task prefix is always retained verbatim. Old action turns are replaced by a
    deterministic provenance ledger, except for evidence that can still govern the
    current change: successful patches, reads of edited files, the latest reads,
    diff, test, failed test, and tool error. Recent action turns remain untouched.
    """
    source = [dict(message) for message in messages]
    before_tokens = estimate_request_tokens(source, tools)
    base_count = max(0, min(base_message_count, len(source)))
    groups = _action_groups(source[base_count:])
    report: dict[str, Any] = {
        "schema_version": "forgeloop.context.v1",
        "policy": "agent-context-efficiency-v1",
        "applied": False,
        "before_estimated_tokens": before_tokens,
        "after_estimated_tokens": before_tokens,
        "messages_before": len(source),
        "messages_after": len(source),
        "turns_compacted": 0,
        "turns_preserved": len(groups),
        "keep_recent_turns": keep_recent_turns,
    }
    if before_tokens < compact_threshold_tokens or len(groups) <= keep_recent_turns:
        report.update(_context_sources(source, tools, base_count=base_count))
        return source, report

    pinned, reasons = _pinned_action_groups(groups, keep_recent_turns)
    compacted_indexes = [index for index in range(len(groups)) if index not in pinned]
    if not compacted_indexes:
        report.update(_context_sources(source, tools, base_count=base_count))
        return source, report

    ledger = _provenance_ledger(groups, compacted_indexes)
    if redactor:
        ledger = redactor.redact_text(ledger)
    recent_start = max(0, len(groups) - keep_recent_turns)
    effective: list[Message] = [*source[:base_count]]
    effective.append({"role": "system", "content": ledger})
    for index, group in enumerate(groups):
        if index in pinned:
            effective.extend(
                group if index >= recent_start else _strip_historical_reasoning(group)
            )

    after_tokens = estimate_request_tokens(effective, tools)
    if after_tokens >= before_tokens:
        report.update(_context_sources(source, tools, base_count=base_count))
        return source, report

    report.update(
        {
            "applied": True,
            "after_estimated_tokens": after_tokens,
            "messages_after": len(effective),
            "turns_compacted": len(compacted_indexes),
            "turns_preserved": len(pinned),
            "preserved_reasons": dict(sorted(reasons.items())),
        }
    )
    report.update(_context_sources(effective, tools, base_count=base_count))
    return effective, report


def _action_groups(messages: Sequence[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    for message in messages:
        if message.get("role") == "assistant" or not groups:
            groups.append([])
        groups[-1].append(message)
    return groups


def _tool_calls(group: Sequence[Message]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in group:
        if message.get("role") != "assistant":
            continue
        for raw in message.get("tool_calls") or ():
            function = raw.get("function") or {}
            arguments = function.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {"raw": arguments}
            calls.append(
                {
                    "id": str(raw.get("id") or ""),
                    "name": str(function.get("name") or "unknown"),
                    "arguments": arguments,
                }
            )
    return calls


def _observations(group: Sequence[Message]) -> dict[str, Message]:
    return {
        str(message.get("tool_call_id") or ""): message
        for message in group
        if message.get("role") == "tool"
    }


def _call_ok(call: dict[str, Any], observations: dict[str, Message]) -> bool:
    content = str(observations.get(call["id"], {}).get("content", ""))
    return content.startswith("OK\n")


def _is_diff_call(call: dict[str, Any]) -> bool:
    if call["name"] == "git_diff":
        return True
    if call["name"] == "git_inspect":
        return call["arguments"].get("operation") == "diff"
    return call["name"] == "shell" and bool(
        re.search(r"(?i)\bgit\s+diff\b", str(call["arguments"].get("command", "")))
    )


def _is_test_call(call: dict[str, Any]) -> bool:
    return call["name"] == "shell" and bool(
        _TEST_COMMAND.search(str(call["arguments"].get("command", "")))
    )


def _pinned_action_groups(
    groups: Sequence[Sequence[Message]], keep_recent_turns: int
) -> tuple[set[int], dict[str, int]]:
    pinned: set[int] = set()
    reasons: dict[str, int] = {}

    def pin(index: int, reason: str) -> None:
        pinned.add(index)
        reasons[reason] = reasons.get(reason, 0) + 1

    recent_start = max(0, len(groups) - keep_recent_turns)
    for index in range(recent_start, len(groups)):
        pin(index, "recent")

    details = [(_tool_calls(group), _observations(group)) for group in groups]
    edited_paths: set[str] = set()
    for index, (calls, observations) in enumerate(details):
        successful_patches = [
            call
            for call in calls
            if call["name"] == "apply_patch" and _call_ok(call, observations)
        ]
        if successful_patches:
            pin(index, "current_patch")
            edited_paths.update(
                str(call["arguments"].get("path") or "")
                for call in successful_patches
            )

    def pin_latest(reason: str, predicate: Any) -> None:
        for index in range(len(groups) - 1, -1, -1):
            calls, observations = details[index]
            if predicate(calls, observations):
                pin(index, reason)
                return

    for path in sorted(filter(None, edited_paths)):
        pin_latest(
            "edited_source",
            lambda calls, _observations, target=path: any(
                call["name"] == "read_file"
                and str(call["arguments"].get("path") or "") == target
                for call in calls
            ),
        )
    read_indexes = [
        index
        for index, (calls, _observations) in enumerate(details)
        if any(call["name"] == "read_file" for call in calls)
    ]
    for index in read_indexes[-2:]:
        pin(index, "latest_source")

    pin_latest("latest_diff", lambda calls, _observations: any(map(_is_diff_call, calls)))
    pin_latest("latest_test", lambda calls, _observations: any(map(_is_test_call, calls)))
    pin_latest(
        "latest_failed_test",
        lambda calls, observations: any(
            _is_test_call(call) and not _call_ok(call, observations) for call in calls
        ),
    )
    pin_latest(
        "latest_error",
        lambda _calls, observations: any(
            str(message.get("content", "")).startswith("ERROR\n")
            for message in observations.values()
        ),
    )
    return pinned, reasons


def _provenance_ledger(
    groups: Sequence[Sequence[Message]], compacted_indexes: Sequence[int]
) -> str:
    lines = [
        "ForgeLoop deterministic history ledger. These older turns were compacted; "
        "the full evidence remains in the trajectory. Treat the task and retained "
        "recent/current evidence as authoritative."
    ]
    for index in compacted_indexes:
        group = groups[index]
        calls = _tool_calls(group)
        observations = _observations(group)
        if not calls:
            text = " ".join(
                str(message.get("content") or "").strip()
                for message in group
                if message.get("content")
            )
            if text:
                lines.append(f"- action {index + 1}: model/user note: {_excerpt(text, 600)}")
            continue
        for call in calls:
            observation = observations.get(call["id"], {})
            content = str(observation.get("content") or "")
            status = "OK" if content.startswith("OK\n") else "ERROR"
            arguments = json.dumps(
                call["arguments"], ensure_ascii=False, sort_keys=True, default=str
            )
            excerpt_limit = 900 if status == "ERROR" else 500
            excerpt = _observation_excerpt(content, excerpt_limit)
            lines.append(
                f"- action {index + 1}: {call['name']} {arguments} -> {status}; "
                f"observation={excerpt}"
            )
        for message in group:
            if message.get("role") == "user" and message.get("content"):
                lines.append(
                    f"- action {index + 1}: controller/user feedback: "
                    f"{_excerpt(' '.join(str(message['content']).split()), 600)}"
                )
    return "\n".join(lines)


def _strip_historical_reasoning(group: Sequence[Message]) -> list[Message]:
    result: list[Message] = []
    for original in group:
        message = dict(original)
        if message.get("role") == "assistant" and "reasoning_content" in message:
            # DeepSeek tool history requires the field, but not the obsolete chain.
            message["reasoning_content"] = " " if message.get("tool_calls") else ""
        result.append(message)
    return result


def _observation_excerpt(content: str, limit: int) -> str:
    flat = " ".join(content.split())
    return json.dumps(_excerpt(flat, limit), ensure_ascii=False)


def _excerpt(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... <{len(value) - limit} chars omitted>"


def _context_sources(
    messages: Sequence[Message],
    tools: Sequence[dict[str, Any]],
    *,
    base_count: int,
) -> dict[str, Any]:
    sources = {
        "system": 0,
        "task": 0,
        "session_context": 0,
        "assistant_content": 0,
        "assistant_reasoning": 0,
        "assistant_tool_calls": 0,
        "tool_observations": 0,
        "controller_feedback": 0,
        "tool_schemas": len(
            json.dumps(list(tools), ensure_ascii=False, default=str, separators=(",", ":"))
        ),
    }
    call_names: dict[str, str] = {}
    by_tool: dict[str, int] = {}
    for index, message in enumerate(messages):
        role = str(message.get("role") or "unknown")
        content_chars = len(str(message.get("content") or ""))
        if role == "system":
            sources["system"] += content_chars
        elif index == base_count - 1 and role == "user":
            sources["task"] += content_chars
        elif index < base_count:
            sources["session_context"] += content_chars
        elif role == "assistant":
            sources["assistant_content"] += content_chars
            sources["assistant_reasoning"] += len(
                str(message.get("reasoning_content") or "")
            )
            raw_calls = message.get("tool_calls") or ()
            sources["assistant_tool_calls"] += len(
                json.dumps(raw_calls, ensure_ascii=False, default=str, separators=(",", ":"))
            )
            for raw in raw_calls:
                call_names[str(raw.get("id") or "")] = str(
                    (raw.get("function") or {}).get("name") or "unknown"
                )
        elif role == "tool":
            sources["tool_observations"] += content_chars
            name = call_names.get(str(message.get("tool_call_id") or ""), "unknown")
            by_tool[name] = by_tool.get(name, 0) + content_chars
        elif role == "user":
            sources["controller_feedback"] += content_chars
    dominant = sorted(sources.items(), key=lambda item: (-item[1], item[0]))[:5]
    return {
        "estimated_input_tokens": estimate_request_tokens(messages, tools),
        "source_chars": sources,
        "tool_observation_chars": dict(
            sorted(by_tool.items(), key=lambda item: (-item[1], item[0]))
        ),
        "dominant_sources": [
            {"source": name, "chars": chars, "estimated_tokens": chars // 4}
            for name, chars in dominant
        ],
    }


def compact_messages(
    messages: Sequence[Message],
    *,
    keep_recent: int = 8,
    redactor: SecretRedactor | None = None,
    context_state: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[list[Message], dict[str, int]]:
    """Deterministically compress older conversational turns without a model call."""
    source = [dict(message) for message in messages]
    before = estimate_tokens(source)
    if len(source) <= keep_recent and not force:
        return source, {"before_tokens": before, "after_tokens": before, "compacted": 0}
    if force and len(source) > 1:
        keep_recent = min(keep_recent, max(1, len(source) // 3))
    older, recent = source[:-keep_recent], source[-keep_recent:]
    lines: list[str] = []
    for message in older:
        role = str(message.get("role", "unknown"))
        content = " ".join(str(message.get("content", "")).split())
        if not content:
            continue
        lines.append(f"{role}: {content[:600]}")
    state = context_state or {}
    sections: list[str] = [
        "ForgeLoop compacted context (authoritative working summary):"
    ]
    labels = (
        ("original_task", "Original task"),
        ("latest_constraints", "Latest user constraints"),
        ("current_plan", "Current plan"),
        ("diff_summary", "Current diff summary"),
        ("key_locations", "Key code locations"),
        ("test_evidence", "Key tests/errors"),
        ("completed", "Completed"),
        ("pending", "Pending"),
        ("tool_evidence", "Important tool evidence"),
    )
    for key, label in labels:
        value = state.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            rendered = "; ".join(str(item) for item in value[-12:])
        else:
            rendered = str(value)
        sections.append(f"{label}: {rendered[:4000]}")
    if lines:
        sections.append("Earlier turns:\n" + "\n".join(lines))
    summary = "\n".join(sections)
    if redactor:
        summary = redactor.redact_text(summary)
    result: list[Message] = [{"role": "system", "content": summary}, *recent]
    return result, {
        "before_tokens": before,
        "after_tokens": estimate_tokens(result),
        "compacted": len(older),
    }
