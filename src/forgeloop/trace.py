from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class TraceError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrajectoryReadResult:
    events: list[dict[str, Any]]
    recovery_warning: str | None = None


def _read_trajectory(
    path: Path, *, recover_incomplete_tail: bool
) -> TrajectoryReadResult:
    events: list[dict[str, Any]] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TraceError(f"Cannot read trajectory {path}: {exc}") from exc
    lines = raw.splitlines()
    terminated = raw.endswith((b"\n", b"\r"))
    recovery_warning: str | None = None
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_incomplete_tail = line_number == len(lines) and not terminated
            if recover_incomplete_tail and is_incomplete_tail and events:
                recovery_warning = (
                    "ignored an incomplete final trajectory record at "
                    f"{path}:{line_number}"
                )
                break
            raise TraceError(
                f"Invalid trajectory JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise TraceError(f"Expected an event object at {path}:{line_number}")
        events.append(event)
    if not events:
        raise TraceError(f"Trajectory contains no events: {path}")
    return TrajectoryReadResult(
        sorted(events, key=lambda event: int(event.get("sequence", 0))),
        recovery_warning,
    )


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    """Load a complete trajectory strictly for replay, analysis, or export."""

    return _read_trajectory(path, recover_incomplete_tail=False).events


def default_trace_roots() -> tuple[Path, ...]:
    return (
        Path(".forgeloop/runs"),
        Path(".forgeloop/eval-runs"),
        Path.home() / ".forgeloop" / "trajectories",
    )


def resolve_trajectory(
    reference: str | Path, *, search_roots: Iterable[Path] | None = None
) -> Path:
    supplied = Path(reference).expanduser()
    if supplied.is_file():
        return supplied.resolve()
    roots = tuple(search_roots or default_trace_roots())
    matches: list[Path] = []
    requested = str(reference)
    for root in roots:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            continue
        for path in resolved_root.rglob("*.jsonl"):
            if path.name in {
                "tasks.jsonl",
                "index.jsonl",
                "sft.jsonl",
                "curated.jsonl",
            }:
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    first = next(line for line in handle if line.strip())
                run_id = str(json.loads(first).get("run_id") or "")
            except (OSError, StopIteration, json.JSONDecodeError, AttributeError):
                continue
            if run_id == requested:
                matches.append(path.resolve())
    unique = sorted(set(matches))
    if not unique:
        raise TraceError(f"Trajectory id or path was not found: {reference}")
    if len(unique) > 1:
        raise TraceError(
            f"Trajectory id is ambiguous ({len(unique)} matches); pass an exact path"
        )
    return unique[0]


def _short(value: Any, limit: int = 110) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tool_title(payload: dict[str, Any]) -> str:
    name = str(payload.get("name") or "unknown")
    arguments = payload.get("arguments") or {}
    if name == "read_file":
        return f"Read {arguments.get('path', '')}"
    if name == "search_files":
        return (
            f"Search {_short(arguments.get('pattern'))} in {arguments.get('path', '.')}"
        )
    if name == "apply_patch":
        return f"Edit {arguments.get('path', '')}"
    if name == "shell":
        return f"Shell {_short(arguments.get('command'))}"
    if name in {"git_diff", "git_inspect"}:
        return f"Git inspect {arguments.get('operation', 'diff')}"
    if name == "finish":
        return f"Finish {arguments.get('status', 'unknown')}"
    return (
        f"Tool {name} {_short(json.dumps(arguments, ensure_ascii=False, default=str))}"
    )


def _effect_title(effect: dict[str, Any]) -> str:
    effect_type = str(effect.get("type") or "unknown")
    target = str(effect.get("target") or "")
    action = effect.get("action") or {}
    result = effect.get("result") or {}
    status = str(result.get("status") or "unknown").upper()
    if effect_type == "shell.exec":
        return f"Effect shell.exec {status}: {_short(action.get('command'))}"
    if effect_type == "test.run":
        return f"Test {status}: {_short(action.get('command'))}"
    if effect_type == "git.change":
        paths = action.get("changed_paths") or []
        detail = ", ".join(str(path) for path in paths[:4]) or "repository state"
        return f"Git change: {detail}"
    if effect_type == "policy.violation":
        return f"Policy violation {status}: {_short(action.get('operation'))} {target}"
    return f"Effect {effect_type} {status}: {target}"


def replay_lines(events: list[dict[str, Any]]) -> list[str]:
    timeline: list[tuple[int, str]] = []
    for event in events:
        sequence = int(event.get("sequence", 0))
        event_type = str(event.get("type") or "")
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        if event_type == "run_started":
            timeline.append((sequence, f"User: {_short(payload.get('request'))}"))
        elif event_type == "model_response":
            content = _short(payload.get("content"))
            calls = payload.get("tool_calls") or []
            if content:
                timeline.append((sequence, f"Model: {content}"))
            if calls:
                names = [
                    str(call.get("name") or "unknown")
                    for call in calls
                    if isinstance(call, dict)
                ]
                timeline.append((sequence, f"Model selected: {', '.join(names)}"))
        elif event_type == "tool_call":
            timeline.append((sequence, _tool_title(payload)))
        elif event_type == "observation":
            ok = payload.get("ok")
            output = _short(payload.get("output"), 90)
            if ok is False:
                timeline.append((sequence, f"Observation ERROR: {output}"))
            elif payload.get("tool") in {"shell", "apply_patch"}:
                timeline.append((sequence, f"Observation OK: {output}"))
        elif event_type == "effect":
            timeline.append((sequence, _effect_title(payload)))
        elif event_type == "eval_verifier":
            status = "PASS" if payload.get("passed") else "FAIL"
            timeline.append(
                (sequence, f"Verifier {status}: {_short(payload.get('command'))}")
            )
        elif event_type == "run_error":
            timeline.append((sequence, f"Run error: {_short(payload.get('message'))}"))
        elif event_type == "run_finished":
            timeline.append(
                (
                    sequence,
                    f"Terminal {payload.get('status', 'unknown')} ({payload.get('stop_reason', 'unknown')})",
                )
            )
    width = max(2, len(str(len(timeline))))
    return [
        f"{index:0{width}d} {title}  [seq={sequence}]"
        for index, (sequence, title) in enumerate(timeline, 1)
    ]


def replay_trajectory(path: Path, events: list[dict[str, Any]] | None = None) -> str:
    read_result = (
        TrajectoryReadResult(events)
        if events is not None
        else _read_trajectory(path, recover_incomplete_tail=True)
    )
    loaded = read_result.events
    run_id = str(loaded[0].get("run_id") or path.stem)
    lines = [
        f"Trajectory: {run_id}",
        f"Source: {path}",
        "Mode: offline evidence only",
    ]
    if read_result.recovery_warning:
        lines.append(f"Evidence warning: {read_result.recovery_warning}")
    lines.append("")
    lines.extend(replay_lines(loaded))
    return "\n".join(lines) + "\n"


def _effect_signature(effect: dict[str, Any]) -> str:
    action = effect.get("action") or {}
    return json.dumps(
        {
            "type": effect.get("type"),
            "target": effect.get("target"),
            "operation": action.get("operation"),
            "command": action.get("command"),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _diff_paths(diff: str) -> set[str]:
    paths: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git a/"):
            continue
        parts = line.split(" b/", 1)
        if len(parts) == 2:
            paths.add(parts[1])
    return paths


def analyze_trajectory(events: list[dict[str, Any]]) -> dict[str, Any]:
    verifier: dict[str, Any] | None = None
    verifier_sequence: int | None = None
    terminal: dict[str, Any] = {}
    effects: list[tuple[int, dict[str, Any]]] = []
    tool_calls: list[tuple[int, dict[str, Any]]] = []
    context_calls: list[dict[str, Any]] = []
    failed_observations: list[tuple[int, dict[str, Any]]] = []
    modified_files: set[str] = set()
    final_diff = ""
    for event in events:
        sequence = int(event.get("sequence", 0))
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        event_type = str(event.get("type") or "")
        if event_type == "effect":
            effects.append((sequence, payload))
            if payload.get("type") in {"file.write", "file.delete"} and payload.get(
                "target"
            ):
                modified_files.add(str(payload["target"]))
            if payload.get("type") == "git.change":
                modified_files.update(
                    str(path)
                    for path in (payload.get("action") or {}).get("changed_paths") or []
                )
        elif event_type == "tool_call":
            tool_calls.append((sequence, payload))
        elif event_type == "context_usage":
            context_calls.append(payload)
        elif event_type == "observation" and payload.get("ok") is False:
            failed_observations.append((sequence, payload))
        elif event_type == "eval_verifier":
            verifier = payload
            verifier_sequence = sequence
        elif event_type == "eval_final_diff":
            final_diff = str(payload.get("diff") or "")
        elif event_type == "run_finished":
            terminal = payload
    modified_files.update(_diff_paths(final_diff))

    tests = [
        {
            "step": effect.get("step"),
            "status": (effect.get("result") or {}).get("status"),
            "command": (effect.get("action") or {}).get("command"),
        }
        for _, effect in effects
        if effect.get("type") == "test.run"
    ]
    signature_events: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for sequence, effect in effects:
        signature_events[_effect_signature(effect)].append((sequence, effect))
    repeated = [
        {
            "count": len(items),
            "type": items[0][1].get("type"),
            "target": items[0][1].get("target"),
            "step": items[1][1].get("step"),
            "command": (items[0][1].get("action") or {}).get("command"),
            "sequence": items[1][0],
        }
        for items in signature_events.values()
        if len(items) >= 2
    ]
    effect_steps_by_call = {
        str(effect.get("tool_call_id") or ""): effect.get("step")
        for _, effect in effects
        if effect.get("tool_call_id")
    }
    tool_signature_events: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(
        list
    )
    for sequence, call in tool_calls:
        signature = json.dumps(
            {"name": call.get("name"), "arguments": call.get("arguments") or {}},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        tool_signature_events[signature].append((sequence, call))
    repeated.extend(
        {
            "count": len(items),
            "type": f"tool.{items[0][1].get('name')}",
            "target": (items[0][1].get("arguments") or {}).get("path") or "",
            "step": effect_steps_by_call.get(str(items[1][1].get("id") or "")),
            "command": (items[0][1].get("arguments") or {}).get("command"),
            "sequence": items[1][0],
        }
        for items in tool_signature_events.values()
        if len(items) >= 2 and items[0][1].get("name") != "finish"
    )
    repeated.sort(key=lambda item: item["sequence"])

    risk_events = []
    risk_flags: set[str] = set()
    for sequence, effect in effects:
        risk = effect.get("risk") or {}
        flags = [str(flag) for flag in risk.get("flags") or []]
        risk_flags.update(flags)
        if (
            effect.get("type") in {"file.delete", "policy.violation"}
            or str(risk.get("level") or "low") in {"high", "critical"}
            or flags
        ):
            risk_events.append((sequence, effect))

    post_pass = None
    if verifier and verifier.get("passed") and verifier_sequence is not None:
        post_pass = next(
            (
                (sequence, effect)
                for sequence, effect in effects
                if sequence > verifier_sequence
            ),
            None,
        )

    failed_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for sequence, effect in effects:
        status = str((effect.get("result") or {}).get("status") or "")
        if status in {"failed", "fail", "error", "blocked"}:
            failed_groups[_effect_signature(effect)].append((sequence, effect))
    effect_error_loops = [items for items in failed_groups.values() if len(items) >= 2]
    effect_error_loops.sort(key=lambda items: items[1][0])
    observation_groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for sequence, observation in failed_observations:
        signature = json.dumps(
            {
                "tool": observation.get("tool"),
                "output": observation.get("output"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        observation_groups[signature].append((sequence, observation))
    observation_error_loops = [
        items for items in observation_groups.values() if len(items) >= 2
    ]
    observation_error_loops.sort(key=lambda items: items[1][0])

    first_issue: dict[str, Any] | None = None
    if risk_events:
        sequence, effect = min(risk_events, key=lambda item: item[0])
        first_issue = {
            "sequence": sequence,
            "step": effect.get("step"),
            "effect": effect,
            "reason": "Recorded destructive/sensitive effect or explicit safety flag.",
        }
    if post_pass and (first_issue is None or post_pass[0] < first_issue["sequence"]):
        sequence, effect = post_pass
        reason = (
            "Verifier had already passed, but the trajectory recorded another effect."
        )
        signature = _effect_signature(effect)
        if len(signature_events[signature]) >= 2:
            reason = "Verifier had already passed, but the Agent continued a repeated action."
        first_issue = {
            "sequence": sequence,
            "step": effect.get("step"),
            "effect": effect,
            "reason": reason,
        }
    if first_issue is None and effect_error_loops:
        sequence, effect = effect_error_loops[0][1]
        first_issue = {
            "sequence": sequence,
            "step": effect.get("step"),
            "effect": effect,
            "reason": "The same failing effect was recorded repeatedly.",
        }
    if first_issue is None and observation_error_loops:
        sequence, observation = observation_error_loops[0][1]
        first_issue = {
            "sequence": sequence,
            "step": effect_steps_by_call.get(
                str(observation.get("tool_call_id") or "")
            ),
            "effect": {
                "type": "tool.error",
                "target": observation.get("tool") or "unknown",
                "result": {"status": "failed"},
            },
            "reason": "The same tool error observation was recorded repeatedly.",
        }
    if first_issue is None and repeated:
        item = repeated[0]
        repeated_effect = next(
            (effect for sequence, effect in effects if sequence == item["sequence"]),
            {
                "type": item["type"],
                "target": item["target"],
                "action": {"command": item["command"]},
                "result": {"status": "repeated"},
            },
        )
        first_issue = {
            "sequence": item["sequence"],
            "step": item["step"],
            "effect": repeated_effect,
            "reason": "The same effect was recorded repeatedly.",
        }

    no_progress_context = None
    if terminal.get("stop_reason") == "no_progress" and effects:
        sequence, effect = effects[-1]
        no_progress_context = {
            "sequence": sequence,
            "step": effect.get("step"),
            "effect": effect,
        }
        if first_issue is None:
            first_issue = {
                **no_progress_context,
                "reason": "This is the last recorded effect before no_progress termination.",
            }

    return {
        "outcome": (
            "PASS"
            if verifier and verifier.get("passed")
            else "FAIL"
            if verifier
            else "UNKNOWN"
        ),
        "verifier": verifier,
        "terminal_state": str(terminal.get("status") or "unknown"),
        "stop_reason": str(terminal.get("stop_reason") or "unknown"),
        "modified_files": sorted(path for path in modified_files if path),
        "tests": tests,
        "context_calls": context_calls,
        "repeated_actions": repeated,
        "error_loops": len(effect_error_loops) + len(observation_error_loops),
        "risk_flags": sorted(risk_flags),
        "first_issue": first_issue,
        "post_verifier_pass_effect": post_pass is not None,
        "no_progress_context": no_progress_context,
    }


def explain_trajectory(path: Path, events: list[dict[str, Any]] | None = None) -> str:
    read_result = (
        TrajectoryReadResult(events)
        if events is not None
        else _read_trajectory(path, recover_incomplete_tail=True)
    )
    analysis = analyze_trajectory(read_result.events)
    lines: list[str] = []
    if read_result.recovery_warning:
        lines.extend([f"Evidence warning: {read_result.recovery_warning}", ""])
    lines.extend(
        [
            f"Outcome: {analysis['outcome']}",
            f"Termination: {analysis['terminal_state']}/{analysis['stop_reason']}",
            "",
            "Affected files:",
        ]
    )
    lines.extend(
        [f"- {path}" for path in analysis["modified_files"]] or ["- none recorded"]
    )
    lines.extend(["", "Test results:"])
    lines.extend(
        [
            f"- Step {test['step']}: {str(test['status']).upper()} {_short(test['command'])}"
            for test in analysis["tests"]
        ]
        or ["- none recorded"]
    )
    lines.extend(["", "Model input context:"])
    lines.extend(
        [_context_call_line(call) for call in analysis["context_calls"]]
        or ["- no per-call context metrics recorded"]
    )
    lines.extend(["", "Repeated actions:"])
    lines.extend(
        [
            f"- {item['count']}x {item['type']} {_short(item['command'] or item['target'])}"
            for item in analysis["repeated_actions"]
        ]
        or ["- none"]
    )
    lines.extend(["", "Safety/risk flags:"])
    lines.extend([f"- {flag}" for flag in analysis["risk_flags"]] or ["- none"])
    issue = analysis["first_issue"]
    lines.extend(["", "First suspicious / unnecessary / harmful effect:"])
    if issue:
        effect = issue["effect"]
        location = (
            f"Step {issue['step']}"
            if issue.get("step") is not None
            else f"Sequence {issue['sequence']}"
        )
        lines.append(f"{location} [seq={issue['sequence']}]")
        lines.append(_effect_title(effect))
        lines.extend(["", "Reason:", issue["reason"]])
    else:
        lines.append("None supported by recorded evidence.")
    context = analysis["no_progress_context"]
    if context:
        lines.extend(
            [
                "",
                "Key event before no_progress:",
                f"Step {context['step']} [seq={context['sequence']}] {_effect_title(context['effect'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def _context_call_line(call: dict[str, Any]) -> str:
    actual = call.get("input_tokens")
    estimated = call.get("estimated_input_tokens")
    size = f"input={actual if actual is not None else 'unknown'}"
    if estimated is not None:
        size += f" estimated={estimated}"
    if call.get("applied"):
        size += (
            f" compacted={call.get('before_estimated_tokens')}"
            f"->{call.get('after_estimated_tokens')}"
        )
    cache = call.get("prompt_cache") or {}
    cache_ratio = cache.get("warm_prefix_hit_ratio")
    if cache_ratio is not None:
        size += f" warm-cache={100 * cache_ratio:.2f}%"
        if cache.get("missed_reusable_tokens"):
            size += f" miss={cache['missed_reusable_tokens']}"
    elif cache.get("status"):
        size += f" cache={cache['status']}"
    dominant = ", ".join(
        f"{item.get('source')}={item.get('chars')} chars"
        for item in (call.get("dominant_sources") or [])[:3]
    )
    return f"- Step {call.get('step')}: {size}" + (
        f"; main: {dominant}" if dominant else ""
    )


__all__ = [
    "TraceError",
    "analyze_trajectory",
    "explain_trajectory",
    "load_trajectory",
    "replay_lines",
    "replay_trajectory",
    "resolve_trajectory",
]
