from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from forgeloop.evals import default_suite_path
from forgeloop.security import SecretRedactor

DATASET_SCHEMA_VERSION = "forgeloop.dataset.sample.v1"
DATASET_MANIFEST_VERSION = "forgeloop.dataset.manifest.v1"
SFT_SCHEMA_VERSION = "forgeloop.sft.conversation.v1"

SFT_CANDIDATE = "sft_candidate"
SUCCESSFUL_BUT_INEFFICIENT = "successful_but_inefficient"
MODEL_FAILURE = "model_failure"
INFRASTRUCTURE_FAILURE = "infrastructure_failure"
CLASSIFICATIONS = (
    SFT_CANDIDATE,
    SUCCESSFUL_BUT_INEFFICIENT,
    MODEL_FAILURE,
    INFRASTRUCTURE_FAILURE,
)
INFRASTRUCTURE_CATEGORIES = {
    "forgeloop_harness_failure",
    "environment_eval_failure",
    "infrastructure_failure",
}
INEFFICIENT_STOP_REASONS = {
    "no_progress",
    "repeated_tool_call",
    "repeated_error",
    "budget_guard",
}
INEFFICIENT_TERMINAL_STATES = {
    "blocked",
    "budget_exceeded",
    "failed",
    "interrupted",
}


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetBuildResult:
    index_path: Path
    manifest_path: Path
    samples: int
    classifications: dict[str, int]
    skipped: dict[str, int]


class TrainingDataSanitizer:
    """Defense-in-depth redaction applied when building and exporting datasets."""

    _sensitive_keys = {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth_token",
        "token",
        "authorization",
        "client_secret",
        "provider_credential",
        "provider_credentials",
        "password",
        "credential",
        "credentials",
    }
    _credential_assignment = re.compile(
        r"(?i)\b((?:[a-z][a-z0-9]*[_-])*(?:api[_-]?key|access[_-]?key|"
        r"access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"provider[_-]?credentials?|password|token|secret|credentials?))"
        r"(\s*[:=]\s*)([\"']?)[^\s,;\"']{6,}"
    )
    _provider_tokens = (
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
        re.compile(r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{16,}"),
        re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}"),
        re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
        re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
        re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{20,}"),
    )
    _windows_home = re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s\"']+")
    _posix_home = re.compile(r"/(?:home|Users)/[^/\s\"']+")

    def __init__(
        self,
        *,
        redactor: SecretRedactor | None = None,
        local_roots: Iterable[Path | str] = (),
    ) -> None:
        self.redactor = redactor or SecretRedactor.from_environment()
        roots = [Path.home(), Path.cwd(), *[Path(root) for root in local_roots]]
        self.local_roots = tuple(
            sorted(
                {str(root.expanduser().resolve()) for root in roots},
                key=len,
                reverse=True,
            )
        )

    def sanitize(self, value: Any, *, key: str | None = None) -> Any:
        if key and key.lower() in self._sensitive_keys:
            return "[REDACTED]"
        if isinstance(value, str):
            return self.sanitize_text(value)
        if isinstance(value, dict):
            return {
                str(item_key): self.sanitize(item, key=str(item_key))
                for item_key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item) for item in value]
        return value

    def sanitize_text(self, value: str) -> str:
        redacted = self.redactor.redact_text(value)
        for root in self.local_roots:
            variants = {root, root.replace("\\", "/"), root.replace("/", "\\")}
            for variant in sorted(variants, key=len, reverse=True):
                redacted = re.sub(
                    re.escape(variant),
                    "[LOCAL_ROOT]",
                    redacted,
                    flags=re.IGNORECASE,
                )
        redacted = self._windows_home.sub("[USER_HOME]", redacted)
        redacted = self._posix_home.sub("[USER_HOME]", redacted)
        redacted = self._credential_assignment.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
            redacted,
        )
        for pattern in self._provider_tokens:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Cannot read JSONL from {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetError(f"Expected a JSON object at {path}:{line_number}")
        events.append(value)
    return events


def discover_suite_paths(
    source_root: Path, explicit: Iterable[Path] = ()
) -> tuple[Path, ...]:
    candidates = [Path(path) for path in explicit]
    candidates.append(default_suite_path())
    candidates.extend((source_root.parent / "foundry").glob("*/tasks.json"))
    candidates.extend((Path.cwd() / ".forgeloop" / "foundry").glob("*/tasks.json"))
    unique: dict[Path, None] = {}
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            unique[resolved] = None
    return tuple(unique)


def load_task_registry(paths: Iterable[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    registry: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        data = _read_json(path)
        suite_id = str(data.get("suite_id") or "unknown-suite")
        for raw_task in data.get("tasks", []):
            if not isinstance(raw_task, dict) or "id" not in raw_task:
                continue
            task = dict(raw_task)
            task["_suite_path"] = str(path)
            task["_suite_kind"] = str(data.get("suite_kind") or "benchmark")
            registry[(suite_id, str(task["id"]))] = task
    return registry


def _normalize_tool_call(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("function"), dict):
        function = raw["function"]
        return {
            "id": str(raw.get("id") or ""),
            "name": str(function.get("name") or ""),
            "arguments": function.get("arguments") or {},
        }
    return {
        "id": str(raw.get("id") or ""),
        "name": str(raw.get("name") or ""),
        "arguments": raw.get("arguments") or {},
    }


def _message_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call["id"],
        "type": "function",
        "function": {"name": call["name"], "arguments": call["arguments"]},
    }


def extract_trajectory(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise DatasetError("Trajectory contains no events")
    ordered = sorted(events, key=lambda event: int(event.get("sequence", 0)))
    trajectory_id = str(ordered[0].get("run_id") or "")
    if not trajectory_id:
        raise DatasetError("Trajectory is missing run_id")

    messages: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    started: dict[str, Any] = {}
    finished: dict[str, Any] = {}
    runtime: dict[str, Any] = {"type": "unknown"}
    verifier: dict[str, Any] | None = None
    final_change: dict[str, Any] = {"diff": "", "status": ""}
    initialized_messages = False

    for event in ordered:
        event_type = str(event.get("type") or "")
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        sequence = int(event.get("sequence", 0))
        if event_type == "run_started":
            started = payload
        elif event_type == "model_request" and not initialized_messages:
            messages = [dict(message) for message in payload.get("messages", [])]
            tools = [dict(tool) for tool in payload.get("tools", [])]
            initialized_messages = True
        elif event_type == "model_response":
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": payload.get("content") or "",
            }
            response_calls = []
            for raw_call in payload.get("tool_calls", []):
                call = _normalize_tool_call(raw_call)
                if call:
                    response_calls.append(_message_tool_call(call))
            if response_calls:
                assistant["tool_calls"] = response_calls
            messages.append(assistant)
        elif event_type == "tool_call":
            call = _normalize_tool_call(payload)
            if call:
                call["sequence"] = sequence
                calls.append(call)
        elif event_type == "observation":
            observation = {
                "sequence": sequence,
                "tool_call_id": str(payload.get("tool_call_id") or ""),
                "tool": payload.get("tool"),
                "ok": payload.get("ok"),
                "output": str(payload.get("output") or ""),
                "metadata": payload.get("metadata") or {},
            }
            observations.append(observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": observation["tool_call_id"],
                    "content": observation["output"],
                }
            )
        elif event_type == "run_finished":
            finished = payload
        elif event_type == "eval_runtime_started":
            runtime = payload
        elif event_type == "eval_verifier":
            verifier = payload
        elif event_type == "eval_final_diff":
            final_change = {
                "diff": str(payload.get("diff") or ""),
                "status": str(payload.get("status") or ""),
            }

    return {
        "trajectory_id": trajectory_id,
        "messages": messages,
        "tools": tools,
        "tool_calls": calls,
        "tool_observations": observations,
        "started": started,
        "finished": finished,
        "runtime": runtime,
        "verifier": verifier,
        "final_change": final_change,
    }


def inefficiency_reasons(sample: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    terminal = str(sample.get("terminal_state") or "")
    stop_reason = str(sample.get("stop_reason") or "")
    if terminal in INEFFICIENT_TERMINAL_STATES:
        reasons.append(f"terminal_state:{terminal}")
    if stop_reason in INEFFICIENT_STOP_REASONS:
        reasons.append(f"stop_reason:{stop_reason}")

    signatures = Counter(
        json.dumps(
            {"name": call.get("name"), "arguments": call.get("arguments")},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for call in sample.get("tool_calls", [])
        if call.get("name") != "finish"
    )
    if signatures and max(signatures.values()) >= 3:
        reasons.append("redundant_tool_calls")

    failed_outputs = Counter(
        (observation.get("tool"), observation.get("output"))
        for observation in sample.get("tool_observations", [])
        if observation.get("ok") is False
    )
    if failed_outputs and max(failed_outputs.values()) >= 2:
        reasons.append("repeated_tool_error")
    return reasons


def classify_sample(sample: dict[str, Any]) -> tuple[str, list[str]]:
    terminal = sample.get("terminal", {})
    failure_category = str(terminal.get("failure_category") or "")
    verifier = sample.get("verifier_result")
    if failure_category in INFRASTRUCTURE_CATEGORIES or not isinstance(verifier, dict):
        reason = failure_category or "missing_verifier_result"
        return INFRASTRUCTURE_FAILURE, [reason]
    passed = bool(verifier.get("passed")) and bool(sample.get("task_success"))
    if not passed:
        return MODEL_FAILURE, [failure_category or "verifier_failed"]
    reasons = inefficiency_reasons(sample)
    if reasons:
        return SUCCESSFUL_BUT_INEFFICIENT, reasons
    return SFT_CANDIDATE, []


def _safe_locator(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _resolve_trajectory(raw_path: Any, run_dir: Path) -> Path | None:
    if not raw_path:
        return None
    supplied = Path(str(raw_path)).expanduser()
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.extend((run_dir / supplied, Path.cwd() / supplied))
    candidates.append(run_dir / "trajectories" / supplied.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _sample_id(trajectory_id: str, task_id: str, attempt: int) -> str:
    digest = hashlib.sha256(
        f"{trajectory_id}\0{task_id}\0{attempt}".encode("utf-8")
    ).hexdigest()
    return f"ds_{digest[:24]}"


class DatasetBuilder:
    def __init__(
        self,
        source_root: Path,
        output_dir: Path,
        *,
        suite_paths: Iterable[Path] = (),
        sanitizer: TrainingDataSanitizer | None = None,
    ) -> None:
        self.source_root = source_root.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.workspace_root = Path.cwd().resolve()
        self.suite_paths = discover_suite_paths(self.source_root, suite_paths)
        self.sanitizer = sanitizer or TrainingDataSanitizer(
            local_roots=(self.source_root, self.output_dir)
        )

    def build(self) -> DatasetBuildResult:
        if not self.source_root.is_dir():
            raise DatasetError(
                f"Trajectory source directory does not exist: {self.source_root}"
            )
        task_registry = load_task_registry(self.suite_paths)
        samples: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
        seen: set[str] = set()
        task_files = sorted(self.source_root.rglob("tasks.jsonl"))
        if not task_files:
            raise DatasetError(
                f"No eval tasks.jsonl files found under {self.source_root}"
            )

        for task_file in task_files:
            run_dir = task_file.parent
            summary_path = run_dir / "summary.json"
            summary = _read_json(summary_path) if summary_path.is_file() else {}
            suite_id = str(summary.get("suite_id") or "unknown-suite")
            eval_run_id = str(summary.get("run_id") or run_dir.name)
            try:
                records = _read_jsonl(task_file)
            except DatasetError:
                skipped["invalid_task_file"] += 1
                continue
            for record in records:
                trajectory_path = _resolve_trajectory(
                    record.get("trajectory_path"), run_dir
                )
                if trajectory_path is None:
                    skipped["missing_trajectory"] += 1
                    continue
                try:
                    trajectory = extract_trajectory(_read_jsonl(trajectory_path))
                    sample = self._build_sample(
                        record,
                        trajectory,
                        trajectory_path,
                        suite_id,
                        eval_run_id,
                        task_registry.get((suite_id, str(record.get("task_id") or ""))),
                    )
                except (DatasetError, OSError, ValueError, TypeError):
                    skipped["invalid_trajectory"] += 1
                    continue
                if sample["id"] in seen:
                    skipped["duplicate_sample"] += 1
                    continue
                validate_sample(sample)
                seen.add(sample["id"])
                samples.append(self.sanitizer.sanitize(sample))

        samples.sort(key=lambda sample: sample["id"])
        counts = Counter(sample["classification"] for sample in samples)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.output_dir / "index.jsonl"
        temporary = self.output_dir / "index.jsonl.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
                )
        temporary.replace(index_path)
        manifest = {
            "schema_version": DATASET_MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "index": index_path.name,
            "source": _safe_locator(self.source_root, self.workspace_root),
            "suite_sources": [
                _safe_locator(path, self.workspace_root) for path in self.suite_paths
            ],
            "samples": len(samples),
            "classifications": {name: counts.get(name, 0) for name in CLASSIFICATIONS},
            "skipped": dict(sorted(skipped.items())),
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.sanitizer.sanitize(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return DatasetBuildResult(
            index_path,
            manifest_path,
            len(samples),
            manifest["classifications"],
            manifest["skipped"],
        )

    def _build_sample(
        self,
        record: dict[str, Any],
        trajectory: dict[str, Any],
        trajectory_path: Path,
        suite_id: str,
        eval_run_id: str,
        task_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = task_metadata or {}
        source = metadata.get("source")
        source = source if isinstance(source, dict) else {}
        task_id = str(record.get("task_id") or metadata.get("id") or "unknown-task")
        attempt = int(record.get("attempt") or 1)
        goal = str(
            record.get("description")
            or metadata.get("description")
            or trajectory["started"].get("request")
            or ""
        )
        base_sha = str(
            record.get("expected_base_sha")
            or metadata.get("base_commit")
            or trajectory["started"].get("git", {}).get("head")
            or "unknown"
        )
        repo = str(source.get("repository") or f"forgeloop-eval://{suite_id}/{task_id}")
        source_type = (
            "real_repository_eval" if source.get("repository") else "fixture_eval"
        )
        verifier = record.get("verifier")
        if not isinstance(verifier, dict):
            verifier = trajectory.get("verifier")
        final_diff = str(record.get("final_diff") or trajectory["final_change"]["diff"])
        final_status = str(
            record.get("final_status") or trajectory["final_change"]["status"]
        )
        finished = trajectory["finished"]
        terminal_state = str(
            record.get("terminal_state") or finished.get("status") or "unknown"
        )
        stop_reason = str(
            record.get("stop_reason") or finished.get("stop_reason") or "unknown"
        )
        provenance = {
            "repo": repo,
            "base_sha": base_sha,
            "task_id": task_id,
            "trajectory_id": trajectory["trajectory_id"],
            "trajectory_path": _safe_locator(trajectory_path, self.workspace_root),
            "source_type": source_type,
            "suite_id": suite_id,
            "eval_run_id": eval_run_id,
            "attempt": attempt,
            "source_repository": source.get("repository"),
            "source_pr": source.get("pr"),
            "source_fix_commit": source.get("fix_commit"),
            "source_base_sha": source.get("base_sha"),
            "suite_path": (
                _safe_locator(Path(metadata["_suite_path"]), self.workspace_root)
                if metadata.get("_suite_path")
                else None
            ),
        }
        sample: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "id": _sample_id(trajectory["trajectory_id"], task_id, attempt),
            "classification": "",
            "classification_reasons": [],
            "task": {
                "id": task_id,
                "goal": goal,
                "mode": str(
                    metadata.get("mode") or trajectory["started"].get("mode") or "task"
                ),
                "attempt": attempt,
                "difficulty": str(
                    record.get("difficulty") or metadata.get("difficulty") or "unknown"
                ),
                "expected_outcome": str(
                    record.get("expected_outcome")
                    or metadata.get("expected_outcome")
                    or "completed"
                ),
                "tags": list(metadata.get("tags") or []),
            },
            "task_success": bool(record.get("success")),
            "repo": repo,
            "base_sha": base_sha,
            "task_provenance": provenance,
            "model": str(
                record.get("model") or trajectory["started"].get("model") or "unknown"
            ),
            "provider": record.get("provider"),
            "messages": trajectory["messages"],
            "tools": trajectory["tools"],
            "tool_calls": trajectory["tool_calls"],
            "tool_observations": trajectory["tool_observations"],
            "final_diff": final_diff,
            "final_status": final_status,
            "verifier_result": verifier,
            "terminal_state": terminal_state,
            "stop_reason": stop_reason,
            "terminal": {
                "state": terminal_state,
                "stop_reason": stop_reason,
                "failure_category": str(record.get("failure_category") or "unknown"),
                "failure_detail": record.get("failure_detail"),
            },
            "runtime": trajectory["runtime"],
            "usage": {
                "steps": int(record.get("steps") or 0),
                "model_calls": int(record.get("model_calls") or 0),
                "tool_calls": int(record.get("tool_calls") or 0),
                "tokens": {
                    "input": record.get("input_tokens"),
                    "output": record.get("output_tokens"),
                    "total": record.get("total_tokens"),
                    "cached": record.get("cached_tokens"),
                    "reasoning": record.get("reasoning_tokens"),
                },
                "cost_usd": record.get("total_cost_usd"),
                "cost_sources": list(record.get("cost_sources") or []),
                "wall_time_seconds": record.get("wall_time_seconds"),
            },
            "source_trajectory_id": trajectory["trajectory_id"],
            "source_type": source_type,
        }
        classification, reasons = classify_sample(sample)
        sample["classification"] = classification
        sample["classification_reasons"] = reasons
        return sample


def validate_sample(sample: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "id",
        "classification",
        "task",
        "repo",
        "base_sha",
        "task_provenance",
        "model",
        "provider",
        "messages",
        "tools",
        "tool_calls",
        "tool_observations",
        "final_diff",
        "verifier_result",
        "terminal_state",
        "runtime",
        "usage",
        "source_trajectory_id",
        "source_type",
    }
    missing = sorted(required - sample.keys())
    if missing:
        raise DatasetError("Dataset sample missing fields: " + ", ".join(missing))
    provenance = sample["task_provenance"]
    required_provenance = {
        "repo",
        "base_sha",
        "task_id",
        "trajectory_id",
        "source_type",
    }
    missing_provenance = sorted(required_provenance - provenance.keys())
    if missing_provenance:
        raise DatasetError(
            "Dataset sample missing provenance: " + ", ".join(missing_provenance)
        )
    if sample["classification"] not in CLASSIFICATIONS:
        raise DatasetError(f"Unknown classification: {sample['classification']}")


def resolve_index_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved / "index.jsonl" if resolved.is_dir() else resolved


def load_dataset(path: Path) -> list[dict[str, Any]]:
    index_path = resolve_index_path(path)
    if not index_path.is_file():
        raise DatasetError(f"Dataset index does not exist: {index_path}")
    samples = _read_jsonl(index_path)
    for sample in samples:
        validate_sample(sample)
    return samples


def inspect_dataset(path: Path) -> dict[str, Any]:
    samples = load_dataset(path)
    classifications = Counter(sample["classification"] for sample in samples)
    sources = Counter(sample["source_type"] for sample in samples)
    models = Counter(sample["model"] for sample in samples)
    repos = Counter(sample["repo"] for sample in samples)
    total_tokens: int | None = 0
    total_cost: float | None = 0.0
    total_steps = 0
    for sample in samples:
        tokens = sample["usage"]["tokens"].get("total")
        cost = sample["usage"].get("cost_usd")
        total_tokens = (
            None
            if total_tokens is None or tokens is None
            else total_tokens + int(tokens)
        )
        total_cost = (
            None if total_cost is None or cost is None else total_cost + float(cost)
        )
        total_steps += int(sample["usage"].get("steps") or 0)
    return {
        "samples": len(samples),
        "classifications": {
            name: classifications.get(name, 0) for name in CLASSIFICATIONS
        },
        "sources": dict(sorted(sources.items())),
        "models": dict(sorted(models.items())),
        "repositories": len(repos),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "average_steps": total_steps / len(samples) if samples else 0.0,
    }


class SFTConversationAdapter:
    """Framework-neutral conversation/tool-calling JSONL adapter."""

    def convert(self, sample: dict[str, Any]) -> dict[str, Any]:
        if sample["classification"] != SFT_CANDIDATE:
            raise DatasetError("Only sft_candidate samples can use the SFT adapter")
        provenance = sample["task_provenance"]
        return {
            "schema_version": SFT_SCHEMA_VERSION,
            "id": sample["id"],
            "messages": sample["messages"],
            "tools": sample["tools"],
            "metadata": {
                "model": sample["model"],
                "provider": sample["provider"],
                "task_id": provenance["task_id"],
                "repo": provenance["repo"],
                "base_sha": provenance["base_sha"],
                "trajectory_id": provenance["trajectory_id"],
                "source_type": provenance["source_type"],
                "verifier_passed": bool(sample["verifier_result"].get("passed")),
                "usage": sample["usage"],
            },
            "outcome": {
                "terminal_state": sample["terminal_state"],
                "final_diff": sample["final_diff"],
                "verifier": sample["verifier_result"],
            },
        }


def export_dataset(
    dataset_path: Path,
    output_path: Path,
    *,
    export_format: str = "sft",
    include_infrastructure: bool = False,
    sanitizer: TrainingDataSanitizer | None = None,
) -> tuple[int, Counter[str]]:
    samples = load_dataset(dataset_path)
    sanitizer = sanitizer or TrainingDataSanitizer(
        local_roots=(resolve_index_path(dataset_path).parent, output_path.parent)
    )
    if export_format == "sft":
        selected = [
            sample for sample in samples if sample["classification"] == SFT_CANDIDATE
        ]
        adapter = SFTConversationAdapter()
        exported = [adapter.convert(sample) for sample in selected]
    elif export_format == "internal":
        selected = [
            sample
            for sample in samples
            if include_infrastructure
            or sample["classification"] != INFRASTRUCTURE_FAILURE
        ]
        exported = selected
    else:
        raise DatasetError("format must be 'sft' or 'internal'")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in exported:
            sanitized = sanitizer.sanitize(sample)
            handle.write(
                json.dumps(sanitized, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(output_path)
    return len(exported), Counter(sample["classification"] for sample in selected)


__all__ = [
    "CLASSIFICATIONS",
    "DATASET_SCHEMA_VERSION",
    "DatasetBuilder",
    "DatasetError",
    "INFRASTRUCTURE_FAILURE",
    "MODEL_FAILURE",
    "SFT_CANDIDATE",
    "SFTConversationAdapter",
    "SUCCESSFUL_BUT_INEFFICIENT",
    "TrainingDataSanitizer",
    "classify_sample",
    "export_dataset",
    "inspect_dataset",
    "load_dataset",
    "validate_sample",
]
