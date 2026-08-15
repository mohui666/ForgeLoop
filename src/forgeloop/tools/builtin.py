from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

from forgeloop.effects import EffectDraft
from forgeloop.runtime import LocalRuntime, Runtime
from forgeloop.security import ShellSafetyPolicy, is_sensitive_path
from forgeloop.tools.base import BaseTool, ToolRegistry, ToolResult
from forgeloop.workspace import Workspace, is_ephemeral_git_path


_GIT_REVIEW_PATHS = (
    ".",
    ":(exclude,glob).forgeloop/**",
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/.pytest_cache/**",
    ":(exclude,glob)**/.mypy_cache/**",
    ":(exclude,glob)**/.ruff_cache/**",
    ":(exclude,glob)**/*.pyc",
    ":(exclude,glob)**/*.pyo",
    ":(exclude,glob)**/.coverage",
)


def _normalize_git_path(runtime: Runtime, value: str) -> str:
    if runtime.shell_environment.syntax == "PowerShell":
        return value.replace("\\", "/")
    return value


def _file_state(runtime: Runtime, path: Path, workspace: Workspace) -> dict:
    kind = runtime.path_kind(path)
    state = {
        "path": workspace.relative(path),
        "kind": kind,
        "exists": kind != "missing",
    }
    if kind == "file":
        content = runtime.read_bytes(path)
        state.update(
            {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    return state


def _status_paths(status: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for line in status.splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        path = line[3:].split(" -> ")[-1]
        if path:
            paths[path] = code
    return paths


def _render_untracked_diff(
    workspace: Workspace,
    runtime: Runtime,
    relative_path: str,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    """Render an untracked file as reviewable new-file evidence."""

    normalized = _normalize_git_path(runtime, relative_path)
    header = (
        f"diff --git a/{normalized} b/{normalized}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{normalized}\n"
    )
    if is_sensitive_path(normalized):
        return header + "[sensitive untracked content withheld]\n", False
    try:
        path = workspace.resolve(relative_path, must_exist=True)
        if runtime.path_kind(path) != "file":
            return header + "[untracked path is not a regular file]\n", False
        content = runtime.read_bytes(path)
    except (OSError, ValueError):
        return header + "[untracked file could not be read]\n", False
    digest = hashlib.sha256(content).hexdigest()
    if b"\0" in content:
        return (
            header + f"Binary file: {len(content)} bytes, sha256={digest}\n",
            True,
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return (
            header + f"Non-UTF-8 file: {len(content)} bytes, sha256={digest}\n",
            True,
        )
    lines = text.splitlines(keepends=True)
    if not lines:
        return header + "@@ -0,0 +0,0 @@\n", True
    body = "".join("+" + line for line in lines)
    if not text.endswith(("\n", "\r")):
        body += "\n\\ No newline at end of file\n"
    rendered = header + f"@@ -0,0 +1,{len(lines)} @@\n" + body
    if len(rendered) <= max_chars:
        return rendered, True
    digest_note = (
        f"\n... <untracked diff truncated; {len(content)} bytes, sha256={digest}>\n"
    )
    available = max(max_chars - len(header) - len(digest_note), 0)
    return header + body[:available] + digest_note, False


def _command_output_complete(*results: object) -> bool:
    for result in results:
        if result is None:
            continue
        if bool(getattr(result, "stdout_truncated", False)) or bool(
            getattr(result, "stderr_truncated", False)
        ):
            return False
        if any(
            "chars omitted" in str(getattr(result, stream, ""))
            for stream in ("stdout", "stderr")
        ):
            return False
    return True


def _shell_quote(runtime: Runtime, value: str) -> str:
    if runtime.shell_environment.syntax == "PowerShell":
        return "'" + value.replace("'", "''") + "'"
    return shlex.quote(value)


def _git_review_pathspec(
    runtime: Runtime,
    path: str | None,
    *,
    sensitive_paths: tuple[str, ...] = (),
) -> str:
    paths = (
        (f":(literal){_normalize_git_path(runtime, path)}",)
        if path
        else _GIT_REVIEW_PATHS
    )
    paths += tuple(
        f":(exclude,literal){_normalize_git_path(runtime, item)}"
        for item in sensitive_paths
    )
    return " -- " + " ".join(_shell_quote(runtime, item) for item in paths)


def _compact_git(snapshot) -> dict:
    return {
        "is_repository": snapshot.is_repository,
        "head": snapshot.head,
        "branch": snapshot.branch,
        "status": snapshot.status,
        "error": snapshot.error,
    }


def _git_effect(
    before,
    after,
    *,
    origin: str,
    before_fingerprint: str | None = None,
    after_fingerprint: str | None = None,
) -> EffectDraft | None:
    if not before.is_repository or not after.is_repository:
        return None
    if (
        before.head == after.head
        and before.branch == after.branch
        and before.status == after.status
        and (before_fingerprint is None or before_fingerprint == after_fingerprint)
    ):
        return None
    before_paths = _status_paths(before.status)
    after_paths = _status_paths(after.status)
    changed_paths = sorted(
        path
        for path in set(before_paths) | set(after_paths)
        if before_paths.get(path) != after_paths.get(path)
    )
    return EffectDraft(
        "git.change",
        ".",
        action={
            "origin": origin,
            "before": _compact_git(before),
            "after": _compact_git(after),
            "changed_paths": changed_paths,
            "content_changed": (
                before_fingerprint != after_fingerprint
                if before_fingerprint is not None
                else None
            ),
        },
        result={"status": "changed"},
        evidence={"changed_file_count": len(changed_paths)},
    )


def _shell_file_effects(before, after, *, command: str) -> tuple[EffectDraft, ...]:
    before_paths = _status_paths(before.status if before.is_repository else "")
    after_paths = _status_paths(after.status if after.is_repository else "")
    effects: list[EffectDraft] = []
    for path, code in after_paths.items():
        if before_paths.get(path) == code:
            continue
        effect_type = "file.delete" if "D" in code else "file.write"
        effects.append(
            EffectDraft(
                effect_type,
                path,
                action={"origin": "shell", "command": command, "git_status": code},
                result={
                    "status": "changed" if effect_type == "file.write" else "deleted"
                },
            )
        )
    return tuple(effects)


def _is_test_command(command: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?:^|[;&|]\s*|\s)(?:python\s+-m\s+)?(?:pytest|unittest|"
            r"cargo\s+test|go\s+test|dotnet\s+test|npm\s+(?:run\s+)?test|"
            r"pnpm\s+(?:run\s+)?test|yarn\s+test)(?:\s|$)",
            command,
        )
    )


def _shell_risk(command: str) -> dict:
    flags: list[str] = []
    if re.search(r"(?i)\b(?:rm|del|erase|remove-item|unlink|rmdir|rd)\b", command):
        flags.append("destructive_operation")
    if re.search(r"(?i)\bgit\s+push\b", command):
        flags.append("remote_git_write")
    return {
        "level": "high" if flags else "low",
        "flags": flags,
        "safety_policy": "ShellSafetyPolicy",
    }


def _excerpt(value: str, limit: int = 2_000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... <{len(value) - limit} chars omitted>"


@dataclass
class ReadFileTool(BaseTool):
    workspace: Workspace
    runtime: Runtime = field(default_factory=LocalRuntime)
    max_chars: int = 40_000
    name = "read_file"
    description = "Read a UTF-8 text file in the workspace, optionally selecting an inclusive line range."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        del timeout_seconds
        target = str(arguments["path"])
        if is_sensitive_path(arguments["path"]):
            return ToolResult(
                False,
                "Reading credential or VCS-internal files is blocked",
                effects=(
                    EffectDraft(
                        "policy.violation",
                        target,
                        action={"operation": "file.read"},
                        result={"status": "blocked"},
                        risk={
                            "level": "high",
                            "flags": ["sensitive_path", "blocked"],
                            "safety_policy": "sensitive_path",
                        },
                        evidence={"reason": "credential or VCS-internal path"},
                    ),
                ),
            )
        path = self.workspace.resolve(arguments["path"])
        kind = self.runtime.path_kind(path)
        if kind != "file":
            return ToolResult(
                False,
                f"Not a file: {arguments['path']}",
                effects=(
                    EffectDraft(
                        "file.read",
                        target,
                        action={"start_line": arguments.get("start_line", 1)},
                        result={"status": "failed", "reason": "not_a_file"},
                    ),
                ),
            )
        start = int(arguments.get("start_line", 1))
        end = arguments.get("end_line")
        if start < 1 or (end is not None and int(end) < start):
            return ToolResult(
                False,
                "Invalid line range",
                effects=(
                    EffectDraft(
                        "file.read",
                        target,
                        action={"start_line": start, "end_line": end},
                        result={"status": "failed", "reason": "invalid_range"},
                    ),
                ),
            )
        lines = (
            self.runtime.read_bytes(path).decode("utf-8", errors="replace").splitlines()
        )
        selected = lines[start - 1 : int(end) if end is not None else None]
        numbered = "\n".join(
            f"{number}: {line}" for number, line in enumerate(selected, start=start)
        )
        if len(numbered) > self.max_chars:
            omitted = len(numbered) - self.max_chars
            numbered = numbered[: self.max_chars] + f"\n... <{omitted} chars omitted>"
        return ToolResult(
            True,
            numbered,
            {"path": self.workspace.relative(path), "line_count": len(lines)},
            effects=(
                EffectDraft(
                    "file.read",
                    self.workspace.relative(path),
                    action={"start_line": start, "end_line": end},
                    result={"status": "success"},
                    evidence={
                        "line_count": len(lines),
                        "selected_lines": len(selected),
                        "bytes_returned": len(numbered.encode("utf-8")),
                    },
                ),
            ),
        )


@dataclass
class SearchFilesTool(BaseTool):
    workspace: Workspace
    runtime: Runtime = field(default_factory=LocalRuntime)
    max_output_chars: int = 40_000
    name = "search_files"
    description = (
        "Search text in workspace files using a regex. Uses ripgrep when available."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression to search for.",
            },
            "path": {
                "type": "string",
                "default": ".",
                "description": "Workspace-relative directory or file.",
            },
            "glob": {
                "type": "string",
                "description": "Optional file glob such as *.py.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "default": 100,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        target_name = str(arguments.get("path", "."))
        if is_sensitive_path(target_name):
            return ToolResult(
                False,
                "Searching credential or VCS-internal files is blocked",
                effects=(
                    EffectDraft(
                        "policy.violation",
                        target_name,
                        action={"operation": "file.search"},
                        result={"status": "blocked"},
                        risk={
                            "level": "high",
                            "flags": ["sensitive_path", "blocked"],
                        },
                    ),
                ),
            )
        target = self.workspace.resolve(arguments.get("path", "."))
        if self.runtime.path_kind(target) == "missing":
            return ToolResult(
                False,
                f"Path does not exist: {arguments.get('path', '.')}",
                effects=(
                    EffectDraft(
                        "file.read",
                        target_name,
                        action={"operation": "search", "pattern": arguments["pattern"]},
                        result={"status": "failed", "reason": "missing_path"},
                    ),
                ),
            )
        max_results = min(int(arguments.get("max_results", 100)), 500)
        result = self.runtime.search_text(
            arguments["pattern"],
            target,
            arguments.get("glob"),
            max_results,
            timeout_seconds,
        )
        if result.error:
            return ToolResult(
                False,
                result.error,
                {"matches": len(result.matches)},
                effects=(
                    EffectDraft(
                        "file.read",
                        target_name,
                        action={"operation": "search", "pattern": arguments["pattern"]},
                        result={"status": "failed", "reason": result.error},
                    ),
                ),
            )
        output = "\n".join(result.matches) or "No matches."
        return ToolResult(
            True,
            output[: self.max_output_chars],
            {"matches": len(result.matches), "timed_out": result.timed_out},
            effects=(
                EffectDraft(
                    "file.read",
                    target_name,
                    action={
                        "operation": "search",
                        "pattern": arguments["pattern"],
                        "glob": arguments.get("glob"),
                    },
                    result={"status": "success"},
                    evidence={"matches": len(result.matches)},
                ),
            ),
        )


@dataclass
class ApplyPatchTool(BaseTool):
    workspace: Workspace
    runtime: Runtime = field(default_factory=LocalRuntime)
    name = "apply_patch"
    description = (
        "Apply an exact text replacement to one workspace file. The old_text must match exactly once "
        "unless replace_all is true. To create a new file, use empty old_text and a path that does not exist."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "old_text": {
                "type": "string",
                "description": "Exact text to replace; empty only when creating a file.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement or full new-file content.",
            },
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        del timeout_seconds
        target = str(arguments["path"])
        if is_sensitive_path(target):
            return ToolResult(
                False,
                "Writing credential or VCS-internal files is blocked",
                effects=(
                    EffectDraft(
                        "policy.violation",
                        target,
                        action={"operation": "file.write"},
                        result={"status": "blocked"},
                        risk={
                            "level": "high",
                            "flags": ["sensitive_path", "blocked"],
                            "safety_policy": "sensitive_path",
                        },
                    ),
                ),
            )
        path = self.workspace.resolve(target)
        old_text = arguments["old_text"]
        new_text = arguments["new_text"]
        replace_all = bool(arguments.get("replace_all", False))
        kind = self.runtime.path_kind(path)
        before_file = _file_state(self.runtime, path, self.workspace)
        before_git = self.workspace.git_snapshot()
        before_git_fingerprint = (
            self.workspace.git_progress_fingerprint()
            if before_git.is_repository
            else None
        )

        def failed(reason: str, output: str) -> ToolResult:
            return ToolResult(
                False,
                output,
                effects=(
                    EffectDraft(
                        "file.write",
                        target,
                        action={
                            "operation": "update" if kind != "missing" else "create"
                        },
                        result={"status": "failed", "reason": reason},
                        evidence={"before": before_file},
                    ),
                ),
            )

        if kind != "missing":
            if kind != "file":
                return failed("not_a_file", f"Not a file: {arguments['path']}")
            if old_text == "":
                return failed(
                    "empty_old_text",
                    "old_text cannot be empty when updating an existing file",
                )
            raw_content = self.runtime.read_bytes(path).decode("utf-8")
            uses_crlf = "\r\n" in raw_content
            content = raw_content.replace("\r\n", "\n")
            old_text = old_text.replace("\r\n", "\n")
            new_text = new_text.replace("\r\n", "\n")
            count = content.count(old_text)
            if count == 0:
                return failed(
                    "old_text_not_found",
                    "old_text was not found; read the latest file and retry",
                )
            if count > 1 and not replace_all:
                return failed(
                    "ambiguous_match",
                    f"old_text matched {count} times; provide more context or set replace_all",
                )
            updated = content.replace(old_text, new_text, -1 if replace_all else 1)
            if uses_crlf:
                updated = updated.replace("\n", "\r\n")
            self.runtime.write_bytes(path, updated.encode("utf-8"))
            after_file = _file_state(self.runtime, path, self.workspace)
            after_git = self.workspace.git_snapshot()
            after_git_fingerprint = (
                self.workspace.git_progress_fingerprint()
                if after_git.is_repository
                else None
            )
            effects = [
                EffectDraft(
                    "file.write",
                    self.workspace.relative(path),
                    action={
                        "operation": "update",
                        "replacement_count": count if replace_all else 1,
                        "replace_all": replace_all,
                    },
                    result={"status": "success"},
                    evidence={"before": before_file, "after": after_file},
                )
            ]
            git_effect = _git_effect(
                before_git,
                after_git,
                origin="apply_patch",
                before_fingerprint=before_git_fingerprint,
                after_fingerprint=after_git_fingerprint,
            )
            if git_effect:
                effects.append(git_effect)
            return ToolResult(
                True,
                f"Updated {self.workspace.relative(path)} ({count if replace_all else 1} replacement(s)).",
                effects=tuple(effects),
            )
        if old_text:
            return failed(
                "missing_file",
                "Cannot replace text in a file that does not exist",
            )
        self.runtime.write_bytes(path, new_text.encode("utf-8"))
        after_file = _file_state(self.runtime, path, self.workspace)
        after_git = self.workspace.git_snapshot()
        after_git_fingerprint = (
            self.workspace.git_progress_fingerprint()
            if after_git.is_repository
            else None
        )
        effects = [
            EffectDraft(
                "file.write",
                self.workspace.relative(path),
                action={"operation": "create"},
                result={"status": "success"},
                evidence={"before": before_file, "after": after_file},
            )
        ]
        git_effect = _git_effect(
            before_git,
            after_git,
            origin="apply_patch",
            before_fingerprint=before_git_fingerprint,
            after_fingerprint=after_git_fingerprint,
        )
        if git_effect:
            effects.append(git_effect)
        return ToolResult(
            True,
            f"Created {self.workspace.relative(path)}.",
            effects=tuple(effects),
        )


@dataclass
class ShellTool(BaseTool):
    workspace: Workspace
    runtime: Runtime
    max_timeout_seconds: float = 120.0
    safety: ShellSafetyPolicy = field(default_factory=ShellSafetyPolicy)
    name = "shell"
    description: str = field(init=False)
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {
                "type": "string",
                "default": ".",
                "description": "Workspace-relative working directory.",
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 120,
                "default": 30,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __post_init__(self) -> None:
        self.description = (
            "Run one independent shell command "
            f"{self.runtime.shell_environment.description}. "
            "No shell state persists between calls."
        )

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        command = str(arguments["command"])
        target_cwd = str(arguments.get("cwd", "."))
        rejection = self.safety.rejection(command)
        if rejection:
            risk = _shell_risk(command)
            risk["flags"] = sorted({*risk["flags"], "blocked"})
            return ToolResult(
                False,
                rejection,
                {"blocked": True},
                effects=(
                    EffectDraft(
                        "policy.violation",
                        target_cwd,
                        action={"operation": "shell.exec", "command": command},
                        result={"status": "blocked"},
                        risk=risk,
                        evidence={"reason": rejection},
                    ),
                ),
            )
        cwd = self.workspace.resolve(target_cwd)
        if self.runtime.path_kind(cwd) != "directory":
            return ToolResult(
                False,
                f"cwd is not a directory: {arguments.get('cwd')}",
                effects=(
                    EffectDraft(
                        "shell.exec",
                        target_cwd,
                        action={"command": command},
                        result={"status": "failed", "reason": "invalid_cwd"},
                        risk=_shell_risk(command),
                    ),
                ),
            )
        requested = float(arguments.get("timeout_seconds", 30))
        effective = max(0.1, min(requested, self.max_timeout_seconds, timeout_seconds))
        before_git = self.workspace.git_snapshot()
        before_fingerprint = (
            self.workspace.git_progress_fingerprint()
            if before_git.is_repository
            else None
        )
        result = self.runtime.run(command, cwd, effective)
        after_git = self.workspace.git_snapshot()
        after_fingerprint = (
            self.workspace.git_progress_fingerprint()
            if after_git.is_repository
            else None
        )
        output = (
            f"exit_code: {result.exit_code}\n"
            f"timed_out: {str(result.timed_out).lower()}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        status = "success" if result.exit_code == 0 else "failed"
        effects = [
            EffectDraft(
                "shell.exec",
                self.workspace.relative(cwd),
                action={"command": command, "timeout_seconds": effective},
                result={
                    "status": status,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                },
                risk=_shell_risk(command),
                evidence={
                    "stdout": _excerpt(result.stdout),
                    "stderr": _excerpt(result.stderr),
                },
            )
        ]
        if _is_test_command(command):
            effects.append(
                EffectDraft(
                    "test.run",
                    self.workspace.relative(cwd),
                    action={"command": command},
                    result={
                        "status": "pass" if result.exit_code == 0 else "fail",
                        "exit_code": result.exit_code,
                        "timed_out": result.timed_out,
                    },
                    evidence={
                        "stdout": _excerpt(result.stdout),
                        "stderr": _excerpt(result.stderr),
                    },
                )
            )
        effects.extend(_shell_file_effects(before_git, after_git, command=command))
        git_effect = _git_effect(
            before_git,
            after_git,
            origin="shell",
            before_fingerprint=before_fingerprint,
            after_fingerprint=after_fingerprint,
        )
        if git_effect:
            effects.append(git_effect)
        return ToolResult(
            result.exit_code == 0,
            output,
            asdict(result),
            tuple(effects),
        )


@dataclass
class ValidateTool(ShellTool):
    """Explicit validation with the same execution and safety policy as shell."""

    name = "validate"

    def __post_init__(self) -> None:
        self.description = (
            "Run one relevant test, behavioral check, lint, typecheck, or build command "
            f"{self.runtime.shell_environment.description}. The command must exercise "
            "the changed behavior and must not edit source files. A syntax/import-only "
            "probe is not sufficient for a behavior change."
        )

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        result = super().execute(arguments, timeout_seconds=timeout_seconds)
        metadata = {**(result.metadata or {}), "explicit_validation": True}
        if any(effect.type == "test.run" for effect in result.effects):
            effects = result.effects
        else:
            effects = (
                *result.effects,
                EffectDraft(
                    "test.run",
                    str(arguments.get("cwd", ".")),
                    action={
                        "command": str(arguments["command"]),
                        "explicit_validation": True,
                    },
                    result={
                        "status": "pass" if result.ok else "fail",
                        "exit_code": metadata.get("exit_code"),
                        "timed_out": bool(metadata.get("timed_out")),
                    },
                    evidence={"output": _excerpt(result.output)},
                ),
            )
        return ToolResult(result.ok, result.output, metadata, effects)


@dataclass
class GitDiffTool(BaseTool):
    workspace: Workspace
    runtime: Runtime
    max_untracked_diff_chars: int = 40_000
    run_base_head: str | None = field(default=None, init=False, repr=False)
    name = "git_diff"
    description = (
        "Show git status and the current diff without modifying the repository."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "cached": {"type": "boolean", "default": False},
            "path": {
                "type": "string",
                "description": "Optional workspace-relative path filter.",
            },
        },
        "additionalProperties": False,
    }

    def bind_run_context(self, *, base_head: str | None) -> None:
        self.run_base_head = base_head

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        path = str(arguments.get("path") or "") or None
        if path:
            self.workspace.resolve(path)
        cached_only = bool(arguments.get("cached", False))
        pathspec = _git_review_pathspec(self.runtime, path)
        timeout = min(timeout_seconds, 30.0)
        status_result = self.runtime.run(
            f"git status --short --untracked-files=all{pathspec}",
            self.workspace.root,
            timeout,
        )
        layers = (
            (("staged", " --cached", ""),)
            if cached_only
            else (
                (
                    "base_to_worktree",
                    "",
                    " " + _shell_quote(self.runtime, self.run_base_head),
                ),
            )
            if self.run_base_head
            else (
                ("staged", " --cached", ""),
                ("unstaged", "", ""),
            )
        )
        name_results = [
            (
                label,
                self.runtime.run(
                    f"git diff --no-renames{flag} --name-only -z{revision}{pathspec}",
                    self.workspace.root,
                    timeout,
                ),
            )
            for label, flag, revision in layers
        ]
        tracked_names_complete = all(
            result.exit_code == 0 and _command_output_complete(result)
            for _, result in name_results
        )
        sensitive_paths: tuple[str, ...] = ()
        if tracked_names_complete:
            sensitive_paths = tuple(
                sorted(
                    {
                        item
                        for _, result in name_results
                        for item in result.stdout.split("\0")
                        if item and is_sensitive_path(item)
                    }
                )
            )
        content_pathspec = _git_review_pathspec(
            self.runtime, path, sensitive_paths=sensitive_paths
        )
        diff_results = (
            [
                (
                    label,
                    self.runtime.run(
                        f"git diff --no-renames{flag}{revision}{content_pathspec}",
                        self.workspace.root,
                        timeout,
                    ),
                )
                for label, flag, revision in layers
            ]
            if tracked_names_complete
            else []
        )
        untracked_sections: list[str] = []
        untracked_paths: list[str] = []
        untracked_complete = True
        untracked_chars = 0
        untracked_contents_included = 0
        untracked_result = None
        if not cached_only:
            untracked_result = self.runtime.run(
                f"git ls-files -z --others --exclude-standard{pathspec}",
                self.workspace.root,
                timeout,
            )
            if untracked_result.exit_code == 0:
                if not _command_output_complete(untracked_result):
                    untracked_complete = False
                else:
                    untracked_paths = [
                        item
                        for item in untracked_result.stdout.split("\0")
                        if item and not is_ephemeral_git_path(item)
                    ]
                for relative_path in untracked_paths:
                    remaining = self.max_untracked_diff_chars - untracked_chars
                    if remaining <= 0:
                        untracked_sections.append(
                            "... <remaining untracked file diffs omitted>\n"
                        )
                        untracked_complete = False
                        break
                    rendered, complete = _render_untracked_diff(
                        self.workspace,
                        self.runtime,
                        relative_path,
                        max_chars=remaining,
                    )
                    untracked_sections.append(rendered)
                    untracked_chars += len(rendered)
                    untracked_complete = untracked_complete and complete
                    untracked_contents_included += int(complete)
            else:
                untracked_complete = False
        tracked_sections = [
            f"{label.replace('_', ' ').title()} changes:\n{result.stdout.rstrip()}"
            for label, result in diff_results
            if result.stdout.rstrip()
        ]
        if sensitive_paths:
            tracked_sections.append(
                "Sensitive tracked diff content withheld: " + ", ".join(sensitive_paths)
            )
        if not tracked_names_complete:
            tracked_sections.append(
                "Tracked diff content withheld because changed-path enumeration "
                "was incomplete."
            )
        command_results = [
            status_result,
            *(result for _, result in name_results),
            *(result for _, result in diff_results),
            untracked_result,
        ]
        errors = [
            item.stderr for item in command_results if item is not None and item.stderr
        ]
        output = "\n".join(
            item.rstrip()
            for item in (
                status_result.stdout,
                *tracked_sections,
                *untracked_sections,
            )
            if item.rstrip()
        )
        if errors:
            output += "\nstderr:\n" + "\n".join(errors)
        ok = all(result is None or result.exit_code == 0 for result in command_results)
        output_complete = (
            tracked_names_complete
            and not sensitive_paths
            and _command_output_complete(*command_results)
            and untracked_complete
        )
        review_scope = (
            "worktree"
            if ok and not path and not cached_only and output_complete
            else "partial"
        )
        metadata_source = diff_results[0][1] if diff_results else status_result
        metadata = asdict(metadata_source) | {
            "review_scope": review_scope,
            "path_filter": path,
            "cached_only": cached_only,
            "review_base": self.run_base_head,
            "tracked_diff_layers": [label for label, _ in diff_results],
            "tracked_sensitive_files_withheld": len(sensitive_paths),
            "untracked_files": len(untracked_paths),
            "untracked_contents_included": untracked_contents_included,
            "output_complete": output_complete,
        }
        return ToolResult(ok, output or "No changes.", metadata)


@dataclass
class ListFilesTool(BaseTool):
    workspace: Workspace
    max_results: int = 500
    name = "list_files"
    description = "List repository files for code navigation without invoking a shell."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "default": "*"},
        },
        "additionalProperties": False,
    }

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        del timeout_seconds
        target = self.workspace.resolve(arguments.get("path", "."), must_exist=True)
        glob = arguments.get("glob", "*")
        candidates = [target] if target.is_file() else target.rglob(glob)
        files = [
            self.workspace.relative(path)
            for path in candidates
            if path.is_file()
            and ".git" not in path.parts
            and not is_sensitive_path(str(path))
        ][: self.max_results]
        return ToolResult(True, "\n".join(files) or "No files.", {"files": len(files)})


@dataclass
class GitInspectTool(BaseTool):
    workspace: Workspace
    runtime: Runtime
    name = "git_inspect"
    description = (
        "Inspect Git status, diff, or recent log without modifying the repository."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["status", "diff", "log"]},
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        operation = arguments["operation"]
        command = {
            "status": "git status --short --branch",
            "diff": "git diff --stat; git diff",
            "log": "git log -n 20 --oneline --decorate",
        }[operation]
        result = self.runtime.run(
            command, self.workspace.root, min(timeout_seconds, 30)
        )
        output = result.stdout + (
            ("\nstderr:\n" + result.stderr) if result.stderr else ""
        )
        return ToolResult(result.exit_code == 0, output or "No output.", asdict(result))


def build_default_tools(
    workspace: Workspace, runtime: Runtime, *, read_only: bool = False
) -> ToolRegistry:
    tools: list[BaseTool] = [
        ReadFileTool(workspace, runtime),
        SearchFilesTool(workspace, runtime),
        ListFilesTool(workspace),
        GitDiffTool(workspace, runtime),
        GitInspectTool(workspace, runtime),
    ]
    if not read_only:
        tools[2:2] = [
            ApplyPatchTool(workspace, runtime),
            ShellTool(workspace, runtime),
            ValidateTool(workspace, runtime),
        ]
    return ToolRegistry(tools)
