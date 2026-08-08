from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

from forgeloop.effects import EffectDraft
from forgeloop.runtime import LocalRuntime, Runtime
from forgeloop.security import ShellSafetyPolicy, is_sensitive_path
from forgeloop.tools.base import BaseTool, ToolRegistry, ToolResult
from forgeloop.workspace import Workspace


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
class GitDiffTool(BaseTool):
    workspace: Workspace
    runtime: Runtime
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

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        path = arguments.get("path")
        if path:
            self.workspace.resolve(path)
        cached = " --cached" if arguments.get("cached", False) else ""
        quoted_path = f" -- '{path.replace(chr(39), chr(39) * 2)}'" if path else ""
        command = f"git status --short; git diff{cached}{quoted_path}"
        result = self.runtime.run(
            command, self.workspace.root, min(timeout_seconds, 30.0)
        )
        output = result.stdout
        if result.stderr:
            output += "\nstderr:\n" + result.stderr
        return ToolResult(
            result.exit_code == 0, output or "No changes.", asdict(result)
        )


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
        tools[2:2] = [ApplyPatchTool(workspace, runtime), ShellTool(workspace, runtime)]
    return ToolRegistry(tools)
