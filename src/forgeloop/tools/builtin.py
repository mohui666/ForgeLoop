from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import ClassVar

from forgeloop.runtime import LocalRuntime, Runtime
from forgeloop.security import ShellSafetyPolicy, is_sensitive_path
from forgeloop.tools.base import BaseTool, ToolRegistry, ToolResult
from forgeloop.workspace import Workspace


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
        if is_sensitive_path(arguments["path"]):
            return ToolResult(
                False, "Reading credential or VCS-internal files is blocked"
            )
        path = self.workspace.resolve(arguments["path"])
        kind = self.runtime.path_kind(path)
        if kind != "file":
            return ToolResult(False, f"Not a file: {arguments['path']}")
        start = int(arguments.get("start_line", 1))
        end = arguments.get("end_line")
        if start < 1 or (end is not None and int(end) < start):
            return ToolResult(False, "Invalid line range")
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
        if is_sensitive_path(arguments.get("path", ".")):
            return ToolResult(
                False, "Searching credential or VCS-internal files is blocked"
            )
        target = self.workspace.resolve(arguments.get("path", "."))
        if self.runtime.path_kind(target) == "missing":
            return ToolResult(
                False, f"Path does not exist: {arguments.get('path', '.')}"
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
            return ToolResult(False, result.error, {"matches": len(result.matches)})
        output = "\n".join(result.matches) or "No matches."
        return ToolResult(
            True,
            output[: self.max_output_chars],
            {"matches": len(result.matches), "timed_out": result.timed_out},
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
        if is_sensitive_path(arguments["path"]):
            return ToolResult(
                False, "Writing credential or VCS-internal files is blocked"
            )
        path = self.workspace.resolve(arguments["path"])
        old_text = arguments["old_text"]
        new_text = arguments["new_text"]
        replace_all = bool(arguments.get("replace_all", False))
        kind = self.runtime.path_kind(path)
        if kind != "missing":
            if kind != "file":
                return ToolResult(False, f"Not a file: {arguments['path']}")
            if old_text == "":
                return ToolResult(
                    False, "old_text cannot be empty when updating an existing file"
                )
            raw_content = self.runtime.read_bytes(path).decode("utf-8")
            uses_crlf = "\r\n" in raw_content
            content = raw_content.replace("\r\n", "\n")
            old_text = old_text.replace("\r\n", "\n")
            new_text = new_text.replace("\r\n", "\n")
            count = content.count(old_text)
            if count == 0:
                return ToolResult(
                    False, "old_text was not found; read the latest file and retry"
                )
            if count > 1 and not replace_all:
                return ToolResult(
                    False,
                    f"old_text matched {count} times; provide more context or set replace_all",
                )
            updated = content.replace(old_text, new_text, -1 if replace_all else 1)
            if uses_crlf:
                updated = updated.replace("\n", "\r\n")
            self.runtime.write_bytes(path, updated.encode("utf-8"))
            return ToolResult(
                True,
                f"Updated {self.workspace.relative(path)} ({count if replace_all else 1} replacement(s)).",
            )
        if old_text:
            return ToolResult(
                False, "Cannot replace text in a file that does not exist"
            )
        self.runtime.write_bytes(path, new_text.encode("utf-8"))
        return ToolResult(True, f"Created {self.workspace.relative(path)}.")


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
        rejection = self.safety.rejection(arguments["command"])
        if rejection:
            return ToolResult(False, rejection, {"blocked": True})
        cwd = self.workspace.resolve(arguments.get("cwd", "."))
        if self.runtime.path_kind(cwd) != "directory":
            return ToolResult(False, f"cwd is not a directory: {arguments.get('cwd')}")
        requested = float(arguments.get("timeout_seconds", 30))
        effective = max(0.1, min(requested, self.max_timeout_seconds, timeout_seconds))
        result = self.runtime.run(arguments["command"], cwd, effective)
        output = (
            f"exit_code: {result.exit_code}\n"
            f"timed_out: {str(result.timed_out).lower()}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        return ToolResult(result.exit_code == 0, output, asdict(result))


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
