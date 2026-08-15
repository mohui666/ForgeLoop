from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from forgeloop.security import is_sensitive_path


def _truncate_command_output(value: str, limit: int) -> str:
    """Keep both command setup and terminal diagnostics when output is large."""

    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    marker = f"\n... <{omitted} chars omitted from middle> ...\n"
    available = limit - len(marker)
    if available <= 0:
        return value[:limit]
    head = available // 2
    tail = available - head
    return value[:head] + marker + value[-tail:]


@dataclass(frozen=True)
class CommandResult:
    command: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass(frozen=True)
class SearchResult:
    matches: tuple[str, ...] = ()
    error: str | None = None
    timed_out: bool = False


@dataclass(frozen=True)
class ShellEnvironment:
    """Shell semantics exposed by a Runtime to agent-facing tools."""

    platform: str
    executable: str
    syntax: str
    location: str
    network_access: bool | None = None
    guidance: str = ""

    @property
    def description(self) -> str:
        network = "network-disabled " if self.network_access is False else ""
        summary = (
            f"in the {network}{self.location} on {self.platform} using "
            f"{self.syntax} syntax ({self.executable})"
        )
        return f"{summary}. {self.guidance}" if self.guidance else summary


class Runtime(Protocol):
    """Execution and workspace I/O boundary used by tools and verifiers."""

    def start(self, workspace_root: Path) -> None: ...

    def close(self) -> None: ...

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult: ...

    def path_kind(self, path: Path) -> str: ...

    def read_bytes(self, path: Path) -> bytes: ...

    def write_bytes(self, path: Path, content: bytes) -> None: ...

    def search_text(
        self,
        pattern: str,
        target: Path,
        glob: str | None,
        max_results: int,
        timeout_seconds: float,
    ) -> SearchResult: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    @property
    def shell_environment(self) -> ShellEnvironment: ...


@dataclass
class LocalRuntime:
    max_output_chars: int = 40_000
    sanitize_environment: bool = True

    def start(self, workspace_root: Path) -> None:
        del workspace_root

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        return {"type": "local"}

    @property
    def shell_environment(self) -> ShellEnvironment:
        if os.name == "nt":
            return ShellEnvironment(
                platform="Windows",
                executable="PowerShell",
                syntax="PowerShell",
                location="local host",
                guidance="Use PowerShell commands and variables",
            )
        return ShellEnvironment(
            platform="Unix",
            executable="/bin/sh",
            syntax="POSIX shell",
            location="local host",
            guidance="Use POSIX shell commands and variables",
        )

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        if os.name == "nt":
            argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            argv = ["/bin/sh", "-lc", command]
        with (
            tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8", errors="replace"
            ) as stdout_file,
            tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8", errors="replace"
            ) as stderr_file,
        ):
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                errors="replace",
                env=self._environment(),
            )
            try:
                process.wait(timeout=timeout_seconds)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_process_tree(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
            if not timed_out:
                return CommandResult(
                    command=command,
                    cwd=str(cwd),
                    exit_code=process.returncode,
                    stdout=self._truncate(stdout),
                    stderr=self._truncate(stderr),
                    stdout_truncated=len(stdout) > self.max_output_chars,
                    stderr_truncated=len(stderr) > self.max_output_chars,
                )
            timeout_stderr = stderr + "\nCommand timed out."
            return CommandResult(
                command=command,
                cwd=str(cwd),
                exit_code=124,
                stdout=self._truncate(stdout),
                stderr=self._truncate(timeout_stderr),
                timed_out=True,
                stdout_truncated=len(stdout) > self.max_output_chars,
                stderr_truncated=len(timeout_stderr) > self.max_output_chars,
            )

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
        process.kill()

    @staticmethod
    def path_kind(path: Path) -> str:
        if path.is_file():
            return "file"
        if path.is_dir():
            return "directory"
        if path.exists():
            return "other"
        return "missing"

    @staticmethod
    def read_bytes(path: Path) -> bytes:
        return path.read_bytes()

    @staticmethod
    def write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def search_text(
        self,
        pattern: str,
        target: Path,
        glob: str | None,
        max_results: int,
        timeout_seconds: float,
    ) -> SearchResult:
        del timeout_seconds
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return SearchResult(error=f"Invalid regex: {exc}")
        root = target if target.is_dir() else target.parent
        candidates = [target] if target.is_file() else target.rglob(glob or "*")
        matches: list[str] = []
        for path in candidates:
            if (
                not path.is_file()
                or ".git" in path.parts
                or is_sensitive_path(str(path))
            ):
                continue
            if glob and not fnmatch.fnmatch(path.name, glob):
                continue
            try:
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1
                ):
                    if regex.search(line):
                        relative = path.relative_to(root).as_posix()
                        matches.append(f"{relative}:{line_number}:{line}")
                        if len(matches) >= max_results:
                            return SearchResult(tuple(matches))
            except (OSError, UnicodeError):
                continue
        return SearchResult(tuple(matches))

    def _truncate(self, value: str) -> str:
        return _truncate_command_output(value, self.max_output_chars)

    @staticmethod
    def _decode(value: str | bytes | None) -> str:
        if value is None:
            return ""
        return value.decode(errors="replace") if isinstance(value, bytes) else value

    def _environment(self) -> dict[str, str] | None:
        if not self.sanitize_environment:
            return None
        markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        return {
            name: value
            for name, value in os.environ.items()
            if not any(marker in name.upper() for marker in markers)
        }


_CONTAINER_SEARCH_SCRIPT = r"""
import fnmatch, json, pathlib, re, sys
root = pathlib.Path('/workspace')
target = root / sys.argv[1]
pattern, file_glob, limit = sys.argv[2], sys.argv[3] or None, int(sys.argv[4])
try:
    regex = re.compile(pattern)
except re.error as exc:
    print(json.dumps({'error': f'Invalid regex: {exc}', 'matches': []}))
    raise SystemExit(0)
candidates = [target] if target.is_file() else target.rglob(file_glob or '*')
matches = []
for path in candidates:
    if not path.is_file() or '.git' in path.parts:
        continue
    if file_glob and not fnmatch.fnmatch(path.name, file_glob):
        continue
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError):
        continue
    for line_number, line in enumerate(lines, 1):
        if regex.search(line):
            matches.append(f'{path.relative_to(root).as_posix()}:{line_number}:{line}')
            if len(matches) >= limit:
                print(json.dumps({'error': None, 'matches': matches}))
                raise SystemExit(0)
print(json.dumps({'error': None, 'matches': matches}))
""".strip()


@dataclass
class DockerRuntime:
    """One disposable Linux container bound to one eval workspace."""

    image: str = "forgeloop-eval:py312"
    dockerfile: Path | None = None
    build_context: Path | None = None
    max_output_chars: int = 40_000
    docker_executable: str | None = None
    container_name: str | None = field(default=None, init=False)
    workspace_root: Path | None = field(default=None, init=False, repr=False)

    @classmethod
    def available(cls) -> bool:
        try:
            docker = cls._find_docker()
            result = subprocess.run(
                [docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                timeout=10,
                check=False,
                env=cls._docker_env(docker),
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def start(self, workspace_root: Path) -> None:
        if self.container_name is not None:
            raise RuntimeError("DockerRuntime is already started")
        root = workspace_root.resolve()
        if not root.is_dir():
            raise RuntimeError(f"Docker workspace does not exist: {root}")
        docker = self.docker_executable or self._find_docker()
        self.docker_executable = docker
        self._ensure_image(docker)
        name = f"forgeloop-eval-{uuid.uuid4().hex[:12]}"
        mount = f"type=bind,source={root},target=/workspace"
        result = self._docker(
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--label",
            "forgeloop.eval=true",
            "--network",
            "none",
            "--mount",
            mount,
            "--workdir",
            "/workspace",
            self.image,
            "sleep",
            "infinity",
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to start Docker eval container: "
                + result.stderr.decode(errors="replace").strip()
            )
        self.container_name = name
        self.workspace_root = root

    def close(self) -> None:
        name = self.container_name
        self.container_name = None
        self.workspace_root = None
        if not name:
            return
        result = self._docker("rm", "--force", name, timeout=30)
        stderr = result.stderr.decode(errors="replace")
        if result.returncode != 0 and "No such container" not in stderr:
            raise RuntimeError(
                f"Failed to remove Docker eval container: {stderr.strip()}"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "type": "docker",
            "image": self.image,
            "container": self.container_name,
            "network": "none",
            "workspace": "/workspace",
        }

    @property
    def shell_environment(self) -> ShellEnvironment:
        return ShellEnvironment(
            platform="Linux",
            executable="/bin/sh",
            syntax="POSIX shell",
            location="Docker container",
            network_access=False,
            guidance=(
                "Use POSIX shell syntax only; Windows and PowerShell commands, "
                "utilities, and variables are unavailable"
            ),
        )

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        container_cwd = self._container_path(cwd)
        timeout_value = max(0.1, timeout_seconds)
        try:
            completed = self._docker(
                "exec",
                "--workdir",
                container_cwd,
                self._container(),
                "timeout",
                "--signal=KILL",
                f"{timeout_value}s",
                "/bin/sh",
                "-lc",
                command,
                timeout=timeout_value + 5,
            )
            stdout = completed.stdout.decode(errors="replace")
            stderr = completed.stderr.decode(errors="replace")
            return CommandResult(
                command=command,
                cwd=container_cwd,
                exit_code=completed.returncode,
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
                timed_out=completed.returncode == 124,
                stdout_truncated=len(stdout) > self.max_output_chars,
                stderr_truncated=len(stderr) > self.max_output_chars,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode(errors="replace")
            stderr = (exc.stderr or b"").decode(
                errors="replace"
            ) + "\nCommand timed out."
            return CommandResult(
                command=command,
                cwd=container_cwd,
                exit_code=124,
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
                timed_out=True,
                stdout_truncated=len(stdout) > self.max_output_chars,
                stderr_truncated=len(stderr) > self.max_output_chars,
            )

    def path_kind(self, path: Path) -> str:
        container_path = self._container_path(path)
        script = (
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "print('file' if p.is_file() else 'directory' if p.is_dir() "
            "else 'other' if p.exists() else 'missing')"
        )
        result = self._exec_python(script, container_path, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace").strip())
        return result.stdout.decode(errors="replace").strip()

    def read_bytes(self, path: Path) -> bytes:
        script = "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"
        result = self._exec_python(script, self._container_path(path), timeout=30)
        if result.returncode != 0:
            raise OSError(result.stderr.decode(errors="replace").strip())
        return result.stdout

    def write_bytes(self, path: Path, content: bytes) -> None:
        script = (
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())"
        )
        result = self._exec_python(
            script, self._container_path(path), timeout=30, input_bytes=content
        )
        if result.returncode != 0:
            raise OSError(result.stderr.decode(errors="replace").strip())

    def search_text(
        self,
        pattern: str,
        target: Path,
        glob: str | None,
        max_results: int,
        timeout_seconds: float,
    ) -> SearchResult:
        relative = self._relative(target).as_posix()
        try:
            result = self._exec_python(
                _CONTAINER_SEARCH_SCRIPT,
                relative,
                pattern,
                glob or "",
                str(max_results),
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return SearchResult(error="Search timed out", timed_out=True)
        if result.returncode != 0:
            return SearchResult(error=result.stderr.decode(errors="replace").strip())
        try:
            payload = json.loads(result.stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            return SearchResult(error=f"Invalid container search response: {exc}")
        return SearchResult(tuple(payload["matches"]), payload.get("error"))

    def _ensure_image(self, docker: str) -> None:
        inspect = self._docker("image", "inspect", self.image, timeout=30)
        if inspect.returncode == 0:
            return
        dockerfile = self.dockerfile or (
            Path(__file__).resolve().parent / "docker" / "eval.Dockerfile"
        )
        if not dockerfile.is_file():
            raise RuntimeError(
                f"Docker eval image is missing and Dockerfile was not found: {dockerfile}"
            )
        build_context = (self.build_context or dockerfile.parent).resolve()
        if not build_context.is_dir():
            raise RuntimeError(f"Docker build context was not found: {build_context}")
        result = self._docker(
            "build",
            "--file",
            str(dockerfile),
            "--tag",
            self.image,
            str(build_context),
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to build Docker eval image: "
                + result.stderr.decode(errors="replace").strip()
            )

    def _exec_python(
        self,
        script: str,
        *arguments: str,
        timeout: float,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        return self._docker(
            "exec",
            "--interactive" if input_bytes is not None else "--workdir",
            *([] if input_bytes is not None else ["/workspace"]),
            self._container(),
            "python",
            "-c",
            script,
            *arguments,
            timeout=timeout,
            input_bytes=input_bytes,
        )

    def _container(self) -> str:
        if not self.container_name:
            raise RuntimeError("DockerRuntime is not started")
        return self.container_name

    def _container_path(self, path: Path) -> str:
        relative = self._relative(path)
        return str(PurePosixPath("/workspace", *relative.parts))

    def _relative(self, path: Path) -> Path:
        if self.workspace_root is None:
            raise RuntimeError("DockerRuntime is not started")
        try:
            return path.resolve().relative_to(self.workspace_root)
        except ValueError as exc:
            raise RuntimeError(f"Path escapes Docker workspace: {path}") from exc

    def _docker(
        self,
        *arguments: str,
        timeout: float,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        docker = self.docker_executable or self._find_docker()
        return subprocess.run(
            [docker, *arguments],
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=self._docker_env(docker),
        )

    @staticmethod
    def _find_docker() -> str:
        found = shutil.which("docker")
        if found:
            return found
        if os.name == "nt":
            candidate = (
                Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
                / "Docker"
                / "Docker"
                / "resources"
                / "bin"
                / "docker.exe"
            )
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError("Docker CLI is not installed or not on PATH")

    @staticmethod
    def _docker_env(docker: str) -> dict[str, str]:
        env = os.environ.copy()
        docker_dir = str(Path(docker).resolve().parent)
        env["PATH"] = docker_dir + os.pathsep + env.get("PATH", "")
        return env

    def _truncate(self, value: str) -> str:
        return _truncate_command_output(value, self.max_output_chars)
