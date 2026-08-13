from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar

from forgeloop import __version__
from forgeloop.agent import AgentLoop, RunMode
from forgeloop.budget import BudgetLimits
from forgeloop.controller import controller_for_policy
from forgeloop.delivery import GitPatchDelivery
from forgeloop.evals import EvalTaskResult, FailureCategory, aggregate_results
from forgeloop.guards import guard_semantics
from forgeloop.models import LiteLLMProvider, ProviderRetryPolicy
from forgeloop.policy import (
    PolicyIdentity,
    provider_policy_identity,
    resolve_policy_api_key,
)
from forgeloop.runtime import CommandResult, SearchResult, ShellEnvironment
from forgeloop.security import SecretRedactor, is_sensitive_path
from forgeloop.tools.base import BaseTool, ToolRegistry, ToolResult
from forgeloop.tools.builtin import (
    ApplyPatchTool,
    GitDiffTool,
    GitInspectTool,
    ReadFileTool,
    SearchFilesTool,
    ShellTool,
    ValidateTool,
)
from forgeloop.trajectory import TrajectoryStore
from forgeloop.verifier import VerifierResult
from forgeloop.workspace import GitSnapshot, WorkspaceError

SUBSET_SCHEMA_VERSION = "forgeloop.deepswe-subset.v1"
DEFAULT_SUBSET_PATH = Path(__file__).with_name("deepswe_assets") / "eval-v2-subset.json"
DEFAULT_CHECKOUT = Path(".forgeloop/external/deep-swe")
DEFAULT_JOBS_DIR = Path(".forgeloop/deepswe-jobs")
DEFAULT_REPORTS_DIR = Path(".forgeloop/eval-v2-runs")
REMOTE_ROOT = "/app"
DEEPSWE_MAX_MODEL_CALLS = 256
DEEPSWE_MAX_TOOL_CALLS = 1024
DEEPSWE_MAX_SECONDS = 5400.0
LEGACY_PIER_OUTPUT_LIMIT = 40_000


class DeepSWEError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactCollectionAudit:
    ok: bool
    status: str
    contract: str
    patch_path: str
    patch_bytes: int
    patch_sha256: str | None
    expected_base_sha: str
    delivery_base_sha: str | None
    delivery_head_sha: str | None
    delivery_branch: str | None
    delivery_patch_bytes: int
    delivery_patch_sha256: str | None
    manifest_status: str
    verifier_apply_bytes: int | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeepSWESubset:
    suite_id: str
    repository: str
    revision: str
    pier_repository: str
    pier_revision: str
    pier_version: str
    seed: int
    method: str
    population_sha256: str
    tasks: tuple[str, ...]
    common_resources: dict[str, Any]
    source_path: Path

    @classmethod
    def load(cls, path: Path = DEFAULT_SUBSET_PATH) -> "DeepSWESubset":
        source = path.expanduser().resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SUBSET_SCHEMA_VERSION:
            raise DeepSWEError(f"DeepSWE subset schema must be {SUBSET_SCHEMA_VERSION}")
        upstream = raw.get("upstream") or {}
        selection = raw.get("selection") or {}
        tasks = tuple(str(item) for item in raw.get("tasks") or ())
        if len(tasks) != 20 or len(set(tasks)) != len(tasks):
            raise DeepSWEError("DeepSWE Eval v2 subset must contain 20 unique tasks")
        return cls(
            suite_id=str(raw["suite_id"]),
            repository=str(upstream["repository"]),
            revision=str(upstream["revision"]),
            pier_repository=str(upstream["pier_repository"]),
            pier_revision=str(upstream["pier_revision"]),
            pier_version=str(upstream["pier_version"]),
            seed=int(selection["seed"]),
            method=str(selection["method"]),
            population_sha256=str(selection["population_sha256"]),
            tasks=tasks,
            common_resources=dict(raw.get("common_resources") or {}),
            source_path=source,
        )

    def validate_checkout(self, checkout: Path) -> dict[str, Any]:
        root = checkout.expanduser().resolve()
        if not (root / ".git").is_dir():
            raise DeepSWEError(f"DeepSWE checkout is missing or not Git: {root}")
        head = _git(root, "rev-parse", "HEAD").strip()
        if head != self.revision:
            raise DeepSWEError(
                f"DeepSWE revision mismatch: expected {self.revision}, got {head}"
            )
        missing = [task for task in self.tasks if not (root / "tasks" / task).is_dir()]
        if missing:
            raise DeepSWEError(
                "Pinned checkout is missing tasks: " + ", ".join(missing)
            )
        all_tasks = sorted(
            path.name for path in (root / "tasks").iterdir() if path.is_dir()
        )
        population_payload = ("\n".join(all_tasks) + "\n").encode()
        population_sha256 = hashlib.sha256(population_payload).hexdigest()
        if population_sha256 != self.population_sha256:
            raise DeepSWEError(
                "DeepSWE task population checksum mismatch: " + population_sha256
            )
        if select_task_ids(all_tasks, self.seed, len(self.tasks)) != self.tasks:
            raise DeepSWEError(
                "DeepSWE subset does not match its recorded selection method"
            )
        crlf_scripts = [
            task
            for task in self.tasks
            if b"\r\n" in (root / "tasks" / task / "tests" / "test.sh").read_bytes()
        ]
        if crlf_scripts:
            raise DeepSWEError(
                "DeepSWE verifier scripts have CRLF line endings; clone with "
                "`git -c core.autocrlf=false clone ...`. Affected tasks: "
                + ", ".join(crlf_scripts)
            )
        return {"checkout": str(root), "revision": head, "task_count": len(all_tasks)}


def check_requirements(
    checkout: Path = DEFAULT_CHECKOUT,
    subset_path: Path = DEFAULT_SUBSET_PATH,
) -> dict[str, Any]:
    subset = DeepSWESubset.load(subset_path)
    root = checkout.expanduser().resolve()
    checkout_info = subset.validate_checkout(root)
    docker = shutil.which("docker")
    if not docker:
        raise DeepSWEError("Docker CLI was not found")
    info = subprocess.run(
        [docker, "info", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    if info.returncode != 0:
        raise DeepSWEError("Docker Engine is unavailable: " + info.stderr.strip())
    try:
        pier_distribution = metadata.distribution("datacurve-pier")
    except metadata.PackageNotFoundError as exc:
        raise DeepSWEError(
            "Pier is not installed in this environment; run `uv sync --extra deepswe`"
        ) from exc
    installed_pier = pier_distribution.version
    if installed_pier != subset.pier_version:
        raise DeepSWEError(
            f"Pier version mismatch: expected {subset.pier_version}, got {installed_pier}"
        )
    direct_url = _distribution_direct_url(pier_distribution)
    vcs = direct_url.get("vcs_info") or {}
    installed_revision = str(vcs.get("commit_id") or "")
    installed_repository = str(direct_url.get("url") or "")
    if installed_revision != subset.pier_revision:
        raise DeepSWEError(
            "Pier revision mismatch: expected "
            f"{subset.pier_revision}, got {installed_revision or 'no Git provenance'}"
        )
    if _normalized_git_url(installed_repository) != _normalized_git_url(
        subset.pier_repository
    ):
        raise DeepSWEError(
            "Pier repository mismatch: expected "
            f"{subset.pier_repository}, got {installed_repository or 'unknown'}"
        )
    _validate_pier_collector_contract(root, subset.tasks)
    usage = shutil.disk_usage(root)
    return {
        **checkout_info,
        "subset": subset.suite_id,
        "subset_tasks": len(subset.tasks),
        "sampling_seed": subset.seed,
        "pier_version": installed_pier,
        "pier_repository": installed_repository,
        "pier_revision": installed_revision,
        "pier_artifact_contract": "verifier.collect -> /logs/artifacts/model.patch",
        "docker_server": json.loads(info.stdout).get("ServerVersion"),
        "disk_free_gb": round(usage.free / (1024**3), 2),
        "declared_task_resources": subset.common_resources,
    }


class PierWorkspace:
    """Workspace-shaped view whose content lives in Pier's official /app image."""

    def __init__(self, virtual_root: Path, runtime: "PierRuntime") -> None:
        virtual_root.mkdir(parents=True, exist_ok=True)
        self.root = virtual_root.resolve()
        self.runtime = runtime

    def resolve(
        self, relative_path: str | Path = ".", *, must_exist: bool = False
    ) -> Path:
        supplied = str(relative_path).replace("\\", "/")
        if supplied == REMOTE_ROOT:
            supplied = "."
        elif supplied.startswith(REMOTE_ROOT + "/"):
            supplied = supplied[len(REMOTE_ROOT) + 1 :]
        candidate = (self.root / supplied).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes workspace: {relative_path}") from exc
        if must_exist and self.runtime.path_kind(candidate) == "missing":
            raise WorkspaceError(f"Path does not exist: {relative_path}")
        return candidate

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix() or "."

    def git_snapshot(self, *, max_chars: int = 20_000) -> GitSnapshot:
        root = self.runtime.run("git rev-parse --show-toplevel", self.root, 10)
        if root.exit_code != 0:
            return GitSnapshot(False, error=root.stderr.strip() or None)
        head = self.runtime.run("git rev-parse HEAD", self.root, 10)
        branch = self.runtime.run("git symbolic-ref --short -q HEAD", self.root, 10)
        status = self.runtime.run(
            "git status --short --untracked-files=all", self.root, 10
        )
        status_text = status.stdout[:max_chars]
        if len(status.stdout) > max_chars:
            status_text += f"\n... <{len(status.stdout) - max_chars} chars omitted>"
        return GitSnapshot(
            True,
            repository_root=root.stdout.strip(),
            head=head.stdout.strip() if head.exit_code == 0 else None,
            branch=branch.stdout.strip() if branch.exit_code == 0 else None,
            status=status_text,
            error=status.stderr.strip() or None if status.exit_code != 0 else None,
        )

    def git_progress_fingerprint(
        self,
        *,
        base_head: str | None = None,
        max_untracked_bytes: int = 1_000_000,
    ) -> str:
        if base_head:
            command = (
                "{ git diff --name-only -z "
                f"{_quote(base_head)}; git ls-files --others --exclude-standard -z; "
                "} | grep -zvE '(^|/)(\\.pytest_cache|\\.mypy_cache|\\.ruff_cache|__pycache__)(/|$)|\\.(pyc|pyo)$|(^|/)\\.coverage$' | "
                "sort -zu | while IFS= read -r -d '' path; do "
                "printf '%s\\0' \"$path\"; "
                'if test -f "$path"; then '
                "test -x \"$path\" && printf '+x' || printf -- '-x'; "
                f'head -c {int(max_untracked_bytes)} "$path"; '
                "else printf '<deleted>'; fi; done | sha256sum"
            )
            result = self.runtime.run(command, self.root, 30)
            return hashlib.sha256(
                (result.stdout + result.stderr).encode("utf-8", errors="replace")
            ).hexdigest()
        command = (
            "git status --porcelain=v1 --untracked-files=all; "
            "git diff --binary HEAD; git diff --binary --cached; "
            "git ls-files --others --exclude-standard -z | "
            f"head -c {int(max_untracked_bytes)} | sha256sum"
        )
        result = self.runtime.run(command, self.root, 30)
        return hashlib.sha256(
            (result.stdout + result.stderr).encode("utf-8", errors="replace")
        ).hexdigest()


class PierRuntime:
    """Synchronous ForgeLoop Runtime bridged to Pier's async environment API."""

    def __init__(self, environment: Any, loop: asyncio.AbstractEventLoop) -> None:
        self.environment = environment
        self.loop = loop
        self.workspace_root: Path | None = None

    def start(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def close(self) -> None:
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "type": "pier_deepswe",
            "remote_root": REMOTE_ROOT,
            "environment": type(self.environment).__name__,
        }

    @property
    def shell_environment(self) -> ShellEnvironment:
        return ShellEnvironment(
            platform="Linux",
            executable="/bin/bash",
            syntax="POSIX shell",
            location="official DeepSWE task container at /app",
            network_access=False,
            guidance="Use POSIX commands; repository commands run inside /app",
        )

    def run(self, command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
        remote_cwd = self._remote_path(cwd)
        started = time.perf_counter()
        try:
            result = self._await(
                self.environment.exec(
                    command,
                    cwd=remote_cwd,
                    timeout_sec=max(1, math.ceil(timeout_seconds)),
                ),
                timeout_seconds + 10,
            )
            return CommandResult(
                command=command,
                cwd=remote_cwd,
                exit_code=int(result.return_code),
                stdout=self._truncate(result.stdout or ""),
                stderr=self._truncate(result.stderr or ""),
                timed_out=False,
            )
        except TimeoutError:
            return CommandResult(
                command=command,
                cwd=remote_cwd,
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {time.perf_counter() - started:.1f}s",
                timed_out=True,
            )

    def path_kind(self, path: Path) -> str:
        remote = self._remote_path(path)
        command = f"if [ -f {_quote(remote)} ]; then echo file; elif [ -d {_quote(remote)} ]; then echo directory; elif [ -e {_quote(remote)} ]; then echo other; else echo missing; fi"
        result = self.run(command, self.workspace_root or path.parent, 10)
        return result.stdout.strip() or "missing"

    def read_bytes(self, path: Path) -> bytes:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "payload"
            self._await(
                self.environment.download_file(self._remote_path(path), target), 60
            )
            return target.read_bytes()

    def write_bytes(self, path: Path, content: bytes) -> None:
        remote = self._remote_path(path)
        parent = remote.rpartition("/")[0] or "/"
        mkdir = self.run(
            f"mkdir -p {_quote(parent)}", self.workspace_root or path.parent, 30
        )
        if mkdir.exit_code != 0:
            raise DeepSWEError(mkdir.stderr or "Could not create remote directory")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "payload"
            source.write_bytes(content)
            self._await(self.environment.upload_file(source, remote), 120)

    def search_text(
        self,
        pattern: str,
        target: Path,
        glob: str | None,
        max_results: int,
        timeout_seconds: float,
    ) -> SearchResult:
        payload = base64.b64encode(
            json.dumps(
                {
                    "root": REMOTE_ROOT,
                    "target": self._remote_path(target),
                    "pattern": pattern,
                    "glob": glob,
                    "limit": max_results,
                }
            ).encode()
        ).decode()
        script = (
            "import base64,fnmatch,json,pathlib,re,sys;"
            "a=json.loads(base64.b64decode(sys.argv[1]));"
            "r=pathlib.Path(a['root']);t=pathlib.Path(a['target']);"
            "rx=re.compile(a['pattern']);"
            "cs=[t] if t.is_file() else t.rglob(a['glob'] or '*');out=[];"
            "[(out.append(f'{p.relative_to(r)}:{n}:{line}')) "
            "for p in cs if p.is_file() and '.git' not in p.parts "
            "for n,line in enumerate(p.read_text(encoding='utf-8',errors='ignore').splitlines(),1) "
            "if rx.search(line) and len(out)<a['limit']];print(json.dumps(out))"
        )
        result = self.run(
            f"python3 -c {_quote(script)} {_quote(payload)}",
            self.workspace_root or target.parent,
            timeout_seconds,
        )
        if result.exit_code != 0:
            return SearchResult(
                error=result.stderr or result.stdout, timed_out=result.timed_out
            )
        try:
            return SearchResult(tuple(json.loads(result.stdout)), timed_out=False)
        except json.JSONDecodeError as exc:
            return SearchResult(error=f"Remote search returned invalid JSON: {exc}")

    def list_files(self, target: Path, glob: str, max_results: int) -> tuple[str, ...]:
        payload = base64.b64encode(
            json.dumps(
                {
                    "root": REMOTE_ROOT,
                    "target": self._remote_path(target),
                    "glob": glob,
                    "limit": max_results,
                }
            ).encode()
        ).decode()
        script = (
            "import base64,json,pathlib,sys;"
            "a=json.loads(base64.b64decode(sys.argv[1]));r=pathlib.Path(a['root']);"
            "t=pathlib.Path(a['target']);cs=[t] if t.is_file() else t.rglob(a['glob']);"
            "print(json.dumps([str(p.relative_to(r)) for p in cs if p.is_file() and '.git' not in p.parts][:a['limit']]))"
        )
        result = self.run(
            f"python3 -c {_quote(script)} {_quote(payload)}",
            self.workspace_root or target.parent,
            60,
        )
        if result.exit_code != 0:
            raise DeepSWEError(result.stderr or result.stdout)
        return tuple(json.loads(result.stdout))

    def _remote_path(self, path: Path) -> str:
        if self.workspace_root is None:
            raise DeepSWEError("PierRuntime has not been started")
        try:
            relative = path.resolve().relative_to(self.workspace_root).as_posix()
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes workspace: {path}") from exc
        return REMOTE_ROOT if relative == "." else f"{REMOTE_ROOT}/{relative}"

    def _await(self, awaitable: Any, timeout: float) -> Any:
        return asyncio.run_coroutine_threadsafe(awaitable, self.loop).result(timeout)

    @staticmethod
    def _truncate(value: str, limit: int = 40_000) -> str:
        if len(value) <= limit:
            return value
        return value[:limit] + f"\n... <{len(value) - limit} chars omitted>"


@dataclass
class PierListFilesTool(BaseTool):
    workspace: PierWorkspace
    runtime: PierRuntime
    # DeepSWE repositories can contain thousands of files. Keep one navigation
    # result comfortably inside the frozen 8192-token policy context.
    max_results: int = 100
    name = "list_files"
    description = "List repository files for code navigation without invoking a shell."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "default": "*"},
        },
        "additionalProperties": False,
    }

    def execute(self, arguments: dict, *, timeout_seconds: float) -> ToolResult:
        del timeout_seconds
        target_name = str(arguments.get("path", "."))
        if is_sensitive_path(target_name):
            return ToolResult(
                False, "Listing credential or VCS-internal paths is blocked"
            )
        target = self.workspace.resolve(target_name, must_exist=True)
        files = self.runtime.list_files(
            target, str(arguments.get("glob", "*")), self.max_results
        )
        return ToolResult(True, "\n".join(files) or "No files.", {"files": len(files)})


def build_pier_tools(workspace: PierWorkspace, runtime: PierRuntime) -> ToolRegistry:
    return ToolRegistry(
        [
            ReadFileTool(workspace, runtime),
            SearchFilesTool(workspace, runtime),
            ApplyPatchTool(workspace, runtime),
            ShellTool(workspace, runtime),
            ValidateTool(workspace, runtime),
            PierListFilesTool(workspace, runtime),
            GitDiffTool(workspace, runtime),
            GitInspectTool(workspace, runtime),
        ]
    )


try:
    from pier.agents.base import BaseAgent as _PierBaseAgent
except ImportError:  # Keep ForgeLoop core and CLI help usable without the extra.

    class _PierBaseAgent:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise DeepSWEError(
                "DeepSWE support requires Python 3.12 and `uv sync --extra deepswe`"
            )


class ForgeLoopPierAgent(_PierBaseAgent):
    """Run the unchanged ForgeLoop AgentLoop against one official Pier task."""

    def __init__(
        self,
        *args: Any,
        policy_manifest: str = "qwen3.5-4b-local",
        max_steps: int | str = DEEPSWE_MAX_MODEL_CALLS,
        max_model_calls: int | str = DEEPSWE_MAX_MODEL_CALLS,
        max_tool_calls: int | str = DEEPSWE_MAX_TOOL_CALLS,
        max_seconds: float | str = DEEPSWE_MAX_SECONDS,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.policy_manifest = policy_manifest
        self.limits = BudgetLimits(
            max_steps=int(max_steps),
            max_model_calls=int(max_model_calls),
            max_tool_calls=int(max_tool_calls),
            max_seconds=float(max_seconds),
        )

    @staticmethod
    def name() -> str:
        return "forgeloop"

    def version(self) -> str:
        return __version__

    async def setup(self, environment: Any) -> None:
        check = await environment.exec(
            "test -d /app && command -v git && command -v python3",
            cwd=REMOTE_ROOT,
            timeout_sec=30,
        )
        if check.return_code != 0:
            raise DeepSWEError(
                check.stderr or check.stdout or "Invalid DeepSWE task image"
            )
        await environment.exec(
            "git config user.name 'ForgeLoop Eval' && git config user.email 'eval@forgeloop.local'",
            cwd=REMOTE_ROOT,
            timeout_sec=30,
        )

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        loop = asyncio.get_running_loop()
        result, trajectory = await asyncio.to_thread(
            self._run_sync, instruction, environment, loop
        )
        usage = result.budget["usage"]
        recorded = _trajectory_usage(trajectory)
        context.n_input_tokens = (
            usage["input_tokens"]
            if usage["input_tokens"] is not None
            else recorded["input_tokens"]
        )
        context.n_cache_tokens = (
            usage["cached_tokens"]
            if usage["cached_tokens"] is not None
            else recorded["cached_tokens"]
        )
        context.n_output_tokens = (
            usage["output_tokens"]
            if usage["output_tokens"] is not None
            else recorded["output_tokens"]
        )
        context.cost_usd = (
            usage["cost_usd"] if usage["cost_usd"] is not None else recorded["cost_usd"]
        )
        context.n_agent_steps = usage["steps"]
        context.metadata = {
            "forgeloop": {
                "terminal_state": result.status.value,
                "stop_reason": result.stop_reason,
                "summary": result.summary,
                "evidence": result.evidence,
                "trajectory_file": trajectory.name,
                "model_calls": usage["model_calls"],
                "tool_calls": usage["tool_calls"],
                "execution_budget": result.budget["limits"],
                "budget_semantics": "forgeloop.execution-budget.v2",
                "guard_semantics": guard_semantics(
                    max_repeated_tool_calls=self.limits.max_repeated_tool_calls,
                    max_repeated_errors=self.limits.max_repeated_errors,
                    max_no_progress_steps=self.limits.max_no_progress_steps,
                ),
                "reasoning_tokens": usage["reasoning_tokens"],
                "cache_miss_tokens": usage.get("cache_miss_tokens"),
                "cached_input_ratio": usage["cached_input_ratio"],
                "warm_cache_reusable_tokens": usage.get(
                    "warm_cache_reusable_tokens", 0
                ),
                "warm_cache_reused_tokens": usage.get("warm_cache_reused_tokens", 0),
                "warm_cache_missed_tokens": usage.get("warm_cache_missed_tokens", 0),
                "warm_cache_hit_ratio": usage.get("warm_cache_hit_ratio"),
                "warm_cache_measured_calls": usage.get(
                    "warm_cache_measured_calls", 0
                ),
                "warm_cache_significant_miss_calls": usage.get(
                    "warm_cache_significant_miss_calls", 0
                ),
                "warm_cache_reset_calls": usage.get("warm_cache_reset_calls", 0),
                "cost_sources": usage["cost_sources"],
                "usage_complete": usage["usage_complete"],
                "usage_records": usage["usage_records"],
                "unavailable_model_calls": usage["unavailable_model_calls"],
                "delivery": result.delivery,
                "provider_reliability": result.provider_reliability,
            }
        }

    def _run_sync(
        self,
        instruction: str,
        environment: Any,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[Any, Path]:
        runtime = PierRuntime(environment, loop)
        virtual_root = self.logs_dir / "virtual-workspace"
        workspace = PierWorkspace(virtual_root, runtime)
        runtime.start(workspace.root)
        trajectory = TrajectoryStore(
            self.logs_dir / "forgeloop-trajectories",
            run_id=f"deepswe-{int(time.time())}",
            redactor=SecretRedactor.from_environment(),
        )
        policy = PolicyIdentity.load(self.policy_manifest)
        api_base = str(policy.serving_config.get("api_base") or "") or None
        provider = LiteLLMProvider(
            model=policy.litellm_model,
            api_base=api_base,
            api_key=resolve_policy_api_key(policy),
            thinking_level=str(policy.serving_config.get("thinking_level") or "auto"),
            policy=policy,
        )
        agent = AgentLoop(
            provider=provider,
            tools=build_pier_tools(workspace, runtime),
            workspace=workspace,
            trajectory=trajectory,
            limits=self.limits,
            controller=controller_for_policy(policy),
            delivery=GitPatchDelivery(runtime),
        )
        trajectory.append("eval_runtime_started", runtime.metadata)
        result = agent.run(
            RunMode.TASK,
            instruction,
            instructions=(
                "This is an official DeepSWE/Pier task. All tools operate on the "
                "official repository at /app inside its isolated container. Follow "
                "the task's branch/commit requirement by finishing only after relevant "
                "validation and final diff review. ForgeLoop's delivery layer creates "
                "the delivery branch and commit from the real worktree after finish; "
                "do not spend model calls committing manually."
            ),
        )
        trajectory.append("eval_runtime_stopped", runtime.metadata)
        return result, trajectory.path


def pier_command(
    checkout: Path,
    jobs_dir: Path,
    job_name: str,
    tasks: tuple[str, ...],
    policy_manifest: str,
    limits: BudgetLimits | None = None,
) -> list[str]:
    resolved_limits = limits or deepswe_budget_limits()
    pier = Path(sys_executable_dir()) / ("pier.exe" if os.name == "nt" else "pier")
    executable = str(pier if pier.is_file() else shutil.which("pier") or "pier")
    command = [
        executable,
        "run",
        "--path",
        str((checkout / "tasks").resolve()),
        "--agent-import-path",
        "forgeloop.deepswe:ForgeLoopPierAgent",
        "--model",
        PolicyIdentity.load(policy_manifest).litellm_model,
        "--agent-kwarg",
        f"policy_manifest={policy_manifest}",
        "--agent-kwarg",
        f"max_steps={resolved_limits.max_steps}",
        "--agent-kwarg",
        f"max_model_calls={resolved_limits.max_model_calls}",
        "--agent-kwarg",
        f"max_tool_calls={resolved_limits.max_tool_calls}",
        "--agent-kwarg",
        f"max_seconds={resolved_limits.max_seconds:g}",
        "--env",
        "docker",
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--jobs-dir",
        str(jobs_dir.resolve()),
        "--job-name",
        job_name,
    ]
    for task in tasks:
        command.extend(["--include-task-name", task])
    return command


def deepswe_budget_limits(
    *,
    max_model_calls: int = DEEPSWE_MAX_MODEL_CALLS,
    max_tool_calls: int = DEEPSWE_MAX_TOOL_CALLS,
    max_seconds: float = DEEPSWE_MAX_SECONDS,
) -> BudgetLimits:
    """Build the configurable long-horizon DeepSWE execution window."""
    return BudgetLimits(
        max_steps=max_model_calls,
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        max_seconds=max_seconds,
    )


def select_task_ids(
    population: list[str] | tuple[str, ...], seed: int, size: int
) -> tuple[str, ...]:
    ordered = sorted(str(item) for item in population)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered[:size])


def run_deepswe(
    *,
    checkout: Path = DEFAULT_CHECKOUT,
    jobs_dir: Path = DEFAULT_JOBS_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    subset_path: Path = DEFAULT_SUBSET_PATH,
    task: str | None = None,
    policy_manifest: str = "qwen3.5-4b-local",
    job_name: str | None = None,
    max_model_calls: int = DEEPSWE_MAX_MODEL_CALLS,
    max_tool_calls: int = DEEPSWE_MAX_TOOL_CALLS,
    max_seconds: float = DEEPSWE_MAX_SECONDS,
) -> tuple[subprocess.CompletedProcess[str], Path | None]:
    subset = DeepSWESubset.load(subset_path)
    check_requirements(checkout, subset_path)
    if task and task not in subset.tasks:
        raise DeepSWEError(f"Task is not in frozen Eval v2 subset: {task}")
    selected = (task,) if task else subset.tasks
    resolved_job = job_name or (
        f"{subset.suite_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    limits = deepswe_budget_limits(
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        max_seconds=max_seconds,
    )
    jobs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source_root = str(Path(__file__).parents[1].resolve())
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        pier_command(
            checkout, jobs_dir, resolved_job, selected, policy_manifest, limits
        ),
        cwd=Path.cwd(),
        env=env,
        text=True,
        errors="replace",
        check=False,
    )
    job_dir = jobs_dir.resolve() / resolved_job
    report_dir = import_pier_results(
        job_dir, reports_dir, subset, policy_manifest, execution_limits=limits
    )
    return completed, report_dir


def import_pier_results(
    job_dir: Path,
    reports_dir: Path,
    subset: DeepSWESubset,
    policy_manifest: str,
    execution_limits: BudgetLimits | None = None,
) -> Path | None:
    result_paths = sorted(
        path for path in job_dir.rglob("result.json") if path.parent != job_dir
    )
    if not result_paths:
        return None
    run_id = job_dir.name
    run_dir = reports_dir.resolve() / run_id
    trajectories = run_dir / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)
    policy = PolicyIdentity.load(policy_manifest)
    results: list[EvalTaskResult] = []
    artifact_audits: list[dict[str, Any]] = []
    for path in result_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(raw["task_name"]).split("/", 1)[-1]
        agent_result = raw.get("agent_result") or {}
        forge = (agent_result.get("metadata") or {}).get("forgeloop") or {}
        verifier_raw = raw.get("verifier_result") or {}
        rewards = verifier_raw.get("rewards") or {}
        verifier_passed = float(rewards.get("reward", 0)) == 1.0
        stdout_path = path.parent / "verifier" / "test-stdout.txt"
        stderr_path = path.parent / "verifier" / "test-stderr.txt"
        stdout = _read_optional(stdout_path)
        stderr = _read_optional(stderr_path)
        duration = _duration(raw.get("verifier"))
        verifier = VerifierResult(
            command="DeepSWE official Pier verifier",
            passed=verifier_passed,
            exit_code=0 if verifier_passed else 1,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )
        trajectory_source = next(
            (path.parent / "agent" / "forgeloop-trajectories").glob("*.jsonl"),
            None,
        )
        trajectory_path: str | None = None
        expected_base_sha = _task_base_sha(raw)
        artifact_audit = _audit_artifact_collection(
            path.parent,
            expected_base_sha=expected_base_sha,
            trajectory_path=trajectory_source,
            verifier_stdout=stdout,
            verifier_rewards=rewards,
        )
        artifact_audits.append({"task_id": task_id, **artifact_audit.to_dict()})
        passed = verifier_passed and artifact_audit.ok
        if trajectory_source:
            target = trajectories / f"{task_id}-{trajectory_source.name}"
            shutil.copy2(trajectory_source, target)
            _append_external_events(target, verifier, path.parent, artifact_audit)
            trajectory_path = str(target)
        exception = raw.get("exception_info") or {}
        failure_category = (
            FailureCategory.NONE.value
            if passed
            else FailureCategory.ENVIRONMENT.value
            if exception or not artifact_audit.ok
            else FailureCategory.MODEL.value
        )
        usage_incomplete = forge.get("usage_complete") is False
        input_tokens = agent_result.get("n_input_tokens")
        output_tokens = agent_result.get("n_output_tokens")
        total_tokens = (
            input_tokens + output_tokens
            if isinstance(input_tokens, int) and isinstance(output_tokens, int)
            else None
        )
        patch_path = path.parent / "artifacts" / "model.patch"
        final_diff = _read_optional(patch_path)
        terminal_state = str(forge.get("terminal_state") or "infrastructure_failed")
        stop_reason = str(forge.get("stop_reason") or "pier_trial_error")
        failure_detail = str(exception.get("exception_message")) if exception else None
        if not artifact_audit.ok:
            terminal_state = "infrastructure_failed"
            stop_reason = f"artifact_collection_{artifact_audit.status}"
            failure_detail = artifact_audit.detail
        results.append(
            EvalTaskResult(
                task_id=task_id,
                description="Official DeepSWE task; see pinned upstream instruction.md",
                model=policy.litellm_model,
                provider=policy.litellm_model.partition("/")[0],
                success=passed,
                terminal_state=terminal_state,
                stop_reason=stop_reason,
                verifier=verifier,
                failure_category=failure_category,
                failure_detail=failure_detail,
                expected_base_sha=expected_base_sha,
                actual_base_sha=artifact_audit.delivery_base_sha,
                initial_dirty=False,
                model_calls=int(forge.get("model_calls") or 0),
                tool_calls=int(forge.get("tool_calls") or 0),
                steps=int(agent_result.get("n_agent_steps") or 0),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=(
                    agent_result.get("n_cache_tokens")
                ),
                reasoning_tokens=forge.get("reasoning_tokens"),
                total_cost_usd=(
                    agent_result.get("cost_usd")
                ),
                cost_sources=tuple(forge.get("cost_sources") or ["unknown"]),
                wall_time_seconds=_duration_between(
                    raw.get("started_at"), raw.get("finished_at")
                ),
                final_diff=final_diff,
                final_status=(
                    "patch_collected_verified"
                    if artifact_audit.ok
                    else artifact_audit.status
                ),
                trajectory_path=trajectory_path,
                difficulty="hard",
                policy_identity=provider_policy_identity(
                    LiteLLMProvider(model=policy.litellm_model, policy=policy)
                ),
                provider_reliability=dict(forge.get("provider_reliability") or {}),
                cache_miss_tokens=forge.get("cache_miss_tokens"),
                cached_input_ratio=forge.get("cached_input_ratio"),
                usage_complete=not usage_incomplete,
                unavailable_model_calls=int(
                    forge.get("unavailable_model_calls") or 0
                ),
                warm_cache_reusable_tokens=int(
                    forge.get("warm_cache_reusable_tokens") or 0
                ),
                warm_cache_reused_tokens=int(
                    forge.get("warm_cache_reused_tokens") or 0
                ),
                warm_cache_missed_tokens=int(
                    forge.get("warm_cache_missed_tokens") or 0
                ),
                warm_cache_hit_ratio=forge.get("warm_cache_hit_ratio"),
                warm_cache_measured_calls=int(
                    forge.get("warm_cache_measured_calls") or 0
                ),
                warm_cache_significant_miss_calls=int(
                    forge.get("warm_cache_significant_miss_calls") or 0
                ),
                warm_cache_reset_calls=int(
                    forge.get("warm_cache_reset_calls") or 0
                ),
            )
        )
    tasks_path = run_dir / "tasks.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in results
        ),
        encoding="utf-8",
    )
    summary = aggregate_results(
        subset.suite_id,
        run_id,
        policy.litellm_model,
        results,
        planned_tasks=len(results),
        planned_repeats=1,
        policy_identity=policy.to_dict(),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    resolved_limits = execution_limits or deepswe_budget_limits()
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "suite_id": subset.suite_id,
                "deep_swe_repository": subset.repository,
                "deep_swe_revision": subset.revision,
                "pier_repository": subset.pier_repository,
                "pier_revision": subset.pier_revision,
                "pier_version": subset.pier_version,
                "sampling_seed": subset.seed,
                "selection_method": subset.method,
                "pier_job_dir": str(job_dir.resolve()),
                "artifact_collection": {
                    "schema_version": "forgeloop.deepswe-artifact-audit.v1",
                    "fail_closed": True,
                    "tasks": artifact_audits,
                },
                "execution_budget": {
                    "schema_version": "forgeloop.execution-budget.v2",
                    "cumulative_tokens": "telemetry_only",
                    "limits": asdict(resolved_limits),
                },
                "provider_reliability": ProviderRetryPolicy.from_config(
                    policy.serving_config.get("provider_reliability")
                ).to_dict(),
                "guard_semantics": guard_semantics(
                    max_repeated_tool_calls=resolved_limits.max_repeated_tool_calls,
                    max_repeated_errors=resolved_limits.max_repeated_errors,
                    max_no_progress_steps=resolved_limits.max_no_progress_steps,
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def sys_executable_dir() -> str:
    import sys

    return str(Path(sys.executable).resolve().parent)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise DeepSWEError(result.stderr.strip() or "Git command failed")
    return result.stdout


def _distribution_direct_url(distribution: metadata.Distribution) -> dict[str, Any]:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _normalized_git_url(value: str) -> str:
    return value.rstrip("/").removesuffix(".git").lower()


def _validate_pier_collector_contract(root: Path, tasks: tuple[str, ...]) -> None:
    """Prove the installed Pier parses the contract used by pinned DeepSWE."""

    try:
        from pier.models.task.config import TaskConfig
    except (ImportError, AttributeError) as exc:
        raise DeepSWEError(
            "Installed Pier does not expose the DeepSWE verifier.collect schema"
        ) from exc

    incompatible: list[str] = []
    for task in tasks:
        task_path = root / "tasks" / task / "task.toml"
        config = TaskConfig.model_validate_toml(task_path.read_text(encoding="utf-8"))
        artifact_sources = [
            item if isinstance(item, str) else item.source for item in config.artifacts
        ]
        base_sha = str(config.metadata.get("base_commit_hash") or "")
        collect_commands = [hook.command for hook in config.verifier.collect]
        expected_range = f"git diff --binary {base_sha} HEAD"
        if (
            "/logs/artifacts/model.patch" not in artifact_sources
            or not config.verifier.collect
            or not base_sha
            or not any(
                expected_range in command and "> /logs/artifacts/model.patch" in command
                for command in collect_commands
            )
        ):
            incompatible.append(task)
    if incompatible:
        raise DeepSWEError(
            "Installed Pier did not parse the pinned DeepSWE artifact contract for: "
            + ", ".join(incompatible)
        )


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _task_base_sha(trial: dict[str, Any]) -> str:
    task_id = trial.get("task_id") or {}
    source = Path(str(task_id.get("path") or "")) / "task.toml"
    text = _read_optional(source)
    match = re.search(r'^base_commit_hash\s*=\s*"([0-9a-f]{40})"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _audit_artifact_collection(
    trial_dir: Path,
    *,
    expected_base_sha: str,
    trajectory_path: Path | None,
    verifier_stdout: str,
    verifier_rewards: dict[str, Any],
) -> ArtifactCollectionAudit:
    patch_path = trial_dir / "artifacts" / "model.patch"
    try:
        patch = patch_path.read_bytes()
    except OSError:
        patch = b""
    patch_bytes = len(patch)
    patch_sha256 = hashlib.sha256(patch).hexdigest() if patch else None
    delivery = _delivery_event(trajectory_path)
    manifest_status = _artifact_manifest_status(
        trial_dir / "artifacts" / "manifest.json"
    )
    apply_match = re.search(r"model\.patch applied \((\d+) bytes\)", verifier_stdout)
    apply_bytes = int(apply_match.group(1)) if apply_match else None

    status = "ok"
    detail = "Collector patch matches ForgeLoop delivery and was applied by verifier."
    if expected_base_sha == "unknown":
        status = "expected_base_missing"
        detail = "Task metadata did not expose base_commit_hash."
    elif not delivery:
        status = "delivery_provenance_missing"
        detail = "ForgeLoop trajectory has no patch_delivery event."
    elif not delivery.get("ok") or not delivery.get("has_patch"):
        status = "delivery_failed"
        detail = str(delivery.get("detail") or "Delivery did not produce a patch.")
    elif delivery.get("base_sha") != expected_base_sha:
        status = "base_mismatch"
        detail = (
            f"Delivery base {delivery.get('base_sha')} does not match task base "
            f"{expected_base_sha}."
        )
    elif patch_bytes == 0:
        status = "patch_missing_or_empty"
        detail = "Pier did not collect a non-empty artifacts/model.patch."
    elif manifest_status != "ok":
        status = "manifest_collection_failed"
        detail = f"Pier artifact manifest status for model.patch is {manifest_status}."
    elif not _patch_matches_delivery(patch, delivery):
        if _patch_matches_legacy_pier_truncation(patch, delivery):
            detail = (
                "Collector patch exactly reproduces the legacy PierRuntime "
                "40,000-character truncated delivery identity and was applied by "
                "verifier."
            )
        else:
            status = "delivery_patch_mismatch"
            detail = (
                "Collected patch does not match delivery patch size/hash provenance."
            )
    elif int(verifier_rewards.get("apply_failed") or 0) == 1 or (
        "submitted model.patch failed to apply" in verifier_stdout
    ):
        status = "patch_apply_failed"
        detail = "The official verifier rejected model.patch as unapplyable."
    elif apply_bytes is None:
        status = "verifier_apply_not_observed"
        detail = "Official verifier output does not confirm model.patch application."
    elif apply_bytes != patch_bytes:
        status = "verifier_apply_size_mismatch"
        detail = (
            f"Verifier reported {apply_bytes} applied bytes, collector stored "
            f"{patch_bytes}."
        )

    return ArtifactCollectionAudit(
        ok=status == "ok",
        status=status,
        contract="DeepSWE v1.1 verifier.collect base-to-HEAD patch",
        patch_path=str(patch_path.resolve()),
        patch_bytes=patch_bytes,
        patch_sha256=patch_sha256,
        expected_base_sha=expected_base_sha,
        delivery_base_sha=(str(delivery.get("base_sha")) if delivery else None),
        delivery_head_sha=(str(delivery.get("head_sha")) if delivery else None),
        delivery_branch=(str(delivery.get("branch")) if delivery else None),
        delivery_patch_bytes=int(delivery.get("patch_bytes") or 0) if delivery else 0,
        delivery_patch_sha256=(
            str(delivery.get("patch_sha256"))
            if delivery and delivery.get("patch_sha256")
            else None
        ),
        manifest_status=manifest_status,
        verifier_apply_bytes=apply_bytes,
        detail=detail,
    )


def _delivery_event(trajectory_path: Path | None) -> dict[str, Any]:
    if trajectory_path is None or not trajectory_path.is_file():
        return {}
    delivery: dict[str, Any] = {}
    for line in trajectory_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "patch_delivery":
            payload = event.get("payload")
            if isinstance(payload, dict):
                delivery = payload
    return delivery


def _artifact_manifest_status(path: Path) -> str:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "missing"
    except (OSError, json.JSONDecodeError):
        return "invalid"
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = raw.get("entries", [])
    else:
        return "invalid"
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("source", "")).rstrip("/").endswith("/model.patch"):
            return str(entry.get("status") or "unknown")
    return "model.patch_not_declared"


def _patch_matches_delivery(patch: bytes, delivery: dict[str, Any]) -> bool:
    normalized = patch.rstrip()
    expected_bytes = int(delivery.get("patch_bytes") or 0)
    expected_hash = delivery.get("patch_sha256")
    if expected_hash:
        return (
            len(normalized) == expected_bytes
            and hashlib.sha256(normalized).hexdigest() == expected_hash
        )
    return len(normalized) == expected_bytes


def _patch_matches_legacy_pier_truncation(
    patch: bytes, delivery: dict[str, Any]
) -> bool:
    """Recognize only the exact identity produced by the old Pier stdout cap."""

    try:
        text = patch.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if len(text) <= LEGACY_PIER_OUTPUT_LIMIT:
        return False
    omitted = len(text) - LEGACY_PIER_OUTPUT_LIMIT
    truncated = (
        text[:LEGACY_PIER_OUTPUT_LIMIT] + f"\n... <{omitted} chars omitted>"
    ).strip()
    encoded = truncated.encode("utf-8")
    expected_hash = delivery.get("patch_sha256")
    if expected_hash:
        return (
            len(encoded) == int(delivery.get("patch_bytes") or 0)
            and hashlib.sha256(encoded).hexdigest() == expected_hash
        )
    return len(encoded) == int(delivery.get("patch_bytes") or 0)


def _duration(timing: dict[str, Any] | None) -> float:
    if not timing:
        return 0.0
    return _duration_between(timing.get("started_at"), timing.get("finished_at"))


def _duration_between(start: str | None, end: str | None) -> float:
    if not start or not end:
        return 0.0
    return round(
        (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 3
    )


def _append_external_events(
    trajectory_path: Path,
    verifier: VerifierResult,
    trial_dir: Path,
    artifact_audit: ArtifactCollectionAudit,
) -> None:
    lines = trajectory_path.read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1]) if lines else {"sequence": -1, "run_id": "unknown"}
    events = (
        ("eval_artifact_collection", artifact_audit.to_dict()),
        ("eval_verifier", asdict(verifier)),
        (
            "eval_final_diff",
            {
                "diff": _read_optional(trial_dir / "artifacts" / "model.patch"),
                "status": (
                    "patch_collected_verified"
                    if artifact_audit.ok
                    else artifact_audit.status
                ),
            },
        ),
    )
    with trajectory_path.open("a", encoding="utf-8", newline="\n") as handle:
        for offset, (event_type, payload) in enumerate(events, 1):
            handle.write(
                json.dumps(
                    {
                        "schema_version": "forgeloop.trajectory.v2",
                        "run_id": last.get("run_id"),
                        "sequence": int(last.get("sequence", -1)) + offset,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": event_type,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _trajectory_usage(path: Path) -> dict[str, int | float | None]:
    totals: dict[str, int | float | None] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cache_miss_tokens": 0,
        "cost_usd": 0.0,
    }
    responses = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("type") != "model_response":
            continue
        responses += 1
        usage = (event.get("payload") or {}).get("usage") or {}
        for key in totals:
            value = usage.get(key)
            if totals[key] is not None and isinstance(value, (int, float)):
                totals[key] += value
            elif value is None:
                totals[key] = None
    if responses == 0:
        return {key: None for key in totals}
    return totals


__all__ = [
    "DEFAULT_CHECKOUT",
    "DEFAULT_JOBS_DIR",
    "DEFAULT_REPORTS_DIR",
    "DEFAULT_SUBSET_PATH",
    "DeepSWEError",
    "DeepSWESubset",
    "ForgeLoopPierAgent",
    "PierRuntime",
    "PierWorkspace",
    "build_pier_tools",
    "check_requirements",
    "import_pier_results",
    "pier_command",
    "run_deepswe",
    "select_task_ids",
]
