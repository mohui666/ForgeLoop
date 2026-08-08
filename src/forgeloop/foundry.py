from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from forgeloop.evals import EVAL_SCHEMA_VERSION, FixtureRepository
from forgeloop.runtime import DockerRuntime

FOUNDRY_SCHEMA_VERSION = "forgeloop.foundry.v1"


def default_catalog_path() -> Path:
    return Path(__file__).parent / "foundry_assets" / "catalogs" / "stage_b.json"


class FoundryError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceTask:
    id: str
    repository: str
    fix_commit: str
    source_pr: str | None
    description: str
    test_paths: tuple[str, ...]
    solution_paths: tuple[str, ...]
    verifier_command: str
    verifier_timeout_seconds: float
    timeout_seconds: float
    difficulty: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class BuildResult:
    suite_path: Path
    accepted: int
    filtered: int
    image: str
    image_id: str


class FoundryBuilder:
    """Build a small, curated real-repository eval suite.

    Candidate discovery and task descriptions remain manual. Commit checkout,
    patch splitting, fixture export, Docker validation, and manifest generation
    are deterministic and automated.
    """

    def __init__(
        self,
        catalog_path: Path,
        output_dir: Path,
        cache_dir: Path,
        *,
        repeats: int = 2,
    ) -> None:
        self.catalog_path = catalog_path.expanduser().resolve()
        self.output_dir = output_dir.expanduser().resolve()
        self.cache_dir = cache_dir.expanduser().resolve()
        if repeats < 2:
            raise ValueError("Foundry validation requires at least two repeats")
        self.repeats = repeats

    def build(self) -> BuildResult:
        if self.output_dir.exists():
            raise FoundryError(
                f"Output already exists; choose a fresh directory: {self.output_dir}"
            )
        catalog = self._load_catalog()
        tasks = self._parse_tasks(catalog)
        docker = catalog["docker"]
        dockerfile = self._catalog_path(str(docker["dockerfile"]), "Dockerfile")
        image = str(docker["image"])

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = self.output_dir.parent / f".{self.output_dir.name}.building"
        if staging.exists():
            raise FoundryError(f"Stale build staging directory exists: {staging}")
        staging.mkdir()
        try:
            docker_dir = staging / "docker"
            docker_dir.mkdir()
            shutil.copy2(dockerfile, docker_dir / "Dockerfile")
            image_id = self._build_image(image, docker_dir / "Dockerfile", docker_dir)
            task_records = [
                self._build_task(task, staging, image, image_id) for task in tasks
            ]
            suite = {
                "schema_version": EVAL_SCHEMA_VERSION,
                "suite_id": str(catalog["suite_id"]),
                "suite_kind": "real-swe",
                "foundry_schema_version": FOUNDRY_SCHEMA_VERSION,
                "generated_at": catalog.get("generated_at"),
                "screening": catalog.get("screening", {}),
                "tasks": task_records,
            }
            suite_path = staging / "tasks.json"
            suite_path.write_text(
                json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staging.rename(self.output_dir)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

        screening = catalog.get("screening", {})
        return BuildResult(
            suite_path=self.output_dir / "tasks.json",
            accepted=len(task_records),
            filtered=int(screening.get("rejected", 0)),
            image=image,
            image_id=image_id,
        )

    def _load_catalog(self) -> dict[str, Any]:
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != FOUNDRY_SCHEMA_VERSION:
            raise FoundryError(
                f"Unsupported foundry schema: {data.get('schema_version')}"
            )
        return data

    def _parse_tasks(self, catalog: dict[str, Any]) -> tuple[SourceTask, ...]:
        tasks: list[SourceTask] = []
        seen: set[str] = set()
        for raw in catalog.get("tasks", []):
            task_id = str(raw["id"])
            if task_id in seen:
                raise FoundryError(f"Duplicate foundry task id: {task_id}")
            seen.add(task_id)
            test_paths = self._safe_paths(raw["test_paths"])
            solution_paths = self._safe_paths(raw["solution_paths"])
            if set(test_paths) & set(solution_paths):
                raise FoundryError(f"Test and solution paths overlap for {task_id}")
            verifier = raw["verifier"]
            tasks.append(
                SourceTask(
                    id=task_id,
                    repository=str(raw["repository"]),
                    fix_commit=str(raw["fix_commit"]),
                    source_pr=raw.get("source_pr"),
                    description=str(raw["description"]),
                    test_paths=test_paths,
                    solution_paths=solution_paths,
                    verifier_command=str(verifier["command"]),
                    verifier_timeout_seconds=float(verifier.get("timeout_seconds", 60)),
                    timeout_seconds=float(raw.get("timeout_seconds", 600)),
                    difficulty=str(raw.get("difficulty", "medium")),
                    tags=tuple(str(tag) for tag in raw.get("tags", [])),
                )
            )
        if not 1 <= len(tasks) <= 10:
            raise FoundryError("Foundry catalogs must contain 1-10 tasks")
        self._validate_screening(catalog.get("screening"), len(tasks))
        return tuple(tasks)

    @staticmethod
    def _validate_screening(screening: Any, task_count: int) -> None:
        if not isinstance(screening, dict):
            raise FoundryError("Foundry catalog must include screening metadata")
        try:
            inspected = int(screening["inspected"])
            accepted = int(screening["accepted"])
            rejected = int(screening["rejected"])
            recorded_yield = float(screening["candidate_to_valid_yield"])
            rejected_candidates = screening["rejected_candidates"]
        except (KeyError, TypeError, ValueError) as exc:
            raise FoundryError("Incomplete screening metadata") from exc
        if accepted != task_count or inspected != accepted + rejected:
            raise FoundryError("Inconsistent inspected/accepted/rejected counts")
        expected_yield = accepted / inspected if inspected else 0.0
        if abs(recorded_yield - expected_yield) > 1e-12:
            raise FoundryError(
                "Candidate-to-valid yield does not match screening counts"
            )
        if (
            not isinstance(rejected_candidates, list)
            or len(rejected_candidates) != rejected
        ):
            raise FoundryError("Rejected candidate records do not match rejected count")
        for candidate in rejected_candidates:
            if not isinstance(candidate, dict) or not all(
                str(candidate.get(field, "")).strip()
                for field in ("repository", "commit", "reason_code", "reason")
            ):
                raise FoundryError(
                    "Each rejected candidate needs repository, commit, reason code, and reason"
                )

    @staticmethod
    def _safe_paths(values: list[str]) -> tuple[str, ...]:
        paths: list[str] = []
        for value in values:
            path = PurePosixPath(str(value))
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise FoundryError(f"Unsafe repository path: {value}")
            paths.append(path.as_posix())
        if not paths:
            raise FoundryError("Patch path lists cannot be empty")
        return tuple(paths)

    def _build_task(
        self, task: SourceTask, staging: Path, image: str, image_id: str
    ) -> dict[str, Any]:
        repository = self._repository(task)
        fix_sha = self._git(repository, "rev-parse", f"{task.fix_commit}^{{commit}}")
        parents = self._git(repository, "show", "-s", "--format=%P", fix_sha).split()
        if len(parents) != 1:
            raise FoundryError(
                f"{task.id} fix commit must have exactly one parent, got {len(parents)}"
            )
        base_sha = parents[0]
        changed = set(
            self._git(
                repository, "diff", "--name-only", base_sha, fix_sha, "--"
            ).splitlines()
        )
        configured = set(task.test_paths) | set(task.solution_paths)
        missing = configured - changed
        if missing:
            raise FoundryError(
                f"Configured paths were not changed by {task.id}: {sorted(missing)}"
            )

        artifact_dir = staging / "artifacts" / task.id
        fixture = staging / "fixtures" / task.id
        artifact_dir.mkdir(parents=True)
        fixture.parent.mkdir(parents=True, exist_ok=True)
        test_patch = self._patch(repository, base_sha, fix_sha, task.test_paths)
        gold_patch = self._patch(repository, base_sha, fix_sha, task.solution_paths)
        (artifact_dir / "test.patch").write_bytes(test_patch)
        (artifact_dir / "gold.patch").write_bytes(gold_patch)
        self._export(repository, base_sha, fixture)
        self._apply_patch(fixture, test_patch, f"{task.id} test patch")

        validation = self._validate(task, fixture, gold_patch, image, image_id)
        with tempfile.TemporaryDirectory(
            prefix=f"forgeloop-fingerprint-{task.id}-"
        ) as raw:
            fingerprint_dir = Path(raw) / "workspace"
            fixture_sha = FixtureRepository().materialize(
                task.id, fixture, fingerprint_dir
            )
        metadata = {
            "task_id": task.id,
            "source": {
                "repository": task.repository,
                "pr": task.source_pr,
                "fix_commit": fix_sha,
                "base_sha": base_sha,
            },
            "patches": {
                "test": "test.patch",
                "gold": "gold.patch",
                "test_sha256": hashlib.sha256(test_patch).hexdigest(),
                "gold_sha256": hashlib.sha256(gold_patch).hexdigest(),
            },
            "docker": {
                "image": image,
                "image_id": image_id,
            },
            "validation": validation,
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "id": task.id,
            "description": task.description,
            "fixture": f"fixtures/{task.id}",
            "base_commit": fixture_sha,
            "mode": "task",
            "verifier": {
                "command": task.verifier_command,
                "timeout_seconds": task.verifier_timeout_seconds,
            },
            "timeout_seconds": task.timeout_seconds,
            "expected_outcome": "completed",
            "tags": list(task.tags),
            "difficulty": task.difficulty,
            "stage": "a",
            "source": metadata["source"],
            "docker": {
                "image": image,
                "dockerfile": "docker/Dockerfile",
                "build_context": "docker",
                "image_id": image_id,
            },
            "foundry_artifact": f"artifacts/{task.id}/metadata.json",
        }

    def _repository(self, task: SourceTask) -> Path:
        normalized = task.repository.replace("\\", "/")
        slug = normalized.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
        target = self.cache_dir / slug
        if not target.exists():
            self._command(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    task.repository,
                    str(target),
                ],
                timeout=300,
            )
        self._command(
            [
                "git",
                "-C",
                str(target),
                "fetch",
                "--depth=2",
                "origin",
                task.fix_commit,
            ],
            timeout=300,
        )
        return target

    def _validate(
        self,
        task: SourceTask,
        fixture: Path,
        gold_patch: bytes,
        image: str,
        image_id: str,
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        for repeat in range(1, self.repeats + 1):
            with tempfile.TemporaryDirectory(
                prefix=f"forgeloop-validate-{task.id}-{repeat}-"
            ) as raw:
                workspace = Path(raw) / "workspace"
                shutil.copytree(fixture, workspace)
                before = self._verify_in_container(task, workspace, image)
                if before["exit_code"] == 0:
                    raise FoundryError(
                        f"{task.id} verifier unexpectedly passed before gold patch: "
                        f"{before['stderr'] or before['stdout']}"
                    )
                self._apply_patch(workspace, gold_patch, f"{task.id} gold patch")
                after = self._verify_in_container(task, workspace, image)
                if after["exit_code"] != 0:
                    raise FoundryError(
                        f"{task.id} verifier failed after gold patch: "
                        f"{after['stderr'] or after['stdout']}"
                    )
                attempts.append({"repeat": repeat, "before": before, "after": after})
        return {
            "status": "accepted",
            "repeats": self.repeats,
            "deterministic": True,
            "image_id": image_id,
            "attempts": attempts,
        }

    @staticmethod
    def _verify_in_container(
        task: SourceTask, workspace: Path, image: str
    ) -> dict[str, Any]:
        runtime = DockerRuntime(image=image)
        started = time.perf_counter()
        try:
            runtime.start(workspace)
            result = runtime.run(
                task.verifier_command,
                workspace,
                task.verifier_timeout_seconds,
            )
            return {
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        finally:
            runtime.close()

    @staticmethod
    def _patch(
        repository: Path, base_sha: str, fix_sha: str, paths: tuple[str, ...]
    ) -> bytes:
        result = FoundryBuilder._command(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--binary",
                "--full-index",
                base_sha,
                fix_sha,
                "--",
                *paths,
            ],
            timeout=60,
            text=False,
        )
        if not result.stdout:
            raise FoundryError(f"Empty patch for paths: {paths}")
        return result.stdout

    @staticmethod
    def _export(repository: Path, sha: str, destination: Path) -> None:
        destination.mkdir()
        with tempfile.TemporaryDirectory(prefix="forgeloop-archive-") as raw:
            archive = Path(raw) / "source.tar"
            FoundryBuilder._command(
                [
                    "git",
                    "-C",
                    str(repository),
                    "archive",
                    "--format=tar",
                    f"--output={archive}",
                    sha,
                ],
                timeout=180,
            )
            with tarfile.open(archive) as handle:
                handle.extractall(destination, filter="data")

    @staticmethod
    def _apply_patch(workspace: Path, patch: bytes, label: str) -> None:
        env = os.environ.copy()
        # Fixtures are often generated below ForgeLoop's own Git worktree. Stop
        # git from discovering that parent repository and prefix-filtering every
        # source patch as an unrelated path.
        env["GIT_CEILING_DIRECTORIES"] = str(workspace.resolve().parent)
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn"],
            cwd=workspace,
            input=patch,
            capture_output=True,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise FoundryError(
                f"Could not apply {label}: {result.stderr.decode(errors='replace')}"
            )

    @staticmethod
    def _git(repository: Path, *arguments: str) -> str:
        result = FoundryBuilder._command(
            ["git", "-C", str(repository), *arguments], timeout=60
        )
        return result.stdout.strip()

    @staticmethod
    def _build_image(image: str, dockerfile: Path, context: Path) -> str:
        docker = DockerRuntime._find_docker()
        result = subprocess.run(
            [
                docker,
                "build",
                "--file",
                str(dockerfile),
                "--tag",
                image,
                str(context),
            ],
            capture_output=True,
            timeout=600,
            check=False,
            env=DockerRuntime._docker_env(docker),
        )
        if result.returncode != 0:
            raise FoundryError(
                "Docker image build failed: "
                + result.stderr.decode(errors="replace")[-4000:]
            )
        inspect = subprocess.run(
            [docker, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=DockerRuntime._docker_env(docker),
        )
        if inspect.returncode != 0:
            raise FoundryError(f"Could not inspect built Docker image: {image}")
        return inspect.stdout.strip()

    def _catalog_path(self, value: str, label: str) -> Path:
        path = (self.catalog_path.parent / value).resolve()
        if not path.is_file():
            raise FoundryError(f"{label} does not exist: {path}")
        return path

    @staticmethod
    def _command(
        argv: list[str], *, timeout: float, text: bool = True
    ) -> subprocess.CompletedProcess[Any]:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=text,
            errors="replace" if text else None,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            stderr = (
                result.stderr
                if isinstance(result.stderr, str)
                else result.stderr.decode(errors="replace")
            )
            raise FoundryError(f"Command failed ({' '.join(argv[:3])}): {stderr}")
        return result
