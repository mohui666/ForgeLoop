from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from forgeloop.agent_types import RunStatus
from forgeloop.workspace import is_ephemeral_git_path


class DeliveryRuntime(Protocol):
    def run(self, command: str, cwd: Any, timeout_seconds: float) -> Any: ...


@dataclass(frozen=True)
class DeliveryResult:
    ok: bool
    status: str
    base_sha: str | None
    head_sha: str | None
    branch: str | None
    committed: bool
    clean: bool
    patch_bytes: int
    detail: str = ""

    @property
    def has_patch(self) -> bool:
        return self.patch_bytes > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"has_patch": self.has_patch}


class RunDelivery(Protocol):
    def start(self, workspace: Any) -> None: ...

    def deliver(self, workspace: Any, terminal_status: RunStatus) -> DeliveryResult: ...


@dataclass
class GitPatchDelivery:
    """Commit the real task tree so a base-to-HEAD collector can extract it."""

    runtime: DeliveryRuntime
    branch_prefix: str = "forgeloop/deepswe-delivery"
    commit_message: str = "Complete coding agent task"
    require_patch_on_completed: bool = True
    base_sha: str | None = None

    def start(self, workspace: Any) -> None:
        snapshot = workspace.git_snapshot()
        self.base_sha = snapshot.head if snapshot.is_repository else None

    def deliver(self, workspace: Any, terminal_status: RunStatus) -> DeliveryResult:
        if not self.base_sha:
            return DeliveryResult(
                False,
                "not_a_git_repository",
                None,
                None,
                None,
                False,
                False,
                0,
                "A Git base commit was not available at run start.",
            )

        try:
            branch = self._read(workspace, "git symbolic-ref --short -q HEAD") or None
            status = self._meaningful_status(workspace)
            head_before_delivery = self._read(workspace, "git rev-parse HEAD") or None
            committed = False
            if status:
                branch = self._ensure_delivery_branch(workspace, branch)
                self._checked(workspace, "git add -A")
                self._checked(
                    workspace,
                    "git reset -- "
                    "':(glob).forgeloop/**' "
                    "':(glob)**/__pycache__/**' "
                    "':(glob)**/.pytest_cache/**' "
                    "':(glob)**/.mypy_cache/**' "
                    "':(glob)**/.ruff_cache/**' "
                    "':(glob)**/*.pyc' "
                    "':(glob)**/*.pyo' "
                    "':(glob)**/.coverage'",
                )
                staged = self.runtime.run(
                    "git diff --cached --quiet", workspace.root, 30
                )
                if staged.exit_code not in {0, 1}:
                    raise RuntimeError(staged.stderr.strip() or staged.stdout.strip())
                if staged.exit_code == 1:
                    self._checked(
                        workspace,
                        "git commit -m " + shlex.quote(self.commit_message),
                        timeout=120,
                    )
                    committed = True
            elif head_before_delivery != self.base_sha:
                # The model may have committed manually. Preserve that genuine tree,
                # but still put the collector-visible HEAD on the delivery branch.
                branch = self._ensure_delivery_branch(workspace, branch)

            head = self._read(workspace, "git rev-parse HEAD") or None
            patch = self._read(
                workspace,
                "git diff --binary " + shlex.quote(self.base_sha) + " HEAD",
                timeout=120,
            )
            remaining = self._meaningful_status(workspace)
            patch_bytes = len(patch.encode("utf-8"))
            clean = not remaining
            missing_required_patch = (
                terminal_status is RunStatus.COMPLETED
                and self.require_patch_on_completed
                and patch_bytes == 0
            )
            ok = clean and not missing_required_patch
            detail = ""
            if remaining:
                detail = "Working tree remained dirty after delivery: " + remaining
            elif missing_required_patch:
                detail = "Completed run produced no base-to-HEAD patch."
            return DeliveryResult(
                ok,
                "patch_ready"
                if ok and patch_bytes
                else "no_changes"
                if ok
                else "delivery_failed",
                self.base_sha,
                head,
                branch,
                committed,
                clean,
                patch_bytes,
                detail,
            )
        except Exception as exc:  # noqa: BLE001 - delivery is a terminal boundary
            snapshot = workspace.git_snapshot()
            return DeliveryResult(
                False,
                "delivery_failed",
                self.base_sha,
                snapshot.head,
                snapshot.branch,
                False,
                not bool(snapshot.status),
                0,
                f"{type(exc).__name__}: {exc}",
            )

    def _ensure_delivery_branch(self, workspace: Any, branch: str | None) -> str:
        if branch and branch not in {"main", "master"}:
            return branch
        suffix = self.base_sha[:8] if self.base_sha else "task"
        target = f"{self.branch_prefix}-{suffix}"
        created = self.runtime.run(
            "git switch -c " + shlex.quote(target), workspace.root, 30
        )
        if created.exit_code != 0:
            switched = self.runtime.run(
                "git switch " + shlex.quote(target), workspace.root, 30
            )
            if switched.exit_code != 0:
                raise RuntimeError(
                    switched.stderr.strip()
                    or created.stderr.strip()
                    or "Could not create delivery branch"
                )
        return target

    def _read(self, workspace: Any, command: str, *, timeout: float = 30) -> str:
        result = self.runtime.run(command, workspace.root, timeout)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    def _meaningful_status(self, workspace: Any) -> str:
        raw = self._read(workspace, "git status --porcelain --untracked-files=all")
        return "\n".join(
            line
            for line in raw.splitlines()
            if line and not is_ephemeral_git_path(line[3:].strip().strip('"'))
        )

    def _checked(self, workspace: Any, command: str, *, timeout: float = 30) -> None:
        self._read(workspace, command, timeout=timeout)


__all__ = ["DeliveryResult", "GitPatchDelivery", "RunDelivery"]
