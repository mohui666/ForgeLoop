from __future__ import annotations

import os
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(ValueError):
    pass


_EPHEMERAL_GIT_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def is_ephemeral_git_path(value: str) -> bool:
    normalized = value.replace("\\", "/").strip("/")
    if normalized == ".forgeloop" or normalized.startswith(".forgeloop/"):
        return True
    parts = set(normalized.split("/"))
    return (
        bool(parts & _EPHEMERAL_GIT_PARTS)
        or normalized.endswith((".pyc", ".pyo"))
        or normalized in {".coverage"}
    )


@dataclass(frozen=True)
class GitSnapshot:
    is_repository: bool
    repository_root: str | None = None
    head: str | None = None
    branch: str | None = None
    status: str = ""
    error: str | None = None


@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        if not root.is_dir():
            raise WorkspaceError(
                f"Workspace does not exist or is not a directory: {root}"
            )
        object.__setattr__(self, "root", root)

    def resolve(
        self, relative_path: str | Path = ".", *, must_exist: bool = False
    ) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes workspace: {relative_path}") from exc
        if must_exist and not candidate.exists():
            raise WorkspaceError(f"Path does not exist: {relative_path}")
        return candidate

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root)).replace(os.sep, "/")

    def git_snapshot(self, *, max_chars: int = 20_000) -> GitSnapshot:
        """Capture read-only Git identity and dirty state for trajectory boundaries."""
        try:
            root_result = self._git("rev-parse", "--show-toplevel")
        except OSError as exc:
            return GitSnapshot(False, error=str(exc))
        if root_result.returncode != 0:
            return GitSnapshot(False, error=root_result.stderr.strip() or None)

        head = self._git("rev-parse", "HEAD")
        branch = self._git("symbolic-ref", "--short", "-q", "HEAD")
        status = self._git("status", "--short", "--untracked-files=all")
        status_text = status.stdout[:max_chars]
        if len(status.stdout) > max_chars:
            status_text += f"\n... <{len(status.stdout) - max_chars} chars omitted>"
        return GitSnapshot(
            True,
            repository_root=root_result.stdout.strip(),
            head=head.stdout.strip() if head.returncode == 0 else None,
            branch=branch.stdout.strip() if branch.returncode == 0 else None,
            status=status_text,
            error=(status.stderr.strip() or None) if status.returncode != 0 else None,
        )

    def git_progress_fingerprint(
        self,
        *,
        base_head: str | None = None,
        max_untracked_bytes: int = 1_000_000,
    ) -> str:
        """Hash the deliverable tree, optionally relative to the run's base commit.

        Using a stable ``base_head`` keeps the fingerprint unchanged when the same
        tree is committed. This lets execution state distinguish a real source edit
        from Git finalization and matches collectors that extract ``base..HEAD``.
        """
        if base_head:
            digest = hashlib.sha256()
            changed = self._git("diff", "--name-only", "-z", base_head)
            untracked = self._git("ls-files", "--others", "--exclude-standard", "-z")
            paths = sorted(
                {
                    item
                    for item in (changed.stdout + untracked.stdout).split("\0")
                    if item
                    and not item.replace("\\", "/").startswith(".forgeloop/")
                    and not is_ephemeral_git_path(item)
                }
            )
            remaining = max_untracked_bytes
            for relative in paths:
                digest.update(relative.encode("utf-8", errors="replace"))
                try:
                    path = self.resolve(relative)
                    if not path.is_file():
                        digest.update(b"<deleted>")
                        continue
                    digest.update(b"+x" if path.stat().st_mode & 0o111 else b"-x")
                    content = path.read_bytes()[:remaining]
                except OSError:
                    digest.update(b"<unreadable>")
                    continue
                digest.update(content)
                remaining -= len(content)
            return digest.hexdigest()
        digest = hashlib.sha256()
        status = self._git(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude).forgeloop",
        )
        digest.update(status.stdout.encode("utf-8", errors="replace"))
        diff = self._git("diff", "--binary", "HEAD")
        if diff.returncode != 0:
            diff = self._git("diff", "--binary")
        digest.update(diff.stdout.encode("utf-8", errors="replace"))
        untracked = self._git("ls-files", "--others", "--exclude-standard", "-z")
        remaining = max_untracked_bytes
        for relative in filter(None, untracked.stdout.split("\0")):
            if relative.replace("\\", "/").startswith(".forgeloop/"):
                continue
            if remaining <= 0:
                break
            try:
                path = self.resolve(relative, must_exist=True)
                if not path.is_file():
                    continue
                content = path.read_bytes()[:remaining]
            except OSError:
                continue
            digest.update(relative.encode("utf-8", errors="replace"))
            digest.update(content)
            remaining -= len(content)
        return digest.hexdigest()

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
