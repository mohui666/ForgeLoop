from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class GitError(RuntimeError):
    pass


def is_git_repo(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def git_output(path: Path, *args: str, timeout: float = 30.0) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


@dataclass(frozen=True)
class Checkpoint:
    id: str
    ref: str
    commit: str
    created_at: str
    index_backup: str
    had_index: bool


class CheckpointManager:
    """Git-object checkpoints that preserve pre-existing index and worktree state."""

    def __init__(self, repo: Path, home: Path, session_id: str) -> None:
        self.repo = repo.resolve()
        self.directory = home.resolve() / "checkpoints" / session_id
        self.session_id = session_id

    def create(self) -> Checkpoint:
        if not is_git_repo(self.repo):
            raise GitError("Checkpoint requires a Git repository")
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint_id = uuid.uuid4().hex
        index_path = Path(
            git_output(self.repo, "rev-parse", "--git-path", "index").strip()
        )
        if not index_path.is_absolute():
            index_path = self.repo / index_path
        had_index = index_path.exists()
        index_backup = self.directory / f"{checkpoint_id}.index"
        if had_index:
            shutil.copy2(index_path, index_backup)
        temp_index = self.directory / f"{checkpoint_id}.tmp-index"
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(temp_index)
        if self._run("read-tree", "HEAD", env=env, check=False).returncode != 0:
            self._run("read-tree", "--empty", env=env)
        self._run("add", "-A", "--", ".", env=env)
        tree = self._run("write-tree", env=env).stdout.strip()
        commit_env = env | {
            "GIT_AUTHOR_NAME": "ForgeLoop",
            "GIT_AUTHOR_EMAIL": "checkpoint@forgeloop.local",
            "GIT_COMMITTER_NAME": "ForgeLoop",
            "GIT_COMMITTER_EMAIL": "checkpoint@forgeloop.local",
        }
        commit = self._run(
            "commit-tree",
            tree,
            "-m",
            f"ForgeLoop checkpoint {checkpoint_id}",
            env=commit_env,
        ).stdout.strip()
        ref = f"refs/forgeloop/checkpoints/{self.session_id}/{checkpoint_id}"
        self._run("update-ref", ref, commit)
        temp_index.unlink(missing_ok=True)
        checkpoint = Checkpoint(
            checkpoint_id,
            ref,
            commit,
            datetime.now(timezone.utc).isoformat(),
            base64.b64encode(index_backup.read_bytes()).decode("ascii")
            if had_index
            else "",
            had_index,
        )
        (self.directory / f"{checkpoint_id}.json").write_text(
            json.dumps(checkpoint.__dict__, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index_backup.unlink(missing_ok=True)
        return checkpoint

    def undo(self, checkpoint_id: str) -> None:
        checkpoint = self.load(checkpoint_id)
        current_tree, temp_index = self._worktree_tree("undo-current")
        added = self._run(
            "diff",
            "--name-only",
            "--diff-filter=A",
            "-z",
            checkpoint.commit,
            current_tree,
        ).stdout.split("\0")
        self._run("checkout", checkpoint.commit, "--", ".")
        for relative in filter(None, added):
            target = (self.repo / relative).resolve()
            try:
                target.relative_to(self.repo)
            except ValueError as exc:
                raise GitError(f"Unsafe checkpoint path: {relative}") from exc
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        index_path = Path(
            git_output(self.repo, "rev-parse", "--git-path", "index").strip()
        )
        if not index_path.is_absolute():
            index_path = self.repo / index_path
        if checkpoint.had_index:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_bytes(base64.b64decode(checkpoint.index_backup))
        else:
            index_path.unlink(missing_ok=True)
        temp_index.unlink(missing_ok=True)
        self._run("update-ref", "-d", checkpoint.ref)
        (self.directory / f"{checkpoint.id}.json").unlink(missing_ok=True)

    def load(self, checkpoint_id: str) -> Checkpoint:
        path = self.directory / f"{checkpoint_id}.json"
        if not path.exists():
            raise GitError(f"Checkpoint not found: {checkpoint_id}")
        return Checkpoint(**json.loads(path.read_text(encoding="utf-8")))

    def _worktree_tree(self, name: str) -> tuple[str, Path]:
        temp_index = self.directory / f"{name}-{uuid.uuid4().hex}.index"
        env = os.environ.copy() | {"GIT_INDEX_FILE": str(temp_index)}
        if self._run("read-tree", "HEAD", env=env, check=False).returncode != 0:
            self._run("read-tree", "--empty", env=env)
        self._run("add", "-A", "--", ".", env=env)
        return self._run("write-tree", env=env).stdout.strip(), temp_index

    def _run(
        self, *args: str, env: dict[str, str] | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
            timeout=60,
            check=False,
        )
        if check and result.returncode != 0:
            raise GitError(result.stderr.strip() or f"git {' '.join(args)} failed")
        return result
