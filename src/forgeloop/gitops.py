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
from typing import Any

from forgeloop.identifiers import validate_portable_identifier
from forgeloop.persistence import atomic_write_text


class GitError(RuntimeError):
    pass


def _validate_checkpoint_identifier(value: str, *, name: str) -> str:
    """Return a path- and Git-ref-safe identifier or fail closed."""
    try:
        return validate_portable_identifier(value, label=name)
    except ValueError as exc:
        raise GitError(str(exc)) from exc


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
        session_id = _validate_checkpoint_identifier(session_id, name="session id")
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
        try:
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
            ref = self._checkpoint_ref(checkpoint_id)
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
            metadata_path = self.directory / f"{checkpoint_id}.json"
            self._run("update-ref", ref, commit)
            try:
                atomic_write_text(
                    metadata_path,
                    json.dumps(checkpoint.__dict__, indent=2, sort_keys=True) + "\n",
                )
            except BaseException as publish_error:
                try:
                    self._run("update-ref", "-d", ref, commit)
                except BaseException as rollback_error:
                    raise GitError(
                        "Checkpoint metadata publication failed and the Git ref "
                        f"rollback also failed: {ref}"
                    ) from rollback_error
                try:
                    metadata_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    raise GitError(
                        "Checkpoint metadata publication failed; the Git ref was "
                        "rolled back but metadata cleanup failed"
                    ) from cleanup_error
                raise GitError(
                    "Checkpoint metadata publication failed"
                ) from publish_error
            return checkpoint
        finally:
            temp_index.unlink(missing_ok=True)
            index_backup.unlink(missing_ok=True)

    def undo(self, checkpoint_id: str) -> None:
        checkpoint_id = _validate_checkpoint_identifier(
            checkpoint_id, name="checkpoint id"
        )
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
        self._run("update-ref", "-d", checkpoint.ref, checkpoint.commit)
        (self.directory / f"{checkpoint.id}.json").unlink(missing_ok=True)

    def load(self, checkpoint_id: str) -> Checkpoint:
        checkpoint_id = _validate_checkpoint_identifier(
            checkpoint_id, name="checkpoint id"
        )
        path = self.directory / f"{checkpoint_id}.json"
        if not path.exists():
            raise GitError(f"Checkpoint not found: {checkpoint_id}")
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GitError(f"Invalid checkpoint metadata: {checkpoint_id}") from exc
        expected_fields = {
            "id",
            "ref",
            "commit",
            "created_at",
            "index_backup",
            "had_index",
        }
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise GitError(f"Invalid checkpoint metadata schema: {checkpoint_id}")
        if any(
            not isinstance(raw[field], str)
            for field in ("id", "ref", "commit", "created_at", "index_backup")
        ) or not isinstance(raw["had_index"], bool):
            raise GitError(f"Invalid checkpoint metadata field types: {checkpoint_id}")
        expected_ref = self._checkpoint_ref(checkpoint_id)
        if raw["id"] != checkpoint_id or raw["ref"] != expected_ref:
            raise GitError(f"Checkpoint metadata identity mismatch: {checkpoint_id}")
        commit = raw["commit"]
        if len(commit) not in (40, 64) or any(
            char not in "0123456789abcdef" for char in commit
        ):
            raise GitError(f"Invalid checkpoint commit id: {checkpoint_id}")
        try:
            created_at = datetime.fromisoformat(raw["created_at"])
        except ValueError as exc:
            raise GitError(
                f"Invalid checkpoint creation time: {checkpoint_id}"
            ) from exc
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise GitError(f"Invalid checkpoint creation time: {checkpoint_id}")
        try:
            decoded_index = base64.b64decode(raw["index_backup"], validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise GitError(f"Invalid checkpoint index backup: {checkpoint_id}") from exc
        if raw["had_index"] != bool(decoded_index):
            raise GitError(f"Inconsistent checkpoint index backup: {checkpoint_id}")
        ref_result = self._run(
            "show-ref", "--verify", "--hash", expected_ref, check=False
        )
        if ref_result.returncode != 0 or ref_result.stdout.strip() != commit:
            raise GitError(f"Checkpoint ref does not match metadata: {checkpoint_id}")
        object_result = self._run("cat-file", "-t", commit, check=False)
        if object_result.returncode != 0 or object_result.stdout.strip() != "commit":
            raise GitError(f"Checkpoint commit object is invalid: {checkpoint_id}")
        return Checkpoint(**raw)

    def _checkpoint_ref(self, checkpoint_id: str) -> str:
        return f"refs/forgeloop/checkpoints/{self.session_id}/{checkpoint_id}"

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
