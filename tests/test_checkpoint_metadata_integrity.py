from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from forgeloop.gitops import CheckpointManager, GitError


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "source.txt")
    _git(repo, "commit", "-m", "base")
    return repo


def _metadata_path(manager: CheckpointManager, checkpoint_id: str) -> Path:
    return manager.directory / f"{checkpoint_id}.json"


def _rewrite_metadata(
    manager: CheckpointManager, checkpoint_id: str, **changes: Any
) -> None:
    path = _metadata_path(manager, checkpoint_id)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(changes)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_create_rolls_back_ref_when_atomic_metadata_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    manager = CheckpointManager(repo, tmp_path / "state", "session")
    (repo / "source.txt").write_text("checkpoint\n", encoding="utf-8")

    def fail_publish(path: object, *_args: object, **_kwargs: object) -> None:
        # Model a durability failure after os.replace made the complete file visible.
        Path(path).write_text("published but not durable", encoding="utf-8")
        raise OSError("injected metadata failure")

    monkeypatch.setattr("forgeloop.gitops.atomic_write_text", fail_publish)

    with pytest.raises(GitError, match="metadata publication failed"):
        manager.create()

    assert _git(repo, "for-each-ref", "--format=%(refname)", "refs/forgeloop/") == ""
    assert list(manager.directory.glob("*.json")) == []
    assert list(manager.directory.glob("*.index")) == []
    assert list(manager.directory.glob("*.tmp-index")) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "0" * 32, "identity mismatch"),
        ("ref", "refs/heads/main", "identity mismatch"),
        ("commit", "not-an-object", "commit id"),
        ("created_at", "yesterday", "creation time"),
        ("index_backup", "%%%", "index backup"),
        ("had_index", False, "Inconsistent"),
    ],
)
def test_undo_rejects_tampered_metadata_before_worktree_mutation(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    repo = _repo(tmp_path)
    manager = CheckpointManager(repo, tmp_path / "state", "session")
    checkpoint = manager.create()
    (repo / "source.txt").write_text("must remain\n", encoding="utf-8")
    _rewrite_metadata(manager, checkpoint.id, **{field: value})

    with pytest.raises(GitError, match=message):
        manager.undo(checkpoint.id)

    assert (repo / "source.txt").read_text(encoding="utf-8") == "must remain\n"
    assert _git(repo, "show-ref", "--verify", checkpoint.ref) != ""


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {"id": "only-one-field"},
        {
            "id": 1,
            "ref": "ref",
            "commit": "0" * 40,
            "created_at": "2026-01-01T00:00:00+00:00",
            "index_backup": "",
            "had_index": False,
        },
    ],
)
def test_load_rejects_invalid_metadata_schema(tmp_path: Path, metadata: object) -> None:
    repo = _repo(tmp_path)
    manager = CheckpointManager(repo, tmp_path / "state", "session")
    checkpoint = manager.create()
    _metadata_path(manager, checkpoint.id).write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(GitError, match="metadata"):
        manager.load(checkpoint.id)


def test_load_rejects_ref_moved_away_from_recorded_commit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    manager = CheckpointManager(repo, tmp_path / "state", "session")
    (repo / "source.txt").write_text("checkpoint\n", encoding="utf-8")
    checkpoint = manager.create()
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", checkpoint.ref, base)

    with pytest.raises(GitError, match="ref does not match"):
        manager.undo(checkpoint.id)

    assert (repo / "source.txt").read_text(encoding="utf-8") == "checkpoint\n"
    assert _git(repo, "rev-parse", checkpoint.ref) == base


def test_load_rejects_extra_metadata_fields(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    manager = CheckpointManager(repo, tmp_path / "state", "session")
    checkpoint = manager.create()
    _rewrite_metadata(manager, checkpoint.id, unexpected=True)

    with pytest.raises(GitError, match="metadata schema"):
        manager.load(checkpoint.id)
