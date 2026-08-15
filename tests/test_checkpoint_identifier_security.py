from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forgeloop.gitops import CheckpointManager, GitError


INVALID_IDENTIFIERS = (
    "",
    ".",
    "..",
    "../escape",
    "..\\escape",
    "/absolute",
    "C:\\absolute",
    "nested/session",
    "nested\\session",
    "refs@{1}",
    "name.lock",
    "CON",
    "a" * 129,
    "with space",
    "é",
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
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


@pytest.mark.parametrize("session_id", INVALID_IDENTIFIERS)
def test_invalid_session_id_is_rejected_before_directory_creation(
    tmp_path: Path, session_id: str
) -> None:
    home = tmp_path / "state"

    with pytest.raises(GitError, match="Invalid session id"):
        CheckpointManager(tmp_path, home, session_id)

    assert not home.exists()


@pytest.mark.parametrize("checkpoint_id", INVALID_IDENTIFIERS)
def test_load_rejects_invalid_checkpoint_id_without_outside_read(
    tmp_path: Path, checkpoint_id: str
) -> None:
    manager = CheckpointManager(tmp_path, tmp_path / "state", "session_123")
    outside = tmp_path / "escape.json"
    outside.write_text("outside sentinel", encoding="utf-8")

    with pytest.raises(GitError, match="Invalid checkpoint id"):
        manager.load(checkpoint_id)

    assert outside.read_text(encoding="utf-8") == "outside sentinel"
    assert not manager.directory.exists()


@pytest.mark.parametrize("checkpoint_id", INVALID_IDENTIFIERS)
def test_undo_rejects_invalid_checkpoint_id_before_git_mutation(
    tmp_path: Path, checkpoint_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    manager = CheckpointManager(repo, tmp_path / "state", "session-123")
    original_head = _git(repo, "rev-parse", "HEAD")
    calls: list[tuple[str, ...]] = []

    def record_run(*args: str, **_kwargs: object) -> None:
        calls.append(args)
        raise AssertionError("invalid checkpoint id reached Git")

    monkeypatch.setattr(manager, "_run", record_run)
    with pytest.raises(GitError, match="Invalid checkpoint id"):
        manager.undo(checkpoint_id)

    assert calls == []
    assert _git(repo, "rev-parse", "HEAD") == original_head
    assert (repo / "source.txt").read_text(encoding="utf-8") == "base\n"


def test_uuid_checkpoint_id_and_safe_session_remain_supported(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    manager = CheckpointManager(repo, tmp_path / "state", "session_123-ABC")
    (repo / "source.txt").write_text("checkpoint state\n", encoding="utf-8")

    checkpoint = manager.create()

    assert len(checkpoint.id) == 32
    assert checkpoint.id.isalnum()
    assert manager.load(checkpoint.id) == checkpoint
    assert checkpoint.ref.startswith("refs/forgeloop/checkpoints/session_123-ABC/")
    assert _git(repo, "show-ref", "--verify", checkpoint.ref)


def test_create_cleans_temporary_index_when_git_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    manager = CheckpointManager(repo, tmp_path / "state", "session")
    original_run = manager._run

    def fail_write_tree(
        *args: str, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if args == ("write-tree",):
            raise GitError("injected write-tree failure")
        return original_run(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(manager, "_run", fail_write_tree)
    with pytest.raises(GitError, match="injected write-tree failure"):
        manager.create()

    assert list(manager.directory.glob("*.tmp-index")) == []
    assert list(manager.directory.glob("*.index")) == []
