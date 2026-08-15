from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

import forgeloop.sessions as sessions_module
from forgeloop import persistence
from forgeloop.sessions import Session, SessionStore


def _saved_session(tmp_path: Path) -> tuple[SessionStore, Session, str, str]:
    store = SessionStore(tmp_path)
    session = Session.create(tmp_path)
    session.last_summary = "known good"
    store.save(session)
    path = store.path_for(session.id)
    return store, session, path.read_text(encoding="utf-8"), session.updated_at


def _assert_failed_save_preserved(
    store: SessionStore,
    session: Session,
    original: str,
    original_updated_at: str,
) -> None:
    path = store.path_for(session.id)
    assert path.read_text(encoding="utf-8") == original
    assert store.load(session.id).last_summary == "known good"
    assert session.updated_at == original_updated_at
    assert list(store.directory.glob(f".{path.name}.*.tmp")) == []


def test_session_save_is_atomic_and_round_trips_without_temp_files(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(tmp_path)
    session.last_summary = "resume here"

    store.save(session)

    loaded = store.load(session.id)
    assert loaded == session
    assert (
        json.loads(store.path_for(session.id).read_text(encoding="utf-8"))["updated_at"]
        == session.updated_at
    )
    assert list(store.directory.glob("*.tmp")) == []


def test_session_save_write_failure_in_common_primitive_preserves_previous_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store, session, original, original_updated_at = _saved_session(tmp_path)
    session.last_summary = "must not commit"

    def fail_write(_stream: Any, _data: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(persistence, "_write_all", fail_write)

    with pytest.raises(OSError, match="injected write failure"):
        store.save(session)

    _assert_failed_save_preserved(store, session, original, original_updated_at)


def test_session_save_replace_failure_preserves_previous_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store, session, original, original_updated_at = _saved_session(tmp_path)
    session.last_summary = "must not commit"

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"injected replace failure: {source} -> {destination}")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        store.save(session)

    _assert_failed_save_preserved(store, session, original, original_updated_at)


def test_session_save_fsync_failure_preserves_previous_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store, session, original, original_updated_at = _saved_session(tmp_path)
    session.last_summary = "must not commit"

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(persistence.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        store.save(session)

    _assert_failed_save_preserved(store, session, original, original_updated_at)


def test_session_save_reconciles_memory_after_post_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store, session, _original, original_updated_at = _saved_session(tmp_path)
    session.last_summary = "complete payload reached disk"
    real_atomic_write = persistence.atomic_write_text

    def replace_then_fail(path: Path, content: str, **kwargs: Any) -> None:
        real_atomic_write(path, content, **kwargs)
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(persistence, "atomic_write_text", replace_then_fail)

    with pytest.raises(OSError, match="directory fsync failure"):
        store.save(session)

    loaded = store.load(session.id)
    assert loaded.last_summary == "complete payload reached disk"
    assert loaded.updated_at == session.updated_at
    assert session.updated_at > original_updated_at
    assert list(store.directory.glob("*.tmp")) == []


def test_session_store_serializes_parallel_saves_and_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(tmp_path)
    real_atomic_write = persistence.atomic_write_text
    written_timestamps: list[str] = []
    observation_lock = threading.Lock()

    def recording_write(path: Path, content: str, **kwargs: Any) -> None:
        real_atomic_write(path, content, **kwargs)
        timestamp = json.loads(content)["updated_at"]
        with observation_lock:
            written_timestamps.append(timestamp)

    monkeypatch.setattr(persistence, "atomic_write_text", recording_write)
    barrier = threading.Barrier(9)
    errors: list[BaseException] = []

    def save_once() -> None:
        try:
            barrier.wait()
            store.save(session)
            # A concurrent read must only ever observe complete JSON.
            assert store.load(session.id).id == session.id
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=save_once) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(written_timestamps) == 8
    assert written_timestamps == sorted(written_timestamps)
    assert len(set(written_timestamps)) == 8
    on_disk = json.loads(store.path_for(session.id).read_text(encoding="utf-8"))
    assert on_disk["updated_at"] == session.updated_at == written_timestamps[-1]


def test_session_save_never_moves_updated_at_backwards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store, session, _original, original_updated_at = _saved_session(tmp_path)
    monkeypatch.setattr(sessions_module, "_now", lambda: "2000-01-01T00:00:00+00:00")

    store.save(session)

    assert session.updated_at > original_updated_at
    assert store.load(session.id).updated_at == session.updated_at


def test_session_secret_rejection_creates_no_file_or_temp(tmp_path: Path) -> None:
    secret = "sk-do-not-persist"

    class NoOpRedactor:
        secrets = (secret,)

        @staticmethod
        def redact(value: Any) -> Any:
            return value

    store = SessionStore(tmp_path, redactor=NoOpRedactor())  # type: ignore[arg-type]
    session = Session.create(tmp_path)
    # A malicious/no-op redactor must still be caught before creating a temp file.
    session.last_summary = secret

    with pytest.raises(ValueError, match="API key"):
        store.save(session)

    assert not store.path_for(session.id).exists()
    assert list(store.directory.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "session_id",
    [
        "",
        ".",
        "..",
        ".hidden",
        "trailing.",
        "../escaped",
        "..\\escaped",
        "nested/session",
        "nested\\session",
        "/absolute/session",
        r"C:\absolute\session",
        r"\\server\share\session",
        "NUL",
        "con.json",
        "a" * 129,
    ],
)
def test_session_identifiers_fail_closed_before_creating_files(
    tmp_path: Path,
    session_id: str,
) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(tmp_path)
    session.id = session_id
    outside = tmp_path / "escaped.json"
    outside.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid session id"):
        store.path_for(session_id)
    with pytest.raises(ValueError, match="Invalid session id"):
        store.save(session)

    assert outside.read_text(encoding="utf-8") == "preserve me"
    assert not store.directory.exists()
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.parametrize(
    "session_id",
    [
        "a",
        "0123456789abcdef0123456789abcdef",
        "550e8400-e29b-41d4-a716-446655440000",
        "legacy_session_01-alpha",
    ],
)
def test_session_identifiers_accept_stable_portable_names(
    tmp_path: Path,
    session_id: str,
) -> None:
    store = SessionStore(tmp_path)
    session = Session.create(tmp_path)
    session.id = session_id

    store.save(session)

    assert store.path_for(session_id).parent == store.directory
    assert store.load(session_id) == session


def test_session_load_preserves_unique_short_prefix_semantics(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    first = Session.create(tmp_path)
    first.id = "abc123-session"
    second = Session.create(tmp_path)
    second.id = "abc456-session"
    store.save(first)
    store.save(second)

    assert store.load("abc1") == first
    with pytest.raises(ValueError, match="ambiguous"):
        store.load("abc")


def test_session_path_rejects_a_session_directory_outside_home(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "home")
    store.directory = tmp_path / "outside"

    with pytest.raises(ValueError, match="escapes the ForgeLoop home"):
        store.path_for("abc123")
