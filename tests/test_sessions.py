from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import forgeloop.sessions as sessions_module
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


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_session_save_write_failures_preserve_previous_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    store, session, original, original_updated_at = _saved_session(tmp_path)
    session.last_summary = "must not commit"
    real_open = Path.open

    class FailingFile:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> "FailingFile":
            self.handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.handle.__exit__(*args)

        def write(self, content: str) -> int:
            if failure == "write":
                raise OSError("injected write failure")
            return self.handle.write(content)

        def flush(self) -> None:
            if failure == "flush":
                raise OSError("injected flush failure")
            self.handle.flush()

        def fileno(self) -> int:
            return self.handle.fileno()

    def failing_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(path, *args, **kwargs)
        if path.suffix == ".tmp":
            return FailingFile(handle)
        return handle

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match=f"injected {failure} failure"):
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

    monkeypatch.setattr(sessions_module.os, "replace", fail_replace)

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

    monkeypatch.setattr(sessions_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        store.save(session)

    _assert_failed_save_preserved(store, session, original, original_updated_at)


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
