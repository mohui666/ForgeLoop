from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import BinaryIO

import pytest

from forgeloop import persistence


class FaultyStream:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        short_write: bool = False,
        zero_write: bool = False,
        fail_write: bool = False,
        fail_flush: bool = False,
        partial_write_failure: bool = False,
        fail_truncate: bool = False,
    ) -> None:
        self.stream = stream
        self.short_write = short_write
        self.zero_write = zero_write
        self.fail_write = fail_write
        self.fail_flush = fail_flush
        self.partial_write_failure = partial_write_failure
        self.fail_truncate = fail_truncate
        self.write_calls = 0

    def __enter__(self) -> FaultyStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.stream.close()

    def write(self, data: bytes | memoryview) -> int:
        self.write_calls += 1
        if self.fail_write:
            raise OSError("injected write failure")
        if self.partial_write_failure:
            self.partial_write_failure = False
            self.stream.write(data[: max(1, len(data) // 2)])
            raise OSError("injected partial write failure")
        if self.zero_write:
            return 0
        if self.short_write and len(data) > 1:
            data = data[: max(1, len(data) // 2)]
        return self.stream.write(data)

    def flush(self) -> None:
        if self.fail_flush:
            self.fail_flush = False
            raise OSError("injected flush failure")
        self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.stream.seek(offset, whence)

    def tell(self) -> int:
        return self.stream.tell()

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)

    def truncate(self, size: int | None = None) -> int:
        if self.fail_truncate:
            raise OSError("injected rollback failure")
        return self.stream.truncate(size)


def test_atomic_write_text_replaces_file_and_cleans_temp(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    persistence.atomic_write_text(target, "new \N{SNOWMAN}")

    assert target.read_text(encoding="utf-8") == "new \N{SNOWMAN}"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_advisory_file_lock_ignores_stale_sidecar_existence(tmp_path: Path) -> None:
    lock_path = tmp_path / "state.lock"
    lock_path.write_bytes(b"stale metadata is not ownership")

    with persistence.advisory_file_lock(lock_path, timeout=0.1):
        assert lock_path.exists()


def test_advisory_file_lock_times_out_while_an_os_lock_is_held(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "state.lock"
    acquired = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with persistence.advisory_file_lock(lock_path, timeout=1):
            acquired.set()
            release.wait(timeout=2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert acquired.wait(timeout=1)
    started = time.monotonic()
    try:
        with pytest.raises(persistence.LockTimeoutError, match="Timed out"):
            with persistence.advisory_file_lock(
                lock_path, timeout=0.05, poll_interval=0.005
            ):
                pass
        assert time.monotonic() - started >= 0.04
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_advisory_lock_release_failure_does_not_hide_transaction_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        import msvcrt

        real_release = msvcrt.locking

        def fail_release(fd: int, mode: int, size: int) -> None:
            if mode == msvcrt.LK_UNLCK:
                raise OSError("injected release failure")
            real_release(fd, mode, size)

        monkeypatch.setattr(msvcrt, "locking", fail_release)
    else:
        import fcntl

        real_release = fcntl.flock

        def fail_release(fd: int, operation: int) -> None:
            if operation == fcntl.LOCK_UN:
                raise OSError("injected release failure")
            real_release(fd, operation)

        monkeypatch.setattr(fcntl, "flock", fail_release)

    lock_path = tmp_path / "release.lock"
    with pytest.raises(persistence.PersistenceError, match="releasing advisory lock"):
        with persistence.advisory_file_lock(lock_path):
            pass

    with pytest.raises(RuntimeError, match="transaction failed"):
        with persistence.advisory_file_lock(lock_path):
            raise RuntimeError("transaction failed")


def test_write_all_handles_short_writes(tmp_path: Path) -> None:
    target = tmp_path / "short.bin"
    with open(target, "wb", buffering=0) as raw:
        stream = FaultyStream(raw, short_write=True)
        persistence._write_all(stream, b"abcdefgh")
    assert target.read_bytes() == b"abcdefgh"
    assert stream.write_calls > 1


def test_write_all_rejects_zero_progress(tmp_path: Path) -> None:
    with open(tmp_path / "zero.bin", "wb", buffering=0) as raw:
        with pytest.raises(OSError, match="no progress"):
            persistence._write_all(FaultyStream(raw, zero_write=True), b"x")


def test_atomic_write_handles_short_writes_in_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.bin"
    real_fdopen = persistence.os.fdopen
    real_replace = persistence.os.replace
    replacements: list[tuple[Path, Path]] = []

    def short_fdopen(*args: object, **kwargs: object) -> FaultyStream:
        return FaultyStream(real_fdopen(*args, **kwargs), short_write=True)

    def recording_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "fdopen", short_fdopen)
    monkeypatch.setattr(persistence.os, "replace", recording_replace)
    persistence.atomic_write_bytes(target, b"abcdefgh")

    assert target.read_bytes() == b"abcdefgh"
    assert replacements[0][0].parent == target.parent
    assert replacements[0][1] == target


def test_atomic_write_chunks_streams_content_in_order(tmp_path: Path) -> None:
    target = tmp_path / "streamed.bin"

    persistence.atomic_write_chunks(target, (b"first", b"-", b"second"))

    assert target.read_bytes() == b"first-second"


def test_atomic_write_chunks_generator_failure_preserves_old_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "streamed.bin"
    target.write_bytes(b"old")

    def failing_chunks():
        yield b"partial"
        raise OSError("injected chunk generation failure")

    with pytest.raises(OSError, match="chunk generation"):
        persistence.atomic_write_chunks(target, failing_chunks())

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize("failure", ["write", "flush", "fsync", "replace"])
def test_atomic_failure_preserves_old_file_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    target = tmp_path / "state.bin"
    target.write_bytes(b"old")

    if failure in {"write", "flush"}:
        real_fdopen = persistence.os.fdopen

        def faulty_fdopen(*args: object, **kwargs: object) -> FaultyStream:
            raw = real_fdopen(*args, **kwargs)
            return FaultyStream(
                raw,
                fail_write=failure == "write",
                fail_flush=failure == "flush",
            )

        monkeypatch.setattr(persistence.os, "fdopen", faulty_fdopen)
    elif failure == "fsync":
        monkeypatch.setattr(
            persistence.os,
            "fsync",
            lambda _fd: (_ for _ in ()).throw(OSError("injected fsync failure")),
        )
    else:
        monkeypatch.setattr(
            persistence.os,
            "replace",
            lambda *_args: (_ for _ in ()).throw(OSError("injected replace failure")),
        )

    with pytest.raises(OSError, match="injected"):
        persistence.atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize("failure", ["write", "flush", "fsync"])
def test_append_failure_rolls_back_complete_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    target = tmp_path / "events.jsonl"
    original = b'{"existing":true}\n'
    target.write_bytes(original)

    if failure in {"write", "flush"}:
        real_open = open
        calls = 0

        def faulty_open(*args: object, **kwargs: object) -> FaultyStream:
            nonlocal calls
            calls += 1
            raw = real_open(*args, **kwargs)
            return FaultyStream(
                raw,
                fail_write=failure == "write",
                fail_flush=failure == "flush" and calls == 1,
            )

        monkeypatch.setattr(persistence, "open", faulty_open, raising=False)
    else:
        calls = 0
        real_fsync = persistence.os.fsync

        def faulty_fsync(fd: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected fsync failure")
            real_fsync(fd)

        monkeypatch.setattr(persistence.os, "fsync", faulty_fsync)

    with pytest.raises(OSError, match="injected"):
        persistence.append_jsonl(target, {"new": True})

    assert target.read_bytes() == original


def test_append_handles_short_writes_and_writes_one_json_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "events.jsonl"
    real_open = open
    holder: list[FaultyStream] = []

    def short_open(*args: object, **kwargs: object) -> FaultyStream:
        stream = FaultyStream(real_open(*args, **kwargs), short_write=True)
        holder.append(stream)
        return stream

    monkeypatch.setattr(persistence, "open", short_open, raising=False)
    persistence.append_jsonl(target, {"message": "\N{SNOWMAN}"}, fsync=False)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"message": "\N{SNOWMAN}"}
    assert holder[0].write_calls > 1


def test_append_zero_write_leaves_existing_file_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "events.jsonl"
    target.write_bytes(b"kept\n")
    real_open = open

    def zero_open(*args: object, **kwargs: object) -> FaultyStream:
        return FaultyStream(real_open(*args, **kwargs), zero_write=True)

    monkeypatch.setattr(persistence, "open", zero_open, raising=False)
    with pytest.raises(OSError, match="no progress"):
        persistence.append_jsonl(target, {"new": True}, fsync=False)
    assert target.read_bytes() == b"kept\n"


def test_append_rollback_failure_is_explicit_and_poisoned_tail_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "events.jsonl"
    original = b'{"existing":true}\n'
    target.write_bytes(original)
    real_open = open

    def faulty_open(*args: object, **kwargs: object) -> FaultyStream:
        return FaultyStream(
            real_open(*args, **kwargs),
            partial_write_failure=True,
            fail_truncate=True,
        )

    monkeypatch.setattr(persistence, "open", faulty_open, raising=False)
    with pytest.raises(persistence.PersistenceError, match="rollback also failed"):
        persistence.append_jsonl(target, {"new": True}, fsync=False)
    monkeypatch.undo()

    assert target.read_bytes().startswith(original)
    with pytest.raises(persistence.PersistenceError, match="incomplete JSONL"):
        persistence.append_jsonl(target, {"must_not_append": True}, fsync=False)


def test_append_rejects_preexisting_incomplete_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "events.jsonl"
    original = b'{"incomplete":true}'
    target.write_bytes(original)

    with pytest.raises(persistence.PersistenceError, match="incomplete JSONL"):
        persistence.append_jsonl(target, {"new": True})

    assert target.read_bytes() == original
