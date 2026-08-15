from __future__ import annotations

import errno
import json
import os
import tempfile
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator


class PersistenceError(RuntimeError):
    """A persistence boundary could not establish an unambiguous file state."""


class LockTimeoutError(PersistenceError):
    """An advisory state lock could not be acquired within its bounded wait."""


@contextmanager
def advisory_file_lock(
    path: str | os.PathLike[str],
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> Iterator[None]:
    """Hold a portable OS advisory lock on a stable sidecar file.

    The sidecar is deliberately retained after release: ownership comes from the
    operating-system lock, not file existence, so an old lock file cannot become
    a stale-lock deadlock. Callers must lock the complete read-modify-write
    transaction. Acquisition is bounded and fails closed instead of proceeding
    with an unsafe concurrent update.
    """

    if timeout < 0:
        raise ValueError("lock timeout must be non-negative")
    if poll_interval <= 0:
        raise ValueError("lock poll_interval must be positive")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with open(target, "a+b", buffering=0) as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                _write_all(stream, b"\0")
            stream.seek(0)

            def acquire() -> None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def release() -> None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            def acquire() -> None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release() -> None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        while True:
            try:
                acquire()
                break
            except (BlockingIOError, OSError) as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise PersistenceError(
                        f"Failed acquiring advisory lock: {target}"
                    ) from error
                if time.monotonic() >= deadline:
                    raise LockTimeoutError(
                        f"Timed out acquiring advisory lock: {target}"
                    ) from error
                time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            try:
                release()
            except OSError as error:
                # Closing the stream releases an OS-owned advisory lock. Do not
                # replace a more useful transaction failure with cleanup noise.
                if not body_failed:
                    raise PersistenceError(
                        f"Failed releasing advisory lock: {target}"
                    ) from error


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if written is None or written <= 0:
            raise OSError("write made no progress")
        offset += written


def _sync_parent_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path.parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_chunks(path: str | os.PathLike[str], chunks: Iterable[bytes]) -> None:
    """Durably replace *path* from byte chunks without buffering the full file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb", buffering=0) as stream:
            fd = -1
            for chunk in chunks:
                _write_all(stream, chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _sync_parent_directory(target)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Durably replace *path* with *data* without exposing a partial file."""

    atomic_write_chunks(path, (data,))


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Encode and durably replace *path* without exposing a partial file."""

    atomic_write_bytes(path, text.encode(encoding))


def append_jsonl(
    path: str | os.PathLike[str],
    value: Any,
    *,
    fsync: bool = True,
) -> None:
    """Append one JSON value, rolling back the complete line on any I/O failure."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    with open(target, "a+b", buffering=0) as stream:
        stream.seek(0, os.SEEK_END)
        original_size = stream.tell()
        if original_size:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise PersistenceError(
                    f"Refusing to append to incomplete JSONL file: {target}"
                )
            stream.seek(0, os.SEEK_END)
        try:
            _write_all(stream, line)
            stream.flush()
            if fsync:
                os.fsync(stream.fileno())
        except BaseException:
            try:
                stream.truncate(original_size)
                stream.flush()
                if fsync:
                    os.fsync(stream.fileno())
            except BaseException as rollback_error:
                raise PersistenceError(
                    "JSONL append failed and rollback also failed; "
                    f"{target} may contain an incomplete record"
                ) from rollback_error
            raise
