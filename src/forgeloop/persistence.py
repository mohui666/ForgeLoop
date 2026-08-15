from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO


class PersistenceError(RuntimeError):
    """A persistence boundary could not establish an unambiguous file state."""


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


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Durably replace *path* with *data* without exposing a partial file."""

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
            _write_all(stream, data)
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
