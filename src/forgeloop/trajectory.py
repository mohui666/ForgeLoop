from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO

from forgeloop.security import SecretRedactor

SCHEMA_VERSION = "forgeloop.trajectory.v2"


class TrajectoryStore:
    """Append-only JSONL event log designed for replay and dataset export.

    Every successful append is flushed to the operating system. ``fsync_every``
    optionally adds a disk durability boundary every N successful events; zero
    (the default) avoids an fsync per long-horizon event while retaining the
    existing flush-level durability.  Append failures are rolled back before
    the sequence is reused.  If rollback itself fails, the store is poisoned
    and rejects later appends rather than risking an ambiguous trajectory.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        run_id: str | None = None,
        redactor: SecretRedactor | None = None,
        fsync_every: int = 0,
    ) -> None:
        if fsync_every < 0:
            raise ValueError("fsync_every must be non-negative")
        self.run_id = run_id or uuid.uuid4().hex
        self.redactor = redactor or SecretRedactor()
        self.fsync_every = fsync_every
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_fingerprint = uuid.uuid5(uuid.NAMESPACE_OID, self.run_id).hex[:8]
        file_nonce = uuid.uuid4().hex[:8]
        self.path = output_dir / (f"{timestamp}-{run_fingerprint}-{file_nonce}.jsonl")
        self._sequence = 0
        self._append_lock = threading.Lock()
        self._poisoned = False

    def append(self, event_type: str, payload: Any) -> None:
        with self._append_lock:
            if self._poisoned:
                raise RuntimeError(
                    "trajectory store is poisoned by an incomplete rollback"
                )
            event = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "payload": self.redactor.redact(self._normalize(payload)),
            }
            record = (
                json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            with self._open_append() as handle:
                handle.seek(0, os.SEEK_END)
                original_size = handle.tell()
                try:
                    self._write_all(handle, record)
                    handle.flush()
                    next_sequence = self._sequence + 1
                    if self.fsync_every and next_sequence % self.fsync_every == 0:
                        self._fsync(handle)
                except BaseException:
                    try:
                        self._rollback(handle, original_size)
                    except BaseException:
                        self._poisoned = True
                    raise
            self._sequence += 1

    def _open_append(self) -> BinaryIO:
        return self.path.open("a+b")

    @staticmethod
    def _fsync(handle: BinaryIO) -> None:
        os.fsync(handle.fileno())

    @staticmethod
    def _write_all(handle: BinaryIO, record: bytes) -> None:
        remaining = memoryview(record)
        while remaining:
            written = handle.write(remaining)
            if written is None or written <= 0:
                raise OSError("trajectory append made no write progress")
            remaining = remaining[written:]

    @staticmethod
    def _rollback(handle: BinaryIO, original_size: int) -> None:
        handle.seek(original_size)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())

    def redact_text(self, value: str) -> str:
        return self.redactor.redact_text(value)

    @property
    def durability_policy(self) -> dict[str, int | bool]:
        """Serializable settings suitable for run provenance."""
        return {"flush_each_append": True, "fsync_every": self.fsync_every}

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if is_dataclass(value):
            return cls._normalize(asdict(value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._normalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)
