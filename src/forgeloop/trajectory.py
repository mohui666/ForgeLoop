from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from forgeloop.security import SecretRedactor

SCHEMA_VERSION = "forgeloop.trajectory.v2"


class TrajectoryStore:
    """Append-only JSONL event log designed for replay and later dataset export."""

    def __init__(
        self,
        output_dir: Path,
        *,
        run_id: str | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.redactor = redactor or SecretRedactor()
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = output_dir / f"{timestamp}-{self.run_id[:8]}.jsonl"
        self._sequence = 0

    def append(self, event_type: str, payload: Any) -> None:
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": self.redactor.redact(self._normalize(payload)),
        }
        self._sequence += 1
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def redact_text(self, value: str) -> str:
        return self.redactor.redact_text(value)

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
