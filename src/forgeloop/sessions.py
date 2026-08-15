from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from forgeloop.config import forgeloop_home
from forgeloop.identifiers import validate_portable_identifier
from forgeloop import persistence
from forgeloop.security import SecretRedactor
from forgeloop.types import Message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    id: str
    repo: str
    created_at: str
    updated_at: str
    mode: str = "build"
    runtime: str = "local"
    provider: str = ""
    model: str = ""
    thinking: str = "auto"
    conversation: list[Message] = field(default_factory=list)
    trajectories: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    compact_count: int = 0
    last_summary: str = ""
    context_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, repo: Path) -> "Session":
        now = _now()
        return cls(uuid.uuid4().hex, str(repo.resolve()), now, now)


class SessionStore:
    def __init__(
        self,
        home: Path | None = None,
        *,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.home = (home or forgeloop_home()).expanduser().resolve()
        self.directory = self.home / "sessions"
        self.redactor = redactor or SecretRedactor()
        self._lock = threading.RLock()

    def save(self, session: Session) -> None:
        with self._lock:
            path = self.path_for(session.id)
            disk_updated_at = self._disk_updated_at(path)
            updated_at = self._next_updated_at(session.updated_at, disk_updated_at)
            serialized_session = asdict(session)
            serialized_session["updated_at"] = updated_at
            payload = self.redactor.redact(serialized_session)
            self._assert_no_known_secret(payload)
            content = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            try:
                persistence.atomic_write_text(path, content)
            except Exception:
                # A durability error can occur after replace(2). Reconcile the
                # caller-visible timestamp when the complete intended payload is
                # already installed, while still reporting the failed durability
                # boundary to the caller.
                if self._content_matches(path, content):
                    session.updated_at = updated_at
                raise
            session.updated_at = updated_at

    def load(self, session_id: str) -> Session:
        with self._lock:
            self._validate_identifier(session_id)
            matches = [item for item in self.list() if item.id.startswith(session_id)]
            if not matches:
                raise ValueError(f"Session not found: {session_id}")
            if len(matches) > 1:
                raise ValueError(f"Session id is ambiguous: {session_id}")
            return matches[0]

    def list(self) -> list[Session]:
        with self._lock:
            if not self.directory.exists():
                return []
            sessions: list[Session] = []
            for path in self.directory.glob("*.json"):
                try:
                    sessions.append(
                        Session(**json.loads(path.read_text(encoding="utf-8")))
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def path_for(self, session_id: str) -> Path:
        self._validate_identifier(session_id)
        path = self.directory / f"{session_id}.json"
        expected_parent = self.directory.resolve(strict=False)
        if expected_parent.parent != self.home:
            raise ValueError("Session directory escapes the ForgeLoop home")
        if path.resolve(strict=False).parent != expected_parent:
            raise ValueError("Session path escapes the session directory")
        return path

    @staticmethod
    def _validate_identifier(session_id: str) -> None:
        validate_portable_identifier(session_id, label="session id")

    def _assert_no_known_secret(self, payload: Any) -> None:
        rendered = json.dumps(payload, ensure_ascii=False)
        for secret in self.redactor.secrets:
            if secret and secret in rendered:
                raise ValueError("Refusing to persist a session containing an API key")

    @staticmethod
    def _disk_updated_at(path: Path) -> str:
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("updated_at", "")
        except (OSError, AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return ""
        return value if isinstance(value, str) else ""

    @staticmethod
    def _next_updated_at(*values: str) -> str:
        now = datetime.fromisoformat(_now()).astimezone(timezone.utc)
        valid: list[datetime] = []
        for value in values:
            try:
                parsed = datetime.fromisoformat(value)
            except (TypeError, ValueError):
                continue
            if parsed.tzinfo is not None:
                valid.append(parsed.astimezone(timezone.utc))
        if valid and max(valid) >= now:
            now = max(valid) + timedelta(microseconds=1)
        return now.isoformat()

    @staticmethod
    def _content_matches(path: Path, expected: str) -> bool:
        try:
            return path.read_text(encoding="utf-8") == expected
        except OSError:
            return False
