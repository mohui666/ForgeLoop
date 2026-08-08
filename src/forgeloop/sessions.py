from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forgeloop.config import forgeloop_home
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

    def save(self, session: Session) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        session.updated_at = _now()
        payload = self.redactor.redact(asdict(session))
        self._assert_no_known_secret(payload)
        self.path_for(session.id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load(self, session_id: str) -> Session:
        matches = [item for item in self.list() if item.id.startswith(session_id)]
        if not matches:
            raise ValueError(f"Session not found: {session_id}")
        if len(matches) > 1:
            raise ValueError(f"Session id is ambiguous: {session_id}")
        return matches[0]

    def list(self) -> list[Session]:
        if not self.directory.exists():
            return []
        sessions: list[Session] = []
        for path in self.directory.glob("*.json"):
            try:
                sessions.append(Session(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def path_for(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def _assert_no_known_secret(self, payload: Any) -> None:
        rendered = json.dumps(payload, ensure_ascii=False)
        for secret in self.redactor.secrets:
            if secret and secret in rendered:
                raise ValueError("Refusing to persist a session containing an API key")
