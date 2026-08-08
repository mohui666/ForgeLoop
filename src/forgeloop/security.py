from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SecretRedactor:
    """Small exact-secret and authorization-header redactor for persisted artifacts."""

    secrets: tuple[str, ...] = ()

    @classmethod
    def from_environment(cls, *extra: str | None) -> "SecretRedactor":
        markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        values = [
            value
            for name, value in os.environ.items()
            if value and any(marker in name.upper() for marker in markers)
        ]
        values.extend(value for value in extra if value)
        return cls(tuple(dict.fromkeys(values)))

    def redact_text(self, value: str) -> str:
        redacted = value
        for secret in self.secrets:
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        redacted = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,\"']+",
            r"\1[REDACTED]",
            redacted,
        )
        return redacted

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {str(key): self.redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        return value


SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}


def is_sensitive_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return (
        name in SENSITIVE_NAMES
        or name.endswith((".pem", ".p12", ".pfx", ".key"))
        or "/.git/" in f"/{normalized}/"
    )


@dataclass(frozen=True)
class ShellSafetyPolicy:
    """Conservative guard for commands with destructive or credential impact."""

    dangerous_patterns: tuple[str, ...] = (
        r"(?i)\brm\s+-(?:[^\s]*r[^\s]*f|[^\s]*f[^\s]*r)",
        r"(?i)\bremove-item\b[^\n;|]*(?:-recurse|-force)",
        r"(?i)\b(del|erase|rmdir|rd)\b[^\n]*(?:/s|/q)",
        r"(?i)\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*f|push\s+[^\n]*--force)",
        r"(?i)\b(format|diskpart|shutdown|restart-computer|stop-computer)\b",
        r"(?i)(?:\bset-content\b|\bout-file\b|>>?)[^\n]*(?:\.env|credentials|secrets|\.pem|\.key)",
        r"(?i)\b(?:get-childitem|dir|env|printenv|set)\b[^\n]*(?:env:|environment)",
    )

    def rejection(self, command: str) -> str | None:
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command):
                return "Command blocked by ForgeLoop safety policy; run it manually after review."
        return None
