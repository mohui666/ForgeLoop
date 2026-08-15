from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SENSITIVE_VALUE_KEYS = {
    "api_key",
    "apikey",
    "access_key",
    "access_token",
    "refresh_token",
    "auth_token",
    "token",
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "x-auth-token",
    "client_secret",
    "provider_credential",
    "provider_credentials",
    "password",
    "credential",
    "credentials",
}

_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b((?:[a-z][a-z0-9]*[_-])*(?:api[_-]?key|access[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"provider[_-]?credentials?|password|token|secret|credentials?))"
    r"(\s*[:=]\s*)([\"']?)[^\s,;\"'&]{6,}"
)
_PROVIDER_TOKENS = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{20,}"),
)
_WINDOWS_HOME = re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s\"']+")
_POSIX_HOME = re.compile(r"/(?:home|Users)/[^/\s\"']+")
_URL_USERINFO = re.compile(r"(?i)(https?://)[^/\s:@]+:[^@/\s]+@")


def is_sensitive_value_key(key: str) -> bool:
    return key.strip().lower() in SENSITIVE_VALUE_KEYS


@dataclass(frozen=True)
class SecretRedactor:
    """Credential-aware recursive redactor for persisted ForgeLoop artifacts."""

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
        redacted = _CREDENTIAL_ASSIGNMENT.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
            redacted,
        )
        for pattern in _PROVIDER_TOKENS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return _URL_USERINFO.sub(r"\1[REDACTED]@", redacted)

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {
                str(key): (
                    "[REDACTED]"
                    if is_sensitive_value_key(str(key))
                    else self.redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        return value


class EvidenceSanitizer:
    """Shared credential and local-path sanitizer for persisted evidence."""

    def __init__(
        self,
        *,
        redactor: SecretRedactor | None = None,
        local_roots: Iterable[Path | str] = (),
    ) -> None:
        self.redactor = redactor or SecretRedactor.from_environment()
        roots = [Path.home(), Path.cwd(), *[Path(root) for root in local_roots]]
        self.local_roots = tuple(
            sorted(
                {str(root.expanduser().resolve()) for root in roots},
                key=len,
                reverse=True,
            )
        )

    def sanitize(self, value: Any, *, key: str | None = None) -> Any:
        if key and is_sensitive_value_key(key):
            return "[REDACTED]"
        if isinstance(value, str):
            return self.sanitize_text(value)
        if isinstance(value, dict):
            return {
                str(item_key): self.sanitize(item, key=str(item_key))
                for item_key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item) for item in value]
        return value

    def sanitize_text(self, value: str) -> str:
        redacted = self.redactor.redact_text(value)
        for root in self.local_roots:
            variants = {root, root.replace("\\", "/"), root.replace("/", "\\")}
            for variant in sorted(variants, key=len, reverse=True):
                redacted = re.sub(
                    re.escape(variant),
                    "[LOCAL_ROOT]",
                    redacted,
                    flags=re.IGNORECASE,
                )
        redacted = _WINDOWS_HOME.sub("[USER_HOME]", redacted)
        return _POSIX_HOME.sub("[USER_HOME]", redacted)


SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credential",
    "credentials",
    "credential.json",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
    "private-key",
    "private_key",
}
SENSITIVE_SUFFIXES = (".pem", ".p12", ".pfx", ".key")


def is_sensitive_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return (
        name in SENSITIVE_NAMES
        or name.endswith(SENSITIVE_SUFFIXES)
        or "/.git/" in f"/{normalized}/"
    )


def sensitive_path_python_source(function_name: str = "is_sensitive_path") -> str:
    """Render the canonical path policy for isolated Python runtimes.

    Local/Docker search helpers and Pier execute in separate Python processes, so
    they cannot import the host package reliably. Generate their predicate from
    the same constants used by :func:`is_sensitive_path` to prevent policy drift.
    """

    names = repr(tuple(sorted(SENSITIVE_NAMES)))
    suffixes = repr(SENSITIVE_SUFFIXES)
    return (
        f"def {function_name}(path):\n"
        "    normalized = str(path).replace('\\\\', '/').lower()\n"
        "    name = normalized.rsplit('/', 1)[-1]\n"
        f"    return (name in {names} or name.endswith({suffixes}) "
        "or '/.git/' in f'/{normalized}/')\n"
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
