from __future__ import annotations

import re


_PORTABLE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def validate_portable_identifier(value: str, *, label: str = "identifier") -> str:
    """Return a path- and Git-ref-safe portable identifier or raise ValueError."""

    if (
        not isinstance(value, str)
        or not _PORTABLE_IDENTIFIER.fullmatch(value)
        or value.upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            f"Invalid {label}: expected 1-128 ASCII letters, digits, '_' or '-'"
        )
    return value
