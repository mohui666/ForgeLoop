from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Any

from forgeloop.models.base import ModelProviderError

PROVIDER_RELIABILITY_SCHEMA_VERSION = "forgeloop.provider-reliability.v1"


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """Bounded retry policy for one incomplete provider request."""

    max_attempts: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 8.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.2
    attempt_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("provider retry max_attempts must be between 1 and 10")
        if self.initial_backoff_seconds < 0:
            raise ValueError("provider retry initial_backoff_seconds cannot be negative")
        if self.max_backoff_seconds < 0:
            raise ValueError("provider retry max_backoff_seconds cannot be negative")
        if self.backoff_multiplier < 1:
            raise ValueError("provider retry backoff_multiplier must be at least 1")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("provider retry jitter_ratio must be between 0 and 1")
        if self.attempt_timeout_seconds <= 0:
            raise ValueError("provider retry attempt_timeout_seconds must be positive")

    @classmethod
    def from_config(cls, value: Any) -> ProviderRetryPolicy:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("serving_config.provider_reliability must be an object")
        supported = {
            "max_attempts",
            "initial_backoff_seconds",
            "max_backoff_seconds",
            "backoff_multiplier",
            "jitter_ratio",
            "attempt_timeout_seconds",
        }
        unsupported = sorted(set(value) - supported)
        if unsupported:
            raise ValueError(
                "Unsupported provider reliability fields: " + ", ".join(unsupported)
            )
        return cls(**value)

    @classmethod
    def from_provider(cls, provider: Any) -> ProviderRetryPolicy:
        identity = getattr(provider, "policy_identity", None)
        serving = getattr(identity, "serving_config", None)
        if not isinstance(serving, dict) and isinstance(identity, dict):
            serving = identity.get("serving_config")
        config = serving.get("provider_reliability") if isinstance(serving, dict) else None
        return cls.from_config(config)

    def backoff_seconds(
        self, failed_attempt: int, *, random_value: float | None = None
    ) -> float:
        base = min(
            self.max_backoff_seconds,
            self.initial_backoff_seconds
            * (self.backoff_multiplier ** max(0, failed_attempt - 1)),
        )
        sample = random.random() if random_value is None else random_value
        jittered = base * (1 + self.jitter_ratio * ((2 * sample) - 1))
        return round(max(0.0, min(self.max_backoff_seconds, jittered)), 6)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": PROVIDER_RELIABILITY_SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True)
class ProviderErrorClassification:
    error_type: str
    retryable: bool
    retry_reason: str
    status_code: int | None = None


_STATUS_PATTERN = re.compile(
    r"(?i)(?:status(?:_code)?|http(?: status)?|error(?: code)?)"
    r"\s*[:=]?\s*([45]\d\d)\b"
)
_PERMANENT_CLASS_MARKERS = (
    "authentication",
    "permissiondenied",
    "badrequest",
    "notfound",
    "unprocessable",
    "contextwindow",
    "contentpolicy",
    "unsupportedparam",
)
_TIMEOUT_CLASS_MARKERS = ("timeout", "readtimeout", "connecttimeout")
_CONNECTION_CLASS_MARKERS = (
    "apiconnection",
    "connecterror",
    "connectionerror",
    "connectionreset",
    "remoteprotocol",
    "readerror",
    "networkerror",
    "brokenpipe",
)


def classify_provider_error(error: BaseException) -> ProviderErrorClassification:
    """Classify conservatively: retry only explicit transient evidence."""

    if isinstance(error, ModelProviderError) and error.retryable is not None:
        return ProviderErrorClassification(
            error_type=error.error_type,
            retryable=error.retryable,
            retry_reason=error.retry_reason or "provider_classified",
            status_code=error.status_code,
        )

    error_type = type(error).__name__
    class_names = " ".join(
        item.__name__.lower() for item in type(error).__mro__ if hasattr(item, "__name__")
    )
    message = f"{error} {getattr(error, 'details', '')}".lower()
    status_code = _status_code(error, message)

    # Explicit SDK request/authentication classes win over incidental numbers in
    # exception text. Retrying a deterministic request failure is never useful.
    if any(marker in class_names for marker in _PERMANENT_CLASS_MARKERS):
        return ProviderErrorClassification(
            error_type, False, "permanent_request", status_code
        )

    hard_limit_markers = (
        "insufficient balance",
        "insufficient quota",
        "insufficient_quota",
        "billing limit",
        "billing hard limit",
        "credit balance",
    )
    if any(marker in message for marker in hard_limit_markers):
        return ProviderErrorClassification(
            error_type, False, "quota_or_billing", status_code
        )

    if status_code in {408, 409, 429}:
        reason = {
            408: "transient_timeout",
            409: "transient_conflict",
            429: "rate_limit",
        }[status_code]
        return ProviderErrorClassification(error_type, True, reason, status_code)
    if status_code is not None and 500 <= status_code <= 599:
        return ProviderErrorClassification(
            error_type, True, "provider_5xx", status_code
        )
    if status_code is not None and 400 <= status_code <= 499:
        reason = (
            "authentication"
            if status_code in {401, 403}
            else "permanent_provider_4xx"
        )
        return ProviderErrorClassification(error_type, False, reason, status_code)

    if isinstance(error, TimeoutError) or any(
        marker in class_names for marker in _TIMEOUT_CLASS_MARKERS
    ):
        return ProviderErrorClassification(error_type, True, "transient_timeout")
    if isinstance(error, ConnectionError) or any(
        marker in class_names for marker in _CONNECTION_CLASS_MARKERS
    ):
        return ProviderErrorClassification(error_type, True, "connection_failure")

    markers = (
        ("incomplete_stream", ("incomplete stream", "incomplete chunk", "incomplete read")),
        (
            "ssl_eof",
            (
                "ssl eof",
                "unexpected_eof_while_reading",
                "unexpected eof while reading",
                "eof occurred in violation of protocol",
            ),
        ),
        (
            "connection_reset",
            (
                "connection reset",
                "connection aborted",
                "server disconnected",
                "peer closed connection",
                "broken pipe",
            ),
        ),
        ("transient_timeout", ("timed out", "timeout")),
        ("rate_limit", ("rate limit", "too many requests")),
        (
            "provider_overloaded",
            (
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "internal server error",
                "provider overloaded",
            ),
        ),
    )
    for reason, needles in markers:
        if any(needle in message for needle in needles):
            return ProviderErrorClassification(error_type, True, reason)

    permanent_markers = (
        "invalid api key",
        "authentication",
        "unauthorized",
        "permission denied",
        "invalid parameter",
        "bad request",
        "context length",
        "unsupported parameter",
    )
    if any(marker in message for marker in permanent_markers):
        return ProviderErrorClassification(error_type, False, "permanent_request")
    return ProviderErrorClassification(error_type, False, "unclassified_permanent")


def normalize_provider_error(error: BaseException) -> ModelProviderError:
    if isinstance(error, ModelProviderError):
        classification = classify_provider_error(error)
        if error.retryable is not None:
            return error
        return ModelProviderError(
            str(error),
            details=error.details,
            error_type=classification.error_type,
            status_code=classification.status_code,
            retryable=classification.retryable,
            retry_reason=classification.retry_reason,
        )
    classification = classify_provider_error(error)
    return ModelProviderError(
        str(error) or classification.error_type,
        details=f"{classification.error_type}: {error}",
        error_type=classification.error_type,
        status_code=classification.status_code,
        retryable=classification.retryable,
        retry_reason=classification.retry_reason,
    )


def _status_code(error: BaseException, message: str) -> int | None:
    candidates = (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    )
    for value in candidates:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    match = _STATUS_PATTERN.search(message)
    return int(match.group(1)) if match else None


__all__ = [
    "PROVIDER_RELIABILITY_SCHEMA_VERSION",
    "ProviderErrorClassification",
    "ProviderRetryPolicy",
    "classify_provider_error",
    "normalize_provider_error",
]
