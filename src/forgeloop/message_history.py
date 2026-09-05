"""Request history and independent canonical conversation snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from forgeloop.types import Message


class AgentMessageHistory(list[Message]):
    """Cache-stable request history with a separate canonical audit history.

    A committed compaction replaces only the request-facing history. The full
    append-only conversation remains available in ``canonical`` for audit and
    provenance. Subsequent messages extend both histories, keeping the compacted
    request prefix byte-stable until another compaction epoch is necessary.
    """

    def __init__(self, messages: Sequence[Message]) -> None:
        initial = deepcopy(list(messages))
        super().__init__(deepcopy(initial))
        self.canonical: list[Message] = initial
        self.compaction_epochs = 0

    def append(self, message: Message) -> None:
        canonical = deepcopy(message)
        self.canonical.append(canonical)
        super().append(deepcopy(message))

    def commit_compaction(self, messages: Sequence[Message]) -> None:
        self[:] = deepcopy(list(messages))
        self.compaction_epochs += 1
