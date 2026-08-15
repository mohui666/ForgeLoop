from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO

import pytest

from forgeloop.trajectory import (
    DEFAULT_CRITICAL_EVENT_TYPES,
    DEFAULT_FSYNC_EVERY,
    TrajectoryStore,
)


def _events(store: TrajectoryStore) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in store.path.read_text(encoding="utf-8").splitlines()
    ]


class _FaultyHandle:
    def __init__(
        self,
        handle: BinaryIO,
        *,
        partial_write: bool = False,
        short_write: bool = False,
        zero_write: bool = False,
        fail_flush: bool = False,
        fail_rollback: bool = False,
    ) -> None:
        self._handle = handle
        self._partial_write = partial_write
        self._short_write = short_write
        self._zero_write = zero_write
        self._fail_flush = fail_flush
        self._fail_rollback = fail_rollback

    def __enter__(self) -> _FaultyHandle:
        self._handle.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        return self._handle.__exit__(*args)

    def __getattr__(self, name: str) -> object:
        return getattr(self._handle, name)

    def write(self, data: bytes | memoryview) -> int:
        if self._partial_write:
            self._partial_write = False
            self._handle.write(data[: max(1, len(data) // 2)])
            raise OSError("injected partial write")
        if self._short_write:
            self._short_write = False
            chunk = data[: max(1, len(data) // 2)]
            return self._handle.write(chunk)
        if self._zero_write:
            self._zero_write = False
            return 0
        return self._handle.write(data)

    def flush(self) -> None:
        if self._fail_flush:
            self._fail_flush = False
            raise OSError("injected flush failure")
        self._handle.flush()

    def truncate(self, size: int | None = None) -> int:
        if self._fail_rollback:
            raise OSError("injected rollback failure")
        return self._handle.truncate(size)


def _inject_handle(store: TrajectoryStore, **faults: bool) -> None:
    original_open = store._open_append
    used = False

    def open_once() -> BinaryIO:
        nonlocal used
        handle = original_open()
        if used:
            return handle
        used = True
        return _FaultyHandle(handle, **faults)  # type: ignore[return-value]

    store._open_append = open_once  # type: ignore[method-assign]


@pytest.mark.parametrize("fault", ["partial_write", "fail_flush"])
def test_failed_append_is_removed_and_sequence_is_reused(
    tmp_path: Path, fault: str
) -> None:
    store = TrajectoryStore(tmp_path, run_id="failure")
    _inject_handle(store, **{fault: True})

    with pytest.raises(OSError, match="injected"):
        store.append("failed", {"value": 0})
    store.append("recovered", {"value": 1})

    events = _events(store)
    assert [(event["sequence"], event["type"]) for event in events] == [
        (0, "recovered")
    ]


def test_short_write_is_completed_as_one_valid_record(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, run_id="short")
    _inject_handle(store, short_write=True)

    store.append("complete", {"value": 1})

    events = _events(store)
    assert [(event["sequence"], event["type"]) for event in events] == [(0, "complete")]


def test_zero_progress_write_fails_and_reuses_sequence(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, run_id="zero")
    _inject_handle(store, zero_write=True)

    with pytest.raises(OSError, match="no write progress"):
        store.append("failed", {})
    store.append("recovered", {})

    assert [event["sequence"] for event in _events(store)] == [0]


def test_fsync_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, run_id="fsync", fsync_every=1)
    attempts = 0

    def fail_once(handle: BinaryIO) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected fsync failure")

    store._fsync = fail_once  # type: ignore[method-assign]

    with pytest.raises(OSError, match="fsync"):
        store.append("failed", {})
    store.append("recovered", {})

    assert [event["sequence"] for event in _events(store)] == [0]
    assert attempts == 2


def test_fsync_interval_counts_only_successful_appends(tmp_path: Path) -> None:
    store = TrajectoryStore(
        tmp_path,
        run_id="batch",
        fsync_every=2,
        critical_event_types=frozenset(),
    )
    synced_after: list[int] = []
    store._fsync = lambda handle: synced_after.append(store._sequence + 1)  # type: ignore[method-assign]

    for value in range(5):
        store.append("event", {"value": value})

    assert synced_after == [2, 4]
    assert store.durability_policy == {
        "flush_each_append": True,
        "fsync_every": 2,
        "critical_event_types": [],
    }


def test_default_policy_bounds_unsynced_noncritical_events(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, run_id="default-cadence")
    synced_after: list[int] = []
    store._fsync = lambda handle: synced_after.append(store._sequence + 1)  # type: ignore[method-assign]

    for value in range(DEFAULT_FSYNC_EVERY * 2 + 1):
        store.append("observation", {"value": value})

    assert synced_after == [DEFAULT_FSYNC_EVERY, DEFAULT_FSYNC_EVERY * 2]


@pytest.mark.parametrize("event_type", sorted(DEFAULT_CRITICAL_EVENT_TYPES))
def test_critical_events_sync_immediately(tmp_path: Path, event_type: str) -> None:
    store = TrajectoryStore(tmp_path, run_id=f"critical-{event_type}")
    synced_after: list[int] = []
    store._fsync = lambda handle: synced_after.append(store._sequence + 1)  # type: ignore[method-assign]

    store.append("observation", {})
    store.append(event_type, {})

    assert synced_after == [2]


def test_zero_cadence_still_syncs_critical_events(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, run_id="critical-only", fsync_every=0)
    synced_after: list[int] = []
    store._fsync = lambda handle: synced_after.append(store._sequence + 1)  # type: ignore[method-assign]

    store.append("observation", {})
    store.append("run_finished", {})

    assert synced_after == [2]


def test_critical_fsync_failure_rolls_back_and_reuses_sequence(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, run_id="critical-failure", fsync_every=0)
    attempts = 0

    def fail_once(handle: BinaryIO) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected critical fsync failure")

    store._fsync = fail_once  # type: ignore[method-assign]

    with pytest.raises(OSError, match="critical fsync"):
        store.append("run_started", {})
    store.append("run_started", {"recovered": True})

    assert [event["sequence"] for event in _events(store)] == [0]
    assert attempts == 2


def test_concurrent_appends_are_complete_unique_and_contiguous(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, run_id="threads")

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(
            executor.map(
                lambda value: store.append("event", {"value": value}), range(200)
            )
        )

    events = _events(store)
    assert len(events) == 200
    assert [event["sequence"] for event in events] == list(range(200))
    assert {event["payload"]["value"] for event in events} == set(range(200))  # type: ignore[index]


def test_stores_created_together_never_share_a_trajectory_path(tmp_path: Path) -> None:
    first = TrajectoryStore(tmp_path, run_id="same-run")
    second = TrajectoryStore(tmp_path, run_id="same-run")

    first.append("first", {})
    second.append("second", {})

    assert first.path != second.path
    assert [event["type"] for event in _events(first)] == ["first"]
    assert [event["type"] for event in _events(second)] == ["second"]


def test_rollback_failure_poisons_store(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, run_id="poisoned")
    _inject_handle(store, partial_write=True, fail_rollback=True)

    with pytest.raises(OSError, match="partial write"):
        store.append("failed", {})
    with pytest.raises(RuntimeError, match="poisoned"):
        store.append("must-not-write", {})


def test_negative_fsync_interval_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TrajectoryStore(tmp_path, fsync_every=-1)


def test_invalid_critical_event_type_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        TrajectoryStore(tmp_path, critical_event_types=frozenset({""}))
