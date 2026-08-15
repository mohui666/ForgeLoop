from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

import pytest

from forgeloop import persistence
from forgeloop.config import ConfigStore, GlobalConfig
from forgeloop.model_capabilities import ModelCache


def _temporary_files(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def _inject_atomic_failure(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    if stage == "write":
        monkeypatch.setattr(
            persistence,
            "_write_all",
            lambda stream, data: (_ for _ in ()).throw(OSError("write failed")),
        )
    elif stage == "fsync":
        monkeypatch.setattr(
            persistence.os,
            "fsync",
            lambda fd: (_ for _ in ()).throw(OSError("fsync failed")),
        )
    elif stage == "replace":
        monkeypatch.setattr(
            persistence.os,
            "replace",
            lambda source, target: (_ for _ in ()).throw(OSError("replace failed")),
        )
    else:  # pragma: no cover - test helper contract
        raise AssertionError(stage)


@pytest.mark.parametrize("stage", ["write", "fsync", "replace"])
def test_config_store_atomic_failure_preserves_old_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    store = ConfigStore(tmp_path / "home")
    store.save(GlobalConfig(provider="deepseek", model="old-model"))
    _inject_atomic_failure(monkeypatch, stage)

    with pytest.raises(OSError, match=f"{stage} failed"):
        store.save(GlobalConfig(provider="custom", model="new-model"))

    assert store.load().model == "old-model"
    assert _temporary_files(store.path) == []


@pytest.mark.parametrize("stage", ["write", "fsync", "replace"])
def test_model_cache_atomic_failure_preserves_old_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    cache = ModelCache(tmp_path)
    cache.remember_manual("custom", "https://models.invalid", "old-model")
    _inject_atomic_failure(monkeypatch, stage)

    with pytest.raises(OSError, match=f"{stage} failed"):
        cache.update(
            "custom",
            "https://models.invalid",
            [("new-model", {"id": "new-model"})],
        )

    assert ModelCache(tmp_path).models("custom", "https://models.invalid") == [
        "old-model"
    ]
    assert _temporary_files(cache.path) == []


def test_config_and_model_cache_successful_json_remains_secret_free(
    tmp_path: Path,
) -> None:
    store = ConfigStore(tmp_path)
    store.save(
        GlobalConfig(
            provider="custom",
            model="private-model",
            provider_configs={"custom": {"api_base": "https://models.invalid"}},
        )
    )
    cache = ModelCache(tmp_path)
    cache.update(
        "custom",
        "https://models.invalid",
        [
            (
                "private-model",
                {
                    "id": "private-model",
                    "context_window": 128_000,
                    "api_key": "must-not-be-persisted",
                },
            )
        ],
    )

    assert store.load().model == "private-model"
    assert cache.models("custom", "https://models.invalid") == ["private-model"]
    assert "must-not-be-persisted" not in cache.path.read_text(encoding="utf-8")
    assert json.loads(store.path.read_text(encoding="utf-8"))["provider"] == "custom"


def test_model_cache_concurrent_manual_updates_do_not_lose_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = ModelCache(tmp_path)
    original_load: Callable[[], dict[str, object]] = cache._load

    def yielding_load() -> dict[str, object]:
        payload = original_load()
        time.sleep(0.01)
        return payload

    monkeypatch.setattr(cache, "_load", yielding_load)
    count = 12
    start = threading.Barrier(count)

    def remember(index: int) -> None:
        start.wait()
        cache.remember_manual("custom", "https://models.invalid", f"manual-{index:02d}")

    threads = [
        threading.Thread(target=remember, args=(index,)) for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert cache.models("custom", "https://models.invalid") == [
        f"manual-{index:02d}" for index in range(count)
    ]
    assert _temporary_files(cache.path) == []
