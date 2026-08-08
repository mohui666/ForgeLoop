from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from forgeloop.evals import EvalSuite
from forgeloop.foundry import FoundryBuilder, FoundryError


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _source_repository(root: Path) -> str:
    root.mkdir()
    (root / "package").mkdir()
    (root / "tests").mkdir()
    (root / "package" / "code.py").write_text("value = 'bug'\n", encoding="utf-8")
    (root / "tests" / "test_regression.py").write_text(
        "def test_placeholder():\n    assert True\n", encoding="utf-8"
    )
    _git(root, "init", "-q", "--initial-branch=main")
    _git(root, "config", "user.name", "ForgeLoop Test")
    _git(root, "config", "user.email", "test@forgeloop.invalid")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    (root / "package" / "code.py").write_text("value = 'fixed'\n", encoding="utf-8")
    (root / "tests" / "test_regression.py").write_text(
        "from package.code import value\n\ndef test_regression():\n    assert value == 'fixed'\n",
        encoding="utf-8",
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix with regression test")
    return _git(root, "rev-parse", "HEAD")


def test_foundry_extracts_hidden_gold_and_builds_loadable_suite(
    tmp_path: Path, monkeypatch
) -> None:
    _git(tmp_path, "init", "-q", "--initial-branch=host")
    repository = tmp_path / "source"
    fix_sha = _source_repository(repository)
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    dockerfile = catalog_dir / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    catalog = {
        "schema_version": "forgeloop.foundry.v1",
        "suite_id": "real-swe-test",
        "docker": {"image": "test/image:fixed", "dockerfile": "Dockerfile"},
        "screening": {
            "inspected": 3,
            "accepted": 1,
            "rejected": 2,
            "candidate_to_valid_yield": 1 / 3,
            "rejected_candidates": [
                {
                    "repository": "https://example.invalid/one",
                    "commit": "1" * 40,
                    "reason_code": "no_test",
                    "reason": "No regression test.",
                },
                {
                    "repository": "https://example.invalid/two",
                    "commit": "2" * 40,
                    "reason_code": "too_broad",
                    "reason": "Patch is too broad.",
                },
            ],
        },
        "tasks": [
            {
                "id": "real-fix",
                "repository": str(repository),
                "fix_commit": fix_sha,
                "description": "Repair the regression.",
                "test_paths": ["tests/test_regression.py"],
                "solution_paths": ["package/code.py"],
                "verifier": {"command": "python -m pytest -q", "timeout_seconds": 30},
                "difficulty": "easy",
            }
        ],
    }
    catalog_path = catalog_dir / "tasks.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    monkeypatch.setattr(
        FoundryBuilder,
        "_build_image",
        staticmethod(lambda image, dockerfile, context: "sha256:test-image"),
    )
    monkeypatch.setattr(
        FoundryBuilder,
        "_validate",
        lambda self, task, fixture, gold_patch, image, image_id: {
            "status": "accepted",
            "repeats": self.repeats,
            "deterministic": True,
            "attempts": [],
        },
    )
    output = tmp_path / "built"
    result = FoundryBuilder(catalog_path, output, tmp_path / "cache", repeats=2).build()

    suite = EvalSuite.load(result.suite_path)
    task = suite.tasks[0]
    fixture = output / "fixtures" / "real-fix"
    gold = output / "artifacts" / "real-fix" / "gold.patch"
    assert result.accepted == 1
    assert result.filtered == 2
    assert suite.kind == "real-swe"
    assert task.source_commit == fix_sha
    assert task.source_base_sha
    assert task.docker_image == "test/image:fixed"
    assert (fixture / "package" / "code.py").read_text(
        encoding="utf-8"
    ) == "value = 'bug'\n"
    assert "assert value == 'fixed'" in (
        fixture / "tests" / "test_regression.py"
    ).read_text(encoding="utf-8")
    assert "value = 'fixed'" in gold.read_text(encoding="utf-8")
    assert not (fixture / "gold.patch").exists()


def test_foundry_rejects_inconsistent_screening_records() -> None:
    screening = {
        "inspected": 2,
        "accepted": 1,
        "rejected": 1,
        "candidate_to_valid_yield": 0.5,
        "rejected_candidates": [],
    }

    with pytest.raises(FoundryError, match="Rejected candidate records"):
        FoundryBuilder._validate_screening(screening, task_count=1)
