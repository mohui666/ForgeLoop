from __future__ import annotations

from pathlib import Path

from forgeloop.runtime import CommandResult, Runtime


def detect_workflow(repo: Path, kind: str) -> str | None:
    choices: dict[str, list[tuple[str, str]]] = {
        "test": [
            ("pyproject.toml", "uv run pytest"),
            ("pytest.ini", "pytest"),
            ("package.json", "npm test"),
            ("Cargo.toml", "cargo test"),
            ("go.mod", "go test ./..."),
        ],
        "lint": [
            ("pyproject.toml", "uv run ruff check ."),
            ("package.json", "npm run lint"),
            ("Cargo.toml", "cargo clippy --all-targets"),
        ],
        "build": [
            ("package.json", "npm run build"),
            ("Cargo.toml", "cargo build"),
            ("go.mod", "go build ./..."),
            ("pyproject.toml", "uv build"),
        ],
    }
    for marker, command in choices[kind]:
        if (repo / marker).exists():
            return command
    return None


def run_workflow(
    runtime: Runtime, repo: Path, kind: str, timeout: float
) -> CommandResult:
    command = detect_workflow(repo, kind)
    if not command:
        raise ValueError(f"No {kind} workflow detected for this repository")
    return runtime.run(command, repo, timeout)
