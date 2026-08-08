from __future__ import annotations

from pathlib import Path


def load_project_instructions(root: Path, *, max_chars: int = 30_000) -> str:
    sections: list[str] = []
    remaining = max_chars
    for name in ("FORGELOOP.md", "AGENTS.md"):
        path = root / name
        if not path.is_file() or remaining <= 0:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:remaining]
        except OSError:
            continue
        sections.append(f"## {name}\n{content}")
        remaining -= len(content)
    return "\n\n".join(sections)
