from __future__ import annotations

import subprocess
from pathlib import Path

from forgeloop.runtime import LocalRuntime
from forgeloop.tools.base import ToolRegistry
from forgeloop.tools.builtin import GitDiffTool
from forgeloop.workspace import Workspace


def test_registry_binds_run_base_to_existing_and_late_tools(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=ForgeLoop Tests",
            "-c",
            "user.email=tests@forgeloop.local",
            "commit",
            "-qm",
            "base",
        ],
        cwd=tmp_path,
        check=True,
    )
    workspace = Workspace(tmp_path)
    base = workspace.git_snapshot().head
    assert base is not None
    existing = GitDiffTool(workspace, LocalRuntime())
    registry = ToolRegistry([existing])

    registry.bind_run_context(base_head=base)

    assert existing.run_base_head == base
    late_registry = ToolRegistry()
    late_registry.bind_run_context(base_head=base)
    late = GitDiffTool(workspace, LocalRuntime())
    late_registry.register(late)
    assert late.run_base_head == base
