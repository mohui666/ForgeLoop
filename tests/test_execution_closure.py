from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from forgeloop.agent import AgentLoop, RunMode, RunStatus
from forgeloop.budget import BudgetLimits
from forgeloop.delivery import GitPatchDelivery
from forgeloop.hybrid_controller import (
    ControllerPolicyConfig,
    ControllerPolicyResult,
    HybridControllerV14ExplicitCloseout,
    HybridDecision,
)
from forgeloop.runtime import LocalRuntime
from forgeloop.tools import build_default_tools
from forgeloop.trajectory import TrajectoryStore
from forgeloop.types import ModelResponse, ModelUsage, ToolCall
from forgeloop.workspace import Workspace


class AlwaysAdvisoryPolicy:
    config = ControllerPolicyConfig.load()

    def decide(self, snapshot):
        del snapshot
        return ControllerPolicyResult(HybridDecision("implement", "edit"), 0.0, 1, 1)


class ScriptedProvider:
    model_id = "test/execution-closure"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages, tools, *, timeout_seconds):
        del messages, tools
        assert timeout_seconds > 0
        self.calls += 1
        return next(self.responses)


def _response(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        tool_calls=calls,
        usage=ModelUsage(10, 5, 0.0),
        finish_reason="tool_calls",
    )


def _repo(path: Path) -> Workspace:
    (path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (path / "b.py").write_text("B = 1\n", encoding="utf-8")
    tests = path / "tests"
    tests.mkdir()
    (tests / "test_values.py").write_text(
        "from a import A\nfrom b import B\n\ndef test_values():\n    assert (A, B) == (2, 2)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@invalid"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)
    return Workspace(path)


def test_edit_validation_review_explicit_finish_and_patch_delivery(
    tmp_path: Path,
) -> None:
    workspace = _repo(tmp_path)
    base = workspace.git_snapshot().head
    runtime = LocalRuntime()
    provider = ScriptedProvider(
        [
            _response(
                ToolCall(
                    "patch-a",
                    "apply_patch",
                    {"path": "a.py", "old_text": "A = 1", "new_text": "A = 2"},
                ),
                ToolCall(
                    "patch-b",
                    "apply_patch",
                    {"path": "b.py", "old_text": "B = 1", "new_text": "B = 2"},
                ),
            ),
            _response(
                ToolCall(
                    "validate",
                    "validate",
                    {"command": "python -m pytest -q", "timeout_seconds": 30},
                )
            ),
            _response(ToolCall("diff", "git_diff", {})),
            _response(ToolCall("decision", "read_file", {"path": "a.py"})),
            _response(
                ToolCall(
                    "finish",
                    "finish",
                    {
                        "status": "completed",
                        "summary": "Updated both values",
                        "evidence": "Validation passed and the worktree was reviewed",
                    },
                )
            ),
        ]
    )
    controller = HybridControllerV14ExplicitCloseout(AlwaysAdvisoryPolicy())
    agent = AgentLoop(
        provider,
        build_default_tools(workspace, runtime),
        workspace,
        TrajectoryStore(tmp_path / ".forgeloop" / "runs", run_id="closure"),
        BudgetLimits(
            max_steps=10,
            max_model_calls=10,
            max_tool_calls=20,
            max_seconds=120,
        ),
        controller=controller,
        delivery=GitPatchDelivery(runtime),
    )

    result = agent.run(RunMode.TASK, "Set both values to two")

    assert result.status is RunStatus.COMPLETED
    assert result.stop_reason == "model_finish_tool"
    assert provider.calls == 5
    assert result.budget["usage"]["total_tokens"] == 75
    last_pass = controller.summary()["execution_closure"]["last_pass"]
    assert last_pass["model_calls_after"] == 3
    assert last_pass["tokens_after"] == 45
    assert result.delivery is not None
    assert result.delivery["has_patch"] is True
    assert result.delivery["committed"] is True
    assert result.delivery["clean"] is True
    assert result.delivery["patch_sha256"]
    patch = subprocess.run(
        ["git", "diff", "--binary", base, "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "a.py" in patch and "b.py" in patch
    assert ".forgeloop" not in patch


def test_completed_delivery_rejects_an_empty_base_to_head_patch(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    delivery = GitPatchDelivery(LocalRuntime())
    delivery.start(workspace)

    result = delivery.deliver(workspace, RunStatus.COMPLETED)

    assert result.ok is False
    assert result.status == "delivery_failed"
    assert result.has_patch is False
    assert "no base-to-HEAD patch" in result.detail


def test_failed_delivery_preserves_a_real_partial_patch(tmp_path: Path) -> None:
    workspace = _repo(tmp_path)
    base = workspace.git_snapshot().head
    delivery = GitPatchDelivery(LocalRuntime())
    delivery.start(workspace)
    (tmp_path / "a.py").write_text("A = 9\n", encoding="utf-8")

    result = delivery.deliver(workspace, RunStatus.FAILED)

    assert result.ok is True
    assert result.has_patch is True
    assert result.committed is True
    patch = subprocess.run(
        ["git", "diff", "--binary", base, "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert result.patch_sha256 == hashlib.sha256(patch.strip().encode()).hexdigest()
    assert "+A = 9" in patch


def test_delivery_hashes_complete_patch_larger_than_runtime_output_cap(
    tmp_path: Path,
) -> None:
    workspace = _repo(tmp_path)
    base = workspace.git_snapshot().head
    runtime = LocalRuntime(max_output_chars=40_000)
    delivery = GitPatchDelivery(runtime)
    delivery.start(workspace)
    (tmp_path / "large.py").write_text(
        "VALUES = [\n" + "".join(f"    {index},\n" for index in range(10_000)) + "]\n",
        encoding="utf-8",
    )

    result = delivery.deliver(workspace, RunStatus.COMPLETED)

    patch = subprocess.run(
        ["git", "diff", "--binary", base, "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout.rstrip()
    assert len(patch) > runtime.max_output_chars
    assert result.ok is True
    assert result.patch_bytes == len(patch)
    assert result.patch_sha256 == hashlib.sha256(patch).hexdigest()


def test_delivery_adopts_a_model_commit_onto_the_delivery_branch(
    tmp_path: Path,
) -> None:
    workspace = _repo(tmp_path)
    delivery = GitPatchDelivery(LocalRuntime())
    delivery.start(workspace)
    (tmp_path / "a.py").write_text("A = 7\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "model commit"], cwd=tmp_path, check=True)

    result = delivery.deliver(workspace, RunStatus.COMPLETED)

    assert result.ok is True
    assert result.has_patch is True
    assert result.committed is False
    assert result.branch is not None
    assert result.branch.startswith("forgeloop/deepswe-delivery-")
    patch = subprocess.run(
        ["git", "diff", "--binary", delivery.base_sha, "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert result.patch_sha256 == hashlib.sha256(patch.strip().encode()).hexdigest()
