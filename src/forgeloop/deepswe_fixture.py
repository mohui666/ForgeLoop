from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pier.agents.base import BaseAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext

from forgeloop.agent_types import RunStatus
from forgeloop.deepswe import PierRuntime, PierWorkspace
from forgeloop.delivery import GitPatchDelivery
from forgeloop.trajectory import TrajectoryStore


class DeterministicPatchCollectionAgent(BaseAgent):
    """No-model agent used only to exercise Pier's patch delivery contract."""

    @staticmethod
    def name() -> str:
        return "forgeloop-deterministic-patch-fixture"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(
            "git config user.name 'ForgeLoop Fixture' && "
            "git config user.email 'fixture@forgeloop.local'",
            cwd="/app",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del instruction
        loop = asyncio.get_running_loop()
        result = await asyncio.to_thread(self._run_sync, environment, loop)
        context.n_input_tokens = 0
        context.n_cache_tokens = 0
        context.n_output_tokens = 0
        context.cost_usd = 0.0
        context.n_agent_steps = 1
        context.metadata = {
            "forgeloop": {
                "terminal_state": "completed",
                "stop_reason": "deterministic_fixture_complete",
                "model_calls": 0,
                "tool_calls": 1,
                "usage_complete": True,
                "cost_sources": ["deterministic_zero"],
                "delivery": result.to_dict(),
            }
        }

    def _run_sync(
        self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop
    ) -> Any:
        runtime = PierRuntime(environment, loop)
        workspace = PierWorkspace(self.logs_dir / "fixture-workspace", runtime)
        runtime.start(workspace.root)
        delivery = GitPatchDelivery(runtime, commit_message="Fixture delivery commit")
        delivery.start(workspace)
        changed = runtime.run(
            "printf 'PATCHED_BY_FORGELOOP\\n' > fixture_source.txt",
            workspace.root,
            30,
        )
        if changed.exit_code != 0:
            raise RuntimeError(changed.stderr or changed.stdout)
        result = delivery.deliver(workspace, RunStatus.COMPLETED)
        trajectory = TrajectoryStore(
            Path(self.logs_dir) / "forgeloop-trajectories",
            run_id="deterministic-patch-collection",
        )
        trajectory.append("patch_delivery", result.to_dict())
        if not result.ok:
            raise RuntimeError(result.detail or result.status)
        return result


__all__ = ["DeterministicPatchCollectionAgent"]
