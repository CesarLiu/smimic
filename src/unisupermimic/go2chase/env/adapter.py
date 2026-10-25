"""The env interface go2-supermimic's trainer expects, on top of the UniLab env.

``step`` returns ``(obs, reward, terminated, truncated, extras)`` with ``obs["actor"]`` the
stacked actor observation and ``extras["log"]`` the episode logs (mjlab key names:
``Episode_Termination/fell_over``, ``Metrics/target/tags``, ...). ``robot`` is the
read-plan ``RobotView``; ``term`` the target command.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from omegaconf import OmegaConf

from .builder import compose, make_env
from .command import TargetCommand

ACTOR_HISTORY = 5


class Go2ChaseEnv:
    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        backend: str = "mjwarp",
        *,
        task: str = "flat",
        play: bool = False,
        overrides: Sequence[str] = (),
    ):
        overrides = [f"env.seed={seed}", *overrides]
        self.task = task
        self.env = make_env(num_envs, backend, play=play, overrides=overrides, task=task)
        term = self.env.command_manager.get_term("target")
        assert isinstance(term, TargetCommand)
        self.term = term
        self.robot = term.robot
        self.num_envs = num_envs
        self.device = torch.device(self.env.device)
        self.step_dt = float(self.env.step_dt)
        cfg = compose(backend, overrides, task)
        rewards = OmegaConf.to_container(cfg.reward, resolve=True)
        assert isinstance(rewards, dict)
        self.reward_weights = {
            str(name): float(term_cfg["weight"]) for name, term_cfg in rewards.items()
        }

    @property
    def common_step_counter(self) -> int:
        """Curriculum step (UniLab's own counter plus the command's offset)."""
        return self.term.progress_step

    @common_step_counter.setter
    def common_step_counter(self, value: int) -> None:
        self.term.step_offset = int(value) - int(getattr(self.env, "common_step_counter", 0))

    @property
    def action_term(self) -> Any:
        return self.env.action_manager.get_term("joint_pos")

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        obs, info = self.env.reset()
        return {"actor": obs["obs"]}, info

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        state = self.env.step(actions.to(dtype=torch.float32).contiguous())
        extras = {"log": state.info.get("log", {})}
        return {"actor": state.obs["obs"]}, state.reward, state.terminated, state.truncated, extras

    def close(self) -> None:
        self.env.close()


__all__ = ["ACTOR_HISTORY", "Go2ChaseEnv"]
