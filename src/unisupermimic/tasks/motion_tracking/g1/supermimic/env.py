"""SuperMimic Manager-Based environment and backend factory."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_cpu_confinement
from unilab.base.entity import EntityCfg
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.base.torch_env import TorchEnvState
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from unisim.backend.base import SimBackend

from unisupermimic.rl.algos.supermimic.geometry import sample_push
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    SuperMimicMdpProtocol,
)

from .manager_terms import (
    SuperMimicCommand,
    make_current_info,
    make_transition_info,
    merge_current_info,
)
from .motion import CONTROL_DT
from .robot import COLLISION_GEOM_NAMES

PUSH_TERM = "supermimic_push"


class SuperMimicManagerBasedRlEnv(ManagerBasedRlEnv):
    """Publish detached SuperMimic transitions around the Manager reset boundary.

    Optional push injection: kicks are applied right after a transition is recorded, so its
    ``s1`` stays the clean physics target, and each transition carries the kick that produced
    its start state (``push``), which the world-model fit replays exactly.
    """

    def __init__(self, cfg: ManagerBasedRlEnvCfg, backend: SimBackend, num_envs: int):
        if cfg.scale_rewards_by_dt:
            raise ValueError(
                "SuperMimic requires scale_rewards_by_dt=False; rewards already include dt"
            )
        if not math.isclose(cfg.ctrl_dt, CONTROL_DT, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"SuperMimic requires ctrl_dt={CONTROL_DT}")
        super().__init__(cfg, backend, num_envs)
        # backend classes live in unisim.backend.<name>.backend
        module = type(backend).__module__.split(".")
        self.backend_type = module[2] if module[:2] == ["unisim", "backend"] else ".".join(module)
        commands: list[SuperMimicCommand] = []
        for name in self.command_manager.active_terms:
            command = self.command_manager.get_term(name)
            if isinstance(command, SuperMimicCommand):
                commands.append(command)
        if len(commands) != 1:
            raise ValueError("SuperMimic requires exactly one SuperMimicCommand term")
        self._supermimic_command = commands[0]
        self._pre_action: dict[str, Any] | None = None
        command = self._supermimic_command
        self._push_scale = 0.0 if command.cfg.eval_mode else float(command.cfg.push_scale)
        self._push_bound = False
        if self._push_scale > 0.0:
            self._bind_push()
        lo, hi = (int(round(float(value) / CONTROL_DT)) for value in command.cfg.push_interval_s)
        self._push_steps = (lo, max(lo, hi))
        # kick that produced each env's current state (zero after a reset or without pushes)
        self._pending_push = torch.zeros((num_envs, 6), dtype=torch.float32, device=self.device)
        self._push_timer = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._reset_push_timer(torch.ones(num_envs, dtype=torch.bool, device=self.device))

    def _bind_push(self) -> None:
        # Linear kicks only (see PUSH_RANGE). Only mjwarp applies them exactly: the MuJoCo host
        # backend's interval velocity delta also perturbs other state, which would corrupt the
        # logged transitions, so pushes fail closed elsewhere.
        if self.backend_type != "mjwarp":
            raise NotImplementedError(
                f"SuperMimic push injection requires the mjwarp backend, got {self.backend_type!r}"
            )
        self._supermimic_command.robot.bind_root_linear_velocity_delta(term_name=PUSH_TERM)
        self._push_bound = True

    def _apply_linear_push(self, push: torch.Tensor, ids: torch.Tensor) -> None:
        if bool((push[ids, 3:6] != 0).any()):
            raise ValueError("SuperMimic push injection supports linear root kicks only")
        self._supermimic_command.robot.apply_root_velocity_delta_to_sim(
            push[ids, 0:3], None, env_ids=ids, term_name=PUSH_TERM
        )
        self.scene._invalidate_state_reads()
        self._refresh_tensor_reads_after_mutation()

    def _reset_push_timer(self, mask: torch.Tensor) -> None:
        lo, hi = self._push_steps
        draw = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
        self._push_timer = torch.where(mask, draw, self._push_timer)

    def apply_root_push(self, push: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Add world-frame root-velocity kicks ``push`` (num_envs, 6) to envs in ``mask``.

        Refreshes the tensor reads and returns the new current SuperMimic state info (also
        published to ``state.info``), so the next observation sees the post-kick state.
        """
        if not self._push_bound:
            self._bind_push()
        ids = mask.nonzero(as_tuple=False).flatten()
        if ids.numel() > 0:
            self._apply_linear_push(push, ids)
        current = self._current_info()
        if self._state is not None:
            self._state.info[SUPERMIMIC_STATE_KEY] = current
        return current

    def _scheduled_push(self, done: torch.Tensor) -> None:
        """Interval kicks (every push_interval_s) on envs that are not about to reset."""
        self._push_timer -= 1
        due = (self._push_timer <= 0) & ~done
        push = sample_push(self.num_envs, self._push_scale, self.device) * due.unsqueeze(-1)
        if bool(due.any()):
            self._apply_linear_push(push, due.nonzero(as_tuple=False).flatten())
        self._pending_push = push
        self._reset_push_timer(self._push_timer <= 0)

    @property
    def supermimic_mdp(self) -> SuperMimicMdpProtocol:
        return self._supermimic_command.mdp

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> torch.Tensor:
        command = self._supermimic_command
        self._pre_action = {
            "state": tuple(value.detach().clone() for value in command.state()),
            "frames": command.f.detach().clone(),
            "context": command.xi.detach().clone(),
            "previous_action": command.a_prev.detach().clone(),
            "ep_step": command.ep_step.detach().clone(),
            "action": actions.detach().clone(),
            "push": self._pending_push.clone(),
        }
        return super().apply_action(actions, state)

    def _update_state_in_read_phase(self, state: TorchEnvState) -> TorchEnvState:
        updated = super()._update_state_in_read_phase(state)
        if self._pre_action is None:
            raise RuntimeError("SuperMimic update_state requires a preceding apply_action snapshot")
        command = self._supermimic_command
        transition = make_transition_info(
            self._pre_action["state"],
            command.state(),
            self._pre_action["action"],
            self._pre_action["previous_action"],
            self._pre_action["context"],
            updated.reward,
            self._pre_action["frames"],
            self._pre_action["ep_step"],
            updated.terminated,
            updated.truncated,
        )
        transition.update(command.last_episode_stats)
        transition["push"] = self._pre_action["push"]
        if self._push_scale > 0.0:  # after s1 was read: the kick shapes the next start state
            self._scheduled_push(updated.terminated | updated.truncated)
        updated.info[SUPERMIMIC_TRANSITION_KEY] = transition
        updated.info[SUPERMIMIC_STATE_KEY] = self._current_info()
        self._pre_action = None
        return updated

    def _current_info(self) -> dict[str, torch.Tensor]:
        command = self._supermimic_command
        return make_current_info(
            command.state(),
            command.f,
            command.xi,
            command.a_prev,
            command.ep_step,
        )

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if env_ids is None:
            reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            ids = resolve_reset_env_ids(
                env_ids,
                num_envs=self.num_envs,
                device=self.device,
                owner="SuperMimicManagerBasedRlEnv.reset",
            )
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[ids] = True
        obs, info = super().reset(env_ids, seed=seed, options=options)
        assert self._state is not None
        self._pending_push[reset_mask] = 0.0
        self._reset_push_timer(reset_mask)
        current = self._current_info()
        previous = self._state.info.get(SUPERMIMIC_STATE_KEY)
        if previous is not None:
            current = merge_current_info(previous, current, reset_mask)
        self._state.info[SUPERMIMIC_STATE_KEY] = current
        info[SUPERMIMIC_STATE_KEY] = current
        return obs, info

    def reset_to_frames(
        self, frames: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset every environment to its requested motion-library frame."""
        self._supermimic_command.request_frames(frames)
        return self.reset()


SUPPORTED_BACKENDS = ("mjwarp", "mujoco", "motrix")


def make_supermimic_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mjwarp",
) -> SuperMimicManagerBasedRlEnv:
    """Construct SuperMimic through the common public backend factory path."""
    if backend_type not in SUPPORTED_BACKENDS:
        supported = ", ".join(repr(backend) for backend in SUPPORTED_BACKENDS)
        raise ValueError(f"make_supermimic_env supports backends {supported}; got {backend_type!r}")
    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError("make_supermimic_env expected ManagerBasedRlEnvCfg")
    assert cfg.scene is not None
    robot_cfg = cfg.scene.entities["robot"]
    if not isinstance(robot_cfg, EntityCfg):
        raise TypeError("SuperMimic robot scene entity must be an EntityCfg")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
    if robot_cfg.root_body_name is None:
        raise ValueError("SuperMimic requires robot.root_body_name")
    if robot_cfg.geom_names is None:
        cfg.scene.entities["robot"] = replace(robot_cfg, geom_names=COLLISION_GEOM_NAMES)
    cfg.validate()
    if cfg.fixed_model_variants is not None:
        raise ValueError("SuperMimic does not support fixed model variants")
    apply_cpu_confinement(cfg.cpu_ids)
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=True,
        base_name=robot_cfg.root_body_name,
        **env_backend_kwargs(cfg),
    )
    try:
        return SuperMimicManagerBasedRlEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


__all__ = ["SuperMimicManagerBasedRlEnv", "make_supermimic_env"]
