"""Backend factory of the Go2 chase task (mirrors the G1 SuperMimic factory)."""

from __future__ import annotations

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.cpu_runtime import apply_cpu_confinement
from unilab.base.entity import EntityCfg
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

SUPPORTED_BACKENDS = ("mjwarp", "mujoco")


def make_go2_chase_env(
    cfg: ManagerBasedRlEnvCfg, num_envs: int = 1, backend_type: str = "mjwarp"
) -> ManagerBasedRlEnv:
    if backend_type not in SUPPORTED_BACKENDS:
        raise ValueError(f"Go2 chase supports backends {SUPPORTED_BACKENDS}; got {backend_type!r}")
    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError("make_go2_chase_env expected ManagerBasedRlEnvCfg")
    assert cfg.scene is not None
    robot_cfg = cfg.scene.entities["robot"]
    if not isinstance(robot_cfg, EntityCfg) or robot_cfg.root_body_name is None:
        raise ValueError("Go2 chase requires a robot EntityCfg with root_body_name")
    cfg.validate()
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
        return ManagerBasedRlEnv(cfg, backend, num_envs)
    except Exception:
        backend.cleanup_scene_assets()
        raise


__all__ = ["SUPPORTED_BACKENDS", "make_go2_chase_env"]
