"""Register the Go2 chase task with UniLab."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from unisupermimic.go2chase.env.factory import SUPPORTED_BACKENDS, make_go2_chase_env

TASK_NAMES = ("Go2ChaseFlat", "Go2ChaseObstacles", "Go2ChaseTerrain")

for task_name in TASK_NAMES:
    registry.register_env_config(task_name, ManagerBasedRlEnvCfg)
    for backend in SUPPORTED_BACKENDS:
        registry.register_env(task_name, make_go2_chase_env, sim_backend=backend)

__all__ = ["TASK_NAMES"]
