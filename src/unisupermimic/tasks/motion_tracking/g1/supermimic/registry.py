"""Register the G1 SuperMimic task with UniLab."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg

from unisupermimic.tasks.motion_tracking.g1.supermimic.env import make_supermimic_env

TASK_NAME = "G1SuperMimic"

registry.register_env_config(TASK_NAME, ManagerBasedRlEnvCfg)
registry.register_env(TASK_NAME, make_supermimic_env, sim_backend="mjwarp")

__all__ = ["TASK_NAME"]
