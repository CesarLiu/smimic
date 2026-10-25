"""Compose the Go2 chase Hydra config and build the UniLab environment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import unisupermimic

PACKAGE_ROOT = Path(unisupermimic.__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "conf" / "go2chase"
SCENE = PACKAGE_ROOT / "assets" / "robots" / "go2" / "scene_go2_chase.xml"
TASKS = ("flat", "obstacles", "terrain")


def scene_for(task: str) -> Path:
    return SCENE if task == "flat" else SCENE.with_name(f"scene_go2_chase_{task}.xml")


def compose(backend: str = "mjwarp", overrides: Sequence[str] = (), task: str = "flat") -> Any:
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    if task not in TASKS:
        raise ValueError(f"Go2 chase task must be one of {TASKS}, got {task!r}")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return hydra_compose(
            config_name="config", overrides=[f"task=go2_chase_{task}/{backend}", *overrides]
        )


def make_env(
    num_envs: int,
    backend: str = "mjwarp",
    *,
    play: bool = False,
    overrides: Sequence[str] = (),
    task: str = "flat",
) -> Any:
    """The Go2 chase env. ``play``: no episode time limit and no observation noise, as
    go2-supermimic's play config (domain randomization stays on, as there)."""
    from unilab.base.config_adapter import EnvConfigAdapter, create_env
    from unilab.base.process_device import (
        apply_collector_tensor_device_override,
        configure_backend_process_device,
    )
    from unilab.training import ensure_registries
    from unilab.training.tensor_runtime import select_collector_tensor_device
    from unilab.utils.device import get_default_device

    cfg = compose(backend, overrides, task)
    ensure_registries()
    device = get_default_device()
    bound = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound is not None:
        device = bound
    override: dict[str, Any] = EnvConfigAdapter(
        cfg, root_dir=Path.cwd(), algo_name="go2chase"
    ).build_task_env_cfg_override()
    override = apply_collector_tensor_device_override(
        override,
        str(cfg.training.sim_backend),
        collector_tensor_device=select_collector_tensor_device(cfg),
        learner_device=str(device),
    )
    scene = override["scene"]
    if isinstance(scene, dict):
        override["scene"] = {**scene, "model_file": str(scene_for(task))}
    else:
        scene.model_file = str(scene_for(task))
    if play:
        override["max_episode_seconds"] = 1.0e9
        observations = override["observations"]
        policy = observations["policy"]
        if isinstance(policy, dict):
            observations["policy"] = {**policy, "enable_corruption": False}
        else:
            policy.enable_corruption = False
    return create_env(cfg, num_envs=num_envs, env_cfg_override=override)


__all__ = ["CONFIG_DIR", "SCENE", "TASKS", "compose", "make_env", "scene_for"]
