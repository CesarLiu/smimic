"""PPO-Sim baseline entrypoint (single GPU, or torchrun via scripts/train_ddp.py --ppo).

  unisupermimic-train-ppo training.log_dir=runs/ppo_g1 [ppo.max_hours=18 ...]
  python -m unisupermimic.scripts.train_ppo training.log_dir=runs/ppo_g1

Network sizes and inputs come from ``algo.*`` (identical to SuperMimic's); PPO settings from
``ppo.*``. Checkpoints are scored by the SuperMimic evaluator (unisupermimic-eval --ckpt ...).
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from unilab.base.config_adapter import create_env
from unilab.base.process_device import configure_backend_process_device
from unilab.training import ensure_registries, resolve_run_log_dir
from unilab.training.experiment import write_run_config_snapshot
from unilab.training.seed import apply_configured_training_seed
from unilab.utils.device import get_default_device

from unisupermimic.rl.algos.supermimic import distributed
from unisupermimic.rl.algos.supermimic.ppo import PPOConfig, PPOLearner
from unisupermimic.scripts.train_supermimic import _close_env, build_env_override
from unisupermimic.sim2sim import extract_contract_snapshot

SHARED = (
    "goal_frames",
    "hist",
    "z_dim",
    "policy_hidden",
    "critic_hidden",
    "gamma",
    "lam",
    "amp",
    "seed",
)


def build_ppo_config(cfg: DictConfig) -> PPOConfig:
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    ppo = OmegaConf.to_container(cfg.ppo, resolve=True)
    assert isinstance(algo, dict) and isinstance(ppo, dict)
    names = {field.name for field in fields(PPOConfig)}
    values = {key: algo[key] for key in SHARED if key in algo}
    values.update({key: value for key, value in ppo.items() if key in names})
    for name in ("policy_hidden", "critic_hidden"):
        values[name] = tuple(values[name])
    return PPOConfig(**values)


def main(cfg: DictConfig) -> None:
    distributed.init_from_env()
    if distributed.world_size() > 1:
        base_seed = int(cfg.algo.seed) if cfg.algo.seed is not None else 0
        OmegaConf.update(cfg, "algo.seed", base_seed)
        OmegaConf.update(cfg, "env.seed", base_seed + 1000 * distributed.rank(), force_add=True)
    ensure_registries()
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = get_default_device()
    bound_device = configure_backend_process_device(str(cfg.training.sim_backend), device)
    if bound_device is not None:
        device = bound_device
    env_override = build_env_override(cfg, eval_mode=False, learner_device=str(device))
    chosen = str(resolve_run_log_dir(cfg)) if distributed.is_main() else None
    chosen = distributed.broadcast_object(chosen)  # rank 0 picks the run directory
    assert chosen is not None
    log_dir = Path(chosen)
    if distributed.is_main():
        write_run_config_snapshot(
            log_dir,
            run_metadata={
                "algo": "ppo",
                "task": str(cfg.training.task_name),
                "sim_backend": str(cfg.training.sim_backend),
                "device": str(device),
            },
            full_cfg=cfg,
            contract_snapshot=extract_contract_snapshot(cfg),
        )
    env = None
    try:
        env = create_env(cfg, num_envs=int(cfg.ppo.num_envs), env_cfg_override=env_override)
        learner = PPOLearner(env, build_ppo_config(cfg), log_dir, device=str(device))
        summary = learner.train()
        if distributed.is_main():
            print(json.dumps(summary, indent=2))
    finally:
        if env is not None:
            _close_env(env)


@hydra.main(version_base="1.3", config_path="../conf/supermimic", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()
