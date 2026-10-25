"""Two-process CPU (gloo) test of data-parallel SuperMimic training."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))  # FakeEnv lives next to this file


def _flat(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).float() for t in module.state_dict().values()])


def _worker(rank: int, world: int, init_file: str, out_dir: str) -> None:
    from test_supermimic_learner import FakeEnv, _config

    from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        learner = SuperMimicLearner(FakeEnv(), _config(), Path(out_dir) / f"rank{rank}")
        learner.collect(3)
        learner.wm_update()
        learner.policy_update()
        data = learner.buffer.data["a"][: learner.buffer.size].clone()
        # results go through files: tensors queued from a process that has exited are gone
        torch.save(
            (
                rank,
                _flat(learner.wm),
                _flat(learner.pi),
                _flat(learner.critic),
                data,
                learner.base_lr,
                (Path(out_dir) / f"rank{rank}" / "args.json").exists(),
            ),
            Path(out_dir) / f"result{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_two_ranks_keep_identical_networks_on_distinct_data(tmp_path) -> None:
    os.environ.pop("WORLD_SIZE", None)
    world = 2
    mp.start_processes(
        _worker,
        args=(world, str(tmp_path / "init"), str(tmp_path)),
        nprocs=world,
        join=True,
        start_method="spawn",
    )
    results = [torch.load(tmp_path / f"result{r}.pt", weights_only=False) for r in range(world)]
    (_, wm0, pi0, v0, data0, lr0, args0), (_, wm1, pi1, v1, data1, lr1, args1) = results
    # gradients and normalizer statistics are averaged: identical networks, buffers included
    assert torch.equal(wm0, wm1)
    assert torch.equal(pi0, pi1)
    assert torch.equal(v0, v1)
    # ranks seed their action noise differently, so they collected different data
    assert not torch.equal(data0, data1)
    # Table A.5: lr scaled by sqrt(world size); only rank 0 writes run files
    assert math.isclose(lr0, 1e-3 * math.sqrt(2)) and lr0 == lr1
    assert args0 and not args1
