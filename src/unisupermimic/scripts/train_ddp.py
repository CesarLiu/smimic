"""Multi-GPU SuperMimic training entrypoint for torchrun.

  torchrun --standalone --nproc_per_node=2 -m unisupermimic.scripts.train_ddp \
      training.log_dir=runs/my_run [hydra overrides ...]

Each rank sees only its own GPU (CUDA_VISIBLE_DEVICES is narrowed to the LOCAL_RANK-th visible
device before torch or warp initialize), so the simulator and learner keep using cuda:0 as in a
single-GPU run. Ranks own their environments and replay buffers; the learner averages gradients
and normalizer statistics (rl/algos/supermimic/distributed.py).
"""

from __future__ import annotations

import os
import sys


def _narrow_visible_devices() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if os.environ.get("SUPERMIMIC_SAME_GPU") == "1":  # debugging: every rank on one GPU (gloo)
        os.environ.setdefault("SUPERMIMIC_DIST_BACKEND", "gloo")
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(64)]
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]


def main() -> None:
    _narrow_visible_devices()
    from unisupermimic.scripts.train_supermimic import hydra_main

    hydra_main()


if __name__ == "__main__":
    sys.exit(main())
