"""Data-parallel helpers for multi-GPU SuperMimic training (torchrun).

Each rank owns its simulator environments and replay buffer; gradients are averaged before
clipping and running-normalizer statistics are reduced globally, so every rank holds identical
networks. With a single process every helper is a no-op.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def is_main() -> bool:
    return rank() == 0


def init_from_env() -> None:
    """Join the torchrun process group when launched with WORLD_SIZE > 1."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or dist.is_initialized():
        return
    backend = os.environ.get("SUPERMIMIC_DIST_BACKEND") or (
        "nccl" if torch.cuda.is_available() else "gloo"
    )
    dist.init_process_group(backend)


def allreduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    if world_size() > 1:
        dist.all_reduce(tensor)
        tensor /= world_size()
    return tensor


def allreduce_grads(parameters) -> None:
    """Average gradients across ranks in one flat buffer."""
    if world_size() == 1:
        return
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= world_size()
    offset = 0
    for grad in grads:
        count = grad.numel()
        grad.copy_(flat[offset : offset + count].view_as(grad))
        offset += count


def broadcast_module(module: torch.nn.Module) -> None:
    """Copy rank 0's parameters and buffers to every rank."""
    if world_size() == 1:
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, 0)


def broadcast_object(value):
    """Rank 0's Python object on every rank."""
    if world_size() == 1:
        return value
    holder = [value]
    dist.broadcast_object_list(holder, 0)
    return holder[0]


def global_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(count, mean, biased variance) of ``x`` rows pooled over all ranks."""
    count = torch.tensor(float(x.shape[0]), device=x.device)
    total = x.sum(0)
    squares = (x * x).sum(0)
    packed = torch.cat((total, squares, count.reshape(1)))
    dist.all_reduce(packed)
    dim = x.shape[-1]
    count = packed[-1]
    mean = packed[:dim] / count
    var = (packed[dim : 2 * dim] / count - mean * mean).clamp_min(0.0)
    return count, mean, var


__all__ = [
    "allreduce_grads",
    "allreduce_mean_",
    "broadcast_module",
    "broadcast_object",
    "global_moments",
    "init_from_env",
    "is_main",
    "rank",
    "world_size",
]
