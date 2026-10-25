"""Fixed-size device-resident replay laid out as ``(time, environment)``."""

from __future__ import annotations

import torch


def _make_fields(
    num_joints: int,
    context_dim: int,
    num_feet: int = 0,
) -> dict[str, tuple[int | tuple[()], torch.dtype]]:
    state_dim = 13 + 2 * num_joints
    fields: dict[str, tuple[int | tuple[()], torch.dtype]] = {
        "s": (state_dim, torch.float32),
        "s1": (state_dim, torch.float32),
        "a": (num_joints, torch.float32),
        "a_prev": (num_joints, torch.float32),
        "xi": (context_dim, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
        "term": ((), torch.bool),  # hard termination (a fall); done also covers timeouts
        # root-velocity kick between the previous transition's s1 and this s (zero without pushes)
        "push": (6, torch.float32),
    }
    if num_feet:  # foot heights of s1 (rare-event world-model sampling)
        fields["feet"] = (num_feet, torch.float32)
    return fields


FIELDS = _make_fields(29, 33)
WINDOW_FIELDS = ("s", "s1", "a", "xi", "done", "term", "push", "mask")


class ReplayBuffer:
    def __init__(
        self,
        T: int,
        N: int,
        device: str | torch.device,
        num_joints: int = 29,
        context_dim: int = 33,
        num_feet: int = 0,
    ):
        self.T = T
        self.N = N
        self.device = torch.device(device)
        self.fields = _make_fields(num_joints, context_dim, num_feet)
        self.window_keys = tuple(k for k in (*WINDOW_FIELDS, "feet") if k in self.fields)
        self.data = {}
        for key, (shape, dtype) in self.fields.items():
            field_shape = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[key] = torch.zeros(field_shape, dtype=dtype, device=self.device)
        self.ptr = 0
        self.size = 0

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        for key in self.fields:
            if key in ("push", "term") and key not in transition:  # optional fields
                self.data[key][self.ptr] = 0
                continue
            self.data[key][self.ptr] = transition[key]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time indices (zero is oldest) to storage rows."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int) -> dict[str, torch.Tensor]:
        """Sample ``k``-step windows and mask transitions after episode ends."""
        if not 1 <= k <= self.size:
            raise ValueError(f"window length must be in [1, {self.size}], got {k}")
        start = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = start.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        envs = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, envs] for key in self.window_keys}
        done = out["done"].float()
        mask = torch.cumprod(
            torch.cat((torch.ones(B, 1, device=self.device), 1 - done[:, :-1]), -1),
            -1,
        )
        out["mask"] = mask
        return out

    def sample_starts(
        self,
        B: int,
        hist: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample imagination starts and oldest-first, boundary-padded histories."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if hist < 1:
            raise ValueError(f"history length must be positive, got {hist}")
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        history_rows = self._row(age.unsqueeze(-1) - back)
        history_envs = env.unsqueeze(-1).expand_as(history_rows)
        start = {key: self.data[key][row, env] for key in ("s", "a_prev", "xi", "f")}
        history = {
            key: self.data[key][history_rows, history_envs] for key in ("s", "a_prev", "xi", "f")
        }
        return start, history


class WindowArchive:
    """Ring buffer of world-model training windows (rare events kept past the replay's
    horizon, as in go2-supermimic): ``add`` stores the selected windows, ``sample`` draws
    uniformly from what is stored."""

    def __init__(self, capacity: int, device: str | torch.device):
        if capacity <= 0:
            raise ValueError("archive capacity must be positive")
        self.capacity = capacity
        self.device = torch.device(device)
        self.data: dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.size = 0

    def add(self, windows: dict[str, torch.Tensor]) -> None:
        count = next(iter(windows.values())).shape[0]
        if count == 0:
            return
        if not self.data:  # allocate on first use, shaped like the windows
            self.data = {
                key: torch.zeros(
                    (self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.device
                )
                for key, value in windows.items()
            }
        count = min(count, self.capacity)
        rows = (self.ptr + torch.arange(count, device=self.device)) % self.capacity
        for key, value in windows.items():
            self.data[key][rows] = value[-count:]
        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count: int) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty archive")
        rows = torch.randint(0, self.size, (count,), device=self.device)
        return {key: value[rows] for key, value in self.data.items()}


__all__ = ["FIELDS", "WINDOW_FIELDS", "ReplayBuffer", "WindowArchive"]
