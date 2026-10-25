"""PPO-Sim baseline (paper §4.1): PPO in the simulator on SuperMimic's task.

Shares SuperMimic's motions, observations, goal, reward, termination, domain randomization,
start-state sampling, and actor and critic networks; only the optimizer differs. PPO settings
follow BeyondMimic (mjlab G1 tracking: lr 1e-3 with adaptive KL 0.01, 24 steps per env, 5 epochs
x 4 minibatches, clip 0.2, entropy 0.005, initial std 1.0), with 8192 environments per GPU (twice
SuperMimic's, as in the paper). Checkpoints use SuperMimic's format, so the same evaluation
scores them (the deterministic policy is the Gaussian mean).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from unilab.rl.contracts.env import EnvProtocol

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    get_supermimic_mdp,
)

from . import distributed, networks
from .geometry import unflatten_state
from .networks import Critic, Policy

_LOG_FIELDS = (
    "iteration",
    "wall_h",
    "sim_steps",
    "sim_rew",
    "ep_ret_norm",
    "term_frac",
    "std",
    "kl",
    "v_loss",
    "lr",
    "it_per_s",
)


@dataclass
class PPOConfig:
    # shared with SuperMimic (same networks and inputs)
    goal_frames: int = 10
    hist: int = 10
    z_dim: int = 32
    policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    gamma: float = 0.99
    lam: float = 0.95
    amp: str = "bf16"
    seed: int = 0
    # BeyondMimic PPO
    steps_per_env: int = 24
    epochs: int = 5
    minibatches: int = 4
    lr: float = 1e-3
    desired_kl: float = 0.01
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.005
    init_std: float = 1.0
    max_grad_norm: float = 1.0
    # budget and checkpoints (wall-clock, as Fig. 4a compares methods by training time)
    iterations: int = 10**9
    max_hours: float = 18.0
    ckpt_minutes: float = 30.0
    log_every: int = 10


class PPOLearner:
    def __init__(self, env: EnvProtocol, cfg: PPOConfig, log_dir: str | Path, device=None):
        self.env, self.cfg = env, cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(env.device if device is None else device)
        self.mdp = get_supermimic_mdp(env)
        torch.manual_seed(cfg.seed + 1000 * distributed.rank())
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        self.world_size = distributed.world_size()
        goal_dim = self.mdp.goal_frame_dim * cfg.goal_frames
        self.pi = Policy(
            self.mdp.obs_dim,
            goal_dim,
            hist=cfg.hist,
            z_dim=cfg.z_dim,
            num_joints=self.mdp.num_joints,
            hidden=cfg.policy_hidden,
        ).to(self.device)
        self.log_std = nn.Parameter(
            torch.full((self.mdp.num_joints,), math.log(cfg.init_std), device=self.device)
        )
        info = self._state_info()
        features = self._critic_input(info["s"], info["f"], info["xi"])
        self.critic = Critic(features.shape[-1], hidden=cfg.critic_hidden).to(self.device)
        for module in (self.pi, self.critic):
            distributed.broadcast_module(module)
        distributed.broadcast_module(nn.ParameterList([self.log_std]))
        self.params = list(self.pi.parameters()) + [self.log_std] + list(self.critic.parameters())
        self.lr = cfg.lr
        self.opt = torch.optim.Adam(self.params, lr=self.lr)
        n = env.num_envs
        self.obs_hist = torch.zeros(n, cfg.hist, self.mdp.obs_dim, device=self.device)
        self.iteration, self.sim_steps, self.wall_time = 0, 0, 0.0
        if distributed.is_main():
            with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
                json.dump(
                    dict(asdict(cfg), algo="ppo", world_size=self.world_size), stream, indent=2
                )

    # ---- inputs (identical to SuperMimic's) ----
    def _state_info(self) -> dict[str, torch.Tensor]:
        state = self.env.state if self.env.state is not None else self.env.init_state()
        info: dict[str, torch.Tensor] = state.info[SUPERMIMIC_STATE_KEY]
        return info

    def _critic_input(self, flat_state, frames, context) -> torch.Tensor:
        state = unflatten_state(flat_state)
        goal = self.mdp.goal(frames, self.cfg.goal_frames)
        features = self.mdp.critic_features(state, frames, self.mdp.normalize_context(context))
        return torch.cat((features, goal), -1)

    def _log_prob(self, action, mean) -> torch.Tensor:
        std = self.log_std.exp()
        return (
            -0.5 * ((action - mean) / std) ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)
        ).sum(-1)

    # ---- rollout ----
    @torch.no_grad()
    def rollout(self) -> dict[str, Any]:
        T, n = self.cfg.steps_per_env, self.env.num_envs
        buf: dict[str, list[torch.Tensor]] = {
            k: [] for k in ("hist", "goal", "critic", "a", "logp", "mu", "v", "r", "done")
        }
        rewards: list[torch.Tensor] = []
        episode_returns: list[torch.Tensor] = []
        terms = dones = 0.0
        for _ in range(T):
            info = self._state_info()
            state = unflatten_state(info["s"])
            obs = self.mdp.add_noise(self.mdp.obs(state, info["a_prev"], info["xi"], info["f"]))
            fresh = info["ep_step"] == 0
            shifted = torch.cat((self.obs_hist[:, 1:], obs.unsqueeze(1)), 1)
            self.obs_hist = torch.where(
                fresh[:, None, None], obs.unsqueeze(1).expand_as(self.obs_hist), shifted
            )
            goal = self.mdp.goal(info["f"], self.cfg.goal_frames)
            critic_x = self._critic_input(info["s"], info["f"], info["xi"])
            self.pi.obs_norm.update(obs)
            self.pi.goal_norm.update(goal)
            self.critic.norm.update(critic_x)
            mean = self.pi(self.obs_hist, goal).float()
            action = mean + self.log_std.exp() * torch.randn_like(mean)
            value = self.critic(critic_x).float()
            transition = self.env.step(action).info[SUPERMIMIC_TRANSITION_KEY]
            reward = transition["r"].float().clone()
            done, term = transition["done"], transition["term"]
            truncated = done & ~term
            if bool(truncated.any()):  # bootstrap time-outs and clip ends
                next_x = self._critic_input(transition["s1"], transition["f"] + 1, transition["xi"])
                reward += self.cfg.gamma * truncated.float() * self.critic(next_x).float()
            for key, item in (
                ("hist", self.obs_hist.clone()),
                ("goal", goal),
                ("critic", critic_x),
                ("a", action),
                ("logp", self._log_prob(action, mean)),
                ("mu", mean),
                ("v", value),
                ("r", reward),
                ("done", done.float()),
            ):
                buf[key].append(item)
            self.sim_steps += n
            rewards.append(transition["r"].mean() / self.mdp.max_step_reward)
            if "ep_ret_norm" in transition:
                episode_returns.append(transition["ep_ret_norm"].mean())
            terms += float(term.float().sum())
            dones += float(done.float().sum())
        info = self._state_info()
        last_value = self.critic(self._critic_input(info["s"], info["f"], info["xi"])).float()
        out: dict[str, Any] = {k: torch.stack(v) for k, v in buf.items()}
        advantages = torch.zeros_like(out["r"])
        gae = torch.zeros(n, device=self.device)
        for t in range(T - 1, -1, -1):
            next_value = last_value if t == T - 1 else out["v"][t + 1]
            nonterminal = 1.0 - out["done"][t]
            delta = out["r"][t] + self.cfg.gamma * next_value * nonterminal - out["v"][t]
            gae = delta + self.cfg.gamma * self.cfg.lam * nonterminal * gae
            advantages[t] = gae
        out["ret"] = advantages + out["v"]
        out["adv"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        out["stats"] = {"r": rewards, "ep_ret": episode_returns, "term": terms, "done": dones}
        return out

    # ---- PPO update ----
    def update(self, batch: dict[str, Any]) -> tuple[float, float]:
        keys = ("hist", "goal", "critic", "a", "logp", "mu", "v", "ret", "adv")
        flat = {k: batch[k].flatten(0, 1) for k in keys}
        total = flat["a"].shape[0]
        kls, value_loss = [], torch.zeros((), device=self.device)
        old_std = self.log_std.exp().detach()
        for _ in range(self.cfg.epochs):
            permutation = torch.randperm(total, device=self.device)
            for indices in permutation.chunk(self.cfg.minibatches):
                mb = {k: v[indices] for k, v in flat.items()}
                mean = self.pi(mb["hist"], mb["goal"]).float()
                std = self.log_std.exp()
                logp = self._log_prob(mb["a"], mean)
                ratio = torch.exp(logp - mb["logp"])
                surrogate = -torch.min(
                    ratio * mb["adv"], ratio.clamp(1 - self.cfg.clip, 1 + self.cfg.clip) * mb["adv"]
                ).mean()
                value = self.critic(mb["critic"]).float()
                clipped = mb["v"] + (value - mb["v"]).clamp(-self.cfg.clip, self.cfg.clip)
                value_loss = torch.max((value - mb["ret"]) ** 2, (clipped - mb["ret"]) ** 2).mean()
                entropy = (self.log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
                loss = (
                    surrogate + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                )
                with torch.no_grad():  # adaptive lr on the KL between old and new Gaussians
                    kl = (
                        (
                            torch.log(std / old_std)
                            + (old_std**2 + (mb["mu"] - mean) ** 2) / (2 * std**2)
                            - 0.5
                        )
                        .sum(-1)
                        .mean()
                    )
                    kl = distributed.allreduce_mean_(kl)  # identical lr on every rank
                    kls.append(float(kl))
                    if kl > 2 * self.cfg.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < self.cfg.desired_kl / 2:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for group in self.opt.param_groups:
                        group["lr"] = self.lr
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                distributed.allreduce_grads(self.params)
                nn.utils.clip_grad_norm_(self.params, self.cfg.max_grad_norm)
                self.opt.step()
        return sum(kls) / max(len(kls), 1), float(value_loss.detach())

    # ---- checkpoints (SuperMimic format: evaluate.load_policy reads "pi" and "args") ----
    def save(self, path: Path) -> None:
        if not distributed.is_main():
            return
        args = dict(asdict(self.cfg), algo="ppo", test_hidden_dims=None)
        checkpoint = {
            "pi": self.pi.state_dict(),
            "log_std": self.log_std.detach(),
            "critic": self.critic.state_dict(),
            "iteration": self.iteration,
            "wall_time": self.wall_time,
            "sim_steps": self.sim_steps * self.world_size,
            "world_size": self.world_size,
            "args": args,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def train(self) -> dict[str, float | int]:
        start = time.time()
        last_ckpt = start
        log_path = self.log_dir / "train_log.csv" if distributed.is_main() else Path(os.devnull)
        rows = 0
        with log_path.open("a", encoding="utf-8") as log:
            log.write(",".join(_LOG_FIELDS) + "\n")
            self.save(self.log_dir / f"ckpt_{0:06d}.pt")
            log_start = time.time()
            for iteration in range(self.cfg.iterations):
                batch = self.rollout()
                kl, value_loss = self.update(batch)
                self.iteration = iteration + 1
                self.wall_time = time.time() - start
                if self.iteration % self.cfg.log_every == 0:
                    s = batch["stats"]
                    mean = lambda xs: float(sum(float(x) for x in xs) / max(len(xs), 1))  # noqa: E731
                    values = torch.tensor(
                        [
                            mean(s["r"]),
                            mean(s["ep_ret"]),
                            s["term"] / max(s["done"], 1.0),
                            float(self.log_std.exp().mean()),
                            kl,
                            value_loss,
                        ],
                        dtype=torch.float64,
                        device=self.device,
                    )
                    values = distributed.allreduce_mean_(values).tolist()
                    row = [
                        self.iteration,
                        self.wall_time / 3600,
                        self.sim_steps * self.world_size,
                        *values[:4],
                        values[4],
                        values[5],
                        self.lr,
                        self.cfg.log_every / (time.time() - log_start),
                    ]
                    log_start = time.time()
                    log.write(
                        ",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n"
                    )
                    log.flush()
                    rows += 1
                if time.time() - last_ckpt > self.cfg.ckpt_minutes * 60:
                    self.save(self.log_dir / f"ckpt_{self.iteration:06d}.pt")
                    last_ckpt = time.time()
                # rank 0 decides when the wall-clock budget is spent, so no rank leaves early
                stop = torch.tensor(
                    float(
                        distributed.is_main() and time.time() - start > self.cfg.max_hours * 3600
                    ),
                    device=self.device,
                )
                if self.world_size > 1:
                    torch.distributed.all_reduce(stop)
                if stop.item() > 0:
                    break
            self.save(self.log_dir / f"ckpt_{self.iteration:06d}.pt")
        return {"iteration": self.iteration, "wall_time": self.wall_time, "log_rows": rows}


__all__ = ["PPOConfig", "PPOLearner"]
