"""SuperMimic training over the injected environment and MDP protocols."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from unilab.rl.contracts.env import EnvProtocol

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
    get_supermimic_mdp,
)
from unisupermimic.rl.logging import SuperMimicLogger

from . import distributed, networks
from .geometry import add_push, sample_push, unflatten_state
from .networks import Critic, Policy, WorldModel, mlp
from .replay import ReplayBuffer, WindowArchive


@dataclass
class SuperMimicConfig:
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    hist: int = 10
    z_dim: int = 32
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_every: int = 50
    log_backend: str = "tensorboard"
    seed: int = 0
    amp: str = "bf16"
    compile: bool = True
    test_hidden_dims: tuple[int, ...] | None = None
    # imagined root-velocity kicks (analytic push injection; 0 = off, the paper)
    imag_push_rate: float = 0.0
    imag_push_scale: float = 1.0
    # multi-GPU (torchrun): scale lr by sqrt(world size), as the paper does for 2 GPUs
    ddp_lr_scale: bool = True
    # ---- training techniques from go2-supermimic; every default keeps the paper's setup ----
    # rare-event world-model batches: a share of each batch is the highest-rarity windows out
    # of 4x as many candidates (contact changes, flight, falls), another share comes from an
    # archive of rare windows (flight, hard vertical base motion, falls) kept past the replay
    wm_prioritized_frac: float = 0.0
    wm_archive_frac: float = 0.0
    archive_size: int = 50_000
    archive_keep_prob: float = 0.02
    contact_height: float = 0.10  # foot height [m] counted as contact (mdp.foot_heights)
    hard_vz: float = 0.6  # |base vertical speed| [m/s] of a take-off or landing
    # temporally correlated exploration: the first explore_frac of the envs collect with
    # Ornstein-Uhlenbeck action noise (stationary std explore_noise, mean reversion
    # explore_theta 1/s) instead of white noise of std action_noise
    explore_frac: float = 0.0
    explore_noise: float = 0.3
    explore_theta: float = 2.0
    # stability: actor lr multiplier; linear actor-lr ramp whenever the actor starts or
    # restarts; collapse guard (two consecutive log windows above collapse_fall_rate falls
    # per 1000 env steps roll actor and critic back to the last snapshot and halve the lr)
    actor_lr_scale: float = 1.0
    actor_ramp_iters: int = 0
    collapse_fall_rate: float = 0.0
    collapse_snapshot_every: int = 500
    # warm start from another run's checkpoint (weights only, schedules start afresh); a
    # policy whose observation gained terms (appended to each frame) starts exactly as the
    # old one, the new inputs with zero weight. A re-initialised world model refits for
    # wm_rewarmup_iters, a fresh critic trains alone for critic_warmup_iters.
    init_from: str | None = None
    reset_critic: bool = False
    critic_warmup_iters: int = 500
    wm_rewarmup_iters: int = 1500
    # data-collection driver: another policy checkpoint (SuperMimic or PPO-Sim, same task
    # inputs) acts in the last driver_frac of the envs, so the world model sees its skills;
    # the share fades linearly to zero between the driver_fade iterations
    driver_ckpt: str | None = None
    driver_frac: float = 0.0
    driver_fade: tuple[int, int] = (1_000_000_000, 1_000_000_000)


_LOG_FIELDS = (
    "iteration",
    "wall_h",
    "sim_steps",
    "wm_loss",
    "wm_err8",
    "pi_loss",
    "v_loss",
    "imag_rew",
    "pi_gradnorm",
    "imag_survival",
    "sim_rew",
    "ep_len",
    "ep_ret_norm",
    "term_frac",
    "it_per_s",
    "lr",
)


def _empty_stats() -> dict[str, list[torch.Tensor]]:
    return {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _same_shapes(module: nn.Module, state: dict[str, torch.Tensor]) -> bool:
    own = module.state_dict()
    return own.keys() == state.keys() and all(own[k].shape == state[k].shape for k in own)


@torch.no_grad()
def expand_policy_inputs(old: dict[str, torch.Tensor], policy: Policy) -> None:
    """Load ``old`` into ``policy`` whose per-frame observation gained terms appended at the
    end of each frame: old inputs keep their weights and normalizer statistics, new ones
    start with zero weight (mean 0, variance 1), so the policy acts exactly as before."""
    state = policy.state_dict()
    old_obs, new_obs = old["obs_norm.mean"].numel(), state["obs_norm.mean"].numel()
    if new_obs <= old_obs:
        raise ValueError(f"policy observation {old_obs} -> {new_obs}: only growth is supported")
    for key, value in old.items():
        if key not in state:
            raise ValueError(f"warm start: unexpected policy key {key}")
        if key not in ("dec.0.weight", "obs_norm.mean", "obs_norm.var"):
            if state[key].shape != value.shape:
                raise ValueError(
                    f"warm start: {key} {tuple(value.shape)} -> {tuple(state[key].shape)}"
                )
            state[key] = value.clone()
    hist = policy.hist
    old_weight, weight = old["dec.0.weight"], state["dec.0.weight"]
    z_dim = old_weight.shape[1] - old_obs * hist
    if z_dim <= 0 or weight.shape != (old_weight.shape[0], z_dim + new_obs * hist):
        raise ValueError("warm start: policy history or hidden sizes differ")
    weight.zero_()
    weight[:, :z_dim] = old_weight[:, :z_dim]
    for frame in range(hist):  # frames are flattened oldest first, each frame's terms in order
        new_start, old_start = z_dim + frame * new_obs, z_dim + frame * old_obs
        weight[:, new_start : new_start + old_obs] = old_weight[:, old_start : old_start + old_obs]
    state["obs_norm.mean"].zero_()
    state["obs_norm.var"].fill_(1.0)
    state["obs_norm.mean"][:old_obs] = old["obs_norm.mean"]
    state["obs_norm.var"][:old_obs] = old["obs_norm.var"]
    policy.load_state_dict(state)


class SuperMimicLearner:
    def __init__(
        self,
        env: EnvProtocol,
        cfg: SuperMimicConfig,
        log_dir: str | Path,
        device: str | torch.device | None = None,
        backend: str = "",
    ):
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        if cfg.test_hidden_dims is not None and not cfg.test_hidden_dims:
            raise ValueError("test_hidden_dims must be nonempty when provided")
        fractions = (
            cfg.wm_prioritized_frac,
            cfg.wm_archive_frac,
            cfg.explore_frac,
            cfg.driver_frac,
        )
        if min(fractions) < 0 or cfg.wm_prioritized_frac + cfg.wm_archive_frac > 1:
            raise ValueError("wm_prioritized_frac + wm_archive_frac must be within [0, 1]")
        if cfg.explore_frac + cfg.driver_frac > 1:
            raise ValueError("explore_frac + driver_frac must be within [0, 1]")
        if cfg.driver_frac > 0 and not cfg.driver_ckpt:
            raise ValueError("driver_frac > 0 needs driver_ckpt")

        self.env = env
        self.cfg = cfg
        self.backend = backend
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_device = torch.device(env.device)
        self.device = self.env_device if device is None else torch.device(device)
        # HOST_BRIDGE backends (e.g. MuJoCo) keep env tensors on CPU while the
        # learner may run on an accelerator (MPS on macOS); when the two differ,
        # the learner works on its own MDP copy and bridges tensors at the env
        # boundary. On CUDA/ROCm the devices always match and nothing changes.
        self._bridged = self.device != self.env_device
        mdp = get_supermimic_mdp(env)
        self.mdp = mdp.to(self.device) if self._bridged else mdp

        # TF32 only pays off for fp32 (amp="none") training: under bf16 autocast
        # the heavy GEMMs already run in bf16, while on ROCm a TF32 bmm over the
        # misaligned strided views produced by unflatten_state triggers per-call
        # rocBLAS kernel selection costing ~100 ms per call.
        torch.backends.cuda.matmul.allow_tf32 = cfg.amp == "none"
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed + 1000 * distributed.rank())  # distinct data on every rank
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None

        self.wm = self._make_world_model().to(self.device)
        self.pi = self._make_policy().to(self.device)
        self.critic = Critic(self._critic_input_dim(), hidden=cfg.critic_hidden).to(self.device)
        if cfg.test_hidden_dims is not None:
            self.critic.net = mlp(self.critic.norm.mean.numel(), cfg.test_hidden_dims, 1).to(
                self.device
            )
        for module in (self.wm, self.pi, self.critic):
            distributed.broadcast_module(module)
        self.world_size = distributed.world_size()
        self.base_lr = cfg.lr * (math.sqrt(self.world_size) if cfg.ddp_lr_scale else 1.0)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=self.base_lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=self.base_lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.base_lr)

        # optional task hook for contact-based window rarity; heights are logged per transition
        self._foot_heights = getattr(self.mdp, "foot_heights", None)
        self._rare_sampling = cfg.wm_prioritized_frac > 0 or cfg.wm_archive_frac > 0
        num_feet = 0
        if self._foot_heights is not None and self._rare_sampling:
            num_feet = int(self._foot_heights(unflatten_state(self._state_info()["s"])).shape[-1])
            self._foot_heights = self._compile(self._foot_heights)  # one call per env step
        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            env.num_envs,
            self.device,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
            num_feet=num_feet,
        )
        self.obs_hist = torch.zeros(env.num_envs, cfg.hist, self.mdp.obs_dim, device=self.device)
        self.err_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((self.mdp.num_joints,), 3e-3),
                torch.full((self.mdp.num_joints,), 5e-2),
            )
        ).to(self.device)

        self.archive = (
            WindowArchive(cfg.archive_size, self.device) if cfg.wm_archive_frac > 0 else None
        )
        num_envs = env.num_envs
        self.env_index = torch.arange(num_envs, device=self.device)
        self.explore = self.env_index < round(cfg.explore_frac * num_envs)
        self.ou_noise = torch.zeros(num_envs, self.mdp.num_joints, device=self.device)
        self.control_dt = float(getattr(env, "step_dt", 0.02))
        self.driver: Policy | None = None
        self.driver_goal_frames = 0
        if cfg.driver_ckpt:
            self.driver, self.driver_goal_frames = self._load_driver(cfg.driver_ckpt)
        # schedules (saved in checkpoints): policy updates start at policy_start, the actor
        # stays still until actor_frozen_until, its lr ramps from ramp_start
        self.lr_scale = 1.0
        self.policy_start = cfg.wm_only_iters
        self.actor_frozen_until = 0
        self.ramp_start = 0
        self.collapse_armed = False
        self.collapse_strikes = 0
        self.rollbacks = 0
        self._snapshot: dict[str, dict[str, torch.Tensor]] | None = None

        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        self.stats = _empty_stats()
        self._wm_rollout_fn = self._compile(self._wm_rollout)
        self._wm_err_fn = self._compile(self._wm_err)
        self._imag_step_fn = self._compile(self._imag_step)
        self._reward_term_fn = self._compile(self._reward_term)
        self._critic_input_fn = self._compile(self._critic_input)

        if distributed.is_main():
            with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
                json.dump(dict(asdict(cfg), world_size=self.world_size), stream, indent=2)

    def _compile(self, function):
        if not self.cfg.compile:
            return function
        if self.device.type == "mps":
            # Inductor Metal codegen is not usable on MPS; keep AOT graph
            # capture without kernel generation.
            return torch.compile(function, backend="aot_eager")
        return torch.compile(function)

    def _load_driver(self, path: str) -> tuple[Policy, int]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        args = checkpoint["args"]
        hist, goal_frames = int(args["hist"]), int(args["goal_frames"])
        if hist > self.cfg.hist or goal_frames < 1:
            raise ValueError(f"driver history {hist} exceeds this run's {self.cfg.hist}")
        driver = Policy(
            self.mdp.obs_dim,
            self.mdp.goal_frame_dim * goal_frames,
            hist=hist,
            z_dim=int(args["z_dim"]),
            num_joints=self.mdp.num_joints,
            hidden=tuple(args.get("test_hidden_dims") or args["policy_hidden"]),
        ).to(self.device)
        driver.load_state_dict(checkpoint["pi"])
        driver.eval()
        _set_requires_grad(driver, False)
        return driver, goal_frames

    def driver_share(self, iteration: int) -> float:
        """driver_frac before driver_fade[0], linearly down to zero at driver_fade[1]."""
        start, end = self.cfg.driver_fade
        fade = 1.0 if iteration <= start else max(0.0, (end - iteration) / max(1, end - start))
        return self.cfg.driver_frac * fade

    def _make_world_model(self) -> WorldModel:
        kwargs = {
            "num_joints": self.mdp.num_joints,
            "context_dim": self.mdp.context_dim,
        }
        if self.cfg.test_hidden_dims is not None:
            return WorldModel(hidden=self.cfg.test_hidden_dims, **kwargs)
        return WorldModel(hidden=self.cfg.wm_hidden, **kwargs)

    def _make_policy(self) -> Policy:
        goal_dim = self.mdp.goal_frame_dim * self.cfg.goal_frames
        policy = Policy(
            self.mdp.obs_dim,
            goal_dim,
            hist=self.cfg.hist,
            z_dim=self.cfg.z_dim,
            num_joints=self.mdp.num_joints,
            hidden=self.cfg.policy_hidden,
        )
        if self.cfg.test_hidden_dims is not None:
            enc_output = policy.enc[-1]
            assert isinstance(enc_output, nn.Linear)
            z_dim = enc_output.out_features
            policy.enc = mlp(goal_dim, self.cfg.test_hidden_dims, z_dim)
            policy.dec = mlp(
                z_dim + self.mdp.obs_dim * self.cfg.hist,
                self.cfg.test_hidden_dims,
                self.mdp.num_joints,
            )
            dec_output = policy.dec[-1]
            assert isinstance(dec_output, nn.Linear)
            with torch.no_grad():
                dec_output.weight.mul_(0.01)
                dec_output.bias.zero_()
        return policy

    def _state_info(self) -> dict[str, torch.Tensor]:
        state = self.env.state
        if state is None:
            state = self.env.init_state()
        info: dict[str, torch.Tensor] = state.info[SUPERMIMIC_STATE_KEY]
        if self._bridged:
            info = {key: value.to(self.device) for key, value in info.items()}
        return info

    def _critic_input_dim(self) -> int:
        info = self._state_info()
        with torch.no_grad():
            features = self.mdp.critic_features(
                unflatten_state(info["s"]),
                info["f"],
                self.mdp.normalize_context(info["xi"]),
            )
        return int(features.shape[-1] + self.mdp.goal_frame_dim * self.cfg.goal_frames)

    def _wm_rollout(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        contexts: torch.Tensor,
        pushes: torch.Tensor,
    ) -> State:
        state = unflatten_state(states[:, 0])
        predictions = []
        for index in range(self.cfg.wm_window):
            if index > 0:  # logged kick between transitions index-1 and index (zero if none)
                state = add_push(state, pushes[:, index])
            state = self.wm.step(
                state,
                actions[:, index],
                self.mdp.normalize_context(contexts[:, index]),
            )
            predictions.append(state)
        return (
            torch.stack([prediction[0] for prediction in predictions], 1),
            torch.stack([prediction[1] for prediction in predictions], 1),
            torch.stack([prediction[2] for prediction in predictions], 1),
            torch.stack([prediction[3] for prediction in predictions], 1),
            torch.stack([prediction[4] for prediction in predictions], 1),
            torch.stack([prediction[5] for prediction in predictions], 1),
        )

    def _wm_err(
        self, predicted: State, next_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        error = (self.wm.state_error(predicted, unflatten_state(next_states)) / std) ** 2
        error = error.mean(-1)
        return (error * mask).sum() / next_states.shape[0], error[:, -1]

    def _imag_step(
        self,
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One imagined step. Returns the pre-kick next state (reward/termination use it, as in
        the simulator) and the state the rollout continues from (equal without imagined pushes)."""
        action = self.pi(obs_hist, goal)
        next_state = self.wm.step(state, action, normalized_context)
        continued = next_state
        if self.cfg.imag_push_rate > 0:
            batch = action.shape[0]
            hit = (torch.rand(batch, 1, device=action.device) < self.cfg.imag_push_rate).float()
            continued = add_push(
                next_state, hit * sample_push(batch, self.cfg.imag_push_scale, action.device)
            )
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(continued, action, context, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        next_goal = self.mdp.goal(next_frames, self.cfg.goal_frames)
        return next_state, continued, action, next_hist, next_frames, next_goal

    def _reward_term(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mdp.reward(state, frames, action, previous_action), self.mdp.soft_termination(
            state, frames
        )

    def _critic_input(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((self.mdp.critic_features(state, frames, normalized_context), goal), -1)

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            info = self._state_info()
            state = unflatten_state(info["s"])
            obs = self.mdp.add_noise(self.mdp.obs(state, info["a_prev"], info["xi"], info["f"]))
            fresh = info["ep_step"] == 0
            shifted = torch.cat((self.obs_hist[:, 1:], obs.unsqueeze(1)), 1)
            self.obs_hist = torch.where(
                fresh[:, None, None], obs.unsqueeze(1).expand_as(self.obs_hist), shifted
            )
            goal = self.mdp.goal(info["f"], self.cfg.goal_frames)
            self.pi.obs_norm.update(obs)
            self.pi.goal_norm.update(goal)
            action = self.pi(self.obs_hist, goal)
            if self.driver is not None:
                driven = int(self.driver_share(self.iteration) * self.env.num_envs)
                if driven > 0:
                    driver_action = self.driver(
                        self.obs_hist[:, -self.driver.hist :],
                        self.mdp.goal(info["f"], self.driver_goal_frames),
                    )
                    mask = self.env_index >= self.env.num_envs - driven
                    action = torch.where(mask.unsqueeze(-1), driver_action, action)
            noise = self.cfg.action_noise * torch.randn_like(action)
            if bool(self.explore.any()):  # Ornstein-Uhlenbeck: correlated over ~1/theta s
                theta, dt = self.cfg.explore_theta, self.control_dt
                sigma = self.cfg.explore_noise * math.sqrt(2 * theta)
                step = self.ou_noise * (1 - theta * dt)
                step = step + sigma * math.sqrt(dt) * torch.randn_like(action)
                self.ou_noise = torch.where(self.explore.unsqueeze(-1), step, 0.0)
                noise = torch.where(self.explore.unsqueeze(-1), self.ou_noise, noise)
            action = torch.clamp(action + noise, -20, 20)
            if self._bridged:
                step_state = self.env.step(action.to(self.env_device))
            else:
                step_state = self.env.step(action)
            transition = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            if self._bridged:
                transition = {key: value.to(self.device) for key, value in transition.items()}
            initial = unflatten_state(transition["s"])
            next_state = unflatten_state(transition["s1"])
            if "feet" in self.buffer.fields and self._foot_heights is not None:
                transition = dict(transition, feet=self._foot_heights(next_state))
            self.buffer.add(transition)
            self.ou_noise[transition["done"]] = 0.0
            self.wm.in_norm.update(
                self.wm.inputs(
                    initial,
                    transition["a"],
                    self.mdp.normalize_context(transition["xi"]),
                )
            )
            self.wm.out_norm.update(self.wm.targets(initial, next_state))
            self.sim_steps += self.env.num_envs
            self.stats["r"].append(transition["r"].mean() / self.mdp.max_step_reward)
            if "ep_len" in transition:
                self.stats["ep_len"].append(transition["ep_len"].mean())
                self.stats["ep_ret"].append(transition["ep_ret_norm"].mean())
                self.stats["term"].append(transition["term"].float().sum())
                self.stats["done"].append(transition["done"].float().sum())

    @torch.no_grad()
    def window_rarity(self, window: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Score windows by contact changes + airborne steps + 4 x a fall (go2-supermimic) and
        flag archive-worthy ones: flight (>= 2 airborne steps), hard vertical base motion, or a
        fall. Contacts need the task's ``foot_heights`` hook; without it only the last two
        count. Steps after an episode end (mask 0) are ignored."""
        mask = window["mask"]
        batch, k = mask.shape
        fell = (window["term"].float() * mask).amax(-1) > 0
        next_states = unflatten_state(window["s1"].flatten(0, 1))
        vertical = next_states[2][..., 2].view(batch, k).abs()
        hard = ((vertical > self.cfg.hard_vz).float() * mask).amax(-1) > 0
        score = 4.0 * fell.float()
        worthy = fell | hard
        if self._foot_heights is not None:
            if "feet" in window:  # heights of the k next states, logged at collection
                heights = window["feet"]
                step_mask = mask[:, 1:]  # changes between consecutive next states
            else:  # kinematics on the first state and the k next states
                states = torch.cat((window["s"][:, :1], window["s1"]), 1)
                heights = self._foot_heights(unflatten_state(states.flatten(0, 1)))
                heights = heights.view(batch, k + 1, -1)
                step_mask = mask
            contact = (heights < self.cfg.contact_height).float()
            changes = ((contact[:, 1:] - contact[:, :-1]).abs().sum(-1) * step_mask).sum(-1)
            airborne = ((contact[:, -k:].sum(-1) == 0).float() * mask).sum(-1)
            score = score + changes + airborne
            worthy = worthy | (airborne >= 2)
        else:
            score = score + hard.float()
        return score, worthy

    def _wm_windows(self) -> dict[str, torch.Tensor]:
        """Uniform windows, plus the rarest of a candidate pool and archived rare windows when
        rare-event sampling is on; always ``wm_batch`` windows."""
        cfg = self.cfg
        batch = cfg.wm_batch
        prioritized = int(batch * cfg.wm_prioritized_frac)
        if prioritized == 0 and self.archive is None:
            return self.buffer.sample_windows(batch, cfg.wm_window)
        archived = int(batch * cfg.wm_archive_frac) if self.archive and self.archive.size else 0
        parts = [self.buffer.sample_windows(batch - prioritized - archived, cfg.wm_window)]
        candidates = self.buffer.sample_windows(4 * max(prioritized, batch // 8), cfg.wm_window)
        score, worthy = self.window_rarity(candidates)
        if self.archive is not None:
            keep = worthy & (torch.rand_like(score) < cfg.archive_keep_prob)
            self.archive.add({key: value[keep] for key, value in candidates.items()})
        if prioritized:
            top = (score + 1e-3 * torch.rand_like(score)).topk(prioritized).indices
            parts.append({key: value[top] for key, value in candidates.items()})
        if archived:
            assert self.archive is not None
            parts.append(self.archive.sample(archived))
        return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self._wm_windows()
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"], window["push"])
        loss, last_error = self._wm_err_fn(predicted, window["s1"], window["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        distributed.allreduce_grads(self.wm.parameters())
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        mask = window["mask"][:, -1]
        mean_last_error = (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)
        return loss.detach() / self.cfg.wm_window, mean_last_error

    def _sample_valid_starts(
        self, batch_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        starts = []
        histories = []
        count = 0
        while count < batch_size:
            start, history = self.buffer.sample_starts(2 * batch_size, self.cfg.hist)
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)

        def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)

    def policy_update(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        state = unflatten_state(start["s"])
        context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
        normalized_context = self.mdp.normalize_context(context)
        with torch.no_grad():
            obs_hist = self.mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
            )
        obs_hist = self.mdp.add_noise(obs_hist)
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)

        batch_size = self.cfg.pi_batch
        initial_goal = self.mdp.goal(frames, self.cfg.goal_frames)
        goal = initial_goal
        initial_state = state
        states, continued, actions, previous_actions, all_frames, goals = [], [], [], [], [], []
        try:
            for _ in range(self.cfg.horizon):
                previous_actions.append(previous_action)
                next_state, state, previous_action, obs_hist, frames, goal = self._imag_step_fn(
                    state,
                    obs_hist,
                    previous_action,
                    frames,
                    context,
                    normalized_context,
                    goal,
                )
                states.append(next_state)  # pre-kick: reward and termination
                continued.append(state)  # post-kick: critic states and the bootstrap
                actions.append(previous_action)
                all_frames.append(frames)
                goals.append(goal)

            def flatten(parts: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(parts).flatten(0, 1)

            all_states = cast(
                State, tuple(flatten([item[index] for item in states]) for index in range(6))
            )
            flat_frames = flatten(all_frames)
            rewards, dones = self._reward_term_fn(
                all_states, flat_frames, flatten(actions), flatten(previous_actions)
            )
            rewards = rewards.view(self.cfg.horizon, batch_size)
            dones = dones.view(self.cfg.horizon, batch_size)
            alive = torch.cumprod(1 - dones, 0)
            continuation = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), 0)
            discount = self.cfg.gamma ** torch.arange(
                self.cfg.horizon, device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            objective = (discount * continuation * rewards).sum(0)
            survival = alive[-1]
            reward_sum = (continuation * rewards).detach().sum(0)
            detached_rewards, detached_dones = rewards.detach(), dones.detach()
            if not self.cfg.no_critic:
                objective = objective + (self.cfg.gamma**self.cfg.horizon) * survival * self.critic(
                    self._critic_input_fn(state, frames, normalized_context, goal)
                )
            loss = -objective.mean()
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            distributed.allreduce_grads(self.pi.parameters())
            grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if torch.isfinite(grad_norm) and self.iteration >= self.actor_frozen_until:
                self.opt_pi.step()
        finally:
            _set_requires_grad(self.wm, True)
            _set_requires_grad(self.critic, True)

        value_loss = torch.zeros((), device=self.device)
        if not self.cfg.no_critic:
            with torch.no_grad():
                critic_states = cast(
                    State,
                    tuple(
                        torch.cat(
                            (
                                initial_state[index].unsqueeze(0),
                                torch.stack([item[index].detach() for item in continued]),
                            ),
                            0,
                        ).flatten(0, 1)
                        for index in range(6)
                    ),
                )
                critic_frames = torch.cat(
                    (start["f"].unsqueeze(0), torch.stack(all_frames)), 0
                ).flatten()
                critic_goals = torch.cat(
                    (initial_goal.unsqueeze(0), torch.stack(goals)), 0
                ).flatten(0, 1)
                critic_context = (
                    normalized_context.unsqueeze(0)
                    .expand(self.cfg.horizon + 1, -1, -1)
                    .flatten(0, 1)
                )
                critic_input = self._critic_input_fn(
                    critic_states, critic_frames, critic_context, critic_goals
                ).view(self.cfg.horizon + 1, batch_size, -1)
            self.critic.norm.update(critic_input)
            with torch.no_grad():
                values = self.critic(critic_input)
                target = values[-1]
                targets = []
                for index in range(self.cfg.horizon - 1, -1, -1):
                    target = detached_rewards[index] + self.cfg.gamma * (
                        1 - detached_dones[index]
                    ) * ((1 - self.cfg.lam) * values[index + 1] + self.cfg.lam * target)
                    targets.append(target)
                targets = torch.stack(targets[::-1])
            flat_input = critic_input[:-1].reshape(-1, critic_input.shape[-1])
            flat_targets = targets.reshape(-1)
            num_samples = flat_input.shape[0]
            minibatch_size = num_samples // self.cfg.critic_minibatches
            if minibatch_size == 0:
                raise ValueError("critic_minibatches exceeds the number of imagined transitions")
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(num_samples, device=self.device)
                for index in range(self.cfg.critic_minibatches):
                    indices = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                    value_loss = (
                        (self.critic(flat_input[indices]) - flat_targets[indices]) ** 2
                    ).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    value_loss.backward()
                    distributed.allreduce_grads(self.critic.parameters())
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()

        imagined_reward = (reward_sum / self.cfg.horizon / self.mdp.max_step_reward).mean()
        return (
            loss.detach(),
            value_loss.detach(),
            imagined_reward,
            grad_norm.detach(),
            survival.mean().detach(),
        )

    def lr_at(self, iteration: int) -> float:
        if iteration >= self.cfg.lr_decay_iters:
            return self.base_lr * 0.03
        return self.base_lr * (
            0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / self.cfg.lr_decay_iters))
        )

    def actor_lr_factor(self, iteration: int) -> float:
        """actor_lr_scale, ramped linearly over actor_ramp_iters from the latest (re)start."""
        if self.cfg.actor_ramp_iters <= 0:
            return self.cfg.actor_lr_scale
        start = max(self.ramp_start, self.actor_frozen_until, self.policy_start)
        ramp = min(1.0, max(0.0, (iteration - start) / self.cfg.actor_ramp_iters))
        return ramp * self.cfg.actor_lr_scale

    def _new_policy_optimizers(self) -> None:
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=self.base_lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.base_lr)

    def _take_snapshot(self) -> None:
        self._snapshot = {
            "pi": {k: v.detach().clone() for k, v in self.pi.state_dict().items()},
            "critic": {k: v.detach().clone() for k, v in self.critic.state_dict().items()},
        }

    def collapse_guard(self, fall_rate: float) -> bool:
        """Called once per log window with the falls per 1000 env steps; True after a rollback.

        Model exploitation shows up as many real falls. The guard arms once the active actor
        is below the threshold; armed, two windows in a row above it restore actor and critic
        from the last snapshot (the world model is kept: it has just learned from the falls),
        with fresh optimizer moments, half the lr, and the actor ramp restarted."""
        threshold = self.cfg.collapse_fall_rate
        if threshold <= 0 or self.iteration < max(self.policy_start, self.actor_frozen_until):
            return False
        if fall_rate <= threshold:
            self.collapse_armed, self.collapse_strikes = True, 0
            return False
        if not self.collapse_armed:
            return False
        self.collapse_strikes += 1
        if self.collapse_strikes < 2 or self._snapshot is None:
            return False
        self.pi.load_state_dict(self._snapshot["pi"])
        self.critic.load_state_dict(self._snapshot["critic"])
        self._new_policy_optimizers()
        self.lr_scale *= 0.5
        self.ramp_start = self.iteration
        self.collapse_strikes = 0
        self.rollbacks += 1
        return True

    def warm_start(self, path: str | Path) -> list[str]:
        """Initialise from another run's checkpoint (see SuperMimicConfig.init_from);
        returns what was loaded, expanded, or re-initialised."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        notes = []
        if _same_shapes(self.wm, checkpoint["wm"]):
            self.wm.load_state_dict(checkpoint["wm"])
            notes.append("world model loaded")
        else:
            self.policy_start = max(self.policy_start, self.cfg.wm_rewarmup_iters)
            notes.append(f"world model re-initialised; policy updates from {self.policy_start}")
        if _same_shapes(self.pi, checkpoint["pi"]):
            self.pi.load_state_dict(checkpoint["pi"])
            notes.append("policy loaded")
        else:
            expand_policy_inputs(checkpoint["pi"], self.pi)
            notes.append("policy expanded (new inputs at zero weight)")
        critic = checkpoint.get("critic")
        if critic is not None and not self.cfg.reset_critic and _same_shapes(self.critic, critic):
            self.critic.load_state_dict(critic)
            notes.append("critic loaded")
        else:
            self.actor_frozen_until = max(
                self.actor_frozen_until, self.policy_start + self.cfg.critic_warmup_iters
            )
            notes.append(f"critic re-initialised; actor still until {self.actor_frozen_until}")
        for module in (self.wm, self.pi, self.critic):
            distributed.broadcast_module(module)
        return notes

    def _schedule(self) -> dict[str, float | int | bool]:
        return {
            "lr_scale": self.lr_scale,
            "policy_start": self.policy_start,
            "actor_frozen_until": self.actor_frozen_until,
            "ramp_start": self.ramp_start,
            "collapse_armed": self.collapse_armed,
            "rollbacks": self.rollbacks,
        }

    def _checkpoint(self, include_optimizers: bool) -> dict[str, Any]:
        checkpoint = {
            "wm": self.wm.state_dict(),
            "pi": self.pi.state_dict(),
            "iteration": self.iteration,
            "wall_time": self.wall_time,
            "sim_steps": self.sim_steps,
            "world_size": self.world_size,
            "schedule": self._schedule(),
            "args": asdict(self.cfg),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if include_optimizers:
            checkpoint.update(
                {
                    "critic": self.critic.state_dict(),
                    "opt_wm": self.opt_wm.state_dict(),
                    "opt_pi": self.opt_pi.state_dict(),
                    "opt_v": self.opt_v.state_dict(),
                }
            )
        return checkpoint

    def save(self, path: str | Path | None = None, include_optimizers: bool = True) -> Path:
        destination = Path(path) if path is not None else self.log_dir / "last.pt"
        if not distributed.is_main():  # every rank holds the same networks
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint(include_optimizers), temporary)
        os.replace(temporary, destination)
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        required = {"critic", "opt_wm", "opt_pi", "opt_v"}
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(
                f"SuperMimic resume requires a full last.pt checkpoint; missing {missing} in {path}"
            )
        self.wm.load_state_dict(checkpoint["wm"])
        self.pi.load_state_dict(checkpoint["pi"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.opt_wm.load_state_dict(checkpoint["opt_wm"])
        self.opt_pi.load_state_dict(checkpoint["opt_pi"])
        self.opt_v.load_state_dict(checkpoint["opt_v"])
        self.iteration = checkpoint["iteration"]
        self.wall_time = checkpoint["wall_time"]
        self.sim_steps = checkpoint["sim_steps"]
        for key, value in checkpoint.get("schedule", {}).items():
            setattr(self, key, value)

    @staticmethod
    def _mean(values: list[torch.Tensor]) -> float:
        return float(sum(float(value) for value in values) / max(len(values), 1))

    def train(self, resume: str | Path | None = None) -> dict[str, float | int | bool]:
        notes: list[str] = []
        if resume is not None:
            self.load(resume)
        elif self.cfg.init_from:
            notes = self.warm_start(self.cfg.init_from)
        start_iteration = self.iteration
        self.stats = _empty_stats()
        logger = SuperMimicLogger(
            env_name=type(self.env).__name__,
            backend=self.backend,
            max_iterations=self.cfg.iterations,
            wm_only_iters=self.cfg.wm_only_iters,
            start_iteration=start_iteration,
            num_envs=self.env.num_envs,
            log_dir=str(self.log_dir),
            log_backend=self.cfg.log_backend if distributed.is_main() else "no_print",
        )
        logger.start(status="Warming up...")
        for note in notes:
            logger.log_status(f"warm start: {note}")
        try:
            if start_iteration == 0:
                self.collect(self.cfg.warmup_steps)
            required_steps = max(self.cfg.hist, self.cfg.wm_window)
            if start_iteration > 0:
                required_steps = max(required_steps, self.cfg.warmup_steps)
            if self.buffer.size < required_steps:
                self.collect(required_steps - self.buffer.size)
            logger.log_buffer_fill(self.buffer.size, self.cfg.buffer_steps)
            logger.start_training_timer()

            log_path = self.log_dir / "train_log.csv"
            if not distributed.is_main():  # only rank 0 writes the CSV (rows are rank-averaged)
                log_path = Path(os.devnull)
            with log_path.open("a", encoding="utf-8", newline="") as log_file:
                if start_iteration == 0:
                    log_file.write(",".join(_LOG_FIELDS) + "\n")
                train_start = time.time() - self.wall_time
                log_start = time.time()
                step_accumulator = 0.0
                window_env_steps = 0
                aggregate: dict[str, list[torch.Tensor]] = {
                    key: [] for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")
                }
                last_metrics: dict[str, float | int] = {}
                for iteration in range(start_iteration, self.cfg.iterations):
                    if self.cfg.compile and iteration == start_iteration:
                        logger.log_status("Compiling world model (one-off, slow on first run)")
                    if self.cfg.compile and iteration == self.policy_start:
                        logger.log_status("Compiling policy update (one-off, slow on first run)")
                    iteration_start = time.time()
                    learning_rate = self.lr_at(iteration) * self.lr_scale
                    actor_rate = learning_rate * self.actor_lr_factor(iteration)
                    for optimizer, rate in (
                        (self.opt_wm, learning_rate),
                        (self.opt_pi, actor_rate),
                        (self.opt_v, learning_rate),
                    ):
                        for group in optimizer.param_groups:
                            group["lr"] = rate
                    step_accumulator += 1.0 / self.cfg.utd
                    collect_steps = int(step_accumulator)
                    step_accumulator -= collect_steps
                    stats_mark = len(self.stats["r"])
                    self.collect(collect_steps)
                    window_env_steps += collect_steps * self.env.num_envs
                    train_phase_start = time.time()
                    wm_loss = torch.zeros((), device=self.device)
                    wm_error = torch.zeros((), device=self.device)
                    for _ in range(self.cfg.wm_batches):
                        wm_loss, wm_error = self.wm_update()
                    aggregate["wm"].append(wm_loss)
                    aggregate["wm8"].append(wm_error)
                    pi_loss: float | None = None
                    v_loss: float | None = None
                    imag_rew: float | None = None
                    pi_gradnorm: float | None = None
                    imag_survival: float | None = None
                    if iteration >= self.policy_start:
                        policy_loss, value_loss, imagined_reward, grad_norm, survival = (
                            self.policy_update()
                        )
                        aggregate["pi"].append(policy_loss)
                        aggregate["v"].append(value_loss)
                        aggregate["ir"].append(imagined_reward)
                        aggregate["gn"].append(grad_norm)
                        aggregate["surv"].append(survival)
                        pi_loss = float(policy_loss)
                        v_loss = float(value_loss)
                        imag_rew = float(imagined_reward)
                        pi_gradnorm = float(grad_norm)
                        imag_survival = float(survival)
                    train_time = time.time() - train_phase_start
                    self.iteration = iteration + 1
                    self.wall_time = time.time() - train_start
                    iteration_time = time.time() - iteration_start

                    iter_rewards = self.stats["r"][stats_mark:]
                    mean_reward = self._mean(iter_rewards) if iter_rewards else None
                    ep_len = self._mean(self.stats["ep_len"]) if self.stats["ep_len"] else None
                    ep_ret_norm = self._mean(self.stats["ep_ret"]) if self.stats["ep_ret"] else None
                    total_done = sum(float(value) for value in self.stats["done"])
                    term_frac = (
                        sum(float(value) for value in self.stats["term"]) / total_done
                        if total_done > 0
                        else None
                    )
                    logger.log_step(
                        self.iteration,
                        wm_loss=float(wm_loss),
                        wm_err8=float(wm_error),
                        pi_loss=pi_loss,
                        v_loss=v_loss,
                        imag_rew=imag_rew,
                        imag_survival=imag_survival,
                        pi_gradnorm=pi_gradnorm,
                        lr=learning_rate,
                        sim_rew=mean_reward,
                        ep_len=ep_len,
                        ep_ret_norm=ep_ret_norm,
                        term_frac=term_frac,
                        iter_sim_steps=collect_steps * self.env.num_envs,
                        collect_time=train_phase_start - iteration_start,
                        learn_time=train_time,
                        iteration_time=iteration_time,
                    )

                    if self.iteration % self.cfg.log_every == 0:
                        iterations_per_second = self.cfg.log_every / (time.time() - log_start)
                        log_start = time.time()
                        term_frac = float(
                            sum(float(value) for value in self.stats["term"])
                            / max(sum(float(value) for value in self.stats["done"]), 1)
                        )
                        row = [
                            self.iteration,
                            self.wall_time / 3600,
                            self.sim_steps * self.world_size,
                            self._mean(aggregate["wm"]),
                            self._mean(aggregate["wm8"]),
                            self._mean(aggregate["pi"]),
                            self._mean(aggregate["v"]),
                            self._mean(aggregate["ir"]),
                            self._mean(aggregate["gn"]),
                            self._mean(aggregate["surv"]),
                            self._mean(self.stats["r"]),
                            self._mean(self.stats["ep_len"]),
                            self._mean(self.stats["ep_ret"]),
                            term_frac,
                            iterations_per_second,
                            learning_rate,
                        ]
                        fall_rate = (
                            1000.0
                            * sum(float(value) for value in self.stats["term"])
                            / max(window_env_steps, 1)
                        )
                        window_env_steps = 0
                        if self.world_size > 1:  # metrics averaged over ranks
                            values = torch.tensor(
                                [*row[3:14], fall_rate], dtype=torch.float64, device=self.device
                            )
                            *row[3:14], fall_rate = distributed.allreduce_mean_(values).tolist()
                        if self.archive is not None:
                            logger.log_status(
                                f"rare-event archive: {self.archive.size} windows; "
                                f"falls per 1000 env steps: {fall_rate:.2f}"
                            )
                        if self.collapse_guard(fall_rate):
                            logger.log_status(
                                f"collapse guard: {fall_rate:.2f} falls per 1000 steps at "
                                f"iteration {self.iteration}; actor and critic rolled back, "
                                f"lr scale {self.lr_scale:g}"
                            )
                        log_file.write(
                            ",".join(
                                f"{value:.6g}" if isinstance(value, float) else str(value)
                                for value in row
                            )
                            + "\n"
                        )
                        log_file.flush()
                        last_metrics = dict(zip(_LOG_FIELDS, row, strict=True))
                        self.stats = _empty_stats()
                        aggregate = {key: [] for key in aggregate}

                    if (
                        self.cfg.collapse_fall_rate > 0
                        and self.iteration >= self.policy_start
                        and self.iteration % self.cfg.collapse_snapshot_every == 0
                        and self.collapse_strikes == 0
                    ):
                        self._take_snapshot()

                    if (
                        self.iteration % self.cfg.ckpt_every == 0
                        or self.iteration == self.cfg.iterations
                    ):
                        self.wall_time = time.time() - train_start
                        self.save()
                        numbered = self.save(
                            self.log_dir / f"ckpt_{self.iteration:06d}.pt",
                            include_optimizers=False,
                        )
                        logger.log_save(str(numbered))

            self.wall_time = time.time() - train_start
            logger.finish()
            return {
                "completed": self.iteration >= self.cfg.iterations,
                "iteration": self.iteration,
                "sim_steps": self.sim_steps,
                "wall_time": self.wall_time,
                **last_metrics,
            }
        finally:
            logger.close()


__all__ = ["SuperMimicConfig", "SuperMimicLearner", "expand_policy_inputs"]
