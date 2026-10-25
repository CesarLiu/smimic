"""Observation, reward and event terms of the Go2 chase task (UniLab port).

Rewards follow go2-supermimic env/mdp.py and mjlab's velocity-task terms; they are only
logged during SuperMimic training (imagination re-implements them, supermimic/imagine.py).
The context event reproduces go2-supermimic's startup domain randomization: foot friction
(slide, spin, roll; shared by the four feet), base COM offset, and joint encoder bias.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import EventTermCfg, ManagerTermBase

from unisupermimic.go2chase.target import collision

from .robot import FOOT_GEOM_NAMES, actuator_constants

if TYPE_CHECKING:
    from unilab.base.entity import Entity

    from .command import TargetCommand

CONTEXT_DIM = 4  # foot slide friction (1), base COM offset (3): go2-supermimic state.xi


def _term(env: Any, command_name: str = "target") -> TargetCommand:
    return cast("TargetCommand", env.command_manager.get_term(command_name))


##
# Observations.
##


def target_observation(env: Any, command_name: str = "target") -> torch.Tensor:
    return _term(env, command_name).command


def _probe(env: Any, width: int) -> torch.Tensor | None:
    """Zeros while the observation manager probes shapes, before the read plan exists."""
    if getattr(env.scene, "read_plan", None) is None:
        return torch.zeros(env.num_envs, width, device=torch.device(env.device))
    return None


def base_ang_vel(env: Any, command_name: str = "target") -> torch.Tensor:
    """Base angular velocity in the base frame (mjlab's IMU gyro on the rigid base)."""
    probe = _probe(env, 3)
    return probe if probe is not None else _term(env, command_name).robot.root_link_ang_vel_b


def projected_gravity(env: Any, command_name: str = "target") -> torch.Tensor:
    """Gravity direction in the base frame."""
    probe = _probe(env, 3)
    return probe if probe is not None else _term(env, command_name).robot.projected_gravity_b


def lidar(env: Any, command_name: str = "target") -> torch.Tensor:
    """Planar LiDAR ranges around the robot to the static obstacles (obstacles task)."""
    term = _term(env, command_name)
    probe = _probe(env, term.cfg.target.obstacles.lidar_rays)
    return probe if probe is not None else term.lidar()


def height_scan(env: Any, command_name: str = "target") -> torch.Tensor:
    """Base height above each point of the local ground grid (terrain task)."""
    from unisupermimic.go2chase.target.terrain import SCAN_POINTS

    probe = _probe(env, SCAN_POINTS)
    return probe if probe is not None else _term(env, command_name).scan()


##
# Chase rewards.
##


def _to_chase_ball(env: Any, command_name: str):
    term = _term(env, command_name)
    robot = term.robot
    rel = term.gen.chase_pos[:, :2] - robot.root_link_pos_w[:, :2]
    dist = rel.norm(dim=-1).clamp(min=1e-6)
    return term, rel / dist.unsqueeze(-1), dist, robot.root_link_lin_vel_w[:, :2]


def chase_approach(env: Any, command_name: str, max_speed: float) -> torch.Tensor:
    """Robot velocity toward the chase ball, clipped to [-max_speed, max_speed]."""
    term, u, _, v = _to_chase_ball(env, command_name)
    return (v * u).sum(-1).clamp(-max_speed, max_speed) * term.has_chase.float()


def chase_tag(env: Any, command_name: str) -> torch.Tensor:
    """Gaussian of the head-to-ball distance while taggable (collision.tag_activation)."""
    term = _term(env, command_name)
    robot = term.robot
    return collision.tag_activation(
        term.gen.state(), robot.root_link_pos_w[:, :2], robot.heading_w, term.cfg.target
    )


def facing_target(env: Any, command_name: str) -> torch.Tensor:
    """cos(angle between heading and the chase ball) mapped to [0, 1]."""
    term, u, _, _ = _to_chase_ball(env, command_name)
    h = term.robot.heading_w
    forward = torch.stack([h.cos(), h.sin()], dim=-1)
    return 0.5 * (1.0 + (forward * u).sum(-1)) * term.has_chase.float()


##
# Dodge rewards.
##


def evade_danger(env: Any, command_name: str, std: float) -> torch.Tensor:
    """Positive when the dodge obstacle is close to the body box; use a negative weight."""
    term = _term(env, command_name)
    d = term.body_clearance().clamp(min=0.0)
    return torch.exp(-((d / std) ** 2)) * term.has_dodge.float()


def evade_anticipate(env: Any, command_name: str, std: float) -> torch.Tensor:
    """Positive when the thrown obstacle is on course to pass close to the body."""
    term = _term(env, command_name)
    d = term.predicted_body_clearance().clamp(min=0.0)
    return torch.exp(-((d / std) ** 2)) * term.has_dodge.float()


def hit(env: Any, command_name: str) -> torch.Tensor:
    return _term(env, command_name).hit


##
# Shared rewards.
##


def crouch(env: Any, command_name: str) -> torch.Tensor:
    """Base below 0.23 m outside of a high-bar duck; use a negative weight."""
    term = _term(env, command_name)
    pos = term.base_local()
    return collision.crouch_violation(term.gen.state(), pos[:, :2], pos[:, 2])


def outside_arena(env: Any, command_name: str, margin: float) -> torch.Tensor:
    """Distance beyond (arena_radius + margin); use a negative weight."""
    term = _term(env, command_name)
    r = (term.robot.root_link_pos_w[:, :2] - term.gen.center).norm(dim=-1)
    return (r - term.cfg.target.arena_radius - margin).clamp(min=0.0)


def upright(env: Any, std: float, command_name: str = "target") -> torch.Tensor:
    g = _term(env, command_name).robot.projected_gravity_b
    return torch.exp(-(g[:, :2].square().sum(-1)) / std**2)


def tilt_rate_l2(env: Any, command_name: str = "target") -> torch.Tensor:
    """Squared roll and pitch rates of the base; use a negative weight."""
    return _term(env, command_name).robot.root_link_ang_vel_b[:, :2].square().sum(-1)


def hard_landing(env: Any, near_ground: float = 0.05, command_name: str = "target") -> torch.Tensor:
    """Squared downward base speed while a foot is near the ground (heights above the
    ground under the base, as in imagination); negative weight."""
    term = _term(env, command_name)
    robot = term.robot
    g = term.ground_under_base()
    feet_z = (robot.limb_positions_w()["feet"][..., 2] - g.unsqueeze(-1)).min(-1).values
    vz = robot.root_link_lin_vel_w[:, 2]
    return (-vz).clamp(min=0.0).square() * (feet_z < near_ground).float()


def joint_deviation_l2(env: Any, command_name: str = "target") -> torch.Tensor:
    robot = _term(env, command_name).robot
    return (robot.joint_pos - robot.default_joint_pos).square().sum(-1)


def base_height_low(env: Any, min_height: float, command_name: str = "target") -> torch.Tensor:
    """How far the base is below min_height above the ground under it; negative weight."""
    z = _term(env, command_name).base_local()[:, 2]
    return (min_height - z).clamp(min=0.0)


def obstacle_contact(env: Any, command_name: str = "target") -> torch.Tensor:
    """Depth the body reaches into the slightly inflated static obstacles; negative weight."""
    return _term(env, command_name).obstacle_penetration()


def joint_pos_limits(env: Any, command_name: str = "target") -> torch.Tensor:
    """Violation of mjlab's soft limits (ranges shrunk by 0.9 about the middle)."""
    robot = _term(env, command_name).robot
    q, limits = robot.joint_pos, robot.soft_joint_pos_limits
    return ((limits[..., 0] - q).clamp(min=0.0) + (q - limits[..., 1]).clamp(min=0.0)).sum(-1)


class JointTorquesL2(ManagerTermBase):
    """Squared actuator torques: the position actuators' PD law, clamped to the limits."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        kp, kd, effort = actuator_constants()
        device = torch.device(env.device)
        self._kp = torch.tensor(kp, device=device)
        self._kd = torch.tensor(kd, device=device)
        self._effort = torch.tensor(effort, device=device)

    def __call__(self, env: Any, command_name: str = "target") -> torch.Tensor:
        robot = _term(env, command_name).robot
        action = env.action_manager.get_term("joint_pos")
        target = action.processed_action - robot.encoder_bias
        torque = self._kp * (target - robot.joint_pos) - self._kd * robot.joint_vel
        return torque.clamp(-self._effort, self._effort).square().sum(-1)


class JointAccL2(ManagerTermBase):
    """Squared joint accelerations, from the velocity change over one control step."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(env)
        self._prev: torch.Tensor | None = None
        self._dt = float(env.step_dt)

    def __call__(self, env: Any, command_name: str = "target") -> torch.Tensor:
        qd = _term(env, command_name).robot.joint_vel
        prev = qd if self._prev is None else self._prev
        self._prev = qd.clone()
        return ((qd - prev) / self._dt).square().sum(-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        self._prev = None
        return {}


##
# Events.
##


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def reset_base(
    env: Any,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    command_name: str = "target",
    entity_name: str = "robot",
) -> None:
    """mjlab reset_root_state_uniform (default pose + uniform x, y, z, yaw offsets, zero
    velocity) around the arena centre the target command assigns (a terrain tile origin)."""
    robot = cast("Entity", env.scene[entity_name])
    device = torch.device(env.device)
    reset_mask = env.event_term_mask_buffer.scatter(env_ids)
    ids = reset_mask.nonzero(as_tuple=False).flatten()
    root = robot.data.default_root_state(device).clone()
    centers = _term(env, command_name).place(ids)
    k = ids.numel()

    def sample(axis: str) -> torch.Tensor:
        lo, hi = pose_range.get(axis, (0.0, 0.0))
        return env.rng.uniform(lo, hi, (k,)).to(device)

    offset = torch.stack((sample("x"), sample("y"), sample("z")), -1)
    root[ids, 0:3] += offset + centers
    yaw = sample("yaw")
    delta = torch.stack(
        (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1
    )
    root[ids, 3:7] = _quat_mul(root[ids, 3:7], delta)
    root[:, 7:13] = 0.0
    robot.write_root_state_tensor_to_sim(root, reset_mask)


class Go2ContextEvent(ManagerTermBase):
    """Startup domain randomization, replayed on every reset (UniLab restores model fields
    on reset unless they are written again). Writes the context xi into the command."""

    uses_startup_rows = True

    def __init__(self, cfg: EventTermCfg, env: Any):
        super().__init__(env)
        if cfg.mode not in ("startup", "reset"):
            raise ValueError("Go2ContextEvent requires startup or reset mode")
        self._startup = cfg.mode == "startup"
        self._command_name = cfg.params.get("command_name", "target")
        self._robot = cast("Entity", env.scene[cfg.params.get("entity_name", "robot")])
        foot_ids, _ = self._robot.find_geoms(FOOT_GEOM_NAMES, preserve_order=True)
        self._foot_ids, self._friction_defaults = self._robot.bind_geom_friction_write(
            foot_ids, term_name="go2_context"
        )
        base_ids, _ = self._robot.find_bodies(("base",), preserve_order=True)
        self._base_ids, self._ipos_defaults = self._robot.bind_body_ipos_write(
            base_ids, term_name="go2_context"
        )

    def __call__(
        self,
        env: Any,
        env_ids: torch.Tensor | None,
        command_name: str = "target",
        entity_name: str = "robot",
        slide: tuple[float, float] = (0.3, 1.5),
        spin: tuple[float, float] = (1e-4, 2e-2),
        roll: tuple[float, float] = (1e-5, 5e-3),
        com_range: tuple[float, float, float] = (0.025, 0.025, 0.03),
        encoder_bias: float = 0.015,
    ) -> None:
        del command_name, entity_name
        ids = resolve_reset_env_ids(
            env_ids, num_envs=env.num_envs, device=env.device, owner="Go2ContextEvent"
        )
        term = _term(env, self._command_name)
        if self._startup:
            n, rng = env.num_envs, env.rng
            device = torch.device(env.device)

            def log_uniform(lo: float, hi: float) -> torch.Tensor:
                return torch.exp(rng.uniform(math.log(lo), math.log(hi), (n,)))

            term.friction = torch.stack(
                (rng.uniform(*slide, (n,)), log_uniform(*spin), log_uniform(*roll)), -1
            ).to(device)
            bound = torch.tensor(com_range, device=device)
            term.com_offset = rng.uniform(-bound, bound, (n, 3)).to(device)
            term.xi = torch.cat((term.friction[:, :1], term.com_offset), -1)
            bias = self._robot.data.encoder_bias_tensor
            bias.copy_(rng.uniform(-encoder_bias, encoder_bias, tuple(bias.shape)).to(bias))
        friction = self._selected(self._friction_defaults, ids)
        friction[...] = term.friction[ids].unsqueeze(1)  # every foot shares one friction
        self._robot.write_geom_friction_to_sim(
            friction, geom_ids=self._foot_ids, env_ids=ids, term_name="go2_context"
        )
        ipos = self._selected(self._ipos_defaults, ids) + term.com_offset[ids].unsqueeze(1)
        write = (
            self._robot.record_startup_body_ipos_to_sim
            if self._startup
            else self._robot.write_body_ipos_to_sim
        )
        write(ipos, body_ids=self._base_ids, env_ids=ids, term_name="go2_context")

    @staticmethod
    def _selected(defaults: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        values = defaults.to(device=ids.device, dtype=torch.float32)
        if values.ndim == 2:  # shared defaults
            return values.expand((ids.numel(), *values.shape)).clone()
        return values.index_select(0, ids).clone()


__all__ = ["CONTEXT_DIM", "Go2ContextEvent", "JointAccL2", "JointTorquesL2"]
