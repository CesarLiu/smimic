"""Bridge between the Go2 chase env (UniLab, env/adapter.py) and the SuperMimic learner.

Defines the world-model state, the sim context xi, the robot planar pose, and
an exact re-implementation of the actor observation so that a policy trained
in imagination can be evaluated (and deployed) in the real env unchanged.

World-model state s (70):
  base height (1), gravity in body frame (3), base linear velocity in the yaw
  frame (3), base angular velocity in body frame (3), joint pos - default (12),
  joint vel (12), then hip, knee and foot positions relative to the base in the
  yaw frame (3 x 4 x 3, legs ordered FR FL RR RL). The limb points make contact
  changes (touchdown / liftoff) explicit and let imagination compute per-leg
  collisions with obstacles exactly as the env does.
The planar pose (x, y, yaw) is not part of s: the model predicts its change in
the previous yaw frame, which keeps the dynamics translation/yaw invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from unisupermimic.go2chase.target import interface as ti
from unisupermimic.go2chase.target.interface import yaw_rotate_inverse
from unisupermimic.go2chase.target.terrain import SCAN_POINTS, ground

STATE_DIM = 70
POSE_DELTA_DIM = 3
CONTEXT_DIM = 4  # foot friction (1), base COM offset (3)
NUM_JOINTS = 12

# Slices into s.
Z = slice(0, 1)
GRAV = slice(1, 4)
V_YAW = slice(4, 7)
W_B = slice(7, 10)
Q = slice(10, 22)
QD = slice(22, 34)
HIPS = slice(34, 46)
KNEES = slice(46, 58)
FEET = slice(58, 70)
LEGS = ("FR", "FL", "RR", "RL")
FOOT_SITES = LEGS
HIP_BODIES = tuple(f"{leg}_thigh" for leg in LEGS)  # thigh frame = hip pitch joint
KNEE_BODIES = tuple(f"{leg}_calf" for leg in LEGS)  # calf frame = knee joint
FOOT_RADIUS = 0.022


def feet_world_z(s: torch.Tensor) -> torch.Tensor:
    """World-frame foot site heights (N, 4) from the state (yaw rotation keeps z)."""
    return s[:, Z] + s[:, FEET].reshape(-1, 4, 3)[..., 2]


def limb_points_world(s: torch.Tensor, pose: torch.Tensor) -> dict[str, torch.Tensor]:
    """World-frame hips, knees and feet (each (N, 4, 3)) from the state and the
    planar pose (x, y, yaw)."""
    c, si = pose[:, 2].cos(), pose[:, 2].sin()
    base = torch.cat([pose[:, :2], s[:, Z]], -1).unsqueeze(1)
    out = {}
    for name, sl in (("hips", HIPS), ("knees", KNEES), ("feet", FEET)):
        rel = s[:, sl].reshape(-1, 4, 3)
        x = c.unsqueeze(-1) * rel[..., 0] - si.unsqueeze(-1) * rel[..., 1]
        y = si.unsqueeze(-1) * rel[..., 0] + c.unsqueeze(-1) * rel[..., 1]
        out[name] = base + torch.stack([x, y, rel[..., 2]], -1)
    return out


# Actor observation terms in env order, with per-frame dims (see env_cfg.py);
# ("lidar", rays) follows with obstacles, ("height_scan", SCAN_POINTS) on terrain.
OBS_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
    ("target", ti.DIM),
)
# Uniform observation noise half-widths, matching env_cfg.py.
OBS_NOISE = {
    "base_ang_vel": 0.2,
    "projected_gravity": 0.05,
    "joint_pos": 0.01,
    "joint_vel": 1.5,
    "lidar": 0.03,
    "height_scan": 0.02,
}


def obs_terms(env_cfg) -> tuple[tuple[str, int], ...]:
    """Actor terms (name, per-frame dim) of an env config, in order."""
    names = list(env_cfg.observations["actor"].terms)
    rays = env_cfg.commands["target"].target.obstacles.lidar_rays
    known = dict(OBS_TERMS) | {"lidar": rays, "height_scan": SCAN_POINTS}
    terms = tuple((n, known[n]) for n in names)
    assert [n for n, _ in terms[: len(OBS_TERMS)]] == [n for n, _ in OBS_TERMS], names
    return terms


@dataclass
class RobotConsts:
    """Per-env constants of the robot and action mapping (all (N, 12) unless noted)."""

    default_q: torch.Tensor
    action_scale: torch.Tensor
    kp: torch.Tensor  # (12,)
    kd: torch.Tensor  # (12,)
    effort: torch.Tensor  # (12,)
    soft_limits: torch.Tensor  # (N, 12, 2)


def robot_consts(env) -> RobotConsts:
    """Constants of the Go2 position actuators and action mapping (env: Go2ChaseEnv)."""
    from unisupermimic.go2chase.env.robot import action_scale, actuator_constants

    robot = env.robot
    kp, kd, effort = actuator_constants()
    dev = env.device
    scale = torch.tensor(action_scale(), device=dev).expand(env.num_envs, NUM_JOINTS)
    return RobotConsts(
        default_q=robot.default_joint_pos.clone(),
        action_scale=scale.clone(),
        kp=torch.tensor(kp, device=dev),
        kd=torch.tensor(kd, device=dev),
        effort=torch.tensor(effort, device=dev),
        soft_limits=robot.soft_joint_pos_limits.clone(),
    )


def limb_positions_w(robot) -> dict[str, torch.Tensor]:
    """World-frame hip, knee and foot positions (each (N, 4, 3), legs FR FL RR RL) of the
    env's RobotView."""
    return robot.limb_positions_w()


def robot_state(env, field=None) -> torch.Tensor:
    """World-model state of the robot. On terrain (a HeightField) the base height
    is measured above the ground under the base."""
    d = env.robot
    heading = d.heading_w
    v_yaw = yaw_rotate_inverse(d.root_link_lin_vel_w, heading)
    limbs = limb_positions_w(env.robot)
    rel = []
    for name in ("hips", "knees", "feet"):
        r = limbs[name] - d.root_link_pos_w.unsqueeze(1)
        rel.append(torch.stack([yaw_rotate_inverse(r[:, i], heading) for i in range(4)], 1))
    base_h = d.root_link_pos_w[:, 2:3] - ground(d.root_link_pos_w[:, :2], field).unsqueeze(-1)
    return torch.cat(
        [
            base_h,
            d.projected_gravity_b,
            v_yaw,
            d.root_link_ang_vel_b,
            d.joint_pos - d.default_joint_pos,
            d.joint_vel,
        ]
        + [r.reshape(-1, 12) for r in rel],
        dim=-1,
    )


def planar_pose(env) -> torch.Tensor:
    d = env.robot
    return torch.cat([d.root_link_pos_w[:, :2], d.heading_w.unsqueeze(-1)], dim=-1)


def pose_delta(pose0: torch.Tensor, pose1: torch.Tensor) -> torch.Tensor:
    """Planar motion from pose0 to pose1, expressed in pose0's yaw frame."""
    c, s = pose0[:, 2].cos(), pose0[:, 2].sin()
    dx, dy = pose1[:, 0] - pose0[:, 0], pose1[:, 1] - pose0[:, 1]
    dyaw = torch.atan2((pose1[:, 2] - pose0[:, 2]).sin(), (pose1[:, 2] - pose0[:, 2]).cos())
    return torch.stack([c * dx + s * dy, -s * dx + c * dy, dyaw], dim=-1)


def apply_pose_delta(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    c, s = pose[:, 2].cos(), pose[:, 2].sin()
    x = pose[:, 0] + c * delta[:, 0] - s * delta[:, 1]
    y = pose[:, 1] + s * delta[:, 0] + c * delta[:, 1]
    return torch.stack([x, y, pose[:, 2] + delta[:, 2]], dim=-1)


class ContextReader:
    """The per-env domain-randomization context xi: foot slide friction and base COM offset,
    sampled at startup by the env's context event (env/terms.py Go2ContextEvent)."""

    def __init__(self, env):
        self.env = env

    def read(self) -> torch.Tensor:
        return self.env.term.xi.clone()


def obs_frame(
    s: torch.Tensor,
    last_action: torch.Tensor,
    target_obs: torch.Tensor,
    encoder_bias: torch.Tensor,
    noise: bool,
    lidar: torch.Tensor | None = None,
    height_scan: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One frame of each actor observation term, computed from the model state."""
    frame = {
        "base_ang_vel": s[:, W_B],
        "projected_gravity": s[:, GRAV],
        "joint_pos": s[:, Q] + encoder_bias,
        "joint_vel": s[:, QD],
        "actions": last_action,
        "target": target_obs,
    }
    if lidar is not None:
        frame["lidar"] = lidar
    if height_scan is not None:
        frame["height_scan"] = height_scan
    if noise:
        for k, w in OBS_NOISE.items():
            if k in frame:
                frame[k] = frame[k] + (torch.rand_like(frame[k]) * 2 - 1) * w
    return frame


def push_history(
    hist: torch.Tensor,
    frame: dict[str, torch.Tensor],
    h: int,
    terms: tuple[tuple[str, int], ...] = OBS_TERMS,
) -> torch.Tensor:
    """Shift each term's flattened history (oldest->newest) and append the new frame."""
    out, i = [], 0
    for name, dim in terms:
        block = hist[:, i : i + dim * h]
        out.append(torch.cat([block[:, dim:], frame[name]], dim=-1))
        i += dim * h
    return torch.cat(out, dim=-1)
