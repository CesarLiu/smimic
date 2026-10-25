"""Go2 constants and a read-plan view of the robot in go2-supermimic's conventions.

``RobotView`` replaces mjlab's ``entity.data`` for the ported code: world-frame root pose
and velocities (env origins included, as in mjlab), heading, projected gravity, joints,
and the hip / knee / foot points the world-model state and the collision model use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat

if TYPE_CHECKING:
    from unilab.base.entity import Entity

# Natural MJCF joint order (= action order): legs FL FR RL RR, each hip, thigh, calf.
JOINT_NAMES = tuple(
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")
)
# Limb points of the world-model state are ordered FR FL RR RL (go2-supermimic state.LEGS).
LEGS = ("FR", "FL", "RR", "RL")
HIP_BODIES = tuple(f"{leg}_thigh" for leg in LEGS)  # thigh frame = hip pitch joint
KNEE_BODIES = tuple(f"{leg}_calf" for leg in LEGS)  # calf frame = knee joint
BODY_NAMES = ("base", *HIP_BODIES, *KNEE_BODIES)
FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in LEGS)
FOOT_SITE_OFFSET = (-0.002, 0.0, -0.213)  # foot sites FR..RL in their calf frame
HEAD_SITE_OFFSET = (0.33, 0.0, 0.0)  # head site in the base frame
HOME_JOINT_POS = (0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8)
SOFT_LIMIT_FACTOR = 0.9

# Actuators (go2-supermimic robot/go2.py): rotor inertia reflected through the gear ratio,
# PD at a 10 Hz natural frequency with damping ratio 2, Menagerie torque limits.
ROTOR_INERTIA = 0.000111842
HIP_GEAR_RATIO = 6.33
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
HIP_EFFORT, KNEE_EFFORT = 23.7, 45.43


def _gains(gear: float) -> tuple[float, float]:
    inertia = ROTOR_INERTIA * gear**2
    return inertia * NATURAL_FREQ**2, 2 * DAMPING_RATIO * inertia * NATURAL_FREQ


def actuator_constants() -> tuple[list[float], list[float], list[float]]:
    """Per-joint stiffness, damping and effort limit, in JOINT_NAMES order."""
    hip, knee = _gains(HIP_GEAR_RATIO), _gains(KNEE_GEAR_RATIO)
    kp, kd, effort = [], [], []
    for name in JOINT_NAMES:
        (p, d), e = (knee, KNEE_EFFORT) if "calf" in name else (hip, HIP_EFFORT)
        kp.append(p)
        kd.append(d)
        effort.append(e)
    return kp, kd, effort


def action_scale() -> list[float]:
    """go2-supermimic GO2_ACTION_SCALE: 0.25 x effort limit / stiffness."""
    kp, _, effort = actuator_constants()
    return [0.25 * e / p for e, p in zip(effort, kp, strict=True)]


def heading(rotation: torch.Tensor) -> torch.Tensor:
    """Yaw of the body x axis (mjlab EntityData.heading_w)."""
    return torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])


class RobotView:
    """Robot state read from the scene's tensor read plan (one refresh per step)."""

    def __init__(self, env: Any, entity_name: str = "robot"):
        self._env = env
        self.entity: Entity = env.scene[entity_name]
        if tuple(self.entity.joint_names) != JOINT_NAMES:
            raise ValueError(
                f"Go2 chase expects joints {JOINT_NAMES}, got {self.entity.joint_names}"
            )
        device = torch.device(env.device)
        self.device = device
        self.default_joint_pos = self.entity.data.default_joint_pos(device).clone()
        # UniLab reports raw joint ranges; mjlab's soft limits shrink them about the middle
        ranges = self.entity.data.soft_joint_pos_limits(device).clone().reshape(-1, 12, 2)
        middle, half = ranges.mean(-1), (ranges[..., 1] - ranges[..., 0]) / 2 * SOFT_LIMIT_FACTOR
        soft = torch.stack((middle - half, middle + half), -1)
        self.soft_joint_pos_limits = soft.expand(env.num_envs, 12, 2).clone()
        self.foot_offset = torch.tensor(FOOT_SITE_OFFSET, device=device)
        self.head_offset = torch.tensor(HEAD_SITE_OFFSET, device=device)

    @property
    def data(self) -> RobotView:
        """mjlab's ``entity.data`` spelling (go2-supermimic viz.status_lines reads it)."""
        return self

    def _plan(self) -> Any:
        plan = self._env.scene.read_plan
        if plan is None:
            raise RuntimeError("Go2 chase requires the scene tensor read plan")
        if not plan.ready:
            plan.refresh()
        return plan

    def _bodies(self) -> Any:
        return self._plan().body_tensor_view(self.entity, BODY_NAMES)

    # root
    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self._bodies().pos_w[:, 0]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        return self._bodies().quat_w[:, 0]

    @property
    def root_rotation(self) -> torch.Tensor:
        return quat_to_mat(self.root_link_quat_w)

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        return self._bodies().lin_vel_w[:, 0]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        return self._bodies().ang_vel_w[:, 0]

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        rotation = self.root_rotation
        return (rotation.transpose(-1, -2) @ self.root_link_ang_vel_w.unsqueeze(-1)).squeeze(-1)

    @property
    def heading_w(self) -> torch.Tensor:
        return heading(self.root_rotation)

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        return -self.root_rotation[..., 2, :]  # R^T @ (0, 0, -1)

    # joints
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._plan().joint_tensor_view(self.entity).joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._plan().joint_tensor_view(self.entity).joint_vel

    @property
    def encoder_bias(self) -> torch.Tensor:
        return self.entity.data.encoder_bias_tensor.to(self.device)

    # limb points (legs FR FL RR RL)
    def limb_positions_w(self) -> dict[str, torch.Tensor]:
        bodies = self._bodies()
        calf_rotation = quat_to_mat(bodies.quat_w[:, 5:9])
        feet = bodies.pos_w[:, 5:9] + calf_rotation @ self.foot_offset
        return {"hips": bodies.pos_w[:, 1:5], "knees": bodies.pos_w[:, 5:9], "feet": feet}

    def head_pos_w(self) -> torch.Tensor:
        return self.root_link_pos_w + self.root_rotation @ self.head_offset


__all__ = [
    "BODY_NAMES",
    "FOOT_GEOM_NAMES",
    "HOME_JOINT_POS",
    "JOINT_NAMES",
    "LEGS",
    "RobotView",
    "action_scale",
    "actuator_constants",
    "heading",
]
