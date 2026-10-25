"""Target command term: owns the chase ball and the dodge obstacles (UniLab port).

Ported from go2-supermimic env/command.py. Each episode samples which are in play: chase
only, dodge only, or both. ``command`` is the two-slot target observation the actor sees
(target/interface.py), filled from ground truth. Three tasks (``TargetCommandCfg.task``):

- ``flat``: open ground.
- ``obstacles``: tall static boxes (heavy boxes on planar joints, placed per episode) and a
  planar LiDAR.
- ``terrain``: go2-supermimic's generated terrain; ball heights, the base height and every
  ball interaction are measured above the ground under the robot (the ball frame), as in
  imagination, where the world-model state holds the base height above ground.

Differences from the mjlab version, imposed by UniLab's lifecycle:

- The reset root pose is read from the reset transaction (``read_reset_root_pose``);
  mjlab read qpos. The tile origin of a terrain env is chosen by ``place`` (called by the
  root reset event, env/terms.py Go2ResetBase) instead of mjlab's env origins.
- UniLab recomputes ``common_step_counter`` every step, so the curricula read it plus
  ``step_offset`` (set by evaluation to pin the difficulty).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch
from unilab.managers import CommandTerm, CommandTermCfg

from unisupermimic.go2chase.target import interface as ti
from unisupermimic.go2chase.target import obstacles as ob
from unisupermimic.go2chase.target import terrain as tr
from unisupermimic.go2chase.target.collision import (
    dodge_clearance,
    obstacle_penetration,
    predicted_clearance,
    robot_geometry,
)
from unisupermimic.go2chase.target.generator import BallTarget, BallTargetCfg

from .robot import BODY_NAMES, RobotView

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers.types import ManagerBasedRlEnv

TASKS = ("flat", "obstacles", "terrain")


def configure_target(cfg: BallTargetCfg, task: str) -> None:
    """go2-supermimic env_cfg.go2_chase_env_cfg's per-task target settings."""
    if task == "terrain":
        # Chase only, or chase while jumping / ducking thrown bars (no balls).
        cfg.kind_probs = (0.0, 0.5, 0.5)
        cfg.bar_speed = (1.2, 2.0)
        cfg.bar_speed_final = (2.0, 3.0)
        cfg.bars_follow_ground = True
        # The robot is a little slower on rough ground, so the chase ball is too.
        cfg.flee_ratio = (0.7, 0.85)
        cfg.flee_ratio_final = (0.8, 1.0)
        cfg.speed_max_final = (1.8, 2.6)
        cfg.escape_speed = (2.5, 3.5)
        cfg.ou_sigma = 3.5
    elif task == "obstacles":
        cfg.obstacles = ob.ObstacleCfg(enabled=True)
        # Running among obstacles the robot is slower, so throws are slower too.
        cfg.throw_speed = (2.5, 3.5)
        cfg.throw_speed_final = (3.0, 4.5)
        cfg.max_time_to_robot = 0.75


def _yaw_of_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class TargetCommand(CommandTerm):
    cfg: TargetCommandCfg

    def __init__(self, cfg: TargetCommandCfg, env: ManagerBasedRlEnv):
        if cfg.task not in TASKS:
            raise ValueError(f"TargetCommandCfg task must be one of {TASKS}, got {cfg.task!r}")
        configure_target(cfg.target, cfg.task)
        super().__init__(cfg, env)
        self.robot = RobotView(env, cfg.entity_name)
        device = self._device
        # BallTarget creates its state tensors with setattr, so type it loosely
        self.gen: Any = BallTarget(cfg.target, self.num_envs, str(device))
        self.step_offset = 0
        self.center = env.scene.env_origins(device).clone()  # arena centres (N, 3)
        # static obstacles: one entity with each box's planar joints (x, y, yaw); UniLab's
        # mjwarp reset does not support mocap bodies (mjlab moved mocap boxes)
        self._obstacles: Entity | None = None
        if cfg.target.obstacles.enabled:
            self._obstacles = cast("Entity", env.scene["obstacles"])
            if len(self._obstacles.joint_names) != 3 * ob.NUM_OBSTACLES:
                raise ValueError("the obstacles entity needs x, y, yaw joints for every slot")
        self.terrain = None
        self.height_field: tr.HeightField | None = None
        if cfg.task == "terrain":
            from .builder import scene_for
            from .terrain import TerrainLayout, height_field

            self.terrain = TerrainLayout(self.num_envs, device)
            self.height_field = height_field(
                scene_for("terrain"),
                self.terrain,
                cfg.target.arena_radius + 1.5,
                cfg.field_res,
                device,
            )
        self.gen.field = self.height_field
        # Adaptive curriculum: running throws / hits per obstacle kind.
        self._prev_flying = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.kind_throws = torch.zeros(3, device=device)
        self.kind_hits = torch.zeros(3, device=device)
        self.kind_hit_rate = torch.zeros(3, device=device)
        # Dodge hit detected in the last update; the reward reads it on the next step.
        self.hit = torch.zeros(self.num_envs, device=device)
        self._obs = torch.zeros(self.num_envs, ti.DIM, device=device)
        # domain-randomization context, sampled by env/terms.py Go2ContextEvent at startup:
        # foot friction (slide, spin, roll), base COM offset, and xi = (slide, COM offset)
        self.friction = torch.ones(self.num_envs, 3, device=device)
        self.com_offset = torch.zeros(self.num_envs, 3, device=device)
        self.xi = torch.zeros(self.num_envs, 4, device=device)
        names = ["near", "tags", "hits", "dist"]
        if self._obstacles is not None:
            names.append("obstacle_contact")
        for name in names:
            self.metrics[name] = torch.zeros(self.num_envs, device=device)

    # Ground truth accessors.

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        """Bodies compiled into the scene's packed read (RobotView reads them)."""
        return BODY_NAMES

    @property
    def command(self) -> torch.Tensor:
        return self._obs

    @property
    def has_chase(self) -> torch.Tensor:
        return self.gen.has_chase

    @property
    def has_dodge(self) -> torch.Tensor:
        return self.gen.has_dodge

    @property
    def progress_step(self) -> int:
        return int(getattr(self._env, "common_step_counter", 0)) + self.step_offset

    @property
    def obstacle_layout(self) -> torch.Tensor | None:
        """Flat obstacle layout (N, K*4), or None without obstacles."""
        return self.gen.obst if self.cfg.target.obstacles.enabled else None

    def ground_under_base(self) -> torch.Tensor:
        """Terrain height under the base (N,); 0 on flat ground."""
        return tr.ground(self.robot.root_link_pos_w[:, :2], self.height_field)

    def base_local(self) -> torch.Tensor:
        """Base position with z above the ground under the base (the ball frame)."""
        pos = self.robot.root_link_pos_w.clone()
        pos[:, 2] -= self.ground_under_base()
        return pos

    def scan(self) -> torch.Tensor:
        """Base height above each point of the local ground grid (N, P)."""
        robot = self.robot
        relief = tr.relief(
            robot.root_link_pos_w[:, :2], robot.heading_w, self.height_field, self.obstacle_layout
        )
        return self.base_local()[:, 2:3] + relief

    def lidar(self) -> torch.Tensor:
        robot = self.robot
        ocfg = self.cfg.target.obstacles
        return ob.lidar(
            robot.root_link_pos_w[:, :2],
            robot.heading_w,
            ob.unpack(self.gen.obst),
            ocfg.lidar_rays,
            ocfg.lidar_range,
        )

    def head_pos_w(self) -> torch.Tensor:
        return self.robot.head_pos_w()

    def head_ball_dist(self) -> torch.Tensor:
        """Horizontal distance from the robot head to the chase ball."""
        return (self.gen.chase_pos[:, :2] - self.head_pos_w()[:, :2]).norm(dim=-1)

    def geometry(self) -> dict[str, torch.Tensor]:
        """Robot collision geometry in the ball frame: base pose plus hips, knees, feet."""
        robot = self.robot
        pos = robot.root_link_pos_w
        g = self.ground_under_base()
        lift = torch.stack([torch.zeros_like(g), torch.zeros_like(g), g], -1)[:, None]
        limbs = {k: v - lift for k, v in robot.limb_positions_w().items()}
        return robot_geometry(pos[:, :2], pos[:, 2] - g, robot.heading_w, limbs)

    def body_clearance(self) -> torch.Tensor:
        """Gap between the dodge obstacle and the robot (<= 0 means touching)."""
        return dodge_clearance(self.gen.state(), self.geometry(), self.cfg.target)

    def predicted_body_clearance(self) -> torch.Tensor:
        """Clearance at the thrown obstacle's predicted closest approach."""
        return predicted_clearance(self.gen.state(), self.geometry(), self.cfg.target)

    def obstacle_penetration(self) -> torch.Tensor:
        """How deep the body reaches into the (slightly inflated) static obstacles."""
        if self._obstacles is None:
            return torch.zeros(self.num_envs, device=self._device)
        return obstacle_penetration(self.geometry(), self.gen.obst)

    def _update_kind_curriculum(self, hit: torch.Tensor) -> None:
        """Throw more of the obstacle kinds that hit the robot most often."""
        launched = self.gen.flying & ~self._prev_flying & self.has_dodge
        self._prev_flying = self.gen.flying.clone()
        kind = self.gen.kind.long()
        self.kind_throws += torch.bincount(kind[launched], minlength=3).float()
        self.kind_hits += torch.bincount(kind[hit], minlength=3).float()
        if self.kind_throws.sum() >= self.cfg.kind_curriculum_window:
            rate = self.kind_hits / self.kind_throws.clamp(min=1.0)
            self.kind_hit_rate = 0.5 * self.kind_hit_rate + 0.5 * rate
            base = torch.tensor(self.cfg.target.kind_probs, device=self._device)
            self.gen.set_kind_weights(base * (self.cfg.kind_curriculum_floor + self.kind_hit_rate))
            self.kind_throws.zero_()
            self.kind_hits.zero_()

    # Reset placement (called by the root reset event, before this term's reset).

    def place(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Arena centres (k, 3) of the reset envs: the env origins, or on terrain a tile
        origin picked by the progress curriculum (mdp.terrain_levels_by_progress)."""
        if self.terrain is not None:
            progress = self.progress_step / max(1, self.cfg.terrain_curriculum_steps)
            self.center[env_ids] = self.terrain.assign(env_ids, progress)
        return self.center[env_ids]

    def _place_obstacles(self, env_ids: torch.Tensor, reset_mask: torch.Tensor) -> None:
        """Move the obstacle boxes to this episode's layout; unused slots park far away."""
        assert self._obstacles is not None
        layout = ob.unpack(self.gen.obst[env_ids])  # (k, K, 4): cx, cy, yaw, active
        active = layout[..., 3:4] > 0.5
        slots = torch.arange(ob.NUM_OBSTACLES, device=self._device, dtype=torch.float32)
        parked = torch.stack(
            (100.0 + 5.0 * slots, torch.full_like(slots, 100.0), torch.zeros_like(slots)), -1
        )
        rows = torch.where(active, layout[..., :3], parked.expand_as(layout[..., :3]))
        # tensor resets take full-width values plus the reset mask
        qpos = torch.zeros(self.num_envs, 3 * ob.NUM_OBSTACLES, device=self._device)
        qpos[env_ids] = rows.reshape(len(env_ids), -1)
        self._obstacles.write_joint_state_to_sim_mask(qpos, torch.zeros_like(qpos), reset_mask)

    # CommandTerm interface.

    def _update_metrics(self) -> None:
        steps = max(int(getattr(self._env, "max_episode_length", 1)), 1)
        base = self.robot.root_link_pos_w[:, :2]
        d = (self.gen.chase_pos[:, :2] - base).norm(dim=-1)
        self.metrics["dist"] += torch.where(self.has_chase, d, torch.zeros_like(d)) / steps
        near = self.has_chase & (self.head_ball_dist() < self.cfg.near_radius)
        self.metrics["near"] += near.float() / steps
        if self._obstacles is not None:
            self.metrics["obstacle_contact"] += (self.obstacle_penetration() > 0.05).float() / steps

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        ids = reset_mask.nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.gen.resample(ids)

    def reset(
        self, env_ids: torch.Tensor | slice | None = None, *, publish_metrics: bool = True
    ) -> dict[str, float]:
        """Spawn this episode's targets around the robot's staged reset pose, then resample
        (go2-supermimic order)."""
        reset_mask = self._reset_mask_buffer.resolve(env_ids, owner="TargetCommand reset").clone()
        ids = reset_mask.nonzero(as_tuple=False).flatten()
        if ids.numel():
            pose = torch.as_tensor(
                self.robot.entity.read_reset_root_pose(ids), dtype=torch.float32
            ).to(self._device)
            robot_xy, robot_yaw = pose[:, :2], _yaw_of_wxyz(pose[:, 3:7])
            # Mode: 0 chase only, 1 dodge only, 2 both.
            probs = torch.tensor(self.cfg.mode_probs, device=self._device, dtype=torch.float32)
            mode = torch.multinomial(probs, len(ids), replacement=True)
            self.gen.set_progress(self.progress_step)
            self.gen.spawn(ids, robot_xy, robot_yaw, self.center[ids, :2], mode != 1, mode != 0)
            if self._obstacles is not None:
                self._place_obstacles(ids, reset_mask)
            self.hit[ids] = 0.0
        extras: dict[str, float] = super().reset(env_ids, publish_metrics=publish_metrics)
        return extras

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        """The scene read is fresh again: publish the reset envs' target observation."""
        if bool(reset_mask.any()):
            self._update_command()

    def post_compute_reset(self) -> None:
        return

    def compute(self, dt: float | torch.Tensor) -> None:
        dt_t = torch.as_tensor(dt, device=self._device, dtype=torch.float32)
        if dt_t.ndim == 0:
            dt_t = dt_t.expand(self.num_envs)
        self._advance(dt_t)
        super().compute(dt)

    def _advance(self, dt: torch.Tensor) -> None:
        robot = self.robot
        base_xy = robot.root_link_pos_w[:, :2]
        self.gen.set_progress(self.progress_step)
        self.gen.step(dt, base_xy, robot.root_link_lin_vel_w[:, :2], robot.heading_w)
        self.metrics["tags"] += self.gen.tagged.float()
        # Dodge hit: the obstacle touches the body.
        hit = self.has_dodge & (self.body_clearance() <= 0.0) & (dt > 0)
        self.hit = hit.float()
        if self.cfg.kind_curriculum:
            self._update_kind_curriculum(hit)
        self.metrics["hits"] += self.hit
        # The obstacle bounces off (balls) and cannot hit again; the generator throws
        # the next one when this one is done.
        self.gen.deflect(hit.nonzero().flatten(), base_xy[hit])

    def _update_command(self) -> None:
        self._obs = ti.oracle_observation(self.gen.state(), self.base_local(), self.robot.heading_w)


@dataclass(kw_only=True)
class TargetCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    task: str = "flat"
    """flat, obstacles or terrain (go2-supermimic's three tasks)."""
    mode_probs: tuple[float, float, float] = (0.3, 0.3, 0.4)
    """Episode mode probabilities: chase only, dodge only, both at once."""
    kind_curriculum: bool = True
    """Throw more of the obstacle kinds with the highest hit rate."""
    kind_curriculum_window: int = 2000
    """Throws between curriculum updates (across all envs)."""
    kind_curriculum_floor: float = 0.15
    """Kind weight = base probability * (floor + hit rate)."""
    near_radius: float = 0.3
    """Chase success metric only: head within this distance of the chase ball."""
    field_res: float = 0.02
    """Height-field grid [m]: bilinear sampling blurs a step over one cell."""
    terrain_curriculum_steps: int = 12_000
    """Env steps over which the harder terrain rows open up."""
    target: BallTargetCfg = field(default_factory=BallTargetCfg)

    def build(self, env: Any) -> TargetCommand:
        return TargetCommand(self, env)


__all__ = ["TASKS", "TargetCommand", "TargetCommandCfg", "configure_target"]
