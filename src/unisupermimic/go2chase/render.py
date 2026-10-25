"""Record a Go2 chase / dodge rollout and render it on CPU MuJoCo with go2-supermimic's status
card (the video of go2-supermimic scripts/eval.py --video).

  source tools/softgl.sh   # headless machines without an EGL driver
  uv run unisupermimic-go2-video --checkpoint weights/flat.pt --task flat --mode both \\
      --seconds 30 --output videos/go2_flat.mp4

The rollout runs on the chosen backend (mjwarp by default), env 0 is recorded every control
step, and the replay draws the robot and obstacles from the task's scene with the chase ball
(green sphere), thrown balls (red spheres) and bars (red capsules) added as visual geoms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

GREEN_RGBA = (0.1, 0.8, 0.2, 1.0)
RED_RGBA = (0.9, 0.2, 0.1, 1.0)
CONTROL_FPS = 50


@torch.inference_mode()
def record(
    checkpoint: str | Path,
    task: str = "flat",
    mode: str | None = "both",
    seconds: float = 30.0,
    backend: str = "mjwarp",
    difficulty: float = 1.0,
    terrain_type: str | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Roll out env 0 and return its per-step drawing state and the scene to draw it in."""
    from unisupermimic.go2chase import evaluate
    from unisupermimic.go2chase.env.builder import make_env, scene_for
    from unisupermimic.go2chase.target import obstacles as ob
    from unisupermimic.go2chase.target.collision import bar_axis
    from unisupermimic.go2chase.target.generator import BALL
    from unisupermimic.go2chase.viz import status_lines

    overrides = []
    if mode is not None:
        probs = ",".join(map(str, evaluate.MODE_PROBS[mode]))
        overrides.append(f"env.commands.target.mode_probs=[{probs}]")
    env = make_env(1, backend, play=True, overrides=overrides, task=task)
    term = env.command_manager.get_term("target")
    term.gen.cfg.curriculum_steps = evaluate.CURRICULUM_STEPS
    term.cfg.terrain_curriculum_steps = evaluate.CURRICULUM_STEPS
    term.step_offset = int(difficulty * evaluate.CURRICULUM_STEPS)
    if term.obstacle_layout is not None:  # always some obstacles on camera
        term.cfg.target.obstacles.empty_prob = 0.0
    if terrain_type is not None:
        term.terrain.types[0] = term.terrain.columns.index(terrain_type)
    obs, _ = env.reset()
    actor = evaluate.load_actor(checkpoint, obs["obs"].shape[-1], torch.device(env.device))
    robot, gen = term.robot, term.gen
    frames: list[dict[str, Any]] = []
    policy_obs = obs["obs"]
    for _ in range(int(seconds * CONTROL_FPS)):
        policy_obs = env.step(actor(policy_obs)).obs["obs"]
        lift = float(term.ground_under_base()[0])  # ball frame -> world
        state = {
            "qpos": torch.cat(
                (robot.root_link_pos_w[0], robot.root_link_quat_w[0], robot.joint_pos[0])
            )
            .cpu()
            .numpy(),
            "rows": status_lines(term, 0),
            "balls": [],
            "bars": [],
        }
        if bool(term.has_chase[0]):
            state["balls"].append((gen.chase_pos[0].cpu().numpy() + [0, 0, lift], GREEN_RGBA))
        if bool(term.has_dodge[0]):
            pos = gen.dodge_pos[0].cpu().numpy() + [0, 0, lift]
            if int(gen.kind[0]) == BALL:
                state["balls"].append((pos, RED_RGBA))
            else:
                axis = bar_axis(gen.state(), robot.root_link_pos_w[:, :2])[0].cpu().numpy()
                offset = np.array([axis[0], axis[1], 0.0]) * float(gen.half_len[0])
                state["bars"].append((pos - offset, pos + offset))
        if term.obstacle_layout is not None:  # static in an episode: x, y, yaw per slot
            layout = ob.unpack(gen.obst[:1])[0].cpu().numpy()
            parked = [[100.0 + 5.0 * k, 100.0, 0.0] for k in range(len(layout))]
            state["obstacles"] = np.concatenate(
                [row[:3] if row[3] > 0.5 else parked[k] for k, row in enumerate(layout)]
            )
        frames.append(state)
    env.close()
    return frames, scene_for(task)


def render(
    frames: list[dict[str, Any]],
    scene: Path,
    output: Path,
    *,
    target_cfg: Any = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 25,
    distance: float = 4.0,
    elevation: float = -20.0,
) -> int:
    """Draw recorded frames (every CONTROL_FPS / fps step) into an mp4; returns the count."""
    import imageio.v2 as imageio
    import mujoco as mujoco_module

    from unisupermimic.go2chase.target.generator import BallTargetCfg
    from unisupermimic.go2chase.viz import status_card

    mujoco: Any = mujoco_module
    cfg = target_cfg or BallTargetCfg()
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = distance, elevation, 135.0
    eye = np.eye(3).flatten()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with imageio.get_writer(
            output, fps=fps, codec="libx264", quality=8, macro_block_size=None
        ) as writer:
            writer = cast(Any, writer)  # imageio's writer type lacks append_data
            for state in frames[:: CONTROL_FPS // fps]:
                data.qpos[:19] = state["qpos"]
                if "obstacles" in state:
                    data.qpos[19 : 19 + len(state["obstacles"])] = state["obstacles"]
                mujoco.mj_kinematics(model, data)
                camera.lookat[:] = [state["qpos"][0], state["qpos"][1], state["qpos"][2]]
                renderer.update_scene(data, camera=camera)
                scn = renderer.scene
                for pos, rgba in state["balls"]:
                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_initGeom(
                            scn.geoms[scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([cfg.ball_radius, 0, 0]),
                            np.asarray(pos, float),
                            eye,
                            np.asarray(rgba, np.float32),
                        )
                        scn.ngeom += 1
                for start, end in state["bars"]:
                    if scn.ngeom < scn.maxgeom:
                        geom = scn.geoms[scn.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3),
                            np.zeros(3),
                            eye,
                            np.asarray(RED_RGBA, np.float32),
                        )
                        mujoco.mjv_connector(
                            geom,
                            mujoco.mjtGeom.mjGEOM_CAPSULE,
                            cfg.bar_radius,
                            np.asarray(start, float),
                            np.asarray(end, float),
                        )
                        scn.ngeom += 1
                writer.append_data(status_card(renderer.render(), state["rows"]))
                written += 1
    finally:
        renderer.close()
    return written


def main(argv: list[str] | None = None) -> None:
    from unisupermimic.go2chase import evaluate

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("flat", "obstacles", "terrain"), default="flat")
    parser.add_argument("--mode", choices=[*evaluate.MODE_PROBS, "task"], default="both")
    parser.add_argument("--terrain", default=None, help="put the robot on this terrain type")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--backend", default="mjwarp", choices=("mjwarp", "mujoco"))
    parser.add_argument("--output", type=Path, default=Path("videos/go2_chase.mp4"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args(argv)
    frames, scene = record(
        args.checkpoint,
        args.task,
        None if args.mode == "task" else args.mode,
        args.seconds,
        args.backend,
        args.difficulty,
        args.terrain,
    )
    # go2-supermimic eval.py cameras: steeper among obstacles, closer and lower on terrain
    distance, elevation = {"obstacles": (4.0, -40.0), "terrain": (3.2, -16.0)}.get(
        args.task, (4.0, -20.0)
    )
    written = render(
        frames,
        scene,
        args.output,
        width=args.width,
        height=args.height,
        distance=distance,
        elevation=elevation,
    )
    print(f"Wrote {written} frames to {args.output}")


if __name__ == "__main__":
    main()
