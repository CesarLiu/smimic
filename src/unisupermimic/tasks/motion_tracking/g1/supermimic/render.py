"""Render policy rollouts with a translucent reference-motion ghost.

One panel per checkpoint, side by side, each with its running score and a banner once it
falls. The rollout is the deterministic evaluation itself (optionally with the push test), so
any backend works; ``--backend mujoco`` with ``CUDA_VISIBLE_DEVICES=`` keeps the GPU free.
Headless machines without an EGL driver: ``source tools/softgl.sh`` (Mesa llvmpipe).

  unisupermimic-video --ckpt runs/a/ckpt_012000.pt runs/b/ckpt_012000.pt \
      --labels "baseline" "push" --clip dance2_subject4 walk1_subject1 --push_speed 1.0
"""

from __future__ import annotations

import argparse
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat, unflatten_state
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SUPPORTED_BACKENDS
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import (
    _evaluate_chunk,
    load_policy,
    make_eval_env,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD, MDP

if TYPE_CHECKING:
    from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SuperMimicManagerBasedRlEnv

CONTROL_FPS = 50
DEFAULT_SCENE = Path("src/unisupermimic/assets/robots/g1/scene_supermimic.xml")


def flattened_qpos(state: torch.Tensor) -> torch.Tensor:
    num_joints = (state.shape[-1] - 13) // 2
    return torch.cat((state[..., :7], state[..., 13 : 13 + num_joints]), dim=-1)


def reference_qpos(motion: Any, frames: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (motion.root_p[frames], mat_to_quat(motion.root_R[frames]), motion.q[frames]),
        dim=-1,
    )


@torch.inference_mode()
def rollout_qpos(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    clip_name: str,
    seconds: float,
    start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    mdp = cast(MDP, env.supermimic_mdp)
    motion = mdp.motion
    if clip_name not in motion.names:
        raise ValueError(f"Unknown motion clip: {clip_name}")
    clip_index = motion.names.index(clip_name)
    max_frames = max(1, round((start + seconds) * CONTROL_FPS))
    args = checkpoint["args"]
    hist = int(args["hist"])
    goal_frames = int(args["goal_frames"])
    policy = load_policy(env, checkpoint)

    _, reset_info = env.reset_to_frames(motion.clip_start.clone())
    current = reset_info[SUPERMIMIC_STATE_KEY]
    history = None
    states = [current["s"][clip_index].cpu()]
    frames = [int(current["f"][clip_index].item())]
    while len(states) < max_frames:
        obs = mdp.obs(
            unflatten_state(current["s"]),
            current["a_prev"],
            current["xi"],
            current["f"],
        )
        if history is None:
            history = obs.unsqueeze(1).repeat(1, hist, 1)
        else:
            history = torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
        action = policy(history, mdp.goal(current["f"], goal_frames))
        state = env.step(action)
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        if bool(transition["done"][clip_index].item()):
            break
        current = state.info[SUPERMIMIC_STATE_KEY]
        states.append(current["s"][clip_index].cpu())
        frames.append(int(current["f"][clip_index].item()))

    start_frame = round(start * CONTROL_FPS)
    if start_frame >= len(states):
        raise ValueError("start exceeds the completed rollout")
    policy_qpos = flattened_qpos(torch.stack(states)).numpy()
    frame_ids = torch.tensor(frames, dtype=torch.long, device=motion.root_p.device)
    reference = reference_qpos(motion, frame_ids).cpu().numpy()
    return reference[start_frame:], policy_qpos[start_frame:]


@dataclass
class Panel:
    """One checkpoint's rollout of one clip, one row per control step (50 Hz); rows after
    the episode ended repeat its last pose."""

    label: str
    policy: np.ndarray  # (T, 36) root position, wxyz quaternion, joint positions
    reference: np.ndarray  # (T, 36) reference pose of the same frames
    score: np.ndarray  # (T,) normalized return so far (reward per step / max step reward)
    fall: int | None  # control step of the termination, None if it never fell
    offset: int = 0  # clip time of row 0, in control steps (after trim)


@torch.inference_mode()
def rollout_panels(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    clip_names: list[str],
    seconds: float,
    label: str = "",
    push_speed: float = 0.0,
    push_every: int = 150,
    push_seed: int = 0,
) -> dict[str, Panel]:
    """Roll out ``clip_names`` from their first frames for ``seconds`` with the deterministic
    evaluation loop (and its push test), recording one panel per clip."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    mdp = cast(MDP, env.supermimic_mdp)
    motion = mdp.motion
    unknown = [name for name in clip_names if name not in motion.names]
    if unknown:
        raise ValueError(f"Unknown motion clips: {unknown}")
    if len(clip_names) > env.num_envs:
        raise ValueError(f"{len(clip_names)} clips need at least as many environments")
    clip_ids = torch.tensor([motion.names.index(name) for name in clip_names], device=env.device)
    k = len(clip_names)
    rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    rewards = torch.zeros(k, device=env.device)
    falls: list[int | None] = [None] * k
    steps = torch.zeros(k, device=env.device)

    def record(transition: dict[str, torch.Tensor], alive: torch.Tensor) -> None:
        alive = alive[:k]
        frames = transition["f"][:k]
        if not rows:
            rows.append((transition["s"][:k].clone(), frames.clone(), torch.zeros(k)))
        last_state, last_frame, last_score = rows[-1]
        rewards.add_(torch.where(alive, transition["r"][:k], 0.0))
        steps.add_(alive.float())
        next_frame = torch.minimum(frames + 1, motion.clip_end[clip_ids])
        rows.append(
            (
                torch.where(alive.unsqueeze(-1), transition["s1"][:k], last_state),
                torch.where(alive, next_frame, last_frame),
                torch.where(
                    alive,
                    rewards / steps.clamp(min=1) / MAX_STEP_REWARD,
                    last_score.to(rewards.device),
                ).cpu(),
            )
        )
        for index in (alive & transition["term"][:k]).nonzero().flatten().tolist():
            falls[index] = len(rows) - 2  # the step that terminated

    args = checkpoint["args"]
    generator = (
        torch.Generator(device=env.device).manual_seed(push_seed) if push_speed > 0 else None
    )
    _evaluate_chunk(
        env,
        load_policy(env, checkpoint),
        mdp,
        clip_ids,
        int(args["goal_frames"]),
        int(args["hist"]),
        push_speed,
        push_every,
        generator,
        on_step=record,
        max_steps=max(1, round(seconds * CONTROL_FPS)),
    )
    states = torch.stack([row[0] for row in rows], dim=1)  # (k, T, state)
    frames = torch.stack([row[1] for row in rows], dim=1)
    scores = torch.stack([row[2] for row in rows], dim=1)
    return {
        name: Panel(
            label=label,
            policy=flattened_qpos(states[index]).cpu().numpy(),
            reference=reference_qpos(motion, frames[index]).cpu().numpy(),
            score=scores[index].numpy(),
            fall=falls[index],
        )
        for index, name in enumerate(clip_names)
    }


def trim(panel: Panel, start: float) -> Panel:
    """Drop the first ``start`` seconds (a fall before then shows from the first row)."""
    first = round(start * CONTROL_FPS)
    if first >= len(panel.policy):
        raise ValueError("start exceeds the completed rollout")
    fall = None if panel.fall is None else max(panel.fall - first, -1)
    return Panel(
        panel.label,
        panel.policy[first:],
        panel.reference[first:],
        panel.score[first:],
        fall,
        panel.offset + first,
    )


def _panel_frame(image: np.ndarray, panel: Panel, index: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    frame = Image.fromarray(image)
    draw = ImageDraw.Draw(frame)
    width, height = frame.size
    draw.rectangle((0, 0, width, 30), fill=(20, 20, 20))
    draw.text((10, 5), panel.label, fill=(255, 255, 255), font=label_font())
    score = panel.score[min(index, len(panel.score) - 1)]
    status = f"t={(index + panel.offset) / CONTROL_FPS:5.1f}s  score {score:.2f}"
    right = draw.textbbox((0, 0), status, font=label_font())[2]
    draw.text((width - right - 10, 5), status, fill=(220, 220, 220), font=label_font())
    if panel.fall is not None and index > panel.fall:
        draw.rectangle((0, height - 32, width, height), fill=(200, 50, 50))
        text = f"FELL at t={(panel.fall + panel.offset + 1) / CONTROL_FPS:.1f}s"
        draw.text((10, height - 28), text, fill=(255, 255, 255), font=label_font())
    draw.line((0, 0, 0, height), fill=(250, 250, 248), width=2)  # separates the panels
    return np.asarray(frame)


def render_panels(
    panels: list[Panel],
    output: Path,
    scene: Path = DEFAULT_SCENE,
    *,
    caption: str,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    global_ghost: bool = False,
) -> int:
    """Write the panels side by side under a caption bar; returns the number of frames.
    The video lasts as long as the longest panel; shorter ones (a rollout that ended early,
    e.g. every clip of that checkpoint fell) hold their last row."""
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("need at least one panel")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must be an .mp4 path")
    if fps <= 0 or CONTROL_FPS % fps != 0:
        raise ValueError(f"fps must be a positive divisor of {CONTROL_FPS}")
    length = max(len(panel.policy) for panel in panels)
    model, data, camera, renderer = _comparison_renderer(scene, width, height)
    bar = Image.new("RGB", (width * len(panels), 30), (250, 250, 248))
    ImageDraw.Draw(bar).text((10, 5), caption, fill=(30, 30, 30), font=label_font())
    caption_bar = np.asarray(bar)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with _video_writer(output, fps) as writer:
            for index in range(0, length, CONTROL_FPS // fps):
                images = []
                for panel in panels:
                    row = min(index, len(panel.policy) - 1)
                    _pose(model, data, panel.policy[row], panel.reference[row], global_ghost)
                    camera.lookat[:] = [panel.policy[row, 0], panel.policy[row, 1], 0.75]
                    renderer.update_scene(data, camera=camera)
                    images.append(_panel_frame(renderer.render(), panel, index))
                writer.append_data(np.concatenate((caption_bar, np.concatenate(images, 1)), 0))
                written += 1
    finally:
        renderer.close()
    return written


def comparison_xml(scene: Path) -> str:
    xml = scene.read_text(encoding="utf-8")
    robot_xml = scene.with_name("supermimic.xml").read_text(encoding="utf-8")
    body = robot_xml[robot_xml.index('<body name="pelvis"') : robot_xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r'rgba="[^"]*"', 'rgba="0.35 0.65 1.0 0.25"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    xml = xml.replace("</worldbody>", ghost + "</worldbody>")
    return re.sub(r"\s*<keyframe>.*?</keyframe>", "", xml, flags=re.DOTALL)


def load_comparison_model(scene: Path) -> Any:
    import mujoco

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=scene.parent,
            prefix=".render_",
            suffix=".xml",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(comparison_xml(scene))
        return mujoco.MjModel.from_xml_path(str(temporary))  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def label_font() -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        return ImageFont.load_default()


def make_comparison_frame(image: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have HxWx3 shape")
    frame = Image.fromarray(image)
    draw = ImageDraw.Draw(frame)
    labels = (
        ((16, 12), label),
        ((16, image.shape[0] - 38), "solid: SuperMimic   blue ghost: Reference"),
    )
    for xy, text in labels:
        bounds = draw.textbbox(xy, text, font=label_font())
        draw.rectangle(
            (bounds[0] - 6, bounds[1] - 4, bounds[2] + 6, bounds[3] + 4),
            fill=(255, 255, 255),
        )
        draw.text(xy, text, fill=(20, 20, 20), font=label_font())
    return np.asarray(frame)


def _comparison_renderer(scene: Path, width: int, height: int) -> tuple[Any, Any, Any, Any]:
    import mujoco

    mujoco = cast(Any, mujoco)
    model = load_comparison_model(scene)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 3.6
    camera.azimuth = 135
    camera.elevation = -12
    renderer = mujoco.Renderer(model, height=height, width=width)
    return model, data, camera, renderer


def _video_writer(output: Path, fps: int) -> Any:
    import imageio.v2 as imageio

    return imageio.get_writer(
        output,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


def _pose(model: Any, data: Any, policy: np.ndarray, reference: np.ndarray, global_ghost: bool):
    import mujoco

    data.qpos[:36] = policy
    data.qpos[36:] = reference
    if not global_ghost:
        data.qpos[36:38] = policy[:2]
    cast(Any, mujoco).mj_kinematics(model, data)


def render_comparison(
    reference: np.ndarray,
    policy: np.ndarray,
    output: Path,
    scene: Path = DEFAULT_SCENE,
    *,
    clip_name: str,
    width: int = 960,
    height: int = 540,
    fps: int = 25,
    global_ghost: bool = False,
) -> int:
    if reference.shape != policy.shape or reference.ndim != 2:
        raise ValueError("reference and policy qpos must have matching frame x qpos shapes")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must be an .mp4 path")
    if fps <= 0 or CONTROL_FPS % fps != 0:
        raise ValueError(f"fps must be a positive divisor of {CONTROL_FPS}")
    model, data, camera, renderer = _comparison_renderer(scene, width, height)
    if reference.shape[1] * 2 != model.nq:
        raise ValueError(f"comparison model expects {model.nq // 2} qpos columns")
    output.parent.mkdir(parents=True, exist_ok=True)
    stride = CONTROL_FPS // fps
    written = 0
    try:
        with _video_writer(output, fps) as writer:
            writer = cast(Any, writer)
            for index in range(0, len(reference), stride):
                _pose(model, data, policy[index], reference[index], global_ghost)
                camera.lookat[:] = [policy[index, 0], policy[index, 1], 0.75]
                renderer.update_scene(data, camera=camera)
                label = f"{clip_name}  t={index / CONTROL_FPS:4.1f}s"
                writer.append_data(make_comparison_frame(renderer.render(), label))
                written += 1
    finally:
        renderer.close()
    return written


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        nargs="+",
        required=True,
        help="one panel per checkpoint (runs may differ)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="panel titles (default: run/checkpoint)"
    )
    parser.add_argument(
        "--backend", choices=SUPPORTED_BACKENDS, default="mjwarp", help="Simulation backend"
    )
    parser.add_argument(
        "--clip",
        nargs="+",
        default=["dance1_subject1"],
        help="several clips write <output stem>_<clip>.mp4 each",
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--start", type=float, default=0.0, help="skip the first seconds")
    parser.add_argument(
        "--push_speed",
        type=float,
        default=0.0,
        help="push test: horizontal kick in m/s (mjwarp backend)",
    )
    parser.add_argument("--push_every", type=int, default=150)
    parser.add_argument("--push_seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/supermimic_comparison.mp4")
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--width", type=int, default=640, help="per panel")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--global-ghost", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    from unisupermimic.sim2sim import policy_load_dim_guard

    args = _argument_parser().parse_args(argv)
    labels = args.labels or [f"{path.parent.name}/{path.stem}" for path in args.ckpt]
    if len(labels) != len(args.ckpt):
        raise SystemExit("need one label per checkpoint")
    by_run: dict[Path, list[int]] = defaultdict(list)
    for index, path in enumerate(args.ckpt):
        by_run[path.parent].append(index)
    panels: list[dict[str, Panel]] = [{} for _ in args.ckpt]
    for run_dir, indices in by_run.items():  # one environment per run (its own config)
        env = make_eval_env(run_dir, backend=args.backend, max_envs=len(args.clip))
        try:
            device = torch.device(env.device)
            for index in indices:
                checkpoint = torch.load(args.ckpt[index], map_location=device, weights_only=True)
                with policy_load_dim_guard(
                    env_obs_dim=env.supermimic_mdp.obs_dim,
                    env_action_dim=env.action_manager.total_action_dim,
                    algo_name="supermimic",
                ):
                    panels[index] = rollout_panels(
                        env,
                        checkpoint,
                        args.clip,
                        args.start + args.seconds,
                        labels[index],
                        args.push_speed,
                        args.push_every,
                        args.push_seed,
                    )
        finally:
            env.close()
            cast(Any, env.backend).close()
    if args.start > 0:
        panels = [{clip: trim(p, args.start) for clip, p in run.items()} for run in panels]
    for clip in args.clip:
        output = (
            args.output
            if len(args.clip) == 1
            else (args.output.with_name(f"{args.output.stem}_{clip}.mp4"))
        )
        caption = f"{clip}   blue ghost: reference motion"
        if args.push_speed > 0:
            caption += f"   push {args.push_speed:g} m/s every {args.push_every / CONTROL_FPS:g}s"
        written = render_panels(
            [run[clip] for run in panels],
            output,
            args.scene,
            caption=caption,
            width=args.width,
            height=args.height,
            fps=args.fps,
            global_ghost=args.global_ghost,
        )
        falls = {run[clip].label: run[clip].fall for run in panels}
        print(f"Wrote {written} frames to {output}  (fall steps: {falls})", flush=True)


__all__ = [
    "Panel",
    "comparison_xml",
    "flattened_qpos",
    "load_comparison_model",
    "main",
    "make_comparison_frame",
    "reference_qpos",
    "render_comparison",
    "render_panels",
    "rollout_panels",
    "rollout_qpos",
    "trim",
]

if __name__ == "__main__":
    main()
