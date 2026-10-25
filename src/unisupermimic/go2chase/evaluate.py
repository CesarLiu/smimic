"""Evaluate a Go2 chase checkpoint on the UniLab env (go2-supermimic scripts/eval.py).

  uv run python -m unisupermimic.go2chase.evaluate --checkpoint weights/flat.pt --mode both

Reports tags per minute, time near the chase ball, per-kind dodge hit rates with the body
height, highest foot and sideways speed while an obstacle is close, falls per minute and
mean speed, at the hardest curriculum setting, with the play config (no observation noise,
no episode time limit, domain randomization on).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

CATEGORY_NAMES = ("HIGH BALL", "MID BALL", "LOW BALL", "LOW BAR (jump)", "HIGH BAR (duck)")
MODE_PROBS = {
    "chase": (1.0, 0.0, 0.0),
    "dodge": (0.0, 1.0, 0.0),
    "both": (0.0, 0.0, 1.0),
    "mixed": (0.3, 0.3, 0.4),
}
CURRICULUM_STEPS = 10_000


def load_actor(path: str | Path, obs_dim: int, device: torch.device) -> torch.nn.Module:
    from unisupermimic.go2chase.supermimic.train import Actor

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(obs_dim, tuple(checkpoint["cfg"]["actor_hidden"]), 12).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


@torch.inference_mode()
def evaluate(
    checkpoint: str | Path,
    mode: str | None = "both",
    num_envs: int = 256,
    seconds: float = 60.0,
    difficulty: float = 1.0,
    backend: str = "mjwarp",
    task: str = "flat",
    terrain_type: str | None = None,
) -> dict:
    from unisupermimic.go2chase.env.builder import make_env

    overrides = []
    if mode is not None:
        overrides.append(f"env.commands.target.mode_probs=[{','.join(map(str, MODE_PROBS[mode]))}]")
    env = make_env(num_envs, backend, play=True, overrides=overrides, task=task)
    term = env.command_manager.get_term("target")
    gen = term.gen
    # Curricula run on env steps: pin the target and terrain difficulty at the requested
    # progress (go2-supermimic eval.py: both curricula over 10k steps).
    gen.cfg.curriculum_steps = CURRICULUM_STEPS
    term.cfg.terrain_curriculum_steps = CURRICULUM_STEPS
    term.step_offset = int(difficulty * CURRICULUM_STEPS)
    if terrain_type is not None:  # env 0 on this terrain type (sub-terrain column)
        if term.terrain is None:
            raise ValueError("--terrain needs --task terrain")
        term.terrain.types[0] = term.terrain.columns.index(terrain_type)
    obs, _ = env.reset()
    device = torch.device(env.device)
    actor = load_actor(checkpoint, obs["obs"].shape[-1], device)

    robot = term.robot
    steps = int(seconds / env.step_dt)
    n = num_envs
    near, falls, speed, tags, contact = (torch.zeros(n, device=device) for _ in range(5))
    nc = len(CATEGORY_NAMES)
    throws, hits = torch.zeros(nc, device=device), torch.zeros(nc, device=device)
    z_sum, foot_sum, side_sum, count = (torch.zeros(nc, device=device) for _ in range(4))

    def category() -> torch.Tensor:
        return torch.where(gen.kind == 0, gen.style.long(), gen.kind.long() + 2)

    prev_flying, prev_cat = gen.flying.clone(), category()
    policy_obs = obs["obs"]
    for _ in range(steps):
        state = env.step(actor(policy_obs))
        policy_obs = state.obs["obs"]
        pos, vel = robot.root_link_pos_w, robot.root_link_lin_vel_w[:, :2]
        near += ((term.head_ball_dist() < term.cfg.near_radius) & term.has_chase).float()
        tags += gen.tagged.float()
        speed += vel.norm(dim=-1)
        falls += state.terminated.float()
        contact += (term.obstacle_penetration() > 0.05).float()

        base_z = pos[:, 2] - term.ground_under_base()
        foot_z = robot.limb_positions_w()["feet"][..., 2].max(-1).values - term.ground_under_base()
        d = gen.dodge_vel[:, :2] / gen.dodge_vel[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        side = (vel * torch.stack([-d[:, 1], d[:, 0]], -1)).sum(-1).abs()
        close = (
            term.has_dodge & gen.flying & ((gen.dodge_pos[:, :2] - pos[:, :2]).norm(dim=-1) < 0.8)
        )
        launched = gen.flying & ~prev_flying & term.has_dodge
        cat = category()
        for k in range(nc):
            throws[k] += (launched & (cat == k)).sum()
            hits[k] += ((term.hit > 0) & (prev_cat == k)).sum()
            m = close & (cat == k)
            z_sum[k] += base_z[m].sum()
            foot_sum[k] += foot_z[m].sum()
            side_sum[k] += side[m].sum()
            count[k] += m.sum()
        prev_flying, prev_cat = gen.flying.clone(), cat

    minutes = seconds / 60.0
    chase = term.has_chase
    result: dict = {
        "task": task,
        "mode_probs": list(term.cfg.mode_probs),
        "num_envs": n,
        "seconds": seconds,
        "falls_per_minute": (falls.mean() / minutes).item(),
        "mean_speed": (speed / steps).mean().item(),
    }
    if bool(chase.any()):
        result["tags_per_minute"] = (tags[chase].mean() / minutes).item()
        result["time_near"] = (near[chase].mean() / steps).item()
    if term.obstacle_layout is not None:
        result["obstacle_contact"] = (contact.mean() / steps).item()
    if bool(term.has_dodge.any()):
        result["dodge"] = {
            name: {
                "throws": int(throws[k]),
                "hit_rate": float(hits[k] / max(throws[k].item(), 1)),
                "base_z": float(z_sum[k] / max(count[k].item(), 1)),
                "top_foot": float(foot_sum[k] / max(count[k].item(), 1)),
                "sideways": float(side_sum[k] / max(count[k].item(), 1)),
            }
            for k, name in enumerate(CATEGORY_NAMES)
        }
    env.close()
    return result


def report(result: dict) -> str:
    lines = [
        f"{result['task']}  modes={tuple(result['mode_probs'])}  envs={result['num_envs']}  "
        f"seconds={result['seconds']}"
    ]
    if "tags_per_minute" in result:
        lines.append(f"  chase  tags per minute     {result['tags_per_minute']:.1f}")
        lines.append(f"  chase  time within 0.3 m  {result['time_near']:.2f}")
    for name, d in result.get("dodge", {}).items():
        lines.append(
            f"  dodge  {name:15s} throws {d['throws']:5d}  hit rate {d['hit_rate']:.2f}  "
            f"base {d['base_z']:.2f} m  top foot {d['top_foot']:.2f} m  "
            f"sideways {d['sideways']:.2f} m/s"
        )
    if "obstacle_contact" in result:
        lines.append(f"  obstacle contact (time)    {result['obstacle_contact']:.3f}")
    lines.append(f"  falls per minute           {result['falls_per_minute']:.3f}")
    lines.append(f"  mean speed                 {result['mean_speed']:.2f} m/s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("flat", "obstacles", "terrain"), default="flat")
    parser.add_argument(
        "--mode",
        choices=[*MODE_PROBS, "task"],
        default="both",
        help="episode modes; 'task' keeps the task's own mix",
    )
    parser.add_argument("--terrain", default=None, help="put env 0 on this terrain type")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--backend", default="mjwarp", choices=("mjwarp", "mujoco"))
    parser.add_argument("--json", type=Path, default=None, help="also write the metrics here")
    args = parser.parse_args(argv)
    mode = None if args.mode == "task" else args.mode
    result = evaluate(
        args.checkpoint,
        mode,
        args.num_envs,
        args.seconds,
        args.difficulty,
        args.backend,
        args.task,
        args.terrain,
    )
    print(report(result))
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
