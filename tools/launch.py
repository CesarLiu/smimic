#!/usr/bin/env python
"""Experiment queue for UniSuperMimic extensions (H100/H200-class GPUs).

  .venv/bin/python tools/launch.py list
  .venv/bin/python tools/launch.py run push_ab                 # groups or experiment names
  .venv/bin/python tools/launch.py run push_ab --gpus 0,1      # one job per free GPU, strict FIFO
  .venv/bin/python tools/launch.py eval push_ab                # final checkpoint: nominal + push tests
  .venv/bin/python tools/launch.py eval push_ab --curve        # every saved checkpoint, nominal only
  .venv/bin/python tools/launch.py run push_ab --dry           # print the commands only
  .venv/bin/python tools/report_push.py                        # summary table of the evaluations

Every job writes to runs/<name>/ (train_log.csv, ckpt_*.pt, ckpt_*_eval*.json, launch.log).
`run` evaluates each job right after it trains unless --no-eval is given.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / ".venv" / "bin"
TRAIN = str(BIN / "unisupermimic-train")
TORCHRUN = str(BIN / "torchrun")
EVAL = str(BIN / "unisupermimic-eval")
PUSH_SPEEDS = (0.5, 1.0, 1.5)  # m/s horizontal kicks every 3 s (see evaluate.evaluate)
FINAL_ITERATION = 12000  # the default algo.iterations

# Analytic push injection: simulator kicks every 1-3 s within BeyondMimic's linear range
# (logged and replayed exactly in world-model windows) and imagined kicks at a matched rate.
PUSH_SIM = ["env.commands.motion.push_scale=1.0"]
PUSH_IMAG = ["algo.imag_push_rate=0.01", "algo.imag_push_scale=1.0"]

E: dict[str, dict] = {}


def experiment(name: str, overrides: list[str], note: str, gpus: int = 1) -> None:
    E[name] = dict(overrides=overrides, note=note, gpus=gpus)


for seed in range(3):
    s = [f"algo.seed={seed}"]
    experiment(f"uni_base_s{seed}", s, "default UniSuperMimic (paper config, 12k iterations)")
    experiment(f"uni_push_s{seed}", s + PUSH_SIM + PUSH_IMAG, "push injection: sim + imagined kicks")
    experiment(f"uni_push_sim_s{seed}", s + PUSH_SIM, "ablation: simulator kicks only")
    experiment(f"uni_push_imag_s{seed}", s + PUSH_IMAG, "ablation: imagined kicks only")

# Paper Fig. 3a-style 2-GPU run (data parallel, lr x sqrt(2) as in Table A.5)
experiment("uni_base_g2_s0", ["algo.seed=0"], "default config on 2 GPUs (torchrun)", gpus=2)
experiment("uni_push_g2_s0", ["algo.seed=0"] + PUSH_SIM + PUSH_IMAG, "push injection on 2 GPUs", gpus=2)

GROUPS = {
    "push_ab": ["uni_base_s0", "uni_push_s0"],
    "push_ablation": ["uni_push_sim_s0", "uni_push_imag_s0"],
    "push_seeds": ["uni_base_s1", "uni_push_s1", "uni_base_s2", "uni_push_s2"],
    "multi_gpu": ["uni_base_g2_s0", "uni_push_g2_s0"],
}


def expand(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        for item in GROUPS.get(name, [name]):
            if item not in E:
                sys.exit(f"unknown experiment {item!r}; see `tools/launch.py list`")
            if item not in out:
                out.append(item)
    return out


def train_command(name: str) -> list[str]:
    args = [f"training.log_dir=runs/{name}", *E[name]["overrides"]]
    n = E[name]["gpus"]
    if n > 1:  # data parallel: one rank per GPU (scripts/train_ddp.py)
        return [TORCHRUN, "--standalone", f"--nproc_per_node={n}", "-m",
                "unisupermimic.scripts.train_ddp", *args]
    return [TRAIN, *args]


def eval_commands(name: str, curve: bool) -> list[list[str]]:
    run = ROOT / "runs" / name
    if curve:
        ckpts = sorted(run.glob("ckpt_*.pt"))
        return [[EVAL, "--backend", "mjwarp", "--ckpt", str(c)] for c in ckpts]
    final = run / f"ckpt_{FINAL_ITERATION:06d}.pt"
    commands = [[EVAL, "--backend", "mjwarp", "--ckpt", str(final)]]
    commands += [
        [EVAL, "--backend", "mjwarp", "--ckpt", str(final), "--push_speed", f"{v:g}"]
        for v in PUSH_SPEEDS
    ]
    return commands


def job_script(name: str, mode: str, curve: bool, evaluate: bool) -> str:
    """One shell line per step, so a job is a single sequential process per GPU."""
    steps = []
    if mode == "run":
        steps.append(train_command(name))
    if mode == "eval" or evaluate:
        if mode == "run" and curve:
            curve = False  # checkpoints are listed at job start; evaluate the curve separately
        steps += eval_commands(name, curve) if mode == "eval" else eval_commands(name, False)
    return " && ".join(" ".join(f"'{part}'" if "[" in part else part for part in step) for step in steps)


def schedule(names: list[str], gpus: list[int], mode: str, dry: bool, curve: bool, evaluate: bool):
    free, queue, running = list(gpus), list(names), []
    while queue or running:
        while queue:
            name = queue[0]
            need = E[name]["gpus"]
            if need > len(gpus):
                print(f"[skip] {name} needs {need} GPUs, only {len(gpus)} given")
                queue.pop(0)
                continue
            if need > len(free):
                break  # strict FIFO: a big job is never starved by later small ones
            use, free = free[:need], free[need:]
            script = job_script(name, mode, curve, evaluate)
            print(f"[start] {name} on GPU {use}: {script}", flush=True)
            queue.pop(0)
            if dry:
                free += use
                continue
            run_dir = ROOT / "runs" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, use)), MUJOCO_GL="disable")
            env.pop("VIRTUAL_ENV", None)
            log = open(run_dir / ("launch.log" if mode == "run" else "launch_eval.log"), "a")
            process = subprocess.Popen(
                ["bash", "-c", script], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            running.append((name, use, process))
        for item in list(running):
            name, use, process = item
            if process.poll() is not None:
                print(f"[done] {name} exit={process.returncode}", flush=True)
                running.remove(item)
                free += use
        if running:
            time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["list", "run", "eval"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--gpus", default=None, help="comma list (default: all visible)")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--curve", action="store_true", help="eval: every checkpoint, nominal only")
    parser.add_argument("--no-eval", action="store_true", help="run: skip the post-training eval")
    args = parser.parse_args()
    if args.mode == "list":
        for group, members in GROUPS.items():
            print(f"group {group:14s}: {' '.join(members)}")
        for name, spec in E.items():
            print(f"  {name:22s} {spec['gpus']} GPU  {spec['note']}")
        return
    if args.gpus is None:
        import torch

        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [int(x) for x in args.gpus.split(",")]
    schedule(expand(args.names), gpus, args.mode, args.dry, args.curve, not args.no_eval)


if __name__ == "__main__":
    main()
