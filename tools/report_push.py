#!/usr/bin/env python
"""Summarize final-checkpoint evaluations (nominal + push tests) per experiment and seed.

  .venv/bin/python tools/report_push.py [--runs runs] [--iteration 12000]

Prints a markdown table: normalized return and failed clips (of 40) per experiment, with the
mean and standard deviation over seeds when several `<name>_s<seed>` runs exist.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

TESTS = (
    ("nominal", "_eval.json"),
    ("push 0.5", "_eval_push0.5.json"),
    ("push 1.0", "_eval_push1.json"),
    ("push 1.5", "_eval_push1.5.json"),
)


def fmt(values: list[float], digits: int) -> str:
    if not values:
        return "-"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--iteration", type=int, default=None, help="default: each run's last")
    args = parser.parse_args()
    table: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    seeds: dict[str, int] = defaultdict(int)
    for run in sorted(Path(args.runs).iterdir()):
        ckpts = sorted(run.glob("ckpt_*.pt")) if run.is_dir() else []
        if args.iteration is not None:
            ckpts = [c for c in ckpts if c.stem == f"ckpt_{args.iteration:06d}"]
        if not ckpts:
            continue
        stem = ckpts[-1].stem
        name = re.sub(r"_s\d+$", "", run.name)
        seeds[name] += 1
        for label, suffix in TESTS:
            path = run / f"{stem}{suffix}"
            if path.is_file():
                result = json.loads(path.read_text())
                table[name][label].append((result["norm_return"], result["n_failed"]))
    header = "| experiment | seeds | " + " | ".join(
        f"{label} return | failed" for label, _ in TESTS
    )
    print(header + " |")
    print("|" + "---|" * (2 + 2 * len(TESTS)))
    for name in sorted(table):
        cells = []
        for label, _ in TESTS:
            rows = table[name][label]
            cells += [fmt([r for r, _ in rows], 3), fmt([float(f) for _, f in rows], 1)]
        print(f"| {name} | {seeds[name]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
