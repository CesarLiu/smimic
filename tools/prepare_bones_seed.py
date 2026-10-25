#!/usr/bin/env python
"""BONES-SEED (G1 retarget) -> packed 50 Hz SuperMimic motion library (paper Appendix E).

  # one-time: accept the license on https://huggingface.co/datasets/bones-studio/seed and
  # export HF_TOKEN=...
  .venv/bin/python tools/prepare_bones_seed.py download --root data/bones_seed   # ~23.5 GB
  .venv/bin/python tools/prepare_bones_seed.py inspect  --root data/bones_seed   # check the layout!
  .venv/bin/python tools/prepare_bones_seed.py build    --root data/bones_seed --out data/bones_pack
  # train on it:  env.commands.motion.motion_pack=data/bones_pack

  # pack any LAFAN-format CSVs (pos, quat xyzw, 29 joints) the same way, e.g. for testing:
  .venv/bin/python tools/prepare_bones_seed.py pack-csv --src_fps 30 --out data/lafan_pack \\
      src/unisupermimic/assets/motions/g1/lafan1_supermimic/*.csv

The G1 CSV layout is not documented on the dataset card; `inspect` and `build` detect an
optional leading index/time column, root position in m or cm, the quaternion order (wxyz or
xyzw, by pelvis uprightness), and joint angles in rad or deg. The source frame rate comes
from the metadata (`move_duration_frames` @ 120 fps) against the row count. Override with
--quat / --fps / --deg / --cm. Clips are resampled with MotionLib's own resample_clip, so a
pack loads to exactly what MotionLib would build from the same CSVs.

Filters (Appendix E): a name filter for terrain/object interactions (paper: removes 12,553 of
142,220), then a feasibility filter on MuJoCo forward kinematics at 50 Hz (paper: removes 153):
elevated support (lowest body > 0.20 m for >= 0.6 s), ground penetration (lowest body
< -0.12 m, or < -0.08 m for > 80% of the clip), and floating feet (both feet > 0.02 m above the
floor for >= 0.5 s while upright, at standing height and nearly stationary). The keyword list
is a reconstruction; compare the printed counts with the paper's and tune it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "src/unisupermimic/assets/robots/g1/supermimic.xml"
OUT_FPS = 50.0

NAME_KEYWORDS = (
    "stair",
    "stairs",
    "step_up",
    "step_down",
    "steps_up",
    "ladder",
    "box",
    "boxes",
    "crate",
    "chair",
    "sit",
    "sitting",
    "sits",
    "seated",
    "bench",
    "stool",
    "sofa",
    "couch",
    "bed",
    "table",
    "desk",
    "climb",
    "climbing",
    "climbs",
    "fence",
    "wall",
    "ledge",
    "platform",
    "jump_off",
    "jumpoff",
    "jumping_off",
    "jump_over",
    "vault",
    "hurdle",
    "railing",
    "rail",
    "ramp",
    "slope",
    "car",
    "bike",
    "bicycle",
    "horse",
    "swing",
    "obstacle",
    "lean_on",
    "leaning_on",
    "kneel_on",
    "stand_on",
    "standing_on",
    "walk_on",
    "onto",
    "up_on",
)


def name_rejected(name: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    joined = "_".join(tokens)
    for keyword in NAME_KEYWORDS:
        if "_" in keyword:
            if keyword in joined:
                return True
        elif any(t == keyword or (len(keyword) >= 4 and t.startswith(keyword)) for t in tokens):
            return True
    return False


# ---------------- CSV layout ----------------


def read_csv(path: str) -> tuple[list[str] | None, np.ndarray]:
    with open(path, encoding="utf-8") as stream:
        first = stream.readline()
    try:
        [float(x) for x in first.strip().split(",") if x != ""]
        header = None
    except ValueError:
        header = first.strip().split(",")
    rows = np.loadtxt(path, delimiter=",", skiprows=1 if header else 0, dtype=np.float64, ndmin=2)
    return header, rows


def detect_layout(samples: list[np.ndarray], quat="auto", deg="auto", cm="auto") -> dict:
    columns = samples[0].shape[1]
    if columns < 36:
        raise ValueError(f"expected >= 36 columns (7 root + 29 joints), got {columns}")
    skip = columns - 36
    x = np.concatenate([s[:, skip:] for s in samples])
    if quat == "auto":

        def upright(q):  # pelvis R22, ~1 when upright
            return np.median(
                1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2) / np.maximum((q * q).sum(1), 1e-9)
            )

        quat = "wxyz" if upright(x[:, 3:7]) >= upright(x[:, [6, 3, 4, 5]]) else "xyzw"
    if deg == "auto":
        deg = bool(np.abs(x[:, 7:]).max() > 2 * np.pi * 1.5)
    if cm == "auto":
        cm = bool(np.median(x[:, 2]) > 5.0)
    return {"skip": skip, "quat": quat, "deg": bool(deg), "cm": bool(cm)}


def to_lafan_rows(rows: np.ndarray, layout: dict) -> np.ndarray:
    """Any detected layout -> LAFAN rows (pos m, quat xyzw unit, 29 joints rad)."""
    a = rows[:, layout["skip"] :]
    pos = a[:, 0:3] / (100.0 if layout["cm"] else 1.0)
    q = a[:, [4, 5, 6, 3]] if layout["quat"] == "wxyz" else a[:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    joints = a[:, 7:36] * (np.pi / 180.0 if layout["deg"] else 1.0)
    return np.concatenate([pos, q, joints], 1).astype(np.float32)


def source_fps(rows: int, duration_frames_120: int, override: float | None) -> float:
    if override:
        return override
    if duration_frames_120 <= 0:
        return 120.0
    raw = 120.0 * (rows - 1) / max(duration_frames_120 - 1, 1)
    return min((30.0, 50.0, 60.0, 100.0, 120.0), key=lambda f: abs(f - raw))


# ---------------- feasibility (MuJoCo forward kinematics at 50 Hz) ----------------

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        feet = {"left": [], "right": []}
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            for side in feet:
                if name.startswith(f"{side}_ankle_roll_link_collision"):
                    assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                    feet[side].append(g)
        _MODEL = (model, mujoco.MjData(model), feet)
    return _MODEL


def feasibility_reason(pos: np.ndarray, quat_wxyz: np.ndarray, joints: np.ndarray) -> str:
    import mujoco

    model, data, feet = _model()
    n = pos.shape[0]
    zmin = np.empty(n)
    foot = np.empty((n, 2))
    for t in range(n):
        data.qpos[0:3], data.qpos[3:7], data.qpos[7:] = pos[t], quat_wxyz[t], joints[t]
        mujoco.mj_kinematics(model, data)
        zmin[t] = data.xpos[1:, 2].min()
        for i, side in enumerate(("left", "right")):
            ids = feet[side]
            foot[t, i] = (data.geom_xpos[ids, 2] - model.geom_size[ids, 0]).min()

    def longest(mask: np.ndarray) -> int:
        best = run = 0
        for value in mask:
            run = run + 1 if value else 0
            best = max(best, run)
        return best

    if longest(zmin > 0.20) >= int(0.6 * OUT_FPS):
        return "elevated_support"
    if zmin.min() < -0.12 or (zmin < -0.08).mean() > 0.8:
        return "ground_penetration"
    x, y = quat_wxyz[:, 1], quat_wxyz[:, 2]
    upright = 1 - 2 * (x * x + y * y)
    speed = (
        np.linalg.norm(np.gradient(pos, 1.0 / OUT_FPS, axis=0), axis=1) if n > 1 else np.zeros(n)
    )
    floating = (foot > 0.02).all(1) & (upright > 0.9) & (pos[:, 2] > 0.65) & (speed < 0.15)
    if longest(floating) >= int(0.5 * OUT_FPS):
        return "floating_feet"
    return ""


# ---------------- worker ----------------


def _process(job: tuple[str, str, float | None, int, float | None, dict | None, bool]):
    """(name, path, fixed fps, metadata duration @120 fps, fps override, layout or None for
    LAFAN rows, run_feasibility) -> (name, clip or None, rejection reason)."""
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import resample_clip

    name, path, fps, duration, override, layout, check = job
    try:
        _, rows = read_csv(path)
        if fps is None:  # BONES-SEED: from the metadata duration against the row count
            fps = source_fps(rows.shape[0], duration, override)
        rows = to_lafan_rows(rows, layout) if layout is not None else rows.astype(np.float32)
        if rows.shape[0] < 2 or rows.shape[1] != 36:
            return name, None, "unreadable"
        position, quaternion, joints = (
            value.numpy() for value in resample_clip(torch.from_numpy(rows), fps)
        )
    except Exception as exc:  # noqa: BLE001 - report and skip unreadable files
        return name, None, f"unreadable: {exc!r}"
    reason = feasibility_reason(position, quaternion, joints) if check else ""
    return name, (position, quaternion, joints), reason


# ---------------- packed library writer ----------------


class PackWriter:
    """Shards to <out>/_shards, then merges into memory-mappable .npy arrays."""

    def __init__(self, out: str):
        self.out = Path(out)
        self.shards = self.out / "_shards"
        self.shards.mkdir(parents=True, exist_ok=True)
        self.names: list[str] = []
        self.lengths: list[int] = []
        self.buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.count = 0

    def add(self, name: str, clip: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.names.append(name)
        self.lengths.append(int(clip[0].shape[0]))
        self.buffer.append(clip)
        if len(self.buffer) >= 2048:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        np.savez(
            self.shards / f"{self.count:05d}.npz",
            root_pos=np.concatenate([c[0] for c in self.buffer]),
            root_quat=np.concatenate([c[1] for c in self.buffer]),
            joints=np.concatenate([c[2] for c in self.buffer]),
        )
        self.count += 1
        self.buffer = []

    def close(self, meta: dict) -> None:
        self._flush()
        frames = int(sum(self.lengths))
        widths = {"root_pos": 3, "root_quat": 4, "joints": 29}
        arrays = {
            k: np.lib.format.open_memmap(
                self.out / f"{k}.npy", mode="w+", dtype=np.float32, shape=(frames, w)
            )
            for k, w in widths.items()
        }
        offset = 0
        for index in range(self.count):
            shard = np.load(self.shards / f"{index:05d}.npz")
            n = shard["root_pos"].shape[0]
            for key in widths:
                arrays[key][offset : offset + n] = shard[key]
            offset += n
        for array in arrays.values():
            array.flush()
        np.save(self.out / "clip_len.npy", np.asarray(self.lengths, dtype=np.int64))
        (self.out / "names.txt").write_text("\n".join(self.names), encoding="utf-8")
        meta = dict(
            meta,
            n_clips=len(self.names),
            n_frames=frames,
            fps=OUT_FPS,
            hours=frames / OUT_FPS / 3600,
        )
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        shutil.rmtree(self.shards)
        print(
            f"pack: {meta['n_clips']} clips, {frames} frames, {meta['hours']:.2f} h -> {self.out}"
        )


# ---------------- commands ----------------


def list_bones(root: str) -> list[tuple[str, str, int]]:
    meta = sorted(glob.glob(os.path.join(root, "metadata", "seed_metadata_v00*.parquet")))
    rows = []
    if meta:
        import pandas as pd

        frame = pd.read_parquet(meta[-1])
        for _, record in frame.iterrows():
            path = os.path.join(root, str(record["move_g1_mujoco_path"]).lstrip("/"))
            if not os.path.exists(path):
                found = glob.glob(
                    os.path.join(root, "g1", "**", os.path.basename(path)), recursive=True
                )
                path = found[0] if found else path
            rows.append(
                (str(record["filename"]), path, int(record.get("move_duration_frames", 0) or 0))
            )
    else:
        for path in sorted(glob.glob(os.path.join(root, "g1", "**", "*.csv"), recursive=True)):
            rows.append((os.path.splitext(os.path.basename(path))[0], path, 0))
    return rows


def cmd_download(args) -> None:
    import tarfile

    from huggingface_hub import hf_hub_download

    os.makedirs(args.root, exist_ok=True)
    for name in ("metadata/seed_metadata_v004.parquet", "g1.tar.gz"):
        print(
            "downloaded",
            hf_hub_download("bones-studio/seed", name, repo_type="dataset", local_dir=args.root),
        )
    with tarfile.open(os.path.join(args.root, "g1.tar.gz")) as archive:
        archive.extractall(args.root, filter="data")


def cmd_inspect(args) -> None:
    rows = list_bones(args.root)
    print(f"{len(rows)} clips listed")
    rng = np.random.default_rng(0)
    sample = [rows[i] for i in rng.choice(len(rows), size=min(20, len(rows)), replace=False)]
    for name, path, _ in sample[:3]:
        header, data = read_csv(path)
        print(
            f"\n{path}\n  header: {header}\n  shape: {data.shape}  first row: "
            f"{np.round(data[0], 4).tolist()}"
        )
    arrays = [read_csv(path)[1] for _, path, _ in sample]
    layout = detect_layout(arrays, args.quat, args.deg, args.cm)
    print("\ndetected layout:", layout)
    print(
        "source fps:",
        sorted({source_fps(a.shape[0], d, args.fps) for a, (_, _, d) in zip(arrays, sample)}),
    )
    lafan = to_lafan_rows(arrays[0], layout)
    print(
        f"root z median {np.median(lafan[:, 2]):.3f} m, joints [{lafan[:, 7:].min():.2f}, "
        f"{lafan[:, 7:].max():.2f}] rad"
    )


def _run(jobs: list, out: str, workers: int, meta: dict) -> None:
    writer = PackWriter(out)
    rejected: dict[str, str] = {}
    with Pool(workers) as pool:
        for done, (name, clip, reason) in enumerate(pool.imap(_process, jobs, chunksize=16), 1):
            if clip is None or reason:
                rejected[name] = reason or "unreadable"
            else:
                writer.add(name, clip)
            if done % 5000 == 0:
                print(f"  {done}/{len(jobs)} processed, kept {len(writer.names)}", flush=True)
    counts = Counter(r.split(":")[0] for r in rejected.values())
    print("feasibility / other rejections:", dict(counts))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "rejected.json").write_text(json.dumps(rejected, indent=0), encoding="utf-8")
    writer.close(dict(meta, rejected=dict(counts)))


def cmd_build(args) -> None:
    rows = list_bones(args.root)
    kept = [r for r in rows if not name_rejected(r[0])]
    print(
        f"{len(rows)} clips; name filter removed {len(rows) - len(kept)} (paper: 12,553) -> {len(kept)}"
    )
    rng = np.random.default_rng(0)
    sample = [kept[i] for i in rng.choice(len(kept), size=min(50, len(kept)), replace=False)]
    layout = detect_layout([read_csv(p)[1] for _, p, _ in sample], args.quat, args.deg, args.cm)
    print("layout:", layout)
    jobs = [(name, path, None, duration, args.fps, layout, True) for name, path, duration in kept]
    _run(
        jobs,
        args.out,
        args.workers,
        dict(source="BONES-SEED g1", layout=layout, name_filter_removed=len(rows) - len(kept)),
    )


def cmd_pack_csv(args) -> None:
    files = sorted({f for pattern in args.files for f in glob.glob(pattern)})
    jobs = [
        (os.path.splitext(os.path.basename(f))[0], f, args.src_fps, 0, None, None, args.feasibility)
        for f in files
    ]
    _run(jobs, args.out, args.workers, dict(source="csv", src_fps=args.src_fps))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("download", "inspect", "build"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="data/bones_seed")
        p.add_argument("--out", default="data/bones_pack")
        p.add_argument("--quat", default="auto", choices=["auto", "wxyz", "xyzw"])
        p.add_argument("--deg", default="auto", choices=["auto", "true", "false"])
        p.add_argument("--cm", default="auto", choices=["auto", "true", "false"])
        p.add_argument("--fps", type=float, default=None)
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p = sub.add_parser("pack-csv")
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--src_fps", type=float, default=30.0)
    p.add_argument("--feasibility", action="store_true", help="also apply the feasibility filter")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    for key in ("deg", "cm"):
        if getattr(args, key, "auto") in ("true", "false"):
            setattr(args, key, getattr(args, key) == "true")
    {
        "download": cmd_download,
        "inspect": cmd_inspect,
        "build": cmd_build,
        "pack-csv": cmd_pack_csv,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
