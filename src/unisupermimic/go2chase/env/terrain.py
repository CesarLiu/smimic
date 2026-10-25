"""Generated terrain of the Go2 chase terrain task: tile layout and the height field.

The terrain is go2-supermimic's (mjlab TerrainGenerator, fixed seed), exported once by
tools/export_go2_terrain.py to assets/robots/go2/terrain_go2_chase.{xml,json}. Envs are
spread over the terrain columns (one sub-terrain type each) in proportion to the
sub-terrain shares, as mjlab does; the row (difficulty) is picked per reset by the
progress curriculum. The height field is go2-supermimic's: the terrain geoms ray-cast
straight down on a 2 cm grid once (CPU MuJoCo), cached by content hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unisupermimic.go2chase.target import terrain as tr

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robots" / "go2"
TERRAIN_JSON = ASSETS / "terrain_go2_chase.json"
CACHE_DIR = Path.home() / ".cache" / "unisupermimic" / "heightfields"


def proportional_counts(num_envs: int, proportions: np.ndarray) -> np.ndarray:
    """mjlab terrain_entity._proportional_counts: at least one env per column when possible,
    the rest by the largest-remainder method."""
    n = len(proportions)
    if num_envs >= n:
        counts = np.ones(n, dtype=int)
        remaining = num_envs - n
    else:
        counts = np.zeros(n, dtype=int)
        remaining = num_envs
    if remaining > 0:
        ideal = proportions * remaining
        floor = np.floor(ideal).astype(int)
        counts += floor
        leftover = remaining - floor.sum()
        if leftover > 0:
            order = np.argsort(-(ideal - floor))
            counts[order[:leftover]] += 1
    return counts


class TerrainLayout:
    """Sub-terrain origins (rows, cols, 3) and each env's column (terrain type)."""

    def __init__(self, num_envs: int, device: torch.device, path: Path = TERRAIN_JSON):
        meta = json.loads(path.read_text())
        self.columns: list[str] = list(meta["columns"])
        self.origins = torch.tensor(meta["origins"], dtype=torch.float32, device=device)
        self.num_rows, self.num_cols = self.origins.shape[:2]
        counts = proportional_counts(num_envs, np.asarray(meta["proportions"]))
        self.types = torch.repeat_interleave(
            torch.arange(self.num_cols, device=device), torch.as_tensor(counts, device=device)
        )
        self.levels = torch.zeros(num_envs, dtype=torch.long, device=device)

    def open_rows(self, progress: float) -> int:
        """Highest open row: row k of R opens once k / R of the curriculum has passed."""
        return int(min(self.num_rows - 1, int(min(1.0, max(0.0, progress)) * self.num_rows)))

    def assign(self, env_ids: torch.Tensor, progress: float) -> torch.Tensor:
        """go2-supermimic mdp.terrain_levels_by_progress: each reset picks a random open row
        (the easier rows keep being practised). Returns the envs' tile origins (k, 3)."""
        top = self.open_rows(progress)
        self.levels[env_ids] = torch.randint(0, top + 1, (len(env_ids),), device=env_ids.device)
        return self.origins[self.levels[env_ids], self.types[env_ids]]


def _cast_rows(job: tuple[str, list[float], list[float], float]) -> np.ndarray:
    """Heights of grid rows ys over columns xs: the terrain body ray-cast straight down
    (go2-supermimic target/terrain.py build_height_field, point for point)."""
    import mujoco as mujoco_module

    mujoco: Any = mujoco_module  # no stubs for pyright
    scene, xs, ys, top = job
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    terrain_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    # cast only against the terrain: every other geom goes to group 5, which is off
    model.geom_group[:] = 5
    model.geom_group[model.geom_bodyid == terrain_body] = 0
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    down = np.array([0.0, 0.0, -1.0])
    gid = np.zeros(1, dtype=np.int32)
    heights = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            d = mujoco.mj_ray(model, data, np.array([x, y, top]), down, groups, 1, -1, gid)
            heights[i, j] = top - d if d >= 0 else 0.0
    return heights


def height_field(
    scene: Path, layout: TerrainLayout, half: float, res: float, device: torch.device
) -> tr.HeightField:
    """Height field of the scene's terrain body over every tile origin +- half, ray-cast once
    (go2-supermimic target/terrain.py build_height_field, in parallel) and cached."""
    origins = layout.origins.reshape(-1, 3)[:, :2].cpu().numpy()
    bounds = (
        float(origins[:, 0].min() - half),
        float(origins[:, 1].min() - half),
        float(origins[:, 0].max() + half),
        float(origins[:, 1].max() + half),
    )
    terrain_xml = scene.parent / "terrain_go2_chase.xml"
    key = hashlib.sha1(terrain_xml.read_bytes() + np.array([*bounds, res]).tobytes()).hexdigest()[
        :12
    ]
    path = CACHE_DIR / f"heightfield_{key}.pt"
    if not path.exists():
        # a fresh process (no CUDA state, nothing to re-import) forks the ray-casting workers
        import subprocess
        import sys

        print(f"go2 terrain: ray-casting the height field once (cached in {path})", flush=True)
        subprocess.run(
            [sys.executable, "-m", __name__, str(scene), str(path), *map(str, (*bounds, res))],
            check=True,
        )
    return tr.HeightField.load(str(path), device)


def _build(scene: str, path: str, bounds: tuple[float, float, float, float], res: float) -> None:
    import multiprocessing
    import os

    x0, y0, x1, y1 = bounds
    xs = np.arange(x0, x1 + 1e-9, res).tolist()
    ys = np.arange(y0, y1 + 1e-9, res)
    workers = max(1, os.cpu_count() or 1)
    chunks = [c.tolist() for c in np.array_split(ys, 8 * workers) if len(c)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        rows = pool.map(_cast_rows, [(scene, xs, c, 20.0) for c in chunks])
    field = tr.HeightField(torch.tensor(np.concatenate(rows, 0)), (float(x0), float(y0)), res)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(".tmp")
    field.save(str(temporary))
    temporary.replace(path)


__all__ = ["TERRAIN_JSON", "TerrainLayout", "height_field", "proportional_counts"]

if __name__ == "__main__":
    import sys

    _scene, _path, *_numbers = sys.argv[1:]
    _x0, _y0, _x1, _y1, _res = (float(v) for v in _numbers)
    _build(_scene, _path, (_x0, _y0, _x1, _y1), _res)
