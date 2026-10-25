#!/usr/bin/env python
"""Fetch the Unitree Go2 meshes for src/unisupermimic/assets/robots/go2 (MuJoCo Menagerie).

  uv run tools/prepare_go2_assets.py                          # download from GitHub
  uv run tools/prepare_go2_assets.py --from ../go2-supermimic/go2_supermimic/robot/assets
  uv run tools/prepare_go2_assets.py --check                  # compile the scenes only
  uv run tools/prepare_go2_assets.py --driver                 # also the terrain-walking driver

The meshes (about 28 MB) and the driver are gitignored like the G1 assets; the robot,
scene and terrain files are tracked. The driver is go2_rl_lab's mixed-terrain MoE-CTS
policy (MIT, Hugging Face wty-yy/go2_rl_gym_data) that go2-supermimic uses to collect
world-model data on terrain (unisupermimic-go2-train --driver-policy).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GO2 = ROOT / "src/unisupermimic/assets/robots/go2"
MESHES = (
    *(f"base_{i}.obj" for i in range(5)),
    "hip_0.obj",
    "hip_1.obj",
    "thigh_0.obj",
    "thigh_1.obj",
    "thigh_mirror_0.obj",
    "thigh_mirror_1.obj",
    "calf_0.obj",
    "calf_1.obj",
    "calf_mirror_0.obj",
    "calf_mirror_1.obj",
    "foot.obj",
)
URL = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_go2/assets/"
DRIVER_REPO = "wty-yy/go2_rl_gym_data"
DRIVER_FILE = "go2_rl_lab/2026-10-13_15-35-09_mixed1.3.2resume_82.5k_0.6929/exported/policy.pt"


def check() -> None:
    import mujoco

    for name in ("scene_go2_chase", "scene_go2_chase_obstacles", "scene_go2_chase_terrain"):
        model = mujoco.MjModel.from_xml_path(str(GO2 / f"{name}.xml"))
        print(f"{name} compiles: nq {model.nq}, nu {model.nu}, ngeom {model.ngeom}")


def fetch_driver() -> None:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(DRIVER_REPO, DRIVER_FILE))
    target = GO2 / "driver" / "policy.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, target)
    print(f"terrain-walking driver -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        default=None,
        help="copy from a local assets directory instead of downloading",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--driver", action="store_true", help="also fetch the MoE-CTS driver")
    args = parser.parse_args()
    if not args.check:
        target = GO2 / "assets"
        target.mkdir(exist_ok=True)
        for name in MESHES:
            if args.source is not None:
                shutil.copyfile(args.source / name, target / name)
            else:
                with urllib.request.urlopen(URL + name, timeout=60) as response:
                    (target / name).write_bytes(response.read())
        print(f"{len(MESHES)} meshes -> {target}")
    if args.driver:
        fetch_driver()
    check()


if __name__ == "__main__":
    sys.exit(main())
