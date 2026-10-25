#!/usr/bin/env python
"""Fetch the Unitree Go2 meshes for src/unisupermimic/assets/robots/go2 (MuJoCo Menagerie).

  uv run tools/prepare_go2_assets.py                          # download from GitHub
  uv run tools/prepare_go2_assets.py --from ../go2-supermimic/go2_supermimic/robot/assets
  uv run tools/prepare_go2_assets.py --check                  # compile the scene only

The meshes (about 28 MB) are gitignored like the G1 ones; go2_chase.xml and
scene_go2_chase.xml are tracked.
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


def check() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(GO2 / "scene_go2_chase.xml"))
    print(f"scene compiles: nq {model.nq}, nu {model.nu}, ngeom {model.ngeom}")


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
    check()


if __name__ == "__main__":
    sys.exit(main())
