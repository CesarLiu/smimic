#!/usr/bin/env python
"""Export go2-supermimic's generated rough terrain (mjlab) to MJCF for the UniLab port.

Run with an environment that has go2-supermimic and mjlab installed (not this repo's):

  python tools/export_go2_terrain.py --out src/unisupermimic/assets/robots/go2

Writes terrain_go2_chase.xml (the terrain body: box tiles and height fields, generated with
go2-supermimic's fixed seed) and terrain_go2_chase.json (sub-terrain origins (rows, cols, 3),
column names and proportions). Textures, materials and lights are dropped: the Uni scene
brings its own look. The terrain is deterministic, so the committed files are the output of
this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    from go2_supermimic.env.terrains import rough_terrain
    from mjlab.terrains.terrain_entity import TerrainEntity

    cfg = rough_terrain()
    cfg.num_envs = 1
    terrain = TerrainEntity(cfg, device="cpu")
    spec = terrain.spec
    for geom in spec.geoms:
        geom.material = ""
    for element in (*spec.textures, *spec.materials, *spec.lights):
        spec.delete(element)
    xml = spec.to_xml().replace(
        '<mujoco model="MuJoCo Model">', '<mujoco model="go2 chase terrain">'
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "terrain_go2_chase.xml").write_text(xml)
    generator = cfg.terrain_generator
    proportions = np.array([s.proportion for s in generator.sub_terrains.values()])
    meta = {
        "source": "go2-supermimic env/terrains.py ROUGH_TERRAINS, mjlab TerrainGenerator",
        "seed": generator.seed,
        "tile_size": list(generator.size),
        "columns": list(generator.sub_terrains),
        "proportions": (proportions / proportions.sum()).tolist(),
        "origins": terrain.terrain_origins.cpu().numpy().round(6).tolist(),
    }
    (args.out / "terrain_go2_chase.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(
        f"terrain: {xml.count('<geom')} geoms, {xml.count('<hfield')} height fields -> {args.out}"
    )


if __name__ == "__main__":
    main()
