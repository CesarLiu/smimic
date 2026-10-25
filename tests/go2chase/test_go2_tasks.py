"""Go2 chase obstacles and terrain tasks: configuration, terrain layout, scenes, rendering,
and (slow, mjwarp) the placed obstacles and terrain tiles."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from unisupermimic.go2chase.env.builder import compose, scene_for
from unisupermimic.go2chase.env.command import configure_target
from unisupermimic.go2chase.env.terrain import TERRAIN_JSON, TerrainLayout, proportional_counts
from unisupermimic.go2chase.target import obstacles as ob
from unisupermimic.go2chase.target.generator import BallTargetCfg

MESHES = scene_for("flat").parent / "assets"
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="mjwarp requires CUDA")


def test_per_task_target_settings_follow_go2_env_cfg() -> None:
    flat, obstacles, terrain = BallTargetCfg(), BallTargetCfg(), BallTargetCfg()
    configure_target(flat, "flat")
    configure_target(obstacles, "obstacles")
    configure_target(terrain, "terrain")
    assert flat == BallTargetCfg()
    assert obstacles.obstacles.enabled and obstacles.throw_speed == (2.5, 3.5)
    assert obstacles.max_time_to_robot == 0.75
    assert terrain.kind_probs == (0.0, 0.5, 0.5) and terrain.bars_follow_ground
    assert terrain.flee_ratio_final == (0.8, 1.0)


def test_column_counts_match_mjlab() -> None:
    proportions = np.array([0.1, 0.2, 0.2, 0.15, 0.15, 0.1, 0.1])
    # mjlab TerrainEntity placed 8 envs on columns [0, 1, 1, 2, 3, 4, 5, 6]
    assert proportional_counts(8, proportions).tolist() == [1, 2, 1, 1, 1, 1, 1]
    counts = proportional_counts(4096, proportions)
    assert counts.sum() == 4096 and (counts >= 1).all()
    assert proportional_counts(3, proportions).sum() == 3


def test_terrain_rows_open_with_progress() -> None:
    layout = TerrainLayout(64, torch.device("cpu"))
    assert layout.origins.shape == (5, 7, 3) and len(layout.columns) == 7
    ids = torch.arange(64)
    origins = layout.assign(ids, progress=0.0)
    assert bool((layout.levels == 0).all())
    torch.testing.assert_close(origins, layout.origins[0, layout.types])
    layout.assign(ids, progress=0.5)
    assert int(layout.levels.max()) <= 2
    torch.manual_seed(0)
    layout.assign(ids, progress=1.0)
    assert int(layout.levels.max()) == 4


def test_terrain_assets_are_consistent() -> None:
    meta = json.loads(TERRAIN_JSON.read_text())
    assert meta["columns"] == [
        "flat",
        "stairs",
        "stairs_inv",
        "boxes",
        "rough",
        "slope",
        "slope_inv",
    ]
    assert np.isclose(sum(meta["proportions"]), 1.0)
    xml = TERRAIN_JSON.with_suffix(".xml").read_text()
    assert xml.count("<hfield") == 15 and xml.count("<geom") == 3579


@pytest.mark.skipif(not MESHES.is_dir(), reason="run tools/prepare_go2_assets.py first")
def test_obstacle_scene_boxes_are_heavy_planar_bodies() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_for("obstacles")))
    assert model.nmocap == 0 and model.nq == 19 + 3 * ob.NUM_OBSTACLES
    for k, (hx, hy) in enumerate(ob.OBSTACLE_HALF):
        geom = model.geom(f"obstacle{k}")
        assert geom.size.tolist() == pytest.approx([hx, hy, ob.OBSTACLE_HEIGHT / 2])
        assert model.body(f"obstacle{k}").mass[0] == pytest.approx(1e6)
        for axis in ("x", "y", "yaw"):
            assert model.joint(f"obstacle{k}_{axis}").damping[0] == pytest.approx(1e6)
    assert len(model.key("home").qpos) == model.nq


def test_task_configs_add_their_terms() -> None:
    obstacles = compose(task="obstacles")
    assert "lidar" in obstacles.env.observations.policy.terms
    assert len(obstacles.env.scene.entities.obstacles.joint_names) == 3 * ob.NUM_OBSTACLES
    assert obstacles.env.commands.target.task == "obstacles"
    assert obstacles.reward.obstacle_contact.weight == -100.0
    terrain = compose(task="terrain")
    assert "height_scan" in terrain.env.observations.policy.terms
    assert terrain.env.commands.target.task == "terrain"
    assert list(terrain.env.commands.target.mode_probs) == [0.6, 0.0, 0.4]
    assert terrain.reward.terminated.weight == -600.0
    assert "lidar" not in compose(task="flat").env.observations.policy.terms


@pytest.mark.skipif(not MESHES.is_dir(), reason="run tools/prepare_go2_assets.py first")
def test_render_draws_balls_and_bars(tmp_path) -> None:
    import mujoco

    from unisupermimic.go2chase import render

    try:  # only a missing OpenGL context skips (source tools/softgl.sh)
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close()
    except Exception as exc:
        pytest.skip(f"no MuJoCo rendering context: {exc}")
    qpos = np.array(
        [0, 0, 0.3, 1, 0, 0, 0, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8]
    )
    frame = {
        "qpos": qpos,
        "rows": [("Chase + dodge", (255, 255, 255)), ("Robot  1.00 m/s", (255, 210, 60))],
        "balls": [(np.array([0.5, 0.0, 0.3]), render.GREEN_RGBA)],
        "bars": [(np.array([1.0, -0.5, 0.2]), np.array([1.0, 0.5, 0.2]))],
    }
    plain = dict(frame, balls=[], bars=[])
    path = tmp_path / "v.mp4"
    assert render.render([frame] * 4, scene_for("flat"), path, width=160, height=120) == 2
    assert path.stat().st_size > 0

    import imageio.v2 as imageio

    drawn = imageio.get_reader(path).get_data(0).astype(float)
    render.render([plain] * 2, scene_for("flat"), tmp_path / "p.mp4", width=160, height=120)
    empty = imageio.get_reader(tmp_path / "p.mp4").get_data(0).astype(float)
    assert np.abs(drawn - empty).mean() > 0.5  # the ball and the bar changed the picture


@pytest.mark.slow
@CUDA
def test_obstacles_are_placed_at_their_layout() -> None:
    from unisupermimic.go2chase.env.adapter import Go2ChaseEnv

    env = Go2ChaseEnv(16, task="obstacles", play=True)
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (16, 430)
        term = env.term
        for _ in range(5):
            env.step(torch.zeros(16, 12, device=env.device))
        # the analytic LiDAR sees the placed obstacles
        layout = ob.unpack(term.gen.obst).cpu().numpy()
        lidar = term.lidar().cpu().numpy()
        assert (layout[..., 3] > 0.5).any()
        rows = np.flatnonzero((layout[..., 3] > 0.5).any(-1))
        assert (lidar[rows] < term.cfg.target.obstacles.lidar_range).any()
    finally:
        env.close()


@pytest.mark.slow
@CUDA
def test_terrain_envs_stand_on_their_tiles() -> None:
    from unisupermimic.go2chase.env.adapter import Go2ChaseEnv

    env = Go2ChaseEnv(14, task="terrain", play=True)
    try:
        term = env.term
        term.cfg.terrain_curriculum_steps = 1
        term.step_offset = 10  # every row open
        obs, _ = env.reset()
        assert obs["actor"].shape == (14, 735)
        tiles = term.terrain.origins[term.terrain.levels, term.terrain.types]
        torch.testing.assert_close(term.center, tiles)
        pos = env.robot.root_link_pos_w
        # mjlab reset: tile origin + default root (z 0.29) + uniform x, y (+-0.5), z (0.01-0.05)
        assert bool(((pos[:, :2] - tiles[:, :2]).abs() <= 0.5 + 1e-4).all())
        lift = pos[:, 2] - tiles[:, 2] - 0.29
        assert bool(((lift >= 0.01 - 1e-4) & (lift <= 0.05 + 1e-4)).all())
        for _ in range(40):
            env.step(torch.zeros(14, 12, device=env.device))
        height = term.base_local()[:, 2]
        assert bool(((height > 0.05) & (height < 0.45)).all())  # on the local ground, not in it
        scan = term.scan()
        assert scan.shape == (14, 77)
        # the scan point under the base matches the base height above ground
        assert bool(((scan.median(-1).values - height).abs() < 0.15).all())
    finally:
        env.close()
