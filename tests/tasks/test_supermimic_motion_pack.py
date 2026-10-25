"""A packed motion library must load to exactly what MotionLib builds from the CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import prepare_bones_seed  # noqa: E402

from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics  # noqa: E402
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import (  # noqa: E402
    MotionLib,
    motion_pack_clip_count,
)

CLIPS = ("walk1_subject1", "dance2_subject4", "fallAndGetUp1_subject1")
CSV_DIR = ROOT / "src/unisupermimic/assets/motions/g1/lafan1_supermimic"
MJCF = ROOT / "src/unisupermimic/assets/robots/g1/supermimic.xml"


@pytest.mark.skipif(not CSV_DIR.is_dir(), reason="run tools/prepare_supermimic_assets.py first")
def test_pack_loads_identically_to_csv(tmp_path) -> None:
    files = [str(CSV_DIR / f"{name}.csv") for name in CLIPS]
    writer = prepare_bones_seed.PackWriter(str(tmp_path / "pack"))
    for name, path in zip(CLIPS, files):
        _, clip, reason = prepare_bones_seed._process((name, path, 30.0, 0, None, None, False))
        assert clip is not None and reason == ""
        writer.add(name, clip)
    writer.close({"source": "test"})
    assert motion_pack_clip_count(tmp_path / "pack") == len(CLIPS)

    kin = G1Kinematics(str(MJCF), "cpu")
    from_csv = MotionLib(files, kin, "cpu", src_fps=30.0)
    from_pack = MotionLib([], kin, "cpu", pack=str(tmp_path / "pack"))
    assert from_pack.names == from_csv.names
    tensors = [k for k, v in vars(from_csv).items() if isinstance(v, torch.Tensor)]
    assert "body_p" in tensors and "frame_bin" in tensors
    for key in tensors:
        assert torch.equal(getattr(from_pack, key), getattr(from_csv, key)), key


def test_name_filter_flags_terrain_and_object_clips() -> None:
    reject = prepare_bones_seed.name_rejected
    for name in (
        "walk_up_stairs_01",
        "sit_down_on_chair",
        "climb_ladder_R",
        "jump_off_box",
        "Lean_On_Wall_002",
    ):
        assert reject(name), name
    for name in (
        "walk_forward_01",
        "dance_routine_3",
        "side_roll_left",
        "position_hold",
        "carry_newspaper",
        "boxing_jab",
    ):
        assert not reject(name), name
