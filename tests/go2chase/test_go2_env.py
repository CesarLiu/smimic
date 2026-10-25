"""Go2 chase on UniLab: model constants, config, and (slow, mjwarp) env consistency."""

from __future__ import annotations

import math

import pytest
import torch

from unisupermimic.go2chase.env import robot
from unisupermimic.go2chase.env.builder import SCENE, compose

MESHES = SCENE.parent / "assets"


def test_actuator_gains_and_action_scale_follow_go2_supermimic() -> None:
    kp, kd, effort = robot.actuator_constants()
    # go2-supermimic robot/go2.py: rotor inertia x gear^2, 10 Hz, damping ratio 2
    hip_inertia = 0.000111842 * 6.33**2
    omega = 20 * math.pi
    assert kp[0] == pytest.approx(hip_inertia * omega**2, rel=1e-6)
    assert kd[0] == pytest.approx(4 * hip_inertia * omega, rel=1e-6)
    assert kp[2] == pytest.approx(hip_inertia * 2.25 * omega**2, rel=1e-6)  # knee: 1.5x gear
    assert effort == [23.7, 23.7, 45.43] * 4
    scale = robot.action_scale()
    assert scale[0] == pytest.approx(0.25 * 23.7 / kp[0])
    assert scale[2] == pytest.approx(0.25 * 45.43 / kp[2])


def test_config_action_scale_matches_the_robot_constants() -> None:
    cfg = compose()
    scales = cfg.env.actions.joint_pos.scale
    assert scales[".*_hip_joint"] == pytest.approx(robot.action_scale()[0])
    assert scales[".*_calf_joint"] == pytest.approx(robot.action_scale()[2])
    assert list(cfg.env.scene.entities.robot.joint_names) == list(robot.JOINT_NAMES)
    assert cfg.env.observations.policy.history_length == 5


@pytest.mark.skipif(not MESHES.is_dir(), reason="run tools/prepare_go2_assets.py first")
def test_scene_has_go2_supermimic_solver_and_actuators() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert (model.opt.timestep, model.opt.impratio, model.opt.ccd_iterations) == (0.005, 10.0, 50)
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    kp, kd, effort = robot.actuator_constants()
    for i, name in enumerate(robot.JOINT_NAMES):
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.removesuffix("_joint"))
        assert model.actuator_gainprm[act, 0] == pytest.approx(kp[i])
        assert model.actuator_biasprm[act, 2] == pytest.approx(-kd[i])
        assert model.actuator_forcerange[act, 1] == pytest.approx(effort[i])
        joint = model.actuator_trnid[act, 0]
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) == name
    key = model.key("home")
    assert key.qpos[7:].tolist() == pytest.approx(list(robot.HOME_JOINT_POS))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="mjwarp requires CUDA")
def test_imagination_rebuilds_the_env_observation_exactly() -> None:
    from unisupermimic.go2chase.env.adapter import ACTOR_HISTORY, Go2ChaseEnv
    from unisupermimic.go2chase.supermimic import state as st

    env = Go2ChaseEnv(16, play=True)
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (16, 350)
        bias = env.robot.encoder_bias.clone()
        assert float(bias.abs().max()) <= 0.015
        xi = env.term.xi
        assert xi.shape == (16, 4)
        assert bool(((xi[:, 0] >= 0.3) & (xi[:, 0] <= 1.5)).all())
        assert bool((xi[:, 1:].abs() <= torch.tensor([0.025, 0.025, 0.03], device=xi.device)).all())
        generator = torch.Generator(device=env.device).manual_seed(0)
        for _ in range(30):
            action = 0.5 * torch.randn(16, 12, device=env.device, generator=generator)
            obs, _, _, _, _ = env.step(action)
            frame = st.obs_frame(st.robot_state(env), action, env.term.command, bias, noise=False)
            i = 0
            for name, dim in st.OBS_TERMS:
                newest = obs["actor"][:, i + dim * (ACTOR_HISTORY - 1) : i + dim * ACTOR_HISTORY]
                torch.testing.assert_close(newest, frame[name], rtol=0, atol=1e-6)
                i += dim * ACTOR_HISTORY
    finally:
        env.close()
