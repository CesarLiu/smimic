"""Runtime tests for analytic push injection in the SuperMimic environment."""

from __future__ import annotations

import pytest
import torch
from test_supermimic_env import _require_backend_runtime, _runtime_cfg  # tests are not a package

from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.env import make_supermimic_env
from unisupermimic.tasks.motion_tracking.g1.supermimic.manager_terms import SuperMimicCommandCfg

# Only mjwarp applies interval root-velocity deltas exactly (see env._bind_push).
PUSH_BACKENDS = ["mjwarp"]


def _push_env(tmp_path, backend_name: str, *, push_scale: float):
    _require_backend_runtime(backend_name)
    cfg = _runtime_cfg(tmp_path, backend=backend_name)
    command_cfg = cfg.commands["motion"]
    assert isinstance(command_cfg, SuperMimicCommandCfg)
    command_cfg.push_scale = push_scale
    command_cfg.push_interval_s = (0.02, 0.02)  # a kick after every control step
    try:
        return make_supermimic_env(cfg, num_envs=2, backend_type=backend_name)
    except (NotImplementedError, RuntimeError) as exc:
        if "velocity" in str(exc).lower():
            pytest.skip(f"{backend_name} lacks interval root-velocity deltas: {exc}")
        raise


def _close(env) -> None:
    env.close()
    env.backend.close()


@pytest.mark.slow
@pytest.mark.parametrize("backend_name", PUSH_BACKENDS)
def test_scheduled_kicks_keep_s1_clean_and_are_logged(tmp_path, backend_name: str) -> None:
    env = _push_env(tmp_path, backend_name, push_scale=1.0)
    try:
        env.reset()
        action = torch.zeros((2, 29), device=env.device)
        previous = None
        checked = 0
        for _ in range(6):
            transition = env.step(action).info[SUPERMIMIC_TRANSITION_KEY]
            if previous is not None:
                alive = ~previous["done"]
                # this start state = previous physics outcome + the logged kick
                expected = previous["s1"].clone()
                expected[:, 7:13] += transition["push"]
                torch.testing.assert_close(
                    transition["s"][alive], expected[alive], atol=2e-4, rtol=0.0
                )
                assert bool((transition["push"][alive].abs().sum(-1) > 0).all())
                checked += int(alive.sum())
            previous = transition
        assert checked > 0
    finally:
        _close(env)


@pytest.mark.slow
@pytest.mark.parametrize("backend_name", PUSH_BACKENDS)
def test_without_pushes_transitions_chain_and_log_zero(tmp_path, backend_name: str) -> None:
    env = _push_env(tmp_path, backend_name, push_scale=0.0)
    try:
        env.reset()
        action = torch.zeros((2, 29), device=env.device)
        previous = None
        for _ in range(4):
            transition = env.step(action).info[SUPERMIMIC_TRANSITION_KEY]
            assert torch.equal(transition["push"], torch.zeros_like(transition["push"]))
            if previous is not None:
                alive = ~previous["done"]
                torch.testing.assert_close(
                    transition["s"][alive], previous["s1"][alive], atol=2e-4, rtol=0.0
                )
            previous = transition
    finally:
        _close(env)


@pytest.mark.slow
@pytest.mark.parametrize("backend_name", PUSH_BACKENDS)
def test_apply_root_push_changes_root_velocity_by_the_kick(tmp_path, backend_name: str) -> None:
    env = _push_env(tmp_path, backend_name, push_scale=0.0)
    try:
        _, info = env.reset()
        before = info[SUPERMIMIC_STATE_KEY]["s"].clone()
        push = torch.tensor(
            [[0.7, -0.4, 0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=env.device
        )
        mask = torch.tensor([True, False], device=env.device)
        current = env.apply_root_push(push, mask)
        torch.testing.assert_close(current["s"][:, 7:13] - before[:, 7:13], push, atol=2e-4, rtol=0.0)
        torch.testing.assert_close(current["s"][:, :7], before[:, :7], atol=1e-5, rtol=0.0)
        assert current is env.state.info[SUPERMIMIC_STATE_KEY]
    finally:
        _close(env)


@pytest.mark.slow
def test_pushes_fail_closed_on_the_mujoco_host_backend(tmp_path) -> None:
    cfg = _runtime_cfg(tmp_path, backend="mujoco")
    command_cfg = cfg.commands["motion"]
    assert isinstance(command_cfg, SuperMimicCommandCfg)
    command_cfg.push_scale = 1.0
    with pytest.raises(NotImplementedError, match="requires the mjwarp backend"):
        make_supermimic_env(cfg, num_envs=2, backend_type="mujoco")


@pytest.mark.slow
def test_angular_kicks_are_rejected(tmp_path) -> None:
    env = _push_env(tmp_path, "mjwarp", push_scale=0.0)
    try:
        env.reset()
        push = torch.zeros((2, 6), device=env.device)
        push[0, 4] = 0.3
        with pytest.raises(ValueError, match="linear root kicks only"):
            env.apply_root_push(push, torch.tensor([True, False], device=env.device))
    finally:
        _close(env)
