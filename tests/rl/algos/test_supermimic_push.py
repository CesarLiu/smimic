"""CPU tests for analytic push injection on the learner side."""

from __future__ import annotations

from dataclasses import replace

import torch
from test_supermimic_learner import FakeEnv, _config  # same directory (tests are not a package)

from unisupermimic.rl.algos.supermimic.geometry import (
    PUSH_RANGE,
    add_push,
    flatten_state,
    sample_push,
    unflatten_state,
)
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer


def _random_state(batch: int, num_joints: int = 2):
    flat = torch.randn(batch, 13 + 2 * num_joints)
    flat[:, 3:7] = torch.nn.functional.normalize(flat[:, 3:7], dim=-1)
    return unflatten_state(flat)


def test_add_push_is_an_exact_differentiable_velocity_jump() -> None:
    state = tuple(value.clone().requires_grad_() for value in _random_state(16))
    push = sample_push(16, 1.0, "cpu")
    pushed = add_push(state, push)
    for index in (0, 1, 4, 5):
        assert pushed[index] is state[index]
    torch.testing.assert_close(pushed[2] - state[2], push[:, :3])
    torch.testing.assert_close(pushed[3] - state[3], push[:, 3:])
    (pushed[2].sum() + pushed[3].sum()).backward()
    assert torch.equal(state[2].grad, torch.ones_like(state[2]))
    assert torch.equal(state[3].grad, torch.ones_like(state[3]))


def test_sample_push_stays_within_scaled_range() -> None:
    push = sample_push(4096, 2.0, "cpu", torch.Generator().manual_seed(0))
    bound = 2.0 * torch.tensor(PUSH_RANGE)
    assert bool((push.abs() <= bound + 1e-6).all())
    active = bound > 0  # linear kicks only (PUSH_RANGE's angular entries are zero)
    assert bool((push.abs().amax(0)[active] > 0.9 * bound[active]).all())  # range covered


def test_replay_defaults_push_to_zero_and_windows_align_it() -> None:
    buffer = ReplayBuffer(8, 1, "cpu", num_joints=2, context_dim=2)
    for time in range(8):
        transition = {
            "s": torch.full((1, 17), float(time)),
            "s1": torch.full((1, 17), float(time) + 0.5),
            "a": torch.zeros(1, 2),
            "a_prev": torch.zeros(1, 2),
            "xi": torch.zeros(1, 2),
            "r": torch.zeros(1),
            "f": torch.tensor([time]),
            "ep_step": torch.tensor([time], dtype=torch.int32),
            "done": torch.tensor([False]),
        }
        if time % 2 == 1:  # only odd steps carry a push
            transition["push"] = torch.full((1, 6), float(time))
        buffer.add(transition)
    assert torch.equal(buffer.data["push"][0::2], torch.zeros(4, 1, 6))
    window = buffer.sample_windows(16, 3)
    start = window["s"][:, :, 0].long()  # s encodes the time index
    expected = torch.where(
        start % 2 == 1, start.float(), torch.zeros_like(start, dtype=torch.float)
    )
    torch.testing.assert_close(window["push"][..., 0], expected)


def test_world_model_window_replays_logged_pushes(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.cfg = replace(learner.cfg, wm_window=4)
    learner.wm.step = lambda state, action, context: state  # identity dynamics isolate the kicks
    batch = 3
    start = flatten_state(_random_state(batch))
    states = start.unsqueeze(1).expand(batch, 4, -1).clone()
    actions = torch.zeros(batch, 4, 2)
    contexts = torch.zeros(batch, 4, 2)
    pushes = torch.randn(batch, 4, 6)
    predicted = learner._wm_rollout(states, actions, contexts, pushes)
    v0, w0 = unflatten_state(start)[2], unflatten_state(start)[3]
    for index in range(4):
        # the kick at index 0 produced the logged start state already: not re-applied
        expected = pushes[:, 1 : index + 1].sum(1)
        torch.testing.assert_close(predicted[2][:, index], v0 + expected[:, :3])
        torch.testing.assert_close(predicted[3][:, index], w0 + expected[:, 3:])


def test_imagined_kick_changes_only_the_continued_state(tmp_path) -> None:
    learner = SuperMimicLearner(
        FakeEnv(), replace(_config(), imag_push_rate=1.0, imag_push_scale=1.0), tmp_path
    )
    learner.wm.step = lambda state, action, context: state
    batch = 4
    state = _random_state(batch)
    hist = torch.zeros(batch, learner.cfg.hist, learner.mdp.obs_dim)
    frames = torch.zeros(batch, dtype=torch.long)
    context = torch.zeros(batch, 2)
    goal = learner.mdp.goal(frames, learner.cfg.goal_frames)
    pre, continued, *_ = learner._imag_step(
        state, hist, torch.zeros(batch, 2), frames, context, context, goal
    )
    for index in (0, 1, 4, 5):
        torch.testing.assert_close(continued[index], pre[index])
    kick = torch.cat((continued[2] - pre[2], continued[3] - pre[3]), -1)
    active = torch.tensor(PUSH_RANGE) > 0
    assert bool((kick[:, active].abs() > 0).all())
    assert bool((kick.abs() <= torch.tensor(PUSH_RANGE) + 1e-6).all())

    no_push = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "off")
    no_push.wm.step = lambda state, action, context: state
    pre, continued, *_ = no_push._imag_step(
        state, hist, torch.zeros(batch, 2), frames, context, context, goal
    )
    assert continued is pre


def test_training_updates_run_with_imagined_pushes(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), replace(_config(), imag_push_rate=0.5), tmp_path)
    learner.collect(3)
    wm_loss, wm_error = learner.wm_update()
    metrics = learner.policy_update()
    for metric in (wm_loss, wm_error, *metrics):
        assert metric.ndim == 0
        assert torch.isfinite(metric)
