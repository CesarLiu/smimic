"""CPU tests for the training techniques ported from go2-supermimic: rare-event world-model
batches, correlated exploration, actor schedule and collapse guard, warm start, drivers."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_supermimic_learner import FakeEnv, _config  # same directory (tests are not a package)

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner, expand_policy_inputs
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.rl.algos.supermimic.replay import WindowArchive
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY

STATE_DIM = 17  # FakeEnv: 13 + 2 joints x 2


class FallingEnv(FakeEnv):
    """Every episode (3 steps) ends in a fall rather than a timeout."""

    def step(self, actions):
        state = super().step(actions)
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        transition["term"] = transition["done"].clone()
        return state


def _window(batch: int, k: int, vz: float = 0.0, term_at: int | None = None) -> dict:
    s = torch.zeros(batch, k, STATE_DIM)
    s[..., 3] = 1.0  # identity quaternion (wxyz) in the flattened (p, quat, v, w, q, qd) layout
    s1 = s.clone()
    s1[..., 9] = vz  # v_z: p(3) + quaternion(4) + v(3) -> index 9
    term = torch.zeros(batch, k, dtype=torch.bool)
    mask = torch.ones(batch, k)
    if term_at is not None:
        term[:, term_at] = True
        mask[:, term_at + 1 :] = 0
    return {
        "s": s,
        "s1": s1,
        "a": torch.zeros(batch, k, 2),
        "xi": torch.zeros(batch, k, 2),
        "done": term.clone(),
        "term": term,
        "push": torch.zeros(batch, k, 6),
        "mask": mask,
    }


def test_archive_is_a_ring_buffer() -> None:
    archive = WindowArchive(3, "cpu")
    archive.add({"x": torch.tensor([[1.0], [2.0]])})
    archive.add({"x": torch.tensor([[3.0], [4.0]])})
    assert archive.size == 3
    assert sorted(archive.data["x"].flatten().tolist()) == [2.0, 3.0, 4.0]  # 1 overwritten
    assert archive.sample(5)["x"].shape == (5, 1)
    archive.add({"x": torch.zeros(0, 1)})  # nothing to keep is fine
    assert archive.size == 3


def test_rarity_scores_falls_flight_and_hard_vertical_motion(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    k = 4
    calm, fall, jump = _window(1, k), _window(1, k, term_at=1), _window(1, k, vz=1.0)
    window = {key: torch.cat((calm[key], fall[key], jump[key])) for key in calm}

    score, worthy = learner.window_rarity(window)  # FakeMdp has no foot_heights hook
    assert worthy.tolist() == [False, True, True]
    assert score[1] >= 4 and score[0] == 0 and score[2] == 1

    # with the hook: feet up (airborne) from step 1 on, as in a jump
    def feet(state):
        height = state[0][..., 2:3].expand(-1, 2)
        return height

    learner._foot_heights = feet
    window["s1"][0, 1:, 2] = 0.3  # calm window: base (and feet) up -> 1 contact change x 2 feet
    score, worthy = learner.window_rarity(window)
    assert score[0] == 2 + 3  # 2 foot liftoffs + 3 airborne steps
    assert bool(worthy[0])  # >= 2 airborne steps = flight


def test_wm_batches_mix_uniform_rare_and_archived_windows(tmp_path) -> None:
    cfg = replace(
        _config(),
        wm_batch=8,
        wm_prioritized_frac=0.25,
        wm_archive_frac=0.25,
        archive_size=64,
        archive_keep_prob=1.0,
    )
    learner = SuperMimicLearner(FallingEnv(num_envs=4), cfg, tmp_path)
    learner.collect(12)

    first = learner._wm_windows()  # archive empty: uniform + prioritized only
    assert first["s"].shape[0] == 8 and learner.archive is not None
    assert learner.archive.size > 0  # windows with falls were archived
    assert bool(learner.archive.data["term"][: learner.archive.size].any(-1).all())
    second = learner._wm_windows()
    assert second["s"].shape[0] == 8
    loss, error = learner.wm_update()
    assert torch.isfinite(loss) and torch.isfinite(error)


def test_correlated_exploration_only_in_its_envs_and_resets_on_done(tmp_path) -> None:
    cfg = replace(_config(), explore_frac=0.5, action_noise=0.0)
    learner = SuperMimicLearner(FakeEnv(num_envs=4), cfg, tmp_path)
    assert learner.explore.tolist() == [True, True, False, False]
    learner.collect(2)
    assert bool((learner.ou_noise[:2] != 0).all()) and bool((learner.ou_noise[2:] == 0).all())
    learner.collect(1)  # FakeEnv episodes end after 3 steps
    assert bool((learner.ou_noise == 0).all())


def test_actor_ramp_scale_and_freeze(tmp_path) -> None:
    cfg = replace(_config(), actor_lr_scale=0.5, actor_ramp_iters=10, wm_only_iters=20)
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    assert learner.actor_lr_factor(20) == 0.0
    assert learner.actor_lr_factor(25) == pytest.approx(0.25)
    assert learner.actor_lr_factor(40) == pytest.approx(0.5)
    assert SuperMimicLearner(FakeEnv(), _config(), tmp_path / "b").actor_lr_factor(0) == 1.0

    learner.collect(3)
    learner.actor_frozen_until = 5  # e.g. a fresh critic warming up
    before = [p.clone() for p in learner.pi.parameters()]
    critic_before = [p.clone() for p in learner.critic.parameters()]
    learner.policy_update()
    assert all(torch.equal(a, b) for a, b in zip(before, learner.pi.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, learner.critic.parameters()))


def test_collapse_guard_arms_then_rolls_back_after_two_strikes(tmp_path) -> None:
    cfg = replace(_config(), collapse_fall_rate=2.0)
    learner = SuperMimicLearner(FakeEnv(), cfg, tmp_path)
    learner.iteration = 10
    learner._take_snapshot()
    saved = {k: v.clone() for k, v in learner.pi.state_dict().items()}
    with torch.no_grad():
        for parameter in learner.pi.parameters():
            parameter.add_(1.0)  # the actor drifts
    old_optimizer = learner.opt_pi

    assert not learner.collapse_guard(5.0)  # not armed yet: early training falls a lot
    assert not learner.collapse_guard(1.0) and learner.collapse_armed
    assert not learner.collapse_guard(5.0)  # strike one
    assert learner.collapse_guard(5.0)  # strike two: roll back
    for key, value in learner.pi.state_dict().items():
        torch.testing.assert_close(value, saved[key])
    assert learner.lr_scale == 0.5 and learner.ramp_start == 10 and learner.rollbacks == 1
    assert learner.opt_pi is not old_optimizer
    assert not learner.collapse_guard(5.0)  # strikes start over


def test_collapse_guard_off_by_default(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.iteration = 10
    learner._take_snapshot()
    assert not any(learner.collapse_guard(100.0) for _ in range(3))


def test_expand_policy_inputs_acts_exactly_as_before() -> None:
    torch.manual_seed(0)
    old = Policy(6, 4, hist=3, z_dim=5, num_joints=2, hidden=(16, 8))
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.normal_()
        old.obs_norm.mean.normal_()
        old.obs_norm.var.uniform_(0.5, 2.0)
        old.obs_norm.count.fill_(100.0)
    new = Policy(9, 4, hist=3, z_dim=5, num_joints=2, hidden=(16, 8))
    expand_policy_inputs(old.state_dict(), new)

    obs, goal = torch.randn(7, 3, 6), torch.randn(7, 4)
    extra = torch.randn(7, 3, 3) * 100  # new terms: any value, zero weight
    torch.testing.assert_close(new(torch.cat((obs, extra), -1), goal), old(obs, goal))
    with pytest.raises(ValueError, match="only growth"):
        expand_policy_inputs(new.state_dict(), old)


def test_warm_start_loads_expands_and_schedules(tmp_path) -> None:
    cfg = replace(_config(), critic_warmup_iters=7, wm_rewarmup_iters=3)
    source = SuperMimicLearner(FakeEnv(), cfg, tmp_path / "source")
    full = source.save(tmp_path / "full.pt")
    small = source.save(tmp_path / "small.pt", include_optimizers=False)  # no critic

    target = SuperMimicLearner(FakeEnv(), replace(cfg, init_from=str(full)), tmp_path / "a")
    notes = target.warm_start(full)
    assert notes == ["world model loaded", "policy loaded", "critic loaded"]
    assert target.policy_start == 0 and target.actor_frozen_until == 0
    for key, value in source.pi.state_dict().items():
        torch.testing.assert_close(target.pi.state_dict()[key], value)

    fresh_critic = SuperMimicLearner(FakeEnv(), cfg, tmp_path / "b")
    assert "critic re-initialised" in fresh_critic.warm_start(small)[-1]
    assert fresh_critic.actor_frozen_until == 7

    other_wm = SuperMimicLearner(
        FakeEnv(),
        replace(
            cfg, test_hidden_dims=None, wm_hidden=(4,), policy_hidden=(16,), critic_hidden=(8,)
        ),
        tmp_path / "c",
    )
    with pytest.raises(ValueError):  # policy of a different width cannot be warm-started
        other_wm.warm_start(full)


def test_train_warm_starts_from_init_from(tmp_path) -> None:
    source = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "source")
    path = source.save(tmp_path / "source.pt")
    learner = SuperMimicLearner(FakeEnv(), replace(_config(), init_from=str(path)), tmp_path / "t")
    result = learner.train()
    assert result["completed"] and learner.iteration == 1


def test_driver_acts_in_the_last_envs_and_fades(tmp_path) -> None:
    source = SuperMimicLearner(FakeEnv(num_envs=4), _config(), tmp_path / "source")
    with torch.no_grad():
        for parameter in source.pi.parameters():
            parameter.normal_()
    path = source.save(tmp_path / "driver.pt", include_optimizers=False)
    cfg = replace(
        _config(), driver_ckpt=str(path), driver_frac=0.5, driver_fade=(10, 20), action_noise=0.0
    )
    learner = SuperMimicLearner(FakeEnv(num_envs=4), cfg, tmp_path / "t")
    assert learner.driver_share(5) == 0.5
    assert learner.driver_share(15) == pytest.approx(0.25)
    assert learner.driver_share(25) == 0.0

    learner.collect(1)
    actions = learner.buffer.data["a"][0]
    expected = source.pi(
        learner.obs_hist[:, -source.cfg.hist :],
        learner.mdp.goal(torch.arange(4), source.cfg.goal_frames),
    )
    own = learner.pi(learner.obs_hist, learner.mdp.goal(torch.arange(4), learner.cfg.goal_frames))
    torch.testing.assert_close(actions[2:], expected[2:].clamp(-20, 20))  # last half: the driver
    torch.testing.assert_close(actions[:2], own[:2].clamp(-20, 20))


def test_invalid_fractions_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="wm_prioritized_frac"):
        SuperMimicLearner(
            FakeEnv(), replace(_config(), wm_prioritized_frac=0.8, wm_archive_frac=0.5), tmp_path
        )
    with pytest.raises(ValueError, match="driver_ckpt"):
        SuperMimicLearner(FakeEnv(), replace(_config(), driver_frac=0.2), tmp_path)


def test_schedule_survives_a_checkpoint_roundtrip(tmp_path) -> None:
    learner = SuperMimicLearner(FakeEnv(), _config(), tmp_path)
    learner.lr_scale, learner.actor_frozen_until, learner.rollbacks = 0.25, 9, 2
    path = learner.save()
    restored = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "r")
    restored.load(path)
    assert (restored.lr_scale, restored.actor_frozen_until, restored.rollbacks) == (0.25, 9, 2)


class FootEnv(FakeEnv):
    """FakeEnv whose MDP has the optional foot_heights hook (feet at the base height)."""

    def __init__(self, num_envs: int = 2):
        super().__init__(num_envs)
        self.mdp.foot_heights = lambda state: state[0][..., 2:3].expand(*state[0].shape[:-1], 2)


def test_foot_heights_are_logged_once_and_used_for_rarity(tmp_path) -> None:
    plain = SuperMimicLearner(FootEnv(), _config(), tmp_path / "plain")
    assert "feet" not in plain.buffer.fields  # only logged when rare-event sampling is on

    cfg = replace(_config(), wm_prioritized_frac=0.5)
    learner = SuperMimicLearner(FootEnv(), cfg, tmp_path / "rare")
    learner.collect(4)
    assert learner.buffer.data["feet"].shape == (16, 2, 2)
    stored = learner.buffer.data["feet"][:4]
    torch.testing.assert_close(stored, learner.buffer.data["s1"][:4, :, 2:3].expand(-1, -1, 2))

    window = learner.buffer.sample_windows(3, 2)
    assert "feet" in window
    window["feet"].fill_(1.0)  # stored heights say: airborne after the first state
    window["mask"].fill_(1.0)
    window["term"].fill_(False)
    score, worthy = learner.window_rarity(window)
    assert bool(worthy.all())  # 2 airborne steps = flight
    assert bool((score >= 2).all())  # 2 airborne steps
