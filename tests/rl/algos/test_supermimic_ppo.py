"""CPU tests for the PPO-Sim baseline."""

from __future__ import annotations

import torch
from test_supermimic_learner import FakeEnv  # same directory (tests are not a package)

from unisupermimic.rl.algos.supermimic.ppo import PPOConfig, PPOLearner


def _config(**overrides) -> PPOConfig:
    base = dict(
        policy_hidden=(8,), critic_hidden=(8,), hist=2, goal_frames=2, amp="none",
        steps_per_env=4, epochs=2, minibatches=2, iterations=2, max_hours=1.0,
        ckpt_minutes=1e9, log_every=1,
    )
    base.update(overrides)
    return PPOConfig(**base)


def test_rollout_and_update_are_finite_and_change_the_policy(tmp_path) -> None:
    learner = PPOLearner(FakeEnv(num_envs=4), _config(), tmp_path)
    before = [p.detach().clone() for p in learner.pi.parameters()]
    batch = learner.rollout()
    assert batch["a"].shape == (4, 4, 2)
    for key in ("logp", "v", "r", "ret", "adv"):
        assert batch[key].shape == (4, 4)
        assert bool(torch.isfinite(batch[key]).all())
    kl, value_loss = learner.update(batch)
    assert kl >= 0 and value_loss >= 0
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.pi.parameters()))


def test_truncated_steps_bootstrap_the_value(tmp_path) -> None:
    # FakeEnv ends every episode at step 3 with term=False: a truncation, so the reward of that
    # step must carry gamma * V(s1) (zero critic weights would hide this, so check it moved)
    learner = PPOLearner(FakeEnv(num_envs=2), _config(steps_per_env=3), tmp_path)
    torch.nn.init.constant_(learner.critic.net[-1].bias, 5.0)
    batch = learner.rollout()
    raw = learner.mdp.max_step_reward  # FakeEnv rewards are <= max_step_reward
    assert bool((batch["r"][-1] > raw).all())  # bootstrapped
    assert bool((batch["r"][:-1] <= raw + 1e-6).all())


def test_checkpoint_loads_into_the_supermimic_evaluator(tmp_path) -> None:
    from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy

    env = FakeEnv(num_envs=2)
    learner = PPOLearner(env, _config(), tmp_path)
    summary = learner.train()
    assert summary["iteration"] == 2
    checkpoint = torch.load(tmp_path / "ckpt_000002.pt", weights_only=True)
    assert checkpoint["args"]["algo"] == "ppo"
    policy = load_policy(env, checkpoint, device="cpu")
    hist = torch.randn(3, 2, env.mdp.obs_dim)
    goal = torch.randn(3, 2 * env.mdp.goal_frame_dim)
    torch.testing.assert_close(policy(hist, goal), learner.pi(hist, goal))  # the Gaussian mean
    rows = (tmp_path / "train_log.csv").read_text().strip().splitlines()
    assert rows[0].startswith("iteration,wall_h,sim_steps") and len(rows) == 3
