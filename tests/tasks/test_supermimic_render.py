"""Tests for SuperMimic reference-ghost video rendering."""

from pathlib import Path

import numpy as np
import pytest
import torch

from unisupermimic.tasks.motion_tracking.g1.supermimic import render

SCENE = Path(__file__).resolve().parents[2] / render.DEFAULT_SCENE


def test_flattened_qpos_extracts_root_and_joint_positions() -> None:
    state = torch.arange(71, dtype=torch.float32).reshape(1, 71)

    qpos = render.flattened_qpos(state)

    assert qpos.shape == (1, 36)
    torch.testing.assert_close(qpos[0, :7], state[0, :7])
    torch.testing.assert_close(qpos[0, 7:], state[0, 13:42])


def test_comparison_xml_adds_noncolliding_reference_ghost() -> None:
    xml = render.comparison_xml(SCENE)

    assert '<body name="ref_pelvis">' in xml
    assert '<freejoint name="ref_floating_base_joint"' in xml
    assert 'rgba="0.35 0.65 1.0 0.25"' in xml
    assert '<geom contype="0" conaffinity="0" group="1"' in xml
    assert "<keyframe>" not in xml


def test_comparison_model_loads_and_removes_temporary_xml() -> None:
    temporary_files = set(SCENE.parent.glob(".render_*.xml"))

    model = render.load_comparison_model(SCENE)

    assert model.nq == 72
    assert set(SCENE.parent.glob(".render_*.xml")) == temporary_files


def test_comparison_frame_preserves_shape_and_adds_labels() -> None:
    image = np.full((540, 960, 3), 128, dtype=np.uint8)

    frame = render.make_comparison_frame(image, "dance1_subject1  t= 0.0s")

    assert frame.shape == image.shape
    assert frame.dtype == np.uint8
    assert not np.array_equal(frame, image)


def test_rollout_qpos_trims_continuous_rollout(monkeypatch) -> None:
    from types import SimpleNamespace

    num_frames = 200
    root_p = torch.zeros(num_frames, 3)
    root_p[:, 2] = torch.arange(num_frames)
    motion = SimpleNamespace(
        names=["jumps1_subject1"],
        clip_start=torch.tensor([0]),
        root_p=root_p,
        root_R=torch.eye(3).expand(num_frames, 3, 3),
        q=torch.zeros(num_frames, 29),
    )

    class FakeMdp:
        def __init__(self) -> None:
            self.motion = motion

        def obs(self, state, previous_action, context, frames):
            return torch.zeros(frames.shape[0], 1)

        def goal(self, frames, num_frames):
            return torch.zeros(frames.shape[0], num_frames)

    class FakeEnv:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.supermimic_mdp = FakeMdp()
            self.steps = 0

        def _current(self, frames):
            state = torch.zeros(1, 71)
            state[0, 2] = frames[0].float()
            state[0, 3] = 1.0
            return {
                "s": state,
                "f": frames.clone(),
                "a_prev": torch.zeros(1, 29),
                "xi": torch.zeros(1, 1),
            }

        def reset_to_frames(self, frames):
            self.current = self._current(frames)
            return {}, {render.SUPERMIMIC_STATE_KEY: self.current}

        def step(self, action):
            self.steps += 1
            self.current = self._current(self.current["f"] + 1)
            return SimpleNamespace(
                info={
                    render.SUPERMIMIC_STATE_KEY: self.current,
                    render.SUPERMIMIC_TRANSITION_KEY: {"done": torch.tensor([False])},
                }
            )

    def policy(history, goal):
        return torch.zeros(history.shape[0], 29)

    monkeypatch.setattr(render, "load_policy", lambda *_: policy)
    env = FakeEnv()

    reference, policy_qpos = render.rollout_qpos(
        env,
        {"args": {"hist": 1, "goal_frames": 1}},
        "jumps1_subject1",
        seconds=0.2,
        start=0.1,
    )

    assert env.steps == 14
    assert reference.shape == (10, 36)
    assert policy_qpos.shape == (10, 36)
    np.testing.assert_allclose(reference[:, 2], np.arange(5, 15))
    np.testing.assert_allclose(policy_qpos[:, 2], np.arange(5, 15))


@pytest.mark.parametrize("backend", render.SUPPORTED_BACKENDS)
def test_video_parser_accepts_backend(backend: str) -> None:
    args = render._argument_parser().parse_args(["--ckpt", "run/ckpt.pt", "--backend", backend])

    assert args.backend == backend


def test_rollout_panels_hold_the_pose_after_a_fall(monkeypatch) -> None:
    from types import SimpleNamespace

    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD

    num_frames = 100
    root_p = torch.zeros(num_frames, 3)
    root_p[:, 2] = torch.arange(num_frames)
    motion = SimpleNamespace(
        names=["walk", "fall"],
        clip_start=torch.tensor([0, 50]),
        clip_end=torch.tensor([49, 99]),
        root_p=root_p,
        root_R=torch.eye(3).expand(num_frames, 3, 3),
        q=torch.zeros(num_frames, 29),
    )

    class FakeMdp:
        def __init__(self) -> None:
            self.motion = motion

        def obs(self, state, previous_action, context, frames):
            return torch.zeros(frames.shape[0], 1)

        def goal(self, frames, num_frames):
            return torch.zeros(frames.shape[0], num_frames)

    def state_at(frames):  # root height = frame index
        state = torch.zeros(frames.shape[0], 71)
        state[:, 2] = frames.float()
        state[:, 3] = 1.0
        return state

    class FakeEnv:  # 3 environments: the 2 clips and a spare one
        num_envs = 3
        device = torch.device("cpu")
        supermimic_mdp = FakeMdp()

        def _current(self, frames):
            n = frames.shape[0]
            zeros = torch.zeros(n, 29)
            return {"s": state_at(frames), "f": frames, "a_prev": zeros, "xi": zeros[:, :1]}

        def reset_to_frames(self, frames):
            self.start, self.t = frames.clone(), 0
            self.current = self._current(frames.clone())
            return {}, {render.SUPERMIMIC_STATE_KEY: self.current}

        def step(self, action):
            frames = self.current["f"]
            term = torch.tensor([False, self.t == 2, False])  # clip "fall" falls at step 2
            self.t += 1
            next_frames = torch.where(term, self.start, frames + 1)  # auto-reset
            transition = {
                "s": state_at(frames),
                "s1": state_at(frames + 1),
                "f": frames,
                "r": torch.full((3,), 0.5 * MAX_STEP_REWARD),
                "term": term,
                "done": term,
            }
            self.current = self._current(next_frames)
            return SimpleNamespace(
                info={
                    render.SUPERMIMIC_STATE_KEY: self.current,
                    render.SUPERMIMIC_TRANSITION_KEY: transition,
                }
            )

    monkeypatch.setattr(render, "load_policy", lambda *_: lambda h, g: torch.zeros(len(h), 29))

    panels = render.rollout_panels(
        FakeEnv(), {"args": {"hist": 1, "goal_frames": 1}}, ["walk", "fall"], 0.1, "run"
    )

    walk, fall = panels["walk"], panels["fall"]
    assert walk.fall is None and fall.fall == 2 and fall.label == "run"
    assert walk.policy.shape == fall.policy.shape == (6, 36)  # start + 5 steps
    np.testing.assert_allclose(walk.policy[:, 2], np.arange(0, 6))
    np.testing.assert_allclose(walk.reference[:, 2], np.arange(0, 6))
    np.testing.assert_allclose(fall.policy[:, 2], [50, 51, 52, 53, 53, 53])  # held, not reset
    np.testing.assert_allclose(fall.reference[:, 2], [50, 51, 52, 53, 53, 53])
    np.testing.assert_allclose(walk.score, [0, 0.5, 0.5, 0.5, 0.5, 0.5])


def test_panel_frame_shows_the_fall_banner_only_after_the_fall() -> None:
    panel = render.Panel(
        label="run",
        policy=np.zeros((10, 36)),
        reference=np.zeros((10, 36)),
        score=np.full(10, 0.5),
        fall=4,
    )
    image = np.full((240, 320, 3), 128, dtype=np.uint8)

    before, after = render._panel_frame(image, panel, 4), render._panel_frame(image, panel, 5)

    assert before.shape == image.shape
    np.testing.assert_array_equal(before[-10:, 250:], image[-10:, 250:])
    assert (after[-10:, 250:, 0] == 200).all()  # red banner (right of its text)


def test_video_parser_accepts_several_checkpoints_and_clips() -> None:
    args = render._argument_parser().parse_args(
        ["--ckpt", "a/c.pt", "b/c.pt", "--labels", "A", "B", "--clip", "x", "y"]
    )

    assert args.ckpt == [Path("a/c.pt"), Path("b/c.pt")]
    assert args.labels == ["A", "B"] and args.clip == ["x", "y"]


def test_render_panels_hold_a_shorter_panel_to_the_longest(monkeypatch, tmp_path) -> None:
    def panel(label: str, length: int, fall: int | None) -> render.Panel:
        qpos = np.zeros((length, 36))
        qpos[:, 3] = 1.0
        return render.Panel(label, qpos, qpos.copy(), np.full(length, 0.5), fall)

    shown: list[tuple[str, int]] = []
    monkeypatch.setattr(
        render, "_panel_frame", lambda image, p, index: shown.append((p.label, index)) or image
    )
    monkeypatch.setattr(render, "_pose", lambda *args: None)

    class FakeRenderer:
        def update_scene(self, data, camera):
            pass

        def render(self):
            return np.zeros((48, 64, 3), dtype=np.uint8)

        def close(self):
            pass

    class FakeCamera:
        lookat = np.zeros(3)

    monkeypatch.setattr(
        render, "_comparison_renderer", lambda *args: (None, None, FakeCamera(), FakeRenderer())
    )
    written = render.render_panels(
        [panel("fell", 5, 3), panel("ok", 11, None)],
        tmp_path / "out.mp4",
        caption="clip",
        width=64,
        height=48,
        fps=25,
    )

    assert written == 6  # rows 0, 2, ..., 10 of the longest panel
    assert [index for label, index in shown if label == "fell"] == [0, 2, 4, 6, 8, 10]
    assert (tmp_path / "out.mp4").stat().st_size > 0


def test_trim_keeps_clip_time_and_shifts_the_fall() -> None:
    rows = np.arange(20, dtype=float)[:, None].repeat(36, 1)
    panel = render.Panel("run", rows, rows.copy(), np.arange(20.0), fall=12)

    trimmed = render.trim(panel, 0.2)  # 10 rows

    assert trimmed.offset == 10 and trimmed.fall == 2
    np.testing.assert_array_equal(trimmed.policy[:, 0], np.arange(10, 20))
    assert render.trim(panel, 0.3).fall == -1  # fell before the window: banner from row 0
    with pytest.raises(ValueError, match="start exceeds"):
        render.trim(panel, 0.4)
