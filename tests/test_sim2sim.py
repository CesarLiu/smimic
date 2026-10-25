"""SuperMimic-owned sim2sim contract tests."""

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf
from unilab.utils.sim2sim import CrossBackendIncompatibleError

from unisupermimic.sim2sim import extract_contract_snapshot, resolve_sim2sim_config


def _cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "algo": {
                "goal_frames": 10,
                "policy_hidden": [2048, 1024, 512, 256],
                "amp": "bf16",
            },
            "env": {
                "observations": {
                    "policy": {
                        "terms": {
                            "obs": {
                                "func": "unisupermimic.tasks.motion_tracking.g1.supermimic.manager_terms.supermimic_obs"
                            }
                        }
                    }
                },
                "actions": {
                    "joint_pos": {
                        "_target": "unisupermimic.tasks.motion_tracking.g1.supermimic.manager_terms.SuperMimicActionCfg"
                    }
                },
                "commands": {
                    "motion": {
                        "goal_frames": 10,
                        "context_dim": 33,
                        "motion_files": [
                            "src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv"
                        ],
                        "mjcf_path": "src/unisupermimic/assets/robots/g1/supermimic.xml",
                    }
                },
            },
        }
    )


def test_snapshot_contains_supermimic_policy_and_motion_fields() -> None:
    snapshot = extract_contract_snapshot(_cfg())

    assert snapshot["algo.goal_frames"] == 10
    assert snapshot["algo.policy_hidden"] == [2048, 1024, 512, 256]
    assert snapshot["env.commands.motion.context_dim"] == 33
    assert len(snapshot["env.commands.motion.motion_files"]) == 1


def test_unilab_to_downstream_relocation_is_strictly_compatible(tmp_path: Path) -> None:
    cfg = _cfg()
    snapshot = extract_contract_snapshot(cfg)
    snapshot["env.commands.motion.motion_files"] = [
        "src/unilab/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv"
    ]
    snapshot["env.commands.motion.mjcf_path"] = "src/unilab/assets/robots/g1/supermimic.xml"
    snapshot = json.loads(
        json.dumps(snapshot).replace(
            "unisupermimic.tasks.motion_tracking.g1.supermimic",
            "unilab.tasks.motion_tracking.g1.supermimic",
        )
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": snapshot}), encoding="utf-8"
    )

    assert resolve_sim2sim_config(tmp_path, cfg, strict=True) is cfg


def test_policy_shape_mismatch_fails_closed(tmp_path: Path) -> None:
    source = _cfg()
    target = _cfg()
    OmegaConf.update(target, "algo.policy_hidden", [512, 256])
    (tmp_path / "run_config.json").write_text(
        json.dumps({"contract_snapshot": extract_contract_snapshot(source)}), encoding="utf-8"
    )

    with pytest.raises(CrossBackendIncompatibleError, match="algo.policy_hidden"):
        resolve_sim2sim_config(tmp_path, target, strict=True)


def _write_snapshot(run_dir: Path, cfg: DictConfig) -> None:
    (run_dir / "run_config.json").write_text(
        json.dumps({"contract_snapshot": extract_contract_snapshot(cfg)}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("env.commands.motion.motion_pack", "data/bones_pack"),  # absent in the source
        ("env.commands.motion.motion_files", ["other.csv"]),
    ],
)
def test_motion_library_change_is_allowed_only_for_evaluation(
    tmp_path: Path, path: str, value: object
) -> None:
    _write_snapshot(tmp_path, _cfg())
    target = _cfg()
    OmegaConf.update(target, path, value, force_add=True)

    with pytest.raises(CrossBackendIncompatibleError, match=path):
        resolve_sim2sim_config(tmp_path, target, strict=True)
    assert (
        resolve_sim2sim_config(tmp_path, target, strict=True, allow_library_change=True) is target
    )


def test_library_change_does_not_excuse_a_network_change(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, _cfg())
    target = _cfg()
    OmegaConf.update(target, "env.commands.motion.motion_pack", "data/bones_pack", force_add=True)
    OmegaConf.update(target, "algo.policy_hidden", [512, 256])

    with pytest.raises(CrossBackendIncompatibleError, match="algo.policy_hidden") as error:
        resolve_sim2sim_config(tmp_path, target, strict=True, allow_library_change=True)
    assert "motion_pack" not in str(error.value)
