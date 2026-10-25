<h1 align="center">UniSuperMimic</h1>

<h3 align="center">SuperMimic reproduction on UniLab</h3>

<p align="center">Languages: English | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.28480"><img src="https://img.shields.io/badge/paper-arXiv--2610.28480-red" alt="Paper"></a>
  <a href="https://pypi.org/project/unilab/"><img src="https://img.shields.io/pypi/v/unilab" alt="UniLab PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

UniSuperMimic is an independent downstream reproduction of [SuperMimic](https://supermimic.github.io) on **UniLab** for Unitree G1 whole-body tracking of LAFAN1 motions, with **mjwarp, MuJoCo, and Motrix** backends. It includes a differentiable world model, a goal-conditioned policy, a Manager-Based task, tools for 40 motion assets, and entry points for training and deterministic full-library evaluation.

<p align="center">
  <a href="docs/assets/supermimic_comparison.mp4"><img src="docs/assets/supermimic_comparison.gif" alt="SuperMimic policy and reference motion comparison" width="95%"></a>
</p>

<p align="center"><em>Solid robot: SuperMimic policy. Blue ghost: reference motion. Click the image to view the MP4.</em></p>

Paper: [*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*](https://arxiv.org/abs/2610.28480), Qiayuan Liao et al. The implementation references `supermimic_vibed` and depends on PyPI `unilab`.

## Repository Structure

```text
unisupermimic/
├── src/unisupermimic/
│   ├── rl/
│   │   ├── algos/supermimic/       # World model, policy, critic, replay, and learner
│   │   └── contracts/supermimic.py # Environment/algorithm tensor contract
│   ├── tasks/motion_tracking/g1/supermimic/
│   │   ├── env.py                  # Manager-Based environment and backend factory
│   │   ├── manager_terms.py        # Command/action/observation/event/reward terms
│   │   ├── motion.py, mdp.py       # Motion library and differentiable MDP
│   │   ├── evaluate.py             # Deterministic full-clip evaluation from frame zero
│   │   ├── render.py               # Policy rollout and reference-motion comparison video
│   │   └── registry.py             # G1SuperMimic registration
│   ├── scripts/train_supermimic.py # Training, resume, and evaluation entry point
│   ├── conf/supermimic/            # Hydra configuration
│   ├── assets/robots/g1/           # G1 MJCF; meshes/URDF are downloaded and untracked
│   ├── assets/motions/             # LAFAN1 CSV files, downloaded and untracked
│   └── sim2sim.py                  # Policy I/O, network shape, and asset identity checks
├── docs/assets/                    # Result curves, MP4, and GIF
├── tools/prepare_supermimic_assets.py
├── tests/
├── pyproject.toml, uv.lock
├── LICENSE, LICENSES/, NOTICE
└── Makefile
```

## Installation

Requirements: Linux, Python 3.10–3.13, and [uv](https://docs.astral.sh/uv/). A CUDA GPU is required for mjwarp training. Run all commands from the repository root. `uv run` uses `uv.lock` to synchronize the project, PyPI `unilab`, all three simulation backends, and development dependencies.

```bash
uv run unisupermimic-train --help
```

Run `uv lock` after changing dependencies.

On AMD ROCm hosts, install from the pre-resolved ROCm lockfile instead (a ROCm
PyTorch build with the same pinned dependency set):

```bash
make sync-rocm
```

CUDA-only backends (mjwarp) stay unsupported there; train with the MuJoCo
host-bridge backend (`task=g1_super_mimic/mujoco`). `make sync-rocm` swaps
`pyproject.toml`/`uv.lock` for the ROCm profile; run `make sync-default` to
switch back. Maintainers regenerate the ROCm lockfile after dependency changes
with `make lock-rocm`.

### Prepare Assets

```bash
# Download LAFAN1/G1 data from Hugging Face, then generate and validate MJCF.
uv run tools/prepare_supermimic_assets.py

# Validate existing assets offline.
uv run tools/prepare_supermimic_assets.py --validate-only
```

Assets come from `lvhaidong/LAFAN1_Retargeting_Dataset`. A complete asset set contains 40 CSV files with 264,705 source frames and 36 columns per frame. Meshes, URDF, CSV files, logs, and checkpoints are excluded by `.gitignore`.

## Training

Run the following commands from the repository root and set `CUDA_VISIBLE_DEVICES` to an available GPU when using CUDA.

```bash
# Default: mjwarp, 4096 parallel environments, 12000 iterations.
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.log_dir=runs/supermimic_mjwarp_001

# Resume from last.pt in the same run and train to 16000 iterations.
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  algo.resume=true \
  algo.load_run=runs/supermimic_mjwarp_001 \
  algo.checkpoint=last.pt \
  algo.iterations=16000

# Select MuJoCo through the task owner.
uv run unisupermimic-train \
  task=g1_super_mimic/mujoco \
  training.log_dir=runs/supermimic_mujoco_001

# Select Motrix through the task owner.
uv run unisupermimic-train \
  task=g1_super_mimic/motrix \
  training.log_dir=runs/supermimic_motrix_001
```

Use the same `task=g1_super_mimic/<backend>` override when resuming or evaluating a MuJoCo or Motrix run. Motrix uses fixed friction; center-of-mass and joint-offset domain randomization remain enabled. Use Hydra syntax for other configuration overrides. For a quick smoke run, override options such as `algo.num_envs`, `algo.iterations`, `algo.wm_hidden`, `algo.policy_hidden`, and `algo.compile=false`. See `src/unisupermimic/conf/supermimic/config.yaml` for the complete default configuration.

## Evaluation

Evaluation creates one environment per motion, starts from its first frame, disables observation and action noise, and runs the complete clip. Strict sim2sim validation checks policy I/O, network shape, and MJCF identity before environment construction; evaluation may use another motion library (`--motion_pack`), and only the library is allowed to differ.

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.play_only=true \
  algo.load_run=runs/supermimic_mjwarp_001 \
  algo.checkpoint=12000
```

Results are written to `ckpt_012000_eval.json` in the same directory. You can also evaluate a checkpoint directly. The standalone evaluator and video exporter accept `--backend mjwarp`, `--backend mujoco`, or `--backend motrix` and default to mjwarp:

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt
```

## Video Export

The following command rolls out `jumps1_subject1` continuously from a checkpoint, trims the result to the jumping segment from 6–14 seconds, and writes a 25 fps comparison MP4:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run unisupermimic-video \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt \
  --clip jumps1_subject1 \
  --start 6 \
  --seconds 8 \
  --output docs/assets/supermimic_comparison.mp4
```

By default, the reference root XY is aligned with the policy to highlight whole-body tracking. Pass `--global-ghost` to preserve the reference motion's global displacement.

Several checkpoints render side by side (one panel each, with the running score and a banner once a panel falls), several clips write `<output stem>_<clip>.mp4` each, and `--push_speed` replays the push test (mjwarp). The rollout is the deterministic evaluation loop itself. To keep the GPU free, render on the CPU MuJoCo backend; `tools/softgl.sh` installs Mesa software OpenGL on first use for machines without an EGL driver:

```bash
source tools/softgl.sh
CUDA_VISIBLE_DEVICES= uv run unisupermimic-video --backend mujoco \
  --ckpt runs/uni_base_s0/ckpt_012000.pt runs/uni_push_s0/ckpt_012000.pt \
  --labels baseline "push injection" \
  --clip dance2_subject4 fallAndGetUp1_subject1 --seconds 10 \
  --output videos/push_ab.mp4
```

## Extensions (branch `smimic-ext`)

### Push injection and the push test

Training with `env.commands.motion.push_scale=1.0` kicks the root every 1–3 s in the simulator (BeyondMimic's range: ±0.5 m/s in x/y, ±0.2 m/s in z; mjwarp only, exact). Each kick is logged in the replay buffer and added back at the same step of world-model training windows. `algo.imag_push_rate=0.01 algo.imag_push_scale=1.0` also kicks imagined rollouts; rewards use the pre-kick state.

The push test kicks every policy with the same seeded horizontal kicks every `--push_every` steps (default 150 = 3 s) and writes `ckpt_*_eval_push<v>.json`:

```bash
uv run unisupermimic-eval --ckpt runs/uni_push_s0/ckpt_012000.pt --push_speed 1.0
```

### PPO-Sim baseline (paper Fig. 4a)

BeyondMimic PPO in the simulator, with the same observations, goal, and critic input as SuperMimic; settings in the `ppo:` section of `config.yaml` (8192 environments per GPU, 18 h, checkpoints every 30 min). Its checkpoints load into `unisupermimic-eval` and `unisupermimic-video`.

```bash
uv run python -m unisupermimic.scripts.train_ppo training.log_dir=runs/ppo_g1
```

### Multi-GPU training

Data parallel with one rank per GPU: gradients and normalizer moments are averaged, the learning rate is scaled by √N (Table A.5), and each rank seeds its environments differently. Rank 0 logs and saves.

```bash
uv run torchrun --standalone --nproc_per_node=2 -m unisupermimic.scripts.train_ddp \
  training.log_dir=runs/g2        # add --ppo before the overrides for PPO-Sim
```

### BONES-SEED motion library

Accept the dataset license on Hugging Face (`bones-studio/seed`) and `export HF_TOKEN=...`. Then download, check the detected layout, and build a packed 50 Hz library. The build applies the name filter and the foot-contact feasibility filter of Appendix E:

```bash
uv run tools/prepare_bones_seed.py download --root data/bones_seed     # ~23.5 GB
uv run tools/prepare_bones_seed.py inspect  --root data/bones_seed
uv run tools/prepare_bones_seed.py build    --root data/bones_seed --out data/bones_pack
uv run unisupermimic-train env.commands.motion.motion_pack=data/bones_pack training.log_dir=runs/bones
uv run unisupermimic-eval --ckpt runs/bones/ckpt_012000.pt --motion_pack data/bones_pack
```

Evaluation runs in chunks of `eval.max_envs` clips (default 8192). Without `--motion_pack`, it uses the LAFAN library: evaluation may change the motion library, but nothing else in the training contract.

### Training techniques from go2-supermimic

Five task-agnostic techniques from [go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) (Go2 chasing and dodging), each off by default (`algo.*`, documented in `SuperMimicConfig`):

| technique | switch | what it does |
|---|---|---|
| rare-event world-model batches | `wm_prioritized_frac`, `wm_archive_frac` | a share of each world-model batch is the rarest of 4× as many candidate windows (foot contact changes, airborne steps, falls); another share comes from an archive of rare windows (flight, hard vertical base motion, falls) kept beyond the replay horizon. Contacts use the task's `foot_heights` hook (G1: ankle-roll links below `contact_height` = 0.10 m) |
| correlated exploration | `explore_frac` | a share of the envs collects with Ornstein-Uhlenbeck action noise (`explore_noise`, `explore_theta`) instead of white noise |
| stability | `actor_lr_scale`, `actor_ramp_iters`, `collapse_fall_rate` | actor lr multiplier; linear actor-lr ramp whenever the actor (re)starts; collapse guard: two log windows in a row above the fall rate (per 1000 env steps) restore actor and critic from the last snapshot with half the lr |
| warm start | `init_from`, `reset_critic` | initialise from another run's checkpoint; a policy whose observation gained terms starts exactly as before (new inputs at zero weight); a fresh critic trains alone for `critic_warmup_iters` |
| data-collection driver | `driver_ckpt`, `driver_frac`, `driver_fade` | another policy (SuperMimic or PPO-Sim checkpoint) acts in a share of the envs so the world model sees its skills; the share fades out between the two `driver_fade` iterations |

```bash
uv run tools/launch.py run go2 --gpus 0,1,2,3     # each of rare / explore / stable alone and all three
uv run unisupermimic-train training.log_dir=runs/driven \
  algo.driver_ckpt=runs/uni_ppo_g1/ckpt_XXXXXX.pt algo.driver_frac=0.25 algo.driver_fade=[2000,6000]
```

### Go2 chase / dodge (go2-supermimic on UniLab)

[go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) trains a Unitree Go2 to chase a fleeing ball while dodging thrown balls and bars, with its own SuperMimic variant (70-dim yaw-frame world-model state with limb points, analytic chase/dodge rewards in imagination). Its algorithm and target dynamics are vendored under `unisupermimic.go2chase` (MIT); the environment is rewritten for UniLab (`go2chase/env`, task `Go2ChaseFlat`, mjwarp or mujoco). Flat ground only so far: static obstacles (mocap boxes, LiDAR) and generated terrain (height scan, terrain driver) are not ported yet.

The port is checked against go2-supermimic's own mjlab environment: its pretrained `weights/flat.pt` scores the same there and here (chase + dodge, 256 envs, 60 s: 33.5 vs 32.5 tags/min, ball hit rates 0.21-0.25 vs 0.23-0.27, falls 0.004 vs 0.008 per minute), and the actor observation rebuilt from the world-model state equals the env's exactly. Actuators are MuJoCo position actuators with go2's gains and torque limits; mjlab's DC-motor torque-speed curve and 0-2 substep command delay are not modelled (removing them in mjlab leaves the metrics unchanged).

```bash
uv run tools/prepare_go2_assets.py                 # Menagerie Go2 meshes (gitignored)
uv run unisupermimic-go2-train --run-name flat     # go2 TrainCfg flags; runs/go2chase/<stamp>_flat
uv run unisupermimic-go2-eval --checkpoint runs/go2chase/<run>/model_40000.pt --mode both
uv run tools/launch.py run go2_robot --gpus 0,1    # go2_flat_s0/s1, 40k iterations + evaluation
```

On one L4 a full-size run (4096 envs) takes about 1.1 s per iteration and 17 GB.

### Experiment launcher (H100/H200)

`tools/launch.py` queues named experiments over the visible GPUs (strict FIFO, one job per free GPU or GPU group). Each job trains, then evaluates the final checkpoint: nominal plus pushes of 0.5, 1.0, and 1.5 m/s, and the BONES pack for BONES runs. `tools/report_push.py` prints the table, with mean ± std over seeds.

```bash
uv run tools/launch.py list                       # experiments and groups
uv run tools/launch.py run push_ab --gpus 0,1     # uni_base_s0 and uni_push_s0, in parallel
uv run tools/launch.py run push_ab --dry          # print the commands only
uv run tools/launch.py eval push_ab --curve       # nominal eval of every saved checkpoint
uv run tools/report_push.py
```

Groups: `push_ab`, `push_ablation` (simulator or imagined kicks only), `push_seeds`, `multi_gpu`, `ppo` (1/4/8 GPUs), `bones`, `go2` (training techniques from go2-supermimic), `go2_robot` (Go2 chase / dodge). On one L4, a default 12k-iteration run takes about 5 h to train and 1.4 h for the four evaluations.

## Results

Configuration: PyPI `unilab 1.5.0`, mjwarp, and 4096 training environments. Evaluation covers 40 LAFAN1/G1 motions. `norm_return` is the macro average across clips; `frame-weighted` is weighted by clip length.

| checkpoint | norm_return | frame-weighted | failed clips |
|---|---:|---:|---:|
| ckpt_012000.pt | 0.827227 | 0.829952 | 0 / 40 |

![Training and evaluation curves through iteration 12000](docs/assets/success_curve.png)

## Tests

```bash
make check       # ruff, mypy, pyright
make test        # Non-slow tests
make test-all    # Include production backend slow tests; mjwarp requires CUDA
uv run tools/prepare_supermimic_assets.py --validate-only
```

## License

The repository code is released under [Apache-2.0](LICENSE). The reference implementation's MIT license and original copyright notice are preserved in [LICENSES/MIT-supermimic.txt](LICENSES/MIT-supermimic.txt); see also [NOTICE](NOTICE). UniLab, UniSim, MuJoCo/mjwarp, MotrixSim, and the datasets retain their respective upstream licenses. This repository does not relicense those external dependencies or downloaded assets.
