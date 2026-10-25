<h1 align="center">UniSuperMimic</h1>

<h3 align="center">基于 UniLab 的 SuperMimic 复现</h3>

<p align="center">语言：简体中文 | <a href="README.md">English</a></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.28480"><img src="https://img.shields.io/badge/paper-arXiv--2610.28480-red" alt="论文"></a>
  <a href="https://pypi.org/project/unilab/"><img src="https://img.shields.io/pypi/v/unilab" alt="UniLab PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 许可"></a>
</p>

UniSuperMimic 是 [SuperMimic](https://supermimic.github.io) 在 **UniLab** 上的独立下游复现，支持 **mjwarp、MuJoCo 和 Motrix** 后端，面向 Unitree G1 的 LAFAN1 全身运动跟踪。仓库包含可微世界模型、目标条件策略、Manager-Based 任务、40 段运动资产工具，以及训练和确定性全库评估入口。

<p align="center">
  <a href="docs/assets/supermimic_comparison.mp4"><img src="docs/assets/supermimic_comparison.gif" alt="SuperMimic 与参考动作对比" width="95%"></a>
</p>

<p align="center"><em>实体机器人为 SuperMimic 策略，蓝色半透明机器人为参考动作。点击图片查看 MP4。</em></p>

论文：[*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*](https://arxiv.org/abs/2610.28480)，Qiayuan Liao 等。实现参考 `supermimic_vibed`，上游依赖为 PyPI `unilab`。

## 仓库结构

```text
unisupermimic/
├── src/unisupermimic/
│   ├── rl/
│   │   ├── algos/supermimic/       # 世界模型、策略、critic、replay 和 learner
│   │   └── contracts/supermimic.py # 环境/算法张量协议
│   ├── tasks/motion_tracking/g1/supermimic/
│   │   ├── env.py                  # Manager-Based 环境与后端工厂
│   │   ├── manager_terms.py        # command/action/observation/event/reward terms
│   │   ├── motion.py, mdp.py       # 运动库与可微 MDP
│   │   ├── evaluate.py             # 首帧确定性全 clip 评估
│   │   ├── render.py               # 策略采样与参考动作对比视频
│   │   └── registry.py             # G1SuperMimic 注册
│   ├── scripts/train_supermimic.py # 训练、续训和 eval 入口
│   ├── conf/supermimic/            # Hydra 配置
│   ├── assets/robots/g1/           # G1 MJCF；mesh/URDF 下载后不纳入 git
│   ├── assets/motions/             # LAFAN1 CSV，下载后不纳入 git
│   └── sim2sim.py                  # 策略 I/O、网络形状和资产身份校验
├── docs/assets/                    # 结果曲线、MP4 和 GIF
├── tools/prepare_supermimic_assets.py
├── tests/
├── pyproject.toml, uv.lock
├── LICENSE, LICENSES/, NOTICE
└── Makefile
```

## 安装

环境：Linux、Python 3.10–3.13、[uv](https://docs.astral.sh/uv/)；mjwarp 训练需要 CUDA GPU。以下命令均在仓库根目录执行，`uv run` 按 `uv.lock` 同步项目、PyPI `unilab`、三种仿真后端和开发依赖。

```bash
uv run unisupermimic-train --help
```

依赖变更后运行 `uv lock`。

AMD ROCm 主机改用预生成的 ROCm lockfile 安装（同样的依赖钉版，torch 为
ROCm 构建）：

```bash
make sync-rocm
```

CUDA-only 后端（mjwarp）在 ROCm 上不可用，请使用 MuJoCo host-bridge 后端
训练（`task=g1_super_mimic/mujoco`）。`make sync-rocm` 会把
`pyproject.toml`/`uv.lock` 换成 ROCm 配置档，切回默认配置档用
`make sync-default`。维护者在依赖变更后用 `make lock-rocm` 重新生成
ROCm lockfile。

### 准备资产

```bash
# 从 Hugging Face 下载 LAFAN1/G1 数据，生成并校验 MJCF
uv run tools/prepare_supermimic_assets.py

# 已有资产时，只做离线校验
uv run tools/prepare_supermimic_assets.py --validate-only
```

资产来源为 `lvhaidong/LAFAN1_Retargeting_Dataset`。完整资产应包含 40 段 CSV、264705 个源帧，每帧 36 列。mesh、URDF、CSV、日志和 checkpoint 均被 `.gitignore` 排除。

## 训练

以下命令从仓库根目录运行；使用 CUDA 时，把 `CUDA_VISIBLE_DEVICES` 改为实际空闲 GPU。

```bash
# 默认：mjwarp、4096 个并行环境、12000 iterations
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.log_dir=runs/supermimic_mjwarp_001

# 从同一 run 的 last.pt 续训到 16000 iterations
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  algo.resume=true \
  algo.load_run=runs/supermimic_mjwarp_001 \
  algo.checkpoint=last.pt \
  algo.iterations=16000

# 通过 task owner 选择 MuJoCo
uv run unisupermimic-train \
  task=g1_super_mimic/mujoco \
  training.log_dir=runs/supermimic_mujoco_001

# 通过 task owner 选择 Motrix
uv run unisupermimic-train \
  task=g1_super_mimic/motrix \
  training.log_dir=runs/supermimic_motrix_001
```

续训或评估 MuJoCo、Motrix run 时，使用与训练相同的 `task=g1_super_mimic/<backend>` 覆盖。Motrix 使用固定摩擦，保留质心和关节偏置域随机化。其他配置覆盖使用 Hydra 语法，例如快速 smoke 可覆盖 `algo.num_envs`、`algo.iterations`、`algo.wm_hidden`、`algo.policy_hidden` 和 `algo.compile=false`。默认完整配置见 `src/unisupermimic/conf/supermimic/config.yaml`。

## 评估

评估为每段运动创建一个环境，从首帧开始，关闭观测/动作噪声，跑完整个 clip。严格 sim2sim 校验会在环境构造前检查策略 I/O、网络形状和 MJCF 身份。评估时可以换用其他运动库（`--motion_pack`），只有运动库允许不同。

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train \
  training.play_only=true \
  algo.load_run=runs/supermimic_mjwarp_001 \
  algo.checkpoint=12000
```

结果写入同目录的 `ckpt_012000_eval.json`。也可直接评估指定 checkpoint。独立评估和视频导出入口支持 `--backend mjwarp`、`--backend mujoco`、`--backend motrix`，默认为 mjwarp：

```bash
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt
```

## 视频导出

以下命令从 checkpoint 连续采样 `jumps1_subject1`，截取第 6–14 秒的跳跃片段，生成 25 fps 的参考动作对比 MP4：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run unisupermimic-video \
  --backend mjwarp \
  --ckpt runs/supermimic_mjwarp_001/ckpt_012000.pt \
  --clip jumps1_subject1 \
  --start 6 \
  --seconds 8 \
  --output docs/assets/supermimic_comparison.mp4
```

默认将参考动作的 root XY 对齐策略，便于观察全身跟踪；传入 `--global-ghost` 可保留参考动作的全局位移。

可同时传入多个 checkpoint，并排渲染：每个 checkpoint 一个画面，显示实时得分，摔倒后出现红色横幅。传入多个片段时，每个片段输出一个 `<output 文件名>_<片段>.mp4`。`--push_speed` 可回放推力测试（需要 mjwarp）。回放直接使用确定性评测的循环。如不想占用 GPU，可用 CPU 上的 MuJoCo 后端渲染。没有 EGL 驱动的机器，首次执行 `tools/softgl.sh` 时会自动安装 Mesa 软件 OpenGL：

```bash
source tools/softgl.sh
CUDA_VISIBLE_DEVICES= uv run unisupermimic-video --backend mujoco \
  --ckpt runs/uni_base_s0/ckpt_012000.pt runs/uni_push_s0/ckpt_012000.pt \
  --labels baseline "push injection" \
  --clip dance2_subject4 fallAndGetUp1_subject1 --seconds 10 \
  --output videos/push_ab.mp4
```

## 扩展（`smimic-ext` 分支）

### 推力注入与推力测试

训练时加 `env.commands.motion.push_scale=1.0`，仿真器每 1–3 秒给 root 一次速度冲量。范围取 BeyondMimic 的设置：x/y ±0.5 m/s，z ±0.2 m/s。只支持 mjwarp，注入是精确的。每次冲量都记录在 replay 中，世界模型训练窗口会在同一步把它加回去。再加 `algo.imag_push_rate=0.01 algo.imag_push_scale=1.0`，想象 rollout 中也会施加冲量；奖励按施加冲量前的状态计算。

推力测试每隔 `--push_every` 步（默认 150 步，即 3 秒）施加一次水平冲量。随机种子固定，所有策略受到的冲量完全相同。结果写入 `ckpt_*_eval_push<v>.json`：

```bash
uv run unisupermimic-eval --ckpt runs/uni_push_s0/ckpt_012000.pt --push_speed 1.0
```

### PPO-Sim 基线（论文图 4a）

在仿真器中运行 BeyondMimic 的 PPO，观测、目标和 critic 输入都与 SuperMimic 相同。配置见 `config.yaml` 的 `ppo:` 一节：每 GPU 8192 个环境，训练 18 小时，每 30 分钟保存一次 checkpoint。保存的 checkpoint 可直接用于 `unisupermimic-eval` 和 `unisupermimic-video`。

```bash
uv run python -m unisupermimic.scripts.train_ppo training.log_dir=runs/ppo_g1
```

### 多 GPU 训练

数据并行，每块 GPU 一个 rank：
- 梯度和归一化统计量在各 rank 间取平均；
- 学习率乘以 √N（表 A.5）；
- 每个 rank 的环境使用不同的种子；
- 只有 rank 0 写日志和保存 checkpoint。

```bash
uv run torchrun --standalone --nproc_per_node=2 -m unisupermimic.scripts.train_ddp \
  training.log_dir=runs/g2        # PPO-Sim 在参数前加 --ppo
```

### BONES-SEED 动作库

先在 Hugging Face 上接受数据集许可（`bones-studio/seed`），并 `export HF_TOKEN=...`。然后依次下载数据、检查自动识别的数据格式、生成 50 Hz 的打包动作库。生成时会按附录 E 应用名称过滤和脚部接触可行性过滤：

```bash
uv run tools/prepare_bones_seed.py download --root data/bones_seed     # 约 23.5 GB
uv run tools/prepare_bones_seed.py inspect  --root data/bones_seed
uv run tools/prepare_bones_seed.py build    --root data/bones_seed --out data/bones_pack
uv run unisupermimic-train env.commands.motion.motion_pack=data/bones_pack training.log_dir=runs/bones
uv run unisupermimic-eval --ckpt runs/bones/ckpt_012000.pt --motion_pack data/bones_pack
```

评测按每块 `eval.max_envs` 个片段分块进行（默认 8192）。不加 `--motion_pack` 时在 LAFAN 上评测。评测时可以更换动作库，但训练配置中的其他部分都不能改。

### 来自 go2-supermimic 的训练技巧

从 [go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic)（Go2 追逐与躲避）移植了五项与任务无关的技巧，默认全部关闭。开关都在 `algo.*` 下，说明见 `SuperMimicConfig`：

| 技巧 | 开关 | 作用 |
|---|---|---|
| 世界模型稀有事件采样 | `wm_prioritized_frac`、`wm_archive_frac` | 每批世界模型训练数据中，一部分取自 4 倍候选窗口里最稀有的（脚部接触切换、腾空、摔倒），另一部分取自稀有窗口存档（腾空、剧烈的竖直方向运动、摔倒），存档保留的时间比 replay 更长。接触判断用任务提供的 `foot_heights`：G1 上踝关节 roll link 低于 `contact_height`（0.10 m）算接触 |
| 时间相关的探索噪声 | `explore_frac` | 一部分环境改用 Ornstein-Uhlenbeck 动作噪声（`explore_noise`、`explore_theta`）采集数据，而不是白噪声 |
| 稳定性 | `actor_lr_scale`、`actor_ramp_iters`、`collapse_fall_rate` | actor 学习率倍数；actor 每次开始或重新开始更新时，学习率线性爬升；崩溃保护：连续两个日志窗口的摔倒率（每 1000 个环境步）超过阈值时，把 actor 和 critic 恢复到上一个快照，学习率减半 |
| 热启动 | `init_from`、`reset_critic` | 从另一个实验的 checkpoint 初始化。如果策略的观测新增了项，新输入的权重从零开始，初始行为与原策略完全一致；新建的 critic 先单独训练 `critic_warmup_iters` 次迭代 |
| 数据采集 driver | `driver_ckpt`、`driver_frac`、`driver_fade` | 用另一个策略（SuperMimic 或 PPO-Sim 的 checkpoint）控制一部分环境，让世界模型见到它的动作；这部分比例在 `driver_fade` 给出的两个迭代数之间逐渐降到零 |

```bash
uv run tools/launch.py run go2 --gpus 0,1,2,3     # rare / explore / stable 各自单独，以及三者一起
uv run unisupermimic-train training.log_dir=runs/driven \
  algo.driver_ckpt=runs/uni_ppo_g1/ckpt_XXXXXX.pt algo.driver_frac=0.25 algo.driver_fade=[2000,6000]
```

### Go2 追逐与躲避（go2-supermimic 移植到 UniLab）

[go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) 训练 Unitree Go2 追逐逃跑的球，同时躲避扔来的球和横杆。它有自己的 SuperMimic 变体：世界模型状态是 yaw 坐标系下的 70 维向量，包含腿部各点位置；想象 rollout 中用解析的追逐 / 躲避奖励。算法和目标动力学原样拷贝在 `unisupermimic.go2chase` 下（MIT 许可），环境为 UniLab 重写（`go2chase/env`，任务名 `Go2ChaseFlat`，支持 mjwarp 和 mujoco 后端）。目前只移植了平地：静态障碍物（mocap 方块、LiDAR）和生成地形（高度扫描、地形 driver）还没有移植。

移植结果已与 go2-supermimic 自己的 mjlab 环境对照验证：
- 预训练策略 `weights/flat.pt` 在两边得分一致。追逐加躲避模式，256 个环境、60 秒：
  - 每分钟追到 33.5 次，原版 32.5 次；
  - 球的命中率 0.21–0.25，原版 0.23–0.27；
  - 每分钟摔倒 0.004 次，原版 0.008 次。
- 从世界模型状态重建的 actor 观测与环境给出的完全相同。

执行器用 MuJoCo 位置执行器，增益和力矩上限与 go2 相同；mjlab 的直流电机力矩-速度曲线和 0–2 个物理子步的指令延迟没有建模（在 mjlab 里去掉这两项，指标不变）。

```bash
uv run tools/prepare_go2_assets.py                 # 下载 Menagerie 的 Go2 网格（不进 git）
uv run unisupermimic-go2-train --run-name flat     # 参数同 go2 的 TrainCfg；输出到 runs/go2chase/<时间戳>_flat
uv run unisupermimic-go2-eval --checkpoint runs/go2chase/<run>/model_40000.pt --mode both
uv run tools/launch.py run go2_robot --gpus 0,1    # go2_flat_s0/s1，4 万次迭代并评测
```

在单块 L4 上，全尺寸训练（4096 个环境）约 1.1 秒一次迭代，显存约 17 GB。

### 实验启动器（H100/H200）

`tools/launch.py` 把命名好的实验按可见 GPU 排队：严格先进先出，每块（或每组）空闲 GPU 跑一个任务。每个任务训练完成后评测最终 checkpoint，包括常规评测和 0.5、1.0、1.5 m/s 三档推力；BONES 实验还会在 BONES 动作库上评测。`tools/report_push.py` 汇总结果表，多个种子时给出均值 ± 标准差。

```bash
uv run tools/launch.py list                       # 列出实验和分组
uv run tools/launch.py run push_ab --gpus 0,1     # uni_base_s0 与 uni_push_s0 并行
uv run tools/launch.py run push_ab --dry          # 只打印命令
uv run tools/launch.py eval push_ab --curve       # 对每个已保存的 checkpoint 做常规评测
uv run tools/report_push.py
```

分组：
- `push_ab`：基线 vs 推力注入；
- `push_ablation`：只有仿真器冲量 / 只有想象冲量；
- `push_seeds`：更多种子；
- `multi_gpu`；
- `ppo`：1/4/8 GPU；
- `bones`；
- `go2`：来自 go2-supermimic 的训练技巧；
- `go2_robot`：Go2 追逐与躲避。

在单块 L4 上，默认的 12k 迭代训练约需 5 小时，四次评测约需 1.4 小时。

## 结果

配置：PyPI `unilab 1.5.0`、mjwarp、4096 个训练环境。评估覆盖 40 段 LAFAN1/G1 运动；`norm_return` 是 clip 宏平均，`frame-weighted` 按帧数加权。

| checkpoint | norm_return | frame-weighted | 失败段数 |
|---|---:|---:|---:|
| ckpt_012000.pt | 0.827227 | 0.829952 | 0 / 40 |

![截至 12000 轮的训练与评估曲线](docs/assets/success_curve.png)

## 测试

```bash
make check       # ruff、mypy、pyright
make test        # 非 slow 测试
make test-all    # 包含真实后端 slow 测试；mjwarp 需要 CUDA
uv run tools/prepare_supermimic_assets.py --validate-only
```

## License

本仓库代码采用 [Apache-2.0](LICENSE)。参考实现的 MIT 许可及原始版权声明完整保存在 [LICENSES/MIT-supermimic.txt](LICENSES/MIT-supermimic.txt)，另见 [NOTICE](NOTICE)。UniLab、UniSim、MuJoCo/mjwarp、MotrixSim 和数据集分别遵循其上游许可；本仓库不重新许可这些外部依赖或下载资产。
