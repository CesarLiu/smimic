# H200 Runbook（`smimic-ext` 分支）

从一台新的 H100/H200 服务器开始，按顺序完成：环境 → 冒烟测试 → 第一个实验 push_ab → 出结果 → 后续实验。所有命令都在仓库根目录执行。各功能的说明见 README 的"扩展（`smimic-ext` 分支）"一节。

## 1. 环境

```bash
# uv（没有的话）
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone -b smimic-ext git@github.com:CesarLiu/smimic.git
cd smimic
uv sync                                          # 安装 unilab、三个仿真后端和开发依赖到 .venv

# LAFAN1/G1 资产：从 Hugging Face 下载，生成并校验 MJCF（40 个 CSV）
uv run tools/prepare_supermimic_assets.py
uv run tools/prepare_supermimic_assets.py --validate-only

nvidia-smi                                       # 记下可用的 GPU 编号
```

注意：
- 如果 shell 里已经激活了别的 virtualenv，uv 会警告 `VIRTUAL_ENV ... does not match`。先 `deactivate`。`tools/launch.py` 启动任务时会自动去掉 `VIRTUAL_ENV`。
- 下面直接用 `.venv/bin/...` 和 `uv run ...` 都可以，两者等价。

## 2. 冒烟测试（约 10–15 分钟）

开大实验之前先确认代码、GPU 和多卡通信都正常。

```bash
# 2.1 CPU 单元测试（约 20 秒）
CUDA_VISIBLE_DEVICES= uv run pytest -q tests          # 预期：103 passed, 2 skipped

# 2.2 mjwarp 慢速测试（含推力注入在仿真器里是否精确）
CUDA_VISIBLE_DEVICES=0 uv run pytest -q -m slow tests

# 2.3 小规模训练：单卡 + 推力注入，30 次迭代
SMALL="algo.num_envs=64 algo.buffer_steps=100 algo.wm_batch=256 algo.pi_batch=256 \
  algo.compile=false algo.wm_hidden=[256,256] algo.policy_hidden=[256,128] \
  algo.warmup_steps=16 algo.wm_only_iters=10 algo.iterations=30 algo.ckpt_every=30 algo.log_every=10"
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=disable uv run unisupermimic-train \
  training.log_dir=runs/smoke_push $SMALL \
  env.commands.motion.push_scale=1.0 algo.imag_push_rate=0.05

# 2.4 对冒烟 checkpoint 做常规评测 + 一档推力评测（只验证流程能跑通，分数没有意义）
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval --ckpt runs/smoke_push/ckpt_000030.pt
CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-eval --ckpt runs/smoke_push/ckpt_000030.pt --push_speed 1.0

# 2.5 双卡 DDP（同样的小配置）
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_GL=disable uv run torchrun --standalone --nproc_per_node=2 \
  -m unisupermimic.scripts.train_ddp training.log_dir=runs/smoke_ddp $SMALL

# 2.6 PPO-Sim 小规模（单卡）
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=disable uv run python -m unisupermimic.scripts.train_ppo \
  training.log_dir=runs/smoke_ppo ppo.num_envs=128 ppo.iterations=4 ppo.log_every=2 \
  'algo.policy_hidden=[256,128]'

# 通过后清掉冒烟输出，免得 report_push.py 把它们列进结果表
rm -rf runs/smoke_*
```

每一步都应该以退出码 0 结束；2.3、2.5 结束时 `runs/smoke_*/` 下有 `train_log.csv` 和 `ckpt_000030.pt`，2.4 写出 `ckpt_000030_eval.json` 和 `ckpt_000030_eval_push1.json`。

## 3. 第一个实验：push_ab（Uni 基线 vs Uni + 推力注入）

两个实验，各 12k 次迭代，各占一块 GPU 并行跑。每个任务训练完自动评测最终 checkpoint：常规评测 + 0.5 / 1.0 / 1.5 m/s 三档推力。

```bash
uv run tools/launch.py run push_ab --dry          # 先看一眼要执行的命令
nohup uv run tools/launch.py run push_ab --gpus 0,1 > launch_push_ab.log 2>&1 &
```

- 必须用 `nohup`（或在 `tmux` 里跑）。否则 SSH 断开时启动器和它启动的训练任务会一起被杀掉。
- `--gpus` 写空闲的 GPU 编号。启动器严格先进先出：每块空闲 GPU 跑一个任务，GPU 不够时后面的任务排队。
- `uni_push_s0` 的推力注入 = 仿真器冲量（`push_scale=1.0`，每 1–3 秒一次）+ 想象冲量（`imag_push_rate=0.01`）。

### 看进度

```bash
tail -f launch_push_ab.log                        # [start] / [done] exit=0
tail -f runs/uni_push_s0/launch.log               # 训练和评测的完整输出
tail -2 runs/uni_push_s0/train_log.csv | cut -d, -f1-3   # 迭代次数, 已用小时, 仿真步数
nvidia-smi
```

- 第一次运行时 `torch.compile` 要编译世界模型（日志里有 `Compiling world model`），会多花几分钟，属于正常现象。
- 显存：默认 4096 个环境，单个任务峰值约 16.4 GB（L4 上测得），H200 绰绰有余。
- 耗时：L4 上训练 5.0 小时、四次评测约 1–1.4 小时。H200 预计每个实验 1–1.5 小时（含评测），第一次跑完后按实际 `train_log.csv` 第 2 列更新这里的数字。
- 每 2000 次迭代保存一次 `ckpt_00X000.pt` 和可续训的 `last.pt`。

## 4. 出结果

```bash
uv run tools/report_push.py                       # 每个实验的常规评测和三档推力：回报 / 失败片段数
```

### 参考数字（L4，Uni 默认配置，12k 次迭代，单种子）

| 评测 | 归一化回报 | 失败片段（共 40） |
|---|---:|---:|
| 常规 | 0.829 | 0 |
| 推力 0.5 m/s | 0.781 | 1 |
| 推力 1.0 m/s | 0.698 | 6 |
| 推力 1.5 m/s | 0.632 | 8 |

H200 上的 `uni_base_s0` 应该落在这附近（常规约 0.82–0.83、0 失败）。明显偏低先查环境和资产，不要急着看推力的对比。

### 怎么判断"推力方向有效"

- 评测本身有噪声：同一个 checkpoint 重跑一次评测，平均回报差约 0.001，单个片段最多差约 0.02。
- 单种子下，训练种子带来的差别更大，所以 push_ab 只能看趋势：重点看 1.0 和 1.5 m/s 两档的失败片段数和回报，同时常规评测不应明显下降。
- 趋势向好时，再跑 `push_seeds`（种子 1、2），用 3 个种子的均值 ± 标准差下结论；`report_push.py` 会自动按 `_s<种子>` 合并。

### 渲染对比视频

```bash
# H200 一般自带 NVIDIA EGL：先直接试 MUJOCO_GL=egl
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-video \
  --ckpt runs/uni_base_s0/ckpt_012000.pt runs/uni_push_s0/ckpt_012000.pt \
  --labels baseline "push injection" \
  --clip dance2_subject4 fallAndGetUp1_subject4 --seconds 12 --push_speed 1.0 \
  --output videos/push_ab.mp4               # 每个片段一个 videos/push_ab_<片段>.mp4

# 报 EGL 错误时改用 Mesa 软件渲染（首次运行自动安装到 tools/.softgl，约 500 MB）
source tools/softgl.sh
```

`--push_speed` 需要 mjwarp 后端（默认就是）。不加 `--push_speed` 时也可以用 `--backend mujoco` 在 CPU 上渲染，不占 GPU。

### 拷回本地

```bash
# 在本地执行；checkpoint 不大，评测 JSON 和视频更小
rsync -av --include='*/' --include='*.json' --include='*.csv' --include='*.mp4' \
  --include='ckpt_012000.pt' --exclude='*' h200:smimic/runs/ ./runs_h200/
rsync -av h200:smimic/videos/ ./videos_h200/
```

## 5. 后续实验（按顺序）

`uv run tools/launch.py list` 列出全部实验和分组。可以一次排多个分组，启动器按顺序把它们分配到空闲 GPU：

```bash
nohup uv run tools/launch.py run push_seeds push_ablation --gpus 0,1,2,3 > launch_next.log 2>&1 &
```

| 顺序 | 分组 | 内容 | GPU | 备注 |
|---|---|---|---|---|
| 1 | `push_seeds` | 基线 / 推力注入，种子 1、2（4 个任务） | 每个 1 块 | push_ab 趋势向好时跑 |
| 2 | `push_ablation` | 只有仿真器冲量 / 只有想象冲量 | 每个 1 块 | 看两部分各自的贡献 |
| 3 | `multi_gpu` | 基线 / 推力注入，双卡 DDP | 每个 2 块 | 学习率自动乘 √2 |
| 4 | `bones` | BONES-SEED 动作库，单卡 / 双卡 | 1 块 / 2 块 | 先准备数据，见下 |
| 5 | `ppo` | PPO-Sim，1 / 4 / 8 卡，各 18 小时 | 1 / 4 / 8 块 | 最耗时，最后跑 |

推力方向确认有效后，下一步移植"按世界模型误差选 reset"，届时会在启动器里加新的实验名。

### BONES-SEED 数据准备（只做一次）

```bash
# 先在 https://huggingface.co/datasets/bones-studio/seed 接受许可
export HF_TOKEN=...
uv run tools/prepare_bones_seed.py download --root data/bones_seed       # 约 23.5 GB
uv run tools/prepare_bones_seed.py inspect  --root data/bones_seed       # 检查自动识别的格式是否合理
uv run tools/prepare_bones_seed.py build    --root data/bones_seed --out data/bones_pack
```

- `build` 会打印名称过滤和可行性过滤各去掉了多少片段。论文中分别是 12,553 / 142,220 和 153，差得太多就调整过滤规则。
- 打包后的动作库约 46 GB；每个训练进程把它加载到自己的 GPU 上，H200 的 141 GB 放得下。
- BONES 实验训练完会在 LAFAN（未参与训练）和 BONES 动作库上各评测一次，后者写成 `ckpt_012000_eval_bones_pack.json`。

## 6. 故障排查

| 现象 | 处理 |
|---|---|
| `CUDA out of memory` | 同一块卡上有别的进程：`nvidia-smi` 查看，换 `--gpus`。启动器每块卡只放一个任务 |
| 训练前几分钟没有新日志 | `torch.compile` 首次编译，正常；之后同一台机器会走缓存 |
| `CrossBackendIncompatibleError: SuperMimic sim2sim contract mismatch` | 评测用的配置和训练时不一致（网络大小、MJCF 等）。评测只允许换动作库（`--motion_pack`），其余字段必须与 `run_config.json` 中记录的一致 |
| 两个 DDP 任务同时跑 | `torchrun --standalone` 自动选空闲端口，不会冲突 |
| SSH 断开后任务没了 | 没用 `nohup`/`tmux`。按下一行续跑 |
| 训练中断，续跑 | 从最近的 `last.pt` 续（最多丢 2000 次迭代）：`CUDA_VISIBLE_DEVICES=0 uv run unisupermimic-train algo.resume=true algo.load_run=runs/<实验名> algo.checkpoint=last.pt <该实验的其余参数>`，其余参数用 `launch.py run <实验名> --dry` 查看；训练完后 `uv run tools/launch.py eval <实验名>` 补评测 |
| 只缺评测 | `uv run tools/launch.py eval <实验名或分组> --gpus 0`（`--curve` 评测每个已保存的 checkpoint，只做常规评测） |
| 想重跑同名实验 | 先把 `runs/<实验名>` 改名或删掉；同名会写进同一个目录 |
