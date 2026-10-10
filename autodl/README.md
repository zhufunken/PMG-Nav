# AutoDL TIC-VLA Bootstrap

本目录用于检测 AutoDL 实例、安装主 baseline TIC-VLA，并完成 checkpoint、Isaac Sim/DynaNav 和短 PPO smoke test。

GOAT/OVON 脚本仍保留，但它们现在只属于 CLIP-conditioned navigation 辅助对照，不是 VLA+RL 主流程。

## 已验收实例

- Stage 1 最终通过：1×RTX 5090 32GB、25 vCPU、90GB RAM、500GB 数据盘；
- Task 7 峰值显存约 22.5GiB，Task 8 closed-loop 复验约 11.4GiB；
- 该结论只覆盖 Stage 1 inference，不代表 RTX 5090 32GB 足以完成 Stage 2/3
  的所有训练任务。

当前 AutoDL host driver 为 R595.71。Isaac Sim 5.0/Kit 107 在该栈上会在 RTX
renderer 中崩溃；Stage 1 实际使用 Isaac Sim 6.0.1/Kit 110.1.2 compatibility
runner。TIC-VLA commit、InternVL3-1B、checkpoint 和 action 模型不变，但
Replicator Agent 0.x dynamic people 未迁移，不能把该结果描述为原始 Isaac
Sim 5.0 环境的逐字节复现。

## 主流程

所有 repo、模型、checkpoint、数据和 Isaac Sim assets 的获取方法统一以
[`../resource_acquisition_guide.md`](../resource_acquisition_guide.md) 为准。特别注意：

- 当前不要运行 `TICVLA_DOWNLOAD_ASSETS=1`；旧脚本的该选项会下载约 538 GB 全量 TIC-VLA 数据；
- 先按资源手册只获取 checkpoint、DynaNav 子集和 InternVL3-1B；
- 中国境内官方源失败时，按资源手册切换 HF mirror、ModelScope 或显式 GitHub 代理；
- IsaacLab 延迟到 Stage 3 RL，Stage 1 closed-loop 只安装 Isaac Sim。

```bash
cd CCFB_paper/multimodal_robot_grounding_nav_rl/main/autodl

bash scripts/00_probe_instance.sh | tee instance_probe.log

# 只安装代码和训练环境；不自动下载大模型/数据
bash scripts/05_setup_ticvla.sh 2>&1 | tee setup_ticvla.log

# 按 ../resource_acquisition_guide.md 第 5–6 节选择性获取资源后再验证

bash scripts/06_smoke_ticvla.sh 2>&1 | tee smoke_ticvla.log
```

`05_setup_ticvla.sh` 不会默认下载 Isaac Sim。原始 TIC-VLA 使用 Isaac Sim 5.0；
仅在 host driver 与 Kit 107 兼容时使用下面的旧安装入口：

```bash
ISAAC_SIM_ARCHIVE=/root/autodl-tmp/installers/isaac-sim-standalone-5.0.0-linux-x86_64.zip \
ISAAC_SIM_ROOT=/root/autodl-tmp/ccfb-ticvla/isaacsim \
  bash scripts/05_setup_ticvla.sh
```

Stage 1 只需验证 Isaac Sim + DynaNav closed-loop。随后完成 Stage 2 PMG inference；进入 Stage 3 时，再按照 TIC-VLA 官方 `rl/README.md` 安装 IsaacLab 2.2.1。IsaacLab 必须与实例上的 Isaac Sim 路径和 Python 环境对应，第一次应保留完整安装日志。

R595/Blackwell 实例使用当前已验收入口：

```bash
export PMG_ROOT=/root/autodl-tmp/ccfb-ticvla
export PROJECT_MAIN="$PMG_ROOT/project/main"
export ISAAC_SIM_ROOT="$PMG_ROOT/isaacsim-6.0.1"
export TICVLA_ISAAC6_COMPAT=1

bash "$PROJECT_MAIN/run.sh" 7
bash "$PROJECT_MAIN/run.sh" 8
```

运行前必须先执行 `scripts/17_probe_nvidia_vulkan.sh`，并确认枚举到 NVIDIA
GPU 而非 `llvmpipe`。AutoDL 的 headless ICD 配置见
`scripts/16_configure_autodl_vulkan.sh`。

## Open-loop checkpoint 验证

从 TIC-VLA Hugging Face 数据仓库取得公开 checkpoint 后：

```bash
export TICVLA_CHECKPOINT_PATH=/root/autodl-tmp/ccfb-ticvla/checkpoints/ticvla.ckpt
export TICVLA_RUN_OPEN_LOOP=1
bash scripts/06_smoke_ticvla.sh 2>&1 | tee open_loop_ticvla.log
```

## 短 PPO 验证

安装 IsaacLab 2.2.1 并验证环境注册后：

```bash
export ISAACLAB_ROOT=/root/autodl-tmp/ccfb-ticvla/IsaacLab
export TICVLA_BASE_MODEL_PATH=/root/autodl-tmp/ccfb-ticvla/models/InternVL3-1B
export TICVLA_CHECKPOINT_PATH=/root/autodl-tmp/ccfb-ticvla/checkpoints/ticvla.ckpt

TICVLA_RL_ITERATIONS=20 \
TICVLA_RL_ROLLOUT_STEPS=128 \
  bash scripts/07_short_rl_ticvla.sh 2>&1 | tee short_rl_ticvla.log
```

这只是接口和更新验证。官方量级为 400×1024 steps；短测试通过后，先测 `seconds/iteration`，再决定是否继续全量训练。

## 数据目录

```text
/root/autodl-tmp/ccfb-ticvla/
├── src/TIC-VLA/
├── models/InternVL3-1B/
├── data/
├── checkpoints/
├── outputs/
├── isaacsim/
└── src/IsaacLab/               # Stage 3 才获取
```

## 成功标准

1. `torch.cuda.is_available()` 为真且 bf16 matmul 成功；
2. InternVL3-1B 和 TIC-VLA checkpoint 能载入；
3. open-loop eval 能产生 waypoint/action；
4. Isaac Sim 能 headless 启动并加载 DynaNav 场景；
5. language instruction 进入 VLM，action expert 在闭环中产生连续控制；
6. PPO 至少完成一次 update，并保存可重新载入的 checkpoint；
7. 保存显存、wall time、reward、SR/SPL/CR 和 collision 日志。

仅完成安装不算 baseline 跑通。

## 旧辅助对照

```bash
# 仅当需要 GOAT/OVON 的 Habitat 非 VLA 对照时运行
bash scripts/01_setup_ovon.sh
bash scripts/02_smoke_ovon.sh
bash scripts/03_setup_goat.sh
bash scripts/04_smoke_goat.sh
```

不要把 AutoDL 密码、Hugging Face token、Matterport token 或 SSH 私钥写入项目脚本。
