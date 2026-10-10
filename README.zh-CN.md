# PMG-Nav

[English](README.md) | **中文**

面向语言引导机器人导航的物理记忆图（Physical Memory Graph），用于处理距离、方位和关系等物理约束。本仓库包含当前研究原型、实验配置、仿真集成脚本和回归测试。

## 项目状态

本项目基于 TIC-VLA。PMG 维护物体、物理关系、可达性、不确定性和证据；约束编译器与目标场读取这些信息，将语言约束转换为候选机器人目标位姿。SCGP 模块负责判断图证据何时可以影响策略。

当前实现仍在开发中。离线 smoke test 不代表仿真导航性能或跨场景泛化能力。完整配对评估和由强化学习控制的图操作仍在研究中。SCGP 默认关闭，可通过 `PMG_SCGP_ENABLED=1` 启用；旧版运行路径仍包含需要单独消融验证的语义启发式规则。

## 目录结构

- `pmg_nav/`：图、约束、目标场、规划器、运行时和控制门代码。
- `configs/pmg_nav/`：实验配置和 smoke test 配置。
- `scripts/pmg_nav/`：标注、评估、回放、审计和调度脚本。
- `tests/pmg_nav/`：现有回归测试。
- `autodl/`：GPU 环境配置以及 TIC-VLA / Isaac Sim 集成脚本。
- `docs/`：当前设计、限制和实现审计说明。
- `outputs/stage2/annotation/memory_hard_split_smoke.jsonl`：测试所需的小型合成样例，其他生成结果均已排除。

## 本地测试

需要 Python 3.10 或更高版本。在仓库根目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests/pmg_nav
```

这些依赖只覆盖本地回归测试，不包含完整 GPU 仿真环境。Baseline 安装、外部资源和仿真器要求请查看 [AutoDL 配置说明](autodl/README.md)。

截至 2026-10-10，在 Python 3.12 环境中测试结果为：111 项通过、9 项失败。原始工程在相同环境下出现相同的 9 项失败，具体见[验证记录](docs/VALIDATION.md)。当前快照保留了原有行为，尚不是全部测试通过的正式版本。

## 离线 Memory Smoke Test

```sh
python scripts/pmg_nav/11_memory_hard_split_smoke.py --help
python scripts/pmg_nav/11_memory_hard_split_smoke.py
python scripts/pmg_nav/12_memory_hook_runtime_smoke.py --help
```

这些脚本使用确定性合成样例，不会启动 Isaac Sim。仿真运行需要另外下载 TIC-VLA 代码、模型权重、DynaNav 资源以及兼容的 GPU 软件环境。运行 GPU 配置脚本前，请先检查其中的路径和环境变量。

## 范围与许可证

本源代码快照不包含大型数据集、模型权重、仿真资产、生成报告、历史版本和论文草稿。目前尚未选择项目许可证。外部代码、模型和数据集仍受其各自许可证约束。TIC-VLA 项目地址：[ucla-mobility/TIC-VLA](https://github.com/ucla-mobility/TIC-VLA)。
