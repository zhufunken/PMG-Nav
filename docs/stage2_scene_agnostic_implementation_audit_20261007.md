# Stage 2 场景无关模块实现审计与最小落地方案

更新时间：2026-10-07

这份文档只记录当前代码已经实现的行为、已经修正的问题和还需要实验验收的部分。没有把设计目标写成实验结论，也没有修改已冻结的 `stage2_85_m1_only_compact_20261004` 结果。

## 1. 先给结论

当前最需要解决的问题不是再训练一个模型，而是让 PMG 的控制权限有明确、可审计的边界。

现有代码的问题可以分成两层：

1. 旧版 M1 的 PMG 已经能产生目标点和路线字段，但它不是完整的场景无关控制模块。`simple_oracle` 原先在每次采样时重建图，跨帧历史不会进入下一次决策。
2. 新的 SCGP 原型此前只存在于独立模块，没有接入 `runtime.action_payload()` 和行为 wrapper，因此不会改变机器人行为。只记录 SCGP 日志不能证明 PMG 接管了动作。

本次修改把 SCGP 接入真实控制链路，同时默认关闭：

```text
PMG_SCGP_ENABLED=0   -> 旧 M0/M1 行为路径，不改变冻结结果
PMG_SCGP_ENABLED=1   -> 开启 persistent PMG + SCGP 有限 lease + fail-closed gate
```

新路径仍然不使用 RL adapter，不修改 upstream TIC-VLA。SCGP 只允许 PMG 在满足物理证据的短窗口内接管；任何证据缺失、路线不可验证、进度停滞或 lease 到期都会回到 TIC-VLA。

## 2. 已核实的实际数据流

### 2.1 M1 开启 SCGP 时

```text
Isaac world pose / perception
        |
        v
PMG graph.observe(stable node_id)
        |
        v
constraint program + candidate goal field
        |
        v
route planner -> waypoint / route validity
        |
        v
CandidateEvidence + SceneState
        |
        v
SceneInvariantExecutor
        |
        +-- PMG_CHECKPOINT + finite lease -> SCGP_ALLOW=1
        +-- FALLBACK + reason code    -> SCGP_ALLOW=0
        |
        v
generated TIC-VLA behavior wrapper
        |
        +-- allow=1 且 terminal gate 满足 -> PMG waypoint/goal 控制
        +-- 其他情况                  -> TIC-VLA policy 控制
```

### 2.2 关闭 SCGP 时

`runtime.py` 仍然产生原有 PMG goal/route 字段；行为 wrapper 看到 `TICVLA_PMG_SCGP_ENABLED=0`，不会把 SCGP 字段当作额外控制条件。这是为了保证历史 M1 和 M0 不因新模块导入而变化。

## 3. 本次代码修改

### 3.1 `pmg_nav/runtime.py`

新增：

- `PMG_SCGP_ENABLED` 显式开关；
- `SCGPConfig` 环境配置；
- 候选点到 `CandidateEvidence` 的转换；
- `SceneState` 帧状态构造；
- 同一候选的几何进度：上一帧距离减去当前帧距离；候选身份变化时进度置零；
- SCGP decision trace、reason 计数、最近 20 条 trace；
- `TICVLA_PMG_SCGP_*` payload 字段；
- `persistent_graph_enabled` 状态。

当前字段含义：

| 字段 | 含义 |
|---|---|
| `TICVLA_PMG_SCGP_ENABLED` | 是否启用 SCGP gate |
| `TICVLA_PMG_SCGP_ALLOW` | 本帧是否允许 PMG 控制 |
| `TICVLA_PMG_SCGP_MODE` | `pmg_checkpoint` 或 `fallback` |
| `TICVLA_PMG_SCGP_REASON` | 稳定、路线、进度、lease 等 reason code |
| `TICVLA_PMG_SCGP_LEASE_UNTIL` | 本次控制 lease 的最后有效帧 |
| `TICVLA_PMG_SCGP_STABLE_COUNT` | 候选身份连续稳定帧数 |
| `TICVLA_PMG_SCGP_NO_PROGRESS_TICKS` | 无几何进度的连续计数 |

几何安全证据目前采用以下保守定义：

- `hard_constraints_satisfied`：来自 `GoalCandidate.hard_valid`；
- `reachable`：只有当前 route planner 返回可达且存在 waypoint 时为真；
- `free_endpoint`：候选没有被 planner 标记为 blocking；
- `orientation_feasible`：若候选没有单独提供该字段，当前 MVP 只使用有限 yaw 和候选硬约束，不能宣称完成了完整朝向验证；
- `footprint_clearance_m`：使用 planner 声明的机器人 footprint envelope。它不是路径上逐点测得的最小 clearance，报告中必须这样标注。

### 3.2 `pmg_nav/scgp.py`

修正了两点：

1. `constraint_checkpoint` 在 runtime 中明确映射为 `GoalRole.CHECKPOINT`，不会再因为控制词汇不同而返回 `UNSUPPORTED_GOAL_ROLE`。
2. lease 不再每帧自动续租。同一候选超过 `lease_until_frame` 后返回 `LEASE_EXPIRED`，必须先回到 policy；只有新的候选身份才可以重新申请 lease。

SCGP 的拒绝顺序为：

```text
candidate id
-> frame order
-> frame transform
-> no-progress watchdog
-> confidence / identity / hard constraint
-> route_valid
-> goal role / terminal verification
-> finite lease
```

### 3.3 `autodl/scripts/08_run_dynanav_baseline.sh`

行为 wrapper 现在解析 SCGP payload。启用 SCGP 且 `SCGP_ALLOW=0` 时，`candidate_valid` 被置为 false，原有 terminal/global PMG override 分支不会执行，动作回到 TIC-VLA policy。trace 中新增：

- `pmg_scgp_enabled`；
- `pmg_scgp_allow`；
- `pmg_scgp_reason`；
- `pmg_scgp_lease_until`；
- `pmg_scgp_stable_count`；
- `pmg_scgp_no_progress_ticks`。

没有 SCGP 字段的旧 payload 在 wrapper 中默认按 SCGP disabled 处理，不会因为缺字段阻断旧实验。

### 3.4 `autodl/scripts/20_run_dynanav_isaac6_compat.py`

兼容 runner 会清除、发布全部 SCGP 字段，并在 action trace 汇总中统计：

- `pmg_scgp_allow_count`；
- `pmg_scgp_deny_count`；
- `pmg_scgp_reason_counts`。

这样可以区分“PMG 生成了 payload”和“PMG 获得了真实控制权限”。

## 4. 当前仍然存在的限制

这些限制必须在结果中明确写出，不能用“场景无关”四个字掩盖：

1. 当前 runtime 的 world-frame 假设通过 `FrameTransform("world", "world")` 显式记录，但还没有接入真实相机外参、深度投影误差或多传感器 frame calibration。使用预测 perception 时，必须先提供投影误差和不确定性，否则只能做 oracle/上界实验。
2. `GoalCandidate` 仍以位置为主，候选 identity 由位置量化生成，尚未把真实 object identity 作为独立字段传到 action adapter。新场景中出现两个相似物体时，必须把 identity association 错误单独统计。
3. route planner 当前提供的是 footprint inflation 和 route reachability，不是逐点的最小 clearance 曲线。不能把 `route_valid=true` 写成“全程安全”。
4. final goal 当前要求 `terminal_verified=true`，但 runtime 还没有接入独立的速度、朝向、身份和停止验证器。因此 SCGP MVP 只适合 checkpoint lease；最终终止控制仍应由 policy/evaluator-neutral verifier 另行实现。
5. persistent graph 只有在 `PMG_SCGP_ENABLED=1` 或 `PMG_PERSISTENT_GRAPH=1` 时打开。冻结 M1 仍是原来的重建图路径，不能把两者混称为同一个实验版本。

## 5. 可直接执行的最小实验

不要先跑 85 条。先用 6 条代表 case，每条只跑以下三种：

| 方法 | SCGP | persistent graph | 目的 |
|---|---:|---:|---|
| M0 | 关 | 关 | TIC-VLA 原始控制 |
| M1-base | 关 | 关 | 与冻结 M1 路径一致 |
| M1-SCGP | 开 | 开 | 检验 PMG 是否真实控制且失败会回退 |

### M1-SCGP 最小配置

```bash
export STAGE2_DIAGNOSTIC_MODE=M1
export PMG_METHOD=M1
export STAGE2_M1_VARIANT=simple_oracle
export PMG_PERCEPTION_SOURCE=oracle
export PMG_ACTION_ENABLED=1
export PMG_ROUTE_PLANNER=1
export PMG_SCGP_ENABLED=1
export PMG_PERSISTENT_GRAPH=1
export PMG_SCGP_STABLE_FRAMES=1
export PMG_SCGP_LEASE_FRAMES=8
export PMG_SCGP_MAX_NO_PROGRESS_TICKS=3
export PMG_SCGP_MIN_CONFIDENCE=0.5
```

### 每条 case 的验收条件

必须同时满足：

1. 结果 JSON 有 `pmg.scgp_enabled=true` 和 `pmg.persistent_graph_enabled=true`；
2. action trace 至少出现一次 `pmg_scgp_allow=true` 或明确记录全程 deny 原因；
3. 当 `pmg_scgp_allow=false` 时，所有对应 tick 的 `pmg_override=false`；
4. 出现 `LEASE_EXPIRED` 或 `NO_PROGRESS` 后，后续 tick 的 `pmg_control_source` 回到 `policy`；
5. M0 与 M1-base 的代码、配置、episode、seed、horizon 不变；
6. 不以 SR 单项决定是否成功，至少同时报告 SR、SPL、navigation error、route valid、SCGP allow/deny、reason 分布和 inference latency。

### 6 条结果后的决策门

| 观察结果 | 下一步 |
|---|---|
| M1-SCGP 的 allow 全为 0 | 先修证据链：route、identity、frame 或 confidence，不跑全量 |
| allow 有记录，但 override 全为 0 | 检查 wrapper 的 gate 或环境字段，不宣称 PMG 接管 |
| override 有记录，且 deny 后能回 policy | 才进入 24 条验证 |
| M1-SCGP 相比 M1-base 下降 | 检查错误接管，优先收紧 route/identity，不调模型 |
| physical constraint 提升但 SR 不变 | 保留 constraint、goal pose、correction 指标，SR 作为辅助 |
| 6 条通过但 24 条方差很大 | 做 paired bootstrap/Wilcoxon，再决定是否跑 85 条 |

## 6. 可复现的本地检查

当前已通过：

```bash
python3 -m compileall -q pmg_nav tests/pmg_nav
bash -n autodl/scripts/08_run_dynanav_baseline.sh
python3 -m py_compile autodl/scripts/20_run_dynanav_isaac6_compat.py
```

另外已完成无 Isaac 的 SCGP smoke test：

- oracle candidate + reachable route 能得到 `pmg_checkpoint`；
- lease 在第 8 帧后不自动续租，返回 `LEASE_EXPIRED`；
- `summary()` 记录 `scgp_decision_counts` 和 `persistent_graph_enabled`。

这些是代码级验证，不是 6 条或 85 条的性能结果。性能提升必须在冻结 episode/seed 下由 paired evaluation 重新测量。

## 7. 最终判断标准

当前可以说：SCGP 已从独立原型变成了一个默认关闭、可以真实阻止 PMG override 的控制 gate；persistent graph 也有了明确的 opt-in 运行路径。

当前不能说：SCGP 已经在所有场景上提升了 SR/SPL，也不能说真实视觉 perception 已经通过了 frame、association 和 clearance 验证。

下一步不是继续盲目调参，而是完成上面的 6 条逐帧审计。只有证明“允许时确实接管、拒绝时确实回退、lease 到期不续租、M0/M1-base 不受影响”，才有理由扩大到 24 条和 85 条。
