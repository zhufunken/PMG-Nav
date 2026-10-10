# Stage 2 场景无关 PMG/SCGP 完整落地方案

版本：2026-10-07  
适用代码：当前活动源码 `main/`；冻结 M1 参考版本：`stage2_85_m1_only_compact_20261004`  
研究主线：TIC-VLA + Physical Memory Graph（PMG）  
本方案不引入 Stage 3 的 RL adapter。

## 1. 方案要解决什么问题

当前 M1 的问题不是“图节点还不够多”，也不是“遇到新场景再训练一次模型”。真正需要解决的是：

1. PMG 候选是否对应了正确的物体和物理目标；
2. 候选是否在当前坐标、footprint 和路线条件下可执行；
3. PMG 是否真的改变了机器人行为，而不是只生成日志或额外 token；
4. PMG 接管失败时，是否能在有限时间内回到 TIC-VLA；
5. 同一套机制能否在 Hospital、Office、Outdoor、Warehouse 等环境中使用，而不读取场景名称或按场景单独调参。

方案名称：**SCGP（Scene-Agnostic Constraint-Geometric Progress Layer，场景无关约束—几何—进度层）**。

SCGP 是 PMG 和执行器之间的确定性控制层，不是新的视觉模型，也不是 RL 策略。它只使用结构化的语言约束、PMG 证据、坐标转换、路线几何和实时进度。

## 2. 当前已知事实和边界

### 2.1 冻结 M1 的已知结果

冻结运行：`stage2_85_m1_only_compact_20261004`，85/85 结果文件有效。当前 M1 使用 `simple_oracle + compact PMG + direct goal conditioning`，不使用 RL adapter，也不是严格 M0/M1 配对实验。

| 场景 | 成功数 | 成功率 |
|---|---:|---:|
| Hospital | 24/25 | 96% |
| Office | 17/25 | 68% |
| Outdoor small | 0/10 | 0% |
| Full warehouse | 3/25 | 12% |

这些数字说明当前实现没有跨场景泛化证据。oracle RGB-D 也不能代表真实视觉模型的泛化。

### 2.2 当前实现中已经修正的部分

- `pmg_nav/contracts.py`：坐标、候选证据、控制引用和角色契约；
- `pmg_nav/scgp.py`：候选稳定性、硬可行性、显式路线验证、有限 lease、无进度回退；
- `pmg_nav/runtime.py`：SCGP 结构化证据和 payload 桥接；`PMG_SCGP_ENABLED=1` 时启用 persistent graph；
- `08_run_dynanav_baseline.sh`：行为 wrapper 真正读取 SCGP allow/deny；拒绝时回到 TIC-VLA；
- `20_run_dynanav_isaac6_compat.py`：发布和统计 SCGP 字段；
- 测试：contracts、SCGP、runtime bridge、lease 到期行为；
- 审计：默认关闭时不改变冻结 M1/M0 路径。

### 2.3 仍不能宣称的内容

- 尚未证明 SCGP 提升 SR、SPL 或其他正式指标；
- 尚未完成 6 条、24 条或 85 条真实 Isaac paired evaluation；
- 当前 route planner 提供的是 footprint envelope 和可达性，不是逐点实测最小 clearance 曲线；
- final goal 仍需要独立的速度、朝向、身份和连续稳定帧验证器；
- 预测感知、深度投影和多传感器坐标标定尚未通过真实视觉实验验证。

## 3. 设计原则

### 3.1 场景无关的定义

场景无关只允许依赖以下观测量：

- 节点和边的数量、置信度、时间戳和不确定度；
- 机器人 footprint、候选距离、路线长度和路线进度；
- 障碍物几何、自由空间和路线可达性；
- 语言编译得到的关系和度量约束。

禁止依赖：

- `scene == hospital` 等场景名分支；
- benchmark goal、success、SPL 或 evaluator trajectory；
- 为某一个 episode 写入固定坐标；
- 看到失败后只修改该场景的阈值；
- Stage 2 中训练 PMG-to-action RL adapter。

### 3.2 控制权限原则

PMG 默认没有永久控制权。它只能申请短 lease：

```text
TIC-VLA policy
    -> PMG candidate/evidence
    -> SCGP allow/deny
    -> PMG checkpoint control（有限帧）或 policy fallback
```

只要证据不完整，SCGP 必须拒绝。拒绝必须是可解释的 reason code，而不是静默地继续使用上一帧 payload。

### 3.3 版本隔离原则

- 冻结 M1 的结果和代码不回写；
- SCGP 通过 `PMG_SCGP_ENABLED=1` 显式开启；
- 默认 `PMG_SCGP_ENABLED=0`；
- M0 不导入 SCGP 控制路径；
- M1-base 和 M1-SCGP 必须使用不同 run id；
- 不允许混合旧 strict-parity、旧 global upper-bound 和新 M1-SCGP 的结果。

## 4. 系统架构

```text
language instruction
        |
        v
ConstraintCompiler
        |
        v
persistent PMG belief graph
  nodes: object / landmark / obstacle / checkpoint
  edges: identity / metric / relation / visibility / reachability
        |
        v
compact constraint closure
        |
        v
PhysicalGoalField
  candidate pose + yaw + role + evidence
        |
        v
ObstacleAwareRoutePlanner
  waypoint + route_valid + footprint envelope
        |
        v
SceneState + CandidateEvidence
        |
        v
SceneInvariantExecutor (SCGP)
  allow / deny + reason + finite lease
        |
        +--> allow: PMG checkpoint / route control
        +--> deny: TIC-VLA policy
```

### 4.1 各模块职责

| 模块 | 输入 | 输出 | 是否学习 |
|---|---|---|---|
| `constraints.py` | instruction | typed physical constraints | 否 |
| `graph.py` | observation + stable node id | PMG objects/relations/evidence | 否 |
| `scene_adapter.py` | graph/route statistics | bounded geometry profile | 否 |
| `goal_field.py` | PMG + constraints | candidate goal poses | 否 |
| `planner.py` | static geometry + robot pose | route/waypoints | 否 |
| `contracts.py` | scene/candidate evidence | validated data contract | 否 |
| `scgp.py` | contracts | control reference/lease/fallback | 否 |
| TIC-VLA | images/instruction/state | policy action | 是 upstream 模型 |
| Stage 3 adapter | PMG + policy/action | learned fusion | 本阶段不启用 |

## 5. PMG 精简结构

### 5.1 节点保留原则

compact graph 不按“离机器人最近”单独截断，而按约束闭包保留：

1. 所有硬约束 anchor；
2. 连接多个 anchor 的最小关系桥；
3. route corridor 内的关键障碍边界；
4. 一个 robot/checkpoint 状态节点；
5. 剩余预算用确定性的覆盖策略选择局部可见节点。

如果节点预算不足以覆盖硬约束闭包，输出 `incomplete=true`，SCGP 拒绝控制。不能为了满足固定节点数而静默删除关键 anchor。

### 5.2 边保留原则

只保留对当前约束或控制有用途的边：

| 边 | 作用 | 是否核心 |
|---|---|---|
| `identity` | 判断两帧是否同一对象 | 是 |
| `metric` | distance/near/far 约束 | 是 |
| `left/right/front/behind/between` | 关系约束 | 是 |
| `reachable` | 目标是否可达 | 是 |
| `clearance` | footprint 与障碍间隙 | 是 |
| `visibility` | 遮挡、重新出现和观测质量 | 重要 |
| `temporal_same_entity` | 跨帧身份一致性 | 重要 |
| `decorative/co-occurrence` | 不影响约束或控制的关系 | 删除 |

### 5.3 不保留的元素

- 不参与约束闭包的重复 filler 节点；
- 只用于可视化、没有证据来源的边；
- 无法关联到 stable identity 的瞬时低置信度节点；
- 只因距离近、但不在候选路线和约束闭包中的对象。

## 6. SCGP 状态机

### 6.1 输入契约

`SceneState`：

- frame id；
- robot position/yaw/radius；
- graph nodes/edges；
- obstacle count；
- candidate count；
- progress delta；
- route valid；
- frame transform round-trip status。

`CandidateEvidence`：

- source；
- confidence；
- identity stable；
- hard constraints satisfied；
- free endpoint；
- reachable；
- orientation feasible；
- blocking/contradiction；
- footprint clearance envelope；
- uncertainty；
- route cost。

### 6.2 允许条件

```text
allow_checkpoint =
    valid_candidate_id
    AND frame_order_ok
    AND frame_transform_ok
    AND no_progress_watchdog_ok
    AND confidence >= threshold
    AND identity_stable
    AND hard_constraints_satisfied
    AND free_endpoint
    AND reachable
    AND orientation_feasible
    AND clearance >= robot_radius * scale
    AND route_valid == true
```

### 6.3 必须回退的条件

| reason code | 触发条件 | 动作 |
|---|---|---|
| `INVALID_CANDIDATE_ID` | 候选 id 为空 | policy |
| `FRAME_ORDER_ERROR` | 帧号倒退或重复 | policy |
| `FRAME_ERROR` | 坐标 round-trip 超限 | policy |
| `IDENTITY_UNSTABLE` | 对象身份未稳定 | policy |
| `LOW_CONFIDENCE` | 置信度低于阈值 | policy |
| `HARD_CONSTRAINT_FAILED` | 硬约束不满足 | policy |
| `ENDPOINT_BLOCKED` | 终点或 footprint 被占用 | policy |
| `ROUTE_UNREACHABLE` | route 未明确为可达 | policy |
| `NO_PROGRESS` | 连续窗口没有几何进度 | policy |
| `LEASE_EXPIRED` | 同一候选 lease 到期 | policy |
| `TERMINAL_NOT_VERIFIED` | final goal 缺少终止验证 | policy |

### 6.4 lease 规则

- 默认 `stable_frames=1`，因为 runtime 已有候选稳定性过滤；后续实验可固定为 2；
- 默认 `lease_frames=8`；
- 同一候选在 lease 到期后不能自动续租；
- 只有新的候选 identity 或重新通过完整证据链，才能重新申请；
- checkpoint lease 不等于 episode success；
- final goal 暂不允许直接终止，直到独立 verifier 完成。

## 7. 当前代码开关和运行配置

### 7.1 冻结路径

```bash
export PMG_SCGP_ENABLED=0
export PMG_PERSISTENT_GRAPH=0
```

这一路径用于重现历史 M0/M1，不改变冻结实验。

### 7.2 新 M1-SCGP MVP

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
export PMG_SCGP_CLEARANCE_SCALE=1.0
export PMG_SCGP_PROGRESS_EPSILON_M=0.02
```

第一轮不要同时调多个阈值。阈值必须写进 run manifest，不能在失败 case 上手工临时修改。

### 7.3 Stage 3 adapter 禁止项

以下开关在 Stage 2 保持关闭：

```bash
export PMG_ENABLE_GOAL_ADAPTER=0
export PMG_ENABLE_VECTOR_ADAPTER=0
```

当前的 direct goal condition 是确定性接口，不是 learned PMG-to-action adapter。

## 8. 实验执行顺序

### 阶段 0：离线接口验收

目标：不启动 Isaac，证明数据契约和状态机正确。

检查：

- contracts 序列化和 frame round-trip；
- route invalid 必须 deny；
- no-progress 必须 deny；
- lease 到期必须 deny；
- 默认关闭时 summary 记录 `scgp_enabled=false`；
- persistent graph 开启时跨观测保留图实例。

验收：全部 smoke test 通过，且没有改变默认路径。

### 阶段 1：6 条代表 case

固定三组：

| 组别 | 方法 | 目的 |
|---|---|---|
| A | M0 | 原始 TIC-VLA baseline |
| B | M1-base | 当前冻结 M1 路径 |
| C | M1-SCGP | persistent PMG + SCGP 控制 |

6 条必须覆盖：

1. Hospital 成功；
2. Office 成功；
3. Office 失败；
4. Outdoor 失败；
5. Warehouse 成功；
6. NE 较小但 success=false 的 case。

每条保存：episode、seed、instruction、start pose、代码 SHA、配置 SHA、action trace、PMG summary、result JSON。

### 阶段 2：单变量消融

按以下顺序，每次只改变一个因素：

| 实验 | 改动 | 回答的问题 |
|---|---|---|
| A0 | M1-base | 当前 PMG 行为参考 |
| A1 | 加 persistent graph | 历史观测是否必要 |
| A2 | 加 checkpoint/final role | 是否把中间目标当终点 |
| A3 | 加 canonical transform gate | 坐标错误是否是根因 |
| A4 | 加 closure compaction | compact 是否丢关键关系 |
| A5 | 加 route progress watchdog | 是否存在路线停滞 |
| A6 | 完整 M1-SCGP | 组合效果 |

### 阶段 3：24 条分层验证

只有 6 条满足以下条件才扩展：

- M1-SCGP 的 allow/deny 字段完整；
- deny 后没有 PMG override；
- lease 到期后恢复 policy；
- 没有未解释的 frame/identity/route 错误；
- M0/M1-base/M1-SCGP 运行条件一致。

24 条必须做 scene-balanced 汇总，不能只报告总体 SR。

### 阶段 4：85 条确认实验

只有 24 条通过后才运行全量。全量运行前冻结：

- 代码 SHA；
- 配置 SHA；
- episode 清单；
- seed；
- horizon；
- rendering resolution；
- route/SCGP 阈值；
- 磁盘和日志保留策略。

## 9. 每条 episode 必须记录的指标

### 9.1 任务结果

- success rate；
- SPL；
- navigation error；
- collision/contact（必须确认 instrumentation 有效）；
- timeout 和失败原因。

### 9.2 PMG/SCGP 结果

- physical constraint satisfaction；
- metric/relation/orientation/reachability satisfaction；
- candidate identity stability；
- candidate confidence 和 uncertainty；
- route valid/reachable；
- footprint clearance envelope；
- route progress delta；
- waypoint index 单调性；
- SCGP allow/deny 次数；
- reason code 分布；
- lease duration；
- policy fallback 次数；
- PMG override fraction；
- PMG control source。

### 9.3 资源和公平性

- inference latency mean/max；
- episode wall time；
- peak GPU memory；
- action count；
- policy input checksum；
- PMG payload checksum；
- RGB/depth frame checksum（严格 paired 时开启）；
- M0 prefix 是否完整。

## 10. 结果判断规则

| 现象 | 结论 | 处理 |
|---|---|---|
| M1-SCGP allow 全为 0 | 证据链未打通 | 查 route、identity、frame、confidence，不调模型 |
| allow 有但 override 全为 0 | wrapper 未消费 gate | 查环境字段和行为分支 |
| override 有但 deny 后仍 override | 控制权限泄漏 | 立即停止全量，修 wrapper |
| M1-SCGP 比 M1-base 下降 | 接管带来损害 | 先定位错误接管和 route stall |
| constraint 提升、SR 不变 | PMG 改善物理指标但未改善任务成功 | 把 constraint/pose/correction 作为主结果，SR 辅助 |
| M1-SCGP 与 M0 几乎无差异 | 任务可能不需要长期 memory 或方差过大 | 做 hard split 和 paired CI |
| oracle PMG 也不优于 M0 | 当前任务不支持 PMG 价值 | 停止扩展当前 Stage 2 方向，改任务或指标 |

显著性不能只看平均值：

- 二元 success 使用 paired bootstrap、McNemar 或 Wilcoxon 适配的逐 episode 差值；
- 连续指标报告 95% paired bootstrap CI；
- 同时报告总体和 scene-balanced macro；
- 不把官方论文单点或非严格历史 M0 当严格配对基线。

## 11. 失败处理和回滚

### 单条失败

1. 保留原始日志、trace、PMG summary 和 result JSON；
2. 记录失败 reason code；
3. 检查代码/config SHA 是否漂移；
4. 只在相同冻结版本和配置下重试；
5. 不从不同版本挑选更好的结果覆盖原结果。

### 立即停止条件

- `SCGP_ALLOW=0` 时仍然出现 PMG override；
- 代码或配置 SHA 漂移；
- action/policy checksum 不满足配对要求；
- collision instrumentation 失效；
- persistent graph 把 evaluator goal 写入图；
- 磁盘剩余空间低于实验预设阈值；
- 结果 JSON 缺少关键 trace 字段。

### 回滚

```bash
export PMG_SCGP_ENABLED=0
export PMG_PERSISTENT_GRAPH=0
```

回滚后运行的是原冻结路径。SCGP 相关文件可以保留在源码包中，不应删除历史结果。

## 12. 代码验收命令

在项目 `main/` 目录执行：

```bash
python3 -m compileall -q pmg_nav tests/pmg_nav
bash -n autodl/scripts/08_run_dynanav_baseline.sh
python3 -m py_compile autodl/scripts/20_run_dynanav_isaac6_compat.py
PYTHONPATH=. python3 tests/pmg_nav/test_contracts.py
PYTHONPATH=. python3 tests/pmg_nav/test_scgp.py
```

如果环境安装了 pytest，再执行：

```bash
PYTHONPATH=. python3 -m pytest -q tests/pmg_nav/test_contracts.py tests/pmg_nav/test_scgp.py tests/pmg_nav/test_runtime_scgp_bridge.py
```

这些是代码和接口验收，不等同于性能提升。

## 13. 论文可用结论与不可用结论

### 目前可以写

SCGP 是一个不按场景名称分支的确定性控制层。它将 PMG 候选转换为带证据的有限控制引用，并在证据不足、路线不可达或进度停滞时回退 TIC-VLA。该接口不需要 Stage 2 的 RL adapter，且通过显式开关保持 baseline 可独立运行。

### 当前不能写

- “所有场景显著提升”；
- “PMG 已解决跨场景泛化”；
- “真实视觉感知已经验证”；
- “PMG 512 维表示已经改善 TIC-VLA action expert”；
- “collision=0 就证明系统安全”；
- “85 条有效结果等于 85 条任务成功”。

## 14. 最终决策门

```text
代码/契约测试通过
        |
        v
6 条：真实接管、拒绝、回退、lease 审计
        |
        +-- 不通过 -> 修证据链，不跑全量
        |
        v
24 条：分层 paired + bootstrap CI
        |
        +-- 不优于 M0 -> 改任务/指标或停止当前方向
        |
        v
85 条：冻结版本确认实验
```

这条路径的核心不是让每个场景都获得一个单独的“优化答案”，而是让 PMG 在任何场景中都必须通过同一组可解释的几何、约束和进度条件，才能影响行为。

## 15. 执行前阻断项与状态转移补充

前文是目标方案；以下内容是当前代码与目标之间必须显式处理的差距。它们未完成前，不能把 SCGP 称为已经验证的安全控制器或跨场景泛化方法。

### 15.1 当前实现边界

- 当前控制模式虽包含 `policy`、`fallback`、`pmg_checkpoint` 和 `pmg_terminal`，但没有完整的 `VERIFY`、`RELEASE`、`RECOVERY`、`COMMIT` 状态转移。由于 runtime 尚没有独立的终止验证器，第一轮只允许评估有限时长的 checkpoint lease，禁止宣称 final goal 已被可靠接管。
- `FrameTransform("world", "world")` 是占位契约。真实相机外参、单位换算、深度投影误差和同步误差必须在真实 perception 实验前接入，或将实验明确标为 oracle 条件。
- 候选身份目前主要由位置量化生成，不能可靠区分相似物体。下一版必须传递 graph `node_id`，并单独统计 association error。
- `identity_stable`、`orientation_feasible` 和 `footprint_clearance_m` 不能使用永久绕过、缺字段默认真或机器人半径占位。它们必须分别来自连续身份证据、可验证的朝向约束和路线上的测量/保守最小 clearance。

### 15.2 行为层 lease 的 fail-closed 规则

行为 wrapper 不能只读取 `TICVLA_PMG_SCGP_ALLOW`。启用 SCGP 时，必须同时满足：`allow=1`、当前仿真帧可解析、`lease_until` 可解析、当前帧不超过 `lease_until`。字段缺失、过期、解析异常或回调延迟都必须把 `candidate_valid` 置为假，控制源回到 policy，并记录 `STALE_SCGP_PAYLOAD`。应加入 `now = lease_until + 1` 的单元/集成测试。

### 15.3 明确的状态机

```text
POLICY_SEARCH
  --(candidate node_id 连续 K 帧 + 完整证据 + route_valid + clearance)--> PMG_CHECKPOINT_LEASE
PMG_CHECKPOINT_LEASE
  --(每 tick 帧未过期且 progress > epsilon)--> PMG_CHECKPOINT_LEASE
  --(到达 checkpoint 半径)--> VERIFY_CHECKPOINT
VERIFY_CHECKPOINT
  --(identity/pose/orientation/velocity 连续 N 帧通过)--> POLICY_RELEASE
  --(失败或冲突)--> POLICY_FALLBACK
任一 lease_expired/no_progress/route_invalid/frame_error
  --> POLICY_FALLBACK（下一个有效 tick 强制 policy）
FINAL_COMMIT
  仅允许独立 evaluator-neutral verifier 全部通过；当前 runtime 的 terminal_verified=False，禁止进入
```

### 15.4 逐 tick 有效性断言

每个 episode 必须记录 `sim_frame`、`scgp_enabled`、`allow`、`lease_until`、`candidate_node_id`、候选位姿、`route_valid`、`measured_min_clearance_m`、`identity_stable`、`orientation_feasible`、`terminal_verified`、`control_source`、`override_applied` 和 `reason`。必须满足：

```text
allow=0  => override_applied=0
allow=1  => current_frame <= lease_until 且 route_valid=true
LEASE_EXPIRED/NO_PROGRESS/FRAME_ERROR/IDENTITY_UNSTABLE
  => 下一个有效 tick 的 control_source=policy
```

`allow>0` 不等于 PMG 已影响行为；必须观察 `allow → candidate_valid → pmg_override → control_source=PMG` 的同 tick 因果链。

### 15.5 统计与回滚的预注册要求

6 条只验收控制链路，24 条做分层发现，85 条才计算 paired delta、95% paired bootstrap CI 和 Wilcoxon/置换检验。代码 SHA、配置/env hash、episode 清单及 checksum、seed、horizon、Isaac/Torch/CUDA/GPU、renderer 和 policy checkpoint 必须固化；缺字段、帧不单调、OOM、仿真崩溃或版本漂移的条目标为 invalid，不得挑选最好结果。回滚时停止新队列、保存 trace、关闭 `PMG_SCGP_ENABLED` 做相同 M0 smoke、校验冻结 M0 checksum，再以新的 run id 继续；不删除旧产物。

### 15.6 历史场景启发式的隔离

当前 runtime 仍包含 office/hospital/door/desk/vending 等历史 alias、半径和语义先验；兼容 runner 也可能按数据文件名处理加载分支。这些代码可以作为旧实验兼容层保留，但不能计入“完全场景无关”的核心方法。下一轮必须增加 `PMG_LEGACY_SEMANTIC_PRIORS=0` 消融，并做 leave-one-scene-out 对照，确认 SCGP 只消费通用的 anchor、几何、约束和路线证据。若禁用 legacy prior 后性能显著下降，应把结论写成“当前方法依赖历史先验”，而不是宣称已解决新场景适应。
