# RSI + Memory 实验入口

入口：`python scripts/run_rsi_memory_experiment.py --output outputs/rsi_memory_demo`。

该标准库 runner 复现一个完整闭环：episode 开始读取并锁定 `active_nav_rules.json`；从同一 PMG 锚点检索候选；规划并执行；仅依据实际控制期间的进度触发 `NO_PROGRESS`；先 `REPLAN_CURRENT`，再把候选写入 episode 级 Attempt Memory 并 `SWITCH_CANDIDATE`，无安全候选时 `FALLBACK_VLA`。episode 结束后聚合诊断，生成受白名单约束的 RulePatch，通过确定性开发验证器后原子更新长期 Rule Memory。

输入、输出与更新规则：

|阶段|输入|输出/写入|
|---|---|---|
|在线|规则、候选、位姿/进度|动作、`NO_PROGRESS`、Attempt Memory|
|离线提案|当前规则、训练摘要|RulePatch（只改触发阈值/修复预算/排序权重）|
|验证|旧/新规则及开发 episode|是否接受、版本归档|

真实系统接入时，将 `DemoEnvironment.route/progress` 替换为现有 PMG Goal Field、route planner 与控制反馈适配器；不要把 oracle goal 注入在线执行器。运行后查看 `run_report.json`、`active_nav_rules.json` 与终端日志，可复现至少一条停滞后恢复案例。
