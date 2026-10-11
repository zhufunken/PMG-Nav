# 实验运行记录

`experiment_runner.py` 将每次实验或模型服务测速追加到 `experiment_runs.jsonl`。每条记录包含 UTC 起止时间、耗时、成功/失败、失败信息（最多 5 条）和可选成本估算。

```bash
python3 scripts/experiment_runner.py run --name stage2-smoke --retries 1 -- bash scripts/pmg_nav/12_run_m1_only_85.sh
MODEL_BASE_URL=https://proxy.example.com MODEL_SERVICE_NAME=proxy-a bash scripts/benchmark_model_service.sh
```

服务测速只调用 `GET /v1/models`。价格通过 `MODEL_INPUT_PRICE`、`MODEL_OUTPUT_PRICE`（每千 token，人民币）显式传入；没有价格时记录为 `null`。Ryan 的资源为 `pending_confirmation`，确认前不纳入运行脚本或对比。
