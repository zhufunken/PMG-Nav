# PMG-Nav

**English** | [中文](README.zh-CN.md)

Physical Memory Graphs for language-guided robot navigation with metric and
relational constraints. This repository contains the current research prototype,
experiment configurations, simulator integration scripts, and regression tests.

## Status

The project builds on TIC-VLA. PMG maintains objects, physical relations,
reachability, uncertainty, and evidence; constraint compilation and goal-field
readout turn language constraints into candidate robot goal poses. The SCGP
module gates when graph evidence may influence the policy.

The implementation is under development. Offline smoke tests are not evidence
of simulator navigation performance or cross-scene generalization. Full paired
evaluation and RL-controlled graph operations remain research work in progress.
SCGP is opt-in (`PMG_SCGP_ENABLED=1`); legacy runtime paths contain semantic
heuristics that require separate ablation.

## Layout

- `pmg_nav/`: graph, constraints, goal fields, planner, runtime, and control gates.
- `configs/pmg_nav/`: experiment and smoke-test configurations.
- `scripts/pmg_nav/`: annotation, evaluation, replay, audit, and scheduling tools.
- `tests/pmg_nav/`: existing regression tests.
- `autodl/`: GPU environment setup and TIC-VLA / Isaac Sim integration.
- `docs/`: current design, limitations, and implementation audit.
- `outputs/stage2/annotation/memory_hard_split_smoke.jsonl`: small synthetic
  fixture required by existing tests; other generated outputs are excluded.

## Local Tests

Use Python 3.10 or newer. Run commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests/pmg_nav
```

These dependencies cover local regression tests, not the full GPU simulator
environment. See [AutoDL setup](autodl/README.md) for baseline installation,
external resources, and simulator requirements.

Snapshot validation on 2026-10-10 (Python 3.12): 111 tests passed and 9 failed.
The original source tree produces the same nine failures. See
[validation notes](docs/VALIDATION.md) for the affected tests. This snapshot
preserves existing behavior and is not a fully passing release.

## Offline Memory Smoke Test

```sh
python scripts/pmg_nav/11_memory_hard_split_smoke.py --help
python scripts/pmg_nav/11_memory_hard_split_smoke.py
python scripts/pmg_nav/12_memory_hook_runtime_smoke.py --help
```

These scripts use a deterministic synthetic fixture and do not start Isaac Sim.
Simulator runs require separately downloaded TIC-VLA code, model weights,
DynaNav resources, and compatible GPU software. Review script paths and
environment variables for your machine before running GPU setup scripts.

## Scope and Licensing

Large datasets, model weights, simulation assets, generated reports, historical
snapshots, and manuscript drafts are not included in this source snapshot.
No project license has been selected yet. External code, models, and datasets
remain subject to their respective licenses. TIC-VLA is available at
https://github.com/ucla-mobility/TIC-VLA.
