#!/usr/bin/env python3
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


PMG_ROOT = Path(os.environ.get("PMG_ROOT", "/root/autodl-tmp/ccfb-ticvla"))
PROJECT_MAIN = Path(os.environ.get("PROJECT_MAIN", PMG_ROOT / "project/main"))
ISAAC_SIM_ROOT = Path(os.environ.get("ISAAC_SIM_ROOT", PMG_ROOT / "isaacsim"))
MANIFESTS = PMG_ROOT / "manifests"


def load_json(name: str) -> dict:
    return json.loads((MANIFESTS / name).read_text(encoding="utf-8"))


def require_status(name: str) -> dict[str, str]:
    path = MANIFESTS / f"{name}_status.env"
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    if values.get("exit_code") != "0":
        raise RuntimeError(f"{name} did not pass: {values}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_first_existing(paths: list[Path], default: str) -> str:
    for path in paths:
        if path.is_file():
            value = path.read_text(encoding="utf-8", errors="replace").strip()
            if value:
                return value.splitlines()[0]
    return default


def main() -> None:
    run_script = PROJECT_MAIN / "run.sh"
    status_names = [
        "task2",
        "task3",
        "task4_setup",
        "task4_smoke",
        "task5",
        "task6",
        "task7_setup",
        "task7_closed_loop",
        "task8_open_loop",
        "task8_closed_loop",
    ]
    statuses = {name: require_status(name) for name in status_names}

    required_files = [
        MANIFESTS / "ticvla_commit.txt",
        MANIFESTS / "ticvla_pip_freeze.txt",
        MANIFESTS / "stage1_resources.tsv",
        MANIFESTS / "stage1_checksums.sha256",
        MANIFESTS / "task5_resource_validation.json",
        MANIFESTS / "task6_open_loop_metrics.json",
        MANIFESTS / "task7_closed_loop_metrics.json",
        MANIFESTS / "task8_open_loop_metrics.json",
        MANIFESTS / "task8_closed_loop_metrics.json",
    ]
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required Stage 1 artifact is missing: {path}")

    open_loop = load_json("task6_open_loop_metrics.json")
    closed_loop = load_json("task7_closed_loop_metrics.json")
    recheck_open = load_json("task8_open_loop_metrics.json")
    recheck_closed = load_json("task8_closed_loop_metrics.json")
    for label, metrics in (
        ("task6 open-loop", open_loop),
        ("task7 closed-loop", closed_loop),
        ("task8 open-loop", recheck_open),
        ("task8 closed-loop", recheck_closed),
    ):
        if metrics.get("status") != "passed":
            raise RuntimeError(f"{label} metrics did not pass")

    tracked_files = [
        run_script,
        PROJECT_MAIN / "autodl/configs/dynanav_stage1_smoke.yaml",
        PROJECT_MAIN / "autodl/scripts/05_setup_ticvla.sh",
        PROJECT_MAIN / "autodl/scripts/06_smoke_ticvla.sh",
        PROJECT_MAIN / "autodl/scripts/08_run_dynanav_baseline.sh",
        PROJECT_MAIN / "autodl/scripts/09_fetch_stage1_resources.sh",
        PROJECT_MAIN / "autodl/scripts/10_run_open_loop_baseline.py",
        PROJECT_MAIN / "autodl/scripts/11_collect_dynanav_metrics.py",
        PROJECT_MAIN / "autodl/scripts/12_finalize_stage1.py",
        PROJECT_MAIN / "autodl/scripts/13_probe_ticvla_env.py",
        PROJECT_MAIN / "autodl/scripts/14_fetch_stage1_hf.py",
        PROJECT_MAIN / "autodl/scripts/15_validate_stage1_resources.py",
        PROJECT_MAIN / "autodl/scripts/16_archive_dynanav_artifacts.py",
        PROJECT_MAIN / "autodl/scripts/16_configure_autodl_vulkan.sh",
        PROJECT_MAIN / "autodl/scripts/17_probe_nvidia_vulkan.sh",
        PROJECT_MAIN / "autodl/scripts/20_run_dynanav_isaac6_compat.py",
        PROJECT_MAIN / "autodl/scripts/21_robot_navigation_manager_isaac6.py",
        MANIFESTS / "isaac6_pip_freeze.txt",
        MANIFESTS / "stage1_resources.tsv",
        MANIFESTS / "stage1_checksums.sha256",
    ]
    commit = (MANIFESTS / "ticvla_commit.txt").read_text(encoding="utf-8").splitlines()[0]
    isaac_version = read_first_existing(
        [ISAAC_SIM_ROOT / "VERSION", ISAAC_SIM_ROOT / "version.txt"],
        ISAAC_SIM_ROOT.name,
    )
    kit_version = read_first_existing(
        [ISAAC_SIM_ROOT / "kit" / "VERSION", ISAAC_SIM_ROOT / "kit" / "version.txt"],
        os.environ.get("ISAAC_SIM_KIT_VERSION", "not separately emitted"),
    )
    driver_version = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    compatibility_note = os.environ.get(
        "ISAAC_SIM_COMPATIBILITY_NOTE",
        "No compatibility migration was declared.",
    )
    manifest_lines = [
        "Stage 1 reproducibility manifest",
        f"ticvla_commit={commit}",
        f"isaac_sim_root={ISAAC_SIM_ROOT}",
        f"isaac_sim_version={isaac_version}",
        f"kit_version={kit_version}",
        f"nvidia_driver={driver_version}",
        f"compatibility_note={compatibility_note}",
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"open_loop_command=bash {run_script} 6",
        f"closed_loop_command=bash {run_script} 7",
        f"final_recheck_command=bash {run_script} 8",
        "",
        "sha256  path",
    ]
    for path in tracked_files:
        manifest_lines.append(f"{sha256(path)}  {path}")
    reproducibility_path = MANIFESTS / "stage1_reproducibility_manifest.txt"
    reproducibility_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    total_task_seconds = sum(int(values["duration_seconds"]) for values in statuses.values())
    hourly_rate = float(os.environ.get("AUTODL_HOURLY_RATE", "2.88"))
    task_runtime_cost = total_task_seconds / 3600 * hourly_rate
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()

    report = f"""# Stage 1 TIC-VLA Baseline Report

## Conclusion

Stage 1 passed. The locked TIC-VLA baseline completed deterministic open-loop inference, a headless DynaNav closed-loop episode, and one independent open-loop and closed-loop recheck from the persisted checkpoint and config.

## Reproducibility

- TIC-VLA commit: `{commit}`
- GPU: `{gpu_name}`
- NVIDIA driver: `{driver_version}`
- Isaac Sim: `{isaac_version}` at `{ISAAC_SIM_ROOT}`
- Kit: `{kit_version}`
- Compatibility note: {compatibility_note}
- Open-loop command: `bash {run_script} 6`
- Closed-loop command: `bash {run_script} 7`
- Final recheck command: `bash {run_script} 8`
- Resource manifest: `{MANIFESTS / 'stage1_resources.tsv'}`
- Checksums: `{MANIFESTS / 'stage1_checksums.sha256'}`
- Locked scripts/config manifest: `{reproducibility_path}`

## Open-Loop Baseline

- Samples: {open_loop['successful_samples']}/{open_loop['requested_samples']}
- Mean ADE: {open_loop['mean_ade_m']:.4f} m
- Mean FDE: {open_loop['mean_fde_m']:.4f} m
- Mean model inference latency: {open_loop['mean_inference_seconds']:.3f} s
- Peak allocated GPU memory: {open_loop['peak_gpu_memory_gib']:.3f} GiB
- Predictions: `{open_loop['predictions_dir']}`

## Closed-Loop DynaNav Baseline

- Normally terminated episodes: {closed_loop['normal_episodes']}/{closed_loop['total_episodes']}
- Successful episodes: {closed_loop['successful_episodes']}
- Mean simulation FPS: {closed_loop['mean_fps']:.3f}
- Mean VLM latency: {closed_loop['mean_vlm_latency_seconds'] if closed_loop['mean_vlm_latency_seconds'] is not None else 'not emitted by upstream'} s
- Peak sampled GPU memory: {closed_loop['peak_gpu_memory_mib']} MiB
- Aggregate metrics: `{json.dumps(closed_loop['average_metrics'], sort_keys=True)}`
- RGB video: `{closed_loop['artifacts']['video_path']}`
- Robot trajectory: `{closed_loop['artifacts']['trajectory_path']}`

## Final Recheck

- Open-loop: {recheck_open['successful_samples']}/{recheck_open['requested_samples']} sample passed
- Closed-loop: {recheck_closed['normal_episodes']}/{recheck_closed['total_episodes']} episode normally terminated
- Recheck RGB video: `{recheck_closed['artifacts']['video_path']}`
- Recheck robot trajectory: `{recheck_closed['artifacts']['trajectory_path']}`
- Recheck outputs: `{PMG_ROOT / 'outputs/stage1_final_recheck'}`

## Runtime And Cost

- Recorded task wall time: {total_task_seconds} s ({total_task_seconds / 3600:.2f} h)
- Compute-only estimate at CNY {hourly_rate:.2f}/h: CNY {task_runtime_cost:.2f}
- This estimate excludes idle time between commands and persistent data-disk charges; the AutoDL console is the billing authority.

## Known Limits

- The closed-loop gate uses the fixed hospital smoke episode, not the full DynaNav benchmark.
- Stage 1 validates released baseline inference only; it does not start PMG, supervised fine-tuning, or reinforcement learning.
"""
    report_path = PROJECT_MAIN / "stage1_baseline_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"REPRODUCIBILITY_MANIFEST={reproducibility_path}")
    print(f"STAGE1_REPORT={report_path}")
    print(f"RECORDED_TASK_SECONDS={total_task_seconds}")
    print(f"COMPUTE_COST_ESTIMATE_CNY={task_runtime_cost:.2f}")


if __name__ == "__main__":
    main()
