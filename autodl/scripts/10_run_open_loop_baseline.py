#!/usr/bin/env python3
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ticvla.training.evaluate import TICVLATester, TestConfig, _default_test_data_dir


def parse_indices() -> list[int]:
    raw = os.environ.get("TICVLA_OPEN_LOOP_INDICES", "0,1,2")
    indices = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not indices or any(index < 0 for index in indices):
        raise ValueError("TICVLA_OPEN_LOOP_INDICES must contain non-negative integers")
    return indices


def tensor_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def main() -> None:
    seed = int(os.environ.get("TICVLA_OPEN_LOOP_SEED", "1"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_dir = Path(os.environ.get("TICVLA_TEST_DATA_DIR", _default_test_data_dir()))
    output_dir = Path(os.environ.get("TICVLA_TEST_OUTPUT_DIR", "outputs/open_loop_eval"))
    work_root = Path(os.environ.get("PMG_ROOT", "/root/autodl-tmp/ccfb-ticvla"))
    metrics_path = Path(
        os.environ.get(
            "TICVLA_OPEN_LOOP_METRICS",
            str(work_root / "manifests/task6_open_loop_metrics.json"),
        )
    )
    checkpoint = Path(os.environ["TICVLA_CHECKPOINT_PATH"])
    model_path = Path(os.environ["TICVLA_BASE_MODEL_PATH"])
    indices = parse_indices()

    for required in (data_dir, checkpoint, model_path / "config.json"):
        if not required.exists():
            raise FileNotFoundError(f"required open-loop input is missing: {required}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    config = TestConfig(
        model_path=str(model_path),
        ckpt_path=str(checkpoint),
        data_dir=str(data_dir),
        num_test_samples=len(indices),
        save_plots=False,
        plot_dir=str(output_dir),
    )

    started = time.time()
    tester = TICVLATester(config)
    if max(indices) >= len(tester.dataset):
        raise IndexError(
            f"requested sample index {max(indices)} but dataset has {len(tester.dataset)} samples"
        )

    torch.cuda.reset_peak_memory_stats()
    records = []
    failures = []
    for index in indices:
        try:
            result = tester.test_model_inference(index)
            predicted = tensor_array(result["pred_waypoints"])
            ground_truth = tensor_array(result["gt_waypoints"])
            if predicted.ndim != 2 or predicted.shape[1] != 2:
                raise ValueError(f"invalid waypoint shape: {predicted.shape}")
            if not np.isfinite(predicted).all():
                raise ValueError("predicted waypoints contain NaN or infinity")
            if np.allclose(predicted, 0.0):
                raise ValueError("predicted waypoints are all zero")

            compared = min(len(predicted), len(ground_truth))
            if compared == 0:
                raise ValueError("ground-truth or predicted trajectory is empty")
            distances = np.linalg.norm(
                predicted[:compared, :2] - ground_truth[:compared, :2], axis=1
            )
            record = {
                "sample_index": index,
                "sample_path": str(result["sample_path"]),
                "instruction": result["instruction"],
                "response": result["response"],
                "waypoint_shape": list(predicted.shape),
                "predicted_waypoints": predicted.tolist(),
                "ground_truth_waypoints": ground_truth.tolist(),
                "inference_seconds": float(result["inference_time"]),
                "ade_m": float(distances.mean()),
                "fde_m": float(distances[-1]),
                "finite": True,
                "all_zero": False,
            }
            records.append(record)
            (output_dir / f"sample_{index:06d}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            failures.append({"sample_index": index, "error": repr(exc)})

    torch.cuda.synchronize()
    inference_times = [record["inference_seconds"] for record in records]
    metrics = {
        "status": "passed" if len(records) == len(indices) else "failed",
        "seed": seed,
        "selected_indices": indices,
        "requested_samples": len(indices),
        "successful_samples": len(records),
        "failed_samples": failures,
        "dataset_samples": len(tester.dataset),
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint),
        "data_dir": str(data_dir),
        "mean_ade_m": float(np.mean([record["ade_m"] for record in records])) if records else None,
        "mean_fde_m": float(np.mean([record["fde_m"] for record in records])) if records else None,
        "mean_inference_seconds": float(np.mean(inference_times)) if inference_times else None,
        "total_inference_seconds": float(np.sum(inference_times)) if inference_times else 0.0,
        "wall_seconds": time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "predictions_dir": str(output_dir),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if failures or not records:
        raise SystemExit(1)
    if any(not math.isfinite(record["ade_m"]) for record in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
