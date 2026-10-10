#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch

from ticvla.data.policy_data import TICVLADataset


def main() -> None:
    data_root = Path(os.environ["TICVLA_DATA_ROOT"])
    checkpoint = data_root / "TIC-VLA-model.ckpt"
    json_root = data_root / "DynaNav" / "DynaNav_json"

    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state_dict = payload.get("state_dict", {}) if isinstance(payload, dict) else {}
    if not state_dict:
        raise RuntimeError("checkpoint has no non-empty state_dict")

    dataset = TICVLADataset(
        data_dir=str(json_root),
        max_sequence_length=90,
        action_horizon_steps=30,
    )
    if len(dataset) == 0:
        raise RuntimeError("DynaNav dataset loader found zero samples")

    sample = dataset[0]
    sample_path = Path(dataset.samples[0])
    candidate_paths = list(sample.get("delayed_images", []))
    if sample.get("current_image"):
        candidate_paths.append(sample["current_image"])
    resolved_paths = [
        dataset._remap_image_path(path, sample_path) for path in candidate_paths if path
    ]
    existing_paths = [path for path in resolved_paths if Path(path).is_file()]
    if not existing_paths:
        raise RuntimeError("sample loader resolved no existing image files")

    manifest = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_state_dict_keys": len(state_dict),
        "dataset_path": str(json_root),
        "dataset_samples": len(dataset),
        "sample_path": str(sample_path),
        "resolved_image_paths": len(resolved_paths),
        "existing_image_paths": len(existing_paths),
        "validation": "passed",
    }
    output = Path(os.environ["TICVLA_VALIDATION_MANIFEST"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
