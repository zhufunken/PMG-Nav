#!/usr/bin/env python3
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    work_root = Path(os.environ.get("PMG_ROOT", "/root/autodl-tmp/ccfb-ticvla"))
    model_root = Path(
        os.environ.get("TICVLA_BASE_MODEL_PATH", work_root / "models/InternVL3-1B")
    )
    data_root = Path(
        os.environ.get("TICVLA_DATA_ROOT", work_root / "data/TIC-VLA")
    )
    checkpoint_link = Path(
        os.environ.get("TICVLA_CHECKPOINT_PATH", work_root / "checkpoints/ticvla.ckpt")
    )

    model_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    checkpoint_link.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="OpenGVLab/InternVL3-1B",
        repo_type="model",
        local_dir=model_root,
    )
    snapshot_download(
        repo_id="handsomeYun/TIC-VLA",
        repo_type="dataset",
        local_dir=data_root,
        allow_patterns=["TIC-VLA-model.ckpt"],
    )

    checkpoint_source = data_root / "TIC-VLA-model.ckpt"
    if not (model_root / "config.json").is_file():
        raise FileNotFoundError("InternVL3-1B config.json is missing")
    if checkpoint_source.stat().st_size != 1_938_501_547:
        raise RuntimeError("TIC-VLA checkpoint has an unexpected byte size")

    checkpoint_link.unlink(missing_ok=True)
    checkpoint_link.symlink_to(checkpoint_source)
    print(f"MODEL_ROOT={model_root}")
    print(f"CHECKPOINT_SOURCE={checkpoint_source}")
    print(f"CHECKPOINT_LINK={checkpoint_link}")


if __name__ == "__main__":
    main()
