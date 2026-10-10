#!/usr/bin/env python3
import os

from huggingface_hub import snapshot_download


def main() -> None:
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '')}")
    print(f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '')}")
    print(f"HF_XET_HIGH_PERFORMANCE={os.environ.get('HF_XET_HIGH_PERFORMANCE', '')}")
    snapshot_download(
        repo_id="OpenGVLab/InternVL3-1B",
        repo_type="model",
        local_dir=os.environ["TICVLA_BASE_MODEL_PATH"],
    )
    snapshot_download(
        repo_id="handsomeYun/TIC-VLA",
        repo_type="dataset",
        local_dir=os.environ["TICVLA_DATA_ROOT"],
        allow_patterns=[
            "TIC-VLA-model.ckpt",
            "DynaNav/DynaNav_json.zip",
        ],
    )


if __name__ == "__main__":
    main()
