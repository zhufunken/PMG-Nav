#!/usr/bin/env python3
import json

import torch
import transformers
import ticvla


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    x = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
    y = x @ x
    if tuple(y.shape) != (2048, 2048):
        raise RuntimeError(f"unexpected bf16 output shape: {tuple(y.shape)}")
    result = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "ticvla_import": "ok",
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "bf16_matmul_shape": list(y.shape),
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
