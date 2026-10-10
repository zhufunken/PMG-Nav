#!/usr/bin/env python3
"""Retry failed full-matrix runs once, then regenerate the final report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/root/autodl-tmp/ccfb-ticvla")
DRIVER = ROOT / "project/main/scripts/pmg_nav/05_run_full_matrix.py"
FINALIZER = ROOT / "project/main/scripts/pmg_nav/06_finalize_full_matrix.py"
PROGRESS = ROOT / "outputs/stage2_full/progress.json"


def main() -> int:
    # Keep all subprocesses rooted in the remote project.  This script is
    # commonly launched by a detached shell whose inherited cwd is /root.
    os.chdir(ROOT)
    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    failed = [run_id for run_id, item in progress.get("runs", {}).items() if item.get("status") == "failed"]
    retry_code = 0
    for run_id in failed:
        parts = run_id.rsplit("_", 2)
        if len(parts) != 3 or not run_id.startswith("stage2_full_"):
            print(f"Skipping unrecognized failed run id: {run_id}", flush=True)
            retry_code = 1
            continue
        method = parts[-2].upper()
        try:
            episode = int(parts[-1][1:])
        except ValueError:
            print(f"Skipping unrecognized failed run id: {run_id}", flush=True)
            retry_code = 1
            continue
        print(f"Retrying {run_id}: method={method} episode={episode}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(DRIVER),
                "--root",
                str(ROOT),
                "--complexity",
                "full",
                "--methods",
                method,
                "--episodes",
                str(episode),
                "--force",
            ],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode != 0:
            retry_code = 1
    finalizer = subprocess.run([sys.executable, str(FINALIZER)], cwd=ROOT, check=False)
    return retry_code or finalizer.returncode


if __name__ == "__main__":
    raise SystemExit(main())
