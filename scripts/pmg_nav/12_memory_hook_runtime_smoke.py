#!/usr/bin/env python3
"""Exercise the PMG memory-hook runtime without Isaac Sim or TIC-VLA.

This is intentionally a short instrumentation gate.  It proves that one
fixture episode produces stable IDs, preserves an occluded node, applies the
contradiction/reappearance correction, and serializes a typed constraint
program.  It is not a navigation benchmark and never reports SR/SPL.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_MAIN = Path(__file__).resolve().parents[2]
if str(PROJECT_MAIN) not in sys.path:
    sys.path.insert(0, str(PROJECT_MAIN))
from pmg_nav.runtime import PMGEpisodeState


DEFAULT_FIXTURE = PROJECT_MAIN / "outputs/stage2/annotation/memory_hard_split_smoke.jsonl"
DEFAULT_REPORT = PROJECT_MAIN / "outputs/stage2/annotation/memory_hook_runtime_smoke_report.json"
DEFAULT_INSTRUCTION = (
    "Use the anchor_a to anchor_b relation, verify the reappearing target, "
    "then stop 0.5 meters right of the target."
)


def run_smoke(
    fixture: Path,
    episode_id: str,
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    frame_stride: int = 15,
) -> dict:
    os.environ["PMG_MEMORY_HOOK_PATH"] = str(fixture.resolve())
    os.environ["PMG_MEMORY_HOOK_EPISODE"] = episode_id
    os.environ.setdefault("PMG_GRAPH_MODE", "compact")
    os.environ.setdefault("STAGE2_M1_VARIANT", "strict")
    state = PMGEpisodeState(instruction, "M1", frame_stride=frame_stride)
    for frame in (0, frame_stride, frame_stride * 2):
        state.update_from_robot(frame, (0.0, 0.0, 0.0))
    summary = state.summary()
    hook = summary["memory_hook"]
    report = {
        "schema_version": 1,
        "type": "memory_hook_runtime_smoke",
        "fixture_only": True,
        "strict_end_to_end": False,
        "isaac_started": False,
        "model_loaded": False,
        "pmg_runtime_used": True,
        "episode_id": episode_id,
        "instruction": instruction,
        "frame_stride": frame_stride,
        "sampled_frames": [0, frame_stride, frame_stride * 2],
        "memory_hook": hook,
        "graph": {
            "nodes": summary["graph_nodes"],
            "edges": summary["graph_edges"],
            "observations": summary["observations"],
            "stable_ids": sorted(state.graph.objects),
            "target_position_after_reappearance": list(state.graph.objects["target"].position),
        },
        "constraint_program": summary["program_metadata"],
        "claims_not_proven": [
            "Isaac Sim/DynaNav navigation success",
            "TIC-VLA action quality or closed-loop SR/SPL",
            "M0 versus M1 causal performance improvement",
            "detector/RGB-D perception accuracy",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--episode", default="memory_test_003")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_smoke(args.fixture.resolve(), args.episode)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
