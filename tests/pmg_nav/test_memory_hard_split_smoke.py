from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pmg_nav"
sys.path.insert(0, str(SCRIPT_DIR))

_SCRIPT = SCRIPT_DIR / "11_memory_hard_split_smoke.py"
_SPEC = importlib.util.spec_from_file_location("memory_hard_split_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_fixture = _MODULE.evaluate_fixture
write_fixture = _MODULE.write_fixture


def test_fixture_is_small_deterministic_and_history_dependent(tmp_path: Path) -> None:
    path = tmp_path / "memory_hard_split.jsonl"
    rows = write_fixture(path, episodes=6, seed=20260903)
    report = evaluate_fixture(rows, seed=20260903)

    assert len(rows) == 6
    assert path.stat().st_size < 1_000_000
    assert report["strict_end_to_end"] is False
    assert report["isaac_started"] is False
    assert report["methods"]["M1"]["sr"] > report["methods"]["M0"]["sr"]
    assert report["paired_deltas_vs_m0"]["M3"]["action_checksum_equal_m0"] is True

    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows
