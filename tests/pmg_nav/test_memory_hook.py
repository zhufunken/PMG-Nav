from __future__ import annotations

from pathlib import Path

import pytest

from pmg_nav.memory_hook import load_memory_hook
from pmg_nav.runtime import PMGEpisodeState


FIXTURE = Path(__file__).resolve().parents[2] / "outputs/stage2/annotation/memory_hard_split_smoke.jsonl"
INSTRUCTION = (
    "Use the anchor_a to anchor_b relation, verify the reappearing target, "
    "then stop 0.5 meters right of the target."
)


def test_hook_selects_history_event_and_keeps_stable_ids() -> None:
    episode = load_memory_hook(FIXTURE, episode_id="memory_test_003")
    first, first_meta = episode.visible_items((0.0, 0.0, 0.0), 0, frame_stride=15)
    hidden, hidden_meta = episode.visible_items((0.0, 0.0, 0.0), 15, frame_stride=15)
    final, final_meta = episode.visible_items((0.0, 0.0, 0.0), 30, frame_stride=15)

    assert first_meta.frame == 0
    assert hidden_meta.frame == 1
    assert final_meta.frame == 2
    assert {item["node_id"] for item in first} == {"anchor_a", "anchor_b", "target"}
    assert {item["node_id"] for item in hidden} == {"anchor_a", "target"}
    assert {item["node_id"] for item in final} == {
        "anchor_b",
        "target",
        "target_distractor_1",
        "target_distractor_2",
    }
    target = next(item for item in final if item["node_id"] == "target")
    assert target["position"] == (8.0, 2.0, 0.0)
    assert episode.provenance()["row_digest"]


def test_runtime_hook_preserves_occluded_nodes_and_corrects_reappearance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMG_MEMORY_HOOK_PATH", str(FIXTURE))
    monkeypatch.setenv("PMG_MEMORY_HOOK_EPISODE", "memory_test_003")
    monkeypatch.setenv("STAGE2_M1_VARIANT", "strict")
    monkeypatch.setenv("PMG_GRAPH_MODE", "compact")
    state = PMGEpisodeState(INSTRUCTION, "M1", frame_stride=15)

    state.update_from_robot(0, (0.0, 0.0, 0.0))
    state.update_from_robot(15, (0.0, 0.0, 0.0))
    # anchor_b is occluded in the second event but remains in PMG history.
    assert "anchor_b" in state.graph.objects
    assert state._memory_hook_contradiction_count == 1
    state.update_from_robot(30, (0.0, 0.0, 0.0))

    assert state.graph.objects["target"].position == (8.0, 2.0, 0.0)
    assert state._memory_hook_reappearance_correction_count == 1
    assert not state._memory_hook_pending_corrections
    summary = state.summary()
    assert summary["memory_hook"]["history_required"] is True
    assert summary["memory_hook"]["contradiction_count"] == 1
    assert summary["memory_hook"]["reappearance_correction_count"] == 1
    assert summary["program_metadata"]["source"] == "memory_hook_constraint_program"


def test_hook_rejects_instruction_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMG_MEMORY_HOOK_PATH", str(FIXTURE))
    monkeypatch.setenv("PMG_MEMORY_HOOK_EPISODE", "memory_test_003")
    with pytest.raises(ValueError, match="instruction"):
        PMGEpisodeState("different instruction", "M1")
