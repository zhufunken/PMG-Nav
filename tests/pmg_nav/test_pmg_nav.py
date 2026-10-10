import json
import importlib.util
from pathlib import Path
import sys
import types

from pmg_nav import ConstraintCompiler, Evidence, HeuristicToolScheduler, PhysicalGoalField, PhysicalMemoryGraph
from pmg_nav.adapter import PMGGoalAdapter
from pmg_nav.adapter512 import PMGAdapter512
from pmg_nav.runtime import PMGEpisodeState
from pmg_nav.goal_field import GoalCandidate
from pmg_nav.control_gate import (
    GLOBAL,
    POLICY_ONLY,
    RECOVERY,
    TERMINAL,
    decide_control_gate,
    normalize_control_policy,
    terminal_hysteresis,
)


def test_compile_and_goal_field() -> None:
    program = ConstraintCompiler().compile("Stop 2 meters right of the vending machine and remain reachable.")
    graph = PhysicalMemoryGraph()
    graph.observe("vending machine", (3.0, 4.0, 0.0), Evidence("oracle"), node_id="vm")
    candidates = PhysicalGoalField().sample_and_rank(graph, program)
    assert candidates[0].hard_valid
    assert candidates[0].position[0] > 3.0
    assert abs(((candidates[0].position[0] - 3.0) ** 2 + (candidates[0].position[1] - 4.0) ** 2) ** 0.5 - 2.0) < 0.4


def test_association_and_round_trip(tmp_path: Path) -> None:
    graph = PhysicalMemoryGraph(association_radius=0.75)
    first = graph.observe("chair", (0.0, 0.0, 0.0), Evidence("frame", confidence=0.8))
    second = graph.observe("chair", (0.2, 0.1, 0.0), Evidence("frame", confidence=0.7))
    graph.observe("chair", (3.0, 0.0, 0.0), Evidence("frame", confidence=0.9))
    assert first.node_id == second.node_id
    assert len(graph.objects) == 2
    graph.recompute_relations()
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = PhysicalMemoryGraph.load(path)
    assert json.loads(path.read_text())["version"] == 1
    assert loaded.anchors() == graph.anchors()


def test_conflict_scheduler() -> None:
    scheduler = HeuristicToolScheduler()
    assert scheduler.decide(0.9, 0.3, 0.8, False).tool == "BACKTRACK_EVIDENCE"
    assert scheduler.decide(0.9, 0.3, 0.0, True).tool == "COMMIT_GOAL"


def test_adapter_contract() -> None:
    adapter = PMGAdapter512()
    graph = PhysicalMemoryGraph()
    assert adapter.encode(graph) == adapter.zero()
    graph.observe("chair", (1.0, 2.0, 0.0), Evidence("oracle"), node_id="chair:1")
    first = adapter.encode(graph, ["near"])
    second = adapter.encode(graph, ["near"])
    assert len(first) == 512
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_structured_goal_adapter_contract() -> None:
    candidate = GoalCandidate((3.0, 4.0, 0.2), 0.5, score=0.91, hard_valid=True)
    condition = PMGGoalAdapter.encode(candidate, (0.0, 0.0, 0.0), stable=True)
    assert condition.delta_world == (3.0, 4.0)
    assert abs(condition.distance_m - 5.0) < 1e-9
    payload = condition.to_environment()
    assert payload["TICVLA_PMG_GOAL_X"] == "3.0"
    assert payload["TICVLA_PMG_HARD_VALID"] == "1"
    assert payload["TICVLA_PMG_STABLE"] == "1"


def test_stage2_uses_direct_goal_condition_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PMG_ENABLE_GOAL_ADAPTER", raising=False)
    state = PMGEpisodeState("Stop 2 meters right of the chair.", "M1")
    state._known_objects = [{
        "node_id": "chair:1",
        "label": "chair",
        "position": (2.0, 0.0, 0.0),
        "distance": 2.0,
    }]
    state._scene_catalog = list(state._known_objects)
    state._preferred_node_ids = {"chair:1"}
    state._ensure_resolvable_program()
    state.observe(0, (0.0, 0.0, 0.0), force=True)
    payload = state.action_payload((0.0, 0.0, 0.0))
    assert payload is not None
    assert payload["TICVLA_PMG_GOAL_X"] == "4.0"
    assert state.summary()["goal_condition_path"] == "direct_goal_condition"
    assert state.summary()["goal_adapter_enabled"] is False
    assert state.summary()["structured_goal_adapter_used"] is False


def test_simple_oracle_lexical_goal_uses_approach_side_and_full_pool(monkeypatch) -> None:
    """A free-form furniture target must not pick the first angular sample.

    The test deliberately keeps the evaluator goal out of the state.  The
    selected pose is determined from the PMG anchor, the start pose, and the
    phrase ``in front of``; it should therefore lie on the robot-facing side
    at the configured physical stand-off.
    """
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    monkeypatch.setenv("PMG_GRAPH_MODE", "compact")
    monkeypatch.setenv("PMG_ROUTE_PLANNER", "0")
    # This unit test isolates lexical/approach ranking.  The production
    # terminal controller enables the stability gate explicitly; bypass it
    # here so the test does not need to synthesize four observations.
    monkeypatch.setenv("PMG_SIMPLE_ORACLE_REQUIRE_STABLE", "0")
    state = PMGEpisodeState(
        "Go straight down the hallway, turn left at the end, then continue forward to enter the office room. Move inside and stop in front of the desk on the right.",
        "M1",
        start_yaw=-270,
    )
    state._start_robot_position = (-3.04, 48.1, 0.01)
    state._last_robot_position = state._start_robot_position
    item = {
        "node_id": "/Root/SM_TableA2_638",
        "label": "sm tablea2 638",
        "position": (-17.437, 56.380, 0.0),
        "distance": 16.6,
    }
    state._known_objects = [item]
    state._scene_catalog = [item]
    state._preferred_node_ids = {item["node_id"]}
    state._ensure_resolvable_program()
    state.graph.observe(item["label"], item["position"], Evidence("oracle"), node_id=item["node_id"])
    state.graph.recompute_relations()
    state.candidates = state.goal_field.sample_and_rank(state.graph, state.program, top_k=96)

    selected = state.recovery_candidate()
    assert selected is not None
    # The selected radial sample is on the start-facing side of the desk and
    # uses the 2.5 m ``in front of`` stand-off prior, rather than the old
    # radius-1 / opposite-side insertion-order sample.
    assert selected.position[0] > item["position"][0]
    assert selected.position[1] < item["position"][1]
    assert abs(((selected.position[0] - item["position"][0]) ** 2 + (selected.position[1] - item["position"][1]) ** 2) ** 0.5 - 2.5) < 0.2
    assert selected.evidence["simple_oracle_approach_alignment"] > 0.95


def test_association_uses_secondary_object_context(monkeypatch) -> None:
    """A desk with a nearby plant should beat an unrelated desk tie."""
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    state = PMGEpisodeState(
        "Stop at the white desk with the plant on it, located on the right side of the wall at the end.",
        "M1",
        start_yaw=90,
    )
    state._start_robot_position = (-15.2, 12.7, 0.01)
    objects = [
        {
            "node_id": "/Root/SM_SecretaryDeskA2_164",
            "label": "sm secretarydeska2 164",
            "position": (-8.35, 47.398, 0.0),
            "distance": 36.35,
        },
        {
            "node_id": "/Root/SM_TableWorkSecurity4",
            "label": "sm tableworksecurity4",
            "position": (-12.5, 49.0, 0.0),
            "distance": 36.4,
        },
        {
            "node_id": "/Root/SM_PlantA",
            "label": "sm planta",
            "position": (-12.6, 49.2, 0.5),
            "distance": 36.5,
        },
    ]
    selected = state._infer_instruction_anchor({item["label"] for item in objects}, objects)
    assert selected == "sm tableworksecurity4"
    ranking = state._last_association_candidates
    table = next(item for item in ranking if item["label"] == "sm tableworksecurity4")
    secretary = next(item for item in ranking if item["label"] == "sm secretarydeska2 164")
    assert table["context_score"] > 0.0
    assert table["context_score"] > secretary["context_score"]


def test_compact_relations_keep_task_core() -> None:
    graph = PhysicalMemoryGraph()
    for index in range(7):
        graph.observe(
            "anchor" if index == 0 else f"distractor-{index}",
            (float(index * 10), 0.0, 0.0),
            Evidence("oracle", frame_id=index),
            node_id=f"node:{index}",
        )
    graph.prune(5, keep_node_ids={"node:0"})
    assert len(graph.objects) == 5
    assert "node:0" in graph.objects
    graph.recompute_relations({"node:0"}, max_edges_per_anchor=2)
    assert len(graph.relations) == 2
    assert all(key.startswith("node:0|") for key in graph.relations)


def test_m3_bypasses_structured_action_adapter(monkeypatch) -> None:
    state = PMGEpisodeState("Stop 2 meters right of the chair.", "M3")
    state._known_objects = [{
        "node_id": "chair:1",
        "label": "chair",
        "position": (2.0, 0.0, 0.0),
        "distance": 2.0,
    }]
    state._ensure_resolvable_program()
    state.observe(0, (0.0, 0.0, 0.0), force=True)
    monkeypatch.setenv("PMG_ACTION_ENABLED", "0")
    state.action_enabled = False
    assert state.action_payload((0.0, 0.0, 0.0)) is None
    assert state.summary()["structured_goal_adapter_used"] is False


def test_m3_parity_builds_m1_payload_without_action_authority(monkeypatch) -> None:
    """M3 parity computes the M1 payload but never exposes it as control."""
    monkeypatch.setenv("STAGE2_DIAGNOSTIC_MODE", "M3")
    monkeypatch.setenv("PMG_METHOD", "M3")
    monkeypatch.setenv("PMG_M3_PARITY_PROFILE", "1")
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    monkeypatch.setenv("PMG_GRAPH_MODE", "compact")
    monkeypatch.setenv("PMG_ROUTE_PLANNER", "1")
    monkeypatch.setenv("PMG_CONTROL_POLICY", "terminal")
    monkeypatch.setenv("PMG_TERMINAL_ENTER_DISTANCE", "2.0")
    monkeypatch.setenv("PMG_ACTION_ENABLED", "0")
    monkeypatch.setenv("PMG_ENABLE_GOAL_ADAPTER", "1")

    state = PMGEpisodeState("Stop near the chair.", "M3")
    state._start_robot_position = (0.0, 0.0, 0.0)
    state._last_robot_position = (0.0, 0.0, 0.0)
    state.graph.observe("chair", (4.0, 0.0, 0.0), Evidence("oracle"), node_id="chair:1")
    state._known_objects = [{
        "node_id": "chair:1",
        "label": "chair",
        "position": (4.0, 0.0, 0.0),
        "distance": 4.0,
    }]
    state.candidates = [GoalCandidate((4.0, 0.0, 0.0), 0.0, score=1.0, hard_valid=True)]
    state.route_planner = ObstacleAwareRoutePlanner(())
    state.route_planner_enabled = True

    far_payload = state.action_payload((0.0, 0.0, 0.0), diagnostic_only=True)
    assert far_payload is not None
    assert far_payload["TICVLA_PMG_DIAGNOSTIC_ONLY"] == "1"
    assert far_payload["TICVLA_PMG_ROUTE_VALID"] == "0"
    assert far_payload["TICVLA_PMG_ROUTE_REASON"] == "deferred_until_terminal"
    assert state._diagnostic_only_payload_calls == 1
    assert state._action_payload_calls == 0
    assert state.summary()["goal_adapter_enabled"] is False

    near_payload = state.action_payload((2.5, 0.0, 0.0), diagnostic_only=True)
    assert near_payload is not None
    assert near_payload["TICVLA_PMG_ROUTE_VALID"] == "1"
    assert state._route_compute_deferred_count == 1
    summary = state.summary()
    assert summary["diagnostic_only_payload"] is True
    assert summary["goal_payload_used"] is False
    assert summary["action_dataflow"] == "simple_oracle_diagnostic_only_bypass"


def test_m3_parity_rejects_non_diagnostic_payload_call(monkeypatch) -> None:
    monkeypatch.setenv("STAGE2_DIAGNOSTIC_MODE", "M3")
    monkeypatch.setenv("PMG_M3_PARITY_PROFILE", "1")
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    monkeypatch.setenv("PMG_ACTION_ENABLED", "0")
    state = PMGEpisodeState("Stop near the chair.", "M3")
    assert state.action_payload((0.0, 0.0, 0.0)) is None
    assert state._action_payload_calls == 0
    assert state._diagnostic_only_payload_calls == 0


def test_generated_behavior_requires_enabled_and_valid_route_flags() -> None:
    """The Isaac behavior must fail closed on an inconsistent route payload."""
    behavior_patch_source = (
        Path(__file__).resolve().parents[2]
        / "autodl"
        / "scripts"
        / "08_run_dynanav_baseline.sh"
    ).read_text(encoding="utf-8")
    assert (
        "if self._pmg_route_enabled and route_valid and wx is not None and wy is not None:"
        in behavior_patch_source
    )
    assert "route_enabled_flag_missing" in behavior_patch_source


def test_trace_summary_counts_only_enabled_routes(tmp_path: Path) -> None:
    """Disabled/deferred behavior ticks are not route outputs."""
    compat_path = (
        Path(__file__).resolve().parents[2]
        / "autodl"
        / "scripts"
        / "20_run_dynanav_isaac6_compat.py"
    )
    module_name = "_pmg_compat_trace_test"
    # The local developer Python does not necessarily ship PyYAML; the
    # compatibility module only needs it later in ``main()``, not for the
    # trace summarizer under test.
    previous_yaml = sys.modules.get("yaml")
    sys.modules["yaml"] = types.SimpleNamespace()
    try:
        spec = importlib.util.spec_from_file_location(module_name, compat_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_yaml is None:
            sys.modules.pop("yaml", None)
        else:
            sys.modules["yaml"] = previous_yaml

    trace_path = tmp_path / "action_trace.jsonl"
    trace_path.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {
                    "kind": "output",
                    "pmg_route_enabled": False,
                    "pmg_route_valid": False,
                    "pmg_route_reason": "deferred_until_terminal",
                    "pmg_override": False,
                    "v_cmd": 0.0,
                    "w_cmd": 0.0,
                },
                {
                    "kind": "output",
                    "pmg_route_enabled": True,
                    "pmg_route_valid": True,
                    "pmg_route_reason": "astar",
                    "pmg_override": True,
                    "pmg_gate_scope": "terminal",
                    "pmg_control_source": "pmg",
                    "v_cmd": 1.0,
                    "w_cmd": 0.1,
                },
                {
                    "kind": "output",
                    "pmg_route_enabled": False,
                    "pmg_route_valid": False,
                    "pmg_route_reason": "no_pmg_payload",
                    "pmg_override": False,
                    "v_cmd": 0.0,
                    "w_cmd": 0.0,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    summary = module.read_action_trace(trace_path)
    assert summary["pmg_route_output_count"] == 1
    assert summary["pmg_route_valid_count"] == 1
    assert summary["pmg_route_deferred_output_count"] == 1


def test_m1_is_stateless_and_m3_corrects_conflict() -> None:
    catalog = [
        {"node_id": "chair:1", "label": "chair", "position": (2.0, 0.0, 0.0), "distance": 2.0},
        {"node_id": "table:1", "label": "table", "position": (4.0, 0.0, 0.0), "distance": 4.0},
    ]
    m1 = PMGEpisodeState("Stop near the chair.", "M1", goal_hint=(2.0, 0.0, 0.0))
    m1._known_objects = catalog
    m1._ensure_resolvable_program()
    m1.observe(0, (0.0, 0.0, 0.0), force=True)
    m1.observe(15, (0.0, 0.0, 0.0), force=True)
    assert all(node.observations == 1 for node in m1.graph.objects.values())

    m3 = PMGEpisodeState("Stop near the chair.", "M3", goal_hint=(2.0, 0.0, 0.0))
    m3._known_objects = catalog
    m3._ensure_resolvable_program()
    for frame in (0, 15, 30, 45):
        m3.observe(frame, (0.0, 0.0, 0.0), force=True)
    assert m3.summary()["corrections"] == 1
    assert any(item["correction_applied"] for item in m3.tool_log)
    assert m3.graph.objects["chair:1"].position == (2.0, 0.0, 0.0)

    # M3 must inject and verify conflict on the language target rather than
    # whichever unrelated object happens to be closest to the robot.
    target_first = [
        {"node_id": "table:1", "label": "table", "position": (1.0, 0.0, 0.0), "distance": 1.0},
        {"node_id": "chair:1", "label": "chair", "position": (4.0, 0.0, 0.0), "distance": 4.0},
    ]
    targeted = PMGEpisodeState("Stop near the chair.", "M3")
    targeted._known_objects = target_first
    targeted._ensure_resolvable_program()
    assert targeted._conflict_target()["node_id"] == "chair:1"


def test_pmg_context_keeps_benchmark_instruction_authoritative() -> None:
    instruction = "Exit the room, turn right, and continue to the hallway end."
    state = PMGEpisodeState(instruction, "M2", goal_hint=(2.0, 0.0, 0.0))
    state._known_objects = [{
        "node_id": "door:1",
        "label": "door",
        "position": (2.0, 0.0, 0.0),
        "distance": 2.0,
    }]
    state._ensure_resolvable_program()
    for frame in (0, 15, 30, 45):
        state.observe(frame, (0.0, 0.0, 0.0), force=True)
    assert state.summary()["candidate_stable"] is True
    assert state.context() == instruction


def test_candidate_hysteresis_survives_one_transient_miss() -> None:
    # ``near`` has a maximum sigmoid score below the production 0.92
    # recovery gate. Use an equivalent metric constraint so this test checks
    # hysteresis rather than depending on a deliberately unreachable score.
    state = PMGEpisodeState("Stop 2 meters right of the chair.", "M3", goal_hint=(2.0, 0.0, 0.0))
    state._known_objects = [{
        "node_id": "chair:1",
        "label": "chair",
        "position": (2.0, 0.0, 0.0),
        "distance": 2.0,
    }]
    state._ensure_resolvable_program()
    for frame in (0, 15, 30, 45):
        state.observe(frame, (0.0, 0.0, 0.0), force=True)
    assert state.recovery_candidate() is not None


def test_simple_oracle_stability_uses_its_dedicated_score_threshold(monkeypatch) -> None:
    """A valid compact-oracle near candidate must be able to latch below 0.92."""
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    monkeypatch.setenv("PMG_SIMPLE_ORACLE_STABILITY_MIN_SCORE", "0.80")
    monkeypatch.setenv("PMG_RECOVERY_MIN_SCORE", "0.92")
    state = PMGEpisodeState("Stop near the chair.", "M1")
    candidate = GoalCandidate((2.0, 0.0, 0.0), 0.0, score=0.8655, hard_valid=True)
    for frame in (0, 15, 30, 45):
        state._update_candidate_stability(frame, candidate)
    assert state._candidate_stable is True
    latched = state.recovery_candidate()
    assert latched is not None
    state._candidate_history.append((60, latched.position, 0.82, True))
    state._candidate_history = state._candidate_history[-4:]
    state._update_candidate_stability(60, GoalCandidate(latched.position, latched.yaw, 0.82, True))
    assert state.recovery_candidate() is not None


def test_pmg_does_not_use_evaluator_goal_as_anchor() -> None:
    state = PMGEpisodeState("Stop near the chair.", "M2", goal_hint=(999.0, 999.0, 0.0))
    state._known_objects = [{
        "node_id": "chair:1",
        "label": "chair",
        "position": (2.0, 0.0, 0.0),
        "distance": 2.0,
    }]
    state._ensure_resolvable_program()
    assert state.program is not None

    # A valid PMG candidate remains usable even when it is far from the
    # evaluator goal; applying that filter would leak benchmark state.
    candidate = GoalCandidate((2.0, 0.0, 0.0), 0.0, score=0.95, hard_valid=True)
    state._candidate_latched = candidate
    state._candidate_stable = True
    state.candidates = [candidate]
    assert state.recovery_candidate() == candidate


def test_unresolved_fallback_cannot_change_control() -> None:
    state = PMGEpisodeState("Proceed to the destination.", "M2")
    state._known_objects = [{
        "node_id": "chair:1",
        "label": "chair",
        "position": (2.0, 0.0, 0.0),
        "distance": 2.0,
    }]
    state.program = None
    state._ensure_resolvable_program()
    assert state._program_fallback_used is True
    candidate = GoalCandidate((2.0, 0.0, 0.0), 0.0, score=0.95, hard_valid=True)
    state._candidate_latched = candidate
    state._candidate_stable = True
    state.candidates = [candidate]
    assert state.recovery_candidate() is None


def test_natural_language_target_matches_local_label_without_goal() -> None:
    state = PMGEpisodeState(
        "Walk straight down the hallway and stop at the red bed located to the right of the doors.",
        "M2",
        goal_hint=(999.0, 999.0, 0.0),
    )
    state._known_objects = [
        {
            "node_id": "bed:1",
            "label": "sm hospitalbed 01b 141",
            "position": (4.0, 0.0, 0.0),
            "distance": 4.0,
        },
        {
            "node_id": "door:1",
            "label": "sm door 01b4",
            "position": (2.0, 0.0, 0.0),
            "distance": 2.0,
        },
    ]
    state._ensure_resolvable_program()
    assert state.program is not None
    assert state.program.metadata.get("source") == "instruction_perception_lexical_match"
    assert state.program.metadata.get("inferred_anchor_label") == "sm hospitalbed 01b 141"
    assert state._program_fallback_used is False


def test_inferred_anchor_is_not_used_for_control_by_default(monkeypatch) -> None:
    state = PMGEpisodeState("Go straight and stop at the vending machine.", "M2")
    state._known_objects = [
        {"node_id": "bottle:1", "label": "sm bottlec10", "position": (3.0, 0.0, 0.0), "distance": 3.0},
        {"node_id": "bottle:2", "label": "sm bottlec10", "position": (3.1, 0.0, 0.0), "distance": 3.1},
    ]
    state.program = ConstraintCompiler().compile("Stop near the chair.")
    state.program = state.program.__class__(
        state.program.instruction,
        state.program.constraints,
        metadata={"source": "instruction_perception_lexical_match"},
    )
    candidate = GoalCandidate((3.0, 0.0, 0.0), 0.0, score=0.95, hard_valid=True)
    state._candidate_latched = candidate
    state._candidate_stable = True
    state.candidates = [candidate]
    monkeypatch.delenv("PMG_RECOVERY_ALLOW_INFERRED", raising=False)
    assert state.recovery_candidate() is None
    monkeypatch.setenv("PMG_RECOVERY_ALLOW_INFERRED", "1")
    assert state.recovery_candidate() == candidate


def test_usd_compound_aliases_and_final_phrase_priority() -> None:
    state = PMGEpisodeState(
        "Move forward toward the staircase, then turn left to enter the hallway. Continue straight ahead and stop in front of the blue hospital bed on the right side of the hallway.",
        "M2",
    )
    labels = {
        "sm staircase 1",
        "sm hospitalbed 01b 141",
        "sm door 01b4",
    }
    assert state._infer_instruction_anchor(labels) == "sm hospitalbed 01b 141"

    state = PMGEpisodeState("Go straight ahead and stop at the red emergency exit door.", "M2")
    assert state._infer_instruction_anchor({"bp panicdoor 347", "sm hospitalbed 01b 141"}) == "bp panicdoor 347"

    state = PMGEpisodeState("Go straight ahead and stop at the red emergency exit door.", "M1")
    objects = [
        {
            "node_id": "/Root/BP_ToiletDoor5",
            "label": "bp toiletdoor5",
            "position": (-1.4, -19.99, 0.0),
            "distance": 8.0,
        },
        {
            "node_id": "/Root/BP_PanicDoor_347",
            "label": "bp panicdoor 347",
            "position": (-6.5, -25.5, 0.0),
            "distance": 14.0,
        },
    ]
    assert state._infer_instruction_anchor(
        {item["label"] for item in objects}, objects
    ) == "bp panicdoor 347"

    state = PMGEpisodeState("Go straight until reach the black and white striped umbrellas.", "M2")
    assert state._infer_instruction_anchor({"umbrella01", "patio umbrella 01 inst"}) == "umbrella01"

    state = PMGEpisodeState(
        "Cross the street straight ahead, then proceed forward to the Northside Branch Library with the circular window at front.",
        "M2",
    )
    assert state._infer_instruction_anchor({"northside branch library", "circular window"}) == "northside branch library"


def test_compound_alias_does_not_match_portable_as_table() -> None:
    """A compact USD name must not win through an arbitrary substring."""
    state = PMGEpisodeState(
        "Go through the hallway and stop at the white desk with the plant on it.",
        "M2",
    )
    labels = {
        "sm boxportablee3 372",
        "sm secretarydesk 01",
    }
    assert state._infer_instruction_anchor(labels) == "sm secretarydesk 01"


def test_standoff_uses_final_stop_clause_instead_of_route_clause() -> None:
    state = PMGEpisodeState(
        "Go straight ahead, passing through the hallway behind the black glass doors, "
        "and then enter the office desk area and stop at the white desk with the plant on it, "
        "located on the right side of the wall at the end.",
        "M1",
    )
    assert state._simple_oracle_target_radius() == 1.25
    front = PMGEpisodeState(
        "Enter the office room and stop in front of the desk.", "M1"
    )
    assert front._simple_oracle_target_radius() == 2.5


def test_plant_context_selects_reception_closet_landmark() -> None:
    """The office USD's plant-side desk landmark uses a compact asset head."""
    state = PMGEpisodeState(
        "Go straight ahead, then stop at the white desk with the plant on it, located at the end.",
        "M1",
        start_yaw=90,
    )
    objects = [
        {
            "node_id": "/Root/SM_ReceptionClosets6",
            "label": "sm receptionclosets6",
            "position": (-11.112, 49.51, 0.0),
            "distance": 36.84,
        },
        {
            "node_id": "/Root/SM_SecretaryDeskA2_164",
            "label": "sm secretarydeska2 164",
            "position": (-8.35, 48.398, 0.0),
            "distance": 36.35,
        },
        {
            "node_id": "/Root/SM_Plant4",
            "label": "sm plant4",
            "position": (-11.453, 50.352, 0.8),
            "distance": 37.75,
        },
    ]
    assert state._infer_instruction_anchor(
        {item["label"] for item in objects}, objects
    ) == "sm receptionclosets6"


def test_route_local_instance_tiebreak_and_explicit_node_anchor() -> None:
    """Opaque USD suffixes must not decide between same-semantic instances."""
    state = PMGEpisodeState(
        "Go straight past the wheelchairs and stop at the water dispenser by the window.",
        "M1",
    )
    objects = [
        {
            "node_id": "/Root/SM_WaterDispenser_01a2",
            "label": "sm waterdispenser 01a2",
            "position": (-34.12, 5.02, 0.0),
            "distance": 43.90,
        },
        {
            "node_id": "/Root/SM_WaterDispenser_01a_15",
            "label": "sm waterdispenser 01a 15",
            "position": (-5.31, -2.36, 0.0),
            "distance": 15.90,
        },
    ]
    assert state._infer_instruction_anchor({item["label"] for item in objects}, objects) == (
        "sm waterdispenser 01a 15"
    )
    state._known_objects = list(objects)
    state._scene_catalog = list(objects)
    state._ensure_resolvable_program()
    assert state.program is not None
    assert state.program.metadata["inferred_anchor_id"] == "/Root/SM_WaterDispenser_01a_15"
    # The actual constraint must use the concrete id, not an ambiguous label.
    assert state.program.constraints[0].anchors == ("/Root/SM_WaterDispenser_01a_15",)
    state.observe(0, (9.75, 2.82, 0.0), force=True)
    assert state.candidates
    assert state.candidates[0].position == (-5.31, -2.36, 0.0)


def test_route_language_selects_red_gurney_at_hallway_end() -> None:
    """A bed request with a down-hallway route must not pick a nearby ward bed.

    The hospital USD uses both ``hospitalbed`` and ``gurney`` authoring
    labels.  The route/side evidence (initial pose and heading only) selects
    the gurney at the narrated hallway end; no evaluator goal is supplied.
    """
    start = (18.7, 29.3, 0.0)
    objects = [
        {
            "node_id": "/Root/SM_HospitalBed_02d12",
            "label": "sm hospitalbed 02d12",
            "position": (14.192, 18.483, 0.0),
            "distance": 11.7,
        },
        {
            "node_id": "/Root/SM_HospitalBed_02d11",
            "label": "sm hospitalbed 02d11",
            "position": (14.192, 14.830, 0.0),
            "distance": 15.1,
        },
        {
            "node_id": "/Root/SM_Gurney_01a5",
            "label": "sm gurney 01a5",
            "position": (15.170, -3.471, 0.0),
            "distance": 33.0,
        },
    ]
    state = PMGEpisodeState(
        "Walk straight down the hallway and stop at the red bed located to the right of the doors.",
        "M1",
        start_yaw=-90,
    )
    state._start_robot_position = start
    state._last_robot_position = start
    assert state._infer_instruction_anchor({item["label"] for item in objects}, objects) == (
        "sm gurney 01a5"
    )
    assert state._last_association_reason == "corridor_alignment_side_right"


def test_route_language_selects_end_room_desk_not_nearest_table() -> None:
    """A turn-left office route resolves ``desk`` to the end-room table."""
    start = (-3.04, 48.1, 0.0)
    objects = [
        {
            "node_id": "/Root/SM_TableB3_325",
            "label": "sm tableb3 325",
            "position": (-7.205, 45.116, 0.0),
            "distance": 5.1,
        },
        {
            "node_id": "/Root/SM_SecretaryDeskA2_164",
            "label": "sm secretarydeska2 164",
            "position": (-8.35, 48.398, 0.0),
            "distance": 5.3,
        },
        {
            "node_id": "/Root/SM_TableA2_638",
            "label": "sm tablea2 638",
            "position": (-17.437, 56.38, 0.0),
            "distance": 16.6,
        },
    ]
    state = PMGEpisodeState(
        "Go straight down the hallway, turn left at the end, then continue forward to enter the office room. Move inside and stop in front of the desk on the right.",
        "M1",
        start_yaw=-270,
    )
    state._start_robot_position = start
    state._last_robot_position = start
    assert state._infer_instruction_anchor({item["label"] for item in objects}, objects) == (
        "sm tablea2 638"
    )
    assert state._last_association_reason == "multi_leg_route_side_right"


def test_explicit_far_end_ranks_route_progress_before_soft_side_score() -> None:
    """Ordinal route intent must outrank a nearby, slightly better-aligned match."""
    objects = [
        {
            "node_id": "/Root/SM_ConcreteBlock_near",
            "label": "sm concreteblock near",
            "position": (10.0, 0.0, 0.0),
            "distance": 10.0,
        },
        {
            "node_id": "/Root/SM_ConcreteBlock_far",
            "label": "sm concreteblock far",
            "position": (20.0, 4.0, 0.0),
            "distance": 20.4,
        },
    ]
    state = PMGEpisodeState(
        "Go straight down the hallway passing the concrete pillars, then stop at the last pillar at the far end.",
        "M1",
        start_yaw=0.0,
    )
    state._start_robot_position = (0.0, 0.0, 0.0)
    state._last_robot_position = state._start_robot_position

    assert state._infer_instruction_anchor(
        {item["label"] for item in objects}, objects
    ) == "sm concreteblock far"
    assert state._last_association_prefer_far is True
    ranking = state._last_association_candidates
    assert ranking[0]["node_id"] == "/Root/SM_ConcreteBlock_far"
    assert ranking[0]["route_progress_m"] > ranking[1]["route_progress_m"]


def test_end_of_route_prefers_far_desk_over_right_side_shortcut() -> None:
    """A far-end modifier is not overridden by a local right-side bonus."""
    objects = [
        {
            "node_id": "/Root/SM_SecretaryDesk_near_right",
            "label": "sm secretarydesk near",
            "position": (-8.35, 48.4, 0.0),
            "distance": 35.9,
        },
        {
            "node_id": "/Root/SM_Table_far_left",
            "label": "sm table far",
            "position": (-12.5, 49.0, 0.0),
            "distance": 36.4,
        },
    ]
    state = PMGEpisodeState(
        "Go through the hallway, then stop at the white desk on the right side of the wall at the end.",
        "M1",
        start_yaw=90.0,
    )
    state._start_robot_position = (-15.2, 12.7, 0.0)
    state._last_robot_position = state._start_robot_position

    assert state._infer_instruction_anchor(
        {item["label"] for item in objects}, objects
    ) == "sm table far"
    assert state._last_association_candidates[0]["route_progress_m"] > (
        state._last_association_candidates[1]["route_progress_m"]
    )


def test_compound_desk_alias_matches_concatenated_usd_name() -> None:
    """``secretarydeska2`` must be recognized as a literal desk label."""
    state = PMGEpisodeState("Stop at the desk.", "M1")
    assert state._infer_instruction_anchor({"sm secretarydeska2 164"}) == (
        "sm secretarydeska2 164"
    )


def test_infrastructure_prim_filter_is_narrow() -> None:
    assert PMGEpisodeState._is_infrastructure_prim("/World/Robot/chassis_link", "chassis_link")
    assert PMGEpisodeState._is_infrastructure_prim("/World/Robot/base_link", "base_link")
    assert PMGEpisodeState._is_infrastructure_prim("/World/Robot/chassis_link/camera", "camera")
    assert not PMGEpisodeState._is_infrastructure_prim("/World/Office/table_01", "table_01")
    assert not PMGEpisodeState._is_infrastructure_prim("/World/Office/base", "base")
    assert not PMGEpisodeState._is_infrastructure_prim("/World/Office/baseball_display", "baseball_display")


def test_structural_catalog_filter_preserves_landmarks() -> None:
    assert PMGEpisodeState._is_structural_label("sm floor 3m 14")
    assert PMGEpisodeState._is_structural_label("sm wall 6m 43")
    assert not PMGEpisodeState._is_structural_label("sm wallselevator 2")
    assert not PMGEpisodeState._is_structural_label("sm doorb door")


def test_catalog_deduplicates_colocated_usd_instances() -> None:
    items = [
        {"node_id": "/Root/BP_Table", "label": "sm table", "position": (1.0, 2.0, 0.0), "distance": 2.0},
        {"node_id": "/Root/BP_Table/InstancedSM_Table", "label": "instancedsm table", "position": (1.03, 2.02, 0.0), "distance": 2.0},
        {"node_id": "/Root/BP_Table_2", "label": "sm table", "position": (4.0, 2.0, 0.0), "distance": 4.0},
    ]
    deduplicated = PMGEpisodeState._deduplicate_catalog(items)
    assert [item["node_id"] for item in deduplicated] == ["/Root/BP_Table", "/Root/BP_Table_2"]


def test_simple_oracle_defaults_to_terminal_handoff() -> None:
    assert normalize_control_policy(None, method="M1", variant="simple_oracle") == TERMINAL
    far = decide_control_gate(
        method="M1",
        variant="simple_oracle",
        candidate_valid=True,
        candidate_distance_m=12.0,
        terminal_active=False,
    )
    assert far.scope == POLICY_ONLY
    assert far.allow_override is False
    assert far.reason == "outside_terminal_radius"
    near = decide_control_gate(
        method="M1",
        variant="simple_oracle",
        candidate_valid=True,
        candidate_distance_m=3.5,
        terminal_active=False,
    )
    assert near.scope == "terminal"
    assert near.allow_override is True
    assert near.reason == "terminal_handoff"
    blocked = decide_control_gate(
        method="M1",
        variant="simple_oracle",
        candidate_valid=True,
        candidate_distance_m=3.5,
        terminal_active=False,
        require_valid_route=True,
        route_valid=False,
    )
    assert blocked.scope == POLICY_ONLY
    assert blocked.allow_override is False
    assert blocked.reason == "route_infeasible_policy_fallback"


def test_global_override_is_explicit_only() -> None:
    global_decision = decide_control_gate(
        method="M1",
        variant="simple_oracle",
        policy="always",
        candidate_valid=True,
        candidate_distance_m=100.0,
    )
    assert global_decision.policy == GLOBAL
    assert global_decision.scope == "global"
    assert global_decision.allow_override is True
    assert global_decision.reason == "explicit_global_override"


def test_terminal_hysteresis_has_separate_enter_and_exit_radii() -> None:
    assert terminal_hysteresis(
        candidate_distance_m=3.9,
        current_active=False,
        enter_distance_m=4.0,
        exit_distance_m=5.0,
    ) is True
    # Once active, remain active through the larger exit radius.
    assert terminal_hysteresis(
        candidate_distance_m=4.8,
        current_active=True,
        enter_distance_m=4.0,
        exit_distance_m=5.0,
    ) is True
    assert terminal_hysteresis(
        candidate_distance_m=5.1,
        current_active=True,
        enter_distance_m=4.0,
        exit_distance_m=5.0,
    ) is False
    assert terminal_hysteresis(
        candidate_distance_m="not-a-number",
        current_active=True,
        enter_distance_m=4.0,
        exit_distance_m=5.0,
    ) is False


def test_recovery_policy_requires_local_trigger() -> None:
    no_trigger = decide_control_gate(
        method="M2",
        policy="recovery",
        candidate_valid=True,
        candidate_distance_m=2.0,
        terminal_active=True,
        recovery_requested=False,
    )
    assert no_trigger.allow_override is False
    assert no_trigger.reason == "terminal_recovery_not_requested"
    triggered = decide_control_gate(
        method="M2",
        policy="recovery",
        candidate_valid=True,
        candidate_distance_m=2.0,
        terminal_active=True,
        recovery_requested=True,
    )
    assert triggered.allow_override is True
    assert triggered.scope == "terminal"


def test_disabled_action_always_fails_closed() -> None:
    decision = decide_control_gate(
        method="M3",
        policy="global",
        action_enabled=False,
        candidate_valid=True,
        candidate_distance_m=0.5,
        terminal_active=True,
    )
    assert decision.allow_override is False
    assert decision.scope == POLICY_ONLY
    assert decision.reason == "method_or_action_disabled"


def test_injected_route_alignment_stops_before_translation() -> None:
    """Keep the generated Isaac behavior's corner guard auditable."""

    script = (
        Path(__file__).resolve().parents[2]
        / "autodl/scripts/08_run_dynanav_baseline.sh"
    ).read_text(encoding="utf-8")
    assert 'PMG_ROUTE_ALIGN_HOLD_RAD", "0.55"' in script
    assert 'PMG_ROUTE_ALIGN_RELEASE_RAD", "0.35"' in script
    assert "if pmg_route_alignment_hold\n                        else float(" in script
    assert '"pmg_route_alignment_hold": bool(pmg_route_alignment_hold)' in script
    assert '"pmg_route_heading_error_rad": pmg_route_heading_error_rad' in script
