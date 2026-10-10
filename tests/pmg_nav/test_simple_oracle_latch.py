"""Offline invariants for the deterministic simple-oracle latch."""

from unittest.mock import Mock

from pmg_nav import ConstraintCompiler, Evidence
from pmg_nav.goal_field import GoalCandidate
from pmg_nav.planner import ObstacleAwareRoutePlanner, RoutePlan
from pmg_nav.runtime import PMGEpisodeState


def _state(monkeypatch):
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    monkeypatch.setenv("PMG_ROUTE_PLANNER", "1")
    monkeypatch.setenv("PMG_CONTROL_POLICY", "global")
    state = PMGEpisodeState("Stop near the chair.", "M1")
    state.program = ConstraintCompiler().compile("Stop near the chair.")
    state._known_objects = [{
        "node_id": "chair:1",
        "label": "chair",
        "position": (5.0, 0.0, 0.0),
        "distance": 5.0,
    }]
    state.graph.observe("chair", (5.0, 0.0, 0.0), Evidence("oracle"), node_id="chair:1")
    state.candidates = [
        GoalCandidate((4.0, 0.0, 0.0), 0.0, score=0.9, hard_valid=True),
        GoalCandidate((5.0, 1.0, 0.0), 0.0, score=0.8, hard_valid=True),
    ]
    planner = ObstacleAwareRoutePlanner([])
    planner.plan = Mock(wraps=planner.plan)
    state.route_planner = planner
    state.route_planner_enabled = True
    return state, planner


def test_candidate_and_route_latch_ignore_motion_and_reordering(monkeypatch):
    state, planner = _state(monkeypatch)
    first = state.action_payload((0.0, 0.0, 0.0))
    assert first is not None
    key = state._simple_oracle_candidate_key
    assert planner.plan.call_count == 1

    # Regenerated candidates arrive in a different order while the robot has
    # moved.  The latched semantic target and route must remain unchanged.
    state.candidates = list(reversed(state.candidates))
    second = state.action_payload((2.5, 3.0, 0.0))
    assert second is not None
    assert state._simple_oracle_candidate_key == key
    assert planner.plan.call_count == 1
    assert second["TICVLA_PMG_GOAL_X"] == first["TICVLA_PMG_GOAL_X"]


def test_arrival_is_sticky_for_single_target(monkeypatch):
    state, planner = _state(monkeypatch)
    state.action_payload((0.0, 0.0, 0.0))
    arrived = state.action_payload((4.0, 0.0, 0.0))
    assert arrived is not None
    assert state._simple_oracle_arrived is True
    assert state._simple_oracle_candidate_key == (4.0, 0.0, 0.0)
    # No neighbouring radial sample is selected at the terminal radius.
    assert planner.plan.call_count == 1


def test_terminal_entry_replans_once_without_changing_candidate(monkeypatch):
    state, planner = _state(monkeypatch)
    monkeypatch.setenv("PMG_CONTROL_POLICY", "terminal")
    monkeypatch.setenv("PMG_TERMINAL_ENTER_DISTANCE", "2.0")
    state.action_payload((0.0, 0.0, 0.0))
    key = state._simple_oracle_candidate_key
    # Fair terminal control defers A* until the semantic candidate enters the
    # local hand-off radius; the first frame must not incur route computation.
    assert planner.plan.call_count == 0
    # Cross from outside to inside the terminal hand-off radius.  The route
    # is refreshed from the local pose exactly once; its semantic target does
    # not change and subsequent frames reuse the local plan.
    state.action_payload((2.5, 0.0, 0.0))
    assert planner.plan.call_count == 1
    assert state._simple_oracle_candidate_key == key
    # No stale start route existed, so the first local plan is not counted as
    # a terminal replan event.
    assert state._simple_oracle_terminal_replan_done is False
    state.action_payload((3.0, 0.0, 0.0))
    assert planner.plan.call_count == 1


def test_terminal_scope_rechecked_after_far_candidate_selection(monkeypatch):
    """A replacement candidate cannot leak a terminal route out of scope.

    The initial semantic latch is inside the 4 m hand-off radius, but a
    feasibility selector may replace it with a candidate at 4.5 m.  That
    candidate is also inside the 5 m hysteresis *exit* radius, so the second
    check must use the pre-selection route state rather than treating the
    newly committed route as already active.
    """
    state, _ = _state(monkeypatch)
    monkeypatch.setenv("PMG_CONTROL_POLICY", "terminal")
    monkeypatch.setenv("PMG_TERMINAL_ENTER_DISTANCE", "4.0")
    monkeypatch.setenv("PMG_TERMINAL_EXIT_DISTANCE", "5.0")
    near = GoalCandidate((2.0, 0.0, 0.0), 0.0, score=1.0, hard_valid=True)
    far = GoalCandidate((4.5, 0.0, 0.0), 0.0, score=0.9, hard_valid=True)
    state._simple_oracle_candidate_latched = near
    state._simple_oracle_candidate_key = state._candidate_position_key(near.position)
    state.candidates = [near, far]

    def _select_far(_robot_position):
        # Model the normal selector side effect: it commits the route before
        # action_payload() performs its post-selection scope check.
        state._route_plan = RoutePlan(
            waypoints=(far.position,),
            reachable=True,
            length_m=4.5,
            expanded_nodes=0,
            direct=True,
            reason="astar",
            resolution_m=0.5,
        )
        state._route_candidate_key = far.position
        return far

    monkeypatch.setattr(state, "_route_select_candidate", _select_far)
    payload = state.action_payload((0.0, 0.0, 0.0))
    assert payload is not None
    assert payload["TICVLA_PMG_GOAL_X"] == "4.5"
    assert payload["TICVLA_PMG_ROUTE_ENABLED"] == "0"
    assert payload["TICVLA_PMG_ROUTE_VALID"] == "0"
    assert payload["TICVLA_PMG_ROUTE_REASON"] == "deferred_until_terminal_selected_candidate"
    assert state._route_compute_allowed is False
    assert state._route_compute_deferred_count == 1
    assert state._route_plan is None


def test_visibility_miss_is_hysteretic(monkeypatch):
    state, _ = _state(monkeypatch)
    state.action_payload((0.0, 0.0, 0.0))
    key = state._simple_oracle_candidate_key
    state.candidates = []
    for frame in (15, 30):
        state._simple_oracle_observation_update(frame)
        assert state._simple_oracle_candidate_key == key
    # The third consecutive observation miss is the first one allowed to
    # release the latch.
    state._simple_oracle_observation_update(45)
    assert state._simple_oracle_candidate_key is None
