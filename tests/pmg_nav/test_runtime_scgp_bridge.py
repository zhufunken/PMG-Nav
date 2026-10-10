from pmg_nav.goal_field import GoalCandidate
from pmg_nav.planner import RoutePlan
from pmg_nav.runtime import PMGEpisodeState


def test_runtime_scgp_reference_is_real_control_authority(monkeypatch):
    monkeypatch.setenv("PMG_SCGP_ENABLED", "1")
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    state = PMGEpisodeState("approach the red chair", method="M1")
    state._route_obstacle_count = 1
    state._route_plan = RoutePlan(
        waypoints=((0.5, 0.0, 0.0),),
        reachable=True,
        length_m=1.0,
        expanded_nodes=1,
        direct=True,
        reason="direct",
        resolution_m=0.5,
    )
    candidate = GoalCandidate(
        position=(1.0, 0.0, 0.0),
        yaw=0.0,
        score=1.0,
        hard_valid=True,
        evidence={"confidence": 1.0},
    )
    state._candidate_stable = True
    state.candidates = [candidate]
    state.update_from_robot(1, (0.0, 0.0, 0.0))
    reference = state._scgp_reference(
        (0.0, 0.0, 0.0), candidate, (0.5, 0.0, 0.0), True
    )
    assert reference is not None
    assert reference.is_pmg_owned
    assert state.summary()["scgp_enabled"] is True
    assert state.summary()["persistent_graph_enabled"] is True


def test_runtime_defaults_preserve_frozen_path(monkeypatch):
    monkeypatch.delenv("PMG_SCGP_ENABLED", raising=False)
    monkeypatch.delenv("PMG_PERSISTENT_GRAPH", raising=False)
    state = PMGEpisodeState("approach the red chair", method="M1")
    assert state.scgp_enabled is False
    assert state.persistent_graph_enabled is False
