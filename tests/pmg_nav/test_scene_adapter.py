from types import SimpleNamespace

from pmg_nav.scene_adapter import SceneGraphAdapter


def _candidate(position, *, hard_valid=True, reachable=True, blocking=False):
    return SimpleNamespace(
        position=position,
        hard_valid=hard_valid,
        evidence={"reachable": reachable, "blocking": blocking},
    )


def test_profile_is_scene_name_agnostic_and_deterministic():
    adapter = SceneGraphAdapter()
    stats = adapter.measure(
        graph_nodes=12,
        graph_edges=24,
        anchor_nodes=2,
        node_positions=[(0, 0, 0), (10, 0, 0), (0, 8, 0)],
        obstacle_count=18,
        candidates=[_candidate((4, 0, 0))],
    )
    first = adapter.adapt(stats)
    second = adapter.adapt(stats)
    assert first == second
    assert first.graph_node_budget > adapter.base_node_budget
    assert first.require_reachable_route is True
    assert first.profile in {"wide_geometry", "constrained_geometry"}


def test_negative_progress_is_preserved_and_rejects_candidate():
    adapter = SceneGraphAdapter()
    stats = adapter.measure(
        graph_nodes=4,
        graph_edges=2,
        anchor_nodes=1,
        candidates=[_candidate((2, 0, 0))],
        progress_delta_m=-0.5,
    )
    assert stats.progress_delta_m == -0.5
    profile = adapter.adapt(stats)
    decision = adapter.evaluate_candidate(
        _candidate((2, 0, 0)),
        robot_position=(0, 0, 0),
        profile=profile,
        route_valid=True,
        progress_delta_m=-0.5,
    )
    assert decision.accepted is False
    assert decision.reason == "negative_progress"


def test_no_route_or_blocked_candidate_fails_closed():
    adapter = SceneGraphAdapter()
    stats = adapter.measure(
        graph_nodes=4,
        graph_edges=2,
        anchor_nodes=1,
        obstacle_count=4,
        candidates=[_candidate((2, 0, 0), blocking=True)],
    )
    profile = adapter.adapt(stats)
    blocked = adapter.evaluate_candidate(
        _candidate((2, 0, 0), blocking=True),
        robot_position=(0, 0, 0),
        profile=profile,
        route_valid=True,
        progress_delta_m=0.2,
    )
    unreachable = adapter.evaluate_candidate(
        _candidate((2, 0, 0)),
        robot_position=(0, 0, 0),
        profile=profile,
        route_valid=False,
        progress_delta_m=0.2,
    )
    assert blocked.reason == "candidate_blocked"
    assert unreachable.reason == "route_not_reachable"


def test_adapter_does_not_consult_scene_name_or_evaluator_state():
    adapter = SceneGraphAdapter()
    stats = adapter.measure(
        graph_nodes=3,
        graph_edges=1,
        anchor_nodes=1,
        node_positions=[(1, 0, 0)],
        candidates=[_candidate((1, 0, 0))],
    )
    profile = adapter.adapt(stats)
    assert "hospital" not in profile.profile
    assert "warehouse" not in profile.profile
    assert not hasattr(stats, "success")
    assert not hasattr(stats, "goal")
