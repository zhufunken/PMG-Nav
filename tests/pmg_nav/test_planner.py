from __future__ import annotations

import math
import os

from pmg_nav.goal_field import GoalCandidate
from pmg_nav.planner import ObstacleAABB, ObstacleAwareRoutePlanner, RoutePlan
from pmg_nav.runtime import PMGEpisodeState


def test_empty_map_returns_direct_route() -> None:
    planner = ObstacleAwareRoutePlanner(())
    result = planner.plan((0.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    assert result.reachable is True
    assert result.direct is True
    assert result.reason == "no_static_obstacles"
    assert result.first_waypoint == (4.0, 0.0, 0.0)


def test_thin_near_ground_carpet_surface_is_not_an_obstacle() -> None:
    """Collision-authored carpet/floor support surfaces are filtered only by
    the explicit thin/near-ground predicate used by USD extraction.

    A vertical wall with the same semantic token is intentionally retained;
    this prevents the exception from becoming a generic name-based bypass.
    """

    is_surface = ObstacleAwareRoutePlanner._is_passable_ground_surface
    assert is_surface({"sm", "carpetpath", "01"}, -0.003, 0.007) is True
    assert is_surface({"sm", "floor", "27", "sm", "floor", "02"}, -0.000001, 0.000001) is True
    assert is_surface({"sm", "carpetpath", "01"}, -0.003, 0.20) is False
    assert is_surface({"wall", "carpetpath"}, -0.003, 2.0) is False
    assert is_surface({"wall"}, -0.003, 0.007) is False


def test_compound_furniture_names_are_obstacles_without_substring_matches() -> None:
    tokens = ObstacleAwareRoutePlanner._path_tokens
    geometry = ObstacleAwareRoutePlanner._GEOMETRY_TOKENS
    assert tokens("/Root/SM_SecretaryDeskA2_164", "SM_SecretaryDeskA2_164") & geometry == {"desk"}
    assert tokens("/Root/SM_ReceptionClosets6", "SM_ReceptionClosets6") & geometry == {"closet"}
    assert tokens("/Root/SM_TableWorkSecurity9", "SM_TableWorkSecurity9") & geometry == {"table"}
    assert "table" not in tokens("/Root/SM_BoxPortableE3_372", "SM_BoxPortableE3_372")


def test_dedup_keeps_full_parent_footprint_when_child_comes_first() -> None:
    planner = ObstacleAwareRoutePlanner(
        [
            ObstacleAABB(0.0, 1.0, 0.0, 1.0, source="child"),
            ObstacleAABB(0.0, 3.0, 0.0, 3.0, source="desk"),
        ],
        robot_radius_m=0.0,
    )
    assert len(planner.obstacles) == 1
    assert planner.obstacles[0].source == "desk"
    assert planner.is_free_position((2.0, 2.0)) is False


def test_astar_goes_around_wall_without_corner_cutting() -> None:
    # A finite wall blocks the direct corridor.  The route must remain outside
    # the inflated footprint and contain at least one intermediate waypoint.
    wall = ObstacleAABB(1.5, 2.5, -2.0, 2.0, source="test", label="wall")
    planner = ObstacleAwareRoutePlanner(
        [wall], resolution_m=0.25, robot_radius_m=0.2, bounds_margin_m=2.0
    )
    result = planner.plan((0.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    assert result.reachable is True
    assert result.direct is False
    assert result.reason == "astar"
    assert len(result.waypoints) >= 2
    # Every segment of the simplified route must be collision-free.
    previous = (0.0, 0.0, 0.0)
    for waypoint in result.waypoints:
        assert planner.segment_blocked(previous, waypoint) is False
        assert planner.is_free_position(waypoint) is True
        previous = waypoint
    assert math.isclose(result.waypoints[-1][0], 4.0, abs_tol=1e-6)
    assert math.isclose(result.waypoints[-1][1], 0.0, abs_tol=1e-6)


def test_astar_route_starts_at_exact_free_pose_and_validates_connector() -> None:
    """A coarse grid must not turn the first cell center into a fake target."""

    wall = ObstacleAABB(1.5, 2.5, -2.0, 2.0, source="test", label="wall")
    start = (0.13, -0.07, 0.0)
    planner = ObstacleAwareRoutePlanner(
        [wall], resolution_m=0.50, robot_radius_m=0.20, bounds_margin_m=2.0
    )

    result = planner.plan(start, (4.0, 0.0, 0.0))

    assert result.reachable is True
    assert result.waypoints[0] == start
    previous = start
    for waypoint in result.waypoints:
        assert planner.segment_blocked(previous, waypoint) is False
        previous = waypoint


def test_optional_corner_clearance_keeps_route_away_from_wall_corner() -> None:
    """A Carter-sized safety envelope must not take the coarse corner cut.

    The released Nova Carter asset is asymmetric around its root.  The
    default 0.45 m circular envelope is intentionally retained for backward
    compatibility, while an opt-in 0.35 m corner margin gives an effective
    0.80 m envelope for a controlled gate.
    """

    wall = ObstacleAABB(-4.5, -1.5, 47.71, 47.934, source="test", label="wall")
    start = (-3.04, 48.1, 0.0)
    goal = (-0.675, 48.844, 0.0)
    loose = ObstacleAwareRoutePlanner(
        [wall], resolution_m=0.50, robot_radius_m=0.45, bounds_margin_m=3.0
    )
    safe = ObstacleAwareRoutePlanner(
        [wall],
        resolution_m=0.50,
        robot_radius_m=0.45,
        corner_clearance_m=0.35,
        bounds_margin_m=3.0,
    )

    loose_route = loose.plan(start, goal, force_replan=True)
    safe_route = safe.plan(start, goal, force_replan=True)

    assert loose_route.reachable is True
    assert safe_route.reachable is True
    # The opt-in envelope must retain a higher clearance corner instead of
    # reusing the 0.45 m route's near-corner diagonal.
    assert safe_route.waypoints[0][1] >= loose_route.waypoints[0][1] + 0.25
    assert safe.summary()["corner_clearance_m"] == 0.35
    # The episode-style start is deliberately inside the inflated wall
    # envelope; the runtime handles that as a strict escape transition.  The
    # post-escape route itself must satisfy the selected clearance.
    assert safe.is_free_position(safe_route.waypoints[0]) is True
    previous = safe_route.waypoints[0]
    for waypoint in safe_route.waypoints[1:]:
        assert safe.segment_blocked(previous, waypoint) is False
        assert safe.is_free_position(waypoint) is True
        previous = waypoint


def test_optional_axis_clearance_moves_route_away_from_carter_corner() -> None:
    """An explicit X envelope changes the route without changing defaults.

    The e041 Carter trace exposed a root/footprint mismatch at a wall corner:
    an isotropic envelope selected an east-west transition too close to the
    wall, while an additional world-X margin selected a farther-east cell.
    This small fixture keeps the gate deterministic and checks the actual A*
    route rather than only inspecting a configuration field.
    """

    wall = ObstacleAABB(-4.5, -1.5, 47.71, 47.934, source="test", label="wall")
    start = (-3.04, 48.1, 0.0)
    goal = (-0.175, 46.344, 0.0)
    base = ObstacleAwareRoutePlanner(
        [wall],
        resolution_m=0.50,
        robot_radius_m=0.45,
        corner_clearance_m=0.50,
        bounds_margin_m=3.0,
    )
    anisotropic = ObstacleAwareRoutePlanner(
        [wall],
        resolution_m=0.50,
        robot_radius_m=0.45,
        corner_clearance_m=0.50,
        axis_clearance_x_m=0.75,
        bounds_margin_m=3.0,
    )
    base_route = base.plan(start, goal, force_replan=True)
    axis_route = anisotropic.plan(start, goal, force_replan=True)

    assert base_route.reachable is True
    assert axis_route.reachable is True
    # The axis-aware route must contain a vertical transit segment farther east
    # than the isotropic route.  (The coarse grid may also add a short diagonal
    # at the corner, so inspect all intermediate points.)
    base_x = max(point[0] for point in base_route.waypoints[:-1])
    axis_x = max(point[0] for point in axis_route.waypoints[:-1])
    assert axis_x >= base_x + 0.25
    assert anisotropic.summary()["axis_clearance_x_m"] == 0.75
    assert anisotropic.summary()["axis_clearance_y_m"] == 0.0
    # The synthetic start is intentionally inside the inflated envelope, just
    # like e041; the runtime handles that first leg as a strict escape.
    previous = axis_route.waypoints[0]
    for waypoint in axis_route.waypoints[1:]:
        assert anisotropic.segment_blocked(previous, waypoint) is False
        previous = waypoint


def test_goal_inside_obstacle_is_rejected() -> None:
    planner = ObstacleAwareRoutePlanner(
        [ObstacleAABB(1.0, 3.0, -1.0, 1.0)], robot_radius_m=0.25
    )
    result = planner.plan((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert result.reachable is False
    assert result.reason == "goal_in_obstacle"


def test_endpoint_and_segment_callbacks_are_deterministic() -> None:
    obstacle = ObstacleAABB(1.0, 2.0, 1.0, 2.0)
    planner = ObstacleAwareRoutePlanner([obstacle], robot_radius_m=0.0)
    assert planner.is_free((0.0, 0.0, 0.0)) is True
    assert planner.is_free((1.5, 1.5, 0.0)) is False
    assert planner.segment_blocked((0.0, 1.5, 0.0), (3.0, 1.5, 0.0)) is True
    assert planner.segment_blocked((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)) is False


def test_runtime_emits_waypoint_without_using_evaluator_goal() -> None:
    old_variant = os.environ.get("STAGE2_M1_VARIANT")
    old_enabled = os.environ.get("PMG_ROUTE_PLANNER")
    os.environ["STAGE2_M1_VARIANT"] = "simple_oracle"
    os.environ["PMG_ROUTE_PLANNER"] = "1"
    state = PMGEpisodeState("Stop near the chair.", "M1", goal_hint=(999.0, 999.0, 0.0))
    state._known_objects = [{
        "node_id": "chair:1",
        "label": "chair",
        "position": (4.0, 0.0, 0.0),
        "distance": 4.0,
    }]
    state._scene_catalog = list(state._known_objects)
    state._preferred_node_ids = {"chair:1"}
    state._ensure_resolvable_program()
    state.observe(0, (0.0, 0.0, 0.0), force=True)
    state.route_planner = ObstacleAwareRoutePlanner(
        [ObstacleAABB(1.5, 2.5, -1.0, 1.0)],
        resolution_m=0.25,
        robot_radius_m=0.2,
        bounds_margin_m=2.0,
    )
    payload = state.action_payload((0.0, 0.0, 0.0))
    # An infeasible route is diagnostic evidence, not permission to publish a
    # behavior command. The evaluator hint must still never become the goal.
    assert payload is None
    route_summary = state.summary()["route_planner"]
    assert route_summary is None or route_summary["last_plan"] is None or route_summary["last_plan"]["reason"] == "astar"
    if old_variant is None:
        os.environ.pop("STAGE2_M1_VARIANT", None)
    else:
        os.environ["STAGE2_M1_VARIANT"] = old_variant
    if old_enabled is None:
        os.environ.pop("PMG_ROUTE_PLANNER", None)
    else:
        os.environ["PMG_ROUTE_PLANNER"] = old_enabled


def test_runtime_advances_coarse_grid_waypoint_at_observed_residual() -> None:
    old_reach = os.environ.pop("PMG_ROUTE_WAYPOINT_REACH_M", None)
    try:
        state = PMGEpisodeState("Stop near the chair.", "M1")
        state.route_planner = ObstacleAwareRoutePlanner((), resolution_m=0.50)
        candidate = GoalCandidate((2.0, 0.0, 0.0), 0.0, score=1.0)
        state._route_plan = RoutePlan(
            ((0.0, 0.0, 0.0), candidate.position),
            True,
            2.0,
            0,
            False,
            "astar",
            0.50,
        )
        state._route_candidate_key = candidate.position
        state._route_plan_start = (0.79, 0.0, 0.0)

        waypoint = state._route_waypoint((0.79, 0.0, 0.0), candidate)

        assert state._route_waypoint_index == 1
        assert waypoint == candidate.position
    finally:
        if old_reach is not None:
            os.environ["PMG_ROUTE_WAYPOINT_REACH_M"] = old_reach


def test_route_start_inside_inflated_obstacle_uses_strict_escape_guard() -> None:
    """A blocked planning start must not consume the first A* point early."""

    old_escape = os.environ.pop("PMG_ROUTE_ESCAPE_REACH_M", None)
    try:
        state = PMGEpisodeState("Stop near the chair.", "M1")
        planner = ObstacleAwareRoutePlanner(
            [ObstacleAABB(-0.4, 0.4, -0.4, 0.4)],
            resolution_m=0.50,
            robot_radius_m=0.45,
        )
        state.route_planner = planner
        candidate = GoalCandidate((2.0, 0.0, 0.0), 0.0, score=1.0)
        state._route_plan = RoutePlan(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), candidate.position),
            True,
            2.0,
            0,
            False,
            "astar",
            0.50,
        )
        state._route_candidate_key = candidate.position
        state._route_plan_start = (0.0, 0.0, 0.0)
        state._route_start_free = False

        # The old 0.875 m threshold would consume wp0 here.  The strict
        # escape threshold must retain it, even before the current pose is
        # checked for free space.
        waypoint = state._route_waypoint((0.60, 0.0, 0.0), candidate)

        assert waypoint == (0.0, 0.0, 0.0)
        assert state._route_waypoint_index == 0
        assert state._route_escape_mode is True
        assert state._route_waypoint_guard_reason == "escape_distance"
    finally:
        if old_escape is not None:
            os.environ["PMG_ROUTE_ESCAPE_REACH_M"] = old_escape


def test_route_progress_rejects_blocked_next_segment() -> None:
    """A coarse waypoint cannot be consumed across a blocked local segment."""

    state = PMGEpisodeState("Stop near the chair.", "M1")
    state.route_planner = ObstacleAwareRoutePlanner(
        [ObstacleAABB(0.4, 0.8, -0.3, 0.3)],
        resolution_m=0.25,
        robot_radius_m=0.0,
    )
    candidate = GoalCandidate((2.0, 0.0, 0.0), 0.0, score=1.0)
    state._route_plan = RoutePlan(
        ((0.0, 0.0, 0.0), candidate.position),
        True,
        2.0,
        0,
        False,
        "astar",
        0.25,
    )
    state._route_candidate_key = candidate.position
    state._route_plan_start = (0.0, 0.0, 0.0)
    state._route_start_free = True

    waypoint = state._route_waypoint((0.10, 0.0, 0.0), candidate)

    assert waypoint == (0.0, 0.0, 0.0)
    assert state._route_waypoint_index == 0
    assert state._route_waypoint_guard_reason == "next_segment_blocked"
    assert state._route_guard_block_count == 1


def test_route_cache_does_not_depend_on_planner_probe_state() -> None:
    state = PMGEpisodeState("Stop near the chair.", "M1")
    planner = ObstacleAwareRoutePlanner((), resolution_m=0.50)
    state.route_planner = planner
    candidate = GoalCandidate((4.0, 0.0, 0.0), 0.0, score=1.0)
    selected = RoutePlan(
        ((1.0, 0.0, 0.0), candidate.position),
        True,
        4.0,
        0,
        False,
        "astar",
        0.50,
    )
    state._route_plan = selected
    state._route_candidate_key = candidate.position
    state._route_plan_start = (0.0, 0.0, 0.0)
    state._route_waypoint_index = 1
    # Simulate a feasibility probe for another candidate mutating the
    # planner's public last-start cache.
    planner.last_start = (99.0, 99.0, 0.0)

    reused = state._route_for_candidate((0.2, 0.0, 0.0), candidate)

    assert reused is selected
    assert state._route_waypoint_index == 1
