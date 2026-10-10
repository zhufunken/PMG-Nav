import math

from pmg_nav.contracts import (
    CandidateEvidence,
    ControlMode,
    ControlReference,
    FrameTransform,
    GoalRole,
    SceneState,
)


def _raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_frame_transform_round_trip_and_audit_gate():
    transform = FrameTransform(
        "sensor",
        "world",
        translation_m=(1.0, -2.0, 0.5),
        yaw_rad=math.pi / 3.0,
        unit_scale=1.25,
    )
    point = (0.75, -1.5, 0.2)
    recovered = transform.inverse().apply(transform.apply(point))
    assert all(abs(left - right) < 1e-9 for left, right in zip(recovered, point))
    assert transform.round_trip_ok
    noisy = FrameTransform("sensor", "world", round_trip_error_m=0.2)
    assert not noisy.round_trip_ok
    assert transform.to_dict()["source_frame"] == "sensor"


def test_candidate_evidence_fails_closed_and_accepts_complete_evidence():
    incomplete = CandidateEvidence()
    assert not incomplete.is_feasible()
    assert "identity_unstable" in incomplete.failure_reasons()
    complete = CandidateEvidence(
        source="oracle_depth",
        frame_id=12,
        confidence=0.9,
        identity_stable=True,
        hard_constraints_satisfied=True,
        free_endpoint=True,
        reachable=True,
        orientation_feasible=True,
        footprint_clearance_m=0.65,
        uncertainty_m=0.08,
        route_cost_m=4.2,
    )
    assert complete.is_feasible(min_clearance_m=0.5, min_confidence=0.8)
    assert complete.failure_reasons(min_clearance_m=0.7) == ("insufficient_clearance",)


def test_scene_state_is_serializable_without_scene_specific_fields():
    state = SceneState(
        frame_id=7,
        robot_position=(1.0, 2.0, 0.0),
        graph_nodes=4,
        graph_edges=6,
        obstacle_count=2,
        candidate_count=3,
        progress_delta_m=0.15,
        route_valid=True,
        frame_transform=FrameTransform("camera", "world"),
    )
    serialized = state.to_dict()
    assert serialized["frame_id"] == 7
    assert serialized["frame_transform"]["target_frame"] == "world"
    assert "scene_name" not in serialized
    _raises_value_error(lambda: SceneState(frame_id=0, robot_position=(0.0, 0.0)))


def test_control_reference_enforces_role_and_lease_window():
    reference = ControlReference(
        mode=ControlMode.PMG_CHECKPOINT,
        candidate_id="candidate-1",
        goal_pose=(3.0, 1.0, 0.0),
        waypoint=(2.0, 1.0, 0.0),
        goal_role=GoalRole.CHECKPOINT,
        issued_frame=4,
        valid_until_frame=8,
        route_valid=True,
        clearance_m=0.6,
        progress_delta_m=0.2,
        reason="stable_reachable_checkpoint",
    )
    assert reference.is_pmg_owned
    assert not reference.is_active(3)
    assert reference.is_active(4)
    assert reference.is_active(8)
    assert not reference.is_active(9)
    assert reference.to_dict()["goal_role"] == "checkpoint"
    _raises_value_error(
        lambda: ControlReference(
            mode=ControlMode.PMG_TERMINAL,
            candidate_id="candidate-2",
            goal_role=GoalRole.CHECKPOINT,
        )
    )
