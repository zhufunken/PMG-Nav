from pmg_nav.contracts import CandidateEvidence, GoalRole, SceneState
from pmg_nav.scgp import SCGPConfig, SceneInvariantExecutor


def _evidence():
    return CandidateEvidence(
        source="oracle_depth",
        confidence=0.9,
        identity_stable=True,
        hard_constraints_satisfied=True,
        free_endpoint=True,
        reachable=True,
        orientation_feasible=True,
        footprint_clearance_m=0.8,
        uncertainty_m=0.05,
        route_cost_m=4.0,
    )


def _scene(frame, progress=0.2):
    return SceneState(
        frame_id=frame,
        robot_position=(0.0, 0.0, 0.0),
        robot_radius_m=0.45,
        candidate_count=1,
        progress_delta_m=progress,
        route_valid=True,
    )


def test_executor_requires_stability_then_issues_checkpoint_lease():
    executor = SceneInvariantExecutor(SCGPConfig(stable_frames=2, lease_frames=4))
    first = executor.decide(
        _scene(1),
        candidate_id="desk:1",
        goal_pose=(2.0, 0.0, 0.0),
        waypoint=(1.0, 0.0, 0.0),
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    second = executor.decide(
        _scene(2),
        candidate_id="desk:1",
        goal_pose=(2.0, 0.0, 0.0),
        waypoint=(1.0, 0.0, 0.0),
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    assert first.mode.value == "fallback"
    assert first.fallback_reason == "IDENTITY_UNSTABLE"
    assert second.mode.value == "pmg_checkpoint"
    assert second.is_active(6)
    assert not second.is_active(7)


def test_executor_fails_closed_on_no_progress_and_unverified_terminal():
    executor = SceneInvariantExecutor(SCGPConfig(stable_frames=1, max_no_progress_ticks=2))
    unverified = executor.decide(
        _scene(1, progress=0.2),
        candidate_id="target:1",
        goal_pose=(2.0, 0.0, 0.0),
        waypoint=None,
        goal_role=GoalRole.FINAL_GOAL,
        evidence=_evidence(),
    )
    stalled = executor.decide(
        _scene(2, progress=0.0),
        candidate_id="target:1",
        goal_pose=(2.0, 0.0, 0.0),
        waypoint=None,
        goal_role=GoalRole.FINAL_GOAL,
        evidence=_evidence(),
    )
    stalled_again = executor.decide(
        _scene(3, progress=0.0),
        candidate_id="target:1",
        goal_pose=(2.0, 0.0, 0.0),
        waypoint=None,
        goal_role=GoalRole.FINAL_GOAL,
        evidence=_evidence(),
    )
    assert unverified.fallback_reason == "TERMINAL_NOT_VERIFIED"
    assert stalled.fallback_reason == "TERMINAL_NOT_VERIFIED"
    assert stalled_again.fallback_reason == "NO_PROGRESS"


def test_executor_does_not_depend_on_scene_name():
    executor = SceneInvariantExecutor(SCGPConfig(stable_frames=1))
    result = executor.decide(
        _scene(1),
        candidate_id="anchor:1",
        goal_pose=(1.0, 0.0, 0.0),
        waypoint=(0.5, 0.0, 0.0),
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    assert result.is_pmg_owned
    assert "scene" not in result.reason.lower()


def test_executor_requires_explicit_route_validation():
    executor = SceneInvariantExecutor(SCGPConfig(stable_frames=1))
    scene = SceneState(
        frame_id=1,
        robot_position=(0.0, 0.0, 0.0),
        robot_radius_m=0.45,
        candidate_count=1,
        progress_delta_m=0.2,
        route_valid=None,
    )
    result = executor.decide(
        scene,
        candidate_id="anchor:1",
        goal_pose=(1.0, 0.0, 0.0),
        waypoint=(0.5, 0.0, 0.0),
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    assert result.mode.value == "fallback"
    assert result.fallback_reason == "ROUTE_UNREACHABLE"


def test_executor_does_not_renew_same_candidate_after_lease_expiry():
    executor = SceneInvariantExecutor(SCGPConfig(stable_frames=1, lease_frames=2))
    first = executor.decide(
        _scene(1),
        candidate_id="anchor:lease",
        goal_pose=(1.0, 0.0, 0.0),
        waypoint=(0.5, 0.0, 0.0),
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    still_active = executor.decide(
        _scene(3),
        candidate_id="anchor:lease",
        goal_pose=(1.0, 0.0, 0.0),
        waypoint=(0.5, 0.0, 0.0),
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    expired = executor.decide(
        _scene(4),
        candidate_id="anchor:lease",
        goal_pose=(1.0, 0.0, 0.0),
        waypoint=(0.5, 0.0, 0.0),
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    assert first.valid_until_frame == 3
    assert still_active.is_pmg_owned
    assert expired.fallback_reason == "LEASE_EXPIRED"


def test_executor_fails_closed_on_invalid_control_inputs():
    executor = SceneInvariantExecutor(SCGPConfig(stable_frames=1))
    empty_id = executor.decide(
        _scene(1),
        candidate_id=" ",
        goal_pose=(1.0, 0.0, 0.0),
        waypoint=None,
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    bad_pose = executor.decide(
        _scene(2),
        candidate_id="target:1",
        goal_pose=(float("nan"), 0.0, 0.0),
        waypoint=None,
        goal_role=GoalRole.CHECKPOINT,
        evidence=_evidence(),
    )
    bad_role = executor.decide(
        _scene(3),
        candidate_id="target:1",
        goal_pose=(1.0, 0.0, 0.0),
        waypoint=None,
        goal_role="unknown",
        evidence=_evidence(),
    )
    assert empty_id.fallback_reason == "INVALID_CANDIDATE_ID"
    assert bad_pose.fallback_reason == "INVALID_CONTROL_REFERENCE"
    assert bad_role.fallback_reason == "UNSUPPORTED_GOAL_ROLE"
