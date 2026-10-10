"""Small, serializable contracts for the scene-agnostic PMG layer.

The contracts in this module intentionally contain no simulator, model, or
evaluator dependencies.  They are an opt-in boundary for a future persistent
graph and progress-aware controller.  Existing Stage 2 code can continue to
use its current dictionaries and environment variables; importing this module
alone does not change that path.

All coordinates are metres in a right-handed XY navigation plane with a Z
component retained for auditability.  Angles are radians.  Validation is
fail-closed: malformed geometry raises ``ValueError`` instead of silently
creating a control reference from invalid data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Sequence


Point3 = tuple[float, float, float]


def _point3(value: Sequence[float], *, name: str) -> Point3:
    """Return a finite three-dimensional point.

    A tuple is stored regardless of whether callers provide a list, numpy
    array, or another sequence.  The module does not import numpy so it stays
    usable in the lightweight M0 environment.
    """

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a numeric 3-vector")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a numeric 3-vector") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    try:
        point = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 3-vector") from exc
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{name} must contain only finite values")
    return point  # type: ignore[return-value]


def _finite(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative(value: float, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _count(value: int, *, name: str) -> int:
    # bool is an int subclass but is almost always an accidental input here.
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


class GoalRole(str, Enum):
    """Role of a PMG pose in the task state machine."""

    NONE = "none"
    CHECKPOINT = "checkpoint"
    FINAL_GOAL = "final_goal"
    RECOVERY = "recovery"


class ControlMode(str, Enum):
    """Controller owner represented by a :class:`ControlReference`."""

    POLICY = "policy"
    PMG_CHECKPOINT = "pmg_checkpoint"
    PMG_TERMINAL = "pmg_terminal"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class FrameTransform:
    """Planar rigid transform plus audit information.

    ``apply`` interprets the transform as ``target = R(yaw) * (scale *
    source) + translation``.  The scale is useful when a sensor reports a
    different linear unit; it is deliberately explicit rather than inferred.
    ``round_trip_error_m`` is supplied by the caller after checking the
    corresponding inverse transform and is used by control code as a gate.
    """

    source_frame: str
    target_frame: str
    translation_m: Point3 = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0
    unit_scale: float = 1.0
    round_trip_error_m: float = 0.0
    max_round_trip_error_m: float = 0.05

    def __post_init__(self) -> None:
        if not str(self.source_frame).strip() or not str(self.target_frame).strip():
            raise ValueError("source_frame and target_frame must be non-empty")
        object.__setattr__(self, "source_frame", str(self.source_frame))
        object.__setattr__(self, "target_frame", str(self.target_frame))
        object.__setattr__(self, "translation_m", _point3(self.translation_m, name="translation_m"))
        object.__setattr__(self, "yaw_rad", _finite(self.yaw_rad, name="yaw_rad"))
        scale = _finite(self.unit_scale, name="unit_scale")
        if scale <= 0.0:
            raise ValueError("unit_scale must be greater than zero")
        object.__setattr__(self, "unit_scale", scale)
        object.__setattr__(
            self,
            "round_trip_error_m",
            _nonnegative(self.round_trip_error_m, name="round_trip_error_m"),
        )
        object.__setattr__(
            self,
            "max_round_trip_error_m",
            _nonnegative(self.max_round_trip_error_m, name="max_round_trip_error_m"),
        )

    @property
    def round_trip_ok(self) -> bool:
        """Whether this transform is safe for a geometry-based hand-off."""

        return self.round_trip_error_m <= self.max_round_trip_error_m

    def apply(self, point: Sequence[float]) -> Point3:
        """Transform a point from ``source_frame`` into ``target_frame``."""

        x, y, z = _point3(point, name="point")
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        scaled_x = self.unit_scale * x
        scaled_y = self.unit_scale * y
        tx, ty, tz = self.translation_m
        return (
            c * scaled_x - s * scaled_y + tx,
            s * scaled_x + c * scaled_y + ty,
            self.unit_scale * z + tz,
        )

    def inverse(self) -> "FrameTransform":
        """Return the exact inverse transform with swapped frame names."""

        inverse_yaw = -self.yaw_rad
        c = math.cos(inverse_yaw)
        s = math.sin(inverse_yaw)
        tx, ty, tz = self.translation_m
        inverse_scale = 1.0 / self.unit_scale
        # The inverse translation is -R^T t / scale.
        inverse_translation = (
            inverse_scale * (-c * tx + s * ty),
            inverse_scale * (-s * tx - c * ty),
            -inverse_scale * tz,
        )
        return FrameTransform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            translation_m=inverse_translation,
            yaw_rad=inverse_yaw,
            unit_scale=inverse_scale,
            round_trip_error_m=self.round_trip_error_m,
            max_round_trip_error_m=self.max_round_trip_error_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"round_trip_ok": self.round_trip_ok}


@dataclass(frozen=True)
class CandidateEvidence:
    """Auditable evidence attached to one candidate goal pose.

    The booleans mirror the invariant checks in the design document.  A
    caller may still use a candidate for diagnostics when the evidence is
    incomplete, but ``is_feasible`` will reject it until all hard checks pass.
    """

    source: str = "unknown"
    frame_id: int | None = None
    confidence: float = 0.0
    identity_stable: bool = False
    hard_constraints_satisfied: bool = False
    free_endpoint: bool = False
    reachable: bool = False
    orientation_feasible: bool = False
    blocking: bool = False
    contradiction: bool = False
    footprint_clearance_m: float = 0.0
    uncertainty_m: float = 0.0
    route_cost_m: float | None = None

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "source", str(self.source))
        if self.frame_id is not None:
            object.__setattr__(self, "frame_id", _count(self.frame_id, name="frame_id"))
        confidence = _finite(self.confidence, name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "footprint_clearance_m",
            _nonnegative(self.footprint_clearance_m, name="footprint_clearance_m"),
        )
        object.__setattr__(self, "uncertainty_m", _nonnegative(self.uncertainty_m, name="uncertainty_m"))
        if self.route_cost_m is not None:
            object.__setattr__(self, "route_cost_m", _nonnegative(self.route_cost_m, name="route_cost_m"))

    def failure_reasons(self, *, min_clearance_m: float = 0.0, min_confidence: float = 0.0) -> tuple[str, ...]:
        """Return stable reason codes for failed hard checks."""

        min_clearance = _nonnegative(min_clearance_m, name="min_clearance_m")
        threshold = _finite(min_confidence, name="min_confidence")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        reasons: list[str] = []
        if self.confidence < threshold:
            reasons.append("low_confidence")
        if not self.identity_stable:
            reasons.append("identity_unstable")
        if not self.hard_constraints_satisfied:
            reasons.append("hard_constraint_failed")
        if not self.free_endpoint:
            reasons.append("endpoint_blocked")
        if not self.reachable:
            reasons.append("route_not_reachable")
        if not self.orientation_feasible:
            reasons.append("orientation_infeasible")
        if self.blocking:
            reasons.append("candidate_blocked")
        if self.contradiction:
            reasons.append("contradictory_evidence")
        if self.footprint_clearance_m < min_clearance:
            reasons.append("insufficient_clearance")
        return tuple(reasons)

    def is_feasible(self, *, min_clearance_m: float = 0.0, min_confidence: float = 0.0) -> bool:
        return not self.failure_reasons(
            min_clearance_m=min_clearance_m,
            min_confidence=min_confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneState:
    """Frame-local physical state consumed by scene-agnostic rules."""

    frame_id: int
    robot_position: Point3
    robot_yaw_rad: float = 0.0
    robot_radius_m: float = 0.45
    graph_nodes: int = 0
    graph_edges: int = 0
    obstacle_count: int = 0
    candidate_count: int = 0
    progress_delta_m: float = 0.0
    route_valid: bool | None = None
    stale_control_ticks: int = 0
    frame_transform: FrameTransform | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _count(self.frame_id, name="frame_id"))
        object.__setattr__(self, "robot_position", _point3(self.robot_position, name="robot_position"))
        object.__setattr__(self, "robot_yaw_rad", _finite(self.robot_yaw_rad, name="robot_yaw_rad"))
        object.__setattr__(self, "robot_radius_m", _nonnegative(self.robot_radius_m, name="robot_radius_m"))
        for field_name in ("graph_nodes", "graph_edges", "obstacle_count", "candidate_count", "stale_control_ticks"):
            object.__setattr__(self, field_name, _count(getattr(self, field_name), name=field_name))
        object.__setattr__(self, "progress_delta_m", _finite(self.progress_delta_m, name="progress_delta_m"))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.frame_transform is not None:
            value["frame_transform"] = self.frame_transform.to_dict()
        return value


@dataclass(frozen=True)
class ControlReference:
    """Bounded control lease or an explicit policy fallback reference."""

    mode: ControlMode | str = ControlMode.POLICY
    candidate_id: str | None = None
    goal_pose: Point3 | None = None
    waypoint: Point3 | None = None
    goal_role: GoalRole | str = GoalRole.NONE
    issued_frame: int = 0
    valid_until_frame: int = 0
    route_valid: bool = False
    clearance_m: float = 0.0
    progress_delta_m: float = 0.0
    reason: str = "policy_default"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, ControlMode) else ControlMode(str(self.mode))
        role = self.goal_role if isinstance(self.goal_role, GoalRole) else GoalRole(str(self.goal_role))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "goal_role", role)
        if self.candidate_id is not None:
            candidate_id = str(self.candidate_id).strip()
            if not candidate_id:
                raise ValueError("candidate_id must be non-empty when provided")
            object.__setattr__(self, "candidate_id", candidate_id)
        if self.goal_pose is not None:
            object.__setattr__(self, "goal_pose", _point3(self.goal_pose, name="goal_pose"))
        if self.waypoint is not None:
            object.__setattr__(self, "waypoint", _point3(self.waypoint, name="waypoint"))
        issued = _count(self.issued_frame, name="issued_frame")
        valid_until = _count(self.valid_until_frame, name="valid_until_frame")
        if valid_until < issued:
            raise ValueError("valid_until_frame must be greater than or equal to issued_frame")
        object.__setattr__(self, "issued_frame", issued)
        object.__setattr__(self, "valid_until_frame", valid_until)
        object.__setattr__(self, "clearance_m", _nonnegative(self.clearance_m, name="clearance_m"))
        object.__setattr__(self, "progress_delta_m", _finite(self.progress_delta_m, name="progress_delta_m"))
        if not str(self.reason).strip():
            raise ValueError("reason must be non-empty")
        object.__setattr__(self, "reason", str(self.reason))
        if self.fallback_reason is not None:
            fallback_reason = str(self.fallback_reason).strip()
            if not fallback_reason:
                raise ValueError("fallback_reason must be non-empty when provided")
            object.__setattr__(self, "fallback_reason", fallback_reason)
        # A PMG lease must identify what it controls.  Policy/fallback are
        # allowed to carry no pose and therefore remain safe defaults.
        if mode in (ControlMode.PMG_CHECKPOINT, ControlMode.PMG_TERMINAL) and self.candidate_id is None:
            raise ValueError("PMG control modes require candidate_id")
        if mode is ControlMode.PMG_CHECKPOINT and role is not GoalRole.CHECKPOINT:
            raise ValueError("pmg_checkpoint mode requires checkpoint goal_role")
        if mode is ControlMode.PMG_TERMINAL and role is not GoalRole.FINAL_GOAL:
            raise ValueError("pmg_terminal mode requires final_goal goal_role")

    def is_active(self, frame_id: int) -> bool:
        """Return whether the reference is inside its finite lease window."""

        current = _count(frame_id, name="frame_id")
        return self.issued_frame <= current <= self.valid_until_frame

    @property
    def is_pmg_owned(self) -> bool:
        return self.mode in (ControlMode.PMG_CHECKPOINT, ControlMode.PMG_TERMINAL)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        value["goal_role"] = self.goal_role.value
        value["is_pmg_owned"] = self.is_pmg_owned
        return value


__all__ = [
    "CandidateEvidence",
    "ControlMode",
    "ControlReference",
    "FrameTransform",
    "GoalRole",
    "Point3",
    "SceneState",
]
