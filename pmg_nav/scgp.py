"""Opt-in scene-agnostic constraint and progress executor.

This module is the smallest executable form of the SCGP design.  It does not
run a model, inspect a benchmark goal, or know a scene name.  A caller gives
it a structured candidate and physical evidence; the executor decides whether
PMG may own a short control lease or whether TIC-VLA should keep control.

The Stage 2 runtime imports this module only behind ``PMG_SCGP_ENABLED=1``.
Keeping the gate explicit lets paired traces validate the state machine without
changing the frozen M1 experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from .contracts import (
    CandidateEvidence,
    ControlMode,
    ControlReference,
    GoalRole,
    SceneState,
)


@dataclass(frozen=True)
class SCGPConfig:
    """Fixed, scene-independent safety and lease parameters."""

    stable_frames: int = 2
    lease_frames: int = 8
    min_confidence: float = 0.5
    clearance_scale: float = 1.0
    progress_epsilon_m: float = 0.02
    max_no_progress_ticks: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.stable_frames, bool) or not math.isfinite(float(self.stable_frames)) or int(self.stable_frames) != self.stable_frames or self.stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        if isinstance(self.lease_frames, bool) or not math.isfinite(float(self.lease_frames)) or int(self.lease_frames) != self.lease_frames or self.lease_frames < 1:
            raise ValueError("lease_frames must be positive")
        if not math.isfinite(float(self.min_confidence)) or not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between zero and one")
        if not math.isfinite(float(self.clearance_scale)) or self.clearance_scale < 0.0:
            raise ValueError("clearance_scale must be non-negative")
        if not math.isfinite(float(self.progress_epsilon_m)) or self.progress_epsilon_m < 0.0:
            raise ValueError("progress_epsilon_m must be non-negative")
        if isinstance(self.max_no_progress_ticks, bool) or not math.isfinite(float(self.max_no_progress_ticks)) or int(self.max_no_progress_ticks) != self.max_no_progress_ticks or self.max_no_progress_ticks < 1:
            raise ValueError("max_no_progress_ticks must be positive")


@dataclass(frozen=True)
class SCGPTrace:
    """Auditable result of one executor decision."""

    frame_id: int
    candidate_id: str | None
    mode: str
    reason: str
    stable_count: int
    no_progress_ticks: int
    lease_until_frame: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "reason": self.reason,
            "stable_count": self.stable_count,
            "no_progress_ticks": self.no_progress_ticks,
            "lease_until_frame": self.lease_until_frame,
        }


class SceneInvariantExecutor:
    """Fail-closed, bounded PMG control hand-off state machine.

    The same equations and thresholds are used for every scene.  The only
    scene-dependent values are measured by the caller and placed in
    :class:`SceneState` or :class:`CandidateEvidence`.
    """

    def __init__(self, config: SCGPConfig | None = None) -> None:
        self.config = config or SCGPConfig()
        self._stable_candidate_id: str | None = None
        self._stable_count = 0
        self._no_progress_ticks = 0
        self._last_frame = -1
        self._lease_candidate_id: str | None = None
        self._lease_until_frame = -1
        self._lease_expired_candidate_id: str | None = None
        self._last_trace = SCGPTrace(0, None, ControlMode.POLICY.value, "init", 0, 0, 0)

    @property
    def last_trace(self) -> SCGPTrace:
        return self._last_trace

    def reset(self) -> None:
        self._stable_candidate_id = None
        self._stable_count = 0
        self._no_progress_ticks = 0
        self._last_frame = -1
        self._lease_candidate_id = None
        self._lease_until_frame = -1
        self._lease_expired_candidate_id = None
        self._last_trace = SCGPTrace(0, None, ControlMode.POLICY.value, "reset", 0, 0, 0)

    def _trace(self, frame_id: int, candidate_id: str | None, mode: str, reason: str, lease_until: int) -> None:
        self._last_trace = SCGPTrace(
            frame_id,
            candidate_id,
            mode,
            reason,
            self._stable_count,
            self._no_progress_ticks,
            lease_until,
        )

    def _fallback(self, scene: SceneState, candidate_id: str | None, reason: str) -> ControlReference:
        self._trace(scene.frame_id, candidate_id, ControlMode.FALLBACK.value, reason, scene.frame_id)
        return ControlReference(
            mode=ControlMode.FALLBACK,
            candidate_id=candidate_id,
            issued_frame=scene.frame_id,
            valid_until_frame=scene.frame_id,
            route_valid=bool(scene.route_valid),
            progress_delta_m=scene.progress_delta_m,
            reason=reason,
            fallback_reason=reason,
        )

    def _update_stability(self, candidate_id: str) -> None:
        if candidate_id == self._stable_candidate_id:
            self._stable_count += 1
        else:
            self._stable_candidate_id = candidate_id
            self._stable_count = 1
            self._lease_candidate_id = None
            self._lease_until_frame = -1
            self._lease_expired_candidate_id = None

    def _update_progress(self, progress_delta_m: float) -> None:
        if progress_delta_m < -self.config.progress_epsilon_m:
            self._no_progress_ticks += 1
        elif progress_delta_m <= self.config.progress_epsilon_m:
            self._no_progress_ticks += 1
        else:
            self._no_progress_ticks = 0

    def decide(
        self,
        scene: SceneState,
        *,
        candidate_id: str,
        goal_pose: Sequence[float],
        waypoint: Sequence[float] | None,
        goal_role: GoalRole | str,
        evidence: CandidateEvidence,
        terminal_verified: bool = False,
    ) -> ControlReference:
        """Return a policy fallback or a finite PMG control reference.

        `terminal_verified` is intentionally supplied by an evaluator-neutral
        verifier (position, orientation, velocity and identity).  It is never
        inferred from benchmark success or from position alone.
        """

        candidate_key = str(candidate_id).strip()
        if not candidate_key:
            return self._fallback(scene, None, "INVALID_CANDIDATE_ID")
        if scene.frame_id <= self._last_frame:
            return self._fallback(scene, candidate_key, "FRAME_ORDER_ERROR")
        self._last_frame = scene.frame_id
        self._update_progress(scene.progress_delta_m)
        self._update_stability(candidate_key)

        transform = scene.frame_transform
        if transform is not None and not transform.round_trip_ok:
            return self._fallback(scene, candidate_key, "FRAME_ERROR")
        if self._no_progress_ticks >= self.config.max_no_progress_ticks:
            return self._fallback(scene, candidate_key, "NO_PROGRESS")
        if not evidence.is_feasible(
            min_clearance_m=scene.robot_radius_m * self.config.clearance_scale,
            min_confidence=self.config.min_confidence,
        ):
            reasons = evidence.failure_reasons(
                min_clearance_m=scene.robot_radius_m * self.config.clearance_scale,
                min_confidence=self.config.min_confidence,
            )
            return self._fallback(scene, candidate_key, reasons[0].upper() if reasons else "INFEASIBLE_CANDIDATE")
        # A positive candidate flag is not enough: the current route must be
        # explicitly validated in this frame.  Unknown route state fails
        # closed so stale geometry cannot obtain a control lease.
        if scene.route_valid is not True:
            return self._fallback(scene, candidate_key, "ROUTE_UNREACHABLE")
        if self._stable_count < self.config.stable_frames:
            return self._fallback(scene, candidate_key, "IDENTITY_UNSTABLE")

        try:
            role = goal_role if isinstance(goal_role, GoalRole) else GoalRole(str(goal_role))
        except ValueError:
            return self._fallback(scene, candidate_key, "UNSUPPORTED_GOAL_ROLE")
        if role is GoalRole.FINAL_GOAL and not terminal_verified:
            return self._fallback(scene, candidate_key, "TERMINAL_NOT_VERIFIED")
        if role is GoalRole.CHECKPOINT:
            mode = ControlMode.PMG_CHECKPOINT
            reason = "STABLE_REACHABLE_CHECKPOINT"
        elif role is GoalRole.FINAL_GOAL:
            mode = ControlMode.PMG_TERMINAL
            reason = "VERIFIED_FINAL_GOAL"
        else:
            return self._fallback(scene, candidate_key, "UNSUPPORTED_GOAL_ROLE")

        # A lease is finite across frames. Do not silently renew it every
        # callback for the same candidate: after expiry policy control must
        # regain ownership until a new candidate identity is observed.
        if (
            self._lease_candidate_id == candidate_key
            and scene.frame_id > self._lease_until_frame
        ):
            self._lease_expired_candidate_id = candidate_key
            self._lease_candidate_id = None
        if self._lease_expired_candidate_id == candidate_key:
            return self._fallback(scene, candidate_key, "LEASE_EXPIRED")
        if self._lease_candidate_id == candidate_key and scene.frame_id <= self._lease_until_frame:
            valid_until = self._lease_until_frame
        else:
            valid_until = scene.frame_id + self.config.lease_frames
            self._lease_candidate_id = candidate_key
            self._lease_until_frame = valid_until
        try:
            reference = ControlReference(
                mode=mode,
                candidate_id=candidate_key,
                goal_pose=goal_pose,
                waypoint=waypoint,
                goal_role=role,
                issued_frame=scene.frame_id,
                valid_until_frame=valid_until,
                route_valid=True,
                clearance_m=evidence.footprint_clearance_m,
                progress_delta_m=scene.progress_delta_m,
                reason=reason,
            )
        except ValueError:
            return self._fallback(scene, candidate_key, "INVALID_CONTROL_REFERENCE")
        self._trace(scene.frame_id, candidate_key, mode.value, reason, valid_until)
        return reference


__all__ = ["SCGPConfig", "SCGPTrace", "SceneInvariantExecutor"]
