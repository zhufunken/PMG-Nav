"""Deterministic scene-adaptation layer for PMG.

This module is deliberately non-learned and scene-name agnostic.  It converts
observable geometry/control statistics into a bounded operating profile.  The
profile can be used by an experiment runner to choose graph budgets, candidate
filters, and control hand-off guards without training a separate policy for
each scene.

The adapter is an opt-in policy component.  It does not inspect evaluator
success, benchmark goal coordinates, or scene names, and it never changes M0
unless a caller explicitly enables it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _distance_xy(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(_finite(a[0]) - _finite(b[0]), _finite(a[1]) - _finite(b[1]))


@dataclass(frozen=True)
class SceneStatistics:
    """Observable, evaluator-independent scene/control summary."""

    graph_nodes: int
    graph_edges: int
    anchor_nodes: int
    spatial_span_m: float
    mean_node_distance_m: float
    obstacle_count: int
    candidate_count: int
    reachable_candidate_count: int
    valid_candidate_count: int
    nearest_candidate_distance_m: float | None
    route_valid: bool | None
    progress_delta_m: float
    stale_control_ticks: int

    @property
    def relation_density(self) -> float:
        denominator = max(1, self.graph_nodes * max(1, self.anchor_nodes))
        return min(1.0, self.graph_edges / denominator)

    @property
    def reachable_fraction(self) -> float:
        return self.reachable_candidate_count / max(1, self.candidate_count)

    @property
    def complexity_score(self) -> float:
        """A bounded complexity score, not a learned confidence."""

        node_term = min(1.0, self.graph_nodes / 16.0)
        edge_term = min(1.0, self.relation_density)
        span_term = min(1.0, self.spatial_span_m / 30.0)
        obstacle_term = min(1.0, self.obstacle_count / 32.0)
        return 0.30 * node_term + 0.25 * edge_term + 0.20 * span_term + 0.25 * obstacle_term

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["relation_density"] = self.relation_density
        value["reachable_fraction"] = self.reachable_fraction
        value["complexity_score"] = self.complexity_score
        return value


@dataclass(frozen=True)
class SceneAdaptationProfile:
    """Bounded operating choices derived from :class:`SceneStatistics`."""

    profile: str
    graph_node_budget: int
    edges_per_anchor: int
    candidate_limit: int
    require_reachable_route: bool
    allow_terminal_override: bool
    terminal_enter_distance_m: float
    terminal_exit_distance_m: float
    minimum_progress_m: float
    max_stale_control_ticks: int
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateDecision:
    """Reasoned result for one candidate before it reaches the control gate."""

    accepted: bool
    reason: str
    candidate_distance_m: float | None
    route_valid: bool


class SceneGraphAdapter:
    """Choose conservative, geometry-driven PMG operating parameters.

    The rules intentionally use wide bounded ranges.  They are not intended
    to be tuned independently for each environment; the same equations are
    applied to every scene and can be evaluated through a fixed ablation.
    """

    def __init__(
        self,
        *,
        base_node_budget: int = 8,
        max_node_budget: int = 16,
        base_edges_per_anchor: int = 4,
        max_edges_per_anchor: int = 8,
    ) -> None:
        self.base_node_budget = max(2, int(base_node_budget))
        self.max_node_budget = max(self.base_node_budget, int(max_node_budget))
        self.base_edges_per_anchor = max(1, int(base_edges_per_anchor))
        self.max_edges_per_anchor = max(self.base_edges_per_anchor, int(max_edges_per_anchor))

    def measure(
        self,
        *,
        graph_nodes: int,
        graph_edges: int,
        anchor_nodes: int,
        node_positions: Iterable[Sequence[float]] = (),
        robot_position: Sequence[float] = (0.0, 0.0, 0.0),
        obstacle_count: int = 0,
        candidates: Iterable[Any] = (),
        progress_delta_m: float = 0.0,
        stale_control_ticks: int = 0,
        route_valid: bool | None = None,
    ) -> SceneStatistics:
        positions = [tuple(position) for position in node_positions]
        distances = [_distance_xy(position, robot_position) for position in positions]
        span = 0.0
        for left in positions:
            for right in positions:
                span = max(span, _distance_xy(left, right))
        candidate_items = list(candidates)
        valid = [item for item in candidate_items if bool(getattr(item, "hard_valid", True))]
        reachable = [
            item for item in valid
            if bool(getattr(item, "evidence", {}).get("reachable", True))
            and not bool(getattr(item, "evidence", {}).get("blocking", False))
        ]
        candidate_distances = [
            _distance_xy(getattr(item, "position", robot_position), robot_position)
            for item in reachable
        ]
        return SceneStatistics(
            graph_nodes=max(0, int(graph_nodes)),
            graph_edges=max(0, int(graph_edges)),
            anchor_nodes=max(0, int(anchor_nodes)),
            spatial_span_m=span,
            mean_node_distance_m=sum(distances) / len(distances) if distances else 0.0,
            obstacle_count=max(0, int(obstacle_count)),
            candidate_count=len(candidate_items),
            reachable_candidate_count=len(reachable),
            valid_candidate_count=len(valid),
            nearest_candidate_distance_m=min(candidate_distances) if candidate_distances else None,
            route_valid=route_valid,
            # Keep progress signed.  A negative delta is a useful safety
            # signal; clamping it here would make the progress gate unable to
            # distinguish regression from no motion.
            progress_delta_m=_finite(progress_delta_m),
            stale_control_ticks=max(0, int(stale_control_ticks)),
        )

    def adapt(self, stats: SceneStatistics) -> SceneAdaptationProfile:
        """Map statistics to one bounded profile using fixed equations."""

        complexity = stats.complexity_score
        # Extra graph capacity is earned by measured complexity, not by a
        # scene label.  The cap prevents the adapter from recreating a dense
        # all-to-all graph.
        extra_nodes = math.ceil(complexity * (self.max_node_budget - self.base_node_budget))
        extra_edges = math.ceil(complexity * (self.max_edges_per_anchor - self.base_edges_per_anchor))
        node_budget = min(self.max_node_budget, self.base_node_budget + extra_nodes)
        edge_budget = min(self.max_edges_per_anchor, self.base_edges_per_anchor + extra_edges)
        reasons: list[str] = []

        if stats.spatial_span_m >= 18.0:
            profile = "wide_geometry"
            reasons.append("large_observed_span")
        elif stats.obstacle_count >= 16 or stats.relation_density >= 0.55:
            profile = "constrained_geometry"
            reasons.append("dense_or_obstructed_geometry")
        elif stats.reachable_fraction < 0.5 and stats.candidate_count:
            profile = "candidate_uncertain"
            reasons.append("low_reachable_candidate_fraction")
        else:
            profile = "regular_geometry"

        if stats.reachable_fraction < 0.5 and stats.candidate_count:
            reasons.append("route_feasibility_required")
        if stats.stale_control_ticks > 0:
            reasons.append("stale_control_guard")

        # A wider/denser scene gets a slightly larger terminal radius, but the
        # bounded range keeps PMG local and leaves global route control to VLA.
        enter = min(6.0, max(3.0, 4.0 + 0.08 * stats.spatial_span_m))
        exit_radius = min(7.0, enter + 1.0)
        return SceneAdaptationProfile(
            profile=profile,
            graph_node_budget=node_budget,
            edges_per_anchor=edge_budget,
            candidate_limit=min(12, max(3, 4 + edge_budget)),
            require_reachable_route=bool(stats.obstacle_count or stats.candidate_count),
            allow_terminal_override=stats.stale_control_ticks == 0 and stats.progress_delta_m >= 0.0,
            terminal_enter_distance_m=enter,
            terminal_exit_distance_m=exit_radius,
            minimum_progress_m=0.05 if stats.stale_control_ticks == 0 else 0.15,
            max_stale_control_ticks=15 if stats.obstacle_count < 16 else 10,
            reasons=tuple(reasons),
        )

    def evaluate_candidate(
        self,
        candidate: Any,
        *,
        robot_position: Sequence[float],
        profile: SceneAdaptationProfile,
        route_valid: bool,
        progress_delta_m: float,
    ) -> CandidateDecision:
        """Apply invariant checks before allowing PMG control hand-off."""

        position = getattr(candidate, "position", None)
        if not position or len(position) < 2 or any(not math.isfinite(_finite(v, math.nan)) for v in position[:3]):
            return CandidateDecision(False, "non_finite_candidate", None, route_valid)
        distance = _distance_xy(position, robot_position)
        if not bool(getattr(candidate, "hard_valid", False)):
            return CandidateDecision(False, "hard_constraint_failed", distance, route_valid)
        evidence = getattr(candidate, "evidence", {}) or {}
        if evidence.get("blocking", False):
            return CandidateDecision(False, "candidate_blocked", distance, route_valid)
        if profile.require_reachable_route and (not route_valid or not evidence.get("reachable", True)):
            return CandidateDecision(False, "route_not_reachable", distance, route_valid)
        if progress_delta_m < -profile.minimum_progress_m:
            return CandidateDecision(False, "negative_progress", distance, route_valid)
        if not profile.allow_terminal_override:
            return CandidateDecision(False, "stale_control_guard", distance, route_valid)
        return CandidateDecision(True, "accepted", distance, route_valid)


__all__ = [
    "CandidateDecision",
    "SceneAdaptationProfile",
    "SceneGraphAdapter",
    "SceneStatistics",
]
