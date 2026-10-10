"""Candidate sampling and constraint-aware physical goal scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

from .constraints import ConstraintEvaluator, ConstraintProgram
from .graph import PhysicalMemoryGraph


@dataclass
class GoalCandidate:
    position: tuple[float, float, float]
    yaw: float
    score: float = 0.0
    hard_valid: bool = True
    satisfaction: list[float] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)


class PhysicalGoalField:
    def __init__(self, radial_samples: int = 16, radii: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0)) -> None:
        self.radial_samples = radial_samples
        self.radii = radii
        self.evaluator = ConstraintEvaluator()

    def sample_and_rank(
        self,
        graph: PhysicalMemoryGraph,
        program: ConstraintProgram,
        *,
        top_k: int = 5,
        reachable_fn: Callable[[tuple[float, float, float]], bool] | None = None,
        blocking_fn: Callable[[tuple[float, float, float]], bool] | None = None,
    ) -> list[GoalCandidate]:
        anchors = graph.anchors()
        relevant = [anchors[name] for item in program.constraints for name in item.anchors if name in anchors]
        if not relevant:
            raise KeyError("program anchors are absent from PMG")
        centers = list(dict.fromkeys(relevant))
        if any(item.relation == "between" for item in program.constraints):
            for item in program.constraints:
                if item.relation == "between":
                    a, b = (anchors[name] for name in item.anchors)
                    centers.append(tuple((a[i] + b[i]) / 2.0 for i in range(3)))
        candidates: list[GoalCandidate] = []
        for center in centers:
            candidates.append(GoalCandidate(center, 0.0))
            for radius in self.radii:
                for index in range(self.radial_samples):
                    angle = 2.0 * math.pi * index / self.radial_samples
                    point = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), center[2])
                    candidates.append(GoalCandidate(point, math.atan2(center[1] - point[1], center[0] - point[0])))
        for candidate in candidates:
            reachable = reachable_fn(candidate.position) if reachable_fn else True
            blocking = blocking_fn(candidate.position) if blocking_fn else False
            scores = self.evaluator.evaluate_program(
                candidate.position,
                program,
                anchors,
                yaw=candidate.yaw,
                reachable=reachable,
                blocking=blocking,
            )
            candidate.satisfaction = [item.score for item in scores]
            candidate.hard_valid = all(score.satisfied for constraint, score in zip(program.constraints, scores) if constraint.hard)
            candidate.score = sum(candidate.satisfaction) / len(candidate.satisfaction) if candidate.hard_valid else 0.0
            candidate.evidence = {"reachable": reachable, "blocking": blocking}
        candidates.sort(key=lambda item: (item.hard_valid, item.score), reverse=True)
        return candidates[:top_k]
