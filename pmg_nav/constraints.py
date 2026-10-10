"""Typed constraint programs and deterministic geometric evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Iterable, Mapping


RELATIONS = {"near", "far", "left", "right", "front", "behind", "between", "distance", "facing", "reachable", "non_blocking"}


@dataclass(frozen=True)
class Constraint:
    relation: str
    anchors: tuple[str, ...] = ()
    value: float | None = None
    tolerance: float = 0.35
    hard: bool = False

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"unsupported relation: {self.relation}")
        if self.relation == "between" and len(self.anchors) != 2:
            raise ValueError("between requires exactly two anchors")
        if self.relation not in {"reachable", "non_blocking"} and not self.anchors:
            raise ValueError(f"{self.relation} requires an anchor")
        if self.relation == "distance" and (self.value is None or self.value < 0):
            raise ValueError("distance requires a non-negative value")


@dataclass(frozen=True)
class ConstraintProgram:
    instruction: str
    constraints: tuple[Constraint, ...]
    target: str = "robot"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "target": self.target,
            "constraints": [asdict(item) for item in self.constraints],
            "metadata": dict(self.metadata),
        }


class ConstraintCompiler:
    """Compile controlled English instructions into a typed program.

    Stage 2 uses generated templates as the authoritative path. Unknown text is
    rejected explicitly instead of silently producing an invalid goal.
    """

    _distance = re.compile(
        r"(?:stop|stand|go)\s+(?P<distance>\d+(?:\.\d+)?)\s*(?:m|meter|meters)\s+"
        r"(?P<relation>left of|right of|in front of|behind|from)\s+(?:the\s+)?(?P<anchor>[\w -]+?)(?:[.,]|\s+and\s+|$)",
        re.IGNORECASE,
    )
    _simple = re.compile(
        r"(?:stop|stand|go|navigate)\s+(?P<relation>near|far from|left of|right of|in front of|behind)\s+"
        r"(?:the\s+)?(?P<anchor>[\w -]+?)(?:[.,]|\s+and\s+|$)",
        re.IGNORECASE,
    )
    _between = re.compile(
        r"between\s+(?:the\s+)?(?P<a>[\w -]+?)\s+and\s+(?:the\s+)?(?P<b>[\w -]+?)(?:[.,]|$)",
        re.IGNORECASE,
    )

    _relation_map = {
        "far from": "far",
        "left of": "left",
        "right of": "right",
        "in front of": "front",
        "from": "distance",
    }

    def compile(self, instruction: str) -> ConstraintProgram:
        constraints: list[Constraint] = []
        between = self._between.search(instruction)
        if between:
            constraints.append(Constraint("between", (self._clean(between["a"]), self._clean(between["b"]))))
        for match in self._distance.finditer(instruction):
            relation = self._relation_map.get(match["relation"].lower(), match["relation"].lower())
            anchor = self._clean(match["anchor"])
            distance = float(match["distance"])
            constraints.append(Constraint("distance", (anchor,), distance, max(0.2, distance * 0.15)))
            if relation != "distance":
                constraints.append(Constraint(relation, (anchor,)))
        for match in self._simple.finditer(instruction):
            relation = self._relation_map.get(match["relation"].lower(), match["relation"].lower())
            constraints.append(Constraint(relation, (self._clean(match["anchor"]),)))
        lowered = instruction.lower()
        if "reachable" in lowered:
            constraints.append(Constraint("reachable", hard=True))
        if "do not block" in lowered or "non-blocking" in lowered:
            constraints.append(Constraint("non_blocking", hard=True))
        if "facing" in lowered:
            facing = re.search(r"facing\s+(?:the\s+)?([\w -]+?)(?:[.,]|$)", instruction, re.I)
            if facing:
                constraints.append(Constraint("facing", (self._clean(facing.group(1)),)))
        unique = tuple(dict.fromkeys(constraints))
        if not unique:
            raise ValueError(f"instruction does not match Stage 2 grammar: {instruction!r}")
        return ConstraintProgram(instruction, unique)

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower().replace("_", " "))


@dataclass(frozen=True)
class ConstraintScore:
    satisfied: bool
    score: float
    residual: float


class ConstraintEvaluator:
    def evaluate(
        self,
        candidate: tuple[float, float, float],
        constraint: Constraint,
        anchors: Mapping[str, tuple[float, float, float]],
        *,
        yaw: float = 0.0,
        reachable: bool = True,
        blocking: bool = False,
    ) -> ConstraintScore:
        relation = constraint.relation
        if relation == "reachable":
            return ConstraintScore(reachable, 1.0 if reachable else 0.0, 0.0 if reachable else 1.0)
        if relation == "non_blocking":
            return ConstraintScore(not blocking, 0.0 if blocking else 1.0, 1.0 if blocking else 0.0)
        points = [anchors[name] for name in constraint.anchors]
        if relation == "between":
            midpoint = tuple((points[0][i] + points[1][i]) / 2.0 for i in range(3))
            residual = self._distance(candidate, midpoint)
            scale = max(0.5, self._distance(points[0], points[1]) * 0.35)
            score = math.exp(-residual / scale)
            return ConstraintScore(score >= 0.5, score, residual)
        anchor = points[0]
        dx, dy = candidate[0] - anchor[0], candidate[1] - anchor[1]
        distance = self._distance(candidate, anchor)
        if relation == "distance":
            residual = abs(distance - float(constraint.value))
            score = math.exp(-residual / max(constraint.tolerance, 1e-6))
            return ConstraintScore(residual <= constraint.tolerance, score, residual)
        if relation in {"near", "far"}:
            threshold = constraint.value if constraint.value is not None else (2.0 if relation == "near" else 4.0)
            satisfied = distance <= threshold if relation == "near" else distance >= threshold
            signed = threshold - distance if relation == "near" else distance - threshold
            return ConstraintScore(satisfied, 1.0 / (1.0 + math.exp(-signed)), abs(signed))
        if relation in {"left", "right", "front", "behind"}:
            margin = {"left": -dx, "right": dx, "front": dy, "behind": -dy}[relation]
            score = 1.0 / (1.0 + math.exp(-2.0 * margin))
            return ConstraintScore(margin > 0.0, score, abs(margin))
        if relation == "facing":
            desired = math.atan2(anchor[1] - candidate[1], anchor[0] - candidate[0])
            residual = abs(math.atan2(math.sin(yaw - desired), math.cos(yaw - desired)))
            score = math.exp(-residual)
            return ConstraintScore(residual <= math.radians(30), score, residual)
        raise AssertionError(relation)

    def evaluate_program(
        self,
        candidate: tuple[float, float, float],
        program: ConstraintProgram,
        anchors: Mapping[str, tuple[float, float, float]],
        **state: Any,
    ) -> list[ConstraintScore]:
        return [self.evaluate(candidate, item, anchors, **state) for item in program.constraints]

    @staticmethod
    def _distance(a: Iterable[float], b: Iterable[float]) -> float:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
