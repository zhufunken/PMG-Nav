"""PMG adapters used at the TIC-VLA wrapper boundary.

``build_pmg_context`` is a textual diagnostic context and is not inserted into
the released model in the strict benchmark. ``PMGGoalAdapter`` is retained as
an optional deterministic bridge; Stage 2 defaults to direct goal-condition
serialization and does not instantiate it. A learned/projected adapter is
reserved for Stage 3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .constraints import ConstraintProgram
from .goal_field import GoalCandidate
from .graph import PhysicalMemoryGraph


@dataclass(frozen=True)
class PMGGoalCondition:
    """Small structured PMG readout consumed by the Stage 2 controller."""

    world_goal: tuple[float, float, float]
    delta_world: tuple[float, float]
    distance_m: float
    yaw: float
    score: float
    hard_valid: bool
    stable: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_environment(self) -> dict[str, str]:
        return {
            "TICVLA_PMG_GOAL_X": str(float(self.world_goal[0])),
            "TICVLA_PMG_GOAL_Y": str(float(self.world_goal[1])),
            "TICVLA_PMG_GOAL_Z": str(float(self.world_goal[2])),
            "TICVLA_PMG_GOAL_DISTANCE": str(float(self.distance_m)),
            "TICVLA_PMG_GOAL_YAW": str(float(self.yaw)),
            "TICVLA_PMG_SCORE": str(float(self.score)),
            "TICVLA_PMG_HARD_VALID": "1" if self.hard_valid else "0",
            "TICVLA_PMG_STABLE": "1" if self.stable else "0",
        }


class PMGGoalAdapter:
    """Optional deterministic bridge from a goal candidate to control fields.

    This class has no trainable weights and is not required by Stage 2. It is
    kept for explicit compatibility/ablation runs and for a future learned
    Stage 3 adapter boundary.
    """

    @staticmethod
    def encode(
        candidate: GoalCandidate,
        robot_position: tuple[float, float, float],
        *,
        stable: bool,
    ) -> PMGGoalCondition:
        dx = float(candidate.position[0]) - float(robot_position[0])
        dy = float(candidate.position[1]) - float(robot_position[1])
        return PMGGoalCondition(
            world_goal=tuple(map(float, candidate.position)),
            delta_world=(dx, dy),
            distance_m=math.hypot(dx, dy),
            yaw=float(candidate.yaw),
            score=float(candidate.score),
            hard_valid=bool(candidate.hard_valid),
            stable=bool(stable),
        )


def build_pmg_context(
    instruction: str,
    program: ConstraintProgram,
    graph: PhysicalMemoryGraph,
    candidates: list[GoalCandidate],
) -> str:
    labels = [anchor for item in program.constraints for anchor in item.anchors]
    nodes_by_id = [graph.objects[label] for label in labels if label in graph.objects]
    nodes = list({node.node_id: node for node in [*nodes_by_id, *graph.retrieve(labels, limit=8)]}.values())
    # Keep the original route instruction authoritative. The previous wording
    # sounded like a second command and could make the VLA abandon a multi-turn
    # route in favor of a noisy local candidate.
    if not nodes and not candidates:
        return instruction
    node_text = "; ".join(
        f"{node.label} at ({node.position[0]:.2f},{node.position[1]:.2f}) confidence {node.confidence:.2f}"
        for node in nodes
    )
    goal_text = "; ".join(
        f"goal{i + 1}=({item.position[0]:.2f},{item.position[1]:.2f}),score={item.score:.2f}"
        for i, item in enumerate(candidates[:3])
    )
    return (
        f"{instruction}\n"
        "[PMG optional memory hint; preserve the route instruction above and use "
        "this only to resolve an ambiguous target] "
        f"observed anchors: {node_text or 'none'}; candidate poses: {goal_text or 'none'}."
    )
