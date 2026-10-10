"""Fixed Stage 2 PMG tool scheduler."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDecision:
    tool: str
    reason: str
    cost: float


class HeuristicToolScheduler:
    def __init__(self, query_threshold: float = 0.55, margin_threshold: float = 0.15, conflict_threshold: float = 0.5) -> None:
        self.query_threshold = query_threshold
        self.margin_threshold = margin_threshold
        self.conflict_threshold = conflict_threshold

    def decide(
        self,
        top1_confidence: float,
        top2_confidence: float,
        contradiction_probability: float,
        hard_constraints_satisfied: bool,
    ) -> ToolDecision:
        if top1_confidence < self.query_threshold:
            return ToolDecision("QUERY_CANDIDATES", "top candidate confidence below threshold", 1.0)
        if top1_confidence - top2_confidence < self.margin_threshold:
            return ToolDecision("VERIFY_CONSTRAINT", "top candidate margin is ambiguous", 1.5)
        if contradiction_probability > self.conflict_threshold:
            return ToolDecision("BACKTRACK_EVIDENCE", "conflicting evidence exceeds threshold", 2.0)
        if hard_constraints_satisfied:
            return ToolDecision("COMMIT_GOAL", "all hard constraints are satisfied", 0.25)
        return ToolDecision("QUERY_CANDIDATES", "hard constraints remain unresolved", 1.0)
