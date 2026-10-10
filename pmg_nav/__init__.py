"""Physical Memory Graph navigation primitives."""

from .constraints import Constraint, ConstraintCompiler, ConstraintEvaluator, ConstraintProgram
from .goal_field import GoalCandidate, PhysicalGoalField
from .graph import Evidence, ObjectNode, PhysicalMemoryGraph
from .adapter import PMGGoalAdapter, PMGGoalCondition
from .tools import HeuristicToolScheduler, ToolDecision
from .diagnostics import DiagnosticContract, resolve_contract, stable_checksum, sequence_checksum
from .planner import ObstacleAABB, ObstacleAwareRoutePlanner, RoutePlan
from .control_gate import (
    GLOBAL,
    POLICY_ONLY,
    RECOVERY,
    TERMINAL,
    ControlGateDecision,
    decide_control_gate,
    normalize_control_policy,
    terminal_hysteresis,
)
from .scene_adapter import (
    CandidateDecision,
    SceneAdaptationProfile,
    SceneGraphAdapter,
    SceneStatistics,
)
from .contracts import (
    CandidateEvidence,
    ControlMode,
    ControlReference,
    FrameTransform,
    GoalRole,
    SceneState,
)
from .scgp import SCGPConfig, SCGPTrace, SceneInvariantExecutor

__all__ = [
    "Constraint",
    "ConstraintCompiler",
    "ConstraintEvaluator",
    "ConstraintProgram",
    "Evidence",
    "GoalCandidate",
    "HeuristicToolScheduler",
    "ObjectNode",
    "PhysicalGoalField",
    "PhysicalMemoryGraph",
    "PMGGoalAdapter",
    "PMGGoalCondition",
    "ToolDecision",
    "DiagnosticContract",
    "resolve_contract",
    "stable_checksum",
    "sequence_checksum",
    "ObstacleAABB",
    "ObstacleAwareRoutePlanner",
    "RoutePlan",
    "GLOBAL",
    "POLICY_ONLY",
    "RECOVERY",
    "TERMINAL",
    "ControlGateDecision",
    "decide_control_gate",
    "normalize_control_policy",
    "terminal_hysteresis",
    "CandidateDecision",
    "SceneAdaptationProfile",
    "SceneGraphAdapter",
    "SceneStatistics",
    "CandidateEvidence",
    "ControlMode",
    "ControlReference",
    "FrameTransform",
    "GoalRole",
    "SceneState",
    "SCGPConfig",
    "SCGPTrace",
    "SceneInvariantExecutor",
]
