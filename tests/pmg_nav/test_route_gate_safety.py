from __future__ import annotations

import re

from pmg_nav.control_gate import POLICY_ONLY, decide_control_gate
from pmg_nav.planner import ObstacleAwareRoutePlanner


def test_numbered_usd_floor_mesh_is_filtered_as_support_surface() -> None:
    normalized = re.sub(r"[_-]+", " ", "/Root/SM_floor27/SM_floor02".lower())
    tokens = set(re.findall(r"[a-z]+|[0-9]+", normalized))

    assert ObstacleAwareRoutePlanner._is_passable_ground_surface(tokens, -1e-6, 1e-6)
    assert not ObstacleAwareRoutePlanner._is_passable_ground_surface(tokens, -1e-6, 0.20)


def test_terminal_m1_falls_back_to_policy_when_route_is_unreachable() -> None:
    decision = decide_control_gate(
        method="M1",
        variant="simple_oracle",
        candidate_valid=True,
        candidate_distance_m=3.5,
        require_valid_route=True,
        route_valid=False,
    )

    assert decision.scope == POLICY_ONLY
    assert not decision.allow_override
    assert decision.reason == "route_infeasible_policy_fallback"
