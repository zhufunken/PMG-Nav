"""Offline checks of the controller code injected into the Isaac behavior."""

from __future__ import annotations

import math
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "autodl/scripts/08_run_dynanav_baseline.sh"


def _injected_function(name: str, start: str, end: str, signature: str, setup: str, result: str):
    """Execute the actual injected source without launching Isaac Sim."""

    script = SCRIPT.read_text(encoding="utf-8")
    assert script.count(start) == 1
    first = script.index(start)
    last = script.index(end, first)
    body = textwrap.dedent(script[first:last])
    source = f"def {name}({signature}):\n    {setup}\n"
    source += textwrap.indent(body, "    ")
    source += f"    return {result}\n"
    namespace = {"math": math, "np": np, "os": os}
    exec(compile(source, str(SCRIPT), "exec"), namespace)
    return namespace[name]


stall_tick = _injected_function(
    "stall_tick",
    "        pmg_recovery_suppressed = False\n",
    "        if (\n            self._pmg_method in {",
    "self, pmg_control_position, route_reason, route_waypoint_active, "
    "simple_oracle_variant=True, pmg_terminal_scope=True, pmg_global_scope=False, "
    "route_alignment_hold=False",
    "pmg_gate_reason = 'terminal_active'",
    "pmg_recovery_suppressed, pmg_gate_reason",
)


detour_output = _injected_function(
    "detour_output",
    "        if (\n            pmg_recovery_suppressed\n",
    "        # Persist one auditable control-source decision per action tick.",
    "self, pmg_recovery_suppressed, pmg_gate_reason, "
    "simple_oracle_variant=True, pmg_terminal_scope=True, pmg_global_scope=False",
    "route_stall_policy_fallback = False; v_max = 1.5; w_max = 1.0; "
    "v_cmd = 1.0; w_cmd = 0.0; pmg_override_applied = False; "
    "pmg_gate_scope = 'policy'; pmg_control_source = 'policy'",
    "pmg_control_source, pmg_gate_reason, pmg_override_applied, v_cmd",
)


route_pre_tick = _injected_function(
    "route_pre_tick",
    "        route_waypoint_key = None\n",
    "        # A direct goal vector cannot reason about an obstacle absent from the",
    "self, route_waypoint_active",
    "pass",
    "route_waypoint_key, route_alignment_hold, progress_epsilon",
)


route_progress_tick = _injected_function(
    "route_progress_tick",
    "            if route_waypoint_active:\n                # During rotate-first alignment",
    "            simple_stop_radius = float(os.getenv(\"PMG_SIMPLE_ORACLE_STOP_RADIUS\"",
    "self, distance, route_waypoint_active, route_waypoint_key, "
    "route_alignment_hold, progress_epsilon, pmg_recovery_suppressed, pmg_gate_reason",
    "pass",
    "pmg_recovery_suppressed, pmg_gate_reason",
)


route_output = _injected_function(
    "route_output",
    "        # A route-progress latch may be raised inside the controller above,",
    "        if (\n            pmg_recovery_suppressed\n            and not route_stall_policy_fallback",
    "self, route_waypoint_active, pmg_recovery_suppressed, pmg_gate_reason, "
    "pmg_global_scope=False",
    "policy_v_cmd = 1.0; policy_w_cmd = 0.0; v_cmd = 0.1; w_cmd = 0.0; "
    "v_max = 1.5; w_max = 1.0; pmg_override_applied = True; "
    "pmg_gate_scope = 'terminal'; pmg_control_source = 'pmg_terminal'",
    "pmg_control_source, pmg_gate_reason, pmg_override_applied, v_cmd, route_stall_policy_fallback",
)


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        _pmg_goal_position=np.array((4.0, 0.0, 0.0)),
        _pmg_route_valid=True,
        _robot_position=np.array((0.0, 0.0, 0.0)),
        _robot_velocity=np.zeros(3),
        _robot_quaternion=(1.0, 0.0, 0.0, 0.0),
        _cmd_v=0.5,
        _pmg_last_control_source="pmg_terminal",
        _pmg_stall_frames=0,
        _pmg_detour_frames=0,
        _pmg_detour_side=1.0,
        _pmg_direct_stall_key=None,
        _pmg_direct_detour_attempted=False,
        _pmg_direct_stall_latched=False,
        _pmg_route_stall_latched=False,
    )


def _route_state() -> SimpleNamespace:
    state = _state()
    state._pmg_route_reason = "astar"
    state._pmg_waypoint_position = np.array((4.0, 0.0, 0.0))
    state._pmg_waypoint_index = 0
    state._pmg_route_aligning = False
    state._pmg_progress_waypoint_key = None
    state._pmg_progress_best_distance = float("inf")
    state._pmg_progress_stall_inferences = 0
    state._pmg_progress_release_count = 0
    state._pmg_progress_release_key = None
    state._pmg_progress_release_best_distance = float("inf")
    state._pmg_progress_release_attempted = False
    state._pmg_route_stall_latched = False
    state._cmd_v = 0.1
    return state


def _route_step(state: SimpleNamespace):
    key, aligning, epsilon = route_pre_tick(state, True)
    suppressed, reason = stall_tick(
        state, state._pmg_waypoint_position, "astar", True,
        route_alignment_hold=aligning,
    )
    if not suppressed:
        distance = float(np.linalg.norm(state._pmg_waypoint_position[:2] - state._robot_position[:2]))
        suppressed, reason = route_progress_tick(
            state, distance, True, key, aligning, epsilon, suppressed, reason,
        )
    source, reason, override, speed, fallback = route_output(state, True, suppressed, reason)
    state._pmg_last_control_source = source
    state._cmd_v = 0.0 if aligning and source == "pmg_terminal" else speed
    state._pmg_route_aligning = aligning
    return source, reason, override, fallback, aligning


class DirectStallGuardTests(unittest.TestCase):
    def test_direct_route_detours_once_then_yields_without_progress(self) -> None:
        state = _state()
        goal = state._pmg_goal_position
        environment = {
            "PMG_ACTION_ENABLED": "1",
            "PMG_STALL_SPEED": "0.05",
            "PMG_STALL_COMMAND_SPEED": "0.16",
            "PMG_STALL_WINDOW_INFERENCES": "8",
            "PMG_DETOUR_INFERENCES": "24",
            "PMG_POLICY_RECOVERY_INFERENCES": "12",
        }
        with patch.dict(os.environ, environment):
            for _ in range(7):
                suppressed, _ = stall_tick(state, goal, "direct_visible", False)
                self.assertFalse(suppressed)
            suppressed, reason = stall_tick(state, goal, "direct_visible", False)
            self.assertTrue(suppressed)
            self.assertTrue(state._pmg_direct_detour_attempted)
            self.assertEqual(state._pmg_detour_frames, 24)

            sources = []
            source, _, _, speed = detour_output(state, suppressed, reason)
            sources.append(source)
            state._pmg_last_control_source, state._cmd_v = source, speed
            for _ in range(24):
                suppressed, reason = stall_tick(state, goal, "direct_visible", False)
                self.assertTrue(suppressed)
                self.assertEqual(state._pmg_stall_frames, 0)
                source, _, _, speed = detour_output(state, suppressed, reason)
                sources.append(source)
                state._pmg_last_control_source, state._cmd_v = source, speed
            self.assertEqual(sources.count("policy"), 12)
            self.assertEqual(sources.count("pmg_detour"), 13)
            self.assertEqual(state._pmg_detour_frames, 0)

            # The previous detour command cannot immediately rearm a stall.
            suppressed, _ = stall_tick(state, goal, "direct_visible", False)
            self.assertFalse(suppressed)
            state._pmg_last_control_source, state._cmd_v = "pmg_terminal", 0.5
            for _ in range(7):
                suppressed, _ = stall_tick(state, goal, "direct_visible", False)
                self.assertFalse(suppressed)
            suppressed, reason = stall_tick(state, goal, "direct_visible", False)
            self.assertTrue(suppressed)
            self.assertEqual(reason, "direct_route_stalled_policy_fallback")
            self.assertTrue(state._pmg_direct_stall_latched)
            self.assertEqual(state._pmg_detour_frames, 0)

            for _ in range(50):
                state._pmg_last_control_source, state._cmd_v = "policy", 1.0
                suppressed, reason = stall_tick(state, goal, "direct_visible", False)
                source, _, override, _ = detour_output(state, suppressed, reason)
                self.assertEqual((source, override), ("policy", False))
                self.assertEqual(reason, "direct_route_stalled_policy_fallback")
                self.assertEqual(state._pmg_detour_frames, 0)

    def test_new_candidate_or_valid_waypoint_route_clears_direct_latch(self) -> None:
        state = _state()
        state._pmg_direct_stall_key = (4.0, 0.0)
        state._pmg_direct_detour_attempted = True
        state._pmg_direct_stall_latched = True
        state._pmg_last_control_source = "policy"
        with patch.dict(os.environ, {"PMG_ACTION_ENABLED": "1"}):
            state._pmg_route_valid = False
            suppressed, reason = stall_tick(state, state._pmg_goal_position, "route_unavailable", False)
            source, _, override, _ = detour_output(state, suppressed, reason)
            self.assertEqual((source, override), ("policy", False))
            self.assertEqual(reason, "direct_route_stalled_policy_fallback")

            state._pmg_route_valid = True
            state._pmg_detour_frames = 7
            state._pmg_goal_position = np.array((5.0, 0.0, 0.0))
            suppressed, _ = stall_tick(state, state._pmg_goal_position, "direct_visible", False)
            self.assertFalse(suppressed)
            self.assertFalse(state._pmg_direct_detour_attempted)
            self.assertFalse(state._pmg_direct_stall_latched)
            self.assertEqual(state._pmg_detour_frames, 0)

            state._pmg_direct_detour_attempted = True
            state._pmg_direct_stall_latched = True
            suppressed, _ = stall_tick(state, state._pmg_goal_position, "astar", True)
            self.assertFalse(suppressed)
            self.assertFalse(state._pmg_direct_stall_latched)

    def test_moving_and_preterminal_ticks_do_not_accumulate_stalls(self) -> None:
        state = _state()
        with patch.dict(os.environ, {"PMG_ACTION_ENABLED": "1"}):
            for _ in range(5):
                stall_tick(state, state._pmg_goal_position, "direct_visible", False)
            self.assertEqual(state._pmg_stall_frames, 5)
            state._robot_velocity = np.array((0.2, 0.0, 0.0))
            stall_tick(state, state._pmg_goal_position, "direct_visible", False)
            self.assertEqual(state._pmg_stall_frames, 0)

            state._pmg_direct_detour_attempted = True
            state._pmg_direct_stall_latched = True
            suppressed, _ = stall_tick(
                state, state._pmg_goal_position, "direct_visible", False,
                pmg_terminal_scope=False,
            )
            self.assertFalse(suppressed)
            self.assertFalse(state._pmg_direct_stall_latched)
            self.assertEqual(state._pmg_detour_frames, 0)

            suppressed, _ = stall_tick(
                state, state._pmg_goal_position, "direct_visible", False,
                simple_oracle_variant=False, pmg_terminal_scope=False,
            )
            self.assertFalse(suppressed)

    def test_invalid_route_cancels_pending_direct_detour(self) -> None:
        state = _state()
        state._pmg_direct_stall_key = (4.0, 0.0)
        state._pmg_direct_detour_attempted = True
        state._pmg_detour_frames = 7
        state._pmg_route_valid = False
        with patch.dict(os.environ, {"PMG_ACTION_ENABLED": "1"}):
            suppressed, reason = stall_tick(state, state._pmg_goal_position, "route_unavailable", False)
            source, _, override, _ = detour_output(state, suppressed, reason)
            self.assertTrue(suppressed)
            self.assertEqual(reason, "route_infeasible_policy_fallback")
            self.assertEqual((source, override), ("policy", False))
            self.assertEqual(state._pmg_detour_frames, 0)


class RouteProgressGuardTests(unittest.TestCase):
    def test_speed_stall_shares_route_release_budget(self) -> None:
        state = _route_state()
        state._pmg_last_control_source = "pmg_terminal"
        state._cmd_v = 0.5
        with patch.dict(os.environ, {
            "PMG_ACTION_ENABLED": "1",
            "PMG_STALL_WINDOW_INFERENCES": "2",
            "PMG_ROUTE_PROGRESS_STALL_INFERENCES": "100",
            "PMG_DETOUR_INFERENCES": "2",
        }):
            self.assertEqual(_route_step(state)[0], "pmg_terminal")
            state._cmd_v = 0.5
            self.assertEqual(_route_step(state)[0], "pmg_detour")
            self.assertTrue(state._pmg_progress_release_attempted)
            self.assertEqual(state._pmg_detour_frames, 2)
            self.assertEqual(_route_step(state)[0], "pmg_detour")
            self.assertEqual(state._pmg_detour_frames, 1)
            self.assertEqual(_route_step(state)[0], "pmg_detour")
            self.assertEqual(state._pmg_detour_frames, 0)
            self.assertEqual(_route_step(state)[0], "pmg_terminal")
            state._cmd_v = 0.5
            self.assertEqual(_route_step(state)[0], "pmg_terminal")
            state._cmd_v = 0.5
            source, reason, override, _, _ = _route_step(state)
            self.assertEqual((source, reason, override), (
                "policy", "route_progress_stalled_policy_fallback", False,
            ))
            self.assertTrue(state._pmg_route_stall_latched)

    def test_astar_countdown_survives_and_same_waypoint_latches(self) -> None:
        state = _route_state()
        environment = {
            "PMG_ACTION_ENABLED": "1",
            "PMG_STALL_COMMAND_SPEED": "0.16",
            "PMG_ROUTE_PROGRESS_EPS_M": "0.05",
            "PMG_ROUTE_PROGRESS_STALL_INFERENCES": "12",
            "PMG_ROUTE_PROGRESS_RELEASE_INFERENCES": "12",
        }
        with patch.dict(os.environ, environment):
            for _ in range(20):
                first = _route_step(state)
                if state._pmg_progress_release_count:
                    break
            self.assertEqual(state._pmg_progress_release_count, 1)
            self.assertEqual(first[0], "pmg_detour")
            self.assertEqual(state._pmg_detour_frames, 12)

            sources = [first[0]]
            for remaining in range(11, -1, -1):
                result = _route_step(state)
                sources.append(result[0])
                self.assertEqual(state._pmg_detour_frames, remaining)
            self.assertEqual(sources, ["pmg_detour"] * 13)
            self.assertEqual(_route_step(state)[0], "pmg_terminal")

            for _ in range(20):
                result = _route_step(state)
                if state._pmg_route_stall_latched:
                    break
            self.assertTrue(state._pmg_route_stall_latched)
            self.assertEqual(state._pmg_progress_release_count, 1)
            self.assertEqual(result[:3], (
                "policy", "route_progress_stalled_policy_fallback", False,
            ))
            self.assertEqual(state._pmg_detour_frames, 0)
            for _ in range(40):
                result = _route_step(state)
                self.assertEqual(result[0], "policy")
                self.assertEqual(state._pmg_progress_release_count, 1)

            # The shared lower detour branch must also remain disabled when
            # the route latch has no countdown left.
            source, reason, override, _ = detour_output(
                state,
                True,
                "route_progress_stalled_policy_fallback",
            )
            self.assertEqual((source, reason, override), (
                "policy", "route_progress_stalled_policy_fallback", False,
            ))

    def test_verified_progress_or_new_waypoint_reopens_route(self) -> None:
        state = _route_state()
        state._pmg_progress_release_key = (0, 4.0, 0.0, 4.0, 0.0)
        state._pmg_progress_release_best_distance = 4.0
        state._pmg_progress_release_attempted = True
        state._pmg_route_stall_latched = True
        state._pmg_progress_waypoint_key = (0, 4.0, 0.0, 4.0, 0.0)
        state._pmg_progress_best_distance = 4.0
        state._pmg_last_control_source = "policy"
        with patch.dict(os.environ, {"PMG_ACTION_ENABLED": "1", "PMG_ROUTE_PROGRESS_EPS_M": "0.05"}):
            state._robot_position = np.array((0.02, 0.0, 0.0))
            self.assertEqual(_route_step(state)[0], "policy")
            self.assertTrue(state._pmg_route_stall_latched)

            state._robot_position = np.array((0.20, 0.0, 0.0))
            self.assertEqual(_route_step(state)[0], "pmg_terminal")
            self.assertFalse(state._pmg_route_stall_latched)
            self.assertFalse(state._pmg_progress_release_attempted)
            self.assertLess(state._pmg_progress_release_best_distance, 4.0)

            state._pmg_route_stall_latched = True
            state._pmg_progress_release_attempted = True
            state._pmg_waypoint_index = 1
            state._pmg_waypoint_position = np.array((5.0, 0.0, 0.0))
            self.assertEqual(_route_step(state)[0], "pmg_terminal")
            self.assertFalse(state._pmg_route_stall_latched)
            self.assertFalse(state._pmg_progress_release_attempted)
            self.assertEqual(state._pmg_progress_release_key, (1, 5.0, 0.0, 4.0, 0.0))

    def test_rotate_first_does_not_count_as_progress_stall(self) -> None:
        state = _route_state()
        state._pmg_waypoint_position = np.array((-4.0, 0.0, 0.0))
        with patch.dict(os.environ, {"PMG_ACTION_ENABLED": "1"}):
            for _ in range(40):
                source, _, _, _, aligning = _route_step(state)
                self.assertTrue(aligning)
                self.assertEqual(source, "pmg_terminal")
                self.assertEqual(state._pmg_progress_stall_inferences, 0)
                self.assertEqual(state._pmg_detour_frames, 0)
            self.assertEqual(state._pmg_progress_release_count, 0)

    def test_invalid_route_clears_route_progress_latch(self) -> None:
        state = _route_state()
        state._pmg_route_valid = False
        state._pmg_progress_waypoint_key = (0, 4.0, 0.0, 4.0, 0.0)
        state._pmg_progress_best_distance = 4.0
        state._pmg_progress_stall_inferences = 9
        state._pmg_progress_release_key = (0, 4.0, 0.0, 4.0, 0.0)
        state._pmg_progress_release_best_distance = 4.0
        state._pmg_progress_release_attempted = True
        state._pmg_route_stall_latched = True
        key, _, _ = route_pre_tick(state, False)
        self.assertIsNone(key)
        self.assertFalse(state._pmg_route_stall_latched)
        self.assertFalse(state._pmg_progress_release_attempted)
        self.assertIsNone(state._pmg_progress_release_key)

        # Rebuilding the same waypoint starts with a fresh bounded release.
        state._pmg_route_valid = True
        source, _, _, _, _ = _route_step(state)
        self.assertEqual(source, "pmg_terminal")
        self.assertFalse(state._pmg_route_stall_latched)
        self.assertFalse(state._pmg_progress_release_attempted)


if __name__ == "__main__":
    unittest.main()
