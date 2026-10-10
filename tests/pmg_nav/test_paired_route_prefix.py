"""Offline regression checks for paired M0 command authority at PMG handoff."""

from __future__ import annotations

import os
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "autodl/scripts/08_run_dynanav_baseline.sh"


def _prefix_tick(self, v_cmd: float, w_cmd: float, *, terminal: bool, global_override: bool = False):
    script = SCRIPT.read_text(encoding="utf-8")
    start = "        simple_oracle_variant = (\n"
    end = "        # A route planner supplies an immediate collision-free waypoint"
    assert script.count(start) == 1
    body = textwrap.dedent(script[script.index(start):script.index(end)])
    source = (
        "def tick(self, v_cmd, w_cmd, terminal, global_override):\n"
        "    pmg_terminal_eligible = terminal\n"
        "    pmg_global_override = global_override\n"
        "    pmg_control_policy = 'global' if global_override else 'terminal'\n"
        "    pmg_gate_reason = 'terminal_enter'\n"
        "    pmg_control_source = 'policy'\n"
        "    policy_v_cmd, policy_w_cmd = v_cmd, w_cmd\n"
        + textwrap.indent(body, "    ")
        + "    return v_cmd, w_cmd, stage2_prefix_applied, route_gate_block, pmg_control_source, pmg_gate_reason\n"
    )
    namespace = {"os": os}
    exec(compile(source, str(SCRIPT), "exec"), namespace)
    return namespace["tick"](self, v_cmd, w_cmd, terminal, global_override)


def _state(*, route_valid: bool, prefix: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        _pmg_method="M1",
        _pmg_route_valid=route_valid,
        _stage2_policy_prefix_enabled=prefix,
        _stage2_policy_prefix_by_frame={118: (1.5, 0.028360127)},
        _pmg_checkpoint_released=False,
    )


class PairedRoutePrefixTests(unittest.TestCase):
    def test_route_rejection_keeps_paired_command_on_first_terminal_tick(self) -> None:
        with patch.dict(os.environ, {
            "BENCHMARK_SIMULATION_FRAME_COUNT": "118",
            "STAGE2_M1_VARIANT": "simple_oracle",
            "PMG_ACTION_ENABLED": "1",
            "PMG_SIMPLE_ORACLE_REQUIRE_VALID_ROUTE": "1",
        }):
            command = _prefix_tick(_state(route_valid=False), 1.5, -0.000597983, terminal=True)

        self.assertEqual(command, (
            1.5, 0.028360127, True, True,
            "m0_policy_prefix", "route_infeasible_policy_fallback",
        ))

    def test_valid_route_releases_prefix_for_terminal_handoff(self) -> None:
        with patch.dict(os.environ, {
            "BENCHMARK_SIMULATION_FRAME_COUNT": "118",
            "STAGE2_M1_VARIANT": "simple_oracle",
            "PMG_ACTION_ENABLED": "1",
            "PMG_SIMPLE_ORACLE_REQUIRE_VALID_ROUTE": "1",
        }):
            command = _prefix_tick(_state(route_valid=True), 1.5, -0.000597983, terminal=True)

        self.assertEqual(command, (1.5, -0.000597983, False, False, "policy", "terminal_enter"))

    def test_route_gate_does_not_change_preterminal_or_global_authority(self) -> None:
        with patch.dict(os.environ, {
            "BENCHMARK_SIMULATION_FRAME_COUNT": "118",
            "STAGE2_M1_VARIANT": "simple_oracle",
            "PMG_ACTION_ENABLED": "1",
            "PMG_SIMPLE_ORACLE_REQUIRE_VALID_ROUTE": "1",
        }):
            preterminal = _prefix_tick(_state(route_valid=False), 1.5, -0.01, terminal=False)
            global_mode = _prefix_tick(
                _state(route_valid=False), 1.5, -0.01,
                terminal=False, global_override=True,
            )

        self.assertEqual(preterminal, (1.5, 0.028360127, True, False, "m0_policy_prefix", "paired_prefix_before_terminal"))
        self.assertEqual(global_mode, (1.5, -0.01, False, False, "policy", "terminal_enter"))

    def test_missing_paired_command_still_fails_closed_on_route_rejection(self) -> None:
        state = _state(route_valid=False)
        state._stage2_policy_prefix_by_frame.clear()
        with patch.dict(os.environ, {
            "BENCHMARK_SIMULATION_FRAME_COUNT": "118",
            "STAGE2_M1_VARIANT": "simple_oracle",
            "PMG_ACTION_ENABLED": "1",
            "PMG_SIMPLE_ORACLE_REQUIRE_VALID_ROUTE": "1",
        }):
            with self.assertRaisesRegex(RuntimeError, "no M0 command for simulation frame 118"):
                _prefix_tick(state, 1.5, -0.000597983, terminal=True)
