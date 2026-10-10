"""Deterministic PMG/TIC-VLA control hand-off policy.

Stage 2 compares a released TIC-VLA policy with a structured PMG readout.  A
fair comparison must keep TIC-VLA in charge of the global route and only let
PMG take over a bounded terminal segment.  This module contains the small,
dependency-free policy decision used by tests and by the Isaac compatibility
wrapper.  It deliberately does *not* inspect evaluator goal or success state.

The compatibility wrapper may run in an Isaac process where the project
package is not importable.  Its generated behavior script therefore mirrors
these few rules inline; keeping the reference implementation here makes the
semantics explicit and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


POLICY_ONLY = "policy_only"
TERMINAL = "terminal"
RECOVERY = "recovery"
GLOBAL = "global"

_ALIASES = {
    "policy": POLICY_ONLY,
    "ticvla": POLICY_ONLY,
    "off": POLICY_ONLY,
    "none": POLICY_ONLY,
    "terminal": TERMINAL,
    "local": TERMINAL,
    "local_terminal": TERMINAL,
    "fair": TERMINAL,
    "recovery": RECOVERY,
    "terminal_recovery": RECOVERY,
    "global": GLOBAL,
    "always": GLOBAL,
    "oracle": GLOBAL,
    "global_oracle": GLOBAL,
}


@dataclass(frozen=True)
class ControlGateDecision:
    """Result of evaluating one PMG candidate at one control tick."""

    policy: str
    scope: str
    allow_override: bool
    terminal_active: bool
    terminal_eligible: bool
    candidate_valid: bool
    candidate_distance_m: float | None
    terminal_enter_distance_m: float
    terminal_exit_distance_m: float
    reason: str

    @property
    def override_fraction_denominator(self) -> str:
        """Name the denominator used by telemetry (all policy action ticks)."""

        return "all_action_ticks"


def normalize_control_policy(
    value: str | None,
    *,
    method: str = "M0",
    variant: str | None = None,
) -> str:
    """Normalize an environment/profile value to a small closed vocabulary.

    ``simple_oracle`` intentionally defaults to :data:`TERMINAL`, while
    legacy M1/M2 recovery keeps :data:`RECOVERY` semantics.  Unknown values
    fail closed to ``terminal`` for simple-oracle and ``recovery`` otherwise;
    this prevents a typo from silently enabling an unrestricted oracle.
    """

    method = str(method or "M0").upper()
    variant = str(variant or "").lower()
    if value is None or not str(value).strip():
        if method == "M0":
            return POLICY_ONLY
        return TERMINAL if (method == "M1" and variant == "simple_oracle") else RECOVERY
    normalized = _ALIASES.get(str(value).strip().lower())
    if normalized is not None:
        return normalized
    if method == "M0":
        return POLICY_ONLY
    return TERMINAL if (method == "M1" and variant == "simple_oracle") else RECOVERY


def terminal_hysteresis(
    *,
    candidate_distance_m: float | None,
    current_active: bool,
    enter_distance_m: float = 4.0,
    exit_distance_m: float | None = None,
    candidate_valid: bool = True,
) -> bool:
    """Update terminal-state hysteresis using only candidate geometry.

    The exit radius is at least the enter radius.  Invalid/missing candidates
    always clear the state, so a stale PMG payload cannot retain control.
    """

    enter = max(0.0, float(enter_distance_m))
    exit_radius = max(enter, float(exit_distance_m if exit_distance_m is not None else enter + 1.0))
    if not candidate_valid or candidate_distance_m is None:
        return False
    try:
        distance = float(candidate_distance_m)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(distance) or distance < 0.0:
        return False
    if current_active:
        return distance <= exit_radius
    return distance <= enter


def decide_control_gate(
    *,
    method: str,
    variant: str | None = None,
    policy: str | None = None,
    action_enabled: bool = True,
    candidate_valid: bool = False,
    candidate_distance_m: float | None = None,
    terminal_active: bool = False,
    terminal_enter_distance_m: float = 4.0,
    terminal_exit_distance_m: float | None = None,
    recovery_requested: bool = False,
    require_valid_route: bool = False,
    route_valid: bool = True,
) -> ControlGateDecision:
    """Decide whether PMG may replace/blend the current TIC-VLA command.

    ``GLOBAL`` is retained only as an explicitly requested diagnostic upper
    bound.  The default ``simple_oracle`` profile is ``TERMINAL`` and keeps
    the learned policy outside the local terminal radius.  ``RECOVERY`` is
    the historical bounded-recovery mode used by M2/M3.
    """

    method = str(method or "M0").upper()
    normalized = normalize_control_policy(policy, method=method, variant=variant)
    enter = max(0.0, float(terminal_enter_distance_m))
    exit_radius = max(enter, float(terminal_exit_distance_m if terminal_exit_distance_m is not None else enter + 1.0))
    active = terminal_hysteresis(
        candidate_distance_m=candidate_distance_m,
        current_active=bool(terminal_active),
        enter_distance_m=enter,
        exit_distance_m=exit_radius,
        candidate_valid=bool(candidate_valid),
    )
    try:
        finite_distance = (
            candidate_distance_m is not None
            and math.isfinite(float(candidate_distance_m))
            and float(candidate_distance_m) >= 0.0
        )
    except (TypeError, ValueError):
        finite_distance = False
    eligible = bool(candidate_valid and finite_distance)
    if method == "M0" or not action_enabled:
        return ControlGateDecision(
            normalized,
            POLICY_ONLY,
            False,
            active,
            eligible,
            bool(candidate_valid),
            candidate_distance_m,
            enter,
            exit_radius,
            "method_or_action_disabled",
        )
    if not candidate_valid:
        return ControlGateDecision(
            normalized,
            POLICY_ONLY,
            False,
            False,
            False,
            False,
            candidate_distance_m,
            enter,
            exit_radius,
            "candidate_invalid",
        )
    if normalized == POLICY_ONLY:
        return ControlGateDecision(
            normalized,
            POLICY_ONLY,
            False,
            active,
            eligible,
            True,
            candidate_distance_m,
            enter,
            exit_radius,
            "policy_only",
        )
    if normalized == TERMINAL and active and require_valid_route and not route_valid:
        return ControlGateDecision(
            normalized,
            POLICY_ONLY,
            False,
            True,
            eligible,
            True,
            candidate_distance_m,
            enter,
            exit_radius,
            "route_infeasible_policy_fallback",
        )
    if normalized == GLOBAL:
        return ControlGateDecision(
            normalized,
            "global",
            True,
            active,
            eligible,
            True,
            candidate_distance_m,
            enter,
            exit_radius,
            "explicit_global_override",
        )
    # An explicit global profile is the only way to hand off before the local
    # terminal radius.  For the normal recovery profile, a caller must also
    # provide the measured local trigger (stall/heading) below.
    if not active:
        return ControlGateDecision(
            normalized,
            POLICY_ONLY,
            False,
            False,
            eligible,
            True,
            candidate_distance_m,
            enter,
            exit_radius,
            "outside_terminal_radius",
        )
    if normalized == RECOVERY and not recovery_requested:
        return ControlGateDecision(
            normalized,
            POLICY_ONLY,
            False,
            True,
            True,
            True,
            candidate_distance_m,
            enter,
            exit_radius,
            "terminal_recovery_not_requested",
        )
    return ControlGateDecision(
        normalized,
        "terminal",
        True,
        True,
        True,
        True,
        candidate_distance_m,
        enter,
        exit_radius,
        "terminal_handoff" if normalized == TERMINAL else "terminal_recovery",
    )


__all__ = [
    "GLOBAL",
    "POLICY_ONLY",
    "RECOVERY",
    "TERMINAL",
    "ControlGateDecision",
    "decide_control_gate",
    "normalize_control_policy",
    "terminal_hysteresis",
]
