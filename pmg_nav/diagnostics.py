"""Reproducible Stage 2 diagnostic contracts and paired statistics.

This module is deliberately independent of TIC-VLA.  It defines the four
diagnostic conditions and small, deterministic helpers used by the runner and
the offline paired evaluator.  A missing predicted-perception backend is a
blocked condition, never silently replaced by simulator oracle data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Iterable


@dataclass(frozen=True)
class DiagnosticContract:
    mode: str
    perception_source: str
    action_enabled: bool
    pmg_built: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONTRACTS: dict[str, DiagnosticContract] = {
    "M0": DiagnosticContract(
        "M0", "none", False, False, "原始 TIC-VLA baseline，无 PMG"
    ),
    "M1": DiagnosticContract(
        "M1", "oracle", True, True, "simulator oracle PMG，允许影响 goal/action"
    ),
    "M2": DiagnosticContract(
        "M2", "predicted", True, True, "真实预测 perception PMG，允许影响 goal/action"
    ),
    "M3": DiagnosticContract(
        "M3", "oracle", False, True, "构建 PMG，但禁止 PMG 影响 action"
    ),
}


def resolve_contract(mode: str | None = None) -> DiagnosticContract:
    value = (mode or "M0").strip().upper()
    if value not in CONTRACTS:
        raise ValueError(f"unsupported Stage 2 diagnostic mode: {value!r}")
    return CONTRACTS[value]


def stable_checksum(value: Any) -> str:
    """Hash JSON-stable values for action/data-flow instrumentation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sequence_checksum(values: Iterable[Any]) -> str:
    return stable_checksum(list(values))


def bootstrap_ci(
    values: Iterable[float],
    *,
    seed: int = 20260903,
    samples: int = 2000,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, samples)):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    low = int((alpha / 2.0) * (len(means) - 1))
    high = int((1.0 - alpha / 2.0) * (len(means) - 1))
    return (means[low], means[high])


def paired_sign_permutation_pvalue(values: Iterable[float], *, seed: int = 20260903, samples: int = 10000) -> float | None:
    """Two-sided paired sign-permutation p-value without a SciPy dependency."""
    values = [float(value) for value in values if math.isfinite(float(value)) and value != 0.0]
    if not values:
        return None
    observed = abs(sum(values) / len(values))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(max(1, samples)):
        permuted = sum(value if rng.random() < 0.5 else -value for value in values) / len(values)
        exceed += int(abs(permuted) >= observed)
    return (exceed + 1) / (samples + 1)
