#!/usr/bin/env python3
"""Reproducible RSI + PMG memory experiment (standard library only).

The simulator is intentionally small, but preserves the method contract:
Rule Memory -> candidate retrieval -> PMG route -> navigation feedback ->
Attempt Memory -> offline patch proposal/verification -> atomic rule update.
Replace ``DemoEnvironment`` with the real PMG/route adapter when integrating.
"""
from __future__ import annotations

import argparse, json, logging, math, os, tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

DEFAULT_RULE = {
    "schema_version": 1, "rule_id": "route_repair_policy", "version": "nav_rule_v01",
    "parent_version": None, "scope": "all_scenes", "diagnostic_event": "NO_PROGRESS",
    "trigger": {"window_s": 2.0, "min_waypoint_progress_m": 0.18},
    "repair": {"max_same_candidate_replans": 1, "switch_to_alternative": True,
               "candidate_cooldown_s": 12.0, "max_switches_per_episode": 2},
    "candidate_ranking": {"weight_constraint": 1.0, "weight_approach": 0.6,
                           "weight_route_length": 0.25, "weight_repeat_failure": 2.0},
    "safety": {"require_hard_valid": True, "require_route_reachable": True}, "status": "accepted",
}

@dataclass
class Candidate:
    key: str; anchor_id: str; constraint: float; approach: float; route_length: float
    hard_valid: bool = True; reachable: bool = True

@dataclass
class Attempt:
    stalls: int = 0; replans: int = 0; switches: int = 0; cooldown_until: float = 0.0

@dataclass
class EpisodeResult:
    episode_id: str; success: bool; actions: list[str]; stalls: int; recoveries: int
    selected: list[str]; final_distance: float

class RuleMemory:
    def __init__(self, path: Path):
        self.path = path
        if path.exists(): self.rule = json.loads(path.read_text())
        else: self.rule = json.loads(json.dumps(DEFAULT_RULE)); self._write()
    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="rule-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as f: json.dump(self.rule, f, indent=2); f.write("\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    def update(self, patch: dict[str, Any], reason: str):
        candidate = json.loads(json.dumps(self.rule)); candidate["parent_version"] = self.rule["version"]
        n = int(self.rule["version"].split("v")[-1]) + 1; candidate["version"] = f"nav_rule_v{n:02d}"
        for section, values in patch.items():
            if section not in candidate or not isinstance(values, dict): continue
            candidate[section].update(values)
        candidate["provenance"] = {"reason": reason, "accepted_by": "deterministic_dev_verifier"}
        self.rule = candidate; self._write()

class DemoEnvironment:
    """Two same-anchor poses: A stalls, B makes progress after switching."""
    def __init__(self):
        self.candidates = [Candidate("goal:A", "cones", .95, .70, 4.0), Candidate("goal:B", "cones", .82, .90, 5.0)]
    def route(self, candidate: Candidate, pose: float) -> list[float] | None:
        return None if not candidate.reachable else [pose, candidate.route_length]
    def progress(self, candidate: Candidate, tick: int) -> float:
        return 0.03 if candidate.key == "goal:A" else 0.35

def eligible(env: DemoEnvironment, attempts: dict[str, Attempt], now: float, rule: dict) -> list[Candidate]:
    w = rule["candidate_ranking"]; valid = []
    for c in env.candidates:
        a = attempts[c.key]
        if rule["safety"]["require_hard_valid"] and not c.hard_valid: continue
        if rule["safety"]["require_route_reachable"] and not c.reachable: continue
        if a.cooldown_until > now: continue
        score = w["weight_constraint"]*c.constraint + w["weight_approach"]*c.approach - w["weight_route_length"]*c.route_length - w["weight_repeat_failure"]*a.stalls
        valid.append((score, c))
    return [c for _, c in sorted(valid, key=lambda x: x[0], reverse=True)]

def run_episode(ep: str, memory: RuleMemory, log: logging.Logger) -> EpisodeResult:
    env, rule, attempts = DemoEnvironment(), memory.rule, {c.key: Attempt() for c in DemoEnvironment().candidates}
    current = env.candidates[0]; actions, selected, stalls, recoveries, now, distance = [], [current.key], 0, 0, 0.0, 1.0
    for tick in range(6):
        delta = env.progress(current, tick); distance = max(0.0, distance-delta); now += 1.0
        log.info("episode=%s step=%d candidate=%s progress=%.2f distance=%.2f", ep, tick, current.key, delta, distance)
        if delta >= rule["trigger"]["min_waypoint_progress_m"]: actions.append("CONTINUE"); recoveries += int(current.key == "goal:B"); break
        stalls += 1; attempts[current.key].stalls += 1; actions.append("NO_PROGRESS")
        if attempts[current.key].replans < rule["repair"]["max_same_candidate_replans"]:
            attempts[current.key].replans += 1; actions.append("REPLAN_CURRENT"); continue
        attempts[current.key].cooldown_until = now + rule["repair"]["candidate_cooldown_s"]
        alternatives = [c for c in eligible(env, attempts, now, rule) if c.anchor_id == current.anchor_id and c.key != current.key]
        if alternatives and attempts[current.key].switches < rule["repair"]["max_switches_per_episode"]:
            current = alternatives[0]; attempts[current.key].switches += 1; selected.append(current.key); actions.append("SWITCH_CANDIDATE"); continue
        actions.append("FALLBACK_VLA"); break
    result = EpisodeResult(ep, current.key == "goal:B" and distance < 1.0, actions, stalls, recoveries, selected, distance)
    log.info("episode=%s result=%s actions=%s", ep, result.success, actions)
    return result

def propose_and_verify(memory: RuleMemory, results: list[EpisodeResult], log: logging.Logger) -> bool:
    # Deterministic stand-in for the constrained Meta-Agent; replace with LLM adapter later.
    if not results or not any(r.stalls for r in results): return False
    patch = {"trigger": {"min_waypoint_progress_m": 0.15}}
    memory.update(patch, "repeated recoverable stalls"); log.info("accepted rule patch version=%s", memory.rule["version"]); return True

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path); p.add_argument("--output", type=Path, default=Path("outputs/rsi_memory_demo")); p.add_argument("--episodes", type=int, default=2); args = p.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", encoding="utf-8")])
    log = logging.getLogger("rsi_memory"); memory = RuleMemory(out / "active_nav_rules.json"); results = [run_episode(f"demo_{i+1}", memory, log) for i in range(args.episodes)]
    updated = propose_and_verify(memory, results, log); report = {"rule_version": memory.rule["version"], "rule_updated": updated, "episodes": [asdict(r) for r in results]}
    (out / "run_report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
