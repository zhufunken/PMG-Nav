#!/usr/bin/env python3
"""Report a strict paired M0/M1 queue without hiding incomplete artifacts.

The compatibility runner writes one ``benchmark_metrics.json`` per process,
but retry/archival tools may place those files below an additional method or
episode directory.  This utility walks that nested layout, keeps every failed
or ambiguous attempt in the machine-readable report, and only computes paired
statistics from one valid record for each method/episode.

The module is deliberately stdlib-only so it can run with the same Python
environment used by the queue.  It never starts a simulator or edits a source
artifact; output files are created only when explicitly requested and are not
overwritten unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Iterable, Iterator, Mapping, Sequence


METHODS = ("M0", "M1")
DEFAULT_EPISODES = tuple(range(1, 86))
_METHOD_RE = re.compile(r"(?:^|[/_\\-])m([01])(?:$|[/_\\-])", re.IGNORECASE)
_EPISODE_RE = re.compile(r"(?:^|[/_\\-])(?:episode|ep|e)[_\\-]?0*(\\d+)(?:$|[/_\\-])", re.IGNORECASE)
_EPISODE_BARE_RE = re.compile(r"(?:^|[/_\\-])0*(\\d{1,3})(?:$|[/_\\-])")

_METRIC_KEYS = frozenset(
    {
        "success",
        "success_rate",
        "spl",
        "navigation_error",
        "navigation_error_m",
        "navigation_error_meters",
        "nav_error",
        "ne",
        "artifact_status",
        "error",
        "duration",
        "duration_frames",
        "episode_seed",
    }
)
_NESTED_RESULT_KEYS = ("metrics", "result", "episode_metrics", "benchmark")
_SKIP_RECURSION_KEYS = frozenset(
    {
        "action_trace",
        "behavior_execution",
        "behavior_provenance",
        "control_gate",
        "frames",
        "observations",
        "pmg",
        "pmg_adapter",
        "telemetry",
        "trace",
    }
)
_PROTOCOL_KEYS = (
    "compatibility_runner",
    "runtime",
    "runtime_num_people",
    "official_num_people",
    "dynamic_people_note",
    "strict_parity",
    "strict_parity_contract",
    "parity_fixed_horizon",
    "parity_deterministic",
    "deterministic_torch_cuda",
    "paired_action_prefix_enabled",
    "paired_action_prefix_policy",
    "result_class",
    "smoke_mode",
    "smoke_frames",
    "m1_variant",
    "graph_mode",
    "control_policy",
    "perception_source",
    "behavior_backend",
)


class Candidate:
    __slots__ = ("source", "method", "episode", "record", "metadata", "section")

    def __init__(
        self,
        source: str,
        method: str | None,
        episode: str | None,
        record: dict[str, Any],
        metadata: dict[str, Any],
        section: tuple[str, ...],
    ) -> None:
        self.source = source
        self.method = method
        self.episode = episode
        self.record = record
        self.metadata = metadata
        self.section = section


def _normalise_method(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text if text in METHODS else None


def _normalise_episode(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        match = re.fullmatch(r"(?:episode|ep|e)[_\-]?0*(\d+)", text)
        if not match and text.isdigit():
            match = re.fullmatch(r"0*(\d+)", text)
        if not match:
            return None
        number = int(match.group(1))
    else:
        return None
    return f"episode_{number}" if 1 <= number <= 999 else None


def _episode_number(value: Any) -> int | None:
    normalised = _normalise_episode(value)
    if normalised is None:
        return None
    return int(normalised.rsplit("_", 1)[1])


def _infer_method(value: Any) -> str | None:
    method = _normalise_method(value)
    if method is not None:
        return method
    if not isinstance(value, str):
        return None
    match = _METHOD_RE.search(value)
    return f"M{match.group(1)}" if match else None


def _infer_episode(value: Any) -> str | None:
    episode = _normalise_episode(value)
    if episode is not None:
        return episode
    if not isinstance(value, str):
        return None
    match = _EPISODE_RE.search(value)
    if match:
        return _normalise_episode(match.group(1))
    match = _EPISODE_BARE_RE.search(value)
    return _normalise_episode(match.group(1)) if match else None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flatten_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Merge common nested metric wrappers without flattening telemetry."""

    flattened = dict(value)
    for key in _NESTED_RESULT_KEYS:
        child = value.get(key)
        if isinstance(child, Mapping):
            child_flat = _flatten_result(child)
            for child_key, child_value in child_flat.items():
                flattened.setdefault(child_key, child_value)
    return flattened


def _record_episode(value: Mapping[str, Any]) -> str | None:
    for key in ("episode", "episode_id", "episode_key", "episode_name"):
        episode = _infer_episode(value.get(key))
        if episode is not None:
            return episode
    for key in ("run_id", "run", "name"):
        episode = _infer_episode(value.get(key))
        if episode is not None:
            return episode
    return None


def _record_method(value: Mapping[str, Any]) -> str | None:
    for key in ("method", "run_method", "policy_method", "mode"):
        method = _infer_method(value.get(key))
        if method is not None:
            return method
    for key in ("run_id", "run", "name"):
        method = _infer_method(value.get(key))
        if method is not None:
            return method
    return None


def _looks_like_record(value: Mapping[str, Any], episode: str | None) -> bool:
    if episode is None:
        return False
    flattened = _flatten_result(value)
    return bool(_METRIC_KEYS.intersection(flattened))


def _protocol_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in _PROTOCOL_KEYS:
        if key in value and isinstance(value[key], (str, int, float, bool)):
            metadata[key] = value[key]
    caveats = value.get("protocol_caveats", value.get("caveats"))
    if isinstance(caveats, str):
        metadata["protocol_caveats"] = [caveats]
    elif isinstance(caveats, list):
        metadata["protocol_caveats"] = [item for item in caveats if isinstance(item, str)]
    return metadata


def _walk_records(
    value: Any,
    *,
    source: str,
    method: str | None = None,
    episode: str | None = None,
    section: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Candidate]:
    if isinstance(value, Mapping):
        local_method = _record_method(value) or method
        local_episode = _record_episode(value) or episode
        local_metadata = dict(metadata or {})
        local_metadata.update(_protocol_metadata(value))
        flattened = _flatten_result(value)
        if _looks_like_record(value, local_episode):
            yield Candidate(
                source=source,
                method=local_method,
                episode=local_episode,
                record=flattened,
                metadata=local_metadata,
                section=section,
            )
            return
        for key, child in value.items():
            if key in _SKIP_RECURSION_KEYS:
                continue
            child_method = _infer_method(key) or local_method
            child_episode = _infer_episode(key) or local_episode
            if isinstance(child, (Mapping, list, tuple)):
                yield from _walk_records(
                    child,
                    source=source,
                    method=child_method,
                    episode=child_episode,
                    section=section + (str(key),),
                    metadata=local_metadata,
                )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_records(
                child,
                source=source,
                method=method,
                episode=episode,
                section=section + (str(index),),
                metadata=metadata,
            )


def _json_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("benchmark_metrics.json"))


def discover_records(root: Path) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Read all benchmark JSONs below ``root`` and retain read failures."""

    candidates: list[Candidate] = []
    failures: list[dict[str, Any]] = []
    paths = _json_paths(root)
    if not paths:
        failures.append({"source": str(root), "reason": "no_benchmark_metrics_files"})
    for path in paths:
        path_method = _infer_method(str(path))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(
                {
                    "source": str(path),
                    "method": path_method,
                    "episode": _infer_episode(str(path)),
                    "reason": "unreadable_benchmark_metrics",
                    "detail": str(exc),
                }
            )
            continue
        found = list(
            _walk_records(
                payload,
                source=str(path),
                method=path_method,
                metadata=_protocol_metadata(payload) if isinstance(payload, Mapping) else None,
            )
        )
        if not found:
            failures.append(
                {
                    "source": str(path),
                    "method": path_method,
                    "episode": _infer_episode(str(path)),
                    "reason": "no_episode_record",
                }
            )
        candidates.extend(found)
        top_level_artifact_status = payload.get("artifact_status") if isinstance(payload, Mapping) else None
        if top_level_artifact_status is not None:
            for candidate in found:
                candidate.metadata.setdefault("_payload_artifact_status", top_level_artifact_status)
        if top_level_artifact_status is not None and top_level_artifact_status != "valid":
            failures.append(
                {
                    "source": str(path),
                    "method": path_method,
                    "episode": _infer_episode(str(path)),
                    "reason": "top_level_invalid_artifact",
                    "artifact_status": top_level_artifact_status,
                }
            )
        if isinstance(payload, Mapping) and isinstance(payload.get("failed_episodes"), list):
            for failed in payload["failed_episodes"]:
                if isinstance(failed, Mapping):
                    failures.append(
                        {
                            "source": str(path),
                            "method": _record_method(failed) or path_method,
                            "episode": _record_episode(failed) or _infer_episode(str(path)),
                            "reason": "reported_failed_episode",
                            "detail": failed.get("error") or failed.get("reason"),
                        }
                    )
                else:
                    failures.append(
                        {
                            "source": str(path),
                            "method": path_method,
                            "reason": "reported_failed_episode",
                            "detail": str(failed),
                        }
                    )
        elif isinstance(payload, Mapping) and isinstance(payload.get("failed_episodes"), int):
            failed_count = int(payload["failed_episodes"])
            if failed_count > 0:
                failures.append(
                    {
                        "source": str(path),
                        "method": path_method,
                        "episode": _infer_episode(str(path)),
                        "reason": "reported_failed_episode_count",
                        "count": failed_count,
                    }
                )
    return candidates, failures


def _value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    control_gate = record.get("control_gate")
    for container in (record, control_gate if isinstance(control_gate, Mapping) else {}):
        for name in names:
            if name in container and container[name] is not None:
                return container[name]
    return None


def _override(record: Mapping[str, Any]) -> tuple[float | None, float | None, float | None, list[str]]:
    warnings: list[str] = []
    raw_fraction = _value(
        record,
        ("pmg_override_fraction", "override_fraction", "override_rate"),
    )
    fraction = _finite(raw_fraction)
    if fraction is None:
        raw_override = _value(record, ("pmg_override", "override"))
        if isinstance(raw_override, bool):
            fraction = 1.0 if raw_override else 0.0
    count = _finite(_value(record, ("pmg_override_count", "override_count", "override_ticks")))
    actions = _finite(
        _value(record, ("action_output_count", "policy_action_count", "action_count", "total_actions"))
    )
    if fraction is None and count is not None and actions is not None and actions > 0:
        fraction = count / actions
    if fraction is not None and not 0.0 <= fraction <= 1.0:
        warnings.append("override_fraction_out_of_range")
        fraction = None
    if count is not None and count < 0.0:
        warnings.append("override_count_negative")
        count = None
    if actions is not None and actions < 0.0:
        warnings.append("action_count_negative")
        actions = None
    return fraction, count, actions, warnings


def validate_candidate(candidate: Candidate) -> dict[str, Any]:
    """Convert one raw candidate into a compact valid/failed episode row."""

    record = candidate.record
    metadata = candidate.metadata
    reasons: list[str] = []
    if record.get("error"):
        reasons.append("runner_error")
    artifact_status = record.get("artifact_status", metadata.get("_payload_artifact_status"))
    payload_artifact_status = metadata.get("_payload_artifact_status")
    if payload_artifact_status is not None and payload_artifact_status != "valid":
        artifact_status = payload_artifact_status
    if artifact_status is None:
        reasons.append("missing_artifact_status")
    elif artifact_status != "valid":
        reasons.append("invalid_artifact")
    invalid_reasons = record.get("artifact_invalid_reasons")
    if isinstance(invalid_reasons, list):
        reasons.extend(str(item) for item in invalid_reasons if item)

    success = record.get("success")
    if isinstance(success, bool):
        success_value: bool | None = success
    else:
        success_rate = _finite(record.get("success_rate"))
        success_value = bool(round(success_rate)) if success_rate in (0.0, 1.0) else None
    if success_value is None:
        reasons.append("missing_success")

    spl = _finite(_value(record, ("spl", "mean_spl")))
    if spl is None:
        reasons.append("missing_spl")
    elif not 0.0 <= spl <= 1.0:
        reasons.append("spl_out_of_range")

    navigation_error = _finite(
        _value(record, ("navigation_error", "navigation_error_m", "nav_error", "mean_navigation_error_m", "navigation_error_meters", "ne"))
    )
    if navigation_error is None:
        reasons.append("missing_navigation_error")
    elif navigation_error < 0.0:
        reasons.append("navigation_error_negative")

    duration = _finite(record.get("duration"))
    if "duration" in record and (duration is None or duration <= 0.0):
        reasons.append("invalid_duration")
    duration_frames = _finite(record.get("duration_frames"))
    if "duration_frames" in record and (duration_frames is None or duration_frames <= 0.0):
        reasons.append("invalid_duration_frames")

    seed = _finite(record.get("episode_seed"))
    expected_seed = 666 + (_episode_number(candidate.episode) or 0)
    if seed is None:
        reasons.append("missing_episode_seed")
    elif int(seed) != expected_seed:
        reasons.append("episode_seed_mismatch")
    for key in ("strict_parity_contract", "parity_fixed_horizon", "deterministic_torch_cuda"):
        if record.get(key, metadata.get(key)) is not True:
            reasons.append(f"{key}_not_true")
    if record.get("smoke_mode", metadata.get("smoke_mode")) is True or record.get("result_class", metadata.get("result_class")) == "development_smoke":
        reasons.append("development_smoke_not_formal")
    if record.get("smoke_frames", metadata.get("smoke_frames")) not in (None, 0):
        reasons.append("smoke_frame_cap_present")

    override_fraction, override_count, action_count, override_warnings = _override(record)
    row: dict[str, Any] = {
        "method": candidate.method,
        "episode": candidate.episode,
        "source": candidate.source,
        "section": list(candidate.section),
        "status": "valid" if not reasons else "failed",
        "success": success_value,
        "spl": spl,
        "navigation_error_m": navigation_error,
        "override_fraction": override_fraction,
        "override_count": override_count,
        "action_count": action_count,
        "override_warnings": override_warnings,
        "failure_reasons": list(dict.fromkeys(reasons)),
        "artifact_status": artifact_status or "unspecified",
        "process_status": None,
        "metadata": dict(candidate.metadata),
    }
    return row


def paired_bootstrap_ci(
    values: Iterable[float],
    *,
    seed: int = 20260822,
    samples: int = 2000,
) -> dict[str, Any] | None:
    numbers = [float(value) for value in values if _finite(value) is not None]
    if not numbers:
        return None
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(numbers, k=len(numbers))) for _ in range(samples)]
    means.sort()
    low_index = min(len(means) - 1, max(0, int(0.025 * samples)))
    high_index = min(len(means) - 1, max(0, int(0.975 * samples)))
    return {
        "low": means[low_index],
        "high": means[high_index],
        "n": len(numbers),
        "samples": samples,
        "seed": seed,
    }


def wilcoxon_signed_rank_pvalue(values: Iterable[float]) -> float | None:
    """Exact two-sided signed-rank p-value with zero differences removed."""

    differences = [
        float(value)
        for value in values
        if _finite(value) is not None and float(value) != 0.0
    ]
    if not differences:
        return None
    absolute = sorted((abs(value), index) for index, value in enumerate(differences))
    ranks = [0.0] * len(differences)
    cursor = 0
    while cursor < len(absolute):
        end = cursor + 1
        while end < len(absolute) and absolute[end][0] == absolute[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[absolute[position][1]] = average_rank
        cursor = end
    rank2 = [int(round(rank * 2.0)) for rank in ranks]
    observed = sum(rank for value, rank in zip(differences, rank2) if value > 0.0)
    total_rank = sum(rank2)
    distribution = [0] * (total_rank + 1)
    distribution[0] = 1
    for rank in rank2:
        for subtotal in range(total_rank, -1, -1):
            count = distribution[subtotal]
            if count:
                distribution[subtotal + rank] += count
    assignments = 2 ** len(rank2)
    lower = sum(distribution[: observed + 1])
    upper = sum(distribution[observed:])
    return min(1.0, 2.0 * min(lower, upper) / assignments)


# Keep the helper names used by the older matrix finalizer available to small
# downstream audits that import this module directly.
_bootstrap_ci = paired_bootstrap_ci
_wilcoxon_signed_rank_pvalue = wilcoxon_signed_rank_pvalue


def _statistic(values: Sequence[float], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    numbers = [float(value) for value in values if _finite(value) is not None]
    return {
        "n": len(numbers),
        "mean_delta": statistics.fmean(numbers) if numbers else None,
        "bootstrap_ci_95": paired_bootstrap_ci(
            numbers, seed=seed, samples=bootstrap_samples
        ),
        "wilcoxon_pvalue": wilcoxon_signed_rank_pvalue(numbers),
    }


def _load_json(path: Path, label: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, {"source": str(path), "reason": f"unreadable_{label}", "detail": str(exc)}
    if not isinstance(payload, dict):
        return None, {"source": str(path), "reason": f"{label}_not_object"}
    return payload, None


def _progress_rows(path: Path | None) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, [], {}
    payload, failure = _load_json(path, "progress")
    if failure is not None or payload is None:
        return {}, [failure or {"source": str(path), "reason": "invalid_progress"}], {}
    runs = payload.get("runs")
    if not isinstance(runs, Mapping):
        return {}, [{"source": str(path), "reason": "progress_runs_missing"}], payload
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for run_id, status in runs.items():
        if not isinstance(status, Mapping):
            status = {"status": status}
        method = _infer_method(str(run_id))
        episode = _infer_episode(str(run_id))
        if method is None or episode is None:
            failures.append({"source": str(path), "run_id": str(run_id), "reason": "unkeyed_progress_run"})
            continue
        indexed[(method, episode)] = {"run_id": str(run_id), **dict(status)}
        if status.get("status") not in {"passed", "cached"}:
            failures.append(
                {
                    "source": str(path),
                    "run_id": str(run_id),
                    "method": method,
                    "episode": episode,
                    "reason": "process_not_passed",
                    "process_status": status.get("status"),
                    "detail": status.get("error") or status.get("reason"),
                }
            )
    return indexed, failures, payload


def _manifest_metadata(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        return {}
    metadata = _protocol_metadata(manifest)
    metadata.update({"methods": manifest.get("methods"), "official_episode_count": manifest.get("official_episode_count")})
    for key in ("run_prefix", "paired_m0_run_prefix", "benchmark_config", "pairing_seed_policy"):
        if isinstance(manifest.get(key), (str, int, float, bool)):
            metadata[key] = manifest[key]
    return {key: value for key, value in metadata.items() if value is not None}


def _manifest_caveats(manifest: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(manifest, Mapping):
        return []
    values: list[str] = []
    for key in ("protocol_caveats", "caveats", "limitations"):
        value = manifest.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return values


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def aggregate_queue(
    root: Path,
    *,
    expected_episodes: Iterable[int] = DEFAULT_EPISODES,
    methods: Iterable[str] = METHODS,
    progress_path: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 20260822,
    default_method: str | None = None,
) -> dict[str, Any]:
    """Aggregate a queue and return a JSON-serialisable report."""

    selected_methods = tuple(_normalise_method(method) for method in methods)
    if any(method is None for method in selected_methods) or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must be a unique subset of M0/M1")
    selected_methods = tuple(method for method in selected_methods if method is not None)
    expected_numbers = tuple(sorted(set(int(item) for item in expected_episodes)))
    expected_keys = {(method, f"episode_{number}") for method in selected_methods for number in expected_numbers}

    candidates, discovery_failures = discover_records(root)
    progress, progress_failures, progress_payload = _progress_rows(progress_path)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    failures = [*discovery_failures, *progress_failures]
    unknown_records: list[dict[str, Any]] = []
    for candidate in candidates:
        method = candidate.method or _normalise_method(default_method)
        episode = candidate.episode
        if method is None or episode is None:
            unknown_records.append(
                {
                    "source": candidate.source,
                    "section": list(candidate.section),
                    "method": method,
                    "episode": episode,
                    "reason": "unkeyed_benchmark_record",
                }
            )
            continue
        row = validate_candidate(
            Candidate(
                source=candidate.source,
                method=method,
                episode=episode,
                record=candidate.record,
                metadata=candidate.metadata,
                section=candidate.section,
            )
        )
        process = progress.get((method, episode))
        if process is not None:
            row["run_id"] = process.get("run_id")
            row["process_status"] = process.get("status")
            if process.get("status") not in {"passed", "cached"}:
                row["failure_reasons"] = list(dict.fromkeys(row["failure_reasons"] + ["process_not_passed"]))
                row["status"] = "failed"
        grouped.setdefault((method, episode), []).append(row)
    failures.extend(unknown_records)

    rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for method in selected_methods:
        for number in expected_numbers:
            episode = f"episode_{number}"
            key = (method, episode)
            attempts = grouped.get(key, [])
            if not attempts:
                process = progress.get(key)
                reason = "missing_benchmark_metrics"
                if process is not None:
                    reason = f"missing_result_for_process_{process.get('status', 'unknown')}"
                row = {
                    "method": method,
                    "episode": episode,
                    "status": "missing",
                    "source": None,
                    "attempts": [],
                    "failure_reasons": [reason],
                    "success": None,
                    "spl": None,
                    "navigation_error_m": None,
                    "override_fraction": None,
                    "override_count": None,
                    "action_count": None,
                    "process_status": process.get("status") if process else None,
                }
                rows.append(row)
                failures.append({"method": method, "episode": episode, "reason": reason})
                continue
            valid_attempts = [attempt for attempt in attempts if attempt["status"] == "valid"]
            if len(valid_attempts) == 1:
                row = dict(valid_attempts[0])
                row["attempts"] = attempts
                if len(attempts) > 1:
                    row["attempt_warning"] = "valid_retry_selected_with_failed_attempts"
                    failures.extend(
                        {
                            "method": method,
                            "episode": episode,
                            "source": attempt.get("source"),
                            "reason": "failed_retry_attempt",
                            "failure_reasons": attempt.get("failure_reasons", []),
                        }
                        for attempt in attempts
                        if attempt is not valid_attempts[0]
                    )
                selected[key] = row
            else:
                reason = "duplicate_valid_attempts" if len(valid_attempts) > 1 else "all_attempts_failed"
                row = dict(attempts[0])
                row.update({"status": "ambiguous_duplicate" if len(valid_attempts) > 1 else "failed", "attempts": attempts})
                row["failure_reasons"] = list(dict.fromkeys(row.get("failure_reasons", []) + [reason]))
                rows.append(row)
                failures.append(
                    {
                        "method": method,
                        "episode": episode,
                        "reason": reason,
                        "attempt_count": len(attempts),
                        "sources": [attempt.get("source") for attempt in attempts],
                    }
                )
                continue
            rows.append(row)

    for key, attempts in sorted(grouped.items()):
        if key not in expected_keys:
            failures.append(
                {
                    "method": key[0],
                    "episode": key[1],
                    "reason": "out_of_scope_benchmark_record",
                    "sources": [attempt.get("source") for attempt in attempts],
                }
            )

    methods_summary: dict[str, Any] = {}
    for method in selected_methods:
        method_rows = [row for row in rows if row.get("method") == method]
        valid_rows = [row for row in method_rows if row.get("status") == "valid"]
        spl_values = [float(row["spl"]) for row in valid_rows if row.get("spl") is not None]
        ne_values = [float(row["navigation_error_m"]) for row in valid_rows if row.get("navigation_error_m") is not None]
        override_rows = [row for row in valid_rows if row.get("override_fraction") is not None]
        count_values = [float(row["override_count"]) for row in valid_rows if row.get("override_count") is not None]
        action_values = [float(row["action_count"]) for row in valid_rows if row.get("action_count") is not None]
        methods_summary[method] = {
            "expected_episodes": len(expected_numbers),
            "valid_episodes": len(valid_rows),
            "failed_episodes": sum(row.get("status") in {"failed", "ambiguous_duplicate"} for row in method_rows),
            "missing_episodes": sum(row.get("status") == "missing" for row in method_rows),
            "successes": sum(bool(row.get("success")) for row in valid_rows),
            "sr": (sum(bool(row.get("success")) for row in valid_rows) / len(valid_rows)) if valid_rows else None,
            "spl": statistics.fmean(spl_values) if spl_values else None,
            "navigation_error_m": statistics.fmean(ne_values) if ne_values else None,
            "override_fraction": statistics.fmean(float(row["override_fraction"]) for row in override_rows) if override_rows else None,
            "override_available_episodes": len(override_rows),
            "override_count": sum(count_values) if count_values else None,
            "action_count": sum(action_values) if action_values else None,
            "override_fraction_weighted": (sum(count_values) / sum(action_values)) if count_values and action_values and sum(action_values) > 0 else None,
        }
        methods_summary[method]["success_rate"] = methods_summary[method]["sr"]
        methods_summary[method]["mean_spl"] = methods_summary[method]["spl"]
        methods_summary[method]["mean_navigation_error_m"] = methods_summary[method]["navigation_error_m"]
        methods_summary[method]["pmg_override_fraction"] = methods_summary[method]["override_fraction"]

    paired_rows: list[dict[str, Any]] = []
    for number in expected_numbers:
        episode = f"episode_{number}"
        base = selected.get(("M0", episode))
        candidate = selected.get(("M1", episode))
        if base is None or candidate is None:
            continue
        paired_rows.append(
            {
                "episode": episode,
                "m0_source": base.get("source"),
                "m1_source": candidate.get("source"),
                "sr_delta": float(candidate["success"]) - float(base["success"]),
                "spl_delta": float(candidate["spl"]) - float(base["spl"]),
                "navigation_error_delta_m": float(candidate["navigation_error_m"]) - float(base["navigation_error_m"]),
                "override_delta": (
                    float(candidate["override_fraction"]) - float(base["override_fraction"])
                    if candidate.get("override_fraction") is not None and base.get("override_fraction") is not None
                    else None
                ),
            }
        )

    paired_statistics: dict[str, Any] = {
        "paired_episodes": len(paired_rows),
        "rows": paired_rows,
        "sr": _statistic([row["sr_delta"] for row in paired_rows], seed=seed, bootstrap_samples=bootstrap_samples),
        "spl": _statistic([row["spl_delta"] for row in paired_rows], seed=seed + 1, bootstrap_samples=bootstrap_samples),
        "navigation_error_m": _statistic([row["navigation_error_delta_m"] for row in paired_rows], seed=seed + 2, bootstrap_samples=bootstrap_samples),
        "override_fraction": _statistic(
            [row["override_delta"] for row in paired_rows if row["override_delta"] is not None],
            seed=seed + 3,
            bootstrap_samples=bootstrap_samples,
        ),
    }
    paired_statistics["success_rate"] = paired_statistics["sr"]
    paired_statistics["navigation_error"] = paired_statistics["navigation_error_m"]

    def _ci_pair(statistic: Mapping[str, Any]) -> list[float] | None:
        interval = statistic.get("bootstrap_ci_95")
        if not isinstance(interval, Mapping):
            return None
        return [float(interval["low"]), float(interval["high"])]

    paired_deltas_vs_m0 = {
        "paired_episodes": paired_statistics["paired_episodes"],
        "success_rate_delta": paired_statistics["sr"]["mean_delta"],
        "success_rate_delta_bootstrap_95ci": _ci_pair(paired_statistics["sr"]),
        "success_rate_wilcoxon_pvalue": paired_statistics["sr"]["wilcoxon_pvalue"],
        "mean_spl_delta": paired_statistics["spl"]["mean_delta"],
        "mean_spl_delta_bootstrap_95ci": _ci_pair(paired_statistics["spl"]),
        "mean_spl_wilcoxon_pvalue": paired_statistics["spl"]["wilcoxon_pvalue"],
        "mean_navigation_error_delta_m": paired_statistics["navigation_error_m"]["mean_delta"],
        "mean_navigation_error_delta_bootstrap_95ci": _ci_pair(paired_statistics["navigation_error_m"]),
        "mean_navigation_error_wilcoxon_pvalue": paired_statistics["navigation_error_m"]["wilcoxon_pvalue"],
        "mean_override_fraction_delta": paired_statistics["override_fraction"]["mean_delta"],
        "mean_override_fraction_delta_bootstrap_95ci": _ci_pair(paired_statistics["override_fraction"]),
        "mean_override_fraction_wilcoxon_pvalue": paired_statistics["override_fraction"]["wilcoxon_pvalue"],
    }

    protocol_values: dict[str, list[Any]] = {}
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        for key, value in metadata.items():
            if key == "protocol_caveats":
                continue
            if value not in protocol_values.setdefault(key, []):
                protocol_values[key].append(value)
    manifest_metadata = _manifest_metadata(manifest)
    for key, value in manifest_metadata.items():
        if value not in protocol_values.setdefault(key, []):
            protocol_values[key].append(value)

    caveats = _manifest_caveats(manifest)
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        embedded = metadata.get("protocol_caveats")
        if isinstance(embedded, str):
            caveats.append(embedded)
        elif isinstance(embedded, list):
            caveats.extend(str(item) for item in embedded if item)
    runtime_values = protocol_values.get("compatibility_runner", [])
    if any("isaac_sim_6" in str(value).lower() or "compatibility" in str(value).lower() for value in runtime_values):
        caveats.append("Isaac Sim 6 compatibility runtime is represented; this is not a released dynamic-people runtime.")
    if any(value == 0 for value in protocol_values.get("runtime_num_people", [])):
        caveats.append("runtime_num_people=0 for at least one artifact; official dynamic-people counts are metadata only.")
    strict_flags = [
        protocol_values.get("strict_parity_contract"),
        protocol_values.get("parity_fixed_horizon"),
        protocol_values.get("deterministic_torch_cuda"),
    ]
    if any(not values or any(value is not True for value in values) for values in strict_flags):
        caveats.append("Strict fixed-horizon/deterministic parity is not proven true for every observed artifact.")
    if any(str(value).lower() in {"simple_oracle", "oracle"} for value in protocol_values.get("m1_variant", []) + protocol_values.get("perception_source", [])):
        caveats.append("M1 includes an oracle/simple_oracle perception condition; interpret it as an upper-bound condition.")
    if any(row.get("status") != "valid" for row in rows):
        caveats.append("Failures and missing/ambiguous episodes are excluded from SR/SPL/NE means and remain listed in failures.")
    if any(len(values) > 1 for values in protocol_values.values()):
        caveats.append("Protocol metadata vary across artifacts; inspect protocol_values before pooling results.")
    if len(paired_rows) != len(expected_numbers):
        caveats.append("Paired statistics use only episodes with one valid M0 and one valid M1 artifact.")

    return {
        "schema": "strict_m0_m1_85_report_v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "expected_methods": list(selected_methods),
        "expected_episodes": list(expected_numbers),
        "expected_rows": len(expected_keys),
        "valid_rows": sum(row.get("status") == "valid" for row in rows),
        "complete": len(rows) == len(expected_keys) and all(row.get("status") == "valid" for row in rows),
        "methods": methods_summary,
        "paired_statistics": paired_statistics,
        "paired_deltas_vs_m0": paired_deltas_vs_m0,
        "rows": rows,
        "failures": failures,
        "protocol_values": protocol_values,
        "protocol_caveats": _unique_strings(caveats),
        "progress": {
            "path": str(progress_path) if progress_path else None,
            "payload_metadata": {
                key: value
                for key, value in progress_payload.items()
                if key not in {"runs"} and isinstance(value, (str, int, float, bool))
            },
        },
        "manifest_metadata": manifest_metadata,
        "statistics_config": {"bootstrap_samples": bootstrap_samples, "seed": seed},
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict M0/M1 85 Queue Report",
        "",
        f"Generated: `{report.get('generated', 'unknown')}`",
        f"Completeness: **{'pass' if report.get('complete') else 'incomplete'}** ({report.get('valid_rows', 0)}/{report.get('expected_rows', 0)} valid rows)",
        "",
        "| Method | Valid/expected | SR | SPL | NE (m) | Override |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, values in report.get("methods", {}).items():
        def fmt(key: str) -> str:
            value = values.get(key)
            return "n/a" if value is None else f"{float(value):.4f}"

        lines.append(
            f"| {method} | {values.get('valid_episodes', 0)}/{values.get('expected_episodes', 0)} | "
            f"{fmt('sr')} | {fmt('spl')} | {fmt('navigation_error_m')} | {fmt('override_fraction')} |"
        )
    lines.extend(["", "## Paired Deltas", "", "| Metric | n | Mean delta | Bootstrap 95% CI | Wilcoxon p |", "|---|---:|---:|---|---:|"])
    labels = (("sr", "SR"), ("spl", "SPL"), ("navigation_error_m", "NE (m)"), ("override_fraction", "Override"))
    for key, label in labels:
        values = report.get("paired_statistics", {}).get(key, {})
        ci = values.get("bootstrap_ci_95")
        ci_text = "n/a" if not ci else f"[{float(ci['low']):.4f}, {float(ci['high']):.4f}]"
        mean = values.get("mean_delta")
        lines.append(f"| {label} | {values.get('n', 0)} | {'n/a' if mean is None else f'{float(mean):.4f}'} | {ci_text} | {values.get('wilcoxon_pvalue', 'n/a')} |")
    lines.extend(["", "## Protocol Caveats", ""])
    for caveat in report.get("protocol_caveats", []):
        lines.append(f"- {caveat}")
    lines.extend(["", "## Failures", "", f"Failure records: **{len(report.get('failures', []))}**."])
    for failure in report.get("failures", []):
        identity = ":".join(str(failure.get(key)) for key in ("method", "episode") if failure.get(key) is not None)
        detail = failure.get("detail") or failure.get("reason") or "unknown"
        lines.append(f"- `{identity or failure.get('source', 'unknown')}`: {detail}")
    return "\n".join(lines) + "\n"


def _write_new(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _check_new(paths: Iterable[Path], *, force: bool) -> None:
    if force:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {', '.join(existing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="queue root or one benchmark_metrics.json")
    parser.add_argument("--progress", type=Path, default=None, help="optional runner progress JSON")
    parser.add_argument("--manifest", type=Path, default=None, help="optional workload manifest JSON")
    parser.add_argument("--output", type=Path, default=None, help="JSON report path; stdout is used when omitted")
    parser.add_argument("--markdown-output", type=Path, default=None, help="optional Markdown report path")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--default-method", choices=METHODS, default=None)
    parser.add_argument("--force", action="store_true", help="allow replacing explicitly selected outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] | None = None
    if args.manifest is not None:
        manifest, failure = _load_json(args.manifest, "manifest")
        if failure is not None:
            raise SystemExit(json.dumps(failure, sort_keys=True))
    report = aggregate_queue(
        args.root.resolve(),
        progress_path=args.progress.resolve() if args.progress else None,
        manifest=manifest,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        default_method=args.default_method,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    try:
        output_paths = [path.resolve() for path in (args.output, args.markdown_output) if path is not None]
        _check_new(output_paths, force=args.force)
        if args.output is None:
            print(encoded)
        else:
            _write_new(args.output.resolve(), encoded + "\n", force=args.force)
        if args.markdown_output is not None:
            _write_new(args.markdown_output.resolve(), render_markdown(report), force=args.force)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
