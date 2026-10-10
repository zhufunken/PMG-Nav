#!/usr/bin/env python3
"""Deterministic benchmark subset selection and subset-gate helpers.

The official benchmark repeats the same route under several ``num_people``
conditions.  This module treats those values as metadata only: selection is
stratified by scene, a stable route-family key, and the official people count,
while episode order and the random seed remain reproducible.

The module intentionally has no dependency on the PMG runtime.  It can be
used by a local selector, by the official runner, and by a lightweight gate
before an expensive full-matrix run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence


# The first optimization loop is intentionally the smallest causal comparison:
# locked baseline versus one single-frame PMG variant.  Later stages must opt
# into M2/M3 explicitly in their manifest.
DEFAULT_METHODS = ("M0", "M1")
DEFAULT_ROUTE_QUANTIZATION_M = 10.0
MANIFEST_SCHEMA_VERSION = 1


class BenchmarkFormatError(ValueError):
    """Raised when benchmark metadata cannot be used safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _parse_scalar(value: str) -> Any:
    """Parse the small scalar subset used by benchmark_full.yaml.

    This is only a last-resort fallback for machines without PyYAML.  The
    primary parser below is PyYAML, and the next fallback delegates to Ruby's
    standard YAML parser.  Keeping this fallback narrow is safer than silently
    treating a malformed benchmark as a valid one.
    """

    value = value.strip()
    if not value:
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith(("'", '"')) and value[-1:] == value[0]:
        return value[1:-1]
    try:
        if any(token in value.lower() for token in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _minimal_benchmark_yaml(text: str) -> dict[str, Any]:
    """Parse normalized benchmark YAML when optional parsers are unavailable.

    ``benchmark_full.yaml`` is a top-level mapping whose episode values are
    mappings with scalar fields and block lists for ``start`` and ``goal``.
    The fallback accepts exactly that shape and raises on anything else.
    """

    result: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    current_key: str | None = None
    pending_list: list[Any] | None = None
    pending_indent = -1

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if stripped.startswith("-"):
            # Block sequence items in the normalized benchmark are aligned
            # with the mapping key (``  start:`` / ``  - 1.0``), so equal
            # indentation is valid here.
            if pending_list is None or indent < pending_indent:
                raise BenchmarkFormatError(f"unsupported YAML list indentation: {raw_line!r}")
            item = stripped[1:].strip()
            if not item:
                raise BenchmarkFormatError(f"nested list item is not supported: {raw_line!r}")
            pending_list.append(_parse_scalar(item))
            continue
        if ":" not in stripped:
            raise BenchmarkFormatError(f"unsupported YAML line: {raw_line!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if indent == 0:
            if pending_list is not None:
                pending_list = None
            if raw_value:
                result[key] = _parse_scalar(raw_value)
                current = None
                current_key = None
            else:
                current = {}
                result[key] = current
                current_key = key
            continue
        if current is None or current_key is None or not current_key.startswith("episode_"):
            raise BenchmarkFormatError(f"unexpected nested value: {raw_line!r}")
        if raw_value:
            current[key] = _parse_scalar(raw_value)
            pending_list = None
        else:
            pending_list = []
            current[key] = pending_list
            pending_indent = indent

    episodes = {key: value for key, value in result.items() if key.startswith("episode_")}
    if not episodes:
        raise BenchmarkFormatError("benchmark has no episode_* entries")
    return result


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with PyYAML, Ruby stdlib, or a narrow fallback."""

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        value = yaml.safe_load(text)
        if not isinstance(value, dict):
            raise BenchmarkFormatError(f"expected mapping in {path}")
        return value

    ruby = shutil.which("ruby")
    if ruby:
        command = [
            ruby,
            "-ryaml",
            "-rjson",
            "-e",
            "puts JSON.generate(YAML.load_file(ARGV.fetch(0)))",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            try:
                value = json.loads(completed.stdout)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                return value

    return _minimal_benchmark_yaml(text)


def dump_yaml_mapping(value: Mapping[str, Any]) -> str:
    """Serialize runner config mappings with the same parser fallbacks."""

    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        return yaml.safe_dump(dict(value), sort_keys=False)

    ruby = shutil.which("ruby")
    if ruby:
        completed = subprocess.run(
            [ruby, "-ryaml", "-rjson", "-e", "puts YAML.dump(JSON.parse(STDIN.read))"],
            input=json.dumps(value, ensure_ascii=True),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout:
            return completed.stdout

    # The AutoDL base image intentionally keeps the system Python minimal and
    # does not ship either PyYAML or Ruby.  Do not force the matrix launcher to
    # run inside Isaac Python just to write its tiny per-episode config: doing
    # so contaminates nested Isaac processes with Kit's Python 3.12 modules
    # and can raise ``_sre.MAGIC`` mismatches.  The runner config has a very
    # small, JSON-like shape, so a strict dependency-free block dumper is both
    # safer and reproducible.  Strings are emitted as JSON double-quoted
    # scalars (valid YAML 1.2), which handles URLs, punctuation and Unicode
    # instructions without relying on YAML quoting heuristics.
    def scalar(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=True)
        if isinstance(item, (int, float)):
            return repr(item)
        raise BenchmarkFormatError(
            f"dependency-free YAML writer cannot serialize scalar {type(item).__name__}"
        )

    def emit_mapping(mapping: Mapping[str, Any], indent: int) -> list[str]:
        lines: list[str] = []
        prefix = " " * indent
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise BenchmarkFormatError("YAML mapping keys must be strings")
            key_text = json.dumps(key, ensure_ascii=True)
            if isinstance(item, Mapping):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(emit_mapping(item, indent + 2))
            elif isinstance(item, (list, tuple)):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(emit_sequence(item, indent + 2))
            else:
                lines.append(f"{prefix}{key_text}: {scalar(item)}")
        return lines

    def emit_sequence(sequence: Sequence[Any], indent: int) -> list[str]:
        lines: list[str] = []
        prefix = " " * indent
        for item in sequence:
            if isinstance(item, Mapping):
                lines.append(f"{prefix}-")
                lines.extend(emit_mapping(item, indent + 2))
            elif isinstance(item, (list, tuple)):
                lines.append(f"{prefix}-")
                lines.extend(emit_sequence(item, indent + 2))
            else:
                lines.append(f"{prefix}- {scalar(item)}")
        return lines

    if not isinstance(value, Mapping):
        raise BenchmarkFormatError("runner YAML root must be a mapping")
    return "\n".join(emit_mapping(value, 0)) + "\n"


def episode_items(source: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Return episode entries ordered by their numeric episode id."""

    entries: list[tuple[int, str, Mapping[str, Any]]] = []
    for key, value in source.items():
        if not str(key).startswith("episode_"):
            continue
        try:
            index = int(str(key).rsplit("_", 1)[1])
        except (ValueError, IndexError):
            raise BenchmarkFormatError(f"invalid episode key: {key!r}") from None
        if not isinstance(value, Mapping):
            raise BenchmarkFormatError(f"episode {key!r} is not a mapping")
        entries.append((index, str(key), value))
    if not entries:
        raise BenchmarkFormatError("benchmark has no episode_* entries")
    entries.sort(key=lambda item: (item[0], item[1]))
    return [(key, value) for _, key, value in entries]


def _number(value: Any, field: str, episode: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise BenchmarkFormatError(f"{episode}: {field} must be numeric") from None
    if not math.isfinite(result):
        raise BenchmarkFormatError(f"{episode}: {field} must be finite")
    return result


def _vector(value: Any, field: str, episode: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BenchmarkFormatError(f"{episode}: {field} must be a coordinate list")
    return tuple(_number(item, field, episode) for item in value)


def _quantize(value: float, quantum: float) -> int:
    # Avoid Python's banker's rounding at exact half-quanta.  This is stable
    # across Python versions and works for negative world coordinates too.
    scaled = value / quantum
    return math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)


def normalize_scene(scene: Any) -> str:
    """Use a stable scene label while preserving the source value in metadata."""

    text = str(scene).strip()
    if not text:
        return "<missing-scene>"
    return text.replace("\\", "/").rsplit("/", 1)[-1]


def route_family_key(
    episode: Mapping[str, Any],
    *,
    episode_key: str = "<episode>",
    quantization_m: float = DEFAULT_ROUTE_QUANTIZATION_M,
) -> str:
    """Build a stable route-family id from scene and coarse geometry.

    Dynamic-person variants and wording edits should remain in one route
    family.  Scene plus quantized start/goal coordinates preserve distinct
    routes while merging tiny benchmark perturbations (for example a 1 m
    start offset).  ``start_yaw`` and instruction are intentionally excluded:
    they are conditions of the same route family, not new map routes.
    """

    if quantization_m <= 0 or not math.isfinite(quantization_m):
        raise ValueError("quantization_m must be a positive finite number")
    scene = normalize_scene(episode.get("scene", ""))
    start = _vector(episode.get("start"), "start", episode_key)
    goal = _vector(episode.get("goal"), "goal", episode_key)
    if len(start) < 2 or len(goal) < 2:
        raise BenchmarkFormatError(f"{episode_key}: start and goal need x/y coordinates")
    start_key = ",".join(str(_quantize(value, quantization_m)) for value in start[:2])
    goal_key = ",".join(str(_quantize(value, quantization_m)) for value in goal[:2])
    robot = str(episode.get("robot_type", "")).strip() or "<missing-robot>"
    raw = f"scene={scene}|start={start_key}|goal={goal_key}|robot={robot}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"route_{digest}"


@dataclass(frozen=True)
class EpisodeRecord:
    key: str
    index: int
    scene: str
    route_family: str
    official_num_people: int
    tie_break: str
    source: Mapping[str, Any]


def _tie_break(seed: int, key: str) -> str:
    return hashlib.sha256(f"subset-v1|{seed}|{key}".encode("utf-8")).hexdigest()


def build_records(
    source: Mapping[str, Any],
    *,
    seed: int,
    quantization_m: float = DEFAULT_ROUTE_QUANTIZATION_M,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for key, episode in episode_items(source):
        try:
            index = int(key.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            raise BenchmarkFormatError(f"invalid episode key: {key!r}") from None
        people_value = episode.get("num_people", episode.get("official_num_people", 0))
        try:
            people_float = float(people_value)
            people = int(people_float)
        except (TypeError, ValueError):
            raise BenchmarkFormatError(f"{key}: num_people must be an integer") from None
        if people_float != people:
            raise BenchmarkFormatError(f"{key}: num_people must be an integer")
        records.append(
            EpisodeRecord(
                key=key,
                index=index,
                scene=normalize_scene(episode.get("scene", "")),
                route_family=route_family_key(episode, episode_key=key, quantization_m=quantization_m),
                official_num_people=people,
                tie_break=_tie_break(seed, key),
                source=episode,
            )
        )
    return records


def _dimension_values(record: EpisodeRecord) -> tuple[tuple[str, str], ...]:
    return (
        ("scene", record.scene),
        ("route_family", record.route_family),
        ("official_num_people", str(record.official_num_people)),
    )


def _counts(records: Iterable[EpisodeRecord]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {"scene": {}, "route_family": {}, "official_num_people": {}}
    for record in records:
        for dimension, value in _dimension_values(record):
            result[dimension][value] = result[dimension].get(value, 0) + 1
    return result


def select_records(records: Sequence[EpisodeRecord], size: int) -> list[EpisodeRecord]:
    """Select a deterministic, coverage-first representative subset.

    The greedy score first rewards unseen values in each stratification
    dimension.  Once the subset covers a value, it rewards the least-covered
    value in that dimension.  A cryptographic tie-break derived from the
    caller's seed makes every decision stable without relying on hash random
    seeding or process order.
    """

    if size <= 0:
        raise ValueError("subset size must be positive")
    if not records:
        raise ValueError("cannot select from an empty benchmark")
    target = min(size, len(records))
    available = list(records)
    population_counts = _counts(available)
    selected: list[EpisodeRecord] = []
    selected_keys: set[str] = set()
    selected_counts: dict[str, Counter[str]] = {
        "scene": Counter(),
        "route_family": Counter(),
        "official_num_people": Counter(),
    }
    while len(selected) < target:
        best: EpisodeRecord | None = None
        best_score: tuple[float, float, float, str] | None = None
        for record in available:
            if record.key in selected_keys:
                continue
            gains: list[float] = []
            balance: list[float] = []
            for dimension, value in _dimension_values(record):
                count = selected_counts[dimension][value]
                population = population_counts[dimension][value]
                gains.append(1.0 if count == 0 else 0.0)
                # A rare category should be filled before a common category
                # once both are already represented.
                balance.append(1.0 / (1.0 + count) / max(1, population))
            # Maximize unseen dimensions first, then balance all dimensions,
            # then use the seed-derived tie break as the final stable order.
            score = (sum(gains), sum(balance), record.tie_break)
            if best_score is None or score > best_score:
                best = record
                best_score = score
        if best is None:
            break
        selected.append(best)
        selected_keys.add(best.key)
        for dimension, value in _dimension_values(best):
            selected_counts[dimension][value] += 1
    return sorted(selected, key=lambda record: record.index)


def _stratification_summary(records: Sequence[EpisodeRecord], selected: Sequence[EpisodeRecord]) -> dict[str, dict[str, dict[str, int]]]:
    all_counts = _counts(records)
    selected_counts = _counts(selected)
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for dimension in ("scene", "route_family", "official_num_people"):
        summary[dimension] = {}
        for value in sorted(all_counts[dimension], key=str):
            summary[dimension][value] = {
                "available": int(all_counts[dimension][value]),
                "selected": int(selected_counts[dimension].get(value, 0)),
            }
    return summary


def make_manifest(
    source: Mapping[str, Any],
    *,
    benchmark_path: Path,
    seed: int,
    size: int,
    methods: Sequence[str] = DEFAULT_METHODS,
    quantization_m: float = DEFAULT_ROUTE_QUANTIZATION_M,
) -> dict[str, Any]:
    records = build_records(source, seed=seed, quantization_m=quantization_m)
    selected = select_records(records, size)
    path_text = str(benchmark_path)
    source_digest = sha256_file(benchmark_path)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "pmg_nav_subset_manifest",
        "benchmark_config": path_text,
        "benchmark_sha256": source_digest,
        "benchmark_episode_count": len(records),
        "seed": int(seed),
        "methods": list(methods),
        "selection": {
            "requested_size": int(size),
            "selected_size": len(selected),
            "available_route_family_count": len({record.route_family for record in records}),
            "selected_route_family_count": len({record.route_family for record in selected}),
            "route_family_coverage_complete": len({record.route_family for record in selected})
            == len({record.route_family for record in records}),
            "route_family_definition": "scene + 10m-quantized start/goal x/y + robot_type; instruction, yaw, timeout and people are conditions",
            "route_quantization_m": float(quantization_m),
            "stratification_dimensions": ["scene", "route_family", "official_num_people"],
            "tie_break": "sha256(subset-v1|seed|episode_key)",
        },
        "episode_keys": [record.key for record in selected],
        "episode_indices": [record.index for record in selected],
        "episodes": [
            {
                "episode": record.key,
                "index": record.index,
                "scene": record.scene,
                "route_family": record.route_family,
                "official_num_people": record.official_num_people,
            }
            for record in selected
        ],
        "strata": _stratification_summary(records, selected),
    }
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkFormatError(f"cannot read subset manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("kind") != "pmg_nav_subset_manifest":
        raise BenchmarkFormatError(f"{path} is not a PMG subset manifest")
    keys = value.get("episode_keys")
    if not isinstance(keys, list) or not keys or not all(isinstance(key, str) for key in keys):
        raise BenchmarkFormatError("manifest episode_keys must be a non-empty list")
    if len(set(keys)) != len(keys):
        raise BenchmarkFormatError("manifest episode_keys contains duplicates")
    return value


def manifest_episode_indices(manifest: Mapping[str, Any]) -> list[int]:
    """Return episode ids for runner consumption, validating key/index parity."""

    keys = manifest.get("episode_keys")
    if not isinstance(keys, list) or not keys:
        raise BenchmarkFormatError("manifest episode_keys must be a non-empty list")
    indices: list[int] = []
    for key in keys:
        if not isinstance(key, str) or not key.startswith("episode_"):
            raise BenchmarkFormatError(f"invalid manifest episode key: {key!r}")
        try:
            index = int(key.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            raise BenchmarkFormatError(f"invalid manifest episode key: {key!r}") from None
        if index < 1:
            raise BenchmarkFormatError(f"manifest episode index must be positive: {key!r}")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise BenchmarkFormatError("manifest episode indices contain duplicates")
    return indices


def _episode_number(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.startswith("episode_"):
        try:
            return int(value.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return None
    return None


def evaluate_subset_gate(
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    candidate_method: str = "M1",
    baseline_method: str = "M0",
    require_improvement: bool = True,
    allow_equal: bool = True,
    min_success_rate_delta: float = 0.0,
    min_spl_delta: float = 0.0,
    max_navigation_error_delta: float = 0.0,
) -> dict[str, Any]:
    """Evaluate completeness and paired metric criteria for a subset result."""

    expected_keys = list(manifest_episode_indices(manifest))
    raw_methods = manifest.get("methods", list(DEFAULT_METHODS))
    if not isinstance(raw_methods, list) or not raw_methods or not all(isinstance(method, str) and method for method in raw_methods):
        expected_methods: list[str] = []
        reasons: list[str] = ["manifest methods must be a non-empty list of strings"]
    elif len(set(raw_methods)) != len(raw_methods):
        expected_methods = list(raw_methods)
        reasons = ["manifest methods contains duplicates"]
    else:
        expected_methods = list(raw_methods)
        reasons = []
    rows = result.get("rows")
    checks: dict[str, Any] = {}
    if isinstance(rows, list):
        expected_pairs = {(method, index) for method in expected_methods for index in expected_keys}
        observed_pairs: list[tuple[str | None, int | None]] = []
        invalid_rows: list[int] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                invalid_rows.append(row_index)
                continue
            method = row.get("method")
            episode = _episode_number(row.get("episode"))
            if not isinstance(method, str) or not method or episode is None:
                invalid_rows.append(row_index)
                continue
            observed_pairs.append((method, episode))
            # Older hand-built summaries omitted the ``valid`` field, so its
            # absence remains compatible; when present it must be true.
            if "valid" in row and row.get("valid") is not True:
                invalid_rows.append(row_index)
        seen = set(observed_pairs)
        duplicate_pairs = sorted(
            [f"{method}:episode_{episode}" for (method, episode), count in Counter(observed_pairs).items() if count > 1]
        )
        unexpected_pairs = sorted(
            [f"{method}:episode_{episode}" for method, episode in seen if (method, episode) not in expected_pairs]
        )
        missing_pairs = sorted(
            [f"{method}:episode_{index}" for method, index in expected_pairs if (method, index) not in seen]
        )
        checks["missing_method_episode_pairs"] = missing_pairs
        checks["duplicate_method_episode_pairs"] = duplicate_pairs
        checks["unexpected_method_episode_pairs"] = unexpected_pairs
        checks["invalid_result_rows"] = invalid_rows
        checks["result_rows"] = len(rows)
        if missing_pairs:
            reasons.append(f"missing {len(missing_pairs)} method/episode result pairs")
        if duplicate_pairs:
            reasons.append(f"duplicate {len(duplicate_pairs)} method/episode result pairs")
        if unexpected_pairs:
            reasons.append(f"unexpected {len(unexpected_pairs)} method/episode result pairs")
        if invalid_rows:
            reasons.append(f"invalid {len(invalid_rows)} result rows")
        if len(rows) != len(expected_pairs):
            reasons.append(
                f"result row count {len(rows)} does not equal expected {len(expected_pairs)}"
            )
    else:
        checks["missing_method_episode_pairs"] = None
        checks["result_rows"] = None
        reasons.append("result JSON has no rows list; completeness cannot be audited")

    methods = result.get("methods")
    if not isinstance(methods, Mapping):
        reasons.append("result JSON has no methods summary")
        methods = {}
    base = methods.get(baseline_method)
    candidate = methods.get(candidate_method)
    if not isinstance(base, Mapping) or not isinstance(candidate, Mapping):
        reasons.append(f"missing method summary: {baseline_method} or {candidate_method}")
        base = base if isinstance(base, Mapping) else {}
        candidate = candidate if isinstance(candidate, Mapping) else {}

    def metric(payload: Mapping[str, Any], key: str) -> float | None:
        value = payload.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    base_success = metric(base, "success_rate")
    cand_success = metric(candidate, "success_rate")
    base_spl = metric(base, "mean_spl")
    cand_spl = metric(candidate, "mean_spl")
    base_nav = metric(base, "mean_navigation_error_m")
    cand_nav = metric(candidate, "mean_navigation_error_m")
    deltas = {
        "success_rate": None if base_success is None or cand_success is None else cand_success - base_success,
        "mean_spl": None if base_spl is None or cand_spl is None else cand_spl - base_spl,
        "mean_navigation_error_m": None if base_nav is None or cand_nav is None else cand_nav - base_nav,
    }
    checks["metric_deltas"] = deltas
    checks["baseline_method"] = baseline_method
    checks["candidate_method"] = candidate_method
    if baseline_method not in expected_methods:
        reasons.append(f"baseline method {baseline_method!r} is not listed in the subset manifest")
    if candidate_method not in expected_methods:
        reasons.append(f"candidate method {candidate_method!r} is not listed in the subset manifest")
    checks["criteria"] = {
        "min_success_rate_delta": float(min_success_rate_delta),
        "min_spl_delta": float(min_spl_delta),
        "max_navigation_error_delta": float(max_navigation_error_delta),
        "require_improvement": bool(require_improvement),
        "allow_equal": bool(allow_equal),
    }

    required_metrics = (deltas["success_rate"], deltas["mean_spl"], deltas["mean_navigation_error_m"])
    if any(value is None for value in required_metrics):
        reasons.append("candidate/baseline metric summaries are incomplete")
    else:
        success_delta, spl_delta, nav_delta = required_metrics
        assert success_delta is not None and spl_delta is not None and nav_delta is not None
        if allow_equal:
            success_ok = success_delta >= min_success_rate_delta
            spl_ok = spl_delta >= min_spl_delta
            nav_ok = nav_delta <= max_navigation_error_delta
        else:
            success_ok = success_delta > min_success_rate_delta
            spl_ok = spl_delta > min_spl_delta
            nav_ok = nav_delta < max_navigation_error_delta
        if not success_ok:
            reasons.append(f"success-rate delta {success_delta:.6f} is below threshold")
        if not spl_ok:
            reasons.append(f"SPL delta {spl_delta:.6f} is below threshold")
        if not nav_ok:
            reasons.append(f"navigation-error delta {nav_delta:.6f} is above threshold")
        if require_improvement and not (success_delta > 0.0 or spl_delta > 0.0 or nav_delta < 0.0):
            reasons.append("no positive change on any primary metric")

    complete_flag = result.get("complete")
    if complete_flag is not True:
        reasons.append("result summary must mark the subset complete=true")
    checks["result_complete_flag"] = complete_flag
    passed = not reasons
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "pmg_nav_subset_gate",
        "passed": passed,
        "manifest_episode_count": len(expected_keys),
        "manifest_episode_keys": [f"episode_{index}" for index in expected_keys],
        "checks": checks,
        "reasons": reasons,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
