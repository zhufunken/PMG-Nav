"""Deterministic, fixture-backed perception hook for memory-dependent pilots.

The hook is deliberately independent of Isaac Sim and TIC-VLA.  A JSONL row
describes one episode's object catalog and a short sequence of visibility
events.  The Isaac Sim 6 compatibility runner may feed those events to PMG in
place of its USD catalog, which makes occlusion, reappearance and contradictory
observations reproducible without downloading or authoring another scene.

The module does not import ``pxr``.  It is therefore safe to validate and unit
test on a workstation without Isaac Sim installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constraints import Constraint, ConstraintProgram


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_position(value: Any, *, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be a 3-vector")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite values")
    return result


def _distance(left: Iterable[float], right: Iterable[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _copy_record(record: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{field} must contain mappings")
    node_id = str(record.get("id", record.get("node_id", ""))).strip()
    if not node_id:
        raise ValueError(f"{field} is missing id")
    label = str(record.get("label", node_id)).strip()
    if not label:
        raise ValueError(f"{field} is missing label")
    position = _finite_position(record.get("position"), field=f"{field}.position")
    try:
        confidence = float(record.get("confidence", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}.confidence must be numeric") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{field}.confidence must be in [0, 1]")
    return {
        "node_id": node_id,
        "label": label,
        "position": position,
        "confidence": confidence,
    }


@dataclass(frozen=True)
class MemoryHookObservation:
    """One logical observation sampled by PMG at a frame-stride boundary."""

    frame: int
    visible: tuple[dict[str, Any], ...]
    occluded: tuple[str, ...]
    contradictions: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "visible": [
                {
                    "id": item["node_id"],
                    "label": item["label"],
                    "position": list(item["position"]),
                    "confidence": item["confidence"],
                }
                for item in self.visible
            ],
            "occluded": list(self.occluded),
            "contradictions": [
                {
                    "id": item["node_id"],
                    "label": item["label"],
                    "position": list(item["position"]),
                    "confidence": item["confidence"],
                }
                for item in self.contradictions
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryHookEpisode:
    """Validated fixture row exposed to the PMG runtime."""

    episode_id: str
    instruction: str
    relation_program: Mapping[str, Any]
    objects: Mapping[str, tuple[float, float, float]]
    object_labels: Mapping[str, str]
    history: tuple[MemoryHookObservation, ...]
    oracle_goal_pose: tuple[float, float, float] | None
    requirements: Mapping[str, Any]
    verification: Mapping[str, Any]
    source_path: str
    row_digest: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, source_path: str = "") -> "MemoryHookEpisode":
        if not isinstance(row, Mapping):
            raise ValueError("memory-hook row must be a mapping")
        episode_id = str(row.get("id", row.get("episode", ""))).strip()
        if not episode_id:
            raise ValueError("memory-hook row is missing id")
        instruction = str(row.get("instruction", "")).strip()
        if not instruction:
            raise ValueError(f"memory-hook {episode_id} is missing instruction")
        raw_objects = row.get("objects")
        if not isinstance(raw_objects, Mapping) or not raw_objects:
            raise ValueError(f"memory-hook {episode_id} has no objects")
        objects: dict[str, tuple[float, float, float]] = {}
        object_labels: dict[str, str] = {}
        for key, value in raw_objects.items():
            node_id = str(key).strip()
            if not node_id:
                raise ValueError(f"memory-hook {episode_id} has an empty object id")
            objects[node_id] = _finite_position(value, field=f"objects.{node_id}")
            # A fixture can provide explicit labels; default to the stable id.
            explicit_label = (row.get("object_labels") or {}).get(node_id)
            object_labels[node_id] = str(explicit_label).strip() if explicit_label else ""

        raw_history = row.get("history")
        if not isinstance(raw_history, list) or not raw_history:
            raise ValueError(f"memory-hook {episode_id} has no history")
        history: list[MemoryHookObservation] = []
        previous_frame = -1
        for index, item in enumerate(raw_history):
            if not isinstance(item, Mapping):
                raise ValueError(f"memory-hook {episode_id} history[{index}] is not a mapping")
            try:
                frame = int(item.get("frame"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"memory-hook {episode_id} history[{index}] frame is invalid") from exc
            if frame < 0 or frame <= previous_frame:
                raise ValueError(
                    f"memory-hook {episode_id} history frames must be strictly increasing"
                )
            previous_frame = frame
            visible = tuple(
                _copy_record(record, field=f"history[{index}].visible[{record_index}]")
                for record_index, record in enumerate(item.get("visible", ()))
            )
            contradictions = tuple(
                _copy_record(record, field=f"history[{index}].contradictions[{record_index}]")
                for record_index, record in enumerate(item.get("contradictions", ()))
            )
            occluded = tuple(sorted(str(value) for value in item.get("occluded", ())))
            history.append(
                MemoryHookObservation(
                    frame=frame,
                    visible=visible,
                    occluded=occluded,
                    contradictions=contradictions,
                    metadata={
                        str(key): value
                        for key, value in item.items()
                        if key not in {"frame", "visible", "occluded", "contradictions"}
                    },
                )
            )

        # Every observed ID must have a catalog position.  This catches a
        # typo in a fixture before it can silently create a new PMG entity.
        for observation in history:
            for record in (*observation.visible, *observation.contradictions):
                if record["node_id"] not in objects:
                    raise ValueError(
                        f"memory-hook {episode_id} observation references unknown id "
                        f"{record['node_id']!r}"
                    )
                # History labels are the detector-facing semantic identity.
                # When the compact fixture omits ``object_labels``, derive the
                # per-instance label from the first observation rather than
                # exposing the synthetic object id as a noun phrase.
                if node_id := record["node_id"]:
                    if not object_labels.get(node_id):
                        object_labels[node_id] = str(record["label"]).strip() or node_id
        for node_id in objects:
            object_labels.setdefault(node_id, node_id)
            if not object_labels[node_id]:
                object_labels[node_id] = node_id

        goal_value = row.get("oracle_goal_pose")
        goal = _finite_position(goal_value, field="oracle_goal_pose") if goal_value is not None else None
        digest_payload = dict(row)
        return cls(
            episode_id=episode_id,
            instruction=instruction,
            relation_program=dict(row.get("relation_program") or {}),
            objects=objects,
            object_labels=object_labels,
            history=tuple(history),
            oracle_goal_pose=goal,
            requirements=dict(row.get("requirements") or {}),
            verification=dict(row.get("verification") or {}),
            source_path=str(source_path),
            row_digest=_canonical_digest(digest_payload),
        )

    @property
    def history_required(self) -> bool:
        return bool(self.requirements.get("history_is_necessary_for_success", True))

    def constraint_program(self, fallback: ConstraintProgram | None) -> ConstraintProgram | None:
        """Attach the fixture's held-out relation family to a typed program.

        The compact fixture's ``hops`` describe object-to-object evidence;
        they are retained as metadata because ``ConstraintEvaluator`` scores
        robot-to-anchor constraints.  The metric clause is represented as a
        normal typed constraint when possible, while a compiled instruction
        remains the action-facing fallback.  No evaluator goal is introduced.
        """
        raw_metric = self.relation_program.get("metric")
        metric = dict(raw_metric) if isinstance(raw_metric, Mapping) else {}
        anchor = str(metric.get("anchor", "")).strip()
        relation = str(metric.get("relation", "")).strip().lower()
        constraints: list[Constraint] = []
        if anchor and relation in {"near_distance", "near"}:
            constraints.append(Constraint("near", (anchor,), value=float(metric.get("value", 2.0))))
        elif anchor and relation in {"metric_distance", "distance"}:
            constraints.append(
                Constraint(
                    "distance",
                    (anchor,),
                    value=max(0.0, float(metric.get("value", 0.5))),
                    tolerance=max(0.2, float(metric.get("tolerance", 0.2))),
                )
            )
        if not constraints:
            return fallback
        metadata = {
            **(dict(fallback.metadata) if fallback is not None else {}),
            "source": "memory_hook_constraint_program",
            "memory_hook_relation_program": dict(self.relation_program),
            "memory_hook_history_required": self.history_required,
        }
        return ConstraintProgram(
            self.instruction,
            tuple(constraints),
            metadata=metadata,
        )

    @property
    def first_frame(self) -> int:
        return self.history[0].frame

    @property
    def last_frame(self) -> int:
        return self.history[-1].frame

    def catalog_items(self, robot_position: Iterable[float]) -> list[dict[str, Any]]:
        """Return the complete deterministic catalog in PMG's item schema."""
        robot = tuple(float(value) for value in robot_position)
        return [
            {
                "node_id": node_id,
                "label": self.object_labels.get(node_id, node_id),
                "position": position,
                "distance": _distance(position, robot),
                "confidence": 1.0,
                "source": "memory_hook_catalog",
            }
            for node_id, position in sorted(self.objects.items())
        ]

    def _history_index(self, frame_id: int, frame_stride: int) -> int:
        logical_frame = max(0, int(frame_id)) // max(1, int(frame_stride))
        selected = 0
        for index, item in enumerate(self.history):
            if item.frame <= logical_frame:
                selected = index
            else:
                break
        return selected

    def observation_for(self, frame_id: int, *, frame_stride: int = 15) -> MemoryHookObservation:
        return self.history[self._history_index(frame_id, frame_stride)]

    def visible_items(
        self,
        robot_position: Iterable[float],
        frame_id: int,
        *,
        frame_stride: int = 15,
    ) -> tuple[list[dict[str, Any]], MemoryHookObservation]:
        """Return current visible records plus explicit contradictions.

        Contradictions are kept as the same stable ``node_id`` with their
        supplied confidence.  PMG's normal association/fusion path therefore
        receives exactly the same identity ambiguity as a detector would, while
        the fixture remains deterministic and auditable.
        """
        robot = tuple(float(value) for value in robot_position)
        observation = self.observation_for(frame_id, frame_stride=frame_stride)
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in (*observation.visible, *observation.contradictions):
            record = dict(item)
            record["distance"] = _distance(record["position"], robot)
            record["source"] = (
                "memory_hook_contradiction"
                if item in observation.contradictions
                else "memory_hook_observation"
            )
            # A single frame should not duplicate the same stable track.  The
            # contradiction wins only when the fixture does not also publish a
            # visible record for that ID.
            if record["node_id"] in seen:
                continue
            seen.add(record["node_id"])
            records.append(record)
        return records, observation

    def provenance(self) -> dict[str, Any]:
        return {
            "active": True,
            "episode_id": self.episode_id,
            "source_path": self.source_path,
            "row_digest": self.row_digest,
            "history_required": self.history_required,
            "history_frames": [item.frame for item in self.history],
            "catalog_objects": len(self.objects),
            "requirements": dict(self.requirements),
            "verification": dict(self.verification),
        }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"memory-hook fixture is missing: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("episodes"), list):
            rows = payload["episodes"]
        elif isinstance(payload, Mapping):
            rows = [payload]
        else:
            raise ValueError(f"unsupported memory-hook payload: {path}")
    if not rows:
        raise ValueError(f"memory-hook fixture is empty: {path}")
    return [dict(row) for row in rows]


def load_memory_hook(path: str | Path, *, episode_id: str | None = None) -> MemoryHookEpisode:
    """Load one deterministic episode row from a JSON/JSONL fixture."""
    resolved = Path(path).expanduser().resolve()
    rows = _load_rows(resolved)
    selected: Mapping[str, Any] | None = None
    if episode_id is not None:
        wanted = str(episode_id)
        for row in rows:
            if str(row.get("id", row.get("episode", ""))) == wanted:
                selected = row
                break
        if selected is None:
            available = [str(row.get("id", row.get("episode", ""))) for row in rows]
            raise KeyError(f"memory-hook episode {wanted!r} is missing; available={available}")
    else:
        selected = rows[0]
    return MemoryHookEpisode.from_row(selected, source_path=str(resolved))


__all__ = ["MemoryHookEpisode", "MemoryHookObservation", "load_memory_hook"]
