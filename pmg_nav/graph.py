"""Serializable, provenance-preserving Physical Memory Graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Evidence:
    source: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: float = 1.0
    frame_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectNode:
    node_id: str
    label: str
    position: tuple[float, float, float]
    covariance: tuple[float, float, float] = (0.25, 0.25, 0.25)
    confidence: float = 1.0
    observations: int = 1
    aliases: set[str] = field(default_factory=set)
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["aliases"] = sorted(self.aliases)
        return result


class PhysicalMemoryGraph:
    def __init__(self, association_radius: float = 1.0, max_evidence_per_node: int | None = None) -> None:
        self.association_radius = association_radius
        self.max_evidence_per_node = max_evidence_per_node
        self.objects: dict[str, ObjectNode] = {}
        self.relations: dict[str, dict[str, Any]] = {}

    def observe(
        self,
        label: str,
        position: tuple[float, float, float],
        evidence: Evidence,
        *,
        node_id: str | None = None,
    ) -> ObjectNode:
        label = self.normalize_label(label)
        match = self._associate(label, position, node_id)
        if match is None:
            resolved_id = node_id or f"{label.replace(' ', '_')}:{len(self.objects):04d}"
            node = ObjectNode(resolved_id, label, tuple(map(float, position)), confidence=evidence.confidence)
            node.evidence.append(evidence)
            self.objects[resolved_id] = node
            return node
        old_weight = max(match.confidence * match.observations, 1e-6)
        new_weight = max(evidence.confidence, 1e-6)
        match.position = tuple(
            (old_weight * old + new_weight * new) / (old_weight + new_weight)
            for old, new in zip(match.position, position)
        )
        match.observations += 1
        match.confidence = 1.0 - (1.0 - match.confidence) * (1.0 - evidence.confidence)
        match.covariance = tuple(max(value * 0.8, 0.01) for value in match.covariance)
        match.evidence.append(evidence)
        if self.max_evidence_per_node is not None:
            match.evidence = match.evidence[-max(1, self.max_evidence_per_node):]
        return match

    def _associate(self, label: str, position: tuple[float, float, float], node_id: str | None) -> ObjectNode | None:
        if node_id and node_id in self.objects:
            return self.objects[node_id]
        candidates = [node for node in self.objects.values() if label == node.label or label in node.aliases]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda node: self._distance(node.position, position))
        return nearest if self._distance(nearest.position, position) <= self.association_radius else None

    def recompute_relations(
        self,
        relevant_node_ids: Iterable[str] | None = None,
        *,
        max_edges_per_anchor: int | None = None,
    ) -> None:
        """Recompute directed metric relations.

        When ``relevant_node_ids`` is supplied, only edges originating at a
        task-relevant anchor are retained.  This preserves every relation the
        constraint program can query while avoiding a dense N-squared graph.
        ``max_edges_per_anchor`` further bounds compact graphs: all other
        task anchors are retained first, then the nearest distractors fill the
        remaining slots.  The default ``None`` preserves the full-graph API.
        """
        self.relations.clear()
        nodes = list(self.objects.values())
        relevant = set(relevant_node_ids) if relevant_node_ids is not None else None
        left_nodes = [node for node in nodes if node.node_id in relevant] if relevant is not None else nodes
        for left in left_nodes:
            candidates: list[tuple[float, str, dict[str, Any]]] = []
            for right in nodes:
                if left.node_id == right.node_id:
                    continue
                dx, dy = left.position[0] - right.position[0], left.position[1] - right.position[1]
                distance = self._distance(left.position, right.position)
                relation = {
                    "distance": distance,
                    "left": dx < 0,
                    "right": dx > 0,
                    "front": dy > 0,
                    "behind": dy < 0,
                    "confidence": min(left.confidence, right.confidence),
                }
                candidates.append((distance, right.node_id, relation))
            if max_edges_per_anchor is not None:
                limit = max(1, int(max_edges_per_anchor))
                # Preserve anchor-to-anchor relations even when anchors are
                # far apart; nearest non-anchor nodes are only distractors.
                required = {
                    node_id for node_id in (relevant or ()) if node_id != left.node_id
                }
                candidates.sort(
                    key=lambda item: (
                        item[1] not in required,
                        item[0],
                        item[1],
                    )
                )
                required_count = sum(item[1] in required for item in candidates)
                candidates = candidates[: max(limit, required_count)]
            for _, right_id, relation in candidates:
                self.relations[f"{left.node_id}|{right_id}"] = relation

    def anchors(self) -> dict[str, tuple[float, float, float]]:
        result: dict[str, tuple[float, float, float]] = {}
        for node in sorted(self.objects.values(), key=lambda item: item.confidence, reverse=True):
            result.setdefault(node.label, node.position)
            result[node.node_id] = node.position
        return result

    def retrieve(self, labels: Iterable[str], limit: int = 20) -> list[ObjectNode]:
        normalized = {self.normalize_label(label) for label in labels}
        matches = [node for node in self.objects.values() if node.label in normalized or normalized.intersection(node.aliases)]
        return sorted(matches, key=lambda node: node.confidence, reverse=True)[:limit]

    def prune(self, max_nodes: int, keep_node_ids: Iterable[str] = ()) -> None:
        """Bound graph size while retaining task anchors and recent evidence."""
        max_nodes = max(1, int(max_nodes))
        if len(self.objects) <= max_nodes:
            return
        keep = set(keep_node_ids)

        def latest_frame(node: ObjectNode) -> int:
            frames = [item.frame_id for item in node.evidence if item.frame_id is not None]
            return max(frames, default=-1)

        ordered = sorted(
            self.objects.values(),
            key=lambda node: (
                node.node_id in keep,
                latest_frame(node),
                node.confidence,
                node.observations,
            ),
            reverse=True,
        )
        retained = {node.node_id for node in ordered[:max_nodes]}
        self.objects = {
            node_id: node for node_id, node in self.objects.items() if node_id in retained
        }
        self.relations = {
            key: value
            for key, value in self.relations.items()
            if all(node_id in retained for node_id in key.split("|", 1))
        }

    def save(self, path: Path) -> None:
        payload = {
            "version": 1,
            "association_radius": self.association_radius,
            "max_evidence_per_node": self.max_evidence_per_node,
            "objects": [node.to_dict() for node in self.objects.values()],
            "relations": self.relations,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PhysicalMemoryGraph":
        payload = json.loads(path.read_text(encoding="utf-8"))
        graph = cls(
            float(payload["association_radius"]),
            payload.get("max_evidence_per_node"),
        )
        for item in payload["objects"]:
            item["position"] = tuple(item["position"])
            item["covariance"] = tuple(item["covariance"])
            item["aliases"] = set(item["aliases"])
            item["evidence"] = [Evidence(**entry) for entry in item["evidence"]]
            node = ObjectNode(**item)
            graph.objects[node.node_id] = node
        graph.relations = payload.get("relations", {})
        return graph

    @staticmethod
    def normalize_label(label: str) -> str:
        return " ".join(label.lower().replace("_", " ").replace("-", " ").split())

    @staticmethod
    def _distance(a: Iterable[float], b: Iterable[float]) -> float:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
