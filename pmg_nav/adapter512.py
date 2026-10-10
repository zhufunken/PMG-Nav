"""Deterministic 512-d PMG feature representation.

This is a legacy metadata/logging representation. It is disabled by default in
the current no-RL Stage 2 and is not consumed by the released TIC-VLA action
expert because no projector is trained or wired into that upstream model. A
future Stage 3 adapter may reuse this boundary.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable


class PMGAdapter512:
    dimension = 512

    def encode(self, graph, constraints: Iterable[str] = (), candidates=()) -> list[float]:
        values = [0.0] * self.dimension
        tokens = []
        for node in sorted(graph.objects.values(), key=lambda item: item.node_id):
            tokens.extend([node.label, node.node_id, *map(str, node.position), str(node.confidence), str(node.observations)])
        tokens.extend(map(str, constraints))
        for candidate in list(candidates)[:8]:
            tokens.extend([*map(str, candidate.position), str(candidate.yaw), str(candidate.score)])
        for index, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                bucket = int.from_bytes(digest[offset:offset + 4], "little") % self.dimension
                sign = 1.0 if digest[(offset + 1) % len(digest)] & 1 else -1.0
                values[bucket] += sign * (1.0 / math.sqrt(index + 1))
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    @classmethod
    def zero(cls) -> list[float]:
        return [0.0] * cls.dimension
