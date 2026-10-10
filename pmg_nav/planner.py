"""Small deterministic obstacle-aware route planner for Stage 2.

The planner is deliberately independent of TIC-VLA and of the benchmark
evaluator.  It consumes only static scene geometry (or a caller supplied set
of 2-D obstacle rectangles), the robot's *current* pose, and a PMG-selected
candidate pose.  It never reads ``TICVLA_GOAL_*``, success, or evaluator
trajectory state.

The implementation is intentionally conservative and bounded:

* static USD collision/geometry bounds are reduced to a small set of AABBs;
* a coarse occupancy grid with deterministic 8-connected A* is used only when
  a straight segment is blocked;
* grid size and node expansions are capped so a malformed USD asset cannot
  consume unbounded CPU/RAM;
* a route is advisory: when no safe route can be found, the caller can retain
  the original PMG candidate and report ``route_unavailable``.

This is a Stage 2 oracle geometry component, not a learned adapter.  It is
kept in a separate module so M0 can run without importing it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import re
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ObstacleAABB:
    """A static world-space obstacle footprint.

    Only the XY footprint is used by the planner.  Z limits are retained for
    auditability and allow callers to discard overhead geometry before
    constructing the map.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float = -math.inf
    max_z: float = math.inf
    source: str = "unknown"
    label: str = ""

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(value)) for value in (self.min_x, self.max_x, self.min_y, self.max_y)):
            raise ValueError("obstacle XY bounds must be finite")
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise ValueError("obstacle bounds must be ordered")

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    def contains(
        self,
        point: Sequence[float],
        *,
        margin: float = 0.0,
        margin_x: float | None = None,
        margin_y: float | None = None,
    ) -> bool:
        """Return whether an XY point lies in an inflated footprint.

        ``margin`` is the backwards-compatible isotropic form.  Optional
        ``margin_x``/``margin_y`` values provide an axis-aligned envelope for
        robots whose root is offset from their physical footprint.  Supplying
        only one axis-specific value leaves the other axis at ``margin``.
        """

        x, y = float(point[0]), float(point[1])
        # Preserve the historical ``margin`` behavior (including callers that
        # deliberately pass a negative value to shrink a test footprint).  The
        # planner's own axis additions are clamped at construction time, so
        # normal route checks remain conservative/non-negative.
        effective_x = float(margin if margin_x is None else margin_x)
        effective_y = float(margin if margin_y is None else margin_y)
        return (
            self.min_x - effective_x <= x <= self.max_x + effective_x
            and self.min_y - effective_y <= y <= self.max_y + effective_y
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoutePlan:
    """Serializable result of one bounded planning request."""

    waypoints: tuple[tuple[float, float, float], ...]
    reachable: bool
    length_m: float
    expanded_nodes: int
    direct: bool
    reason: str
    resolution_m: float

    @property
    def first_waypoint(self) -> tuple[float, float, float] | None:
        return self.waypoints[0] if self.waypoints else None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["waypoints"] = [list(item) for item in self.waypoints]
        return value


class ObstacleAwareRoutePlanner:
    """Deterministic 2-D occupancy/A* planner with static-scene extraction."""

    # Labels that are useful collision geometry even when a USD prim does not
    # expose a CollisionAPI.  We intentionally do not include tiny decorative
    # objects: they create noisy maps and do not explain the long stalls seen
    # in the Stage 2 traces.
    _GEOMETRY_TOKENS = {
        "wall", "door", "pillar", "column", "table", "desk", "bed", "gurney",
        "cabinet", "closet", "rack", "shelf", "chair", "counter", "sofa", "bench",
        "elevator", "stairs", "staircase", "cart", "forklift", "vehicle",
        "crate", "box", "block", "concrete", "barrier", "fence", "planter", "trashcan",
    }
    _BLOCKED_PATH_TOKENS = {
        "robot", "carter", "chassis", "wheel", "caster", "sensor", "camera",
        "lidar", "laser", "imu", "bumper", "joint",
        "navmesh", "navigation", "ground", "floor", "ceiling", "roof",
    }
    # These prims describe traversable horizontal surfaces rather than
    # footprint obstacles.  Some Isaac assets (notably SM_CarpetPath_01)
    # author them with CollisionAPI, so the collision schema alone is not a
    # sufficient obstacle criterion.
    _PASSABLE_SURFACE_TOKENS = {
        "carpet", "carpetpath", "rug", "mat", "floor", "ground", "terrain",
        "tile", "tiles", "pavement", "road",
    }

    @classmethod
    def _path_tokens(cls, path: str, name: str) -> set[str]:
        """Split USD CamelCase labels without matching accidental substrings.

        Lowercasing before tokenization hid furniture such as
        ``SM_SecretaryDeskA2`` and ``SM_ReceptionClosets6`` from the static
        obstacle map. A substring test would misread ``Portable`` as a table.
        """

        raw = f"{path} {name}"
        words = re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+",
            raw,
        )
        tokens = {word.lower() for word in words}
        for token in tuple(tokens):
            if token.endswith("s") and token[:-1] in cls._GEOMETRY_TOKENS:
                tokens.add(token[:-1])
        return tokens

    def __init__(
        self,
        obstacles: Iterable[ObstacleAABB] = (),
        *,
        resolution_m: float = 0.50,
        robot_radius_m: float = 0.45,
        corner_clearance_m: float = 0.0,
        axis_clearance_x_m: float = 0.0,
        axis_clearance_y_m: float = 0.0,
        bounds_margin_m: float = 3.0,
        max_cells: int = 24000,
        max_expansions: int | None = None,
        max_obstacles: int = 512,
    ) -> None:
        self.resolution_m = max(0.10, float(resolution_m))
        self.robot_radius_m = max(0.0, float(robot_radius_m))
        # ``robot_radius_m`` is retained as the public/base footprint
        # parameter.  Carter's released asset is not centred on its root and
        # has a roughly 1.24 x 0.91 m footprint, so a circular radius of 0.45
        # m can still cut a wall corner.  This optional, explicit margin is a
        # route-planning safety envelope; it is zero by default so existing
        # M0/M1 runs remain bit-for-bit compatible unless a gate opts in.
        self.corner_clearance_m = max(0.0, float(corner_clearance_m))
        # Additional world-axis margins model a non-centred/asymmetric robot
        # root without changing the isotropic baseline envelope.  These are
        # explicit opt-in diagnostics and remain zero by default.
        self.axis_clearance_x_m = max(0.0, float(axis_clearance_x_m))
        self.axis_clearance_y_m = max(0.0, float(axis_clearance_y_m))
        self.bounds_margin_m = max(0.5, float(bounds_margin_m))
        self.max_cells = max(256, int(max_cells))
        self.max_expansions = max_expansions
        self.max_obstacles = max(1, int(max_obstacles))
        self.obstacles = self._deduplicate_obstacles(obstacles)
        # Populated by ``from_stage`` for auditability; manually supplied
        # obstacle maps have no extraction-time surface filtering.
        self.filtered_ground_surface_count = 0
        self.last_plan: RoutePlan | None = None
        self.last_start: tuple[float, float, float] | None = None
        self.last_goal: tuple[float, float, float] | None = None

    @classmethod
    def from_stage(
        cls,
        stage: Any,
        xform_cache: Any,
        bbox_cache: Any,
        *,
        exclude_paths: Iterable[str] = (),
        resolution_m: float = 0.50,
        robot_radius_m: float = 0.45,
        corner_clearance_m: float = 0.0,
        axis_clearance_x_m: float = 0.0,
        axis_clearance_y_m: float = 0.0,
        bounds_margin_m: float = 3.0,
        max_cells: int = 24000,
        max_obstacles: int = 512,
    ) -> "ObstacleAwareRoutePlanner":
        """Extract conservative static footprints from an already-open USD stage.

        ``pxr`` is imported lazily so the planner remains unit-testable on a
        normal Python installation.  A prim is retained when it has an
        explicit physics collision API, or when its semantic label looks like
        a structural/furniture obstacle.  Robot/sensor/navigation prims are
        always excluded.  ``exclude_paths`` is intended for the selected PMG
        target instance, not evaluator state.
        """

        excluded = {str(path).rstrip("/") for path in exclude_paths}
        try:
            from pxr import UsdGeom, UsdPhysics
        except Exception:
            # Isaac-only dependency unavailable: return an empty but usable
            # planner rather than making M0/M1 library imports fail.
            return cls(
                (),
                resolution_m=resolution_m,
                robot_radius_m=robot_radius_m,
                corner_clearance_m=corner_clearance_m,
                axis_clearance_x_m=axis_clearance_x_m,
                axis_clearance_y_m=axis_clearance_y_m,
                bounds_margin_m=bounds_margin_m,
                max_cells=max_cells,
                max_obstacles=max_obstacles,
            )

        records: list[ObstacleAABB] = []
        filtered_ground_surface_count = 0
        for prim in stage.Traverse():
            try:
                if not prim.IsActive() or not prim.IsValid() or prim.IsPseudoRoot():
                    continue
                path = str(prim.GetPath())
                # Preserve CamelCase boundaries as well as numeric suffixes.
                # Lowercasing first silently dropped compound desks/closets
                # from the map, making a furniture-blocked path look direct.
                tokens = cls._path_tokens(path, str(prim.GetName()))
                if tokens.intersection(cls._BLOCKED_PATH_TOKENS):
                    continue
                # Excluding a selected target parent also excludes its child
                # collision meshes.  Keep unrelated furniture/walls.
                if any(path == item or path.startswith(item + "/") for item in excluded):
                    continue
                imageable = UsdGeom.Imageable(prim)
                if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                    # Collision-only hidden meshes are still relevant; a
                    # hidden mesh with CollisionAPI is retained below.
                    has_collision = cls._has_collision_api(prim, UsdPhysics)
                    if not has_collision:
                        continue
                has_collision = cls._has_collision_api(prim, UsdPhysics)
                semantic = bool(tokens.intersection(cls._GEOMETRY_TOKENS))
                if not has_collision and not semantic:
                    continue
                bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                minimum, maximum = bounds.GetMin(), bounds.GetMax()
                min_x, max_x = float(minimum[0]), float(maximum[0])
                min_y, max_y = float(minimum[1]), float(maximum[1])
                min_z, max_z = float(minimum[2]), float(maximum[2])
                if not all(math.isfinite(v) for v in (min_x, max_x, min_y, max_y, min_z, max_z)):
                    continue
                width, depth = max_x - min_x, max_y - min_y
                # Degenerate points are not obstacles.  Conversely, giant
                # scene roots/terrain bounds are almost certainly containers,
                # not traversable barriers.
                if min(width, depth) < 0.05 or max(width, depth) > 120.0:
                    continue
                # Ignore overhead geometry that cannot contact the robot.
                if max_z < -0.25 or min_z > 2.5:
                    continue
                # A collision-authored carpet/floor is a traversable support
                # surface, not a vertical footprint obstacle.  Restrict this
                # exception to named surface prims that are both near the
                # ground and genuinely thin; thin furniture/low walls remain
                # in the map and continue to be checked conservatively.
                if cls._is_passable_ground_surface(tokens, min_z, max_z):
                    filtered_ground_surface_count += 1
                    continue
                records.append(
                    ObstacleAABB(min_x, max_x, min_y, max_y, min_z, max_z, path, str(prim.GetName()))
                )
            except Exception:
                # A malformed prim must not abort an otherwise valid episode.
                continue
        planner = cls(
            records,
            resolution_m=resolution_m,
            robot_radius_m=robot_radius_m,
            corner_clearance_m=corner_clearance_m,
            axis_clearance_x_m=axis_clearance_x_m,
            axis_clearance_y_m=axis_clearance_y_m,
            bounds_margin_m=bounds_margin_m,
            max_cells=max_cells,
            max_obstacles=max_obstacles,
        )
        planner.filtered_ground_surface_count = filtered_ground_surface_count
        return planner

    @staticmethod
    def _has_collision_api(prim: Any, usd_physics: Any) -> bool:
        for name in ("CollisionAPI", "MeshCollisionAPI", "RigidBodyAPI"):
            try:
                api = getattr(usd_physics, name, None)
                if api is not None and prim.HasAPI(api):
                    return True
            except Exception:
                continue
        # Some Isaac assets author collision as an attribute without applying
        # the typed schema.  This fallback is intentionally narrow.
        try:
            return any(
                prim.HasAttribute(name)
                for name in ("physics:collisionEnabled", "physics:approximation", "physxConvexHull")
            )
        except Exception:
            return False

    @classmethod
    def _is_passable_ground_surface(
        cls,
        tokens: Iterable[str],
        min_z: float,
        max_z: float,
    ) -> bool:
        """Identify thin named support surfaces during USD extraction.

        This is deliberately a narrow geometry predicate, kept separate from
        ``from_stage`` so its assumptions can be unit-tested without Isaac
        Sim.  It does not affect manually supplied ``ObstacleAABB`` records.
        """

        try:
            token_set = set(tokens)
            lower_z = float(min_z)
            upper_z = float(max_z)
        except (TypeError, ValueError):
            return False
        return bool(
            token_set.intersection(cls._PASSABLE_SURFACE_TOKENS)
            and upper_z - lower_z <= 0.05
            and lower_z >= -0.25
            and upper_z <= 0.15
        )

    def _deduplicate_obstacles(self, obstacles: Iterable[ObstacleAABB]) -> tuple[ObstacleAABB, ...]:
        ordered: list[ObstacleAABB] = []
        for obstacle in obstacles:
            if not isinstance(obstacle, ObstacleAABB):
                try:
                    obstacle = ObstacleAABB(**dict(obstacle))  # type: ignore[arg-type]
                except Exception:
                    continue
            if obstacle.width < 0.05 or obstacle.depth < 0.05:
                continue
            # Parent/child USD bounds often coincide. Merge overlapping
            # footprints so a small child encountered first cannot hide the
            # rest of a later parent desk or cabinet from route planning.
            duplicate = False
            for index, prior in enumerate(ordered):
                overlap_x = min(obstacle.max_x, prior.max_x) - max(obstacle.min_x, prior.min_x)
                overlap_y = min(obstacle.max_y, prior.max_y) - max(obstacle.min_y, prior.min_y)
                if overlap_x > 0.90 * min(obstacle.width, prior.width) and overlap_y > 0.90 * min(obstacle.depth, prior.depth):
                    representative = (
                        obstacle
                        if obstacle.width * obstacle.depth > prior.width * prior.depth
                        else prior
                    )
                    ordered[index] = ObstacleAABB(
                        min(prior.min_x, obstacle.min_x),
                        max(prior.max_x, obstacle.max_x),
                        min(prior.min_y, obstacle.min_y),
                        max(prior.max_y, obstacle.max_y),
                        min(prior.min_z, obstacle.min_z),
                        max(prior.max_z, obstacle.max_z),
                        representative.source,
                        representative.label,
                    )
                    duplicate = True
                    break
            if not duplicate:
                ordered.append(obstacle)
            if len(ordered) >= self.max_obstacles:
                break
        return tuple(ordered)

    def _inflated_contains(self, point: Sequence[float], *, extra_margin: float = 0.0) -> bool:
        margin = (
            self.robot_radius_m
            + self.corner_clearance_m
            + max(0.0, float(extra_margin))
        )
        margin_x = margin + self.axis_clearance_x_m
        margin_y = margin + self.axis_clearance_y_m
        return any(
            obstacle.contains(point, margin_x=margin_x, margin_y=margin_y)
            for obstacle in self.obstacles
        )

    def is_free_position(self, position: Sequence[float], *, extra_margin: float = 0.0) -> bool:
        """Cheap endpoint feasibility predicate for ``PhysicalGoalField``."""

        try:
            return not self._inflated_contains(position, extra_margin=extra_margin)
        except (TypeError, ValueError, IndexError):
            return False

    # Alias used by callers that prefer the shorter callback name.
    is_free = is_free_position

    def segment_blocked(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        step_m: float | None = None,
    ) -> bool:
        """Conservatively test a straight XY segment against inflated AABBs."""

        try:
            sx, sy = float(start[0]), float(start[1])
            gx, gy = float(goal[0]), float(goal[1])
        except (TypeError, ValueError, IndexError):
            return True
        distance = math.hypot(gx - sx, gy - sy)
        if distance <= 1e-9:
            return not self.is_free_position(start)
        step = max(0.10, float(step_m or self.resolution_m * 0.5))
        samples = max(2, int(math.ceil(distance / step)) + 1)
        for index in range(samples):
            fraction = index / (samples - 1)
            point = (sx + fraction * (gx - sx), sy + fraction * (gy - sy))
            if not self.is_free_position(point):
                return True
        return False

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        force_replan: bool = False,
    ) -> RoutePlan:
        """Plan from the current robot pose to a PMG candidate pose."""

        start_point = self._point3(start)
        goal_point = self._point3(goal)
        if start_point is None or goal_point is None:
            result = RoutePlan((), False, math.inf, 0, False, "invalid_pose", self.resolution_m)
            self.last_plan = result
            return result
        if not force_replan and self.last_plan is not None and self.last_start == start_point and self.last_goal == goal_point:
            return self.last_plan
        self.last_start, self.last_goal = start_point, goal_point
        if not self.obstacles:
            result = RoutePlan((goal_point,), True, self._distance2(start_point, goal_point), 0, True, "no_static_obstacles", self.resolution_m)
            self.last_plan = result
            return result
        if not self.is_free_position(goal_point):
            result = RoutePlan((), False, math.inf, 0, False, "goal_in_obstacle", self.resolution_m)
            self.last_plan = result
            return result
        if self.is_free_position(start_point) and not self.segment_blocked(start_point, goal_point):
            result = RoutePlan((goal_point,), True, self._distance2(start_point, goal_point), 0, True, "direct_visible", self.resolution_m)
            self.last_plan = result
            return result

        result = self._astar(start_point, goal_point)
        self.last_plan = result
        return result

    def _astar(self, start: tuple[float, float, float], goal: tuple[float, float, float]) -> RoutePlan:
        min_x = min(start[0], goal[0]) - self.bounds_margin_m
        max_x = max(start[0], goal[0]) + self.bounds_margin_m
        min_y = min(start[1], goal[1]) - self.bounds_margin_m
        max_y = max(start[1], goal[1]) + self.bounds_margin_m
        # Include nearby obstacle corners so a route can go around a wall just
        # outside the start/goal bounding box.
        for obstacle in self.obstacles:
            if obstacle.max_x >= min_x and obstacle.min_x <= max_x and obstacle.max_y >= min_y and obstacle.min_y <= max_y:
                min_x = min(min_x, obstacle.min_x - self.bounds_margin_m)
                max_x = max(max_x, obstacle.max_x + self.bounds_margin_m)
                min_y = min(min_y, obstacle.min_y - self.bounds_margin_m)
                max_y = max(max_y, obstacle.max_y + self.bounds_margin_m)
        resolution = self.resolution_m
        width = max(1, int(math.ceil((max_x - min_x) / resolution)) + 1)
        height = max(1, int(math.ceil((max_y - min_y) / resolution)) + 1)
        while width * height > self.max_cells:
            resolution *= 1.25
            width = max(1, int(math.ceil((max_x - min_x) / resolution)) + 1)
            height = max(1, int(math.ceil((max_y - min_y) / resolution)) + 1)
        def to_cell(point: tuple[float, float, float]) -> tuple[int, int]:
            return (int(round((point[0] - min_x) / resolution)), int(round((point[1] - min_y) / resolution)))
        def to_point(cell: tuple[int, int], z: float) -> tuple[float, float, float]:
            return (min_x + cell[0] * resolution, min_y + cell[1] * resolution, z)
        def in_bounds(cell: tuple[int, int]) -> bool:
            return 0 <= cell[0] < width and 0 <= cell[1] < height
        def free(cell: tuple[int, int]) -> bool:
            return in_bounds(cell) and self.is_free_position(to_point(cell, start[2]))
        # A free grid center is not enough when an obstacle cuts through the
        # start cell: the controller would then receive a first waypoint whose
        # connector from the measured pose is blocked.  For a genuinely free
        # start, require that exact connector to be collision-free.  A start
        # already inside the inflated footprint intentionally keeps the old
        # nearest-free escape behavior; runtime handles that transition with
        # its strict escape guard.
        start_is_free = self.is_free_position(start)
        if start_is_free:
            def start_connected(cell: tuple[int, int]) -> bool:
                return free(cell) and not self.segment_blocked(
                    start, to_point(cell, start[2])
                )
            start_cell = self._nearest_free(to_cell(start), start_connected, width, height)
        else:
            start_cell = self._nearest_free(to_cell(start), free, width, height)
        goal_cell = self._nearest_free(to_cell(goal), free, width, height)
        if start_cell is None or goal_cell is None:
            reason = "start_connector_blocked" if start_is_free and start_cell is None else "start_or_goal_blocked"
            return RoutePlan((), False, math.inf, 0, False, reason, resolution)
        if start_cell == goal_cell:
            return RoutePlan((goal,), True, self._distance2(start, goal), 0, False, "same_free_cell", resolution)
        neighbors = (
            (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, math.sqrt(2.0)),
            (0, -1, 1.0), (0, 1, 1.0),
            (1, -1, math.sqrt(2.0)), (1, 0, 1.0), (1, 1, math.sqrt(2.0)),
        )
        open_heap: list[tuple[float, float, tuple[int, int]]] = []
        g_cost: dict[tuple[int, int], float] = {start_cell: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        def heuristic(cell: tuple[int, int]) -> float:
            return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])
        heapq.heappush(open_heap, (heuristic(start_cell), 0.0, start_cell))
        expansions = 0
        limit = self.max_expansions or self.max_cells * 2
        found = False
        while open_heap and expansions < limit:
            _, current_cost, current = heapq.heappop(open_heap)
            if current_cost > g_cost.get(current, math.inf) + 1e-9:
                continue
            expansions += 1
            if current == goal_cell:
                found = True
                break
            for dx, dy, step_cost in neighbors:
                candidate = (current[0] + dx, current[1] + dy)
                if not free(candidate):
                    continue
                # Prevent diagonal corner cutting through a pair of touching
                # obstacles, a frequent source of false routes in coarse maps.
                if dx and dy and (not free((current[0] + dx, current[1]) ) or not free((current[0], current[1] + dy))):
                    continue
                new_cost = current_cost + step_cost
                if new_cost + 1e-9 < g_cost.get(candidate, math.inf):
                    g_cost[candidate] = new_cost
                    parent[candidate] = current
                    heapq.heappush(open_heap, (new_cost + heuristic(candidate), new_cost, candidate))
        if not found:
            reason = "expansion_limit" if expansions >= limit else "no_path"
            return RoutePlan((), False, math.inf, expansions, False, reason, resolution)
        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(parent[cells[-1]])
        cells.reverse()
        points = [to_point(cell, goal[2]) for cell in cells]
        points = self._simplify(points)
        # Preserve the exact measured start as the first route point whenever
        # it is free.  This makes the route contract explicit and prevents a
        # coarse-grid cell-center residual from becoming an artificial first
        # target.  The connector was checked above; retain the fail-closed
        # guard in case a future simplifier changes the first point.
        if start_is_free and points:
            if self._distance2(start, points[0]) > 1e-9:
                if self.segment_blocked(start, points[0]):
                    return RoutePlan((), False, math.inf, expansions, False, "start_connector_blocked", resolution)
                points.insert(0, start)
            else:
                points[0] = start
        if points and self._distance2(points[-1], goal) > resolution * 0.75:
            points.append(goal)
        elif points:
            points[-1] = goal
        route_length = sum(self._distance2(a, b) for a, b in zip([start, *points[:-1]], points))
        return RoutePlan(tuple(points), True, route_length, expansions, False, "astar", resolution)

    @staticmethod
    def _nearest_free(
        origin: tuple[int, int],
        free: Any,
        width: int,
        height: int,
        radius: int = 8,
    ) -> tuple[int, int] | None:
        if free(origin):
            return origin
        for distance in range(1, radius + 1):
            for dx in range(-distance, distance + 1):
                for dy in (-distance, distance):
                    cell = (origin[0] + dx, origin[1] + dy)
                    if 0 <= cell[0] < width and 0 <= cell[1] < height and free(cell):
                        return cell
            for dy in range(-distance + 1, distance):
                for dx in (-distance, distance):
                    cell = (origin[0] + dx, origin[1] + dy)
                    if 0 <= cell[0] < width and 0 <= cell[1] < height and free(cell):
                        return cell
        return None

    def _simplify(self, points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        if len(points) <= 2:
            return points
        simplified = [points[0]]
        anchor = 0
        for index in range(2, len(points)):
            if self.segment_blocked(points[anchor], points[index]):
                simplified.append(points[index - 1])
                anchor = index - 1
        simplified.append(points[-1])
        return simplified

    @staticmethod
    def _point3(value: Sequence[float]) -> tuple[float, float, float] | None:
        try:
            point = tuple(float(value[index]) for index in range(3))
        except (TypeError, ValueError, IndexError):
            return None
        return point if all(math.isfinite(item) for item in point) else None

    @staticmethod
    def _distance2(a: Sequence[float], b: Sequence[float]) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "obstacle_count": len(self.obstacles),
            # Keep extraction filtering visible in every PMG artifact.  This
            # distinguishes a planner that ignored support surfaces from one
            # that deliberately removed thin carpet/floor collision meshes.
            "filtered_ground_surface_count": int(self.filtered_ground_surface_count),
            "resolution_m": self.resolution_m,
            "robot_radius_m": self.robot_radius_m,
            "corner_clearance_m": self.corner_clearance_m,
            "axis_clearance_x_m": self.axis_clearance_x_m,
            "axis_clearance_y_m": self.axis_clearance_y_m,
            "bounds_margin_m": self.bounds_margin_m,
            "max_cells": self.max_cells,
            "last_plan": self.last_plan.to_dict() if self.last_plan else None,
            "obstacles": [item.to_dict() for item in self.obstacles],
        }


__all__ = ["ObstacleAABB", "ObstacleAwareRoutePlanner", "RoutePlan"]
