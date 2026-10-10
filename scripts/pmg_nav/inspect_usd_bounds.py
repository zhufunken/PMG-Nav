"""Read-only diagnostic: print USD prim bounds near an episode corner."""
from __future__ import annotations

import math
import os

from pxr import Usd, UsdGeom

scene = os.environ["PMG_INSPECT_SCENE"]
stage = Usd.Stage.Open(scene)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for prim in stage.Traverse():
    try:
        if not prim.IsActive() or prim.IsPseudoRoot():
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
        min_x, max_x = float(minimum[0]), float(maximum[0])
        min_y, max_y = float(minimum[1]), float(maximum[1])
        min_z, max_z = float(minimum[2]), float(maximum[2])
        if not all(math.isfinite(v) for v in (min_x, max_x, min_y, max_y, min_z, max_z)):
            continue
        if max_x < -1.0 or min_x > 1.5 or max_y < 46.5 or min_y > 49.5:
            continue
        if (max_x - min_x) <= 0.05 or (max_y - min_y) <= 0.05:
            continue
        print(str(prim.GetPath()), prim.GetName(), f"x=[{min_x:.3f},{max_x:.3f}]", f"y=[{min_y:.3f},{max_y:.3f}]", f"z=[{min_z:.3f},{max_z:.3f}]")
    except Exception:
        continue
