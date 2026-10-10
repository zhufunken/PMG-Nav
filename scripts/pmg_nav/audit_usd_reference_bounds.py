"""Read-only audit of authored USD references and approximate world bounds.

The DynaNav office layer references Omniverse assets by HTTPS URI.  A plain
``Usd.Stage.Open`` in a bare Kit Python process cannot resolve those URIs, so
its BBoxCache can misleadingly report no geometry.  This helper reads only
the authored main layer, downloads selected referenced assets with
``omni.client`` into ``/tmp``, and applies each instance's composed transform
to the referenced asset's local bounds.  It is intended for geometry triage,
not for modifying a USD stage or running simulation.

Environment:
  PMG_INSPECT_SCENE: path to the root USD layer
  PMG_BOUNDS_X/Y: optional target intervals (defaults to x[-1,1.5], y[46.5,49.5])
  PMG_BOUNDS_PAD: optional expansion around the target interval
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import omni.client
from pxr import Gf, Sdf, Usd, UsdGeom


def _interval(name: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = os.environ.get(name)
    if not raw:
        return default
    vals = [float(x.strip()) for x in raw.split(",")]
    if len(vals) != 2 or vals[0] > vals[1]:
        raise ValueError(f"{name} must be 'min,max', got {raw!r}")
    return vals[0], vals[1]


def _materialize(url: str, directory: Path, seen: set[str]) -> Path | None:
    """Materialize one URI and its relative references into a local folder."""
    if url in seen:
        name = Path(urlparse(url).path).name
        return directory / name if name else None
    seen.add(url)
    result, _version, content = omni.client.read_file(url)
    if result != omni.client.Result.OK:
        print(f"ASSET_READ_FAIL url={url} result={result}", flush=True)
        return None
    name = Path(urlparse(url).path).name or "root.usd"
    # Preserve the basename because relative references in the Office assets
    # point to siblings such as ``SM_Wall_6m.usd``.
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != len(content):
        path.write_bytes(content)
    layer = Sdf.Layer.FindOrOpen(str(path))
    if not layer:
        return path
    for ref in layer.GetExternalReferences():
        # Absolute HTTPS references remain absolute; relative references are
        # resolved against the URI of the layer that authored them.
        child_url = ref if "://" in ref else urljoin(url, ref)
        _materialize(child_url, directory, seen)
    return path


def _asset_local_bound(url: str, cache: dict[str, tuple[Gf.Vec3d, Gf.Vec3d] | None]):
    if url in cache:
        return cache[url]
    # Keep temporary files outside the repository and make the directory
    # stable so relative nested references can be resolved by USD.
    digest = hashlib.sha256(url.encode()).hexdigest()[:20]
    directory = Path(tempfile.gettempdir()) / f"pmg_usd_refs_{digest}"
    path = _materialize(url, directory, set())
    if path is None:
        cache[url] = None
        return None
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if not stage:
        print(f"ASSET_STAGE_FAIL url={url}", flush=True)
        cache[url] = None
        return None
    cache_obj = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root = stage.GetDefaultPrim()
    if not root:
        roots = list(stage.GetPseudoRoot().GetChildren())
        root = roots[0] if roots else None
    if not root:
        cache[url] = None
        return None
    try:
        aligned = cache_obj.ComputeWorldBound(root).ComputeAlignedRange()
        mn, mx = aligned.GetMin(), aligned.GetMax()
        vals = [float(v) for v in (*mn, *mx)]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("non-finite extent")
        # USD represents an empty range with +/- float max.  Treat it as
        # unresolved/empty rather than transforming it into a false hit that
        # spans the entire world.
        if any(float(mx[i]) < float(mn[i]) for i in range(3)):
            raise ValueError("empty extent")
        if any(abs(v) > 1.0e6 for v in vals):
            raise ValueError("sentinel/implausible extent")
        bound = (Gf.Vec3d(*[float(v) for v in mn]), Gf.Vec3d(*[float(v) for v in mx]))
    except Exception as exc:  # pragma: no cover - asset-specific fallback
        print(f"ASSET_BOUND_FAIL url={url} error={exc}", flush=True)
        bound = None
    cache[url] = bound
    if bound:
        print(
            "ASSET_BOUND",
            Path(url).name,
            "local=[%.3f,%.3f,%.3f]-[%.3f,%.3f,%.3f]" % (*bound[0], *bound[1]),
            flush=True,
        )
    return bound


def _world_bound(matrix: Gf.Matrix4d, bound: tuple[Gf.Vec3d, Gf.Vec3d]):
    mn, mx = bound
    corners = [
        Gf.Vec3d(x, y, z)
        for x in (mn[0], mx[0])
        for y in (mn[1], mx[1])
        for z in (mn[2], mx[2])
    ]
    points = [matrix.Transform(c) for c in corners]
    return (
        Gf.Vec3d(min(p[0] for p in points), min(p[1] for p in points), min(p[2] for p in points)),
        Gf.Vec3d(max(p[0] for p in points), max(p[1] for p in points), max(p[2] for p in points)),
    )


def _overlap(a: tuple[Gf.Vec3d, Gf.Vec3d], x: tuple[float, float], y: tuple[float, float]) -> bool:
    mn, mx = a
    return not (mx[0] < x[0] or mn[0] > x[1] or mx[1] < y[0] or mn[1] > y[1])


def main() -> int:
    scene = os.environ["PMG_INSPECT_SCENE"]
    target_x = _interval("PMG_BOUNDS_X", (-1.0, 1.5))
    target_y = _interval("PMG_BOUNDS_Y", (46.5, 49.5))
    pad = float(os.environ.get("PMG_BOUNDS_PAD", "0"))
    target_x = target_x[0] - pad, target_x[1] + pad
    target_y = target_y[0] - pad, target_y[1] + pad

    stage = Usd.Stage.Open(scene, load=Usd.Stage.LoadNone)
    if not stage:
        raise RuntimeError(f"unable to open {scene}")
    scene_path = Path(scene)
    scene_layer = Sdf.Layer.FindOrOpen(scene)
    print(
        "SCENE path=%s size=%d sha256=%s external_refs=%d"
        % (
            scene,
            scene_path.stat().st_size,
            hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            len(scene_layer.GetExternalReferences()) if scene_layer else -1,
        ),
        flush=True,
    )
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    asset_cache: dict[str, tuple[Gf.Vec3d, Gf.Vec3d] | None] = {}
    considered = 0
    hits = 0
    # Only geometry-like references are relevant to this corner audit.  This
    # avoids reading hundreds of lights, stationery, and animation assets.
    asset_re = re.compile(r"(wall|building|door|elevator|stair|floor)", re.I)
    for prim in stage.Traverse():
        if prim.IsPseudoRoot() or not prim.IsActive():
            continue
        specs = prim.GetPrimStack()
        refs = []
        for spec in specs:
            try:
                refs.extend(spec.referenceList.GetAppliedItems())
            except Exception:
                continue
        for ref in refs:
            url = str(getattr(ref, "assetPath", ""))
            if not url or not asset_re.search(url):
                continue
            considered += 1
            bound = _asset_local_bound(url, asset_cache)
            if not bound:
                continue
            world = _world_bound(xforms.GetLocalToWorldTransform(prim), bound)
            if _overlap(world, target_x, target_y):
                hits += 1
                mn, mx = world
                print(
                    "HIT",
                    str(prim.GetPath()),
                    Path(url).name,
                    "world=[%.3f,%.3f,%.3f]-[%.3f,%.3f,%.3f]" % (*mn, *mx),
                    flush=True,
                )
    print(
        "SUMMARY scene=%s target_x=%s target_y=%s considered=%d unique_assets=%d hits=%d"
        % (scene, target_x, target_y, considered, len(asset_cache), hits),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
