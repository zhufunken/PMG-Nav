#!/usr/bin/env python3
"""Inspect a USD scene for stable object identity, pose, semantics, and bounds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument("--experience", default="")
    return parser.parse_args()


def clean_label(name: str) -> str:
    value = re.sub(r"[_-]+", " ", name)
    value = re.sub(r"\b\d+\b", "", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def main() -> None:
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "renderer": "RayTracedLighting", "fast_shutdown": True},
        experience=args.experience,
    )
    try:
        import omni.kit.app
        import omni.usd
        from pxr import Usd, UsdGeom

        context = omni.usd.get_context()
        result = context.open_stage(args.scene)
        if result is False:
            raise RuntimeError(f"failed to open {args.scene}")
        deadline = time.monotonic() + 300
        while context.get_stage_loading_status()[2] and time.monotonic() < deadline:
            app.update()
        if context.get_stage_loading_status()[2]:
            raise TimeoutError("scene loading exceeded 300 seconds")
        for _ in range(args.warmup_frames):
            app.update()
        stage = context.get_stage()
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        objects = []
        for prim in stage.Traverse():
            if not prim.IsActive() or not prim.IsValid() or prim.IsPseudoRoot():
                continue
            imageable = UsdGeom.Imageable(prim)
            if not imageable or imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            name = prim.GetName()
            label = clean_label(name)
            if not label or label in {"mesh", "xform", "geometry", "looks", "world", "default prim"}:
                continue
            matrix = cache.GetLocalToWorldTransform(prim)
            translation = matrix.ExtractTranslation()
            semantic = {}
            for attribute in prim.GetAttributes():
                attr_name = attribute.GetName()
                if "semantic" in attr_name.lower() or "class" in attr_name.lower():
                    try:
                        value = attribute.Get()
                    except Exception:
                        continue
                    if value is not None:
                        semantic[attr_name] = str(value)
            try:
                bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                minimum, maximum = bbox.GetMin(), bbox.GetMax()
                extent = [float(maximum[i] - minimum[i]) for i in range(3)]
                finite_extent = all(0.01 < value < 100.0 for value in extent)
            except Exception:
                extent, finite_extent = [0.0, 0.0, 0.0], False
            if not semantic and not finite_extent:
                continue
            objects.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "name": name,
                    "label": label,
                    "type": prim.GetTypeName(),
                    "position": [float(translation[i]) for i in range(3)],
                    "extent": extent,
                    "semantic": semantic,
                }
            )
        unique_paths = len({item["prim_path"] for item in objects})
        semantic_count = sum(bool(item["semantic"]) for item in objects)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scene": args.scene,
            "object_count": len(objects),
            "unique_prim_paths": unique_paths,
            "semantic_object_count": semantic_count,
            "has_stable_identity": unique_paths == len(objects) and unique_paths > 0,
            "has_world_pose": all(len(item["position"]) == 3 for item in objects),
            "objects": objects,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({key: payload[key] for key in payload if key != "objects"}, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
