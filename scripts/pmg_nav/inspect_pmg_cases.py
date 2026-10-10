#!/usr/bin/env python3
"""Export inspectable PMG case summaries and dependency-free SVG diagrams.

The script reads only serialized artifacts. It does not start Isaac Sim and it
does not alter an experiment. The graph path is resolved from each episode's
PMG payload, so the report can distinguish the current compact run from older
graph snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="experiment root")
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--episodes", nargs="+", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-visible-nodes", type=int, default=12)
    p.add_argument("--max-visible-edges", type=int, default=36)
    return p.parse_args()


def find_metrics(root: Path, prefix: str, episode: int) -> Path:
    candidates = [
        root / "outputs" / "dynanav_closed_loop" / f"{prefix}_m1_e{episode:03d}" / "benchmark_metrics.json",
        root / "outputs" / "dynanav_closed_loop" / f"{prefix}_m1_e{episode:03d}" / "benchmark_metrics.json",
        root / "runtime" / "dynanav_closed_loop" / f"{prefix}_m1_e{episode:03d}" / "benchmark_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.glob(f"**/{prefix}_m1_e{episode:03d}/benchmark_metrics.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"metrics not found for episode {episode}: {prefix}")


def resolve_graph(metrics_path: Path, pmg: dict[str, Any]) -> Path:
    raw = pmg.get("graph_path") or pmg.get("graph_file")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = metrics_path.parent / path
        if path.exists():
            return path
    candidates = [metrics_path.with_suffix(".graph.json"), metrics_path.parent / "pmg.graph.json"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"serialized graph not found beside {metrics_path}")


def short_label(value: str, limit: int = 30) -> str:
    value = " ".join(str(value).replace("_", " ").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def select_nodes(graph: dict[str, Any], pmg: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    objects = list(graph.get("objects", []))
    context = str(pmg.get("context", ""))
    match = re.search(r"Physical memory: .*? at \(([-+0-9.eE]+),\s*([-+0-9.eE]+)", context)
    target = (float(match.group(1)), float(match.group(2))) if match else None

    def distance(item: dict[str, Any]) -> float:
        if target is None:
            return float("inf")
        pos = item.get("position", [0.0, 0.0, 0.0])
        return math.hypot(float(pos[0]) - target[0], float(pos[1]) - target[1])

    robot = next((x for x in objects if "robot" in str(x.get("label", "")).lower() or "carter" in str(x.get("label", "")).lower()), None)
    target_node = min(objects, key=distance) if objects and target is not None else None
    ranked = sorted(objects, key=lambda x: (float(x.get("confidence", 0.0)), int(x.get("observations", 0))), reverse=True)
    selected: list[dict[str, Any]] = []
    for item in [robot, target_node, *ranked]:
        if item is None or item in selected:
            continue
        selected.append(item)
        if len(selected) >= max(2, limit):
            break
    return selected


def build_case(metrics_path: Path, episode: int, args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_payload = (payload.get("episodes") or [{}])[0]
    pmg = episode_payload.get("pmg") or {}
    graph_path = resolve_graph(metrics_path, pmg)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    selected = select_nodes(graph, pmg, args.max_visible_nodes)
    selected_ids = {item.get("node_id") for item in selected}
    relations = graph.get("relations", {})
    edges = []
    for key, value in relations.items() if isinstance(relations, dict) else []:
        left, sep, right = str(key).partition("|")
        if sep and left in selected_ids and right in selected_ids:
            edges.append({"source": left, "target": right, **value})
    edges.sort(key=lambda x: (float(x.get("distance", 0.0)), x["source"], x["target"]))
    trace = episode_payload.get("action_trace") or {}
    return {
        "episode": episode,
        "metrics_path": str(metrics_path),
        "graph_path": str(graph_path),
        "scene": episode_payload.get("scene"),
        "success": episode_payload.get("success"),
        "instruction": pmg.get("instruction") or episode_payload.get("instruction"),
        "context": pmg.get("context"),
        "graph_nodes_summary": pmg.get("graph_nodes"),
        "graph_edges_summary": pmg.get("graph_edges"),
        "graph_nodes_serialized": len(graph.get("objects", [])),
        "graph_edges_serialized": len(relations) if isinstance(relations, dict) else len(relations or []),
        "graph_precision": pmg.get("graph_precision"),
        "graph_recall": pmg.get("graph_recall"),
        "constraint_satisfaction": pmg.get("constraint_satisfaction"),
        "relation_brier_score": pmg.get("relation_brier_score"),
        "navigation_error": episode_payload.get("navigation_error"),
        "spl": episode_payload.get("spl"),
        "duration": episode_payload.get("duration"),
        "action_trace": {
            key: trace.get(key)
            for key in (
                "pmg_override_fraction",
                "pmg_override_count",
                "policy_action_count",
                "action_output_count",
                "control_source_counts",
                "pmg_terminal_override_count",
                "pmg_detour_override_count",
                "inference_latency_mean_s",
            )
            if key in trace
        },
        "nodes": [
            {
                "id": item.get("node_id"),
                "label": item.get("label"),
                "position": item.get("position"),
                "confidence": item.get("confidence"),
                "observations": item.get("observations"),
            }
            for item in selected
        ],
        "edges": edges[: args.max_visible_edges],
    }


def svg_for_case(case: dict[str, Any]) -> str:
    nodes = case["nodes"]
    edges = case["edges"]
    if not nodes:
        return "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 900 520\"><text x=\"20\" y=\"40\">No serialized PMG nodes</text></svg>"
    xs = [float((x.get("position") or [0, 0])[0]) for x in nodes]
    ys = [float((x.get("position") or [0, 0])[1]) for x in nodes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    sx = 760.0 / max(xmax - xmin, 1.0)
    sy = 400.0 / max(ymax - ymin, 1.0)
    points = {}
    for i, node in enumerate(nodes):
        pos = node.get("position") or [0, 0]
        x = 70 + (float(pos[0]) - xmin) * sx
        y = 460 - (float(pos[1]) - ymin) * sy
        points[node["id"]] = (x, y)
    lines = []
    for edge in edges:
        if edge["source"] not in points or edge["target"] not in points:
            continue
        x1, y1 = points[edge["source"]]
        x2, y2 = points[edge["target"]]
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#9aa4b2" stroke-opacity="0.28" stroke-width="1"/>')
    circles = []
    for node in nodes:
        x, y = points[node["id"]]
        label = short_label(node.get("label", "node"), 26).replace("&", "&amp;").replace("<", "&lt;")
        low = str(node.get("label", "")).lower()
        fill = "#e06c75" if "robot" in low or "carter" in low else "#4f9da6" if node is nodes[1] else "#8892a6"
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{fill}"/><text x="{x + 12:.1f}" y="{y + 4:.1f}" font-size="12" fill="#253047">{label}</text>')
    status = "SUCCESS" if case.get("success") else "FAILURE"
    title = f'e{int(case["episode"]):03d} · {status} · {case.get("scene", "")}'
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 520" role="img" aria-label="'+title+'"><rect width="900" height="520" fill="#f8fafc"/><text x="20" y="28" font-size="16" font-weight="600" fill="#182235">'+title+'</text><text x="20" y="50" font-size="12" fill="#5b667a">serialized nodes='+str(case.get("graph_nodes_serialized"))+' edges='+str(case.get("graph_edges_serialized"))+'; visible subset shown</text>'+''.join(lines)+''.join(circles)+'</svg>'


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    errors = []
    for episode in args.episodes:
        try:
            cases.append(build_case(find_metrics(args.root, args.run_prefix, episode), episode, args))
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            errors.append({"episode": episode, "error": str(exc)})
    (args.output / "cases.json").write_text(json.dumps({"run_prefix": args.run_prefix, "cases": cases, "errors": errors}, indent=2, ensure_ascii=False), encoding="utf-8")
    for case in cases:
        (args.output / f"e{case['episode']:03d}.svg").write_text(svg_for_case(case), encoding="utf-8")
    lines = [f"# PMG case export: `{args.run_prefix}`", "", "This report is generated from serialized artifacts; it does not run simulation.", ""]
    for case in cases:
        lines += [f"## e{case['episode']:03d} ({'success' if case['success'] else 'failure'})", "", f"- Scene: `{case.get('scene')}`", f"- Metrics: `{case.get('metrics_path')}`", f"- Graph: `{case.get('graph_path')}`", f"- Serialized graph: {case.get('graph_nodes_serialized')} nodes, {case.get('graph_edges_serialized')} edges", f"- Visible graph subset: {len(case['nodes'])} nodes, {len(case['edges'])} edges", f"- NE: {case.get('navigation_error')}; SPL: {case.get('spl')}; duration: {case.get('duration')}", "", f"Instruction: {case.get('instruction')}", ""]
    if errors:
        lines += ["## Errors", ""] + [f"- e{x['episode']:03d}: {x['error']}" for x in errors]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
