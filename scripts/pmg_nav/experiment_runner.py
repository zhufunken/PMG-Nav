#!/usr/bin/env python3
"""Reusable experiment runner and model-service benchmark.

Examples:
  python scripts/experiment_runner.py run --name smoke -- bash scripts/06_smoke_ticvla.sh
  python scripts/experiment_runner.py http --name proxy-a --url "$MODEL_BASE_URL/v1/models" --requests 5

Each invocation appends one JSON object to ``EXPERIMENT_RECORDS`` (JSONL).
Secrets are never written to the record; URLs are redacted to scheme/host/path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlsplit, urlunsplit


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_url(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_command(args: argparse.Namespace) -> int:
    started = time.monotonic()
    started_at = now()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    record = {
        "kind": "experiment",
        "name": args.name,
        "started_at": started_at,
        "command": command,
        "cwd": str(Path.cwd()),
        "attempts": 0,
        "status": "failed",
    }
    output = ""
    for attempt in range(1, args.retries + 2):
        record["attempts"] = attempt
        try:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            output = (completed.stdout + completed.stderr)[-args.max_output_chars:]
            record["exit_code"] = completed.returncode
            if completed.returncode == 0:
                record["status"] = "success"
                break
        except OSError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        if attempt <= args.retries:
            time.sleep(args.retry_delay)
    record["finished_at"] = now()
    record["duration_seconds"] = round(time.monotonic() - started, 3)
    record["output_tail"] = output
    if args.price_per_hour is not None:
        record["estimated_cost_cny"] = round(record["duration_seconds"] / 3600 * args.price_per_hour, 4)
    append_record(Path(args.records), record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["status"] == "success" else int(record.get("exit_code", 1))


def http_benchmark(args: argparse.Namespace) -> int:
    latencies: list[float] = []
    errors: list[str] = []
    started = time.monotonic()
    for _ in range(args.requests):
        t0 = time.monotonic()
        try:
            request = urllib.request.Request(args.url, headers={"User-Agent": "ccfb-experiment-runner/1"})
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response.read(args.max_response_bytes)
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}")
            latencies.append(time.monotonic() - t0)
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    total = time.monotonic() - started
    success = len(latencies)
    record = {
        "kind": "model_service_benchmark",
        "name": args.name,
        "url": redact_url(args.url),
        "started_at": now(),
        "requests": args.requests,
        "successes": success,
        "failures": len(errors),
        "reliability": round(success / args.requests, 4),
        "total_seconds": round(total, 3),
        "latency_seconds": {
            "mean": round(mean(latencies), 4) if latencies else None,
            "median": round(median(latencies), 4) if latencies else None,
            "min": round(min(latencies), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
        "errors": errors[:5],
        "status": "success" if success == args.requests else "partial" if success else "failed",
        "pricing": {"input_cny_per_1k_tokens": args.input_price, "output_cny_per_1k_tokens": args.output_price},
    }
    append_record(Path(args.records), record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if success else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run", help="run any experiment command with retries and timing")
    r.add_argument("--name", required=True)
    r.add_argument("--records", default=os.getenv("EXPERIMENT_RECORDS", "records/experiment_runs.jsonl"))
    r.add_argument("--retries", type=int, default=0)
    r.add_argument("--retry-delay", type=float, default=2.0)
    r.add_argument("--price-per-hour", type=float, default=None)
    r.add_argument("--max-output-chars", type=int, default=4000)
    r.add_argument("command", nargs=argparse.REMAINDER)
    r.set_defaults(func=run_command)
    h = sub.add_parser("http", help="measure endpoint latency and reliability")
    h.add_argument("--name", required=True)
    h.add_argument("--url", required=True)
    h.add_argument("--requests", type=int, default=5)
    h.add_argument("--timeout", type=float, default=20)
    h.add_argument("--max-response-bytes", type=int, default=1_000_000)
    h.add_argument("--records", default=os.getenv("EXPERIMENT_RECORDS", "records/experiment_runs.jsonl"))
    h.add_argument("--input-price", type=float, default=None)
    h.add_argument("--output-price", type=float, default=None)
    h.set_defaults(func=http_benchmark)
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    if parsed.mode == "run" and not parsed.command:
        parser().error("run requires a command after --")
    raise SystemExit(parsed.func(parsed))
