"""Operator CLI: validation, local tasks, serving and explicitly requested benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .contracts import TaskRequest
from .engine import execute_sync
from .resources import discover_devices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oflow", description="OFlow single-host workload engine")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "serve", "devices"):
        command = commands.add_parser(name)
        command.add_argument("--config")
        if name == "serve":
            command.add_argument("--host")
            command.add_argument("--port", type=int)
        if name == "run":
            command.add_argument("--task", default="text.score")
            command.add_argument("--input", required=True, help="JSON object")
            command.add_argument("--profile", default="balanced")
        if name == "devices":
            command.add_argument(
                "--cuda", action="store_true", help="Explicitly opt into CUDA runtime probing"
            )
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--items", type=int, default=128)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument("--output", default="reports/local-benchmark.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark":
            from .benchmark import benchmark_suite

            report = asyncio.run(
                benchmark_suite(items=args.items, repeats=args.repeats, warmups=args.warmups)
            )
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Benchmark correctness: {report['correct']}; report: {output}")
            return 0 if report["correct"] else 1
        config = load_config(args.config)
        if args.command == "validate":
            print("Configuration schema is valid (models and plugins were not loaded).")
        elif args.command == "devices":
            from dataclasses import asdict

            from .config import DeviceConfig

            devices = config.devices
            if args.cuda:
                # Visibility is still controlled by CUDA_VISIBLE_DEVICES. Probe inventory first.
                import torch

                devices = tuple(d for d in devices if d.id == "cpu") + tuple(
                    DeviceConfig(id=f"cuda:{i}") for i in range(torch.cuda.device_count())
                )
            print(json.dumps([asdict(d) for d in discover_devices(devices)], indent=2))
        elif args.command == "run":
            result = execute_sync(
                TaskRequest(task=args.task, input=json.loads(args.input), profile=args.profile),
                config,
            )
            print(result.model_dump_json(indent=2))
            return 1 if result.error else 0
        elif args.command == "serve":
            import uvicorn

            from .api import create_app

            if args.host is not None or args.port is not None:
                api = config.api.model_dump()
                api.update(
                    {
                        key: value
                        for key, value in {"host": args.host, "port": args.port}.items()
                        if value is not None
                    }
                )
                config = type(config).model_validate({**config.model_dump(), "api": api})
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            uvicorn.run(
                create_app(config),
                host=config.api.host,
                port=config.api.port,
                limit_concurrency=config.api.max_concurrent_requests + 8,
                timeout_keep_alive=5,
                access_log=False,
                workers=1,
            )
        return 0
    except (ValueError, OSError, ImportError) as exc:
        # ValidationError can include config secrets/paths; operator sees a concise failure only.
        print(
            f"OFlow command failed: {type(exc).__name__}. Check configuration and installed extras.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
