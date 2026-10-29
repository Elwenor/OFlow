"""Explicit, portable benchmarks. No benchmark or autotuning runs on normal requests."""

from __future__ import annotations

import asyncio
import hashlib
import math
import platform
import random
import statistics
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .config import EngineConfig
from .contracts import TaskRequest, TaskState, TenantIdentity
from .engine import OFlowEngine


def equivalent(reference, candidate, *, atol: float = 5e-4, rtol: float = 1e-5) -> bool:
    """Full structural comparison, exact integers/booleans, finite float abs+relative tolerance."""
    if isinstance(reference, bool) or isinstance(candidate, bool):
        return type(reference) is type(candidate) and reference == candidate
    if isinstance(reference, int) and isinstance(candidate, int):
        return reference == candidate
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        return (
            math.isfinite(float(reference))
            and math.isfinite(float(candidate))
            and math.isclose(float(reference), float(candidate), abs_tol=atol, rel_tol=rtol)
        )
    if type(reference) is not type(candidate):
        return False
    if isinstance(reference, dict):
        return reference.keys() == candidate.keys() and all(
            equivalent(reference[k], candidate[k], atol=atol, rtol=rtol) for k in reference
        )
    if isinstance(reference, (list, tuple)):
        return len(reference) == len(candidate) and all(
            equivalent(a, b, atol=atol, rtol=rtol)
            for a, b in zip(reference, candidate, strict=True)
        )
    return reference == candidate


def autotune(
    reference: Callable,
    candidates: dict[str, Callable],
    *,
    environment: dict,
    warmups: int = 1,
    repeats: int = 3,
    seed: int = 42,
) -> dict:
    """Benchmark each candidate in a shuffled order per repetition; every output must pass."""
    if warmups < 1 or repeats < 3 or not candidates or not environment:
        raise ValueError("provide candidates, environment, >=1 warmup and >=3 measured repetitions")
    expected = reference()
    rows: dict[str, dict[str, Any]] = {
        name: {"seconds": [], "correct": True} for name in candidates
    }
    rng = random.Random(seed)
    orders = []
    for index in range(warmups + repeats):
        names = list(candidates)
        rng.shuffle(names)
        orders.append(names)
        for name in names:
            started = time.perf_counter()
            try:
                output = candidates[name]()
                valid = equivalent(expected, output)
            except Exception:
                valid = False
            elapsed = time.perf_counter() - started
            rows[name]["correct"] = rows[name]["correct"] and valid
            if index >= warmups:
                rows[name]["seconds"].append(elapsed)
    valid_names = [name for name, result in rows.items() if result["correct"]]
    winner = (
        min(valid_names, key=lambda name: statistics.median(rows[name]["seconds"]))
        if valid_names
        else None
    )
    return {
        "environment": environment,
        "seed": seed,
        "warmups": warmups,
        "repeats": repeats,
        "orders": orders,
        "candidates": rows,
        "winner": winner,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class Scenario:
    name: str
    clients: int
    profile: str = "latency_first"
    fail: bool = False
    deadline: bool = False
    mixed: bool = False


SCENARIOS = (
    Scenario("A_single_interactive", 1),
    Scenario("B_multiple_interactive", 8),
    Scenario("C_bulk", 16, "throughput_first"),
    Scenario("D_mixed", 8, mixed=True),
    Scenario("E_backend_failures", 8, fail=True),
    Scenario("F_deadlines", 4, deadline=True),
    Scenario("G_repeated_memory", 8, "balanced"),
)


async def benchmark_suite(
    *, items: int = 128, warmups: int = 1, repeats: int = 3, seed: int = 42
) -> dict:
    """Scheduler measurements on synthetic deterministic work, not model-inference claims."""
    from datetime import datetime, timedelta

    if items < 8 or repeats < 3 or warmups < 1:
        raise ValueError("use items>=8, repeats>=3, warmups>=1")
    source_digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for source in sorted(package.rglob("*.py")):
        source_digest.update(source.relative_to(package).as_posix().encode())
        source_digest.update(source.read_bytes().replace(b"\r\n", b"\n"))
    report: dict[str, Any] = {
        "kind": "synthetic_scheduler",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "oflow": version("oflow-engine"),
            "pydantic": version("pydantic"),
            "backend": "deterministic_mock",
            "model_version": "none",
            "runtime": "thread",
            "source_tree_sha256": source_digest.hexdigest(),
        },
        "items_per_run": items,
        "warmups": warmups,
        "repeats": repeats,
        "seed": seed,
        "runs": [],
        "order": [],
    }
    rng = random.Random(seed)
    for repeat in range(warmups + repeats):
        scenarios = list(SCENARIOS)
        rng.shuffle(scenarios)
        report["order"].append([s.name for s in scenarios])
        for scenario in scenarios:
            config = EngineConfig.model_validate(
                {
                    "backends": [
                        {
                            "id": "synthetic",
                            "kind": "mock",
                            "concurrency": 2,
                            "options": {
                                "delay_seconds": 0.01 if scenario.deadline else 0.001,
                                "fail": scenario.fail,
                            },
                        }
                    ],
                    "scheduler": {"history_limit": 8},
                    "telemetry": {"structured_logs": False},
                }
            )
            async with OFlowEngine(config) as engine:
                tenants = [
                    TenantIdentity(f"client-{i}", concurrency_limit=4)
                    for i in range(scenario.clients)
                ]
                results: list = []
                iterator = iter(range(items))
                tracemalloc.start()
                before = tracemalloc.get_traced_memory()[0]
                start = time.perf_counter()

                async def client(
                    identity, iterator=iterator, scenario=scenario, results=results, engine=engine
                ):
                    for index in iterator:
                        deadline = (
                            datetime.now(UTC) + timedelta(milliseconds=1)
                            if scenario.deadline
                            else None
                        )
                        profile = (
                            "throughput_first" if scenario.mixed and index % 2 else scenario.profile
                        )
                        req = TaskRequest(
                            task="test.echo",
                            input={"value": index},
                            profile=profile,
                            deadline=deadline,
                        )
                        result = await engine.execute(req, identity=identity)
                        results.append((index, result))

                await asyncio.gather(*(client(t) for t in tenants))
                elapsed = time.perf_counter() - start
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                expected = (
                    TaskState.TIMED_OUT
                    if scenario.deadline
                    else TaskState.FAILED
                    if scenario.fail
                    else TaskState.COMPLETED
                )
                valid = len(results) == items and all(
                    r.status == expected
                    and (expected != TaskState.COMPLETED or r.output == {"value": index})
                    for index, r in results
                )
                if repeat >= warmups:
                    report["runs"].append(
                        {
                            "scenario": asdict(scenario),
                            "repeat": repeat - warmups + 1,
                            "configuration": config.model_dump(mode="json"),
                            "correct": valid,
                            "seconds": elapsed,
                            "requests_per_second": items / elapsed,
                            "completed_per_second": sum(
                                r.status == TaskState.COMPLETED for _, r in results
                            )
                            / elapsed,
                            "latency_p50_ms": percentile(
                                [r.total_seconds * 1000 for _, r in results], 0.5
                            ),
                            "latency_p95_ms": percentile(
                                [r.total_seconds * 1000 for _, r in results], 0.95
                            ),
                            "latency_p99_ms": percentile(
                                [r.total_seconds * 1000 for _, r in results], 0.99
                            ),
                            "queue_p95_ms": percentile(
                                [r.queue_seconds * 1000 for _, r in results], 0.95
                            ),
                            "python_retained_bytes": current - before,
                            "python_peak_bytes": peak,
                            "status_counts": dict(Counter(r.status.value for _, r in results)),
                            "devices": engine.resources.snapshot(),
                        }
                    )
    report["correct"] = all(r["correct"] for r in report["runs"])
    return report
