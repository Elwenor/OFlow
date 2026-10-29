"""Bounded telemetry and optional metadata-only SQLite audit storage."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from contextlib import nullcontext
from typing import Protocol

from .contracts import TaskResult

LOG = logging.getLogger("oflow")


class AuditRepository(Protocol):
    def record(self, metadata: dict) -> None: ...
    def close(self) -> None: ...


class SQLiteAuditRepository:
    def __init__(self, path: str, retention: int):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.retention = retention
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS audit "
            "(sequence INTEGER PRIMARY KEY, metadata TEXT NOT NULL)"
        )

    def record(self, metadata: dict) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO audit(metadata) VALUES (?)", (json.dumps(metadata),)
            )
            self.connection.execute(
                "DELETE FROM audit WHERE sequence <= "
                "(SELECT COALESCE(MAX(sequence), 0) - ? FROM audit)",
                (self.retention,),
            )

    def close(self) -> None:
        self.connection.close()


class Telemetry:
    buckets = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 30, 120, float("inf"))

    def __init__(self, structured_logs: bool = True, tracing: bool = False):
        self.structured_logs = structured_logs
        self.tracer = None
        if tracing:
            from opentelemetry import trace

            self.tracer = trace.get_tracer("oflow", "1.0.0b1")
        self.counts: Counter = Counter()
        self.latency = [0] * len(self.buckets)
        self.queue_latency = [0] * len(self.buckets)
        self.total_seconds = 0.0
        self.queue_seconds = 0.0

    def span(self, **attributes):
        return (
            self.tracer.start_as_current_span("oflow.execute", attributes=attributes)
            if self.tracer
            else nullcontext()
        )

    def record(
        self, result: TaskResult, *, task: str, tenant: str, profile: str, worker: str | None
    ) -> dict:
        self.counts[result.status.value] += 1
        self.total_seconds += result.total_seconds
        self.queue_seconds += result.queue_seconds
        for index, bound in enumerate(self.buckets):
            if result.total_seconds <= bound:
                self.latency[index] += 1
            if result.queue_seconds <= bound:
                self.queue_latency[index] += 1
        metadata = {
            "request_id": result.request_id,
            "correlation_id": result.metadata.get("correlation_id"),
            "task": task,
            "tenant": tenant,
            "backend": result.backend,
            "profile": profile,
            "worker": worker,
            "status": result.status.value,
            "queue_seconds": result.queue_seconds,
            "execution_seconds": result.execution_seconds,
            "total_seconds": result.total_seconds,
            "error_category": result.error.category.value if result.error else None,
        }
        if self.structured_logs:
            LOG.info(json.dumps(metadata, separators=(",", ":")))
        return metadata

    def prometheus(self, scheduler, supervisor, resources) -> str:
        lines = ["# TYPE oflow_requests_total counter"]
        for status, count in sorted(self.counts.items()):
            lines.append(f'oflow_requests_total{{status="{status}"}} {count}')
        lines += [
            "# TYPE oflow_active_requests gauge",
            f"oflow_active_requests {scheduler.inflight}",
            "# TYPE oflow_queued_requests gauge",
            f"oflow_queued_requests {len(scheduler.queue)}",
        ]
        for name, counts, total in [
            ("latency", self.latency, self.total_seconds),
            ("queue", self.queue_latency, self.queue_seconds),
        ]:
            lines.append(f"# TYPE oflow_{name}_seconds histogram")
            for bound, count in zip(self.buckets, counts, strict=True):
                label = "+Inf" if bound == float("inf") else str(bound)
                lines.append(f'oflow_{name}_seconds_bucket{{le="{label}"}} {count}')
            lines += [
                f"oflow_{name}_seconds_count {sum(self.counts.values())}",
                f"oflow_{name}_seconds_sum {total}",
            ]
        for worker in supervisor.workers.values():
            identity = worker.config.id
            lines += [
                f'oflow_worker_healthy{{backend="{identity}"}} {int(worker.healthy)}',
                f'oflow_worker_active{{backend="{identity}"}} {worker.active}',
            ]
        for device in resources.snapshot():
            lines += [
                f'oflow_device_reserved_memory_mb{{device="{device["id"]}"}} '
                f"{device['reserved_memory_mb']}",
                f'oflow_device_active{{device="{device["id"]}"}} {device["active"]}',
            ]
        return "\n".join(lines) + "\n"
