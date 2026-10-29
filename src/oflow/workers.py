"""Persistent bounded local workers. Transport boundary is independent of the scheduler."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .config import BackendConfig
from .contracts import (
    Backend,
    BatchBackend,
    ExecutionContext,
    ResourceLease,
    ResourceRequirements,
    TaskRequest,
    WorkerCrashed,
    WorkerDescriptor,
)
from .resources import ResourceManager


class WorkerTransport(Protocol):
    async def execute(self, requests: list[TaskRequest], contexts: list[ExecutionContext]): ...


class LocalWorker:
    def __init__(self, backend: Backend, config: BackendConfig, resources: ResourceManager):
        if (
            backend.descriptor.requirements.device != config.device
            or backend.descriptor.max_concurrency != config.concurrency
        ):
            raise ValueError("backend placement/concurrency differs from operator configuration")
        self.backend, self.config, self.resources = backend, config, resources
        self.executor = ThreadPoolExecutor(
            max_workers=backend.descriptor.max_concurrency, thread_name_prefix=f"oflow-{config.id}"
        )
        self.active, self.restarts = 0, 0
        self.healthy, self.closed = False, False
        self.model_lease: ResourceLease | None = None
        self.unavailable_reason = "not_started"

    async def _call(self, function, *args):
        def guarded():
            try:
                return function(*args)
            except Exception:
                raise
            except BaseException as exc:
                raise WorkerCrashed("worker terminated") from exc

        # Even lifecycle cancellation must not detach physical work from its reservation.
        future = asyncio.wrap_future(self.executor.submit(guarded))
        interrupted = False
        while True:
            try:
                result = await asyncio.shield(future)
                break
            except asyncio.CancelledError:
                interrupted = True
            except Exception:
                if interrupted:
                    raise asyncio.CancelledError() from None
                raise
        if interrupted:
            raise asyncio.CancelledError()
        return result

    async def start(self) -> None:
        if self.config.device.startswith("cuda:") and self.config.model_memory_mb <= 0:
            raise ValueError("CUDA workers require an explicit positive model_memory_mb budget")
        self.model_lease = self.resources.acquire(
            ResourceRequirements(
                device=self.config.device, cpu_slots=0, memory_mb=self.config.model_memory_mb
            ),
            persistent=True,
        )
        if self.model_lease is None:
            self.unavailable_reason = "device_or_budget_unavailable"
            return
        try:
            await self._call(self.backend.initialize)
            self.healthy = (await self._call(self.backend.health)).healthy
            self.unavailable_reason = "ready" if self.healthy else "backend_unhealthy"
        except Exception:
            self.unavailable_reason = "initialization_failed"
            self.healthy = False
        if not self.healthy:
            try:
                await self._call(self.backend.shutdown)
            finally:
                self.resources.release(self.model_lease)
                self.model_lease = None

    async def execute(self, requests: list[TaskRequest], contexts: list[ExecutionContext]):
        if len(requests) == 1:
            return [await self._call(self.backend.execute, requests[0], contexts[0])]
        if not isinstance(self.backend, BatchBackend):
            raise ValueError("backend does not support batching")
        return await self._call(self.backend.execute_batch, requests, contexts)

    async def restart(self) -> None:
        if self.active or self.closed or self.restarts >= self.config.max_restarts:
            return
        self.restarts += 1
        await self._call(self.backend.shutdown)
        await self._call(self.backend.initialize)
        self.healthy = (await self._call(self.backend.health)).healthy
        self.unavailable_reason = "ready" if self.healthy else "backend_unhealthy"

    async def close(self) -> None:
        if self.closed:
            return
        self.closed, self.healthy = True, False
        try:
            await self._call(self.backend.shutdown)
        finally:
            if self.model_lease:
                self.resources.release(self.model_lease)
                self.model_lease = None
            self.executor.shutdown(wait=True, cancel_futures=True)

    def snapshot(self) -> WorkerDescriptor:
        return WorkerDescriptor(
            id=f"local:{self.config.id}",
            backend=self.config.id,
            device=self.config.device,
            healthy=self.healthy,
            active=self.active,
            capacity=self.backend.descriptor.max_concurrency,
            restarts=self.restarts,
            reason=self.unavailable_reason,
        )


class WorkerSupervisor:
    def __init__(
        self,
        backends: dict[str, Backend],
        configs: tuple[BackendConfig, ...],
        resources: ResourceManager,
    ):
        self.workers = {c.id: LocalWorker(backends[c.id], c, resources) for c in configs}

    async def start(self) -> None:
        # Serial initialization avoids overlapping heavyweight model loads.
        try:
            for worker in self.workers.values():
                await worker.start()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        errors = []
        for worker in self.workers.values():
            try:
                await worker.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("one or more backends failed during shutdown") from errors[0]
