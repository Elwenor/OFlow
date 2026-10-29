"""Application services used directly by the SDK and by the ASGI adapter."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backends import BUILTIN_FACTORIES, register_builtin_tasks
from .config import EngineConfig, load_config
from .contracts import Backend, ExecutionPolicy, TaskRequest, TaskResult, TenantIdentity
from .registry import BackendRegistry, RulesRouter, TaskRegistry, load_plugins
from .resources import Device, ResourceManager, discover_devices
from .scheduler import Scheduler, TaskHandle
from .telemetry import SQLiteAuditRepository, Telemetry
from .workers import WorkerSupervisor


class OFlowEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        tasks: TaskRegistry | None = None,
        backends: list[Backend] | None = None,
        inventory: list[Device] | None = None,
        policy: ExecutionPolicy | None = None,
        identity: TenantIdentity | None = None,
    ):
        self.config = config or EngineConfig()
        self.tasks = tasks or TaskRegistry()
        register_builtin_tasks(self.tasks)
        self.registry = BackendRegistry(self.tasks)
        plugins = load_plugins(self.config, self.tasks)
        if set(plugins) & set(BUILTIN_FACTORIES):
            raise ValueError("plugins may not replace built-in factory names")
        factories = {**BUILTIN_FACTORIES, **plugins}
        if backends is None:
            for cfg in self.config.backends:
                if cfg.kind not in factories:
                    raise ValueError(f"unknown backend kind: {cfg.kind}")
                self.registry.register(factories[cfg.kind](cfg, self.tasks))
        else:
            for backend in backends:
                self.registry.register(backend)
        if set(self.registry.backends) != {b.id for b in self.config.backends}:
            raise ValueError("backend instances must match configured backend IDs")
        self.router = RulesRouter(self.registry, self.config.routing.rules)
        self.resources = ResourceManager(
            inventory if inventory is not None else discover_devices(self.config.devices)
        )
        self.supervisor = WorkerSupervisor(
            self.registry.backends, self.config.backends, self.resources
        )
        self.telemetry = Telemetry(
            self.config.telemetry.structured_logs, self.config.telemetry.tracing
        )
        self.identity = identity or TenantIdentity(
            "local", concurrency_limit=32, queue_limit=128, max_priority=100
        )
        self.audit_queue: asyncio.Queue | None = None
        self.audit_task: asyncio.Task | None = None
        self.audit_executor: ThreadPoolExecutor | None = None
        self.audit_dropped = 0
        self.scheduler = Scheduler(
            self.config.scheduler,
            self.config.profiles,
            self.tasks,
            self.router,
            self.supervisor,
            self.resources,
            self.telemetry,
            policy,
            self._audit,
        )
        self.started, self.closed = False, False
        self.loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def from_config(cls, path: str | Path, **overrides) -> OFlowEngine:
        return cls(load_config(path, **overrides))

    def _audit(self, metadata: dict) -> None:
        if self.audit_queue is not None:
            try:
                self.audit_queue.put_nowait(metadata)
            except asyncio.QueueFull:
                self.audit_dropped += 1

    async def _audit_loop(self, repository: SQLiteAuditRepository) -> None:
        assert self.audit_queue is not None
        try:
            while True:
                metadata = await self.audit_queue.get()
                try:
                    if metadata is None:
                        return
                    await asyncio.get_running_loop().run_in_executor(
                        self.audit_executor, repository.record, metadata
                    )
                except Exception:
                    self.audit_dropped += 1
                finally:
                    self.audit_queue.task_done()
        finally:
            await asyncio.get_running_loop().run_in_executor(self.audit_executor, repository.close)

    async def start(self) -> OFlowEngine:
        if self.closed:
            raise RuntimeError("engine is single-lifecycle; create a new instance after close")
        if self.started:
            self._check_loop()
            return self
        self.loop = asyncio.get_running_loop()
        await self.supervisor.start()
        if self.config.telemetry.metadata_sqlite:
            self.audit_queue = asyncio.Queue(maxsize=128)
            self.audit_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="oflow-audit"
            )
            try:
                repository = await self.loop.run_in_executor(
                    self.audit_executor,
                    SQLiteAuditRepository,
                    self.config.telemetry.metadata_sqlite,
                    self.config.telemetry.audit_retention,
                )
            except BaseException:
                await self.supervisor.close()
                self.audit_executor.shutdown(wait=True)
                self.closed = True
                raise
            self.audit_task = asyncio.create_task(self._audit_loop(repository))
        await self.scheduler.start()
        self.started = True
        return self

    def _check_loop(self) -> None:
        if not self.started or self.closed or self.loop is not asyncio.get_running_loop():
            raise RuntimeError("use the engine inside its async context on one event loop")

    async def submit(
        self, request: TaskRequest, *, identity: TenantIdentity | None = None
    ) -> TaskHandle:
        self._check_loop()
        return await self.scheduler.submit(request, identity or self.identity)

    async def execute(
        self, request: TaskRequest, *, identity: TenantIdentity | None = None
    ) -> TaskResult:
        return await (await self.submit(request, identity=identity)).result()

    def update_routing(self, rules: dict[str, list[str]]) -> None:
        """Trusted SDK operator change; already admitted work retains its routing decision."""
        self._check_loop()
        validated = EngineConfig.model_validate(
            {**self.config.model_dump(), "routing": {"rules": rules}}
        )
        router = RulesRouter(self.registry, validated.routing.rules)
        self.config = validated
        self.router = self.scheduler.router = router

    async def execute_many(
        self,
        requests: list[TaskRequest],
        *,
        identity: TenantIdentity | None = None,
        concurrency: int = 4,
    ) -> list[TaskResult]:
        """Sliding window: no per-item unlimited futures or scheduler-bypassing fast path."""
        if not 1 <= concurrency <= self.config.scheduler.max_inflight:
            raise ValueError("invalid batch client concurrency")
        results: list[TaskResult | None] = [None] * len(requests)
        iterator = iter(enumerate(requests))

        async def consume():
            for index, request in iterator:
                results[index] = await self.execute(request, identity=identity)

        consumers = [asyncio.create_task(consume()) for _ in range(min(concurrency, len(requests)))]
        try:
            await asyncio.gather(*consumers)
        finally:
            for consumer in consumers:
                if not consumer.done():
                    consumer.cancel()
            await asyncio.gather(*consumers, return_exceptions=True)
        assert all(result is not None for result in results)
        return [result for result in results if result is not None]

    async def close(self) -> None:
        if self.closed:
            return
        if self.started:
            self._check_loop()
            await self.scheduler.close()
        try:
            await self.supervisor.close()
        finally:
            if self.audit_task:
                assert self.audit_queue is not None and self.audit_executor is not None
                await self.audit_queue.put(None)
                await self.audit_task
                self.audit_executor.shutdown(wait=True)
            self.closed, self.started = True, False

    async def __aenter__(self) -> OFlowEngine:
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.close()


def execute_sync(request: TaskRequest, config: EngineConfig | None = None) -> TaskResult:
    """One-shot convenience; persistent/heavyweight callers should use the async context."""

    async def run():
        async with OFlowEngine(config) as engine:
            return await engine.execute(request)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run())
    raise RuntimeError("execute_sync cannot run inside an event loop; await engine.execute instead")
