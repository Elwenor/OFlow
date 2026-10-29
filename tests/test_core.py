import asyncio
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from oflow import OFlowEngine, TaskRequest, TaskState, TenantIdentity, execute_sync
from oflow.backends import MockBackend, register_builtin_tasks
from oflow.config import BackendConfig, EngineConfig, load_config
from oflow.contracts import ErrorCategory, ResourceRequirements, WorkerCrashed
from oflow.registry import BackendRegistry, TaskRegistry
from oflow.resources import Device, ResourceManager


def run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, 15))


def request(**kwargs):
    return TaskRequest(task="test.echo", input={"value": "ok"}, profile="latency_first", **kwargs)


def mock_config(**changes):
    data = {
        "backends": [{"id": "mock", "kind": "mock", "options": {"delay_seconds": 0.02}}],
        "telemetry": {"structured_logs": False},
    }
    data.update(changes)
    return EngineConfig.model_validate(data)


def test_cpu_sdk_and_sync():
    result = execute_sync(TaskRequest(task="math.statistics", input={"values": [1, 2, 3]}))
    assert result.status == TaskState.COMPLETED
    assert result.output == {"count": 3, "mean": 2.0, "minimum": 1.0, "maximum": 3.0}


def test_basic_import_does_not_load_torch_or_http():
    code = "import sys; sys.path.insert(0,'src'); import oflow; print(any(x in sys.modules for x in ['torch','fastapi','httpx']))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize(
    "data",
    [
        {"schema_version": "2"},
        {"mystery": 1},
        {"scheduler": {"queue_limit": 0}},
        {"backends": [{"id": "x", "device": "cuda:9"}]},
        {"routing": {"rules": {"text.score": ["missing"]}}},
        {"backends": [{"id": "x"}, {"id": "x"}]},
        {
            "plugins": [
                {"name": "untrusted", "version": "1", "entry_point": "x:y", "capabilities": []}
            ]
        },
    ],
)
def test_invalid_config(data):
    with pytest.raises(ValueError):
        EngineConfig.model_validate(data)


def test_yaml_precedence(tmp_path):
    path = tmp_path / "oflow.yaml"
    path.write_text("schema_version: '1'\nscheduler:\n  queue_limit: 7\n")
    assert load_config(path).scheduler.queue_limit == 7
    assert load_config(path, scheduler={"queue_limit": 3}).scheduler.queue_limit == 3


def test_public_identity_and_options_rejected():
    with pytest.raises(ValidationError):
        TaskRequest(task="test.echo", input={}, tenant="admin")
    with pytest.raises(ValidationError):
        request(deadline=datetime.now())

    async def scenario():
        async with OFlowEngine(mock_config()) as e:
            r = await e.execute(request(options={"import_module": "os"}))
            assert r.error.category == ErrorCategory.INVALID_REQUEST
            bad = await e.execute(TaskRequest(task="test.echo", input={"wrong": "private"}))
            assert bad.error.category == ErrorCategory.INVALID_REQUEST

    run(scenario())


def test_backend_registration_and_compatibility():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    registry = BackendRegistry(tasks)
    backend = MockBackend(BackendConfig(id="mock", kind="mock"), tasks)
    registry.register(backend)
    with pytest.raises(ValueError):
        registry.register(backend)
    with pytest.raises(ValueError):
        OFlowEngine(EngineConfig.model_validate({"routing": {"rules": {"test.echo": ["cpu"]}}}))


def test_routing_order_fallback_permissions_and_profiles():
    config = EngineConfig.model_validate(
        {
            "backends": [{"id": "first", "kind": "mock"}, {"id": "second", "kind": "mock"}],
            "routing": {"rules": {"text.score": ["second", "first"]}},
        }
    )

    async def scenario():
        async with OFlowEngine(config) as e:
            task = TaskRequest(task="text.score", input={"text": "hello"})
            assert (await e.execute(task)).backend == "second"
            e.supervisor.workers["second"].healthy = False
            assert (
                await e.execute(task.model_copy(update={"request_id": "fallback"}))
            ).backend == "first"
            denied = TenantIdentity("limited", tasks=frozenset({"test.echo"}))
            assert (
                await e.execute(task, identity=denied)
            ).error.category == ErrorCategory.FORBIDDEN
            limited = TenantIdentity("limited", backends=frozenset({"first"}))
            assert (await e.execute(task, identity=limited)).backend == "first"
            assert (
                await e.execute(request(preferred_backend="second"), identity=limited)
            ).error.category == ErrorCategory.FORBIDDEN
            assert (
                await e.execute(
                    TaskRequest(task="text.score", input={"text": "x"}, profile="bogus")
                )
            ).error.category == ErrorCategory.INVALID_REQUEST

    run(scenario())


def test_queue_bounds_tenant_limits_and_priority():
    config = mock_config(
        scheduler={"queue_limit": 2},
        backends=[{"id": "mock", "kind": "mock", "options": {"delay_seconds": 0.12}}],
    )

    async def scenario():
        async with OFlowEngine(config) as e:
            tenant = TenantIdentity("one", concurrency_limit=1, queue_limit=2)
            first = await e.submit(request(), identity=tenant)
            while first.state != TaskState.RUNNING:
                await asyncio.sleep(0.001)
            low = await e.submit(request(priority=-10), identity=tenant)
            high = await e.submit(request(priority=0), identity=tenant)
            extra = await e.execute(request(), identity=tenant)
            assert extra.error.category == ErrorCategory.OVERLOADED
            assert len(e.scheduler.queue) == 2
            assert e.scheduler.tenant_running["one"] == 1
            await first.result()
            await high.result()
            assert low.state != TaskState.COMPLETED
            assert (await low.result()).status == TaskState.COMPLETED

    run(scenario())


def test_running_and_queued_cancellation_deadlines_release_resources():
    async def scenario():
        async with OFlowEngine(mock_config()) as e:
            running = await e.submit(request())
            while running.state != TaskState.RUNNING:
                await asyncio.sleep(0.001)
            queued = await e.submit(request())
            assert queued.cancel()
            assert (await queued.result()).status == TaskState.CANCELLED
            assert running.cancel()
            assert (await running.result()).status == TaskState.CANCELLED
            deadline = await e.execute(
                request(deadline=datetime.now(UTC) + timedelta(milliseconds=1))
            )
            assert deadline.status == TaskState.TIMED_OUT
            expired = await e.execute(request(deadline=datetime.now(UTC) - timedelta(seconds=1)))
            assert expired.status == TaskState.TIMED_OUT
        assert not e.resources.leases
        assert e.scheduler.inflight == 0
        assert not e.scheduler.tenant_running

    run(scenario())


def test_cancelled_noncooperative_worker_keeps_lease():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    started, release = threading.Event(), threading.Event()

    class NonCooperative(MockBackend):
        def execute(self, req, ctx):
            started.set()
            release.wait(2)
            return {"value": "finished"}

    config = mock_config()
    backend = NonCooperative(config.backends[0], tasks)

    async def scenario():
        async with OFlowEngine(config, tasks=tasks, backends=[backend]) as e:
            handle = await e.submit(request())
            while not started.is_set():
                await asyncio.sleep(0.001)
            handle.cancel()
            await handle.result()
            assert e.scheduler.inflight == 1
            assert e.resources.active["cpu"] == 1
            release.set()
            while e.scheduler.inflight:
                await asyncio.sleep(0.001)
            assert e.resources.active["cpu"] == 0

    run(scenario())


def test_client_coroutine_cancellation_cancels_execution():
    async def scenario():
        async with OFlowEngine(mock_config()) as e:
            req = request()
            task = asyncio.create_task(e.execute(req))
            await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert e.scheduler.status(req.request_id, e.identity) == TaskState.CANCELLED

    run(scenario())


def test_worker_lifecycle_cancellation_cannot_release_active_thread_lease():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    config = mock_config()
    entered, release = threading.Event(), threading.Event()

    class Blocking(MockBackend):
        def execute(self, request, context):
            entered.set()
            release.wait(2)
            return {"value": "done"}

    async def scenario():
        async with OFlowEngine(
            config, tasks=tasks, backends=[Blocking(config.backends[0], tasks)]
        ) as engine:
            handle = await engine.submit(request())
            while not entered.is_set():
                await asyncio.sleep(0.001)
            physical_task = next(iter(engine.scheduler.running))
            physical_task.cancel()
            await asyncio.sleep(0.01)
            assert engine.resources.active["cpu"] == 1
            assert engine.scheduler.inflight == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await physical_task
            assert (await handle.result()).status == TaskState.CANCELLED
            assert engine.resources.active["cpu"] == 0

    run(scenario())


def test_multiple_tenants_round_robin_and_no_starvation():
    async def scenario():
        async with OFlowEngine(mock_config()) as e:
            bulk = TenantIdentity("bulk", concurrency_limit=1)
            ui = TenantIdentity("ui", concurrency_limit=1)
            handles = [await e.submit(request(), identity=bulk) for _ in range(10)]
            while handles[0].state != TaskState.RUNNING:
                await asyncio.sleep(0.001)
            interactive = await e.submit(request(), identity=ui)
            assert (await interactive.result()).status == TaskState.COMPLETED
            assert handles[-1].state != TaskState.COMPLETED
            assert all(
                r.status == TaskState.COMPLETED
                for r in await asyncio.gather(*(h.result() for h in handles))
            )

    run(scenario())


def test_priority_aging_eventually_promotes_old_requests():
    from oflow.config import default_profiles
    from oflow.scheduler import FairPriorityPolicy

    policy = FairPriorityPolicy(0.01)
    old = request(priority=-100)
    new = request(priority=100)
    assert policy.rank([(old, 0), (new, 3)], default_profiles(), 3)[0] == 0


def test_worker_crash_restarts_and_no_exception_details_leak():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    config = mock_config()

    class Crashing(MockBackend):
        calls = 0

        def execute(self, req, ctx):
            self.calls += 1
            if self.calls == 1:
                raise WorkerCrashed("TOP-SECRET traceback")
            return {"value": "recovered"}

    backend = Crashing(config.backends[0], tasks)

    async def scenario():
        async with OFlowEngine(config, tasks=tasks, backends=[backend]) as e:
            first = await e.execute(request())
            assert first.status == TaskState.FAILED
            assert "TOP-SECRET" not in first.model_dump_json()
            while (
                e.supervisor.workers["mock"].restarts == 0
                or not e.supervisor.workers["mock"].healthy
            ):
                await asyncio.sleep(0.001)
            assert (await e.execute(request())).status == TaskState.COMPLETED

    run(scenario())


def test_invalid_output_and_failure_cleanup():
    async def scenario():
        async with OFlowEngine(
            mock_config(backends=[{"id": "mock", "kind": "mock", "options": {"fail": True}}])
        ) as e:
            r = await e.execute(request())
            assert r.status == TaskState.FAILED
        assert not e.resources.leases

    run(scenario())


def test_repeated_execution_has_bounded_history_and_no_inputs():
    async def scenario():
        async with OFlowEngine(
            mock_config(scheduler={"history_limit": 8}, backends=[{"id": "mock", "kind": "mock"}])
        ) as e:
            for _ in range(200):
                assert (await e.execute(request())).status == TaskState.COMPLETED
            await asyncio.sleep(0.01)
            assert len(e.scheduler.records) == 8
            assert all(j.request.input == {} for j in e.scheduler.records.values())
            assert len(e.resources.leases) == 1  # persistent worker lease only
            assert not e.scheduler.tenant_running
            assert not e.scheduler.tenant_queued

    run(scenario())


def test_graceful_shutdown_drains_then_stops_admission():
    async def scenario():
        e = OFlowEngine(mock_config())
        await e.start()
        handles = [await e.submit(request()) for _ in range(4)]
        await e.close()
        assert all(
            r.status == TaskState.COMPLETED
            for r in await asyncio.gather(*(h.result() for h in handles))
        )
        assert not e.resources.leases
        with pytest.raises(RuntimeError):
            await e.start()

    run(scenario())


def test_duplicate_request_ids_and_tenant_status_isolation():
    async def scenario():
        async with OFlowEngine(mock_config()) as e:
            r = request(request_id="same")
            first = await e.submit(r)
            duplicate = await e.execute(r)
            assert duplicate.error.category == ErrorCategory.CONFLICT
            other = TenantIdentity("other")
            assert e.scheduler.status("same", other) is None
            assert not e.scheduler.cancel("same", other)
            assert (await e.execute(r, identity=other)).status == TaskState.COMPLETED
            await first.result()

    run(scenario())


def test_multiple_gpu_placement_mock_inventory_and_budgets():
    resources = ResourceManager(
        [
            Device("cpu", "host", "CPU", 1024, 4),
            Device("cuda:0", "GPU-A", "Fake GPU 1", 100, 1),
            Device("cuda:1", "GPU-B", "Fake GPU 2", 200, 2),
        ]
    )
    model = resources.acquire(
        ResourceRequirements(device="cuda:0", cpu_slots=0, memory_mb=80), persistent=True
    )
    first = resources.acquire(ResourceRequirements(device="cuda:0", memory_mb=20))
    assert model and first
    assert resources.acquire(ResourceRequirements(device="cuda:0", memory_mb=1)) is None
    second = resources.acquire(ResourceRequirements(device="cuda:1", memory_mb=100))
    assert second
    resources.release(first)
    resources.release(first)  # idempotent cleanup
    resources.release(model)
    resources.release(second)
    assert not resources.leases


def test_missing_gpu_does_not_break_cpu_execution():
    config = EngineConfig.model_validate({"devices": [{"id": "cpu"}, {"id": "cuda:0"}]})
    inventory = [
        Device("cpu", "host", "CPU", 1024, 1),
        Device("cuda:0", "missing", "missing", 1024, 1, healthy=False),
    ]

    async def scenario():
        async with OFlowEngine(config, inventory=inventory) as e:
            assert (
                await e.execute(TaskRequest(task="text.score", input={"text": "cpu"}))
            ).status == TaskState.COMPLETED

    run(scenario())
