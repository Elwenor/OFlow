import asyncio
import gc
import tracemalloc

from oflow import OFlowEngine, TaskRequest, TaskState
from oflow.backends import CPUBackend, MockBackend, register_builtin_tasks
from oflow.config import EngineConfig
from oflow.contracts import ErrorCategory
from oflow.registry import TaskRegistry


def echo(**kwargs):
    return TaskRequest(task="test.echo", input={"value": "ok"}, profile="latency_first", **kwargs)


def test_bad_cost_estimator_does_not_kill_dispatcher():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    config = EngineConfig()

    class BrokenCost(CPUBackend):
        def estimate_cost(self, request):
            raise ValueError("bad estimator")

    backend = BrokenCost(config.backends[0], tasks)

    async def scenario():
        async with OFlowEngine(config, tasks=tasks, backends=[backend]) as e:
            r = await e.execute(TaskRequest(task="text.score", input={"text": "x"}))
            assert r.error.category == ErrorCategory.BACKEND
            assert not e.scheduler.dispatcher.done()

    asyncio.run(scenario())


def test_bad_policy_fails_queued_work_without_hanging():
    class BrokenPolicy:
        def rank(self, *args):
            raise ValueError("bad policy")

    async def scenario():
        async with OFlowEngine(policy=BrokenPolicy()) as e:
            r = await asyncio.wait_for(
                e.execute(TaskRequest(task="text.score", input={"text": "x"})), 2
            )
            assert r.status == TaskState.FAILED
            assert not e.scheduler.accepting

    asyncio.run(scenario())


def test_impossible_resource_budget_rejected_before_waiting():
    config = EngineConfig.model_validate(
        {
            "devices": [{"id": "cpu", "memory_mb": 1}],
            "backends": [{"id": "cpu", "request_memory_mb": 2}],
        }
    )

    async def scenario():
        async with OFlowEngine(config) as e:
            r = await e.execute(TaskRequest(task="text.score", input={"text": "x"}))
            assert r.error.category == ErrorCategory.UNAVAILABLE
            assert not e.scheduler.queue

    asyncio.run(scenario())


def test_invalid_or_oversized_output_is_normalized():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    config = EngineConfig.model_validate(
        {"backends": [{"id": "mock", "kind": "mock"}], "scheduler": {"max_output_bytes": 256}}
    )

    class HugeOutput(MockBackend):
        def execute(self, request, context):
            return {"value": "x" * 1024}

    backend = HugeOutput(config.backends[0], tasks)

    async def scenario():
        async with OFlowEngine(config, tasks=tasks, backends=[backend]) as e:
            result = await e.execute(echo())
            assert result.status == TaskState.FAILED
            assert result.error.category == ErrorCategory.BACKEND

    asyncio.run(scenario())


def test_backend_initialization_failure_releases_persistent_memory():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    config = EngineConfig.model_validate(
        {"backends": [{"id": "mock", "kind": "mock", "model_memory_mb": 4}]}
    )

    class BadInitialize(MockBackend):
        def initialize(self):
            raise RuntimeError("initialization failed")

    async def scenario():
        async with OFlowEngine(
            config, tasks=tasks, backends=[BadInitialize(config.backends[0], tasks)]
        ) as e:
            assert not e.resources.leases
            result = await e.execute(echo())
            assert result.error.category == ErrorCategory.UNAVAILABLE

    asyncio.run(scenario())


def test_history_ttl_and_repeated_request_python_memory():
    async def scenario():
        config = EngineConfig.model_validate(
            {
                "backends": [{"id": "mock", "kind": "mock"}],
                "telemetry": {"structured_logs": False},
                "scheduler": {"history_limit": 4, "history_ttl_seconds": 0.02},
            }
        )
        async with OFlowEngine(config) as e:
            tracemalloc.start()
            for _ in range(100):
                await e.execute(echo())
            gc.collect()
            before = tracemalloc.get_traced_memory()[0]
            for _ in range(400):
                await e.execute(echo())
            gc.collect()
            after = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            # Detect retained per-request graphs, while allowing interpreter/cache noise.
            assert after - before < 500000
            await asyncio.sleep(0.05)
            e.scheduler._prune()
            assert len(e.scheduler.records) == 0

    asyncio.run(scenario())
