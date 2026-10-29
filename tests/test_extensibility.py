import asyncio
import json
import sqlite3
import sys
import types

import pytest

from oflow import OFlowEngine, TaskRequest
from oflow.backends import MockBackend, register_builtin_tasks
from oflow.config import EngineConfig
from oflow.contracts import Contract, ResourceRequirements
from oflow.registry import TaskRegistry
from oflow.resources import Device


def test_trusted_plugin_adds_new_task_without_scheduler_changes(monkeypatch):
    module = types.ModuleType("test_oflow_plugin")
    module.OFLOW_PLUGIN_API_VERSION = "1"
    module.OFLOW_PLUGIN_VERSION = "0.1"

    class Input(Contract):
        value: str

    class Output(Contract):
        value: str

    class UpperBackend(MockBackend):
        def __init__(self, config, tasks):
            super().__init__(config.model_copy(update={"kind": "mock"}), tasks)
            self.descriptor = self.descriptor.model_copy(
                update={"capabilities": (tasks.capability("demo.upper"),)}
            )

        def execute(self, request, context):
            return {"value": request.input["value"].upper()}

    def register(tasks):
        tasks.register("demo.upper", Input, Output)
        return {"uppercase": UpperBackend}

    module.register = register
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = EngineConfig.model_validate(
        {
            "backends": [{"id": "upper", "kind": "uppercase"}],
            "trusted_plugin_allowlist": ["uppercase_plugin"],
            "plugins": [
                {
                    "name": "uppercase_plugin",
                    "version": "0.1",
                    "api_version": "1",
                    "entry_point": "test_oflow_plugin:register",
                    "capabilities": ["uppercase"],
                }
            ],
        }
    )

    async def scenario():
        async with OFlowEngine(config) as engine:
            result = await engine.execute(TaskRequest(task="demo.upper", input={"value": "hello"}))
            assert result.output == {"value": "HELLO"}

    asyncio.run(scenario())
    module.OFLOW_PLUGIN_API_VERSION = "99"
    with pytest.raises(ValueError):
        OFlowEngine(config)


def test_microbatch_is_capability_driven_and_keeps_association():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    config = EngineConfig.model_validate({"backends": [{"id": "mock", "kind": "mock"}]})
    sizes = []

    class Batch(MockBackend):
        def execute_batch(self, requests, contexts):
            sizes.append(len(requests))
            return [{"value": r.input["value"]} for r in requests]

    backend = Batch(config.backends[0], tasks)

    async def scenario():
        async with OFlowEngine(config, tasks=tasks, backends=[backend]) as engine:
            requests = [
                TaskRequest(task="test.echo", input={"value": i}, profile="throughput_first")
                for i in range(8)
            ]
            results = await engine.execute_many(requests, concurrency=8)
            assert [r.output["value"] for r in results] == list(range(8))
            assert max(sizes) == 8

    asyncio.run(scenario())


def test_runtime_routing_update_is_validated():
    config = EngineConfig.model_validate(
        {"backends": [{"id": "a", "kind": "mock"}, {"id": "b", "kind": "mock"}]}
    )

    async def scenario():
        async with OFlowEngine(config) as engine:
            engine.update_routing({"text.score": ["b"]})
            result = await engine.execute(TaskRequest(task="text.score", input={"text": "x"}))
            assert result.backend == "b"
            with pytest.raises(ValueError):
                engine.update_routing({"text.score": ["missing"]})
            assert engine.router.rules == {"text.score": ["b"]}

    asyncio.run(scenario())


def test_independent_workers_execute_on_multiple_mock_gpu_devices():
    tasks = TaskRegistry()
    register_builtin_tasks(tasks)
    config = EngineConfig.model_validate(
        {
            "devices": [{"id": "cpu", "max_concurrency": 4}, {"id": "cuda:0"}, {"id": "cuda:1"}],
            "backends": [
                {"id": "gpu0", "kind": "fake", "device": "cuda:0", "model_memory_mb": 80},
                {"id": "gpu1", "kind": "fake", "device": "cuda:1", "model_memory_mb": 80},
            ],
        }
    )

    class Placed(MockBackend):
        def __init__(self, cfg):
            super().__init__(cfg.model_copy(update={"device": "cpu"}), tasks)
            self.descriptor = self.descriptor.model_copy(
                update={"requirements": ResourceRequirements(device=cfg.device)}
            )

        def execute(self, request, context):
            return {"value": context.device}

    inventory = [
        Device("cpu", "host", "CPU", 1024, 4),
        Device("cuda:0", "GPU-UUID-A", "Mock GPU A", 100, 1),
        Device("cuda:1", "GPU-UUID-B", "Mock GPU B", 100, 1),
    ]

    async def scenario():
        async with OFlowEngine(
            config, tasks=tasks, backends=[Placed(c) for c in config.backends], inventory=inventory
        ) as engine:
            results = await asyncio.gather(
                *[
                    engine.execute(
                        TaskRequest(
                            task="test.echo", input={"value": 0}, preferred_backend=f"gpu{i}"
                        )
                    )
                    for i in range(2)
                ]
            )
            assert [r.output for r in results] == [{"value": "cuda:0"}, {"value": "cuda:1"}]
            assert all(d["reserved_memory_mb"] == 80 for d in engine.resources.snapshot()[1:])
        assert not engine.resources.leases

    asyncio.run(scenario())


def test_optional_sqlite_audit_retention_and_privacy(tmp_path):
    path = tmp_path / "audit.sqlite"
    config = EngineConfig.model_validate(
        {"telemetry": {"metadata_sqlite": str(path), "audit_retention": 3}}
    )

    async def scenario():
        async with OFlowEngine(config) as engine:
            for _ in range(10):
                await engine.execute(
                    TaskRequest(task="text.score", input={"text": "SECRET-PROMPT-NEVER-STORE"})
                )

    asyncio.run(scenario())
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT metadata FROM audit").fetchall()
    assert len(rows) == 3
    assert "SECRET-PROMPT-NEVER-STORE" not in str(rows)
    assert all(json.loads(row[0])["status"] == "COMPLETED" for row in rows)
