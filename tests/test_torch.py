import asyncio

import pytest

from oflow import OFlowEngine, TaskRequest, TaskState
from oflow.config import EngineConfig
from oflow.contracts import ErrorCategory
from oflow.resources import Device


def test_optional_torch_backend_real_cpu_execution_and_rng_isolation():
    torch = pytest.importorskip("torch")
    initial = torch.random.get_rng_state().clone()
    config = EngineConfig.model_validate(
        {"backends": [{"id": "embeddings", "kind": "torch_embed"}]}
    )

    async def scenario():
        async with OFlowEngine(config) as engine:
            results = await engine.execute_many(
                [
                    TaskRequest(task="text.embed", input={"text": text})
                    for text in ["abc", "abc", "def"]
                ]
            )
            assert all(r.status == TaskState.COMPLETED for r in results)
            assert results[0].output == results[1].output
            assert results[0].output != results[2].output
            assert len(results[0].output["embedding"]) == 16

    asyncio.run(scenario())
    assert torch.equal(initial, torch.random.get_rng_state())


def test_gpu_configs_validate_without_torch_or_available_gpu():
    from oflow.config import load_config

    for path, expected_count in [("configs/gpu.yaml", 1), ("configs/multi_gpu.yaml", 2)]:
        config = load_config(path)
        gpu_backends = [
            backend for backend in config.backends if backend.device.startswith("cuda:")
        ]
        assert len(gpu_backends) == expected_count
        assert all(
            backend.kind == "torch_embed" and backend.model_memory_mb > 0
            for backend in gpu_backends
        )


def test_unavailable_gpu_does_not_silently_execute_embedding_on_cpu():
    from oflow.config import load_config

    config = load_config("configs/gpu.yaml")
    inventory = [
        Device("cpu", "host", "CPU", 1024, 4),
        Device("cuda:0", "unavailable", "unavailable", 1024, 1, healthy=False),
    ]

    async def scenario():
        async with OFlowEngine(config, inventory=inventory) as engine:
            result = await engine.execute(TaskRequest(task="text.embed", input={"text": "x"}))
            assert result.error.category == ErrorCategory.UNAVAILABLE
            assert result.backend is None
            cpu = await engine.execute(
                TaskRequest(task="text.score", input={"text": "still available"})
            )
            assert cpu.status == TaskState.COMPLETED and cpu.backend == "cpu"

    asyncio.run(scenario())


@pytest.mark.cuda
def test_single_cuda_execution_matches_cpu_and_releases_repeated_allocations():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a CUDA-enabled PyTorch build and physical GPU are required")
    inputs = ["abc", "zażółć gęślą jaźń", "", "GPU " * 1000]

    async def scenario():
        cpu_config = EngineConfig.model_validate(
            {"backends": [{"id": "embeddings", "kind": "torch_embed"}]}
        )
        async with OFlowEngine(cpu_config) as cpu:
            reference = await cpu.execute_many(
                [TaskRequest(task="text.embed", input={"text": text}) for text in inputs]
            )
        torch.cuda.synchronize(0)
        before = torch.cuda.memory_allocated(0)
        async with OFlowEngine.from_config("configs/gpu.yaml") as engine:
            worker = engine.supervisor.workers["embeddings-gpu0"]
            assert worker.healthy
            weights = worker.backend.weights
            assert str(weights.device) == "cuda:0"
            address = weights.data_ptr()
            del weights
            for _ in range(20):
                results = await engine.execute_many(
                    [
                        TaskRequest(
                            task="text.embed",
                            input={"text": text},
                            preferred_backend="embeddings-gpu0",
                            profile="throughput_first",
                        )
                        for text in inputs
                    ]
                )
                assert worker.backend.weights.data_ptr() == address
                for actual, expected in zip(results, reference, strict=True):
                    assert actual.status == TaskState.COMPLETED
                    assert actual.backend == "embeddings-gpu0"
                    assert actual.output["embedding"] == pytest.approx(
                        expected.output["embedding"], abs=1e-5
                    )
        assert not engine.resources.leases
        torch.cuda.synchronize(0)
        assert torch.cuda.memory_allocated(0) <= before + 1048576

    asyncio.run(scenario())


@pytest.mark.cuda
@pytest.mark.multi_gpu
def test_two_physical_cuda_workers_execute_and_release_their_devices():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two visible physical CUDA GPUs are required")
    before = [torch.cuda.memory_allocated(index) for index in range(2)]

    async def scenario():
        async with OFlowEngine.from_config("configs/multi_gpu.yaml") as engine:
            for index in range(2):
                worker = engine.supervisor.workers[f"embeddings-gpu{index}"]
                assert worker.healthy
                assert str(worker.backend.weights.device) == f"cuda:{index}"
            results = await asyncio.gather(
                *(
                    engine.execute(
                        TaskRequest(
                            task="text.embed",
                            input={"text": "same input"},
                            preferred_backend=f"embeddings-gpu{index % 2}",
                            profile="latency_first",
                        )
                    )
                    for index in range(8)
                )
            )
            assert all(result.status == TaskState.COMPLETED for result in results)
            assert {result.backend for result in results} == {"embeddings-gpu0", "embeddings-gpu1"}
            for result in results[1:]:
                assert result.output["embedding"] == pytest.approx(
                    results[0].output["embedding"], abs=1e-5
                )
        assert not engine.resources.leases
        for index in range(2):
            torch.cuda.synchronize(index)
            assert torch.cuda.memory_allocated(index) <= before[index] + 1048576

    asyncio.run(scenario())
